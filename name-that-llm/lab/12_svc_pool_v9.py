"""Step 12: does Open Pool v9 beat nlm_pool_v5 for the 0.663 notebook's 4-view n-gram SVC member? (CPU)

What: rebuilds the `svc4` member of the 0.663 notebook (infer.py, `--svc_w 1.5`): four TF-IDF views (char 3-5,
word 1-3, letter-shape 2-6, function-word 1-3, weights 1/0.7/0.5/0.5), LinearSVC(C=0.5, balanced), fitted on
official train (sample weight 1.0) + nlm_matched_clean (0.5) + a pool (0.35), with the notebook's NFKC/whitespace
exact-key de-duplication (train keys first, then extra, then pool; first occurrence wins). Scored out of fold on
train.csv with prompt-grouped GroupKFold(5) (the lab's folds): each fold's train prompts are held out, and, as in the
notebook's --sim mode, every extra/pool row whose key matches a held-out passage is dropped. Settings:
  none = train + extra only (reference), v5 = + nlm_pool_v5 (what the notebook uses), v9 = + Open Pool v9 instead.
Reports the leak check (pool rows whose normalised text equals a train passage), balanced accuracy per setting
(argmax and after a 300-iteration Sinkhorn over all OOF rows), per-class recall, and a paired bootstrap over prompts.

Why: Open Pool v9 (crtproduction/name-that-llm-open-pool, CC BY-SA 4.0) is larger and cleaned; swapping the pool is a
change to how a member is trained, the only kind of change worth a submission.

Deviation (ponytail): the vectorisers are fitted once per setting on all rows (held-out train texts included, labels
never), not once per fold; min_df/max_features shift by ~1% of rows at most. Upgrade path: refit per fold.

Output: <out>/svc12_<setting>.npz (OOF decision scores, cached; delete to refit) and <out>/svc12_pool_v9.json.
Usage: python 12_svc_pool_v9.py --train data/train.csv --extra nlm_matched_clean.parquet \
           --v5 nlm_pool_v5.parquet --v9 ntl_open_pool.parquet [--out artifacts] [--boot 2000] [--jobs 2] [--v9-rows N]
"""
import argparse
import importlib.util
import json
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score as bacc, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('s08', os.path.join(HERE, '08_blend_oof.py'))
s08 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s08)   # the same four views as the notebook (char, word, shape, small/function words)
LABELS, K, log = s08.LABELS, s08.K, s08.t07.log
SRC_W = {'train': 1.0, 'extra': 0.5, 'pool': 0.35}   # infer.py: official 1.0, --extra_w 0.5, pool 0.35


def key(t):   # infer.py _norm_text_key
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(t or ''))).strip()


def load(path):
    d = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path)
    d = d[d.model.isin(LABELS)][['text', 'model']].copy()
    d['text'] = d.text.fillna('').astype(str)
    return d.reset_index(drop=True)


def clean(df, seen):
    """Drop empty rows and rows whose key is already seen (train keys, earlier sources, earlier rows); updates seen."""
    keep = []
    for k in df.text.map(key):
        keep.append(bool(k) and k not in seen)
        seen.add(k)
    return df[keep].reset_index(drop=True)


def features(texts):
    A = []
    for view, lo, hi, mf, w in s08.VIEWS:
        prep = s08.shape_text if view == 'shape' else s08.small_words_text if view == 'small' else (lambda s: s)
        an = 'char' if view == 'shape' else 'word' if view == 'small' else view
        o = dict(analyzer=an, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf,
                 dtype=np.float32)
        if an == 'word':
            o['token_pattern'] = r'(?u)\b\w+\b|[^\w\s]'
        M = TfidfVectorizer(**o).fit_transform([prep(t) for t in texts])
        M.data *= w
        A.append(M)
        log(f'  view {view}: nnz {M.nnz:,}')
    X = hstack(A, format='csr')
    del A, M
    return X


def oof_scores(train, y, folds, extra_df, pool_df, jobs):
    n = len(train)
    fit = pd.concat([train[['text', 'model']].assign(_w=SRC_W['train'])]
                    + [d.assign(_w=SRC_W[s]) for s, d in (('extra', extra_df), ('pool', pool_df)) if d is not None],
                    ignore_index=True)
    X = features(fit.text.tolist())
    del fit['text']
    fy, sw = fit.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy(), fit._w.to_numpy(np.float32)

    def one(f):
        # Held-out rows stay in X with sample weight 0 (dual box [0, 0]: no effect) instead of slicing X, which would
        # copy ~3 GB; class weights are 'balanced' over the training rows only, as a slice would give.
        tr = np.r_[np.where(folds != f)[0], np.arange(n, len(fit))]
        cw = dict(enumerate(compute_class_weight('balanced', classes=np.arange(K), y=fy[tr])))
        w = np.zeros(len(fit), np.float32)
        w[tr] = sw[tr]
        svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight=cw)
        svc.fit(X, fy, sample_weight=w)
        va = np.where(folds == f)[0]
        log(f'  fold {f}: bacc {bacc(y[va], svc.decision_function(X[va]).argmax(1)):.4f}')
        return va, svc.decision_function(X[va])

    S = np.zeros((n, K))
    with ThreadPoolExecutor(jobs) as ex:   # liblinear releases the GIL
        for va, s in ex.map(one, range(folds.max() + 1)):
            S[va] = s
    return S


def boot(y, a, b, groups, n_boot, seed=0):
    """Paired bootstrap over prompts: P(bacc(b) > bacc(a)) and the mean difference."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        d.append(bacc(y[rows], b[rows]) - bacc(y[rows], a[rows]))
    d = np.array(d)
    return float((d > 0).mean()), float(d.mean()), [float(np.percentile(d, 5)), float(np.percentile(d, 95))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--extra', required=True)
    ap.add_argument('--v5', required=True)
    ap.add_argument('--v9', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'artifacts'))
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--v9-rows', type=int, default=0,
                    help='subsample the cleaned v9 pool to this many rows (0 = all; the full v9 fit needs >16 GB RAM)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train = pd.read_csv(a.train)
    train['text'] = train.text.fillna('').astype(str)
    y = train.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    folds = np.zeros(len(train), int)
    for f, (_, va) in enumerate(GroupKFold(n_splits=5).split(train, groups=train.prompt_id)):
        folds[va] = f
    tkeys = set(train.text.map(key))

    raw = {'extra': load(a.extra), 'v5': load(a.v5), 'v9': load(a.v9)}
    leak = {s: dict(rows=len(d), train_text_matches=int(d.text.map(key).isin(tkeys).sum()),
                    train_passages_hit=len(tkeys & set(d.text.map(key))),
                    internal_duplicates=int(d.text.map(key).duplicated().sum())) for s, d in raw.items()}
    seen = set(tkeys)
    extra = clean(raw['extra'], seen)
    pools = {}
    for s in ('v5', 'v9'):   # each pool is cleaned against train + extra on its own
        pools[s] = clean(raw[s], set(seen))
        leak[s]['kept_for_fit'] = len(pools[s])
    leak['extra']['kept_for_fit'] = len(extra)
    if a.v9_rows:   # ponytail: size-matched v9 (quality, not volume) because full v9 OOMs on 15 GB; upgrade: bigger box
        pools['v9'] = pools['v9'].sample(a.v9_rows, random_state=0).reset_index(drop=True)
        leak['v9']['subsampled_to'] = a.v9_rows
    log('leak check', json.dumps(leak))

    S = {}
    for s, pool in (('none', None), ('v5', pools['v5']), ('v9', pools['v9'])):
        p = os.path.join(a.out, f'svc12_{s}{"_n%d" % a.v9_rows if s == "v9" and a.v9_rows else ""}.npz')
        if os.path.exists(p):
            S[s] = np.load(p)['scores']
        else:
            log(f'setting {s}: fitting')
            S[s] = oof_scores(train, y, folds, extra, pool, a.jobs)
            np.savez(p, scores=S[s], fold=folds, ids=train.id.astype(str).to_numpy(dtype=str))
    pred = {s: v.argmax(1) for s, v in S.items()}
    pred_sk = {s: s08.sinkhorn(s08.softmax(v)).argmax(1) for s, v in S.items()}
    res = dict(leak=leak, bacc={s: round(bacc(y, p), 4) for s, p in pred.items()},
               bacc_sinkhorn={s: round(bacc(y, p), 4) for s, p in pred_sk.items()},
               fold_bacc={s: [round(bacc(y[folds == f], p[folds == f]), 4) for f in range(5)] for s, p in pred.items()},
               recall={s: dict(zip(LABELS, np.round(recall_score(y, p, labels=range(K), average=None), 3).tolist()))
                       for s, p in pred.items()})
    for name, P in (('argmax', pred), ('sinkhorn', pred_sk)):
        for base, new in (('v5', 'v9'), ('none', 'v5'), ('none', 'v9')):
            pb, md, ci = boot(y, P[base], P[new], train.prompt_id.values, a.boot)
            res[f'boot_{name}_{new}_vs_{base}'] = dict(p_better=round(pb, 3), mean_diff=round(md, 4), ci90=ci)
    json.dump(res, open(os.path.join(a.out, f'svc12_pool_v9{"_n%d" % a.v9_rows if a.v9_rows else ""}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'recall'}, indent=1))
    print(pd.DataFrame(res['recall']).assign(d_v9_v5=lambda d: d.v9 - d.v5).sort_values('d_v9_v5').to_string())


if __name__ == '__main__':
    main()
