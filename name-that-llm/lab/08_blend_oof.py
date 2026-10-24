"""Step 8: OOF evidence for the rebuilt encoder members: each member alone, and its gain in the blend (CPU).

What: loads the members' OOF logits from 07 (oof_<member>.npz) and reproduces the 0.620 base's n-gram SVC out of fold:
the notebook's four TF-IDF views and LinearSVC, on the same prompt-grouped folds, fitted on the pool plus the train folds.
It then scores, by balanced accuracy and after Sinkhorn as in the notebooks:
- each member alone;
- SVC + members at the public notebooks' OURS weights;
- a paired bootstrap over prompts, giving P(blend better than SVC alone).

Why: the base's own OOF (LoRAs, DeBERTa) is not published, so the SVC is the only base member we can rebuild out of fold.
The gain over SVC alone is an upper bound on the gain over the full 0.620 base, whose stronger members overlap more with
ours.

Output: <out>/oof_svc.npz (cached; delete it to refit) and <out>/blend_oof.json; the table is printed.
Usage: python 08_blend_oof.py --train data/train.csv --pool pool_v4.parquet --oof-dir DIR_WITH_oof_npz --out DIR
"""
import argparse
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('t07', os.path.join(HERE, '07_train_encoders.py'))
t07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t07)
LABELS, K = t07.LABELS, t07.K

# The base notebook's SVC cell, unchanged.
SMALL_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("small", 1, 3, 50000, 0.5)]
WEIGHTS = {'nuridudak 0.624 / mrpicklerick 0.624': dict(modernbert=0.25, desklib=0.10),
           'nuridudak 0.626': dict(modernbert=0.40, desklib=0.10),
           'mrpicklerick 0.627': dict(modernbert=0.30, desklib=0.05)}


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def small_words_text(t):
    return " ".join(w.lower() if w.lower() in SMALL_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


def svc_scores(fit_texts, fit_y, pred_texts):
    A, B = [], []
    for view, lo, hi, max_features, weight in VIEWS:
        prep = shape_text if view == "shape" else small_words_text if view == "small" else (lambda s: s)
        analyzer = "char" if view == "shape" else "word" if view == "small" else view
        opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                    max_features=max_features, dtype=np.float32)
        if analyzer == "word":
            opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
        vec = TfidfVectorizer(**opts)
        A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
        B.append(vec.transform([prep(t) for t in pred_texts]) * weight)
    svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)
    return svc.decision_function(hstack(B, format="csr"))


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn(P, iters=300):
    c = np.ones(K)
    target = np.full(K, len(P) / K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)


def blend(parts, rows=slice(None)):
    """parts: list of (weight, probabilities). Sinkhorn over the rows scored together, as on the test set."""
    L = sum(w * np.log(np.clip(P[rows], 1e-9, 1)) for w, P in parts)
    return sinkhorn(softmax(L)).argmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--oof-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train = pd.read_csv(a.train)
    train['text'] = train['text'].fillna('').astype(str)
    lab = {l: i for i, l in enumerate(LABELS)}
    y = train.model.map(lab).values
    members = {}
    for n in ('modernbert', 'desklib'):
        p = os.path.join(a.oof_dir, f'oof_{n}.npz')
        if os.path.exists(p):
            z = np.load(p)
            assert list(z['ids']) == train.id.astype(str).tolist() and list(z['labels']) == LABELS
            members[n] = dict(P=softmax(z['oof'].astype(np.float64)), P1=softmax(z['stage1'].astype(np.float64)))
            folds = z['fold']
    if not members:   # same folds as 07
        from sklearn.model_selection import GroupKFold
        folds = np.zeros(len(train), int)
        for f, (_, va) in enumerate(GroupKFold(n_splits=5).split(train, groups=train.prompt_id)):
            folds[va] = f

    svc_path = os.path.join(a.out, 'oof_svc.npz')
    if os.path.exists(svc_path):
        S = np.load(svc_path)['scores']
    else:
        pool = pd.read_parquet(a.pool)
        pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
        keep = t07.leak_filter(pool.text.tolist(), train.text.tolist()) & ~pool.text.isin(set(train.text))
        pool = pool[keep].reset_index(drop=True)   # the same honest pool as 07's stage 1
        S = np.zeros((len(train), K))
        for f in range(folds.max() + 1):
            tr, va = folds != f, folds == f
            S[va] = svc_scores(train.text[tr].tolist() + pool.text.tolist(),
                               np.r_[y[tr], pool.model.map(lab).values], train.text[va].tolist())
            t07.log(f'svc fold {f}: {balanced_accuracy_score(y[va], S[va].argmax(1)):.4f}')
        np.savez(svc_path, scores=S, fold=folds, ids=train.id.astype(str).to_numpy(dtype=str))
    Psvc = softmax(S)

    res = {'svc alone (argmax)': balanced_accuracy_score(y, Psvc.argmax(1)),
           'svc alone + sinkhorn': balanced_accuracy_score(y, blend([(1.0, Psvc)]))}
    for n, m in members.items():
        res[f'{n} alone (argmax)'] = balanced_accuracy_score(y, m['P'].argmax(1))
        res[f'{n} stage-1 only (argmax)'] = balanced_accuracy_score(y, m['P1'].argmax(1))
    if len(members) == 2:
        res['modernbert+desklib 1:1 + sinkhorn'] = balanced_accuracy_score(
            y, blend([(1.0, members['modernbert']['P']), (1.0, members['desklib']['P'])]))
    configs = {name: [(1.0, Psvc)] + [(w[n], members[n]['P']) for n in members] for name, w in WEIGHTS.items()}
    configs['svc + modernbert 1.0 + desklib 0.5'] = [(1.0, Psvc)] + [(dict(modernbert=1.0, desklib=0.5)[n], members[n]['P'])
                                                                    for n in members]
    base_pred = blend([(1.0, Psvc)])
    preds = {name: blend(parts) for name, parts in configs.items()}
    for name, pr in preds.items():
        res[f'svc + OURS [{name}] + sinkhorn'] = balanced_accuracy_score(y, pr)

    # Paired bootstrap over prompts: resample the 150 prompts, rescore both (predictions fixed, Sinkhorn already done).
    rng = np.random.default_rng(0)
    groups = train.prompt_id.values
    ug = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in ug}
    boot = {}
    for name, pr in preds.items():
        d = []
        for _ in range(a.boot):
            idx = np.concatenate([gidx[g] for g in rng.choice(ug, len(ug))])
            d.append(balanced_accuracy_score(y[idx], pr[idx]) - balanced_accuracy_score(y[idx], base_pred[idx]))
        d = np.array(d)
        boot[name] = dict(mean_gain=float(d.mean()), p_better=float((d > 0).mean()),
                          ci90=[float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))])
    for k, v in res.items():
        print(f'{k:60s} {v:.4f}')
    for k, v in boot.items():
        print(f'bootstrap [{k}] gain {v["mean_gain"]:+.4f}, 90% CI {v["ci90"][0]:+.4f}..{v["ci90"][1]:+.4f}, '
              f'P(better) {v["p_better"]:.3f}')
    json.dump(dict(scores=res, bootstrap=boot), open(os.path.join(a.out, 'blend_oof.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
