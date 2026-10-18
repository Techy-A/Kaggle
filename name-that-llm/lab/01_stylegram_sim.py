"""Step 1: the stylegram member of the 0.556 notebook on a simulated hidden test (CPU, no submissions).

The 0.556 blend is dominated by the stylegram (char/word/shape/function-word TF-IDF + LinearSVC, blend weight 1.52 against
<= 0.19 for every other member), so changes to it are tested here first. The hidden test has only ~2-3 passages per
prompt, while train has all 18 models on every prompt, so plain prompt-grouped CV says nothing about prompt-centering.
This script rebuilds the test's structure:

  * 5 prompt-grouped folds; per fold a plain SVC and centered twins, trained on the other prompts only;
  * a pseudo-test: for every prompt, K passages from random models (K drawn from --sizes), the whole pool clustered
    with the notebook's own topic clustering (median-normalized content-word TF-IDF, average linkage, cut 0.90);
  * each pseudo-test row predicted by the models of the fold that held its prompt out;
  * balanced accuracy over all pseudo-test rows, averaged over --draws pools (same pools for every variant).

Variants (applied to rows in clusters of 2-18 passages; other rows keep the plain head):
  plain     plain SVC only (no centering)
  hard      the 0.556 notebook: replace by the centered twin (trained on full 18-model prompt groups)
  small     like hard, but the twin is trained on centered random subsets of 2-3 passages per prompt, the way the
            test's clusters look
  add_b     plain log-probability + b x centered log-probability (both kept, b in --betas)

Usage: python 01_stylegram_sim.py [--draws 5] [--sizes 2 3] [--betas 0.5 1.0]
Output: one line per variant (mean and sd of balanced accuracy over draws) + cluster pair precision/recall.
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags, hstack
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

# ---- the notebook's stylegram, unchanged (FUNCTION_WORDS, SPEC, shape_text, function_text, clustering) ----
FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
SPEC = [('char', 3, 5, False, 250000, 1.0), ('word', 1, 3, False, 180000, 0.7),
        ('shape', 2, 6, False, 60000, 0.5), ('function', 1, 3, False, 50000, 0.5)]


def shape_text(text):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", text)


def function_text(text):
    return " ".join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else "CONTENT"
                    for t in re.findall(r"\w+|[^\w\s]", text))


def stylegram_features(texts):
    blocks = []
    for analyzer, lo, hi, lower, maxf, w in SPEC:
        tf = shape_text if analyzer == 'shape' else function_text if analyzer == 'function' else str
        actual = 'char' if analyzer == 'shape' else 'word' if analyzer == 'function' else analyzer
        opts = dict(analyzer=actual, ngram_range=(lo, hi), lowercase=lower, sublinear_tf=True, min_df=2,
                    max_features=maxf, dtype=np.float64)
        if actual == 'word':
            opts['token_pattern'] = r"(?u)\b\w+\b|[^\w\s]"
        blocks.append(TfidfVectorizer(**opts).fit_transform([tf(t) for t in texts]) * w)
    return hstack(blocks, format='csr')


def center(X, groups, min_size=2, max_size=18):
    """Subtract each group's mean (groups outside the size range unchanged), as _stylegram_center_sparse."""
    X = csr_matrix(X)
    _, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    valid = np.flatnonzero((cnt >= min_size) & (cnt <= max_size))
    remap = np.full(len(cnt), -1)
    remap[valid] = np.arange(len(valid))
    mapped = remap[inv]
    rows = np.flatnonzero(mapped >= 0)
    M = csr_matrix((np.ones(len(rows)), (rows, mapped[rows])), shape=(X.shape[0], len(valid)))
    return (X - M @ (diags(1.0 / cnt[valid]) @ (M.T @ X))).tocsr()


def cluster_ids(texts, threshold=0.90):
    """The notebook's _stylegram_cluster_ids: median-normalized content-word TF-IDF distances, average linkage."""
    xt = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_features=100_000, dtype=np.float32).fit_transform(texts)
    d = np.clip(1.0 - (xt @ xt.T).toarray(), 0.0, 1.0)
    np.fill_diagonal(d, 0.0)
    d /= max(float(np.median(d[np.triu_indices(len(texts), 1)])), 1e-6)
    np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(d, checks=False), method='average'), threshold, criterion='distance')


def svc_logp(model, X, K):
    """The notebook's probability mapping: softmax of the SVC decision values."""
    s = model.decision_function(X)
    s = np.exp(s - s.max(1, keepdims=True))
    p = np.zeros((X.shape[0], K))
    p[:, model.classes_] = s / s.sum(1, keepdims=True)
    return np.log(np.clip(p, 1e-9, 1))


def small_groups(prompts, sizes, rng):
    """Split every prompt's rows into random groups of the given sizes (the test-like centering target)."""
    g, nxt = np.empty(len(prompts), int), 0
    for p in np.unique(prompts):
        idx = rng.permutation(np.flatnonzero(prompts == p))
        while len(idx):
            k = int(rng.choice(sizes))
            g[idx[:k]], idx, nxt = nxt, idx[k:], nxt + 1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--sizes', type=int, nargs='+', default=[2, 3])
    ap.add_argument('--betas', type=float, nargs='+', default=[0.5, 1.0])
    args = ap.parse_args()

    train = pd.read_csv('data/train.csv')
    labels = sorted(train.model.unique())
    y = train.model.map({l: i for i, l in enumerate(labels)}).values
    pid = train.prompt_id.to_numpy(dtype=object)
    K = len(labels)
    # ponytail: one TF-IDF fit on all train text (vocabulary and idf only, no labels), so every fold and every cluster
    # share a feature space; the notebook fits on train and applies to test, which this mirrors transductively.
    X = stylegram_features(train.text.tolist())
    fold = np.empty(len(train), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(X, y, pid)):
        fold[va] = k

    models = []   # per fold: plain, hard-centered twin (full prompt groups), small-group twin
    for k in range(5):
        tr = fold != k
        rng = np.random.default_rng(k)
        fit = lambda A: LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(A, y[tr])
        models.append(dict(plain=fit(X[tr]), hard=fit(center(X[tr], pid[tr])),
                           small=fit(center(X[tr], small_groups(pid[tr], args.sizes, rng)))))
        print(f'fold {k} fitted', flush=True)

    scores = {}
    prec, rec = [], []
    for d in range(args.draws):
        rng = np.random.default_rng(100 + d)
        rows = np.concatenate([rng.choice(np.flatnonzero(pid == p), int(rng.choice(args.sizes)), replace=False)
                               for p in np.unique(pid)])
        g = cluster_ids(train.text.to_numpy(dtype=object)[rows].tolist())
        size = np.bincount(g)[g]
        on = (size >= 2) & (size <= 18)
        same_c, same_p = g[:, None] == g[None, :], pid[rows][:, None] == pid[rows][None, :]
        off = ~np.eye(len(rows), dtype=bool)
        prec.append((same_c & same_p & off).sum() / max((same_c & off).sum(), 1))
        rec.append((same_c & same_p & off).sum() / max((same_p & off).sum(), 1))

        Xc = center(X[rows], g)   # cluster-centered pseudo-test features (clusters outside 2-18 unchanged)
        lp = {v: np.zeros((len(rows), K)) for v in ('plain', 'hard', 'small')}
        for k in range(5):
            m = fold[rows] == k
            for v in lp:
                lp[v][m] = svc_logp(models[k][v], X[rows][m] if v == 'plain' else Xc[m], K)
        yt = y[rows]
        out = {'plain': lp['plain'].argmax(1)}
        for v in ('hard', 'small'):
            out[v] = np.where(on, lp[v].argmax(1), out['plain'])
            for b in (args.betas if v == 'hard' else []):
                out[f'add_{v}_{b:g}'] = np.where(on, (lp['plain'] + b * lp[v]).argmax(1), out['plain'])
        for v, pred in out.items():
            scores.setdefault(v, []).append(balanced_accuracy_score(yt, pred))
        print(f'draw {d}: {len(rows)} passages, clustered rows {on.sum()}, pair precision {prec[-1]:.3f} '
              f'recall {rec[-1]:.3f}', flush=True)

    base = np.array(scores['plain'])
    print(f'\nsizes {args.sizes}, {args.draws} draws (paired deltas vs plain):')
    for v, s in scores.items():
        s = np.array(s)
        print(f'  {v:16s} balanced acc {s.mean():.4f} (sd {s.std():.4f}) | delta {np.mean(s - base):+.4f} '
              f'(wins {int((s > base).sum())}/{len(s)})')


if __name__ == '__main__':
    main()
