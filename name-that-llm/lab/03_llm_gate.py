"""Step 3: does a GLM-vs-Claude gate on frozen LLM features fix the biggest cross-lab confusion? (CPU)

Step 2 showed GLM and Claude blur into each other, and that gates retrained on the stylegram's own features add
nothing. This gate uses different information: the frozen Qwen3-8B (L7+L22) and GLM-4-9B (L8+L3+L24) features the 0.556
notebook already computes (cached once by jobs/ntl_featcache), so it costs no extra GPU at inference.

Same simulated hidden test as steps 1-2 (prompt-grouped folds, 2-3 passages per prompt, the notebook's clustering), base
= stylegram plain + centered evidence. Per fold, a StandardScaler + LogisticRegression (C as the notebook's LM heads) is
trained on the GLM + Claude training rows only; when the base predicts one of those 7 labels, the 7-way decision becomes
base log-probability + lam x gate log-probability. Also reported: each LLM head as a full 18-way member added to the
base (log-probability x w), to see how much these features know beyond the stylegram.

Usage: python 03_llm_gate.py [--draws 5] [--lams 0.5 1 2] [--feats q3_8b glm9b both]
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sim', os.path.join(HERE, '01_stylegram_sim.py'))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)


def lr_logp(model, E, K):
    p = np.full((len(E), K), 1e-9)
    p[:, model.classes_] = np.clip(model.predict_proba(E), 1e-9, 1)
    return np.log(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--lams', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    ap.add_argument('--feats', nargs='+', default=['q3_8b', 'glm9b', 'both'])
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(HERE, 'data/train.csv'))
    z = np.load(os.path.join(HERE, 'data/featcache/ntl_feats.npz'), allow_pickle=True)
    assert (z['ids'] == train['id'].values).all()
    E = {'q3_8b': z['q3_8b'].astype(np.float32), 'glm9b': z['glm9b'][:, :8192].astype(np.float32)}   # glm: L8+L3
    E['both'] = np.hstack([E['q3_8b'], E['glm9b']])
    labels = sorted(train.model.unique())
    K = len(labels)
    y = train.model.map({l: i for i, l in enumerate(labels)}).values
    pid = train.prompt_id.to_numpy(dtype=object)
    texts = train.text.to_numpy(dtype=object)
    gc_idx = np.array([i for i, l in enumerate(labels) if l.startswith(('glm', 'claude'))])

    X = sim.stylegram_features(texts.tolist())
    fold = np.empty(len(train), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(X, y, pid)):
        fold[va] = k
    head = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.0025, max_iter=2000))
    models = []
    for k in range(5):
        tr = fold != k
        r = tr & np.isin(y, gc_idx)
        m = dict(plain=LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(X[tr], y[tr]),
                 hard=LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(sim.center(X[tr], pid[tr]), y[tr]))
        for f in args.feats:
            m[f'gate_{f}'] = head().fit(E[f][r], y[r])
            m[f'full_{f}'] = head().fit(E[f][tr], y[tr])
        models.append(m)
        print(f'fold {k} fitted', flush=True)

    res = {}
    for d in range(args.draws):
        rng = np.random.default_rng(100 + d)
        rows = np.concatenate([rng.choice(np.flatnonzero(pid == p), int(rng.choice([2, 3])), replace=False)
                               for p in np.unique(pid)])
        g = sim.cluster_ids(texts[rows].tolist())
        size = np.bincount(g)[g]
        on = (size >= 2) & (size <= 18)
        Xc = sim.center(X[rows], g)
        base = np.zeros((len(rows), K))
        lp = {key: np.zeros((len(rows), K)) for key in models[0] if key.startswith(('gate_', 'full_'))}
        for k in range(5):
            m = fold[rows] == k
            base[m] = sim.svc_logp(models[k]['plain'], X[rows][m], K) + \
                np.where(on[m, None], sim.svc_logp(models[k]['hard'], Xc[m], K), 0.0)
            for key in lp:
                lp[key][m] = lr_logp(models[k][key], E[key.split('_', 1)[1]][rows][m], K)
        yt, pred = y[rows], base.argmax(1)
        out = {'base': pred}
        hit = np.isin(pred, gc_idx)
        for f in args.feats:
            for lam in args.lams:
                s = base[:, gc_idx] + lam * lp[f'gate_{f}'][:, gc_idx]
                out[f'gate_{f}_lam{lam:g}'] = np.where(hit, gc_idx[s.argmax(1)], pred)
                out[f'full_{f}_w{lam:g}'] = (base + lam * lp[f'full_{f}']).argmax(1)
        for key, p in out.items():
            res.setdefault(key, []).append(balanced_accuracy_score(yt, p))
        print(f'draw {d} done', flush=True)

    b = np.array(res['base'])
    print(f'\nbase (stylegram plain + centered): {b.mean():.4f}')
    for key, s in res.items():
        s = np.array(s)
        print(f'  {key:22s} {s.mean():.4f}  delta {np.mean(s - b):+.4f}  (wins {int((s > b).sum())}/{len(s)})')


if __name__ == '__main__':
    main()
