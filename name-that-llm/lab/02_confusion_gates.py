"""Step 2: where the stylegram's errors are (18x18 confusion by lab) and whether within-lab gates fix them (CPU).

Same simulated hidden test as 01_stylegram_sim.py (prompt-grouped folds, 2-3 passages per prompt, the notebook's
clustering), with the best stylegram from step 1 as the base: plain + 1.0 x centered evidence on clustered rows.

  * Confusion matrix pooled over --draws pools, rows/columns sorted by lab; how many errors stay inside the lab.
  * Within-lab gates (the 0.556 notebook already has one for Claude): when the base predicts a label of a lab with 2+
    models, a LinearSVC trained only on that lab's training rows re-decides inside the lab, scored as
    base log-probability (restricted to the lab) + lam x gate log-probability, lam in --lams.

Usage: python 02_confusion_gates.py [--draws 5] [--lams 0.5 1 2]
Output: the confusion matrix (confusion_add.csv), per-class recall, within-lab error share, gate deltas per lab.
"""
import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('sim', os.path.join(HERE, '01_stylegram_sim.py'))
sim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sim)   # reuse step 1's stylegram, centering, clustering and probability mapping


def lab_of(label):
    return label.split('-')[0]   # gpt, gemini, claude, glm, kimi, qwen3.8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=5)
    ap.add_argument('--lams', type=float, nargs='+', default=[0.5, 1.0, 2.0])
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(HERE, 'data/train.csv'))
    labels = sorted(train.model.unique())
    K = len(labels)
    y = train.model.map({l: i for i, l in enumerate(labels)}).values
    pid = train.prompt_id.to_numpy(dtype=object)
    texts = train.text.to_numpy(dtype=object)
    labs = sorted({lab_of(l) for l in labels})
    lab_idx = {b: np.array([i for i, l in enumerate(labels) if lab_of(l) == b]) for b in labs}
    gated = [b for b in labs if len(lab_idx[b]) > 1]

    X = sim.stylegram_features(texts.tolist())
    fold = np.empty(len(train), int)
    for k, (_, va) in enumerate(GroupKFold(5).split(X, y, pid)):
        fold[va] = k
    svc = lambda A, t: LinearSVC(C=0.5, random_state=42, max_iter=4000).fit(A, t)
    models = []
    for k in range(5):
        tr = fold != k
        m = dict(plain=svc(X[tr], y[tr]), hard=svc(sim.center(X[tr], pid[tr]), y[tr]))
        for b in gated:   # within-lab gate on plain features, the lab's training rows only
            r = tr & np.isin(y, lab_idx[b])
            m[b] = svc(X[r], y[r])
        models.append(m)
        print(f'fold {k} fitted', flush=True)

    Y, BASE, GATED = [], [], {(b, lam): [] for b in gated for lam in args.lams}
    for d in range(args.draws):
        rng = np.random.default_rng(100 + d)
        rows = np.concatenate([rng.choice(np.flatnonzero(pid == p), int(rng.choice([2, 3])), replace=False)
                               for p in np.unique(pid)])
        g = sim.cluster_ids(texts[rows].tolist())
        size = np.bincount(g)[g]
        on = (size >= 2) & (size <= 18)
        Xc = sim.center(X[rows], g)
        base = np.zeros((len(rows), K))
        gate = {b: np.full((len(rows), K), -np.inf) for b in gated}
        for k in range(5):
            m = fold[rows] == k
            lp = sim.svc_logp(models[k]['plain'], X[rows][m], K)
            lc = sim.svc_logp(models[k]['hard'], Xc[m], K)
            base[m] = lp + np.where(on[m, None], lc, 0.0)
            for b in gated:
                gate[b][m] = sim.svc_logp(models[k][b], X[rows][m], K)
        pred = base.argmax(1)
        Y.append(y[rows]); BASE.append(pred)
        for b in gated:
            hit = np.isin(pred, lab_idx[b])
            for lam in args.lams:
                s = base[:, lab_idx[b]] + lam * gate[b][:, lab_idx[b]]
                GATED[b, lam].append(np.where(hit, lab_idx[b][s.argmax(1)], pred))
        print(f'draw {d} done', flush=True)

    # ---- confusion by lab (pooled over draws) ----
    yt, yp = np.concatenate(Y), np.concatenate(BASE)
    order = [i for b in labs for i in lab_idx[b]]
    cm = confusion_matrix(yt, yp, labels=order)
    names = [labels[i] for i in order]
    pd.DataFrame(cm, index=names, columns=names).to_csv(os.path.join(HERE, 'confusion_add.csv'))
    err = yt != yp
    same_lab = np.array([lab_of(labels[a]) == lab_of(labels[b]) for a, b in zip(yt, yp)])
    print(f'\nbase (plain + centered evidence): balanced acc {balanced_accuracy_score(yt, yp):.4f} over {len(yt)} rows')
    print(f'errors {err.sum()}: inside the true lab {(err & same_lab).sum()} ({(err & same_lab).sum() / err.sum():.0%}), '
          f'across labs {(err & ~same_lab).sum()}')
    print('\nper-class recall (sorted by lab) and the most common wrong label:')
    for i in order:
        r = yt == i
        wrong = pd.Series(yp[r & err]).map(lambda j: labels[j]).value_counts()
        top = f'{wrong.index[0]} ({wrong.iloc[0]})' if len(wrong) else '-'
        print(f'  {labels[i]:24s} recall {np.mean(yp[r] == i):.2f}   most confused with {top}')
    print('\nlab-level accuracy (is the predicted lab right?):', f'{np.mean(~err | same_lab):.3f}')

    # ---- gates ----
    base_acc = balanced_accuracy_score(yt, yp)
    print('\nwithin-lab gates (delta in balanced accuracy vs base, pooled; per-draw wins):')
    per_draw_base = [balanced_accuracy_score(a, b) for a, b in zip(Y, BASE)]
    for b in gated:
        for lam in args.lams:
            gp = np.concatenate(GATED[b, lam])
            wins = sum(balanced_accuracy_score(a, c) > p for a, c, p in zip(Y, GATED[b, lam], per_draw_base))
            print(f'  {b:8s} lam {lam:<4g} {balanced_accuracy_score(yt, gp) - base_acc:+.4f}  (wins {wins}/{args.draws})')


if __name__ == '__main__':
    main()
