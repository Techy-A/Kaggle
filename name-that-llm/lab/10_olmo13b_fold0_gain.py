"""Step 10: how much does the OLMo-2-13B member add to the stack, on fold 0 only (CPU, seconds).

What: takes the OLMo fold-0 OOF logits (plain head, triplet-centred head; 540 rows, 30 per class) and the stack proxy
that exists out of fold in the lab: SVC (08's rebuild) + modernbert + desklib at the nuridudak 0.626 weights (08's WEIGHTS)
+ the two token-attention members, all on the same prompt-grouped fold 0 rows. Blends in log space, Sinkhorn over the 540
rows as in 08, and scores balanced accuracy for W (plain head weight) x TWIN_W (centred head weight). A paired bootstrap
over prompts gives P(better) against W=0 / TWIN_W=0.
Why: the 0.639 notebook's own members have no OOF, so the proxy is weaker (about 0.58-0.60, not 0.639) and the gain over
it is an upper bound on the gain over the real stack. Weights are in the proxy's scale (SVC 1.0); the notebook's scale is
not identical, so W is a guide, not a transfer. Fold 0 is 540 rows: the noise floor is about +-0.01.
Output: <out>/olmo13b_fold0_gain.json (cached; delete to refit) and the printed table.
Usage: python 10_olmo13b_fold0_gain.py [--olmo DIR] [--oof data/oof] [--out artifacts] [--boot 2000]
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score as bacc

K = 18
WS, TWS = [0, 0.03, 0.05, 0.1, 0.15, 0.2], [0, 0.03, 0.05, 0.1]
OURS = dict(modernbert=0.40, desklib=0.10)   # 08_blend_oof.py WEIGHTS['nuridudak 0.626']
TOK = 0.3                                    # each token-attention member (proxy choice, not tuned)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn(P, iters=300):
    c = np.ones(K)
    for _ in range(iters):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= (len(P) / K) / R.sum(0)
    R = P * c
    return R / R.sum(1, keepdims=True)


def pick(z, ids, key):
    m = {i: j for j, i in enumerate(z['ids'])}
    return z[key][[m[i] for i in ids]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--olmo', default='/home/user/work/ds_ntl_olmo13b_member/oof_fold0.npz')
    ap.add_argument('--oof', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/oof'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts'))
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'olmo13b_fold0_gain.json')
    if os.path.exists(path):
        print(open(path).read())
        return
    o = np.load(a.olmo, allow_pickle=True)
    ids = o['ids']
    olabels = [str(l) for l in o['labels']]
    labels = sorted(olabels)   # the lab's members use alphabetical order; the OLMo run uses the notebook's own order
    perm = [olabels.index(l) for l in labels]
    y = np.array([labels.index(olabels[i]) for i in o['y']])
    Pp, Pc = softmax(o['plain'].astype(float)[:, perm]), softmax(o['centred'].astype(float)[:, perm])
    enc = os.path.join(a.oof, 'encrebuild')
    base = [(1.0, softmax(pick(np.load(f'{enc}/oof_svc.npz'), ids, 'scores')))]
    for n, w in OURS.items():
        z = np.load(f'{enc}/oof_{n}.npz')
        assert [str(l) for l in z['labels']] == labels
        base.append((w, softmax(pick(z, ids, 'oof').astype(float))))
    for f in ('tokattn/tokattn_oof.npz', 'tokattn2/tokattn_q8l22_oof.npz', 'tokattn2/tokattn_debl23_oof.npz'):
        z = np.load(os.path.join(a.oof, f), allow_pickle=True)
        if [str(l) for l in z['classes']] == labels:
            base.append((TOK, np.clip(pick(z, ids, 'probs'), 1e-9, 1)))
    # prompt ids for the grouped bootstrap
    pid = dict(zip(*(lambda z: (z['ids'], z['prompt_id']))(np.load(f'{enc}/oof_modernbert.npz'))))
    groups = np.array([pid[i] for i in ids])
    lb = sum(w * np.log(np.clip(P, 1e-9, 1)) for w, P in base)

    def pred(w, tw):
        return sinkhorn(softmax(lb + w * np.log(np.clip(Pp, 1e-9, 1)) + tw * np.log(np.clip(Pc, 1e-9, 1)))).argmax(1)

    res = {'olmo plain alone': bacc(y, Pp.argmax(1)), 'olmo centred alone': bacc(y, Pc.argmax(1)),
           'proxy stack (members=%d)' % len(base): bacc(y, pred(0, 0))}
    grid = {(w, tw): pred(w, tw) for w in WS for tw in TWS}
    rng = np.random.default_rng(0)
    ug = np.unique(groups)
    gidx = {g: np.where(groups == g)[0] for g in ug}
    draws = [np.concatenate([gidx[g] for g in rng.choice(ug, len(ug))]) for _ in range(a.boot)]
    b0 = np.array([bacc(y[i], grid[(0, 0)][i]) for i in draws])
    table = []
    for (w, tw), pr in grid.items():
        d = np.array([bacc(y[i], pr[i]) for i in draws]) - b0
        table.append(dict(W=w, TWIN_W=tw, bacc=float(bacc(y, pr)), gain=float(bacc(y, pr) - res['proxy stack (members=%d)' % len(base)]),
                          p_better=float((d > 0).mean()), ci90=[float(np.quantile(d, .05)), float(np.quantile(d, .95))]))
    out = dict(scores={k: float(v) for k, v in res.items()}, grid=table, n_groups=int(len(ug)))
    json.dump(out, open(path, 'w'), indent=1)
    for k, v in res.items():
        print(f'{k:35s} {v:.4f}')
    print(f'{"W":>5} {"TW":>5} {"bacc":>7} {"gain":>7} {"P(b)":>6}  ci90')
    for r in table:
        print(f'{r["W"]:5.2f} {r["TWIN_W"]:5.2f} {r["bacc"]:7.4f} {r["gain"]:+7.4f} {r["p_better"]:6.3f}  {r["ci90"][0]:+.3f}..{r["ci90"][1]:+.3f}')


if __name__ == '__main__':
    main()
