"""Step 11: Sinkhorn variants on our own out-of-fold blend: argmax, 1/3/10/50/300 iterations, cross-fitted offsets (CPU).

What: builds the best honest prompt-grouped OOF blend the lab holds (n-gram SVC + modernbert + desklib at the
nuridudak 0.626 weights, three token-attention members at 0.3, and the 5-fold LoRA24 plain + centred readouts), then
scores, as the notebooks do (Sinkhorn over all rows scored together):
- plain argmax;
- Sinkhorn with 1, 3, 10, 50 and 300 iterations (300 is the notebooks' current step);
- cross-fitted per-class log-offsets (GoPain, thread 752365): the offsets are the log Sinkhorn column scalings learned on
  the other four folds' OOF rows and applied to the held-out fold (no row sees its own offsets).
Evaluated (a) on the balanced OOF (2,699 rows; paired bootstrap over prompts for P(better) vs 300 iterations) and
(b) on simulated uneven test pools (~2,000 rows drawn from the OOF with class mix Dirichlet(alpha), alpha 1, 5, 20 and
balanced), many draws, reporting balanced accuracy and regret vs an oracle that knows the true class mix
(argmax of logP + log(pi_true), the Bayes prior-shift correction), with a paired draw-level P(better) vs 300 iterations.
Why: every notebook in the chain ends with a 300-iteration Sinkhorn; partial or offset balancing may lose less on uneven
test mixes. This is a correction step, the one kind of change the project rules allow.
Caveats: NTL has no length crops (that is an SHL notion); rows are full passages. The real 0.639 members (14B LoRAs,
ATLAS) have no OOF, so the blend here is weaker (about 0.6) and leans differently; the answer transfers only loosely.
Output: <out>/lora24_oof.npz and <out>/sinkhorn_variants.json (cached; delete to refit) and the printed tables.
Usage: python 11_sinkhorn_variants.py [--oof data/oof] [--train data/train.csv] [--out artifacts] [--draws 200]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score as bacc

K = 18
ITERS = [1, 3, 10, 50, 300]
NAMES = ['argmax'] + [f'sk{i}' for i in ITERS] + ['offsets']
W = dict(svc=1.0, modernbert=0.40, desklib=0.10, tok=0.3, lora_plain=1.0, lora_centred=1.0)


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def sinkhorn_all(P, iters=ITERS):
    """Same iteration as 08/10; returns {k: probabilities after k iterations} and the final column scalings c."""
    c = np.ones(K)
    target = len(P) / K
    out = {}
    for k in range(1, max(iters) + 1):
        R = P * c
        R /= R.sum(1, keepdims=True)
        c *= target / R.sum(0)
        if k in iters:
            out[k] = c.copy()
    return out


def preds_for(L, P=None):
    """Predictions of every variant (except offsets) over the rows scored together; L = blended log-probabilities."""
    P = softmax(L)
    cs = sinkhorn_all(P)
    r = {'argmax': L.argmax(1)}
    for k in ITERS:
        r[f'sk{k}'] = (np.log(P + 1e-30) + np.log(cs[k])).argmax(1)
    return r


def lora_oof(oof, train, path):
    if os.path.exists(path):
        z = np.load(path)
        return z['plain'], z['centred']
    lab = sorted(train.model.unique())
    pid = train.prompt_id.values
    idx = {i: k for k, i in enumerate(train.id.astype(str))}
    Lp, Lc = np.zeros((len(train), K)), np.zeros((len(train), K))
    for f in range(5):
        d = f'{oof}/lora/lora24/outer{f}/'
        z, r = np.load(d + 'validation_features.npz', allow_pickle=True), np.load(d + 'readouts.npz', allow_pickle=True)
        assert list(r['classes']) == lab
        X = z['features']
        ii = np.array([idx[str(i)] for i in z['ids']])
        Xc = X.copy()
        for p in np.unique(pid[ii]):   # centre by prompt, as the centred head was trained
            m = pid[ii] == p
            Xc[m] -= X[m].mean(0)
        Lp[ii] = X @ r['plain_coef'].T + r['plain_intercept']
        Lc[ii] = Xc @ r['center_coef'].T + r['center_intercept']
    np.savez(path, plain=Lp, centred=Lc)
    return Lp, Lc


def load_blend(a):
    train = pd.read_csv(a.train)
    lab = sorted(train.model.unique())
    y = train.model.map({l: i for i, l in enumerate(lab)}).values
    ids = train.id.astype(str).tolist()
    parts, single = [], {}
    z = np.load(f'{a.oof}/encrebuild/oof_svc.npz', allow_pickle=True)
    assert list(z['ids']) == ids
    folds = z['fold']
    single['svc'] = softmax(z['scores'])
    for n in ('modernbert', 'desklib'):
        z = np.load(f'{a.oof}/encrebuild/oof_{n}.npz', allow_pickle=True)
        assert list(z['ids']) == ids and list(z['fold']) == list(folds)
        single[n] = softmax(z['oof'].astype(np.float64))
    for n, f in [('tok1', 'tokattn/tokattn_oof'), ('tok2', 'tokattn2/tokattn_q8l22_oof'), ('tok3', 'tokattn2/tokattn_debl23_oof')]:
        z = np.load(f'{a.oof}/{f}.npz', allow_pickle=True)
        assert list(z['ids']) == ids and list(z['classes']) == lab
        single[n] = z['probs'].astype(np.float64)
    Lp, Lc = lora_oof(a.oof, train, os.path.join(a.out, 'lora24_oof.npz'))
    single['lora_plain'], single['lora_centred'] = softmax(Lp), softmax(Lc)
    wt = dict(svc=W['svc'], modernbert=W['modernbert'], desklib=W['desklib'], tok1=W['tok'], tok2=W['tok'],
              tok3=W['tok'], lora_plain=W['lora_plain'], lora_centred=W['lora_centred'])
    L = sum(wt[n] * np.log(np.clip(single[n], 1e-9, 1)) for n in wt)
    return train, y, folds, L, {n: bacc(y, P.argmax(1)) for n, P in single.items()}


def crossfit_offsets(L, folds):
    """Per row: log column scalings learned (300-iteration Sinkhorn) on the other four folds' rows."""
    B = np.zeros_like(L)
    for f in np.unique(folds):
        c = sinkhorn_all(softmax(L[folds != f]), [300])[300]
        B[folds == f] = np.log(c)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof', default='data/oof')
    ap.add_argument('--train', default='data/train.csv')
    ap.add_argument('--out', default='artifacts')
    ap.add_argument('--draws', type=int, default=200)
    ap.add_argument('--pool', type=int, default=2000)
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, 'sinkhorn_variants.json')
    if os.path.exists(path):
        print(open(path).read())
        return
    train, y, folds, L, singles = load_blend(a)
    n = len(y)
    B = crossfit_offsets(L, folds)
    res = {'members_argmax_bacc': singles}
    print('members (argmax bacc):', {k: round(v, 4) for k, v in singles.items()})

    # (a) balanced OOF: all rows scored together; bootstrap over prompts vs sk300
    pr = preds_for(L)
    pr['offsets'] = (L + B).argmax(1)
    res['balanced'] = {k: bacc(y, p) for k, p in pr.items()}
    rng = np.random.default_rng(0)
    groups = train.prompt_id.values
    ug = np.unique(groups)
    gidx = [np.where(groups == g)[0] for g in ug]
    D = {k: [] for k in NAMES}
    for _ in range(a.boot):
        idx = np.concatenate([gidx[i] for i in rng.integers(0, len(ug), len(ug))])
        base = bacc(y[idx], pr['sk300'][idx])
        for k in NAMES:
            D[k].append(bacc(y[idx], pr[k][idx]) - base)
    res['balanced_boot_vs_sk300'] = {k: dict(gain=float(np.mean(v)), p_better=float((np.array(v) > 0).mean()),
                                             ci90=[float(np.quantile(v, .05)), float(np.quantile(v, .95))])
                                     for k, v in D.items() if k != 'sk300'}
    print('\nbalanced OOF, bacc and paired bootstrap over prompts vs sk300')
    for k in NAMES:
        b = res['balanced_boot_vs_sk300'].get(k)
        print(f'  {k:8s} {res["balanced"][k]:.4f}' + ('' if b is None else
              f'  gain {b["gain"]:+.4f} CI {b["ci90"][0]:+.4f}..{b["ci90"][1]:+.4f} P(better) {b["p_better"]:.3f}'))

    # (b) simulated uneven pools
    byclass = [np.where(y == c)[0] for c in range(K)]
    sim = {}
    for alpha in (1, 5, 20, None):
        rng = np.random.default_rng(1)
        S = {k: [] for k in NAMES + ['oracle']}
        for _ in range(a.draws):
            pi = np.full(K, 1 / K) if alpha is None else rng.dirichlet(np.full(K, alpha))
            cnt = rng.multinomial(a.pool, pi)
            idx = np.concatenate([rng.choice(byclass[c], cnt[c]) for c in range(K) if cnt[c]])
            Ls, ys = L[idx], y[idx]
            p = preds_for(Ls)
            p['offsets'] = (Ls + B[idx]).argmax(1)
            p['oracle'] = (Ls + np.log(np.clip(pi, 1e-12, 1))).argmax(1)
            for k in S:
                S[k].append(bacc(ys, p[k]))
        S = {k: np.array(v) for k, v in S.items()}
        key = 'balanced' if alpha is None else f'alpha{alpha}'
        sim[key] = {k: dict(bacc=float(v.mean()), regret=float((S['oracle'] - v).mean()),
                            gain_vs_sk300=float((v - S['sk300']).mean()),
                            p_better=float((v > S['sk300']).mean() + 0.5 * (v == S['sk300']).mean()))
                    for k, v in S.items()}
        print(f'\nsim {key}: pool {a.pool}, {a.draws} draws (bacc / regret vs oracle / gain vs sk300 / P(draw better))')
        for k in NAMES + ['oracle']:
            s = sim[key][k]
            print(f'  {k:8s} {s["bacc"]:.4f}  {s["regret"]:.4f}  {s["gain_vs_sk300"]:+.4f}  {s["p_better"]:.2f}')
    res['sim'] = sim
    json.dump(res, open(path, 'w'), indent=1)


if __name__ == '__main__':
    main()
