"""Tune the prior weights on saved per-candidate features from compare_models.py.

Usage: python tune_prior.py <work_dir> <model_name>
Ranks by z(fingerprint log-likelihood) + w_pop * popularity + w_np * natural-product flag, de-duplicated by
InChIKey14, and reports MRR@25 on S1 (library natural products: flatters priors) and S2 (no public spectra:
penalises them). Weights that hold up on both are the safer choice for the hidden test.
"""
import sys

import numpy as np
import pandas as pd


def reciprocal_rank(feat, score):
    best = pd.Series(score).groupby(feat['ids']).max()
    if feat['key'] not in best.index:
        return 0.0
    rank = int((best > best[feat['key']]).sum()) + 1
    return 1.0 / rank if rank <= 25 else 0.0


def main(work, model):
    feats = pd.read_pickle(f'{work}/compare_features.pkl')
    out = []
    for w_pop in (0, 0.1, 0.25, 0.5, 0.75, 1.0):
        for w_np in (0, 0.5, 1, 2):
            rr = {}
            for f in feats:
                fp = f['fp'][model]
                z = (fp - fp.mean()) / (fp.std() + 1e-9)
                rr.setdefault(f['split'], []).append(reciprocal_rank(f, z + w_pop * f['pop'] + w_np * f['natural']))
            row = dict(w_pop=w_pop, w_np=w_np, **{k: np.mean(v) for k, v in rr.items()})
            out.append(row)
    res = pd.DataFrame(out)
    s1, s2 = [c for c in res.columns if c.startswith('S1')][0], [c for c in res.columns if c.startswith('S2')][0]
    res['mean'] = (res[s1] + res[s2]) / 2
    res['worse_split'] = res[[s1, s2]].min(axis=1)
    print(res.round(3).sort_values('mean', ascending=False).to_string(index=False))
    res.to_csv(f'{work}/tune_prior.csv', index=False)


if __name__ == '__main__':
    main(*sys.argv[1:3])
