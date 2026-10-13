"""Tune extra per-candidate terms (fragmentation, analogs) on top of the hand prior, as pipeline.prior_order applies them.

Usage: python tune_extra.py <work_dir> [model_name]
Score per candidate row: z(fp) + w_pop * pop + w_np * natural + sum_k w_k * z_k, where z_k is a feature z-scored
over the candidates that have it within the molecule (others get MISSING). One score per InChIKey14 (max).
MRR@25 on S1 (library natural products) and S2 (enveda-180 without public spectra), answers only via PubChem.
"""
import itertools
import sys

import numpy as np
import pandas as pd

MISSING = -3.0


def zmap(df, col):
    """{(qid, id): z} for one feature column, z-scored within each molecule."""
    g = df.groupby('qid')[col]
    z = (df[col] - g.transform('mean')) / (g.transform('std', ddof=0) + 1e-9)
    return dict(zip(zip(df.qid, df.id), z))


def evaluate(feats, model, w_pop, w_np, extras):
    """extras: list of (weight, {(qid, id): z})."""
    rr = {}
    for qid, f in enumerate(feats):
        fp = f['fp'][model]
        z = (fp - fp.mean()) / (fp.std() + 1e-9)
        score = z + w_pop * f['pop'] + w_np * np.asarray(f['natural'], float)
        for w, zm in extras:
            if w:
                score = score + w * np.array([zm.get((qid, k), MISSING) for k in f['ids']])
        best = pd.Series(score).groupby(f['ids']).max()
        rank = int((best > best.get(f['key'], -np.inf)).sum()) + 1 if f['key'] in best.index else 99
        rr.setdefault(f['split'], []).append(1 / rank if rank <= 25 else 0.0)
    return {k: np.mean(v) for k, v in rr.items()}


def main(work, model='fpnet_v2h'):
    feats = pd.read_pickle(f'{work}/compare_features.pkl')
    frag = pd.read_parquet(f'{work}/frag_features.parquet')
    an = pd.read_parquet(f'{work}/analog_features.parquet')
    fz = zmap(frag, 'frag_int')
    rows = []
    for col in ['an_max', 'an_top', 'an_wmean']:
        az = zmap(an, col)
        for w_frag, w_an in itertools.product([0.0, 0.5], [0.0, 0.25, 0.5, 1.0]):
            r = evaluate(feats, model, 0.25, 1.0, [(w_frag, fz), (w_an, az)])
            rows.append(dict(feature=col, w_frag=w_frag, w_an=w_an, **r))
    res = pd.DataFrame(rows)
    res['mean'] = res.iloc[:, -2:].mean(axis=1)
    print(res.round(3).to_string(index=False))


if __name__ == '__main__':
    main(*sys.argv[1:3])
