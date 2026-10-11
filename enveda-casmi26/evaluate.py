"""Score fingerprint ranking, cosine search and their blend on the simulated novelty classes.

Usage: python evaluate.py <train.parquet> <work_dir>
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from casmi import TEST_ADDUCTS, Library, make_validation, molecule_mass, mrr_at_25, rank_molecule, read_rows
from fpmodel import SpectrumStore, fp_loglik, make_model, predict_logits

path, work = sys.argv[1], sys.argv[2]
PPM = 10

meta = pd.read_parquet(path, columns=['inchikey14', 'ingest_lib', 'adduct'])
_, novelty, _ = make_validation(meta)
banned = set(novelty.index[novelty == 2])

structs = pd.read_parquet(f'{work}/structures.parquet')
fps = np.load(f'{work}/fps.npy')
masses, keys = structs.mass.to_numpy(), structs.ik14.to_numpy()
model = make_model(fps.shape[1] * 8)
model.load_state_dict(torch.load(f'{work}/fpnet_val.pt'))

queries = read_rows(path, lambda df: (df.ingest_lib == 'enveda-np-examples') & df.adduct.isin(TEST_ADDUCTS))
store = SpectrumStore.from_frame(queries)
logits = predict_logits(model, store, np.arange(len(queries)))

t = time.time()
lib = Library.build(path, keep=lambda df: (df.ingest_lib != 'enveda-np-examples')
                    & df.adduct.isin(TEST_ADDUCTS) & ~df.inchikey14.isin(banned))
print(f'cosine library built in {time.time() - t:.0f}s')

rows = []
for key, spectra in queries.groupby('inchikey14'):
    m = molecule_mass(spectra)
    lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
    cands = keys[lo:hi].copy()   # copy: a view would pin the whole 711k-row array per molecule
    fp = fp_loglik(logits[spectra.index.to_numpy()], fps[lo:hi]) / len(spectra)
    _, cos = rank_molecule(lib, spectra, ppm=PPM)
    cos = np.array([cos.get(k, 0.0) for k in cands])
    rows.append(dict(key=key, novelty=novelty[key], cands=cands, fp=fp, cos=cos,
                     in_pool=key in set(cands)))
print(f'scored {len(rows)} molecules')


def ranked(r, w_cos, w_fp=1.0, gate=None):
    fp = r['fp'] - r['fp'].max()                      # 0 for the best fingerprint match, negative otherwise
    score = w_fp * fp + w_cos * r['cos']
    if gate is not None:                              # confident library matches go first, by cosine
        score = np.where(r['cos'] >= gate, 1e6 + r['cos'], fp)
    return list(r['cands'][np.argsort(-score, kind='stable')])


def report(name, w_cos, w_fp=1.0, gate=None):
    rr = pd.DataFrame([dict(novelty=r['novelty'], rr=mrr_at_25(ranked(r, w_cos, w_fp, gate), r['key'])) for r in rows])
    by = rr.groupby('novelty').rr.mean()
    print(f'{name:28s} class1 {by[1]:.3f}  class2/3 {by[2]:.3f}  all {rr.rr.mean():.3f}  '
          f'@16% class1: {0.16 * by[1] + 0.84 * by[2]:.3f}')
    return by


cov = pd.DataFrame([dict(novelty=r['novelty'], in_pool=r['in_pool'], n=len(r['cands'])) for r in rows])
print(cov.groupby('novelty').agg(in_pool=('in_pool', 'mean'), median_candidates=('n', 'median')))
report('cosine only', 1.0, 0.0)
report('fingerprint only', 0.0)
for w in (5, 10, 20, 50, 100):
    report(f'fingerprint + {w} x cosine', w)
for g in (0.5, 0.6, 0.7, 0.8, 0.9):
    report(f'gate: cosine >= {g} first', 0, gate=g)
pd.to_pickle(rows, f'{work}/eval_rows.pkl')   # own output, reused for quick what-ifs
