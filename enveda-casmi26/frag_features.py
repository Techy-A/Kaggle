"""Fragmentation features for the strongest candidates of each validation molecule.

Usage: python frag_features.py <train.parquet> <work_dir> <pubchem_dir> <popularity_dir>
Writes <work_dir>/frag_features.parquet: qid, id, frag_int, frag_peaks.
Candidates are chosen from features only (top 40 by fingerprint fit plus top 10 by popularity), never by label.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from casmi import TEST_ADDUCTS, molecule_mass, read_rows
from fragments import explain_scores, fragment_masses, molecule_peaks
from learned_ranker import candidate_frame
from pubchem import PubChemTier

N_FIT, N_POP = 40, 10


def score_task(task):
    qid, cid, smiles, mz, it, positive = task
    f1, f2 = fragment_masses(smiles)
    e_int, e_peaks = explain_scores(mz, it, f1, f2, positive)
    return qid, cid, e_int, e_peaks


def main(path, work, pc_dir, pop_dir, model='fpnet_v2h'):
    feats = pd.read_pickle(f'{work}/compare_features.pkl')
    holdout = set(open(f'{work}/holdout_e180.txt').read().split())
    queries = read_rows(path, lambda df: (((df.ingest_lib == 'enveda-np-examples')
                                           | ((df.ingest_lib == 'enveda-180') & df.inchikey14.isin(holdout)))
                                          & df.adduct.isin(TEST_ADDUCTS)))
    spectra_of = dict(list(queries.groupby('inchikey14')))
    pool_smiles = pd.read_parquet(f'{work}/structures.parquet', columns=['ik14', 'smiles']).set_index('ik14').smiles
    pc = PubChemTier(pc_dir)
    pc_ik14 = np.load(f'{pop_dir}/pc_ik14.npy', mmap_mode='r')

    tasks = []
    for qid, f in enumerate(feats):
        spectra = spectra_of[f['key']]
        df = candidate_frame(f['ids'], f['fp'][model], f['pop'], f['natural'])
        chosen = list(dict.fromkeys(list(df.nlargest(N_FIT, 'fp').id) + list(df.nlargest(N_POP, 'pop').id)))
        lo, hi = pc.window(molecule_mass(spectra), 10)
        window = dict(zip(np.asarray(pc_ik14[lo:hi]).astype(str)[::-1], pc.smiles(lo, hi)[::-1]))  # first row wins
        mz, it = molecule_peaks(spectra)
        positive = (spectra.ionization_mode == 'positive').mean() >= 0.5
        for cid in chosen:
            smi = pool_smiles.get(cid) if cid in pool_smiles.index else window.get(cid)
            if smi:
                tasks.append((qid, cid, smi, mz, it, positive))
    t = time.time()
    with Pool() as p:
        rows = p.map(score_task, tasks, chunksize=50)
    out = pd.DataFrame(rows, columns=['qid', 'id', 'frag_int', 'frag_peaks'])
    out.to_parquet(f'{work}/frag_features.parquet')
    print(f'{len(out):,} candidates fragmented in {time.time() - t:.0f}s')


if __name__ == '__main__':
    main(*sys.argv[1:6])
