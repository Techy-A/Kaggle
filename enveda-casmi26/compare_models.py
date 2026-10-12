"""Compare fingerprint models on splits where the answer is reachable only through PubChem.

Usage: python compare_models.py <train.parquet> <work_dir> <pubchem_dir> <popularity_dir> name:fps_bytes:hidden ...
   e.g. fpnet_v1h:256:1536 fpnet_v2h:277:2048

S1 'np-examples': the 250 timsTOF natural products (all have public spectra, so popularity flatters them).
S2 'enveda-180 holdout': 500 timsTOF structures with no public spectra, held out of training (class-2-like).
In both, every answer is removed from the training-structure tier, and models rank all candidates by
fingerprint likelihood alone (no cosine, no prior). Candidates are de-duplicated by InChIKey14.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd


def fp_v2_repaired(smiles):
    from rdkit import RDLogger
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    from pubchem import repair
    RDLogger.DisableLog('rdApp.*')
    mol = repair(smiles)
    if mol is None:
        return np.zeros(277, np.uint8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    maccs = np.zeros(168, np.uint8)
    maccs[:167] = np.array(list(MACCSkeys.GenMACCSKeys(mol)), np.uint8)
    return np.packbits(np.concatenate([gen.GetFingerprintAsNumPy(mol).astype(np.uint8), maccs]))


def main(path, work, pc_dir, pop_dir, *models):
    import torch
    from casmi import TEST_ADDUCTS, molecule_mass, read_rows
    from fpmodel import SpectrumStore, fp_loglik, make_model, predict_logits
    from pubchem import PubChemTier

    holdout = set(open(f'{work}/holdout_e180.txt').read().split())
    q_np = read_rows(path, lambda df: (df.ingest_lib == 'enveda-np-examples') & df.adduct.isin(TEST_ADDUCTS))
    q_e = read_rows(path, lambda df: (df.ingest_lib == 'enveda-180') & df.inchikey14.isin(holdout)
                    & df.adduct.isin(TEST_ADDUCTS))
    queries = pd.concat([q_np.assign(split='S1 np-examples'), q_e.assign(split='S2 enveda-180 holdout')],
                        ignore_index=True)
    answers = set(queries.inchikey14)
    print(f'{queries.inchikey14.nunique()} molecules, {len(queries)} spectra', flush=True)

    structs = pd.read_parquet(f'{work}/structures.parquet')
    keep = ~structs.ik14.isin(answers).to_numpy()
    fps_v2 = np.load(f'{work}/fps_v2.npy')[keep]
    structs = structs[keep].reset_index(drop=True)
    masses, keys = structs.mass.to_numpy(), structs.ik14.to_numpy()
    pc = PubChemTier(pc_dir)
    pc_ik14 = np.load(f'{pop_dir}/pc_ik14.npy', mmap_mode='r')
    pc_lsid = np.load(f'{pop_dir}/pc_lsid.npy', mmap_mode='r')
    pc_lpmid = np.load(f'{pop_dir}/pc_lpmid.npy', mmap_mode='r')
    pc_npf = np.load(f'{pop_dir}/pc_np.npy', mmap_mode='r')
    from pipeline import pool_priors
    s_pop, s_np = pool_priors(structs, pop_dir)
    saved = []                                            # per-molecule features, for tuning ranking rules

    store = SpectrumStore.from_frame(queries)
    nets = []
    for spec in models:
        name, nbytes, hidden = spec.split(':')
        model = make_model(int(nbytes) * 8, hidden=int(hidden))
        model.load_state_dict(torch.load(f'{work}/{name}.pt', map_location='cpu'))
        nets.append((name, int(nbytes), predict_logits(model, store, np.arange(len(queries)))))
        del model

    groups = list(queries.groupby('inchikey14'))
    results, t = [], time.time()
    with Pool() as p:
        for b in range(0, len(groups), 40):                 # batches keep memory small
            batch = groups[b:b + 40]
            windows, flat = [], []
            for key, spectra in batch:
                lo, hi = pc.window(molecule_mass(spectra), 10)
                windows.append((len(flat), lo, hi))
                flat.extend(pc.smiles(lo, hi))
            pc_fps = np.stack(p.map(fp_v2_repaired, flat, chunksize=2000)) if flat else np.zeros((0, 277), np.uint8)
            del flat
            for (key, spectra), (start, plo, phi) in zip(batch, windows):
                m = molecule_mass(spectra)
                lo, hi = np.searchsorted(masses, [m * (1 - 1e-5), m * (1 + 1e-5)])
                ids = np.concatenate([keys[lo:hi], np.asarray(pc_ik14[plo:phi]).astype(str)])
                rows = spectra.index.to_numpy()
                feat = dict(key=key, split=spectra.split.iloc[0], ids=ids, fp={},
                            pop=np.concatenate([s_pop[lo:hi], np.asarray(pc_lsid[plo:phi], np.float32)
                                                + np.asarray(pc_lpmid[plo:phi], np.float32)]),
                            natural=np.concatenate([s_np[lo:hi], np.asarray(pc_npf[plo:phi]) > 0]))
                saved.append(feat)
                for name, nbytes, logits in nets:
                    fp = np.concatenate([fp_loglik(logits[rows], fps_v2[lo:hi, :nbytes]),
                                         fp_loglik(logits[rows], pc_fps[start:start + phi - plo, :nbytes])])
                    feat['fp'][name] = (fp / len(spectra)).astype(np.float32)
                    best = pd.Series(fp).groupby(ids).max()   # one score per identity
                    covered = key in best.index
                    rank = int((best > best[key]).sum()) + 1 if covered else None
                    results.append(dict(model=name, split=spectra.split.iloc[0], covered=covered,
                                        candidates=len(best), rr=(1 / rank if covered and rank <= 25 else 0.0),
                                        top1=covered and rank == 1, top25=covered and rank <= 25))
            print(f'{min(b + 40, len(groups))}/{len(groups)} molecules {time.time() - t:.0f}s', flush=True)
    pd.to_pickle(saved, f'{work}/compare_features.pkl')   # own output, read by tune_prior.py
    res = pd.DataFrame(results)
    res.to_csv(f'{work}/compare_models.csv', index=False)
    print(res.groupby(['split', 'model']).agg(molecules=('rr', 'size'), answer_in_pool=('covered', 'mean'),
                                              median_candidates=('candidates', 'median'), MRR=('rr', 'mean'),
                                              top1=('top1', 'mean'), top25=('top25', 'mean')).round(4).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:])
