"""Validation with PubChem candidates, where class-2/3 answers are reachable only through PubChem.

    python evaluate_pubchem.py build <train.parquet> <work_dir> <pubchem_dir> <popularity_dir>
    python evaluate_pubchem.py rank  <work_dir>

Pool = PubChem (+-10 ppm) U COCONUT U training structures. For the simulated class-2/3 molecules the answer is
removed from the training-structure tier as well, so (as for real class-2 molecules) only PubChem can supply it.
`build` scores every candidate once and saves per-candidate features; `rank` compares ranking rules on them.
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

PPM = 10


def fp_packed_repaired(smiles):
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from pubchem import repair
    RDLogger.DisableLog('rdApp.*')
    mol = repair(smiles)
    if mol is None:
        return np.zeros(256, np.uint8)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))


def build(path, work, pc_dir, pop_dir):
    import torch
    from casmi import TEST_ADDUCTS, Library, make_validation, molecule_mass, rank_molecule, read_rows
    from fpmodel import SpectrumStore, fp_loglik, make_model, predict_logits
    from pubchem import PubChemTier

    meta = pd.read_parquet(path, columns=['inchikey14', 'ingest_lib', 'adduct'])
    _, novelty, _ = make_validation(meta)
    banned = set(novelty.index[novelty == 2])
    del meta

    t = time.time()   # built first: its peak memory is highest while little else is loaded
    lib = Library.build(path, keep=lambda df: (df.ingest_lib != 'enveda-np-examples')
                        & df.adduct.isin(TEST_ADDUCTS) & ~df.inchikey14.isin(banned))
    print(f'cosine library: {time.time() - t:.0f}s')

    structs = pd.read_parquet(f'{work}/structures.parquet')
    fps = np.load(f'{work}/fps.npy')
    keep = ~structs.ik14.isin(banned).to_numpy()                 # answers of class 2/3 leave the train tier
    structs, fps = structs[keep].reset_index(drop=True), fps[keep]
    masses, keys = structs.mass.to_numpy(), structs.ik14.to_numpy()
    pool_pop = pd.read_csv(f'{pop_dir}/pool_popularity.csv', usecols=['inchikey14', 'pop', 'np_coconut', 'np_lotus',
                                                                      'np_npatlas']).drop_duplicates('inchikey14')
    pool_pop = pool_pop.set_index('inchikey14')
    s_pop = pool_pop['pop'].reindex(keys).fillna(0).to_numpy(np.float32)
    s_np = (pool_pop[['np_coconut', 'np_lotus', 'np_npatlas']].sum(1) > 0).reindex(keys).fillna(False).to_numpy(bool).copy()
    s_np |= (structs.source == 'coconut').to_numpy()

    pc = PubChemTier(pc_dir)
    pc_ik14 = np.load(f'{pop_dir}/pc_ik14.npy', mmap_mode='r')
    pc_lsid = np.load(f'{pop_dir}/pc_lsid.npy', mmap_mode='r')
    pc_lpmid = np.load(f'{pop_dir}/pc_lpmid.npy', mmap_mode='r')
    pc_npf = np.load(f'{pop_dir}/pc_np.npy', mmap_mode='r')

    model = make_model(2048)
    model.load_state_dict(torch.load(f'{work}/fpnet_val.pt', map_location='cpu'))
    queries = read_rows(path, lambda df: (df.ingest_lib == 'enveda-np-examples') & df.adduct.isin(TEST_ADDUCTS))
    logits = predict_logits(model, SpectrumStore.from_frame(queries), np.arange(len(queries)))

    groups = list(queries.groupby('inchikey14'))
    windows, flat = [], []
    for key, spectra in groups:
        lo, hi = pc.window(molecule_mass(spectra), PPM)
        windows.append((len(flat), lo, hi))
        flat.extend(pc.smiles(lo, hi))
    t = time.time()
    with Pool() as pool:
        pc_fps = np.stack(pool.map(fp_packed_repaired, flat, chunksize=2000))
    n_flat = len(flat)
    del flat
    print(f'PubChem: {n_flat:,} candidates fingerprinted in {time.time() - t:.0f}s')

    rows = []
    for (key, spectra), (start, plo, phi) in zip(groups, windows):
        n = phi - plo
        m = molecule_mass(spectra)
        lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
        lg = logits[spectra.index.to_numpy()]
        _, cos = rank_molecule(lib, spectra, ppm=PPM)
        rows.append(dict(
            key=key, novelty=novelty[key],
            ids=np.concatenate([keys[lo:hi], np.asarray(pc_ik14[plo:phi]).astype(str)]),
            fp=np.concatenate([fp_loglik(lg, fps[lo:hi]), fp_loglik(lg, pc_fps[start:start + n])]) / len(spectra),
            cos=np.concatenate([[cos.get(k, 0.0) for k in keys[lo:hi]], np.zeros(n)]),
            pop=np.concatenate([s_pop[lo:hi], np.asarray(pc_lsid[plo:phi], np.float32)
                                + np.asarray(pc_lpmid[plo:phi], np.float32)]),
            np=np.concatenate([s_np[lo:hi], np.asarray(pc_npf[plo:phi]) > 0]),
            src=np.concatenate([np.zeros(hi - lo, int), np.ones(n, int)])))
    pd.to_pickle(rows, f'{work}/eval_pubchem_rows.pkl')    # own output, reused by `rank`
    print('saved', len(rows))


def reciprocal_rank(r, order):
    seen, rank = set(), 0
    for i in order:
        if r['ids'][i] in seen:          # the same InChIKey14 twice takes one slot
            continue
        seen.add(r['ids'][i])
        rank += 1
        if r['ids'][i] == r['key']:
            return 1.0 / rank
        if rank == 25:
            break
    return 0.0


def order(r, gate=0.7, w_pop=0.0, w_np=0.0):
    fp = r['fp']
    z = (fp - fp.mean()) / (fp.std() + 1e-9)
    score = z + w_pop * r['pop'] + w_np * r['np']
    score = np.where(r['cos'] >= gate, 1e6 + r['cos'], score)
    return np.argsort(-score, kind='stable')


def rank(work):
    rows = pd.read_pickle(f'{work}/eval_pubchem_rows.pkl')
    cov = pd.DataFrame([dict(novelty=r['novelty'], covered=(r['ids'] == r['key']).any(), n=len(r['ids'])) for r in rows])
    print(cov.groupby('novelty').agg(answer_in_pool=('covered', 'mean'), median_candidates=('n', 'median')))
    res = []
    for w_pop in (0, 0.1, 0.25, 0.5, 1.0):
        for w_np in (0, 0.5, 1, 2):
            for gate in (0.6, 0.7):
                rr = pd.DataFrame([dict(novelty=r['novelty'], rr=reciprocal_rank(r, order(r, gate, w_pop, w_np)))
                                   for r in rows]).groupby('novelty').rr.mean()
                res.append(dict(gate=gate, w_pop=w_pop, w_np=w_np, class1=rr[1], class23=rr[2],
                                expected=0.16 * rr[1] + 0.84 * rr[2]))
    res = pd.DataFrame(res).round(3).sort_values('expected', ascending=False)
    print(res.head(15).to_string(index=False))
    print(res[(res.w_pop == 0) & (res.w_np == 0)].to_string(index=False))
    res.to_csv(f'{work}/rank_grid.csv', index=False)


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build(*sys.argv[2:6])
    else:
        rank(sys.argv[2])
