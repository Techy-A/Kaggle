"""End-to-end ranking for the hidden test: PubChem + COCONUT + training candidates, fingerprint likelihood,
popularity / natural-product prior, cosine gate.

    python pipeline.py <train.parquet> <test.parquet> <structures.parquet> <fps.npy> <model.pt>
                       <pubchem_dir> <popularity_dir> [w_pop] [w_np] [gate]
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from casmi import TEST_ADDUCTS, Library, molecule_mass, rank_molecule
from fpmodel import SpectrumStore, fp_loglik, make_model, predict_logits
from pubchem import PubChemTier, repair

PPM = 10


def fp_and_smiles(smiles):
    """Packed Morgan fingerprint and a clean canonical SMILES for one PubChem candidate."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    mol = repair(smiles)
    if mol is None:
        return np.zeros(256, np.uint8), ''
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8)), Chem.MolToSmiles(mol, isomericSmiles=False)


def prior_order(fp, cos, pop, natural, w_pop, w_np, gate):
    """Confident library matches first (cosine >= gate); the rest by z-scored fingerprint likelihood + prior."""
    z = (fp - fp.mean()) / (fp.std() + 1e-9)
    score = np.where(cos >= gate, 1e6 + cos, z + w_pop * pop + w_np * natural)
    return np.argsort(-score, kind='stable')


def pool_priors(structs, pop_dir):
    pp = pd.read_csv(f'{pop_dir}/pool_popularity.csv',
                     usecols=['inchikey14', 'pop', 'np_coconut', 'np_lotus', 'np_npatlas']).drop_duplicates('inchikey14')
    pp = pp.set_index('inchikey14').reindex(structs.ik14.to_numpy())
    pop = pp['pop'].fillna(0).to_numpy(np.float32)
    natural = (pp[['np_coconut', 'np_lotus', 'np_npatlas']].fillna(0).sum(1) > 0).to_numpy()
    return pop, natural | (structs.source == 'coconut').to_numpy()


def run(train_path, test_path, structs_path, fps_path, model_path, pc_dir, pop_dir,
        w_pop=0.25, w_np=1.0, gate=0.7, n_out=25):
    import torch
    t0 = time.time()
    test = pd.read_parquet(test_path).reset_index(drop=True)
    test = test[test.adduct.isin(TEST_ADDUCTS)].reset_index(drop=True)
    all_ids = pd.read_parquet(test_path, columns=['molecule_id']).molecule_id.unique()

    lib = Library.build(train_path, keep=lambda df: df.adduct.isin(TEST_ADDUCTS))
    print(f'cosine library {time.time() - t0:.0f}s', flush=True)

    structs = pd.read_parquet(structs_path)
    fps = np.load(fps_path)
    masses, keys, smiles = structs.mass.to_numpy(), structs.ik14.to_numpy(), structs.smiles.to_numpy()
    s_pop, s_np = pool_priors(structs, pop_dir)

    pc = PubChemTier(pc_dir)
    pc_ik14 = np.load(f'{pop_dir}/pc_ik14.npy', mmap_mode='r')
    pc_lsid = np.load(f'{pop_dir}/pc_lsid.npy', mmap_mode='r')
    pc_lpmid = np.load(f'{pop_dir}/pc_lpmid.npy', mmap_mode='r')
    pc_npf = np.load(f'{pop_dir}/pc_np.npy', mmap_mode='r')

    model = make_model(2048)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    logits = predict_logits(model, SpectrumStore.from_frame(test), np.arange(len(test)))

    groups = list(test.groupby('molecule_id', sort=False))
    windows, flat = [], []
    for mol, spectra in groups:
        lo, hi = pc.window(molecule_mass(spectra), PPM)
        windows.append((len(flat), lo, hi))
        flat.extend(pc.smiles(lo, hi))
    with Pool() as p:
        res = p.map(fp_and_smiles, flat, chunksize=2000)
    del flat
    pc_fps = np.stack([r[0] for r in res])
    pc_smiles = np.array([r[1] for r in res], dtype=object)
    del res
    print(f'PubChem candidates {len(pc_fps):,} fingerprinted {time.time() - t0:.0f}s', flush=True)

    out = {}
    for (mol, spectra), (start, plo, phi) in zip(groups, windows):
        n = phi - plo
        m = molecule_mass(spectra)
        lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
        lg = logits[spectra.index.to_numpy()]
        _, cos = rank_molecule(lib, spectra, ppm=PPM)
        ids = np.concatenate([keys[lo:hi], np.asarray(pc_ik14[plo:phi]).astype(str)])
        smi = np.concatenate([smiles[lo:hi], pc_smiles[start:start + n]])
        fp = np.concatenate([fp_loglik(lg, fps[lo:hi]), fp_loglik(lg, pc_fps[start:start + n])]) / len(spectra)
        cosv = np.concatenate([[cos.get(k, 0.0) for k in keys[lo:hi]], np.zeros(n)])
        pop = np.concatenate([s_pop[lo:hi], np.asarray(pc_lsid[plo:phi], np.float32)
                              + np.asarray(pc_lpmid[plo:phi], np.float32)])
        natural = np.concatenate([s_np[lo:hi], np.asarray(pc_npf[plo:phi]) > 0])
        if len(ids) == 0:
            continue
        guesses, seen = [], set()
        for i in prior_order(fp, cosv, pop, natural, w_pop, w_np, gate):
            if not smi[i] or ids[i] in seen:        # skip unparsable rows and duplicate InChIKey14s
                continue
            seen.add(ids[i])
            guesses.append(smi[i])
            if len(guesses) == n_out:
                break
        out[mol] = ';'.join(guesses)
    sub = pd.DataFrame({'molecule_id': all_ids})
    sub['smiles'] = sub.molecule_id.map(out).fillna('C').replace('', 'C')
    print(f'done {time.time() - t0:.0f}s, molecules without candidates: {(sub.smiles == "C").sum()}')
    return sub


if __name__ == '__main__':
    a = sys.argv[1:]
    sub = run(*a[:7], *(float(x) for x in a[7:10]))
    sub.to_csv('submission.csv', index=False)
