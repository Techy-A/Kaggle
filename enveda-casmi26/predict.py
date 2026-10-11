"""Rank candidates (cosine-gated fingerprint likelihood) and write submission.csv.

Usage: python predict.py <train.parquet> <test.parquet> <work_dir> <model.pt> [gate]
work_dir must hold structures.parquet and fps.npy (from fingerprints.py).
"""
import sys
import time

import numpy as np
import pandas as pd
import torch

from casmi import TEST_ADDUCTS, Library, molecule_mass, rank_molecule
from fpmodel import SpectrumStore, fp_loglik, gated_order, make_model, predict_logits

PPM = 10


def rank_all(test, lib, structs, fps, logits, gate=0.7, n=25):
    masses, keys, smiles = structs.mass.to_numpy(), structs.ik14.to_numpy(), structs.smiles.to_numpy()
    out = []
    for mol, spectra in test.groupby('molecule_id', sort=False):
        spectra = spectra[spectra.adduct.isin(TEST_ADDUCTS)]
        if len(spectra) == 0:
            out.append((mol, 'C'))
            continue
        m = molecule_mass(spectra)
        lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
        if hi == lo:
            out.append((mol, 'C'))
            continue
        fp = fp_loglik(logits[spectra.index.to_numpy()], fps[lo:hi]) / len(spectra)
        _, cos = rank_molecule(lib, spectra, ppm=PPM)
        cos = np.array([cos.get(k, 0.0) for k in keys[lo:hi]])
        best = gated_order(fp, cos, gate)[:n]
        out.append((mol, ';'.join(smiles[lo:hi][best])))
    return pd.DataFrame(out, columns=['molecule_id', 'smiles'])


def main(train_path, test_path, work, model_path, gate=0.7):
    test = pd.read_parquet(test_path).reset_index(drop=True)
    structs = pd.read_parquet(f'{work}/structures.parquet')
    fps = np.load(f'{work}/fps.npy')
    model = make_model(fps.shape[1] * 8)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    ok = test.adduct.isin(TEST_ADDUCTS)
    logits = np.zeros((len(test), fps.shape[1] * 8), np.float32)
    store = SpectrumStore.from_frame(test[ok])
    logits[np.flatnonzero(ok)] = predict_logits(model, store, np.arange(ok.sum()))
    t = time.time()
    lib = Library.build(train_path, keep=lambda df: df.adduct.isin(TEST_ADDUCTS))
    print(f'library built in {time.time() - t:.0f}s')
    sub = rank_all(test, lib, structs, fps, logits, float(gate))
    sub.to_csv('submission.csv', index=False)
    print(sub.shape)


if __name__ == '__main__':
    main(*sys.argv[1:])
