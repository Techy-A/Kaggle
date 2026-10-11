"""Morgan fingerprints for the candidate pool (training structures + COCONUT).

Usage: python fingerprints.py <train.parquet> <coconut.parquet> <out_dir>
Writes structures.parquet (ik14, smiles, mass, source) and fps.npy (packed bits, row-aligned).
"""
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from casmi import formula_mass

N_BITS = 2048
RDLogger.DisableLog('rdApp.*')


def fp_packed(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(N_BITS // 8, np.uint8)
    return np.packbits(gen.GetFingerprintAsNumPy(mol).astype(np.uint8))


def main(train_path, coconut_path, out):
    train = (pd.read_parquet(train_path, columns=['inchikey14', 'normalized_smiles', 'molecular_formula'])
             .drop_duplicates('inchikey14'))
    train = pd.DataFrame({'ik14': train.inchikey14, 'smiles': train.normalized_smiles,
                          'mass': train.molecular_formula.map(formula_mass), 'source': 'train'})
    coco = pd.read_parquet(coconut_path).assign(source='coconut')
    structs = pd.concat([train, coco], ignore_index=True).dropna(subset=['mass'])
    structs = structs.drop_duplicates('ik14').sort_values('mass', ignore_index=True)
    with Pool() as pool:
        fps = np.stack(pool.map(fp_packed, structs.smiles, chunksize=2000))
    structs.to_parquet(f'{out}/structures.parquet')
    np.save(f'{out}/fps.npy', fps)
    print(structs.source.value_counts().to_dict(), fps.shape)


if __name__ == '__main__':
    main(*sys.argv[1:4])
