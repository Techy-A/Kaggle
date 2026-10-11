"""PubChem candidate tier (ahmedberatozer/casmi26-pubchem-tier): ~106M mass-sorted structures, 150-1250 Da.

pc_mass.npy   float64, ascending neutral monoisotopic masses
pc_off.npy    int64 offsets into pc_smiles (n + 1 entries)
pc_smiles.npy uint8, all SMILES strings concatenated
"""
from pathlib import Path

import numpy as np


class PubChemTier:
    def __init__(self, folder):
        folder = Path(folder)
        self.mass = np.load(folder / 'pc_mass.npy', mmap_mode='r')
        self.off = np.load(folder / 'pc_off.npy', mmap_mode='r')
        self.blob = np.load(folder / 'pc_smiles.npy', mmap_mode='r')

    def window(self, mass, ppm=10):
        tol = mass * ppm * 1e-6
        return np.searchsorted(self.mass, [mass - tol, mass + tol])

    def smiles(self, lo, hi):
        start, end = int(self.off[lo]), int(self.off[hi])
        raw = bytes(self.blob[start:end])
        cuts = np.asarray(self.off[lo:hi + 1]) - start
        return [raw[a:b].decode() for a, b in zip(cuts[:-1], cuts[1:])]


def repair(smiles):
    """Parse a stereo-stripped PubChem SMILES, undoing the false radicals left by stripping '[C@H]' to '[CH]'.

    Returns an RDKit Mol (or None). Every radical becomes a hydrogen, which is right for these organic
    structures: PubChem stores closed-shell molecules.
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    changed = False
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            atom.SetNumRadicalElectrons(0)
            atom.SetNoImplicit(False)
            changed = True
    if changed:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    return mol
