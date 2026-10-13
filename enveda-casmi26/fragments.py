"""In-silico fragmentation score ("how much of the measured spectrum can this structure explain?").

Breaks every bond and every pair of bonds of a candidate's heavy-atom graph, takes the masses of the resulting
pieces, and matches them (as ions, allowing +-2 hydrogen shifts) against the molecule's measured peaks.
Peaks explained by one cut count fully, by two cuts half. Returns intensity- and count-based explain scores.
"""
import numpy as np

H = 1.0078250319
PROTON = 1.007276
MAX_ATOMS = 70          # bigger molecules: single cuts only (pairs grow quadratically)


def _atom_masses(mol):
    from rdkit.Chem import Descriptors  # noqa: F401  (ensures RDKit is initialised)
    masses = []
    for a in mol.GetAtoms():
        masses.append(a.GetMass() if a.GetIsotope() else _MONO.get(a.GetSymbol(), a.GetMass())
                      + a.GetTotalNumHs() * H)
    return np.array(masses)


_MONO = {'C': 12.0, 'H': H, 'N': 14.0030740052, 'O': 15.9949146221, 'P': 30.97376151, 'S': 31.97207069,
         'F': 18.99840320, 'Cl': 34.96885271, 'Br': 78.9183376, 'I': 126.904468, 'Si': 27.9769265327,
         'B': 11.0093055, 'Se': 79.9165218}


def fragment_masses(smiles):
    """Neutral masses of pieces from 1- and 2-bond cuts: returns (masses_1cut, masses_2cut)."""
    from rdkit import Chem
    from pubchem import repair
    mol = repair(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return np.zeros(0), np.zeros(0)
    am = _atom_masses(mol)
    n = len(am)
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    adj = [[] for _ in range(n)]
    for k, (i, j) in enumerate(bonds):
        adj[i].append((j, k))
        adj[j].append((i, k))

    def pieces(cut):
        seen = np.zeros(n, bool)
        out = []
        for s in range(n):
            if seen[s]:
                continue
            stack, tot = [s], 0.0
            seen[s] = True
            while stack:
                u = stack.pop()
                tot += am[u]
                for v, k in adj[u]:
                    if k not in cut and not seen[v]:
                        seen[v] = True
                        stack.append(v)
            out.append(tot)
        return out

    one, two = set(), set()
    for k in range(len(bonds)):
        p = pieces({k})
        if len(p) > 1:
            one.update(round(x, 4) for x in p)
    if n <= MAX_ATOMS:
        for a in range(len(bonds)):
            for b in range(a + 1, len(bonds)):
                p = pieces({a, b})
                if len(p) > 1:
                    two.update(round(x, 4) for x in p)
    return np.array(sorted(one)), np.array(sorted(two - one))


def ion_masses(neutral, positive):
    """All ion m/z for the pieces with -2..+2 hydrogen shifts."""
    if len(neutral) == 0:
        return neutral
    shifts = np.arange(-2, 3) * H
    ions = neutral[:, None] + shifts[None, :] + (PROTON if positive else -PROTON)
    return np.sort(ions.ravel())


def explain_scores(peaks_mz, peaks_int, frag1, frag2, positive, ppm=10.0, min_tol=0.003):
    """(intensity fraction explained, peak fraction explained), 1-cut matches weight 1, 2-cut 0.5."""
    if len(peaks_mz) == 0:
        return 0.0, 0.0
    w = np.zeros(len(peaks_mz))
    for ions, weight in ((ion_masses(frag1, positive), 1.0), (ion_masses(frag2, positive), 0.5)):
        if len(ions) == 0:
            continue
        tol = np.maximum(peaks_mz * ppm * 1e-6, min_tol)
        idx = np.searchsorted(ions, peaks_mz)
        lo = np.abs(ions[np.clip(idx - 1, 0, len(ions) - 1)] - peaks_mz)
        hi = np.abs(ions[np.clip(idx, 0, len(ions) - 1)] - peaks_mz)
        hit = np.minimum(lo, hi) <= tol
        w = np.maximum(w, hit * weight)
    return float((w * peaks_int).sum() / (peaks_int.sum() + 1e-12)), float(w.mean())


def molecule_peaks(spectra, top=40):
    """Merged peak list of a molecule's spectra: per spectrum top peaks, sqrt intensities, below precursor."""
    mzs, its = [], []
    for a, b, p in zip(spectra.ms2_mzs, spectra.ms2_normalized_intensities, spectra.precursor_mz):
        mz, it = np.asarray(a, float), np.asarray(b, float)
        keep = mz < p - 1.0                           # fragments only, not the precursor
        mz, it = mz[keep], it[keep]
        if len(mz) > top:
            sel = np.argpartition(-it, top)[:top]
            mz, it = mz[sel], it[sel]
        mzs.append(mz)
        its.append(np.sqrt(it))
    if not mzs:
        return np.zeros(0), np.zeros(0)
    mz, it = np.concatenate(mzs), np.concatenate(its)
    order = np.argsort(mz)
    return mz[order], it[order]
