"""Analog propagation: candidates that resemble the structures behind similar library spectra score higher.

A molecule with no reference spectrum of its own usually still has close analogs in the library (same scaffold,
different side chain). Their spectra share fragments and neutral losses with the query. We find the most similar
library spectra (binned fragment + neutral-loss cosine, any precursor mass, same ion mode), keep the best spectrum
per structure, and score each candidate by Tanimoto similarity to those hit structures, weighted by spectral
similarity.

Usage (validation): python analogs.py <train.parquet> <work_dir> <pubchem_dir> <popularity_dir>
Writes <work_dir>/analog_features.parquet: qid, id, an_max, an_top, an_wmean (same candidates as frag_features).
"""
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from casmi import TEST_ADDUCTS, clean_peaks

BIN = 0.01                  # Da
MZ_MAX, NL_MAX = 1000.0, 500.0
N_MZ, N_NL = int(MZ_MAX / BIN), int(NL_MAX / BIN)
DIM = N_MZ + N_NL
TOP_PEAKS = 30
PER_STRUCT = 3              # library spectra kept per structure and ion mode
N_HITS = 20                 # hit structures per query molecule


def spectrum_vector(mzs, ints, precursor_mz):
    """Sparse (bins, weights): fragments in 0.01 Da bins, neutral losses offset after them, each half the norm."""
    mz, it = clean_peaks(mzs, ints, precursor_mz, TOP_PEAKS)
    keep = mz < precursor_mz - 0.5
    mz, it = mz[keep], it[keep]
    nl = precursor_mz - mz
    f = (mz / BIN).astype(np.int64)
    ok_f = f < N_MZ
    n = (nl / BIN).astype(np.int64)
    ok_n = n < N_NL
    bins = np.concatenate([f[ok_f], N_MZ + n[ok_n]])
    w = np.concatenate([it[ok_f], it[ok_n]]).astype(np.float32)
    return bins, w


def to_matrix(rows, spread=0):
    """CSR matrix from (bins, weights) rows, L2-normalised. spread=1 also fills the neighbouring bins (query side)."""
    indptr, idx, val = [0], [], []
    for bins, w in rows:
        if spread:
            bins = np.concatenate([bins + s for s in range(-spread, spread + 1)])
            w = np.tile(w, 2 * spread + 1)
            bins = np.clip(bins, 0, DIM - 1)
        u, inv = np.unique(bins, return_inverse=True)
        v = np.zeros(len(u), np.float32)
        np.maximum.at(v, inv, w)
        nrm = np.linalg.norm(v)
        idx.append(u), val.append(v / nrm if nrm > 0 else v)
        indptr.append(indptr[-1] + len(u))
    return sparse.csr_matrix((np.concatenate(val) if val else np.zeros(0, np.float32),
                              np.concatenate(idx) if idx else np.zeros(0, np.int64), indptr), shape=(len(rows), DIM))


class AnalogLibrary:
    """Binned library spectra (a few per structure) for fast one-against-all similarity search."""

    def __init__(self, keys, positive, matrix):
        self.keys, self.positive, self.matrix = keys, positive, matrix

    @classmethod
    def build(cls, path, exclude=(), batch_size=100_000):
        exclude = set(exclude)
        cols = ['inchikey14', 'ionization_mode', 'adduct', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
        count, keys, pos, rows = {}, [], [], []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()
            df = df[df.adduct.isin(TEST_ADDUCTS) & ~df.inchikey14.isin(exclude)]
            for k, mode, p, a, b in zip(df.inchikey14, df.ionization_mode, df.precursor_mz, df.ms2_mzs,
                                        df.ms2_normalized_intensities):
                c = count.get((k, mode), 0)
                if c >= PER_STRUCT:
                    continue
                bins, w = spectrum_vector(a, b, p)
                if len(bins) < 3:
                    continue
                count[(k, mode)] = c + 1
                keys.append(k), pos.append(mode == 'positive'), rows.append((bins, w))
        return cls(np.array(keys), np.array(pos), to_matrix(rows))

    def hits(self, spectra, n=N_HITS):
        """Best library structures for a molecule: per structure the best similarity over query spectra."""
        rows = [spectrum_vector(a, b, p) for a, b, p in
                zip(spectra.ms2_mzs, spectra.ms2_normalized_intensities, spectra.precursor_mz)]
        q = to_matrix(rows, spread=1)
        q = q * np.float32(np.sqrt(3.0))             # spreading over 3 bins divides a 1-bin match by sqrt(3)
        sims = (self.matrix @ q.T).toarray()         # (library spectra, query spectra)
        positive = (spectra.ionization_mode == 'positive').to_numpy()
        sims[self.positive[:, None] != positive[None, :]] = 0.0
        best = sims.max(axis=1)
        top = np.argpartition(-best, min(n * PER_STRUCT * 2, len(best) - 1))[:n * PER_STRUCT * 2]
        s = pd.Series(best[top], index=self.keys[top]).groupby(level=0).max().nlargest(n)
        return s.index.to_numpy(), s.to_numpy()


def tanimoto(a, b):
    """Tanimoto between packed fingerprints a (n, B) and b (m, B) -> (n, m)."""
    pc = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    inter = np.zeros((len(a), len(b)))
    for j in range(len(b)):
        inter[:, j] = pc[a & b[j]].sum(1)
    na, nb = pc[a].sum(1), pc[b].sum(1)
    return inter / np.maximum(na[:, None] + nb[None, :] - inter, 1)


def fp_morgan(smiles):
    """Packed Morgan 2048 (as fps.npy), PubChem false radicals repaired; zeros when unparsable."""
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from pubchem import repair
    RDLogger.DisableLog('rdApp.*')
    mol = repair(smiles) if smiles else None
    if mol is None:
        return np.zeros(256, np.uint8)
    return np.packbits(rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                       .GetFingerprintAsNumPy(mol).astype(np.uint8))


def analog_scores(cand_fps, hit_fps, hit_sims):
    """an_max: max over hits of sim * tanimoto; an_top: tanimoto to the best hit; an_wmean: sim^4-weighted mean."""
    if len(hit_fps) == 0 or len(cand_fps) == 0:
        z = np.zeros(len(cand_fps))
        return z, z, z
    t = tanimoto(cand_fps, hit_fps)
    w = hit_sims ** 4
    return (t * hit_sims[None, :]).max(1), t[:, 0], (t * w[None, :]).sum(1) / (w.sum() + 1e-12)


AN_FEATURES = ['an_max', 'an_top', 'an_wmean', 'hit_sim', 'an_max_gap', 'an_wmean_gap']


def add_analog_features(df, an):
    """Merge analog scores (strongest candidates only; others NaN) and their gaps to the molecule's best."""
    df = df.merge(an, on=['qid', 'id'], how='left')
    g = df.groupby('qid')
    for c in ['an_max', 'an_wmean']:
        df[c + '_gap'] = df[c] - g[c].transform('max')
    return df


def molecule_analogs(lib, spectra, cands, cand_fps, hit_fp):
    """Analog scores for one molecule's candidates. hit_fp(key) -> packed Morgan of a library structure or None."""
    hk, hs = lib.hits(spectra)
    pairs = [(hit_fp(k), s) for k, s in zip(hk, hs)]
    pairs = [(f, s) for f, s in pairs if f is not None]
    hit_fps = np.stack([f for f, _ in pairs]) if pairs else np.zeros((0, cand_fps.shape[1]), np.uint8)
    sims = np.array([s for _, s in pairs])
    a, b, c = analog_scores(cand_fps, hit_fps, sims)
    return pd.DataFrame({'id': cands, 'an_max': a, 'an_top': b, 'an_wmean': c,
                         'hit_sim': sims[0] if len(sims) else 0.0})


def main(path, work, pc_dir, pop_dir, fps_file='fps.npy'):
    from multiprocessing import Pool

    from casmi import molecule_mass, read_rows
    from pubchem import PubChemTier

    feats = pd.read_pickle(f'{work}/compare_features.pkl')
    frag = pd.read_parquet(f'{work}/frag_features.parquet')       # the candidates to score
    holdout = set(open(f'{work}/holdout_e180.txt').read().split())
    answers = {f['key'] for f in feats}
    t = time.time()
    lib = AnalogLibrary.build(path, exclude=answers)               # answers' own spectra never act as analogs
    print(f'library {lib.matrix.shape[0]:,} spectra, {lib.matrix.nnz:,} nonzeros, {time.time() - t:.0f}s', flush=True)

    queries = read_rows(path, lambda df: (((df.ingest_lib == 'enveda-np-examples')
                                           | ((df.ingest_lib == 'enveda-180') & df.inchikey14.isin(holdout)))
                                          & df.adduct.isin(TEST_ADDUCTS)))
    spectra_of = dict(list(queries.groupby('inchikey14')))
    structs = pd.read_parquet(f'{work}/structures.parquet', columns=['ik14', 'smiles'])
    struct_fps = np.load(f'{work}/{fps_file}', mmap_mode='r')        # first 256 bytes: Morgan 2048
    row_of = pd.Series(np.arange(len(structs)), index=structs.ik14)
    row_of = row_of[~row_of.index.duplicated()]
    pool_smiles = structs.set_index('ik14').smiles
    pc = PubChemTier(pc_dir)
    pc_ik14 = np.load(f'{pop_dir}/pc_ik14.npy', mmap_mode='r')

    out = []
    with Pool() as p:
        for qid, f in enumerate(feats):
            spectra = spectra_of[f['key']]
            cands = frag[frag.qid == qid].id.to_numpy()
            lo, hi = pc.window(molecule_mass(spectra), 10)
            window = dict(zip(np.asarray(pc_ik14[lo:hi]).astype(str)[::-1], pc.smiles(lo, hi)[::-1]))
            smi = [pool_smiles.get(c) if c in pool_smiles.index else window.get(c) for c in cands]
            fps = np.stack(p.map(fp_morgan, smi, chunksize=8))
            hit_fp = lambda k: struct_fps[row_of[k], :256] if k in row_of.index else None   # noqa: E731
            out.append(molecule_analogs(lib, spectra, cands, fps, hit_fp).assign(qid=qid))
            if qid % 100 == 0:
                print(f'{qid}/{len(feats)} {time.time() - t:.0f}s', flush=True)
    pd.concat(out, ignore_index=True).to_parquet(f'{work}/analog_features.parquet')
    print(f'done {time.time() - t:.0f}s')


if __name__ == '__main__':
    main(*sys.argv[1:6])
