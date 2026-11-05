"""61: newer public library spectra (MassBank 2026.03 open records, GNPS community libraries 2026-10-22) against the test.

What: keeps library MS2 spectra whose structure (InChIKey-14) is NOT in train.parquet, then for every test molecule finds
the best library match: same ion mode, precursor m/z within 10 ppm + 0.002 Da, square-root-intensity cosine on 0.01 Da
bins and the number of shared peaks. Thresholds come from false hits on training molecules: a training molecule's own
structure is in train, so every hit it gets from this library is a wrong structure by construction.
Why: heliosli's 0.482 notebook puts such a match at rank 1 (his private dataset casmi26-extlib-v1); the host allows newer
releases of train.parquet's sources (thread 750684, 27 Oct). This counts how many test molecules could gain.
Output: artifacts/61_extlib_summary.json, artifacts/61_test_hits.csv. CPU, a few minutes.
Usage: python enveda-casmi26/61_extlib_hits.py [--lib /home/user/work/extlib] [--ncal 6000] [--msnlib]
--msnlib: use only MSnLib v8 MS2 (Zenodo 23494858, CC BY 4.0, from MSnLib doi:10.5281/zenodo.21105617) instead of
MassBank + GNPS; outputs get the suffix _msnlib.
"""
import argparse, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
D = '/home/user/enveda_data'
BIN, PPM, ABS = 0.01, 10e-6, 0.002
TAUS, KS = [0.6, 0.7, 0.8, 0.85, 0.9], [3, 5, 8]


def vec(mz, it, prec):
    mz = np.asarray(mz, float); it = np.asarray(it, float)
    keep = mz < prec - 0.5                      # drop the precursor peak region
    b = np.round(mz[keep] / BIN).astype(np.int64); w = np.sqrt(np.clip(it[keep], 0, None))
    if len(b) == 0: return {}
    s = pd.Series(w).groupby(b).max()
    n = np.sqrt((s.values ** 2).sum())
    return dict(zip(s.index, s.values / n)) if n > 0 else {}


def load_msnlib(path):
    m = pq.read_table(f'{path}/msnlib.parquet', columns=['inchikey', 'precursor_mz', 'mzs', 'intensities', 'charge']).to_pandas()
    m = m.dropna(subset=['mzs', 'intensities', 'inchikey'])
    m['mz'] = m.mzs.map(lambda x: np.fromstring(x, sep=','))
    m['intensity'] = m.intensities.map(lambda x: np.fromstring(x, sep=','))
    m['ion_mode'] = np.where(m.charge > 0, 'positive', 'negative')
    m['src'] = 'msnlib'
    return m[['src', 'inchikey', 'ion_mode', 'precursor_mz', 'mz', 'intensity']]


def load_lib(path, train_keys, msnlib=False):
    rows = []
    if msnlib:
        df = load_msnlib(path)
        df['k'] = df.inchikey.fillna('').str.upper().str[:14]
        df['mode'] = df.ion_mode.str[:3]
        lib = df[(df.k.str.len() == 14) & (df.precursor_mz > 50)][['src', 'k', 'mode', 'precursor_mz', 'mz', 'intensity']]
        n_all, k_all = len(lib), lib.k.nunique()
        lib = lib[~lib.k.isin(train_keys)].sort_values(['mode', 'precursor_mz']).reset_index(drop=True)
        return lib, dict(lib_spectra_all=n_all, lib_structures_all=k_all, new_spectra=len(lib), new_structures=int(lib.k.nunique()))
    g = pq.read_table(f'{path}/gnps.parquet', columns=['ion_mode', 'inchikey', 'precursor_mz', 'mz', 'intensity', 'ms_level']).to_pandas()
    g['src'] = 'gnps'
    m = pq.read_table(f'{path}/massbank.parquet', columns=['ion_mode', 'inchikey', 'precursor_mz', 'mz', 'intensity', 'spectrum_type']).to_pandas()
    m = m[m.spectrum_type == 'MS2']; m['src'] = 'massbank'
    for df in (g, m):
        df['k'] = df.inchikey.fillna('').str.upper().str[:14]
        df['mode'] = df.ion_mode.fillna('').str.lower().str[:3]
        df = df[(df.k.str.len() == 14) & df['mode'].isin(['pos', 'neg']) & (pd.to_numeric(df.precursor_mz, errors='coerce') > 50)]
        rows.append(df[['src', 'k', 'mode', 'precursor_mz', 'mz', 'intensity']])
    lib = pd.concat(rows, ignore_index=True)
    lib['precursor_mz'] = lib.precursor_mz.astype(float)
    n_all, k_all = len(lib), lib.k.nunique()
    lib = lib[~lib.k.isin(train_keys)].sort_values(['mode', 'precursor_mz']).reset_index(drop=True)
    return lib, dict(lib_spectra_all=n_all, lib_structures_all=k_all, new_spectra=len(lib), new_structures=int(lib.k.nunique()))


def best_hits(q, lib, vecs):
    """q: rows with mode, precursor_mz, mz, intensity, mol. Returns per row the best (cos, shared, key)."""
    out = []
    for mode, sub in lib.groupby('mode'):
        P = sub.precursor_mz.values; idx = sub.index.values
        for _, r in q[q['mode'] == mode].iterrows():
            tol = r.precursor_mz * PPM + ABS
            lo, hi = np.searchsorted(P, r.precursor_mz - tol), np.searchsorted(P, r.precursor_mz + tol, 'right')
            if hi <= lo: continue
            a = vec(r.mz, r.intensity, r.precursor_mz)
            if not a: continue
            best = (0.0, 0, None)
            for j in idx[lo:hi]:
                b = vecs[j]
                sh = [k for k in a if k in b]
                c = sum(a[k] * b[k] for k in sh)
                if c > best[0]: best = (c, len(sh), lib.k[j])
            out.append((r.mol, best[0], best[1], best[2]))
    return pd.DataFrame(out, columns=['mol', 'cos', 'shared', 'key'])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lib', default='/home/user/work/extlib'); ap.add_argument('--ncal', type=int, default=6000); ap.add_argument('--msnlib', action='store_true')
    a = ap.parse_args()
    meta = pd.read_parquet(f'{D}/train_meta.parquet', columns=['inchikey14'])
    train_keys = set(meta.inchikey14)
    lib, S = load_lib(a.lib, train_keys, a.msnlib)
    sfx = '_msnlib' if a.msnlib else ''
    print(S, flush=True)
    vecs = {j: vec(r.mz, r.intensity, r.precursor_mz) for j, r in lib.iterrows()}
    # test
    te = pd.read_parquet(f'{D}/test.parquet', columns=['molecule_id', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
    te = te.rename(columns={'molecule_id': 'mol', 'ms2_mzs': 'mz', 'ms2_normalized_intensities': 'intensity'})
    te['mode'] = te.ionization_mode.str.lower().str[:3]
    th = best_hits(te, lib, vecs)
    # calibration: random training spectra; any hit is a wrong structure by construction
    rng = np.random.default_rng(0)
    n = pq.ParquetFile(f'{D}/train.parquet').metadata.num_rows
    pick = np.sort(rng.choice(n, a.ncal, replace=False))
    tr = ds.dataset(f'{D}/train.parquet').take(pick, columns=['inchikey14', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
    tr = tr.rename(columns={'inchikey14': 'mol', 'ms2_mzs': 'mz', 'ms2_normalized_intensities': 'intensity'})
    tr['mode'] = tr.ionization_mode.str.lower().str[:3]
    ch = best_hits(tr, lib, vecs)
    grid = []
    for t in TAUS:
        for k in KS:
            fh = ch[(ch.cos >= t) & (ch.shared >= k)].mol.nunique() / tr.mol.nunique()
            hits = th[(th.cos >= t) & (th.shared >= k)]
            grid.append(dict(tau=t, k=k, false_hit_rate_train=round(fh, 4), test_molecules_hit=int(hits.mol.nunique()),
                             expected_public=round(hits.mol.nunique() * 132 / 400, 1)))
    S['grid'] = grid
    S['n_test_molecules'] = int(te.mol.nunique()); S['n_cal_molecules'] = int(tr.mol.nunique())
    os.makedirs(ART, exist_ok=True)
    json.dump(S, open(f'{ART}/61_extlib_summary{sfx}.json', 'w'), indent=1)
    th.sort_values('cos', ascending=False).to_csv(f'{ART}/61_test_hits{sfx}.csv', index=False)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
