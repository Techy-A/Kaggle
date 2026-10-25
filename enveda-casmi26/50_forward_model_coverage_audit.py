"""Forward-model evidence coverage audit of the real test set (CPU only, offline).

What: for each of the 400 test molecules, which forward-model evidence the current 0.465/0.466 pipeline can give it,
decided from the spectra it has (adduct / ionization mode) and the rules coded in the pipeline:
  * ICEBERG  : positive [M+H]+ or [M+Na]+ spectra only (fuse.build_ice_input covered_only=True,
               benchdata/dl_fuse_1024/fuse.py).
  * GLACIER  : real [M+H]+ spectra only (v29); a molecule WITHOUT [M+H]+ gets N1A proxies from [M+NH4]+ /
               [M-H2O+H]+ spectra (relabelled [M+H]+); [M+Na]+, [M+K]+, [M-2H2O+H]+ and every negative adduct are never proxied.
  * N1 challenger needs ICE and GL together on one molecule.
Class proxy: the real classes are hidden, so each molecule is labelled "library-matched" (class 1-like) when its best
binned cosine (0.01 Da, sqrt intensity) to a train spectrum with the same neutral mass (+-0.003 Da, adducts converted)
reaches 0.7 (the gate threshold of the gen20 notes); the rest is class 2/3 (not separable without structures).
Sensitivity at 0.5 and 0.9 is written too.
Why: CASMI26.md Methods digest 25 Oct, next experiment 3: is any adduct group blind, so that extending the N1 family
could still buy a +0.008 step (one public molecule = 0.0076)?
What it cannot do: candidate-level coverage (which final-list candidates lack an ICE/GL score) and the real run's
n_mols_scored need the 0.466 run logs, which are not in any local cache; only molecule-level coverage is computed.
Output: artifacts/coverage_audit_molecules.csv, artifacts/coverage_audit_summary.json (cached; delete to recompute).
usage: python 50_forward_model_coverage_audit.py
"""
import json
import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

D = '/home/user/enveda_data'
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
os.makedirs(ART, exist_ok=True)
OUT_CSV, OUT_JSON = f'{ART}/coverage_audit_molecules.csv', f'{ART}/coverage_audit_summary.json'
ADD = {'[M+H]+': 1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158, '[M-H2O+H]+': -17.003289,
       '[M-H]-': -1.007276, '[M+CH2O2-H]-': 44.997655, '[M+Cl]-': 34.969402, '[M-2H2O+H]+': -35.013854}
TOL, BIN = 0.003, 0.01


def vec(mz, it):
    k = np.round(np.asarray(mz) / BIN).astype(np.int64)
    v = np.sqrt(np.asarray(it, dtype=float))
    d = {}
    for a, b in zip(k, v):
        d[a] = d.get(a, 0.0) + b
    n = np.sqrt(sum(x * x for x in d.values())) or 1.0
    return {a: b / n for a, b in d.items()}


def cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def evidence(ads):
    ads = set(ads)
    ice = bool(ads & {'[M+H]+', '[M+Na]+'})
    gl_real = '[M+H]+' in ads
    gl_proxy = (not gl_real) and bool(ads & {'[M+NH4]+', '[M-H2O+H]+'})
    return ice, gl_real, gl_proxy


def build():
    te = pd.read_parquet(f'{D}/test.parquet')
    te['nm'] = te.precursor_mz - te.adduct.map(ADD)
    tm = pq.read_table(f'{D}/train_meta.parquet', columns=['adduct', 'precursor_mz']).to_pandas()
    tm['nm'] = tm.precursor_mz - tm.adduct.map(ADD)
    tn = tm.nm.values                                   # NaN for adducts not in ADD (never match)
    ok = np.where(~np.isnan(tn))[0]
    order = ok[np.argsort(tn[ok])]
    sn = tn[order]
    want = {}                                           # train row -> list of test row ids
    for i, nm in enumerate(te.nm.values):
        lo, hi = np.searchsorted(sn, [nm - TOL, nm + TOL])
        for r in order[lo:hi]:
            want.setdefault(int(r), []).append(i)
    print('train rows to read', len(want), flush=True)
    tv = [vec(m, s) for m, s in zip(te.ms2_mzs, te.ms2_normalized_intensities)]
    best = np.zeros(len(te))
    f = pq.ParquetFile(f'{D}/train.parquet')
    base = 0
    for g in range(f.num_row_groups):
        n = f.metadata.row_group(g).num_rows
        rows = [r for r in want if base <= r < base + n]
        if rows:
            t = f.read_row_group(g, columns=['ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
            for r in rows:
                v = vec(t.ms2_mzs.iloc[r - base], t.ms2_normalized_intensities.iloc[r - base])
                for i in want[r]:
                    best[i] = max(best[i], cos(tv[i], v))
        base += n
        print('row group', g, flush=True)
    te['best_cos'] = best
    rows = []
    for mid, g in te.groupby('molecule_id'):
        ads = list(g.adduct)
        ice, gl, glp = evidence(ads)
        rows.append(dict(molecule_id=mid, adducts='|'.join(sorted(set(ads))), n_spectra=len(g), best_cos=g.best_cos.max(),
                         ice=ice, gl_real=gl, gl_proxy=glp, n1=ice and (gl or glp), gl_any=gl or glp))
    m = pd.DataFrame(rows)
    m['group'] = np.select([m.gl_any & m.ice, m.gl_any, m.ice], ['ICE+GL', 'GL only', 'ICE only (Na fallback)'], 'none')
    m.to_csv(OUT_CSV, index=False)
    return m


def main():
    m = pd.read_csv(OUT_CSV) if os.path.exists(OUT_CSV) else build()
    S = dict(n=len(m), by_group=m.group.value_counts().to_dict(),
             by_adduct_set={a: g.group.value_counts().to_dict() for a, g in m.groupby('adducts')})
    for th in (0.5, 0.7, 0.9):
        lib = m.best_cos >= th
        S[f'cos>={th}'] = dict(lib_matched=int(lib.sum()),
                               lib_groups=m[lib].group.value_counts().to_dict(),
                               nolib_groups=m[~lib].group.value_counts().to_dict())
    S['none_adducts_cos>=0.7'] = {a: dict(n=len(g), lib=int((g.best_cos >= 0.7).sum())) for a, g in m[m.group == 'none'].groupby('adducts')}
    json.dump(S, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
