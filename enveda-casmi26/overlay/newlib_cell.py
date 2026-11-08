# ===== newer-library identity overlay (LB TEST, 8 Nov; 71_newlib_build.py) =====
# Library: MassBank open 2026.03 + GNPS community 2026-10 spectra of structures ABSENT from train.parquet (4,823 spectra,
# 1,394 structures). For each test molecule, the best square-root cosine (0.01 Da bins, precursor region dropped) over
# its spectra against library spectra of the same ion mode within 10 ppm + 0.002 Da. A hit (cos >= 0.85, >= 5 shared
# peaks; 61's train-side false-hit rate at this gate: 0.23% of molecules) puts that structure at rank 1 when the
# pipeline's rank 1 has no strong train-library match (library_max < 0.9), else at rank 2. Any error leaves the
# submission unchanged.
def _nl_run(sub_csv, test_parquet, lib_parquet, ev_parquet=None):
    import shutil, traceback
    import numpy as _np, pandas as _pd
    from rdkit import Chem as _C, RDLogger as _RL
    _RL.DisableLog('rdApp.*')
    TAU, K, LIBMAX = 0.85, 5, 0.9
    def vec(mz, it, prec):
        mz = _np.asarray(mz, float); it = _np.asarray(it, float); keep = mz < prec - 0.5
        b = _np.round(mz[keep] / 0.01).astype(_np.int64); w = _np.sqrt(_np.clip(it[keep], 0, None))
        if len(b) == 0: return {}
        s = _pd.Series(w).groupby(b).max(); n = _np.sqrt((s.values ** 2).sum())
        return dict(zip(s.index, s.values / n)) if n > 0 else {}
    try:
        sub = _pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
        te = _pd.read_parquet(test_parquet, columns=['molecule_id', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
        te = te[te.molecule_id.isin(set(sub.molecule_id))]
        lib = _pd.read_parquet(lib_parquet).sort_values(['mode', 'precursor_mz']).reset_index(drop=True)
        V = [vec(a, b, p) for a, b, p in zip(lib.mz, lib.intensity, lib.precursor_mz)]
        lmax = {}
        if ev_parquet:
            ev = _pd.read_parquet(ev_parquet, columns=['molecule_id', 'smiles', 'library_max'])
            lmax = {(m, s): float(v) for m, s, v in zip(ev.molecule_id, ev.smiles, ev.library_max.fillna(0))}
        best = {}
        for mode, g in lib.groupby('mode'):
            P = g.precursor_mz.values; idx = g.index.values
            for r in te[te.ionization_mode.str.lower().str[:3] == mode].itertuples():
                tol = r.precursor_mz * 10e-6 + 0.002
                lo, hi = _np.searchsorted(P, r.precursor_mz - tol), _np.searchsorted(P, r.precursor_mz + tol, 'right')
                if hi <= lo: continue
                a = vec(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
                if not a: continue
                for j in idx[lo:hi]:
                    b = V[j]; sh = [k for k in a if k in b]; c = sum(a[k] * b[k] for k in sh)
                    if c >= TAU and len(sh) >= K and c > best.get(r.molecule_id, (0,))[0]:
                        best[r.molecule_id] = (c, len(sh), lib.k[j], lib.smiles[j])
        kc = {}
        def k14(s):
            if s not in kc:
                x = _C.MolFromSmiles(s); kc[s] = _C.MolToInchiKey(x)[:14] if x is not None else None
            return kc[s]
        L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
        n1 = n2 = same = 0
        for m, (c, sh, key, smi) in best.items():
            l = L[m]
            if k14(l[0]) == key: same += 1; continue
            rest = [x for x in l if k14(x) != key]
            pos = 0 if lmax.get((m, l[0]), 0.0) < LIBMAX else 1
            n1 += pos == 0; n2 += pos == 1
            L[m] = (rest[:pos] + [smi] + rest[pos:])[:25]
            assert len(set(L[m])) == len(L[m])
        res = sub.copy(); res['smiles'] = res.molecule_id.map(lambda m: ';'.join(L[m]))
        assert res.smiles.str.split(';').map(len).between(1, 25).all() and len(res) == len(sub)
        shutil.copy2(sub_csv, str(sub_csv) + '.pre_newlib')
        res.to_csv(sub_csv, index=False)
        print('[newlib] applied', dict(molecules=len(sub), hits=len(best), already_rank1=same, set_rank1=n1, set_rank2=n2,
                                       hit_cos=[round(v[0], 3) for v in best.values()][:20]), flush=True)
    except Exception as e:
        traceback.print_exc(); print('[newlib] failed; submission unchanged:', e, flush=True)


_nl_lib = sorted(_g.glob('/kaggle/input/**/71_newlib.parquet', recursive=True))
if _te and _nl_lib:
    for _p in _subs:
        _nl_run(str(_p), _te[0], _nl_lib[0], _ev[0] if _ev else None)
else:
    print('[newlib] skipped: test or library file not found; submission unchanged', _te[:1], _nl_lib[:1], flush=True)
