# ===== partner-transform generation (LB TEST, 8 Nov; 65/66 policy P_c): rank 1 never changes =====
# For each molecule B with a test-side spectral partner A (xspec RULE: shift-aware cosine >= 0.7, >= 6 matched peaks,
# same ion mode) whose neutral-mass difference matches a biotransformation (+/-O, +/-CH2, +/-H2, +/-hexose,
# +/-acetyl, +/-SO3, +/-glucuronide; 0.005 Da), A's rank-1 structure is transformed at every valid site; the product
# with the best untrained MetFrag-lite score on B's spectra is inserted at rank 2 (existing copies removed, list cut
# to 25). NP leak-free bench: +0.0113, P 0.87, class 3 +0.022. Any error leaves the submission unchanged.
def _pg_run(sub_csv, test_parquet):
    import shutil, sys, traceback
    import numpy as _np, pandas as _pd
    from rdkit import Chem as _C, RDLogger as _RL
    from rdkit.Chem import AllChem as _AC
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _F
    _RL.DisableLog('rdApp.*')
    try:
        _froot = [str(p) for p in (_P('/kaggle/working'), _root) if (p / 'frag_rescore.py').is_file()]
        assert _froot, 'frag_rescore.py not found'
        if _froot[0] not in sys.path: sys.path.insert(0, _froot[0])
        import frag_rescore as _fr
        ADD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
               '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M-H2O+H]+': 1.007276 - 18.010565}
        TOL = 0.005
        ADDS = {'+O': (15.994915, ['[#6;!H0:1]>>[#6:1]O']), '+CH2': (14.015650, ['[O,N;!H0:1]>>[*:1]C']),
                '+H2': (2.015650, ['[C:1]=[C:2]>>[C:1]-[C:2]', '[C:1]=[O:2]>>[C:1]-[O:2]']),
                '-H2': (-2.015650, ['[C;!H0;!$(C=*):1]-[C;!H0;!$(C=*):2]>>[C:1]=[C:2]']),
                '+hexose': (162.052824, ['[OX2H1:1]>>[O:1]C1OC(CO)C(O)C(O)C1O']),
                '+acetyl': (42.010565, ['[O,N;!H0:1]>>[*:1]C(C)=O']),
                '+SO3': (79.956815, ['[OX2H1:1]>>[O:1]S(=O)(=O)O']),
                '+glucuronide': (176.032088, ['[OX2H1:1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O'])}
        DELS = {'-O': (-15.994915, 'H2O'), '-CH2': (-14.015650, 'CH4'), '-hexose': (-162.052824, 'C6H12O6'),
                '-acetyl': (-42.010565, 'C2H4O'), '-SO3': (-79.956815, 'H2O3S'), '-glucuronide': (-176.032088, 'C6H10O7')}
        RXN = {k: (d, [_AC.ReactionFromSmarts(s) for s in ss]) for k, (d, ss) in ADDS.items()}
        def key(m):
            try: return _C.MolToInchiKey(m)[:14]
            except Exception: return None
        def generate(smi, dm):
            m = _C.MolFromSmiles(smi)
            if m is None: return {}
            out = {}
            for name, (d, rx) in RXN.items():
                if abs(d - dm) > TOL: continue
                for r in rx:
                    for ps in r.RunReactants((m,)):
                        p = ps[0]
                        try:
                            _C.SanitizeMol(p); k = key(p)
                            if k: out.setdefault(k, _C.MolToSmiles(p))
                        except Exception: pass
            for name, (d, f) in DELS.items():
                if abs(d - dm) > TOL: continue
                for b in m.GetBonds():
                    if b.IsInRing() or b.GetBondType() != _C.BondType.SINGLE: continue
                    rw = _C.RWMol(m); rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                    try:
                        for a in rw.GetAtoms(): a.SetNoImplicit(False); a.SetNumExplicitHs(0) if a.GetIsAromatic() is False and a.GetSymbol() == 'C' else None
                        _C.SanitizeMol(rw); frags = _C.GetMolFrags(rw, asMols=True)
                    except Exception: continue
                    if len(frags) != 2: continue
                    for i in (0, 1):
                        if _F(frags[i]) == f:
                            k = key(frags[1 - i])
                            if k: out.setdefault(k, _C.MolToSmiles(frags[1 - i]))
            return out
        sub = _pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
        test = _pd.read_parquet(test_parquet)
        test = test[test.molecule_id.isin(set(sub.molecule_id))]
        L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
        t = test[test.adduct.isin(ADD)].copy(); t['nm'] = t.precursor_mz - t.adduct.map(ADD); NM = t.groupby('molecule_id').nm.median().to_dict()
        nbr = partners(test)
        kc = {}
        def k14(s):
            if s not in kc:
                x = _C.MolFromSmiles(s); kc[s] = key(x) if x is not None else None
            return kc[s]
        def spectra(m):
            g = test[test.molecule_id == m]
            return [dict(mz=_np.asarray(r.ms2_mzs, float), it=_np.asarray(r.ms2_normalized_intensities, float) / max(_np.max(r.ms2_normalized_intensities), 1e-12),
                         adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1) for r in g.itertuples() if len(r.ms2_mzs)]
        new, n_ins = {}, 0
        for b, l in L.items():
            new[b] = l
            if b not in NM or b not in nbr or len(l) < 2: continue
            P = {}
            for a in nbr[b]:
                if a not in NM or a not in L: continue
                dm = NM[b] - NM[a]
                if abs(dm) < TOL: continue
                for k, s in generate(L[a][0], dm).items():
                    if k and k != k14(l[0]): P.setdefault(k, s)
            if not P: continue
            smis = list(P.values()); sc = _fr.score_molecule(spectra(b), smis, cache={})
            best = sorted(zip(smis, sc), key=lambda z: -(z[1] or 0))[0][0]
            bk = k14(best)
            rest = [x for x in l if x == l[0] or k14(x) != bk]
            nl = (rest[:1] + [best] + rest[1:])[:25]
            assert nl[0] == l[0] and len(set(nl)) == len(nl)
            new[b] = nl; n_ins += 1
        res = sub.copy(); res['smiles'] = res.molecule_id.map(lambda m: ';'.join(new[m]))
        assert (res.smiles.str.split(';').str[0].values == sub.smiles.str.split(';').str[0].values).all()
        assert res.smiles.str.split(';').map(len).between(1, 25).all()
        shutil.copy2(sub_csv, str(sub_csv) + '.pre_pgen')
        res.to_csv(sub_csv, index=False)
        print('[pgen] applied', dict(molecules=len(sub), with_partner=len(nbr), inserted=n_ins), flush=True)
    except Exception as e:
        traceback.print_exc(); print('[pgen] failed; submission unchanged:', e, flush=True)


if _te:
    for _p in _subs:
        _pg_run(str(_p), _te[0])
else:
    print('[pgen] skipped: test file not found; submission unchanged', flush=True)
