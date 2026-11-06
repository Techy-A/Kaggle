"""Rank-1-locked tail reranker overlay (63_tail_reranker.py, final model trained on all 250 bench477 molecules).

Reorders ranks 2-25 of each list with a LightGBM lambdarank over the pipeline's own exported evidence
(frozen_candidate_evidence.parquet: library_max, ICE, GLACIER, proxy GLACIER, main / prefusion / engine-2 / PubChem
ranks) plus the list position and the mass error from the instrument centre. No neural features. Rank 1 never moves;
any error leaves the submission unchanged. This is a leaderboard TEST of a bench result that may be a bench reversal.
"""
import numpy as np
import pandas as pd
FEATS = ['pos', 'library_max', 'ice', 'gl', 'proxy_gl', 'main_rank', 'prefusion_rank', 'engine2_rank', 'pubchem_rank', 'ppm_abs', 'd_lib_vs_top']
ADD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
       '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M-H2O+H]+': 1.007276 - 18.010565}


def features(sub, ev, test):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    ev = ev.drop_duplicates(['molecule_id', 'smiles']).set_index(['molecule_id', 'smiles'])
    t = test[test.adduct.isin(ADD)].copy(); t['nm'] = t.precursor_mz - t.adduct.map(ADD); nm = t.groupby('molecule_id').nm.median()
    mw, rows = {}, []
    for m, s in zip(sub.molecule_id, sub.smiles):
        for p, c in enumerate(s.split(';')[:25], 1):
            if c not in mw:
                x = Chem.MolFromSmiles(c); mw[c] = ExactMolWt(x) if x is not None else np.nan
            r = dict(mol=m, smiles=c, pos=p)
            if (m, c) in ev.index:
                r.update(ev.loc[(m, c)].to_dict())
            r['ppm'] = (mw[c] - nm[m]) / nm[m] * 1e6 if m in nm.index else np.nan
            rows.append(r)
    d = pd.DataFrame(rows)
    centre = d[d.pos == 1].ppm.median()
    d['ppm_abs'] = (d.ppm - centre).abs() if abs(centre) <= 5 else np.nan
    top = d[d.pos == 1].set_index('mol')
    d['d_lib_vs_top'] = pd.to_numeric(d.library_max, errors='coerce') - d.mol.map(pd.to_numeric(top.library_max, errors='coerce'))
    for f in FEATS: d[f] = pd.to_numeric(d[f], errors='coerce')
    return d, float(centre)


def rerank(sub, ev, test, model_txt):
    import lightgbm as lgb
    bst = lgb.Booster(model_str=model_txt)
    d, centre = features(sub, ev, test)
    d['s'] = bst.predict(d[FEATS]); d.loc[d.pos == 1, 's'] = np.inf
    out = {}
    for m, g in d.groupby('mol', sort=False):
        out[m] = list(g.sort_values(['s', 'pos'], ascending=[False, True]).smiles)
    new, moved = [], 0
    for m, s in zip(sub.molecule_id, sub.smiles):
        full = s.split(';')
        lst = out.get(m, full[:25]) + full[25:]
        assert lst[0] == full[0] and sorted(lst) == sorted(full)
        moved += lst != full; new.append(';'.join(lst))
    res = sub.copy(); res['smiles'] = new
    return res, dict(centre_ppm=round(centre, 3), lists_reordered=moved,
                     rank1_changed=int(sum(a.split(';')[0] != b.split(';')[0] for a, b in zip(sub.smiles, res.smiles))))


def run_tail_rr(sub_csv, ev_parquet, test_parquet, model_txt):
    import shutil
    try:
        sub = pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
        ev = pd.read_parquet(ev_parquet)
        test = pd.read_parquet(test_parquet, columns=['molecule_id', 'adduct', 'precursor_mz'])
        new, stats = rerank(sub, ev, test[test.molecule_id.isin(set(sub.molecule_id))], model_txt)
        assert stats['rank1_changed'] == 0 and len(new) == len(sub)
        shutil.copy2(sub_csv, str(sub_csv) + '.pre_tailrr'); new.to_csv(sub_csv, index=False)
        print('[tailrr] applied', stats, flush=True); return stats
    except Exception as e:
        import traceback; traceback.print_exc(); print('[tailrr] failed; submission unchanged:', e, flush=True); return None
