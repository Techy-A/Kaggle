"""Isomer-exclusion overlay (rule R3). Distinct test molecules are distinct compounds, so one structure can be the
answer for at most one of them. When molecule A's rank 1 is a confident library match (library_max >= 0.87) and
molecule B has the same neutral mass (<= 5 ppm, so same formula), A's rank-1 structure (InChIKey-14) is removed from
B's list; everything below it moves up one place. B's rank 1 changes only if it was A's anchor.
Why: same-mass test pairs are distinct compounds (placeholder test: 45 pairs, max spectral cosine 0.65, median 0.16).
Bench477: 1 removal, MRR unchanged (only 8 same-formula groups in 250 molecules); kept because it is logically safe
unless a library_max >= 0.87 anchor is wrong.
Usage in a notebook: from iso_exclusion import run_iso_overlay; run_iso_overlay(sub_csv, evidence_parquet, test_parquet)
"""
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
ISO_ADDUCT = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
              '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M-H2O+H]+': 1.007276 - 18.010565}
ISO_RULE = dict(ppm=5.0, conf_libmax=0.87)


def iso_groups(test):
    t = test[test.adduct.isin(ISO_ADDUCT)].copy()
    t['nm'] = t.precursor_mz - t.adduct.map(ISO_ADDUCT)
    M = t.groupby('molecule_id').nm.median().sort_values()
    ids, V, out = M.index.tolist(), M.values, {}
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            if (V[j] - V[i]) / V[i] * 1e6 > ISO_RULE['ppm']:
                break
            out.setdefault(ids[i], set()).add(ids[j]); out.setdefault(ids[j], set()).add(ids[i])
    return out


def iso_exclude(sub, ev, test):
    L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
    lib = {(m, s): v for m, s, v in zip(ev.molecule_id, ev.smiles, ev.library_max.fillna(0.0))}
    kc = {}
    def key(s):
        if s not in kc:
            m = Chem.MolFromSmiles(s); kc[s] = Chem.MolToInchiKey(m)[:14] if m is not None else s
        return kc[s]
    G = iso_groups(test)
    anchors = {m: key(l[0]) for m, l in L.items() if l and lib.get((m, l[0]), 0.0) >= ISO_RULE['conf_libmax']}
    removed, touched = 0, 0
    for a, k in anchors.items():
        for b in G.get(a, ()):
            if b not in L or anchors.get(b) == k:
                continue
            new = [s for s in L[b] if key(s) != k]
            if new and len(new) < len(L[b]):
                removed += len(L[b]) - len(new); touched += 1; L[b] = new
    res = sub.copy(); res['smiles'] = res.molecule_id.map({m: ';'.join(l) for m, l in L.items()})
    return res, dict(groups=sum(1 for v in G.values() if v), anchors=len(anchors), lists_touched=touched, removed=removed,
                     rank1_changed=int(sum(a.split(';')[0] != b.split(';')[0] for a, b in zip(sub.smiles, res.smiles))))


def run_iso_overlay(sub_csv, ev_parquet, test_parquet):
    import shutil
    try:
        sub = pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
        ev = pd.read_parquet(ev_parquet, columns=['molecule_id', 'smiles', 'library_max'])
        test = pd.read_parquet(test_parquet, columns=['molecule_id', 'adduct', 'precursor_mz'])
        new, stats = iso_exclude(sub, ev, test[test.molecule_id.isin(set(sub.molecule_id))])
        assert len(new) == len(sub) and new.smiles.notna().all() and (new.smiles.str.len() > 0).all()
        assert stats['rank1_changed'] <= 0.05 * len(sub), f'too many rank-1 changes {stats}'
        shutil.copy2(sub_csv, str(sub_csv) + '.pre_iso')
        new.to_csv(sub_csv, index=False)
        print('[iso] applied', stats, flush=True)
        return stats
    except Exception as e:
        import traceback; traceback.print_exc()
        print('[iso] failed; submission unchanged:', e, flush=True)
        return None
