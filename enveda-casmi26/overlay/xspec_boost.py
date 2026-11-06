"""Cross-test-spectrum boost overlay (port of 58_cross_spectrum_boost.py, pre-registered RULE unchanged).

When molecule A has a confident rank 1 (library_max >= 0.87) and A and B are spectral partners (best modified cosine
>= 0.7 with >= 6 matched peaks, same ion mode), candidates in B's ranks 2-25 whose Morgan Tanimoto to A's rank 1 is
in [0.5, 1) move up 3 places. Rank 1 never changes (asserted). Any error leaves the submission unchanged.
Bench (58, 250 molecules): +0.00128 MRR, P(better) 0.83, 0 rank-1 changes.
Usage in a notebook: from xspec_boost import run_overlay; run_overlay(sub_csv, evidence_parquet, test_parquet)
"""
from itertools import combinations
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
RULE = dict(spec_cos=0.7, spec_match=6, conf_libmax=0.87, tani=0.5, shift=3, max_rank=25)
TOL = 0.01
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return FPG.GetFingerprint(m) if m is not None else None


def prep(mz, it, prec):
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    k = (mz < prec - 0.5) & (it >= 0.01 * it.max())
    mz, it = mz[k], it[k]
    o = np.argsort(-it)[:50]
    mz, it = mz[o], np.sqrt(it[o])
    return mz, it / (np.linalg.norm(it) + 1e-12)


def greedy(mza, ia, mzb, ib, shift=None):
    d = np.subtract.outer(mza, mzb)
    ok = np.abs(d) <= TOL
    if shift is not None and abs(shift) > TOL:
        ok |= np.abs(d - shift) <= TOL
    ai, bi = np.nonzero(ok)
    if len(ai) == 0:
        return 0.0, 0
    p = ia[ai] * ib[bi]
    ua, ub, s, n = set(), set(), 0.0, 0
    for j in np.argsort(-p):
        if ai[j] in ua or bi[j] in ub:
            continue
        ua.add(ai[j]); ub.add(bi[j]); s += p[j]; n += 1
    return s, n


def partners(test):
    S = []
    for r in test.itertuples():
        if len(r.ms2_mzs) == 0:
            continue
        mz, it = prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        S.append((r.molecule_id, r.ionization_mode, r.precursor_mz, mz, it))
    best = {}
    for a, b in combinations(range(len(S)), 2):
        ma, mb = S[a], S[b]
        if ma[0] == mb[0] or ma[1] != mb[1] or len(ma[3]) == 0 or len(mb[3]) == 0:
            continue
        mc, mn = greedy(ma[3], ma[4], mb[3], mb[4], shift=ma[2] - mb[2])
        if mn >= RULE['spec_match']:
            key = tuple(sorted((ma[0], mb[0])))
            best[key] = max(best.get(key, 0.0), mc)
    nbr = {}
    for (a, b), c in best.items():
        if c >= RULE['spec_cos']:
            nbr.setdefault(a, []).append(b); nbr.setdefault(b, []).append(a)
    return nbr


def boost(sub, ev, test):
    """sub: molecule_id, smiles (';'-joined). ev: molecule_id, smiles, library_max. Returns (new sub, stats)."""
    L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
    lib = ev.set_index(['molecule_id', 'smiles']).library_max.astype(float).to_dict()
    nbr = partners(test)
    conf = {m: lib.get((m, l[0]), 0.0) >= RULE['conf_libmax'] for m, l in L.items()}
    cache = {}
    F = lambda s: cache[s] if s in cache else cache.setdefault(s, fp(s))
    out, touched, moved = {}, 0, 0
    for m, full in L.items():
        lst, tail = full[:RULE['max_rank']], full[RULE['max_rank']:]
        anchors = [F(L[o][0]) for o in nbr.get(m, []) if conf.get(o) and o in L]
        anchors = [a for a in anchors if a is not None]
        boosted = []
        if anchors:
            for i in range(1, len(lst)):
                f = F(lst[i])
                if f is not None and any(RULE['tani'] <= x < 1.0 for x in DataStructs.BulkTanimotoSimilarity(f, anchors)):
                    boosted.append(i)
        score = {i: float(i) for i in range(1, len(lst))}
        for i in boosted:
            score[i] = max(1.0, i - RULE['shift']) - 0.5
        new = [lst[0]] + [lst[i] for i in sorted(score, key=lambda i: (score[i], i))] + tail
        assert new[0] == full[0] and sorted(new) == sorted(full)
        touched += bool(boosted); moved += new != full
        out[m] = ';'.join(new)
    res = sub.copy(); res['smiles'] = res.molecule_id.map(out)
    stats = dict(molecules=len(L), with_partner=sum(1 for m in L if nbr.get(m)), confident_anchors=int(sum(conf.values())),
                 lists_touched=touched, lists_reordered=moved, rank1_changed=int(sum(a.split(';')[0] != b.split(';')[0] for a, b in zip(sub.smiles, res.smiles))))
    return res, stats


def run_overlay(sub_csv, ev_parquet, test_parquet):
    import shutil
    try:
        sub = pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
        ev = pd.read_parquet(ev_parquet, columns=['molecule_id', 'smiles', 'library_max'])
        test = pd.read_parquet(test_parquet, columns=['molecule_id', 'ionization_mode', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
        test = test[test.molecule_id.isin(set(sub.molecule_id))]
        new, stats = boost(sub, ev, test)
        assert stats['rank1_changed'] == 0 and len(new) == len(sub) and new.smiles.notna().all()
        shutil.copy2(sub_csv, str(sub_csv) + '.pre_xspec')
        new.to_csv(sub_csv, index=False)
        print('[xspec] applied', stats, flush=True)
        return stats
    except Exception as e:
        import traceback; traceback.print_exc()
        print('[xspec] failed; submission unchanged:', e, flush=True)
        return None


# ===== instrument mass-accuracy prior (ranks 2-25 only; rank 1 never changes) =====
# Bench477 (250 molecules): candidate neutral-mass error vs the measured precursor, centred on the instrument offset
# (median signed ppm of all rank-1 candidates, -1.49 on the bench, no answers used): answers beyond 5 ppm 0.6%, other
# candidates 7.4%. Moving ranks 2-25 beyond 5 ppm to the end of the 25: +0.0007 MRR, P(better) 0.95, 3 up / 0 down.
ADDUCT_MASS = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
               '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M-H2O+H]+': 1.007276 - 18.010565}
MASS_RULE = dict(tail_ppm=5.0, max_rank=25)


def mass_prior(sub, test):
    from rdkit.Chem.Descriptors import ExactMolWt
    t = test[test.adduct.isin(ADDUCT_MASS)].copy()
    t['nm'] = t.precursor_mz - t.adduct.map(ADDUCT_MASS)
    nm = t.groupby('molecule_id').nm.median().to_dict()
    cache = {}
    def mw(s):
        if s not in cache:
            m = Chem.MolFromSmiles(s); cache[s] = ExactMolWt(m) if m is not None else float('nan')
        return cache[s]
    L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
    top = [(mw(l[0]) - nm[m]) / nm[m] * 1e6 for m, l in L.items() if m in nm and l and l[0]]
    centre = float(np.nanmedian(top)) if top else 0.0
    out, moved = {}, 0
    for m, full in L.items():
        if m not in nm:
            out[m] = ';'.join(full); continue
        lst, tail = full[:MASS_RULE['max_rank']], full[MASS_RULE['max_rank']:]
        far = [i for i in range(1, len(lst)) if abs((mw(lst[i]) - nm[m]) / nm[m] * 1e6 - centre) > MASS_RULE['tail_ppm']]
        new = [lst[0]] + [lst[i] for i in range(1, len(lst)) if i not in far] + [lst[i] for i in far] + tail
        assert new[0] == full[0] and sorted(new) == sorted(full)
        moved += new != full; out[m] = ';'.join(new)
    res = sub.copy(); res['smiles'] = res.molecule_id.map(out)
    return res, dict(centre_ppm=round(centre, 3), lists_reordered=moved,
                     rank1_changed=int(sum(a.split(';')[0] != b.split(';')[0] for a, b in zip(sub.smiles, res.smiles))))


def run_mass_overlay(sub_csv, test_parquet):
    import shutil
    try:
        sub = pd.read_csv(sub_csv, dtype=str, keep_default_na=False)
        test = pd.read_parquet(test_parquet, columns=['molecule_id', 'adduct', 'precursor_mz'])
        new, stats = mass_prior(sub, test[test.molecule_id.isin(set(sub.molecule_id))])
        assert abs(stats['centre_ppm']) <= 5, f"implausible instrument centre {stats['centre_ppm']} ppm (placeholder lists?)"
        assert stats['rank1_changed'] == 0 and len(new) == len(sub) and new.smiles.notna().all()
        shutil.copy2(sub_csv, str(sub_csv) + '.pre_mass')
        new.to_csv(sub_csv, index=False)
        print('[mass] applied', stats, flush=True)
        return stats
    except Exception as e:
        import traceback; traceback.print_exc()
        print('[mass] failed; submission unchanged:', e, flush=True)
        return None
