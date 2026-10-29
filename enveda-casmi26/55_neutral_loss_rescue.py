"""Step 4: CPU replay (ablated) of Spectral Parallax's "disagreement rescue" on the genfix bench.

What: own rewrite of romantamrazov/casmi-2026-spectral-parallax `neutral_loss_score` and the spectral gates of
`choose_disagreement` (thresholds exactly as published). For each enveda-np-examples bench list (fin1 top 25;
enveda-180 `t_` lists never used) the contender is the same-formula candidate in fin1 top 15 with the highest
frag_rescore score; it is moved to rank 1 only if every computable published gate passes (lib_max < 0.955, >= 3
spectrum bins, per-bin fragment-ion and neutral-loss margins). The neural z-margin / cosine gates and the "0.477 merge
discarded it" condition cannot be computed on our bench (no busyaprime neural features or variant lists) and are
dropped; see research note §11.
Why: the layer is the only new evidence source among the public 0.477 forks (Step 4 of the CASMI next-steps plan).
Output: artifacts/55_frag_rescore.py (our V22 copy of frag_rescore, extracted), 55_per_list.csv, 55_summary.json.
usage: python 55_neutral_loss_rescue.py
"""
import ast
import importlib
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
m51 = importlib.import_module('51_isomer_reranker')
m53 = importlib.import_module('53_fiora_negmode_bench250')
D, ART = m51.D, m51.ART
LIB_TAU, SRC_RANK, MIN_BINS = 0.955, 15, 3


def load_frag():
    """frag_rescore from our own V22 notebook (cell 34 holds its source as the FRAG_SRC string)."""
    path = f'{ART}/55_frag_rescore.py'
    if not os.path.exists(path):
        nb = json.load(open(f'{HERE}/variants/v22_gen20.ipynb'))
        cell = ''.join(nb['cells'][34]['source'])
        line = next(ln for ln in cell.splitlines() if ln.strip().startswith('FRAG_SRC ='))
        open(path, 'w').write(ast.literal_eval(line.split('=', 1)[1].strip()))
    sys.path.insert(0, ART)
    return importlib.import_module('55_frag_rescore')


frag = load_frag()


def fnum(x, default=float('nan')):
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def spectrum_bins(rows, max_spectra=9):
    """Published bucketing: best (most peaks) spectrum per (adduct, round(CE/10)), >= 9 peaks, top 120 peaks."""
    groups = defaultdict(list)
    for r in rows:
        mz, it = np.asarray(r['ms2_mzs'], float), np.asarray(r['ms2_normalized_intensities'], float)
        n = min(len(mz), len(it))
        mz, it = mz[:n], it[:n]
        good = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        if good.sum() < 9:
            continue
        pk = np.where(good)[0]
        if len(pk) > 120:
            pk = pk[np.argsort(it[pk])[-120:]]
        ce = r['collision_energy_ev']
        energy = fnum(ce[0] if ce is not None and len(ce) == 1 else None, 30)   # float(list) fails -> 30, as published
        sp = {'mz': mz[pk], 'it': it[pk], 'prec': fnum(r['precursor_mz'], mz.max()),
              'mode': 1 if str(r['ionization_mode']).lower() == 'positive' else -1, 'adduct': str(r['adduct'])}
        groups[(sp['adduct'], round(energy / 10))].append((len(pk), sp))
    bins = [max(g, key=lambda a: a[0])[1] for _, g in sorted(groups.items())]
    if len(bins) > max_spectra:
        idx = np.unique(np.linspace(0, len(bins) - 1, max_spectra).round().astype(int))
        bins = [bins[i] for i in idx]
    return bins


def neutral_loss_score(spec, levels):
    """Intensity share of observed losses (precursor - peak) matching M - fragment (level 1 w 1, level 2 w 0.55)."""
    if levels is None:
        return float('nan')
    mass = float(levels[0][0])
    loss, cost = [], []
    for k, lv in enumerate(levels[1:3], start=1):
        l = mass - np.asarray(lv, float)
        ok = (l > 5) & (l < mass - 5)
        loss.extend(l[ok])
        cost.extend([1.0 if k == 1 else 0.55] * int(ok.sum()))
    if not loss:
        return 0.0
    o = np.argsort(loss)
    loss, cost = np.asarray(loss)[o], np.asarray(cost)[o]
    obs, inten = spec['prec'] - spec['mz'], np.asarray(spec['it'], float)
    ok = np.isfinite(obs) & (obs > 6) & np.isfinite(inten) & (inten > 0)
    obs, inten = obs[ok], inten[ok]
    if len(obs) < 5:
        return float('nan')
    best = np.zeros(len(obs))
    for h in (-2, -1, 0, 1, 2):
        tgt = obs - h * frag.H
        idx = np.searchsorted(loss, tgt, side='left')
        for j in (-1, 0):
            t = np.clip(idx + j, 0, len(loss) - 1)
            best = np.maximum(best, cost[t] * np.maximum(0, 1 - np.abs(loss[t] - tgt) / 0.012))
    return float(inten @ best / max(inten.sum(), 1e-10))


def rescue(lst, smi, fml, lib_max, bins, cache):
    """Returns (new list, reason). lst/smi/fml aligned (inchikey14, SMILES, formula) for fin1 top 25."""
    lead = smi[0]
    pool = [i for i in range(1, min(SRC_RANK, len(lst))) if fml[i] is not None and fml[i] == fml[0]]
    if fml[0] is None or not pool:
        return lst, 'no_same_formula'
    sc = frag.score_molecule(bins, [smi[i] for i in pool], cache=cache) if bins else []
    sc = [-1 if s is None else s for s in sc]
    if not sc or max(sc) <= -1:
        return lst, 'no_contender'
    c = pool[int(np.argmax(sc))]
    s_lead = frag.score_molecule(bins, [lead], cache=cache)[0]
    if s_lead is not None and s_lead >= max(sc):
        return lst, 'leader_best_fragment'      # fragment check keeps the leader: no disagreement
    if lib_max >= LIB_TAU:
        return lst, 'protected_library_hit'
    if len(bins) < MIN_BINS:
        return lst, 'too_few_acquisitions'
    if s_lead is None:
        return lst, 'bad_fragments'
    a, b = frag.candidate_levels(lead, cache=cache), frag.candidate_levels(smi[c], cache=cache)
    if a is None or b is None:
        return lst, 'no_fragments'
    ion = np.asarray([frag.score_molecule([s], [lead, smi[c]], cache=cache) for s in bins], float)
    nl = np.asarray([[neutral_loss_score(s, a), neutral_loss_score(s, b)] for s in bins], float)
    if not np.isfinite(ion).all() or not np.isfinite(nl).all():
        return lst, 'invalid_scores'
    di, dl = ion[:, 1] - ion[:, 0], nl[:, 1] - nl[:, 0]
    if (np.median(di) < .022 or di.mean() < .045 or (di > .004).mean() < .70
            or np.median(dl) < .015 or dl.mean() < .024 or (dl > .001).mean() < .67):
        return lst, 'inconsistent_fragment_and_loss_evidence'
    return [lst[c]] + [x for i, x in enumerate(lst) if i != c], 'fired'


def main():
    recs = {r['mid']: r for r in m51.bench_records() if 'fin1' in r and r['mid'].startswith('b_')}
    bt = pd.read_parquet(f'{ART}/51_bench_feats.parquet', columns=['qid', 'pos', 'smiles', 'formula'])
    bt = bt[bt.qid.isin(recs)]
    meta = pd.read_parquet(f'{D}/train.parquet', columns=['ingest_lib', 'inchikey14'])
    keys = {m[2:]: m for m in recs}
    rows = np.where((meta.ingest_lib == 'enveda-np-examples') & meta.inchikey14.isin(keys))[0]
    spec = m53.read_rows(rows, ['inchikey14', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities',
                                'collision_energy_ev', 'ionization_mode', 'adduct'])
    by = defaultdict(list)
    for r in spec.values():
        by[keys[r['inchikey14']]].append(r)
    frag.warmup()
    cache, out = {}, []
    for mid, r in sorted(recs.items()):
        lst = r['fin1'][:25]
        c = bt[bt.qid == mid].sort_values('pos')
        smi = list(c.smiles)[:len(lst)]
        fml = [None if s == '' else f for s, f in zip(smi, list(c.formula)[:len(lst)])]
        bins = spectrum_bins(by[mid])
        if len(smi) < len(lst):
            new, why = lst, 'missing_smiles'
        else:
            new, why = rescue(lst, smi, fml, float(r['lib_max']), bins, cache)
        a = mid[2:]
        out.append({'mid': mid, 'cls': r['cls'], 'lib_max': r['lib_max'], 'bins': len(bins), 'reason': why,
                    'rr0': m51.rr(lst, a), 'rr1': m51.rr(new, a), 'top1_0': lst[0] == a, 'top1_1': new[0] == a})
    df = pd.DataFrame(out)
    df.to_csv(f'{ART}/55_per_list.csv', index=False)
    d = (df.rr1 - df.rr0).values
    res = {'n': len(df), 'reasons': df.reason.value_counts().to_dict(), 'fired': int((df.reason == 'fired').sum()),
           'mrr0': df.rr0.mean(), 'mrr1': df.rr1.mean(), 'boot': m51.boot(d), 'per_class': {}}
    for k, g in df.groupby('cls'):
        res['per_class'][int(k)] = {'n': len(g), 'fired': int((g.reason == 'fired').sum()), 'mrr0': g.rr0.mean(),
                                    'mrr1': g.rr1.mean(), 'rank1_gains': int((~g.top1_0 & g.top1_1).sum()),
                                    'rank1_losses': int((g.top1_0 & ~g.top1_1).sum())}
    res['accept'] = bool(res['boot'][3] >= 0.9 and res['per_class'].get(1, {}).get('rank1_losses', 0) == 0)
    json.dump(res, open(f'{ART}/55_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
