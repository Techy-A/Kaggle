"""Step 3: rank-preserving FIORA-OS fusion on genfix-bench base lists (CPU only).

What: within the rank-1-formula slots of each fin1 list (lib_max < 0.9), reorder by
z(-log base_rank) + 0.25 z(FIORA entropy sim at CE 10, summed over the molecule's negative-mode spectra); stable,
no margin rule. (a) the 214 [M-H]- lists of 53 (cached predictions reused; sanity only), (b) fresh decision set: bench
lists not in (a) whose query has other negative adducts only (formate, Cl, 2M-H), precursor converted to [M-H]-.
Rule, weight, sets and gates pre-registered in the research note §10 before scoring.
Why: §9 showed a reranker ignoring base rank destroys the bench; this keeps the base rank as the prior.
Output: artifacts/54_fresh_fiora_out.mgf (cache), 54_per_list.csv, 54_summary.json.
usage: python 54_fiora_rank_preserving.py
"""
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
m53 = importlib.import_module('53_fiora_negmode_bench250')
m52, m51, D, ART = m53.m52, m53.m51, m53.D, m53.ART
W, LIB_TAU, z = 0.25, 0.9, m53.z
TO_MH = {'[M-H]-': lambda p: p, '[M+CH2O2-H]-': lambda p: p - 46.00548, '[M+Cl]-': lambda p: p - 35.97668,
         '[2M-H]-': lambda p: (p + 1.00728) / 2 - 1.00728}


def load_lists():
    recs = {r['mid']: r for r in m51.bench_records() if 'fin1' in r}
    used = set(pd.read_csv(f'{ART}/53_bench_per_molecule.csv').mid)
    meta = pd.read_parquet(f'{D}/train.parquet', columns=['ingest_lib', 'inchikey14', 'adduct'])
    meta['row'] = np.arange(len(meta))
    mids = {m[2:]: m for m in recs}
    q = meta[meta.adduct.isin(TO_MH) & meta.inchikey14.isin(mids)]
    q = q[q.ingest_lib == q.inchikey14.map(lambda k: 'enveda-np-examples' if mids[k][0] == 'b' else 'enveda-180')]
    spec = m53.read_rows(q.row, ['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
    bt = pd.read_parquet(f'{ART}/51_bench_feats.parquet', columns=['qid', 'pos', 'smiles', 'formula'])
    out = []
    for k, g in q.groupby('inchikey14'):
        mid = mids[k]
        s = 'a' if mid in used else 'b'
        g = g[g.adduct == '[M-H]-'] if s == 'a' else g      # (a) exactly the spectra of 53
        if s == 'b' and (g.adduct == '[M-H]-').any():
            continue
        c = bt[bt.qid == mid].sort_values('pos')
        out.append({'mid': mid, 'set': s, 'rec': recs[mid], 'spectra': [(spec[i], a) for i, a in zip(g.row, g.adduct)],
                    'smiles': list(c.smiles), 'formula': [None if x == '' else f for x, f in zip(c.smiles, c.formula)]})
    return out


def rerank(L, pred):
    r, fs = L['rec'], L['formula']
    lst = r['fin1'][:25]
    slots = [i for i, f in enumerate(fs[:len(lst)]) if f is not None and f == fs[0]] \
        if float(r['lib_max']) < LIB_TAU else []
    new = list(lst)
    if len(slots) >= 2:
        en = np.zeros(len(slots))
        for sp, ad in L['spectra']:
            pmh = TO_MH[ad](sp['precursor_mz'])
            qm, qi = m52.clean(sp['ms2_mzs'], sp['ms2_normalized_intensities'], pmh)
            for j, i in enumerate(slots):
                p = pred.get(f'{L["mid"]}|{i}|10')
                pm, pi = m52.clean(p[:, 0], p[:, 1], pmh) if p is not None and len(p) else (np.zeros(0),) * 2
                en[j] += m52.entropy_sim(qm, qi, pm, pi)
        score = z(-np.log(np.array(slots) + 1.0)) + W * z(en)
        for slot, o in zip(slots, np.argsort(-score, kind='stable')):
            new[slot] = lst[slots[o]]
    a = r['mid'][2:]
    return {'mid': r['mid'], 'set': L['set'], 'cls': int(r['cls']), 'lib_max': float(r['lib_max']),
            'n_slots': len(slots), 'rr0': m51.rr(lst, a), 'rr1': m51.rr(new, a), 'top1_changed': new[0] != lst[0]}


def summarize(pm):
    b = {'n_lists': len(pm), 'n_scored': int((pm.n_slots >= 2).sum()), 'changed_lists': int((pm.rr1 != pm.rr0).sum()),
         'top1_changed': int(pm.top1_changed.sum()), 'mrr0': pm.rr0.mean(), 'mrr1': pm.rr1.mean(),
         'boot': m51.boot((pm.rr1 - pm.rr0).values)}                     # mean, lo, hi, P(better)
    for c in (1, 2, 3):
        x = pm[pm.cls == c]
        b[f'class{c}'] = {'n': len(x), 'mrr0': x.rr0.mean(), 'mrr1': x.rr1.mean(),
                          'rank1_losses': int(((x.rr0 == 1) & (x.rr1 < 1)).sum()),
                          'rank1_gains': int(((x.rr0 < 1) & (x.rr1 == 1)).sum())}
    return b


def main():
    lists = load_lists()
    pred = m52.read_mgf(f'{ART}/53_bench_fiora_out.mgf')
    fm = f'{ART}/54_fresh_fiora_out.mgf'
    fresh = [L for L in lists if L['set'] == 'b']
    if not os.path.exists(fm):
        rows = m53.bench_fiora_input(fresh)
        fin = pd.DataFrame(rows, columns=['Name', 'SMILES', 'Precursor_type', 'CE', 'Instrument_type'])
        timing = m53.run_rows(fin, f'{ART}/54_fresh_in.csv', fm)
        json.dump(timing, open(f'{ART}/54_timing.json', 'w'))
    pred.update(m52.read_mgf(fm))
    pm = pd.DataFrame([rerank(L, pred) for L in lists])
    pm.to_csv(f'{ART}/54_per_list.csv', index=False)
    res = {'weight': W, **{s: summarize(pm[pm.set == s]) for s in ('a', 'b')}}
    res['accept'] = bool(res['b']['boot'][3] >= 0.9 and res['a']['class1']['rank1_losses'] == 0
                         and res['b']['class1']['rank1_losses'] == 0)
    json.dump(res, open(f'{ART}/54_summary.json', 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
