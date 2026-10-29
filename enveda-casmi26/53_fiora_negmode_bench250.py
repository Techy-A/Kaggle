"""FIORA-OS negative-mode isomer bench, enlarged to ~250 new held-out molecules + guarded genfix-bench check (CPU only).

What: (1) 250 new [M-H]- queries picked exactly as in 52 (tof instruments, single CE, >= 5 peaks, no enveda-180 rows as
query or isomer source, no genfix bench answer, no structure with a pluskal_ms2 row, one molecule per formula, 10-24
same-formula isomers) and additionally no formula shared with any of the 60 molecules of 52 (tune half + held-out 31);
(2) FIORA-OS predicts [M-H]- / HCD at CE 10 only; frozen settings from the 52 tune half: entropy similarity, 0.01 Da,
and the pre-registered combination z(fragment) + z(FIORA) within each list; nothing is retuned here; (3) top-1 / MRR
for FIORA, fragment, random, combination, paired bootstrap over molecules, alone and pooled with the 31 held-out of 52;
(4) if comb vs fragment passes P(better) >= 0.9: guarded rerank of the genfix-bench base lists (fin1) whose pseudo-test
query has [M-H]- spectra: only slots with the rank-1 formula move (ordered by comb), rank 1 moves only if the comb
margin >= 1.0, lists with lib_max >= 0.9 untouched; FIORA and fragment scores are averaged over the molecule's [M-H]-
spectra (rule fixed before scoring). Rank-1 losses per class and the MRR change with a bootstrap.
Why: CASMI isomer reranker research note §8/§9 - the 31-molecule result (P 0.94) was too small to decide step 3.
Output: artifacts/53_queries.parquet, 53_fiora_out.mgf, 53_bench_fiora_out.mgf (caches), 53_scores.parquet,
53_bench_per_molecule.csv, 53_summary.json.
usage: python 53_fiora_negmode_bench250.py
"""
import importlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fragments import explain_scores, fragment_masses  # noqa: E402

m52 = importlib.import_module('52_fiora_negmode_bench')
m51 = m52.m51
D, ART = m52.D, m52.ART
m52.CES = [10]                              # frozen choice from the 52 tune half (entropy, CE 10)
SEED, N_MOL, MARGIN, LIB_TAU = 1, 250, 1.0, 0.9
z = lambda g: (g - g.mean()) / (g.std() + 1e-9)  # noqa: E731


def read_rows(rows, cols):
    f, want, spec, off = pq.ParquetFile(f'{D}/train.parquet'), set(rows), {}, 0
    for rg in range(f.num_row_groups):
        n = f.metadata.row_group(rg).num_rows
        sel = [i - off for i in want if off <= i < off + n]
        if sel:
            for i, x in zip(sel, f.read_row_group(rg, columns=cols).take(sel).to_pylist()):
                spec[i + off] = x
        off += n
    return spec


def pick_queries():
    meta = pd.read_parquet(f'{D}/train.parquet', columns=['ingest_lib', 'inchikey14', 'normalized_smiles',
                                                          'molecular_formula', 'instrument_type', 'adduct',
                                                          'collision_energy_ev', 'num_peaks'])
    meta['row'] = np.arange(len(meta))
    meta = meta[meta.ingest_lib != 'enveda-180'].copy()
    meta['collision_energy_ev'] = meta.collision_energy_ev.map(m52._one_ce)
    bench = {r['mid'][2:] for r in m51.bench_records()}
    msnlib = set(meta.inchikey14[meta.ingest_lib == 'pluskal_ms2'])
    old_f = set(pd.read_parquet(f'{ART}/52_queries.parquet').formula)
    st = meta.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']]
    nform = st.molecular_formula.value_counts()
    q = meta[(meta.adduct == '[M-H]-') & meta.instrument_type.fillna('').str.lower().str.contains('tof')
             & meta.collision_energy_ev.between(5, 80) & (meta.num_peaks >= 5)
             & ~meta.inchikey14.isin(bench | msnlib) & (meta.molecular_formula.map(nform) >= 11)
             & ~meta.molecular_formula.isin(old_f) & (meta.normalized_smiles.str.len() <= 80)]
    q = q.sample(frac=1, random_state=SEED).drop_duplicates('inchikey14').drop_duplicates('molecular_formula')
    q = q.head(N_MOL)
    rng = np.random.default_rng(SEED)
    by_f = st.groupby('molecular_formula')
    rows = []
    for r in q.itertuples(index=False):
        oth = by_f.get_group(r.molecular_formula)
        oth = oth[oth.inchikey14 != r.inchikey14].normalized_smiles.values
        c = [r.normalized_smiles] + list(rng.choice(oth, min(len(oth), int(rng.integers(10, 25))), replace=False))
        rows.append({'qid': r.inchikey14, 'row': r.row, 'formula': r.molecular_formula, 'instrument': r.instrument_type,
                     'ingest_lib': r.ingest_lib, 'ce_ev': r.collision_energy_ev, 'cands': c, 'split': 'test'})
    qq = pd.DataFrame(rows)
    cols = ['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']
    spec = read_rows(qq.row, cols)
    for c in cols:
        qq[c] = qq.row.map(lambda i: spec[i][c])  # noqa: B023
    return qq


def run_rows(df, csv, mgf, n_shards=4):
    """FIORA predict, 4 single-threaded shards (one process with default threads is ~60x slower)."""
    env = dict(os.environ, PYTHONPATH=f'{m52.FIORA_SITE}:{m52.FIORA_SRC}', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    t0, procs = time.time(), []
    for k in range(n_shards):
        df.iloc[k::n_shards].to_csv(f'{csv}.{k}', index=False)
        procs.append(subprocess.Popen([sys.executable, '-m', 'fiora.cli.predict', '-i', f'{csv}.{k}', '-o',
                                       f'{mgf}.{k}.mgf', '--dev', 'cpu'], env=env, stdout=subprocess.DEVNULL))
    assert all(p.wait() == 0 for p in procs)
    with open(mgf, 'w') as out:
        for k in range(n_shards):
            out.write(open(f'{mgf}.{k}.mgf').read())
            os.remove(f'{mgf}.{k}.mgf')
            os.remove(f'{csv}.{k}')
    return {'n_predictions': len(df), 'wall_s': time.time() - t0}


def evaluate(df):
    df = df.assign(comb=df.groupby('qid').frag.transform(z) + df.groupby('qid').ent_10.transform(z))
    rr = {c: m52.per_mol(df, c) for c in ['ent_10', 'frag', 'rand', 'comb']}
    res = {c: {'mrr': float(v.mean()), 'top1': float((v == 1).mean())} for c, v in rr.items()}
    res['n'] = int(df.qid.nunique())
    res['boot_comb_vs_frag'] = m51.boot((rr['comb'] - rr['frag']).values)    # mean, lo, hi, P(better)
    res['boot_fiora_vs_frag'] = m51.boot((rr['ent_10'] - rr['frag']).values)
    res['boot_comb_vs_fiora'] = m51.boot((rr['comb'] - rr['ent_10']).values)
    return res


def bench_lists():
    """[M-H]- pseudo-test query spectra (enveda rows of the answer structure) + fin1 candidates with smiles/formula."""
    recs = {r['mid']: r for r in m51.bench_records() if 'fin1' in r}
    meta = pd.read_parquet(f'{D}/train.parquet', columns=['ingest_lib', 'inchikey14', 'adduct'])
    meta['row'] = np.arange(len(meta))
    q = meta[(meta.adduct == '[M-H]-') & meta.ingest_lib.isin(['enveda-180', 'enveda-np-examples'])]
    mids = {k: m for m in recs for k in [m[2:]]}
    # b_ queries are enveda-np-examples rows, t_ queries enveda-180 rows (build_genfix_bench.py)
    q = q[q.inchikey14.isin(mids) & (q.ingest_lib == q.inchikey14.map(
        lambda k: 'enveda-np-examples' if mids.get(k, 'x')[0] == 'b' else 'enveda-180'))]
    spec = read_rows(q.row, ['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
    bt = pd.read_parquet(f'{ART}/51_bench_feats.parquet', columns=['qid', 'pos', 'smiles', 'formula'])
    out = []
    for k, g in q.groupby('inchikey14'):
        mid = mids[k]
        c = bt[bt.qid == mid].sort_values('pos')
        out.append({'mid': mid, 'rec': recs[mid], 'spectra': [spec[i] for i in g.row],
                    'smiles': list(c.smiles), 'formula': [None if s == '' else f for s, f in zip(c.smiles, c.formula)]})
    return out


def bench_check(lists, pred):
    rows = []
    for L in lists:
        r, fs, sm = L['rec'], L['formula'], L['smiles']
        lst = r['fin1'][:25]
        slots = [i for i, f in enumerate(fs) if f is not None and f == fs[0]] if float(r['lib_max']) < LIB_TAU else []
        new, moved = list(lst), False
        if len(slots) >= 2:
            fr, en = np.zeros(len(slots)), np.zeros(len(slots))
            for sp in L['spectra']:
                qm, qi = m52.clean(sp['ms2_mzs'], sp['ms2_normalized_intensities'], sp['precursor_mz'])
                top = np.argsort(-qi)[:40]
                for j, i in enumerate(slots):
                    try:
                        f1, f2 = fragment_masses(sm[i])
                    except Exception:
                        f1 = f2 = np.zeros(0)
                    fr[j] += explain_scores(qm[top], np.sqrt(qi[top]), f1, f2, positive=False)[0]
                    p = pred.get(f'{L["mid"]}|{i}|10')
                    pm, pi = m52.clean(p[:, 0], p[:, 1], sp['precursor_mz']) if p is not None and len(p) \
                        else (np.zeros(0),) * 2
                    en[j] += m52.entropy_sim(qm, qi, pm, pi)
            comb = z(fr) + z(en)              # sums over spectra; z makes this the mean rule
            order = list(np.argsort(-comb, kind='stable'))
            if order[0] != 0 and comb[order[0]] - comb[0] < MARGIN:
                order = [0] + [o for o in order if o != 0]
            for slot, o in zip(slots, order):
                new[slot] = lst[slots[o]]
            moved = True
        a = r['mid'][2:]
        rows.append({'mid': r['mid'], 'cls': int(r['cls']), 'lib_max': float(r['lib_max']), 'n_slots': len(slots),
                     'scored': moved, 'rr0': m51.rr(lst, a), 'rr1': m51.rr(new, a), 'top1_changed': new[0] != lst[0]})
    return pd.DataFrame(rows)


def bench_fiora_input(lists):
    rows = []
    for L in lists:
        fs = L['formula']
        if float(L['rec']['lib_max']) >= LIB_TAU:
            continue
        rows += [(f'{L["mid"]}|{i}|10', L['smiles'][i], '[M-H]-', 10, 'HCD')
                 for i, f in enumerate(fs) if f is not None and f == fs[0]]
    return rows


def main():
    qf, mgf = f'{ART}/53_queries.parquet', f'{ART}/53_fiora_out.mgf'
    if not os.path.exists(qf):
        pick_queries().to_parquet(qf)
    qq = pd.read_parquet(qf)
    qq['cands'] = qq.cands.map(list)
    if not os.path.exists(mgf):
        rows = [(f'{r.qid}|{i}|10', s, '[M-H]-', 10, 'HCD') for r in qq.itertuples() for i, s in enumerate(r.cands)]
        fin = pd.DataFrame(rows, columns=['Name', 'SMILES', 'Precursor_type', 'CE', 'Instrument_type'])
        json.dump(run_rows(fin, f'{ART}/53_fiora_in.csv', mgf), open(f'{ART}/53_timing.json', 'w'))
    df = m52.score(qq, m52.read_mgf(mgf))
    df.to_parquet(f'{ART}/53_scores.parquet')
    old = pd.read_parquet(f'{ART}/52_scores.parquet')
    old = old[old.split == 'test'][df.columns.intersection(old.columns)]
    res = {'timing': json.load(open(f'{ART}/53_timing.json')), 'missing_preds': int(df.missing.sum()),
           'mean_cands': float(df.groupby('qid').size().mean()),
           'instruments': qq.instrument.value_counts().to_dict(), 'libs': qq.ingest_lib.value_counts().to_dict(),
           'query_ce_ev': qq.ce_ev.describe().round(1).to_dict(),
           'expected_random_mrr': float(np.mean([np.mean(1 / np.arange(1, n + 1)) for n in df.groupby('qid').size()])),
           'new': evaluate(df), 'pooled_with_52_test31': evaluate(pd.concat([df, old], ignore_index=True))}
    gate = res['new']['boot_comb_vs_frag'][3] >= 0.9
    res['gate_pass'] = bool(gate)
    if gate:
        lists = bench_lists()
        bm = f'{ART}/53_bench_fiora_out.mgf'
        if not os.path.exists(bm):
            rows = bench_fiora_input(lists)
            fin = pd.DataFrame(rows, columns=['Name', 'SMILES', 'Precursor_type', 'CE', 'Instrument_type'])
            res['bench_timing'] = run_rows(fin, f'{ART}/53_bench_in.csv', bm)
        pm = bench_check(lists, m52.read_mgf(bm))
        pm.to_csv(f'{ART}/53_bench_per_molecule.csv', index=False)
        b = {'n_lists_mh': len(pm), 'n_gated_lib': int((pm.lib_max >= LIB_TAU).sum()),
             'n_scored': int(pm.scored.sum()), 'changed_lists': int((pm.rr1 != pm.rr0).sum()),
             'top1_changed': int(pm.top1_changed.sum()), 'natural': m51.boot((pm.rr1 - pm.rr0).values)}
        for c in (1, 2, 3):
            x = pm[pm.cls == c]
            b[f'class{c}'] = {'n': len(x), 'mrr0': x.rr0.mean(), 'mrr1': x.rr1.mean(),
                              'rank1_losses': int(((x.rr0 == 1) & (x.rr1 < 1)).sum()),
                              'rank1_gains': int(((x.rr0 < 1) & (x.rr1 == 1)).sum())}
        res['bench'] = b
    json.dump(res, open(f'{ART}/53_summary.json', 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
