"""FIORA-OS negative-mode isomer bench (CPU only, offline; ~1 h on 4 cores incl. FIORA).

What: (1) pseudo-test of 60 [M-H]- molecules from train.parquet: one QTOF-like spectrum each (instrument_type contains
'tof', collision_energy_ev known, >= 5 peaks), EXCLUDING every ingest_lib == enveda-180 row (test copies), structures in
the genfix bench answers, and structures with any pluskal_ms2 row (MSnLib = FIORA-OS training data, leakage guard);
each query gets 10-24 same-formula isomers drawn from other (non-enveda-180) training structures;
(2) FIORA-OS v1.0.0 predicts every candidate as [M-H]- / HCD at a CE grid fixed in advance (10, 20, 30, 40) plus the
grid mean; (3) score = cosine (sqrt intensities, 0.01 Da greedy match, peaks >= precursor - 1 removed on both sides)
and spectral entropy similarity; (4) CE option chosen on the tune half (formula-grouped split) by MRR, reported on the
held-out half: top-1 and MRR vs fragments.explain_scores (negative mode) vs random, paired bootstrap over molecules.
Why: CASMI isomer reranker research note §7, step 2 (A2) - is FIORA a stronger isomer signal than fragment matching?
Output: artifacts/52_queries.parquet, 52_fiora_in.csv, 52_fiora_out.mgf (caches; delete to recompute),
artifacts/52_scores.parquet, 52_summary.json.
usage: python 52_fiora_negmode_bench.py   (FIORA at /home/user/work/ext_fiora, set FIORA_SITE / FIORA_SRC to move it)
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

m51 = importlib.import_module('51_isomer_reranker')
D, ART = m51.D, m51.ART
FIORA_SITE = os.environ.get('FIORA_SITE', '/home/user/work/ext_fiora/site')
FIORA_SRC = os.environ.get('FIORA_SRC', '/home/user/work/ext_fiora/src')
SEED, N_MOL, TOL, N_SHARDS = 0, 60, 0.01, 4
CES = [10, 20, 30, 40]                      # fixed before any FIORA run
OPTS = [str(c) for c in CES] + ['mean']


def _one_ce(v):
    """Single collision energy in eV, NaN when missing or stepped (several energies)."""
    v = np.atleast_1d(np.asarray(v, float)) if v is not None else np.zeros(0)
    return float(v[0]) if len(v) == 1 else np.nan


def pick_queries():
    meta = pd.read_parquet(f'{D}/train.parquet', columns=['ingest_lib', 'inchikey14', 'normalized_smiles',
                                                          'molecular_formula', 'instrument_type', 'adduct',
                                                          'collision_energy_ev', 'num_peaks'])
    meta['row'] = np.arange(len(meta))
    meta = meta[meta.ingest_lib != 'enveda-180'].copy()
    meta['collision_energy_ev'] = meta.collision_energy_ev.map(_one_ce)
    bench = {r['mid'][2:] for r in m51.bench_records()}
    msnlib = set(meta.inchikey14[meta.ingest_lib == 'pluskal_ms2'])
    st = meta.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']]
    nform = st.molecular_formula.value_counts()
    q = meta[(meta.adduct == '[M-H]-') & meta.instrument_type.fillna('').str.lower().str.contains('tof')
             & meta.collision_energy_ev.between(5, 80) & (meta.num_peaks >= 5)
             & ~meta.inchikey14.isin(bench | msnlib) & (meta.molecular_formula.map(nform) >= 11)
             & (meta.normalized_smiles.str.len() <= 80)]
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
                     'ingest_lib': r.ingest_lib, 'ce_ev': r.collision_energy_ev, 'cands': c})
    qq = pd.DataFrame(rows)
    f, want, spec = pq.ParquetFile(f'{D}/train.parquet'), set(qq.row), {}
    off = 0
    for rg in range(f.num_row_groups):
        n = f.metadata.row_group(rg).num_rows
        sel = [i - off for i in want if off <= i < off + n]
        if sel:
            t = f.read_row_group(rg, columns=['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']).take(sel)
            for i, x in zip(sel, t.to_pylist()):
                spec[i + off] = x
        off += n
    for c in ['precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities']:
        qq[c] = qq.row.map(lambda i: spec[i][c])  # noqa: B023
    qq['split'] = np.where(rng.random(len(qq)) < 0.5, 'tune', 'test')   # one molecule per formula -> grouped
    return qq


def run_fiora(qq, csv, mgf):
    rows = [(f'{r.qid}|{i}|{ce}', s, '[M-H]-', ce, 'HCD')
            for r in qq.itertuples() for i, s in enumerate(r.cands) for ce in CES]
    df = pd.DataFrame(rows, columns=['Name', 'SMILES', 'Precursor_type', 'CE', 'Instrument_type'])
    df.to_csv(csv, index=False)
    env = dict(os.environ, PYTHONPATH=f'{FIORA_SITE}:{FIORA_SRC}', OMP_NUM_THREADS='1')
    t0, procs = time.time(), []
    for k in range(N_SHARDS):                   # ~2.4 s per prediction single-process: shard over the 4 cores
        df.iloc[k::N_SHARDS].to_csv(f'{csv}.{k}', index=False)
        procs.append(subprocess.Popen([sys.executable, '-m', 'fiora.cli.predict', '-i', f'{csv}.{k}', '-o',
                                       f'{mgf}.{k}.mgf', '--dev', 'cpu'], env=env, stdout=subprocess.DEVNULL))
    assert all(p.wait() == 0 for p in procs)
    with open(mgf, 'w') as out:
        for k in range(N_SHARDS):
            out.write(open(f'{mgf}.{k}.mgf').read())
            os.remove(f'{mgf}.{k}.mgf')
            os.remove(f'{csv}.{k}')
    return len(rows), time.time() - t0


def read_mgf(path):
    out, name, pk = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('TITLE='):
            name, pk = line[6:], []
        elif line == 'END IONS':
            out[name] = np.array(pk).reshape(-1, 2)
        elif line and line[0].isdigit():
            pk.append([float(v) for v in line.split()[:2]])
    return out


def clean(mz, it, prec):
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    k = (mz < prec - 1.0) & (it > 0)
    return mz[k], it[k]


def match(a_mz, a_it, b_mz, b_it):
    """Greedy one-to-one peak matching within TOL, largest product first; returns matched index pairs."""
    if not len(a_mz) or not len(b_mz):
        return []
    d = np.abs(a_mz[:, None] - b_mz[None, :])
    i, j = np.where(d <= TOL)
    pairs, ua, ub = [], set(), set()
    for k in np.argsort(-(a_it[i] * b_it[j])):
        if i[k] not in ua and j[k] not in ub:
            pairs.append((i[k], j[k]))
            ua.add(i[k])
            ub.add(j[k])
    return pairs


def cosine(a_mz, a_it, b_mz, b_it):
    a, b = np.sqrt(a_it), np.sqrt(b_it)
    p = match(a_mz, a, b_mz, b)
    if not p:
        return 0.0
    return float(sum(a[i] * b[j] for i, j in p) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _ent(x):
    x = x[x > 0] / x.sum()
    return float(-(x * np.log(x)).sum())


def entropy_sim(a_mz, a_it, b_mz, b_it):
    """Unweighted spectral entropy similarity (Li et al. 2021) on matched peaks, 0.01 Da."""
    if not len(a_mz) or not len(b_mz):
        return 0.0
    a, b = a_it / a_it.sum(), b_it / b_it.sum()
    p = match(a_mz, a, b_mz, b)
    am, bm = a.copy(), b.copy()
    mix = [(a[i] + b[j]) / 2 for i, j in p]
    for i, j in p:
        am[i] = bm[j] = 0
    m = np.concatenate([np.array(mix), am[am > 0] / 2, bm[bm > 0] / 2])
    return float(1 - (2 * _ent(m) - _ent(a) - _ent(b)) / np.log(4))


def score(qq, pred):
    rng = np.random.default_rng(SEED)
    rows = []
    for r in qq.itertuples():
        qm, qi = clean(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        top = np.argsort(-qi)[:40]
        qm40, qs40 = qm[top], np.sqrt(qi[top])
        for i, s in enumerate(r.cands):
            row = {'qid': r.qid, 'pos': i, 'split': r.split, 'rand': rng.random()}
            try:
                f1, f2 = fragment_masses(s)
            except Exception:
                f1 = f2 = np.zeros(0)
            row['frag'] = explain_scores(qm40, qs40, f1, f2, positive=False)[0]
            cs, es = [], []
            for ce in CES:
                p = pred.get(f'{r.qid}|{i}|{ce}')
                pm, pi = clean(p[:, 0], p[:, 1], r.precursor_mz) if p is not None and len(p) else (np.zeros(0),) * 2
                cs.append(cosine(qm, qi, pm, pi))
                es.append(entropy_sim(qm, qi, pm, pi))
                row[f'cos_{ce}'], row[f'ent_{ce}'] = cs[-1], es[-1]
            row['cos_mean'], row['ent_mean'] = np.mean(cs), np.mean(es)
            row['missing'] = int(any(f'{r.qid}|{i}|{ce}' not in pred for ce in CES))
            rows.append(row)
    return pd.DataFrame(rows)


def per_mol(df, col):
    """Reciprocal rank of the true structure (pos 0), expected value under random tie-breaking."""
    out = {}
    for q, g in df.groupby('qid'):
        v = g.set_index('pos')[col]
        o = v.drop(0)
        b, t = int((o > v[0]).sum()), int((o == v[0]).sum())
        out[q] = float(np.mean(1.0 / (b + 1 + np.arange(t + 1))))
    return pd.Series(out)


def main():
    os.makedirs(ART, exist_ok=True)
    qf, csv, mgf = f'{ART}/52_queries.parquet', f'{ART}/52_fiora_in.csv', f'{ART}/52_fiora_out.mgf'
    if not os.path.exists(qf):
        pick_queries().to_parquet(qf)
    qq = pd.read_parquet(qf)
    qq['cands'] = qq.cands.map(list)
    if not os.path.exists(mgf):
        n, sec = run_fiora(qq, csv, mgf)
        json.dump({'n_predictions': n, 'wall_s': sec, 's_per_prediction': sec / n}, open(f'{ART}/52_timing.json', 'w'))
    timing = json.load(open(f'{ART}/52_timing.json'))
    df = score(qq, read_mgf(mgf))
    df.to_parquet(f'{ART}/52_scores.parquet')
    tune, test = df[df.split == 'tune'], df[df.split == 'test']
    tune_mrr = {f'{k}_{o}': float(per_mol(tune, f'{k}_{o}').mean()) for k in ('cos', 'ent') for o in OPTS}
    best = max(tune_mrr, key=tune_mrr.get)
    res = {'n_mol': int(df.qid.nunique()), 'n_tune': int(tune.qid.nunique()), 'n_test': int(test.qid.nunique()),
           'mean_cands': float(df.groupby('qid').size().mean()), 'missing_preds': int(df.missing.sum()),
           'instruments': qq.instrument.value_counts().to_dict(), 'libs': qq.ingest_lib.value_counts().to_dict(),
           'query_ce_ev': qq.ce_ev.describe().round(1).to_dict(), 'tune_mrr': tune_mrr, 'chosen': best,
           'timing': timing, 'expected_random_mrr_test': float(np.mean(
               [np.mean(1 / np.arange(1, n + 1)) for n in test.groupby('qid').size()]))}
    rr = {c: per_mol(test, c) for c in [best, 'frag', 'rand', 'cos_mean', 'ent_mean']}
    for c, v in rr.items():
        res[f'test_{c}'] = {'mrr': float(v.mean()), 'top1': float((v == 1).mean())}
    for c in ('frag', 'rand'):
        res[f'boot_{best}_vs_{c}'] = m51.boot((rr[best] - rr[c]).values)    # mean, lo, hi, P(better)
    # simple combination, fixed in advance: z(frag) + z(chosen FIORA score) within each list
    z = lambda g: (g - g.mean()) / (g.std() + 1e-9)  # noqa: E731
    test = test.assign(comb=test.groupby('qid').frag.transform(z) + test.groupby('qid')[best].transform(z))
    rc = per_mol(test, 'comb')
    res['test_comb'] = {'mrr': float(rc.mean()), 'top1': float((rc == 1).mean())}
    res['boot_comb_vs_frag'] = m51.boot((rc - rr['frag']).values)
    full = {c: per_mol(df, c) for c in [best, 'frag']}
    res['all60_mrr'] = {c: float(v.mean()) for c, v in full.items()}
    json.dump(res, open(f'{ART}/52_summary.json', 'w'), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main()
