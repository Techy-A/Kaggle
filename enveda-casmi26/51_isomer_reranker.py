"""Same-formula isomer reranker: LightGBM lambdarank on fragment-match features (CPU only, offline, ~1 h on 4 cores).

What: (1) builds ~3,000 training queries from train.parquet: one [M+H]+ structure plus 10-24 same-formula isomers drawn
from other training structures, 5 folds held out by Murcko scaffold of the query structure (bench answers excluded);
(2) per candidate, fragment-match features from fragments.fragment_masses (1-cut and 2-cut matches separately, NeckBeard
settings: 2-cut weight 0.6, linear intensity, 0.005 Da, H shifts -2..+3), each also z-scored within its list;
(3) LightGBM lambdarank (depth 6, 300 rounds), fold-held-out isomer-set MRR against the fragment-score-only baseline and
random; (4) applies the 5 fold models (averaged) to the genfix-bench base lists (bench/results/genfix_bench_log.txt.gz,
`fin1`, 450 molecules), reordering only the slots holding the rank-1 candidate's formula, by z(-log rank) + ALPHA *
z(model), with guards fixed before looking at the bench: lists with lib_max >= 0.9 untouched, rank 1 changes only if the
challenger beats it by MARGIN combined z-units. Paired bootstrap over molecules (10k); zero thin class-1 rank-1 losses
required. (5) cheap check of the test-in-train anomaly: exact array equality of a test sample against train rows.
No training-set-membership feature, no cross-test-spectrum feature.
Why: CASMI isomer reranker research note, 'recommended first experiment' (user step 3, 29 Oct 2026).
Caveat: bench query peaks are the answer's own [M+H]+ train spectra (the bench's hidden queries are copies of those).
Output: artifacts/51_train_feats.parquet, 51_bench_feats.parquet (caches; delete to recompute), 51_summary.json.
usage: python 51_isomer_reranker.py
"""
import gzip
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragments import fragment_masses  # noqa: E402

D = '/home/user/enveda_data'
HERE = os.path.dirname(os.path.abspath(__file__))
ART = f'{HERE}/artifacts'
BENCH_LOG = f'{HERE}/bench/results/genfix_bench_log.txt.gz'
KEYS_PKL = f'{HERE}/bench/results/calibration_keys.pkl'
SEED, N_QUERIES, N_FOLDS = 0, 3000, 5
H, PROTON, TOL, W2 = 1.0078250319, 1.007276, 0.005, 0.6
SHIFTS = np.arange(-2, 4) * H
ALPHA, MARGIN, LIB_TAU = 1.0, 1.0, 0.9          # fixed before the bench was scored
RAW = ['i1', 'c1', 'i2', 'c2', 'score', 'top5', 'base', 'nf1', 'nf2']


def peaks(spectra, top=40, max_spec=5):
    """Merged fragment peaks (below precursor - 1) of up to max_spec spectra, linear relative intensity."""
    mzs, its = [], []
    for a, b, p in list(zip(spectra.ms2_mzs, spectra.ms2_normalized_intensities, spectra.precursor_mz))[:max_spec]:
        mz, it = np.asarray(a, float), np.asarray(b, float)
        keep = mz < p - 1.0
        mz, it = mz[keep], it[keep]
        if len(mz) > top:
            sel = np.argpartition(-it, top)[:top]
            mz, it = mz[sel], it[sel]
        if len(it) and it.max() > 0:
            mzs.append(mz)
            its.append(it / it.max())
    if not mzs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(mzs), np.concatenate(its)


def _hit(mz, neutral):
    if len(neutral) == 0 or len(mz) == 0:
        return np.zeros(len(mz), bool)
    ions = np.sort((neutral[:, None] + SHIFTS[None, :] + PROTON).ravel())
    idx = np.searchsorted(ions, mz)
    lo = np.abs(ions[np.clip(idx - 1, 0, len(ions) - 1)] - mz)
    hi = np.abs(ions[np.clip(idx, 0, len(ions) - 1)] - mz)
    return np.minimum(lo, hi) <= TOL


_FRAG = {}


def feats(mz, it, smiles):
    if smiles not in _FRAG:
        try:
            _FRAG[smiles] = fragment_masses(smiles)
        except Exception:
            _FRAG[smiles] = (np.zeros(0), np.zeros(0))
    f1, f2 = _FRAG[smiles]
    h1, h2 = _hit(mz, f1), _hit(mz, f2) & ~_hit(mz, f1)
    tot = it.sum() + 1e-12
    w = np.where(h1, 1.0, np.where(h2, W2, 0.0))
    top = np.argsort(-it)[:5]
    return [(h1 * it).sum() / tot, h1.mean() if len(mz) else 0, (h2 * it).sum() / tot, h2.mean() if len(mz) else 0,
            (w * it).sum() / tot, (w[top] > 0).mean() if len(mz) else 0, float(w[top[0]] > 0) if len(mz) else 0,
            np.log1p(len(f1)), np.log1p(len(f2))]


def score_group(args):
    qid, mz, it, cands = args
    return [[qid, i, s] + feats(mz, it, s) for i, s in enumerate(cands)]


def add_z(df):
    g = df.groupby('qid')
    for c in RAW:
        df[c + '_z'] = (df[c] - g[c].transform('mean')) / (g[c].transform('std').fillna(0) + 1e-9)
    return df


def load_spectra(keys):
    """[M+H]+ spectra of the given inchikey14s, streamed row group by row group."""
    f, out = pq.ParquetFile(f'{D}/train.parquet'), []
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=['inchikey14', 'adduct']).to_pandas()
        sel = np.where(t.inchikey14.isin(keys) & (t.adduct == '[M+H]+'))[0]
        if len(sel):
            r = f.read_row_group(rg, columns=['inchikey14', 'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities'])
            out.append(r.take(sel).to_pandas())
    return pd.concat(out, ignore_index=True)


def bench_records():
    return [json.loads(l[9:]) for l in gzip.open(BENCH_LOG, 'rt') if l.startswith('GFB_JSON')]


def build_train(meta, bench_keys):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    rng = np.random.default_rng(SEED)
    st = meta.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']]
    mh = set(meta.inchikey14[meta.adduct == '[M+H]+'])
    nform = st.molecular_formula.map(st.molecular_formula.value_counts())
    pool = st[(nform >= 11) & st.inchikey14.isin(mh) & ~st.inchikey14.isin(bench_keys)]
    q = pool.sample(N_QUERIES, random_state=SEED)
    by_f = st.groupby('molecular_formula')
    groups = []
    for k, s, fml in q.itertuples(index=False):
        oth = by_f.get_group(fml)
        oth = oth[oth.inchikey14 != k].normalized_smiles.values
        n = min(len(oth), int(rng.integers(10, 25)))
        groups.append((k, [s] + list(rng.choice(oth, n, replace=False))))
    scaf = {}
    for k, c in groups:
        m = Chem.MolFromSmiles(c[0])
        scaf[k] = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else k
    us = sorted(set(scaf.values()))
    perm = rng.permutation(len(us))
    fold = {u: perm[i] % N_FOLDS for i, u in enumerate(us)}
    return groups, {k: fold[scaf[k]] for k in scaf}


def run_pool(tasks):
    with Pool(4) as p:
        rows = [r for g in p.imap_unordered(score_group, tasks, chunksize=8) for r in g]
    return pd.DataFrame(rows, columns=['qid', 'pos', 'smiles'] + RAW)


def mrr(df, col):
    r = df.groupby('qid')[col].rank(ascending=False, method='first')
    return float((1 / r[df.pos == 0]).mean())


def train_eval(tr):
    import lightgbm as lgb
    feats_ = RAW + [c + '_z' for c in RAW]
    tr = tr.sort_values(['qid', 'pos']).reset_index(drop=True)
    tr['y'] = (tr.pos == 0).astype(int)
    tr['oof'] = 0.0
    models = []
    for f in range(N_FOLDS):
        a, b = tr[tr.fold != f], tr[tr.fold == f]
        m = lgb.LGBMRanker(objective='lambdarank', max_depth=6, num_leaves=31, n_estimators=300, learning_rate=0.05,
                           min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                           random_state=SEED, verbose=-1)
        m.fit(a[feats_], a.y, group=a.groupby('qid', sort=False).size().values)
        tr.loc[b.index, 'oof'] = m.predict(b[feats_])
        models.append(m)
    rng = np.random.default_rng(SEED)
    tr['rand'] = rng.random(len(tr))
    res = {'n_queries': int(tr.qid.nunique()), 'n_rows': len(tr),
           'mrr_lgbm_oof': mrr(tr, 'oof'), 'mrr_fragscore': mrr(tr, 'score'), 'mrr_random': mrr(tr, 'rand'),
           'expected_random': float(np.mean([np.mean(1 / np.arange(1, n + 1)) for n in tr.groupby('qid').size()]))}
    res['per_fold'] = {int(f): [mrr(tr[tr.fold == f], 'oof'), mrr(tr[tr.fold == f], 'score')] for f in range(N_FOLDS)}
    top1 = tr.groupby('qid').apply(lambda g: g.pos.values[np.argmax(g.oof.values)] == 0).mean()
    res['top1_lgbm_oof'], res['top1_fragscore'] = float(top1), float(
        tr.groupby('qid').apply(lambda g: g.pos.values[np.argmax(g.score.values)] == 0).mean())
    return models, feats_, res


def rerank(lst, model_score, formulas, lib_max):
    """Returns the new list; only slots holding the rank-1 formula are permuted."""
    if lib_max >= LIB_TAU or len(lst) < 2:
        return list(lst)
    slots = [i for i, f in enumerate(formulas) if f is not None and f == formulas[0]]
    if len(slots) < 2:
        return list(lst)
    ms = np.array([model_score[i] for i in slots])
    br = -np.log(np.array(slots) + 1.0)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)  # noqa: E731
    comb = z(br) + ALPHA * z(ms)
    order = list(np.argsort(-comb, kind='stable'))
    if order[0] != 0 and comb[order[0]] - comb[0] < MARGIN:
        order = [0] + [o for o in order if o != 0]
    new = list(lst)
    for slot, o in zip(slots, order):
        new[slot] = lst[slots[o]]
    return new


def rr(lst, ans):
    return 1.0 / (lst.index(ans) + 1) if ans in lst[:25] else 0.0


def anomaly():
    te = pd.read_parquet(f'{D}/test.parquet')
    s = te.sample(40, random_state=SEED)
    f = pq.ParquetFile(f'{D}/train.parquet')
    found = []
    pm = np.round(s.precursor_mz.values, 4)
    for rg in range(f.num_row_groups):
        t = f.read_row_group(rg, columns=['precursor_mz', 'num_peaks'])
        tp = np.round(t.column('precursor_mz').to_numpy(), 4)
        sel = np.where(np.isin(tp, pm))[0]
        if not len(sel):
            continue
        r = f.read_row_group(rg, columns=['ingest_lib', 'inchikey14', 'adduct', 'precursor_mz', 'ms2_mzs',
                                          'ms2_normalized_intensities']).take(sel).to_pandas()
        for _, q in s.iterrows():
            for _, x in r[np.round(r.precursor_mz, 4) == round(q.precursor_mz, 4)].iterrows():
                if len(x.ms2_mzs) == len(q.ms2_mzs) and np.array_equal(np.asarray(x.ms2_mzs), np.asarray(q.ms2_mzs)):
                    found.append({'molecule_id': q.molecule_id, 'ingest_lib': x.ingest_lib, 'ik14': x.inchikey14,
                                  'adduct': x.adduct, 'int_equal': bool(np.allclose(
                                      x.ms2_normalized_intensities, q.ms2_normalized_intensities))})
    fd = pd.DataFrame(found)
    return {'sample': 40, 'test_cols': list(te.columns),
            'n_test_with_exact_mz_match': int(fd.molecule_id.nunique()) if len(fd) else 0,
            'n_train_rows_matched': len(fd),
            'ingest_lib_counts': fd.ingest_lib.value_counts().to_dict() if len(fd) else {},
            'intensities_equal_frac': float(fd.int_equal.mean()) if len(fd) else None,
            'distinct_ik14_per_test_spectrum': float(fd.groupby('molecule_id').ik14.nunique().mean()) if len(fd) else None,
            'examples': fd.head(8).to_dict('records')}


def boot(d, n=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), float((b > 0).mean())


def main():
    os.makedirs(ART, exist_ok=True)
    recs = bench_records()
    bench_keys = {r['mid'][2:] for r in recs}
    meta = None
    tf = f'{ART}/51_train_feats.parquet'
    if not os.path.exists(tf):
        meta = pd.read_parquet(f'{D}/train.parquet', columns=['inchikey14', 'normalized_smiles',
                                                                'molecular_formula', 'adduct'])
        groups, fold = build_train(meta, bench_keys)
        sp = load_spectra(set(fold))
        tasks = []
        for k, c in groups:
            mz, it = peaks(sp[sp.inchikey14 == k])
            tasks.append((k, mz, it, c))
        tr = run_pool(tasks)
        tr['fold'] = tr.qid.map(fold)
        tr.to_parquet(tf)
    tr = add_z(pd.read_parquet(tf))
    models, fcols, res = train_eval(tr)
    print(json.dumps(res, indent=1))

    bf = f'{ART}/51_bench_feats.parquet'
    if not os.path.exists(bf):
        import pickle
        k2s = {}
        for s, k in pickle.load(open(KEYS_PKL, 'rb')).items():
            k2s.setdefault(k, s)
        if meta is None:
            meta = pd.read_parquet(f'{D}/train.parquet', columns=['inchikey14', 'normalized_smiles'])
        for k, s in meta.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles']].itertuples(index=False):
            k2s.setdefault(k, s)
        sp = load_spectra(bench_keys)
        tasks = []
        for r in recs:
            mz, it = peaks(sp[sp.inchikey14 == r['mid'][2:]])
            lst = r['fin1'][:25]
            tasks.append((r['mid'], mz, it, [k2s.get(k, '') for k in lst]))
        bt = run_pool(tasks)
        from rdkit import Chem
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        def fml(s):  # noqa: E306
            m = Chem.MolFromSmiles(s) if s else None
            return CalcMolFormula(m) if m else None
        bt['formula'] = bt.smiles.map(fml)
        bt.to_parquet(bf)
    bt = add_z(pd.read_parquet(bf))
    bt['m'] = np.mean([m.predict(bt[fcols]) for m in models], 0)
    bt.loc[bt.smiles == '', 'm'] = np.nan
    rows = []
    for r in recs:
        g = bt[bt.qid == r['mid']].sort_values('pos')
        lst = r['fin1'][:25]
        ms = g.m.fillna(g.m.min() if g.m.notna().any() else 0).values
        fs = [None if s == '' else f for s, f in zip(g.smiles, g.formula)]
        new = rerank(lst, ms, fs, r['lib_max'])
        a = r['mid'][2:]
        rows.append({'mid': r['mid'], 'cls': r['cls'], 'thin': r['mid'][:2], 'lib_max': r['lib_max'],
                     'rr0': rr(lst, a), 'rr1': rr(new, a), 'top1_changed': new[0] != lst[0],
                     'ans_iso_of_top1': a in lst and fs[lst.index(a)] == fs[0] if a in lst else False,
                     'missing_smiles': int((g.smiles == '').sum())})
    pm = pd.DataFrame(rows)
    pm.to_csv(f'{ART}/51_bench_per_molecule.csv', index=False)
    out = {'train': res, 'alpha': ALPHA, 'margin': MARGIN, 'lib_tau': LIB_TAU}
    d = (pm.rr1 - pm.rr0).values
    out['natural'] = boot(d)
    w = pm.cls.map({1: 0.28, 2: 0.20, 3: 0.52}) / pm.cls.map(pm.cls.value_counts(normalize=True))
    out['mix_28_20_52'] = boot(d * w.values)
    for c in (1, 2, 3):
        x = pm[pm.cls == c]
        out[f'class{c}'] = {'n': len(x), 'mrr0': x.rr0.mean(), 'mrr1': x.rr1.mean(),
                            'rank1_losses': int(((x.rr0 == 1) & (x.rr1 < 1)).sum()),
                            'rank1_gains': int(((x.rr0 < 1) & (x.rr1 == 1)).sum()),
                            'top1_changed': int(x.top1_changed.sum())}
    out['changed_lists'] = int((pm.rr1 != pm.rr0).sum())
    out['gated_lib'] = int((pm.lib_max >= LIB_TAU).sum())
    out['missing_smiles_mean'] = float(pm.missing_smiles.mean())
    out['anomaly'] = anomaly()
    json.dump(out, open(f'{ART}/51_summary.json', 'w'), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != 'train'}, indent=1, default=str))


if __name__ == '__main__':
    main()
