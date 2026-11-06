"""63: rank-1-locked tail reranker on the bench (ranks 2-25 only), grouped cross-validation.

What: on the bench477 final lists (250 enveda natural products, the merged477 pipeline), learn a LightGBM lambdarank
over ranks 2-25 from per-candidate evidence the pipeline already exports (neural cos/fp/score/std, ICE, GLACIER, library
max, engine and PubChem ranks, the list position, mass error from the instrument centre). Rank 1 is never moved. 5-fold
molecule-grouped CV, 5 seeds; out-of-fold lists are scored with MRR@25 against the unchanged lists, paired bootstrap.
Why: with rank 1 locked the headroom is +0.056 (every ranks-2-25 hit moved to rank 2); the earlier learned fusion was
rejected because it moved rank 1 ("trust PubChem"). This asks whether evidence the pipeline already has reorders
ranks 2-25 better than its fixed rule. Gate: P(better) >= 0.9, no class-1 losses.
Output: artifacts/63_summary.json. CPU, about a minute.
Usage: python enveda-casmi26/63_tail_reranker.py [feature prefixes to drop, e.g. neural d_neural]
"""
import json, os
import numpy as np, pandas as pd, lightgbm as lgb
from rdkit import Chem, RDLogger
from rdkit.Chem.Descriptors import ExactMolWt
RDLogger.DisableLog('rdApp.*')
B = '/home/user/work/out_bench477/'
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
ADD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
       '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M-H2O+H]+': 1.007276 - 18.010565}
FEATS = ['pos', 'library_max', 'ice', 'gl', 'proxy_gl', 'main_rank', 'prefusion_rank', 'engine2_rank', 'pubchem_rank',
         'neural_cos', 'neural_fp', 'neural_score', 'neural_std', 'ppm_abs', 'd_neural_vs_top', 'd_lib_vs_top']


import sys
DROP=[f for f in sys.argv[1:]]
def main():
    global FEATS
    FEATS=[f for f in FEATS if not any(f.startswith(x) or f==x for x in DROP)]
    print('features',FEATS)
    sub = pd.read_csv(B + 'submission.csv', dtype=str, keep_default_na=False)
    ct = pd.read_parquet(B + 'bench_candidate_truth.parquet'); truth = ct[ct.is_answer].groupby('molecule_id').smiles.apply(set)
    cls = pd.read_csv(B + 'bench_answers.csv').set_index('molecule_id').cls
    ev = pd.read_parquet(B + 'neural_candidate_features.parquet').drop_duplicates(['molecule_id', 'smiles']).set_index(['molecule_id', 'smiles'])
    te = pd.read_parquet(B + 'bench_comp/test.parquet', columns=['molecule_id', 'adduct', 'precursor_mz'])
    te = te[te.adduct.isin(ADD)]; te['nm'] = te.precursor_mz - te.adduct.map(ADD); nm = te.groupby('molecule_id').nm.median()
    mw = {}
    rows = []
    for m, s in zip(sub.molecule_id, sub.smiles):
        l = s.split(';')[:25]
        for p, c in enumerate(l, 1):
            if c not in mw:
                x = Chem.MolFromSmiles(c); mw[c] = ExactMolWt(x) if x else np.nan
            r = dict(mol=m, smiles=c, pos=p, ans=int(c in truth.get(m, set())))
            r.update(ev.loc[(m, c)].to_dict() if (m, c) in ev.index else {})
            r['ppm'] = (mw[c] - nm[m]) / nm[m] * 1e6 if m in nm.index else np.nan
            rows.append(r)
    d = pd.DataFrame(rows)
    centre = d[d.pos == 1].ppm.median(); d['ppm_abs'] = (d.ppm - centre).abs()
    top = d[d.pos == 1].set_index('mol')
    d['d_neural_vs_top'] = d.neural_score - d.mol.map(top.neural_score); d['d_lib_vs_top'] = d.library_max - d.mol.map(top.library_max)
    for f in FEATS: d[f] = pd.to_numeric(d[f], errors='coerce')
    mols = np.array(sorted(d.mol.unique())); rng0 = np.random.default_rng(0)

    def rr(order_df):
        out = {}
        for m, g in order_df.groupby('mol', sort=False):
            a = g.ans.values; k = np.argmax(a) if a.any() else -1
            out[m] = 1 / (k + 1) if k >= 0 else 0.0
        return pd.Series(out)

    base = rr(d.sort_values(['mol', 'pos']))
    deltas = []
    for seed in range(5):
        perm = np.random.default_rng(seed).permutation(mols); folds = np.array_split(perm, 5)
        new = []
        for k in range(5):
            tr = d[~d.mol.isin(folds[k]) & (d.pos > 1)]; va = d[d.mol.isin(folds[k])]
            tr = tr.sort_values(['mol', 'pos']); grp = tr.groupby('mol', sort=False).size().values
            mdl = lgb.LGBMRanker(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=20, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
            mdl.fit(tr[FEATS], tr.ans, group=grp)
            va = va.copy(); va['s'] = mdl.predict(va[FEATS]); va.loc[va.pos == 1, 's'] = np.inf      # rank 1 locked
            new.append(va.sort_values(['mol', 's'], ascending=[True, False]))
        o = rr(pd.concat(new)); deltas.append((o - base).reindex(mols).values)
    D = np.mean(deltas, axis=0)
    bs = np.array([D[rng0.integers(0, len(D), len(D))].mean() for _ in range(5000)])
    c = cls.reindex(mols).values
    out = dict(n=len(mols), base_mrr=float(base.mean()), delta=float(D.mean()), p_better=float((bs > 0).mean()),
               ci90=[float(np.quantile(bs, .05)), float(np.quantile(bs, .95))], up=int((D > 1e-9).sum()), down=int((D < -1e-9).sum()),
               by_class={int(k): dict(delta=float(D[c == k].mean()), down=int((D[c == k] < -1e-9).sum())) for k in (1, 2, 3)},
               per_seed=[float(x.mean()) for x in deltas], headroom_rank2=0.0562, centre_ppm=float(centre))
    os.makedirs(ART, exist_ok=True); json.dump(out, open(f'{ART}/63_summary' + ('_drop_' + '_'.join(DROP) if DROP else '') + '.json', 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
