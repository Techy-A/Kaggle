"""Leak-free bench calibration (pre-registered 7 Nov, CASMI26.md): replay known LB moves on the unseen bench lists.

What: on casmi26-bench477-unseen outputs (0.477-era pipeline on 320 Enveda-180 timsTOF structures absent from
train.parquet; classes 2/3 = 128/192), compute paired MRR@25 deltas with bootstrap P(better) for
  (1) merged477 vs baseline0465                 LB +0.012  -> must be > 0
  (2) tail reranker (63_final_model.txt) on merged477  LB -0.021  -> must be <= 0
  (3) xspec boost + mass prior on merged477      LB +0.001  -> must be >= -0.001
Pass = all three signs right. Output: artifacts/68_unseen_calibration.json.
Usage: python 68_unseen_calibration.py [unseen_out_dir] [test_parquet]   (8 Nov: test path optional, default unchanged)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE / 'overlay'))
import tail_rr, xspec_boost
U = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/user/work/out_unseen')
test = pd.read_parquet(sys.argv[2] if len(sys.argv) > 2 else '/home/user/benchdata/ds_unseen/unseen_test.parquet')
ans = pd.read_csv(U / 'bench_answers.csv', dtype=str, keep_default_na=False)
keys = {m: {a, b} - {''} for m, a, b in zip(ans.molecule_id, ans.inchikey14, ans.metric_key)}
ev = pd.read_parquet(U / 'frozen_candidate_evidence.parquet')
kc = {}
def k14(s):
    if s not in kc:
        x = Chem.MolFromSmiles(s); kc[s] = Chem.MolToInchiKey(x)[:14] if x is not None else None
    return kc[s]
def rr(sub):
    L = dict(zip(sub.molecule_id, sub.smiles.str.split(';')))
    out = []
    for m in ans.molecule_id:
        ks = [k14(x) for x in L[m][:25]]
        out.append(next((1 / (i + 1) for i, k in enumerate(ks) if k in keys[m]), 0.0))
    return np.array(out)
def compare(a, b, name, need):
    d = rr(b) - rr(a); rng = np.random.default_rng(0); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    r = dict(delta=round(float(d.mean()), 5), p_better=round(float((bs > 0).mean()), 3), up=int((d > 0).sum()), down=int((d < 0).sum()),
             rank1_changed=int(sum(x.split(';')[0] != y.split(';')[0] for x, y in zip(a.smiles, b.smiles))), need=need)
    print(name, r, flush=True); return r
base = pd.read_csv(U / 'submission_baseline0465.csv', dtype=str, keep_default_na=False)
merged = pd.read_csv(U / 'variants/submission_merged477.csv', dtype=str, keep_default_na=False)
print('own scoring check: baseline', round(rr(base).mean(), 4), 'merged477', round(rr(merged).mean(), 4), '(bench_results: 0.1129 / 0.1150)')
res = {'1_merged_vs_baseline': compare(base, merged, '(1) merged477 vs baseline0465', '> 0')}
tr, st = tail_rr.rerank(merged, ev, test, (HERE / 'artifacts/63_final_model.txt').read_text())
res['2_tail_reranker'] = compare(merged, tr, '(2) tail reranker', '<= 0'); res['2_tail_reranker']['stats'] = st
xb, sx = xspec_boost.boost(merged, ev, test)
xm, sm = xspec_boost.mass_prior(xb, test)
res['3_xspec_mass'] = compare(merged, xm, '(3) xspec + mass prior', '>= -0.001'); res['3_xspec_mass']['stats'] = dict(xspec=sx, mass=sm)
ok = [res['1_merged_vs_baseline']['delta'] > 0, res['2_tail_reranker']['delta'] <= 0, res['3_xspec_mass']['delta'] >= -0.001]
res['signs_right'] = ok; res['PASS'] = all(ok)
print('signs right:', ok, '-> PASS' if all(ok) else '-> FAIL')
(HERE / ('artifacts/68_unseen_calibration.json' if len(sys.argv) <= 2 else f'artifacts/68_calibration_{U.name}.json')).write_text(json.dumps(res, indent=1, default=str))
