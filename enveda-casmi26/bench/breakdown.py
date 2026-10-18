"""Per-class loss breakdown of V22 on the bench, mixing the honest slices of two runs (CPU, existing results only).

Class 1 comes from the np bench (run2: query spectra hidden, the structure keeps its other library spectra, which is
what class 1 is). Classes 2 and 3 come from the leak-free unseen bench (unseen1: structures no model has seen); the np
run's class 2/3 rows are left out because the models had memorised those structures.
For every class: MRR@25, and where each molecule's points go, using analyze_bench.py's loss buckets:
  top1     answer ranked first;
  ranked   in the final 25 but not first (points lost = 1 - 1/rank);
  fusion   in some candidate source but pushed out of the final 25 by the fusion;
  missing  never a candidate anywhere (best_tc = closest candidate's Tanimoto to the answer).
'Share of all lost points' weights each class by the test's class mix (--mix, community fit 28/20/52).
usage: python breakdown.py [--mix 0.28 0.20 0.52]
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--mix', type=float, nargs=3, default=[0.28, 0.20, 0.52])
args = ap.parse_args()
np_run = pd.read_csv('results/run2_analysis_per_molecule.csv')
unseen = pd.read_csv('results/unseen1_analysis_per_molecule.csv')
mol = pd.concat([np_run[np_run.cls == 1].assign(src='np'), unseen.assign(src='unseen')], ignore_index=True)
mol['lost'] = 1 - mol.rr

rows = []
for c, share in zip((1, 2, 3), args.mix):
    m = mol[mol.cls == c]
    lost = m.groupby('loss').lost.sum() / len(m)   # points lost per molecule of this class, by bucket
    rows.append(dict(cls=c, n=len(m), src=m.src.iloc[0], mrr=m.rr.mean(), test_share=share,
                     **{f'{b}_n': int((m.loss == b).sum()) for b in ('top1', 'ranked', 'fusion', 'missing')},
                     **{f'lost_{b}': share * lost.get(b, 0.0) for b in ('ranked', 'fusion', 'missing')}))
t = pd.DataFrame(rows)
total = t[[c for c in t if c.startswith('lost_')]].to_numpy().sum()
print(f"test-mix MRR estimate {sum(t.mrr * t.test_share):.3f} (class mix {args.mix})\n")
for r in t.itertuples():
    print(f"class {r.cls} ({r.src}, n={r.n}): MRR {r.mrr:.3f} | top1 {r.top1_n}, ranked 2-25 {r.ranked_n}, "
          f"fusion-dropped {r.fusion_n}, missing {r.missing_n}")
print('\nshare of all lost points (weighted by the test mix):')
for r in t.itertuples():
    print(f"  class {r.cls}: ranked 2-25 {r.lost_ranked / total:5.1%} | fusion-dropped {r.lost_fusion / total:5.1%} | "
          f"never a candidate {r.lost_missing / total:5.1%}")
miss3 = mol[(mol.cls == 3) & (mol.loss == 'missing')]
print(f"\nclass-3 never-a-candidate: {len(miss3)}, of which a near miss (best Tanimoto >= 0.7): {(miss3.best_tc >= 0.7).sum()}")
r1 = mol[(mol.cls == 1) & (mol.loss == 'ranked')]
print(f"class-1 ranked 2-25: {len(r1)}, of which rank 2-3: {(r1.rr >= 1 / 3).sum()} (rank 2 -> 1 is worth +0.5 each)")
