"""Error analysis of a CASMI26 bench run: where does V22 lose points, per simulated class?

For every pseudo-test molecule:
  final_rank   rank of the answer in the final submission (0 = not in the 25)
  in_<source>  rank of the answer in each candidate source (engine 2 top 40, PubChem 25, base tail 60, prefusion)
  loss         'top1' | 'ranked' (in final 25, not first) | 'fusion' (in some source, not in final 25) | 'missing'
  best_tc      best Tanimoto (Morgan r2, 2048) between the answer and any candidate seen anywhere: how close we got
The 'missing' class-3 molecules with a near miss (best_tc >= 0.7) are what a candidate generator could win.

usage: python analyze_bench.py <run_dir>   (run_dir holds bench_answers.csv, submission csvs, *_lists.json,
                                            e004_candidate_inputs.json; any missing file is skipped)
"""
import json
import os
import sys

import numpy as np
import pandas as pd


def main(run):
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    taut = rdMolStandardize.TautomerEnumerator()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    kcache, fcache = {}, {}

    def key(s):
        if s not in kcache:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            if m is None:
                kcache[s] = None
            else:
                Chem.RemoveStereochemistry(m)
                kcache[s] = Chem.MolToInchiKey(taut.Canonicalize(m))[:14]
        return kcache[s]

    def fp(s):
        if s not in fcache:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            fcache[s] = gen.GetFingerprint(m) if m is not None else None
        return fcache[s]

    ans = pd.read_csv(os.path.join(run, 'bench_answers.csv'), dtype=str, keep_default_na=False)
    ans['cls'] = ans.cls.astype(int)
    ans['akey'] = [key(s) for s in ans.normalized_smiles]

    def lists_from_csv(p):
        f = pd.read_csv(p, dtype=str, keep_default_na=False)
        return {m: [x for x in s.split(';') if x] for m, s in zip(f.molecule_id, f.smiles)}

    sources = {}
    for name, fn in [('final', 'submission.csv'), ('prefusion_csv', 'baseline_submission.csv'),
                     ('e2r2_base', 'submission_base_e2r2.csv')]:
        p = os.path.join(run, fn)
        if os.path.exists(p):
            sources[name] = lists_from_csv(p)
    p = os.path.join(run, 'eng_lists.json')
    if os.path.exists(p):
        sources['engine2'] = {m: v['smiles'] for m, v in json.load(open(p)).items()}
    p = os.path.join(run, 'pc_lists.json')
    if os.path.exists(p):
        sources['pubchem'] = {m: v.get('pc', []) for m, v in json.load(open(p)).items()}
    p = os.path.join(run, 'e004_candidate_inputs.json')
    if os.path.exists(p):
        ci = json.load(open(p))
        for part in ['prefusion', 'engine2', 'pubchem', 'base_tail', 'fusion_overflow']:
            if part in ci and isinstance(ci[part], dict):
                sources.setdefault(part if part not in sources else part + '_ci',
                                   {m: (v if isinstance(v, list) else v.get('smiles', [])) for m, v in ci[part].items()})
    assert 'final' in sources, 'need submission.csv'
    print('sources:', {k: len(v) for k, v in sources.items()})

    rows = []
    for r in ans.itertuples():
        row = dict(molecule_id=r.molecule_id, cls=r.cls)
        seen = set()
        for name, L in sources.items():
            lst = L.get(r.molecule_id, [])
            rank = next((i + 1 for i, s in enumerate(lst) if key(s) == r.akey), 0)
            row[f'in_{name}'] = rank
            seen.update(lst[:60])
        fr = row['in_final']
        anywhere = any(row[f'in_{n}'] > 0 for n in sources if n != 'final')
        row['loss'] = 'top1' if fr == 1 else 'ranked' if fr > 1 else 'fusion' if anywhere else 'missing'
        a = fp(r.normalized_smiles)
        tcs = [DataStructs.TanimotoSimilarity(a, f) for f in (fp(s) for s in seen) if f is not None and a is not None]
        row['best_tc'] = max(tcs) if tcs else 0.0
        row['rr'] = 1.0 / fr if fr else 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run, 'analysis_per_molecule.csv'), index=False)

    print('\n== MRR@25 and loss buckets per class (share of molecules)')
    g = df.groupby('cls')
    tab = pd.DataFrame({'n': g.size(), 'MRR': g.rr.mean().round(3)})
    for b in ['top1', 'ranked', 'fusion', 'missing']:
        tab[b] = g.loss.apply(lambda s, b=b: round((s == b).mean(), 2))
    print(tab.to_string())
    # points available per bucket: what a perfect fix of that bucket would add to overall MRR
    n = len(df)
    print('\n== MRR points left on the table (overall MRR gain if every molecule in the bucket became rank 1)')
    for b in ['ranked', 'fusion', 'missing']:
        sub = df[df.loss == b]
        print(f'  {b:8s} {len(sub):4d} molecules  +{(1 - sub.rr).sum() / n:.3f}')
    miss3 = df[(df.cls == 3) & (df.loss == 'missing')]
    print('\n== class-3 misses: how close did the nearest candidate get? (Tanimoto, Morgan r2)')
    print(pd.cut(miss3.best_tc, [0, 0.4, 0.55, 0.7, 0.85, 1.0], include_lowest=True).value_counts().sort_index().to_string())
    print(f'  near misses (>= 0.7): {(miss3.best_tc >= 0.7).sum()} of {len(miss3)}')
    for c in sorted(df.cls.unique()):
        sub = df[df.cls == c]
        print(f'\nclass {c}: source recall (answer anywhere in the list):',
              {s: int((sub[f'in_{s}'] > 0).sum()) for s in sources})


if __name__ == '__main__':
    main(sys.argv[1])
