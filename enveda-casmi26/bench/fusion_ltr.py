"""Learned fusion: can a small learning-to-rank model beat V22's hand-written E004 fusion on the bench? (CPU)

V22 orders its final 25 with fixed rules (PubChem slots, promotion, overflow). The bench saved the five lists that
fusion stage sees, for every pseudo-test molecule (e004_candidate_inputs.json): prefusion (25), engine 2 (ranked keys),
PubChem (ranked keys + fingerprint-fit scores), the engine-1 base tail (60) and the overflow (15).

Honest molecules only: np-bench class 1 (the structure keeps its other library spectra) and unseen-bench classes 2-3.
Per molecule every candidate of the union gets: its rank in each list (0 = absent), the number of lists it is in, the
PubChem z-score and pc_fz, top-PubChem popularity S/top_pop, and the candidate's rank in V22's own final list. A
LightGBM lambdarank model (5 molecule-grouped folds, class-stratified) ranks the union; the out-of-fold top 25 is
scored with MRR@25 on the tautomer-canonical stereo-free InChIKey14 and compared with V22's final list per class, plus
a paired bootstrap over molecules at the test's class mix.

Usage: python fusion_ltr.py [--mix 0.28 0.20 0.52] [--seeds 3]
"""
import argparse
import json
import os
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RUNS = {'np': ('/home/user/benchdata/run2', [1]), 'unseen': ('/home/user/benchdata/unseen1', [2, 3])}
LISTS = ['prefusion', 'engine2', 'pubchem', 'base_tail', 'fusion_overflow', 'final']


def metric_key(smi):
    """Tautomer-canonical, stereo-free InChIKey first block (the competition's identity)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        m = rdMolStandardize.TautomerEnumerator().Canonicalize(m)
        Chem.RemoveStereochemistry(m)
        return Chem.MolToInchiKey(m)[:14] or None
    except Exception:
        return None


def load(run_dir, classes):
    ans = pd.read_csv(os.path.join(run_dir, 'bench_answers.csv'))
    ans = ans[ans.cls.isin(classes)].set_index('molecule_id')
    e = json.load(open(os.path.join(run_dir, 'e004_candidate_inputs.json')))
    eng = json.load(open(os.path.join(run_dir, 'eng_lists.json')))
    sub = pd.read_csv(os.path.join(run_dir, 'submission.csv')).set_index('molecule_id').smiles
    known = {}   # smiles -> key from the lists that carry keys (engine 1, engine 2, PubChem)
    for src in (eng, e['engine2']):
        for v in src.values():
            known.update(zip(v['smiles'], v['keys']))
    for v in e['pubchem'].values():
        known.update(zip(v['pc'], v['pc_keys']))
    mols = {}
    for mid in ans.index:
        lists = {'prefusion': e['prefusion'].get(mid, []), 'engine2': e['engine2'].get(mid, {}).get('smiles', []),
                 'pubchem': e['pubchem'].get(mid, {}).get('pc', []), 'base_tail': e['base_tail'].get(mid, []),
                 'fusion_overflow': e['fusion_overflow'].get(mid, []),
                 'final': str(sub.get(mid, '')).split(';')[:25] if mid in sub.index else []}
        pc = e['pubchem'].get(mid, {})
        mols[mid] = dict(cls=int(ans.loc[mid, 'cls']), answer=ans.loc[mid, 'metric_key'], lists=lists,
                         pc_fz=list(pc.get('pc_fz', [])), S=float(pc.get('S') or 0), top_pop=float(pc.get('top_pop') or 0))
    return mols, known


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mix', type=float, nargs=3, default=[0.28, 0.20, 0.52])
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()

    mols, known = {}, {}
    for name, (d, cl) in RUNS.items():
        m, k = load(d, cl)
        mols.update(m); known.update(k)
    todo = sorted({s for m in mols.values() for l in m['lists'].values() for s in l} - set(known))
    print(f'{len(mols)} molecules; {len(todo)} SMILES need a metric key (RDKit)', flush=True)
    with Pool(4) as p:
        known.update(zip(todo, p.map(metric_key, todo, chunksize=200)))

    rows = []
    for mid, m in mols.items():
        ranks = {}
        for name, lst in m['lists'].items():
            for r, s in enumerate(lst, 1):
                k = known.get(s)
                if k is not None:
                    ranks.setdefault(k, {}).setdefault(name, r)   # first (best) rank of the key in each list
        pcz = dict(zip([known.get(s) for s in m['lists']['pubchem']], m['pc_fz']))
        for k, rk in ranks.items():
            f = {f'r_{n}': rk.get(n, 0) for n in LISTS}
            f.update(n_lists=sum(n in rk for n in LISTS if n != 'final'), pc_fz=pcz.get(k, np.nan), S=m['S'],
                     top_pop=m['top_pop'], cls_hint=0)
            rows.append(dict(mid=mid, key=k, cls=m['cls'], y=int(k == m['answer']), **f))
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c.startswith('r_') and c != 'r_final'] + ['n_lists', 'pc_fz', 'S', 'top_pop']
    print(f'candidates {len(df)}; answer in the union for {df.groupby("mid").y.max().sum()} of {len(mols)} molecules',
          flush=True)

    def mrr_of(order_scores):
        out = {}
        for mid, g in df.assign(s=order_scores).groupby('mid'):
            top = g.sort_values('s', ascending=False).head(25)
            hit = np.flatnonzero(top.y.values)
            out[mid] = 1 / (hit[0] + 1) if len(hit) else 0.0
        return pd.Series(out)

    # V22 baseline: its own final order (absent from the final 25 = 0)
    base = mrr_of(np.where(df.r_final > 0, -df.r_final, -1e9))
    mids = np.array(sorted(mols))
    cls = pd.Series({k: v['cls'] for k, v in mols.items()})
    oof = np.zeros(len(df))
    for seed in range(args.seeds):
        for tr_m, va_m in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(mids, cls[mids], mids):
            tr, va = df.mid.isin(mids[tr_m]).values, df.mid.isin(mids[va_m]).values
            dtr = df[tr].sort_values('mid')
            model = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15,
                                   min_child_samples=20, random_state=seed, verbose=-1,
                                   lambdarank_truncation_level=25)
            model.fit(dtr[feats], dtr.y, group=dtr.groupby('mid', sort=True).size().values)
            oof[va] += model.predict(df.loc[va, feats]) / args.seeds
    ltr = mrr_of(oof)

    print('\nMRR@25 per class (V22 final vs learned fusion, out-of-fold):')
    w = dict(zip((1, 2, 3), args.mix))
    for c in (1, 2, 3):
        i = cls[cls == c].index
        print(f'  class {c} (n={len(i)}): V22 {base[i].mean():.4f} | learned {ltr[i].mean():.4f} | '
              f'delta {ltr[i].mean() - base[i].mean():+.4f} (better {int((ltr[i] > base[i]).sum())}, worse {int((ltr[i] < base[i]).sum())})')
    mix = lambda s: sum(w[c] * s[cls[cls == c].index].mean() for c in (1, 2, 3))
    rng = np.random.default_rng(0)
    groups = {c: cls[cls == c].index.values for c in (1, 2, 3)}
    wins = 0
    for _ in range(2000):   # paired bootstrap within each class, at the test's class mix
        smp = {c: rng.choice(groups[c], len(groups[c])) for c in (1, 2, 3)}
        wins += sum(w[c] * (ltr[smp[c]].mean() - base[smp[c]].mean()) for c in (1, 2, 3)) > 0
    print(f'\nat the test mix {args.mix}: V22 {mix(base):.4f} | learned {mix(ltr):.4f} | delta {mix(ltr) - mix(base):+.4f} '
          f'| P(learned better) {wins / 2000:.2f}')


if __name__ == '__main__':
    main()
