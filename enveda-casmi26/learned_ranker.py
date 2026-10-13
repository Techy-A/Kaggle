"""Learned candidate ranker (LightGBM LambdaRank) over our PubChem-only validation features.

Usage: python learned_ranker.py <work_dir> [model_name]

Reads compare_features.pkl (from compare_models.py): for each of 740 validation molecules (S1 = 250 library
natural products, S2 = 490 enveda-180 structures without public spectra), every candidate's fingerprint
log-likelihood, popularity and natural-product flag, with answers reachable only through PubChem.
Candidates are collapsed to one row per InChIKey14 (the metric's identity). The ranker is scored with
5-fold cross-validation grouped by molecule, against the hand-tuned prior from tune_prior.py.
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

TOP = 300          # candidates per molecule kept: top TOP by fingerprint score plus top 50 by popularity


def candidate_frame(ids, fp, pop, natural, n_total=None):
    """One row per InChIKey14 with the ranker features; kept: top TOP by fit plus top 50 by popularity."""
    df = pd.DataFrame({'id': ids, 'fp': fp, 'pop': pop, 'nat': np.asarray(natural, float)})
    df = df.groupby('id', sort=False).agg(fp=('fp', 'max'), pop=('pop', 'max'), nat=('nat', 'max')).reset_index()
    keep = set(df.nlargest(TOP, 'fp').index) | set(df.nlargest(50, 'pop').index)   # strong fit OR well documented
    df = df.loc[sorted(keep)].sort_values('fp', ascending=False).reset_index(drop=True)
    fp, pop = df.fp.to_numpy(), df['pop'].to_numpy()
    df['fp_z'] = (fp - fp.mean()) / (fp.std() + 1e-9)
    df['fp_gap'] = fp - fp.max()
    df['fp_rank'] = np.arange(len(df)) / max(len(df) - 1, 1)
    df['pop_z'] = (pop - pop.mean()) / (pop.std() + 1e-9)
    df['pop_rank'] = pd.Series(pop).rank(ascending=False, pct=True).to_numpy()
    df['pop_gap'] = pop - pop.max()
    df['n_cands'] = np.log1p(len(ids) if n_total is None else n_total)
    return df


def molecule_frame(f, model, qid):
    df = candidate_frame(f['ids'], f['fp'][model], f['pop'], f['natural'])
    df['label'] = (df.id == f['key']).astype(int)
    df['qid'] = qid
    df['split'] = f['split']
    df['prior'] = df.fp_z + 0.10 * df['pop'] + 1.0 * df.nat      # the hand-tuned baseline
    return df


FEATURES = ['fp_z', 'fp_gap', 'fp_rank', 'pop', 'pop_z', 'pop_rank', 'pop_gap', 'nat', 'n_cands']


def mrr(df, score_col):
    out = {}
    for split, g in df.groupby('split'):
        rr = []
        for _, m in g.groupby('qid'):
            order = m.sort_values(score_col, ascending=False).label.to_numpy()[:25]
            hit = np.flatnonzero(order)
            rr.append(1.0 / (hit[0] + 1) if len(hit) else 0.0)
        out[split] = np.mean(rr)
    return out


def main(work, model='fpnet_v2h'):
    feats = pd.read_pickle(f'{work}/compare_features.pkl')
    df = pd.concat([molecule_frame(f, model, i) for i, f in enumerate(feats)], ignore_index=True)
    df['oof'] = np.nan
    qids = df.qid.to_numpy()
    for tr, va in GroupKFold(n_splits=5).split(df, groups=qids):
        a, b = df.iloc[tr].sort_values('qid'), df.iloc[va]
        ranker = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15,
                                min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                lambdarank_truncation_level=25, verbose=-1)
        ranker.fit(a[FEATURES], a.label, group=a.groupby('qid', sort=True).size().to_numpy())
        df.loc[df.index[va], 'oof'] = ranker.predict(b[FEATURES])
    res = pd.DataFrame({'fingerprint only': mrr(df, 'fp_z'), 'hand prior (0.10 pop, 1 np)': mrr(df, 'prior'),
                        'learned ranker (5-fold OOF)': mrr(df, 'oof')}).T
    res['mean'] = res.mean(axis=1)
    print(f'{df.qid.nunique()} molecules, top-{TOP} by fit + top-50 by popularity, answer kept in {df.groupby("qid").label.max().mean():.0%}')
    print(res.round(3).to_string())
    full = lgb.LGBMRanker(objective='lambdarank', n_estimators=300, learning_rate=0.05, num_leaves=15,
                          min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          lambdarank_truncation_level=25, verbose=-1)
    d = df.sort_values('qid')
    full.fit(d[FEATURES], d.label, group=d.groupby('qid', sort=True).size().to_numpy())
    imp = pd.Series(full.booster_.feature_importance('gain'), index=FEATURES).sort_values(ascending=False)
    print('feature importance (gain):', (imp / imp.sum()).round(3).to_dict())
    full.booster_.save_model(f'{work}/learned_ranker.txt')


if __name__ == '__main__':
    main(*sys.argv[1:3])
