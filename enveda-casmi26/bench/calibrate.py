"""Bench calibration: can any honest scoring of the local benches reproduce the known public-LB ordering?

What: re-scores every per-molecule bench output we have (submission-format csvs, or the bench's own per-molecule
reciprocal ranks where the csv was overwritten), pairs each variant with its own base, and asks
  1. per bench and variant: MRR@25 overall / per class, hit@25, top-1, paired stratified bootstrap vs the base
     (delta CI, P(better)), next to the known public LB delta;
  2. whether re-weighting the classes (a simplex grid of test mixes) makes every bench delta take the LB sign
     (rel100, rel250, gen20 negative; E2R3 / depth7 ties not called a gain), and how robust that is (bootstrap);
  3. for gen20, which molecules it changes, how it displaces candidates that V22 already had (the proxy for correct
     library / COCONUT answers), and how likely LB -0.010 is on ~124 public molecules given each bench's deltas.
Why: a bench is trusted only if it reproduces known LB results before it is used to judge new ideas (gen20 lost
0.010 on the LB while the np bench said +0.033).
Benches (all in /home/user/benchdata):
  np      run2 (V22) + run3 (V22 rerun) + np_g20 (gen20 main pass and CPU-tail replays on the gen20 base), 250 molecules
  unseen  unseen1 (V22) + unseen_g20 (gen20 and replays), 320 leak-free molecules, classes 2/3 only
  honest  class 1 from np, classes 2/3 from unseen (breakdown.py's slices)
  engine1 gensweep engine-1 lists (base vs B20), 130 class-3 + 40 class-1/2 np molecules
Variants replayed on the gen20 base are compared with the gen20 replay (additive assumption for the LB).
Output: results/calibration.md (tables, diagnosis, reading); results/calibration_gen20_molecules.csv (per-molecule
gen20 vs V22 on the np bench); results/calibration_keys.pkl (SMILES -> metric key cache, reused on reruns).
usage: python -I calibrate.py [--boot 4000] [--step 0.02]
"""
import argparse
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

BD = '/home/user/benchdata'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'calibration.md')
CACHE = os.path.join(HERE, 'results', 'calibration_keys.pkl')
LB_BASE = 0.451
LB = {'gen20': 0.441, 'rel100': 0.445, 'rel250': 0.445, 'e2r3': 0.450, 'depth7': 0.451, 'no_e2r2': None}
DECISIVE = {'gen20', 'rel100', 'rel250'}        # |LB delta| >= 0.006: the bench must call these losses
TIE_TOL = 0.004                                 # ties (e2r3, depth7): the bench must not call a gain above noise
COMMUNITY_MIX = (0.28, 0.20, 0.52)
N_PUBLIC, RUN_SD = 124, 0.004


READING = ['## Reading (19 Oct run; re-check the numbers above if the inputs change)', '',
    '- **No bench, and no class mix, reproduces the LB.** np: rel250 is a gain at every mix (its class 2 is memorised) and '
    'gen20 is a loss only when class 3 is < ~6% of the test. unseen: rel100 and gen20 are gains at every mix. honest '
    '(np class 1 + unseen class 2/3): all four signs come out right only for mixes with >= 56% class 1 and <= 30% class 3, '
    'none of which is level-plausible (V22 would score >= 0.54 there, not 0.451), and even the best of those mixes gets all '
    'signs right in only 43% of bootstrap draws (14% among level-plausible mixes).',
    '- **gen20 vs the honest bench:** +0.004 [-0.001, +0.011] at 28/20/52; LB -0.010 is a ~1% tail of the 124-molecule '
    'simulation. Either the test has a property the bench lacks, or the LB draw was unlucky (1 molecule at rank 1 = 0.008).',
    '- **Mechanism seen on the bench:** gen20 hardly crowds (no new candidate ever takes rank 1 on class 1/2) but it '
    're-scores the candidates V22 already had: engine 1 features are list-relative (rank_norm, z, gap-to-max, softmax, '
    'formula_share; fragment scoring capped at frag_max_cand=600), so adding up to 350 generated candidates reorders the '
    'top 5 in ~50-60% of molecules and moved the correct class-1 answer off rank 1 in 2/70 engine-1 lists (1/70 after fusion).',
    '- **Why the bench is blind to it:** bench answers have ~180 library spectra each vs a median of 6 for training '
    'structures, so class-1/2 answers win by margins a re-scoring rarely overturns, while class-3 gains are leak-inflated '
    '(engine-1 models saw those structures). The test needs a correct-answer flip rate of ~4-11% (table above) for -0.010.',
    '- **Minimal bench fix:** a "thin class 1" slice: class-1 molecules keep only 1-5 of their other library spectra '
    '(match the training n_spec distribution), scored with the engine-1 lists (CPU, like gensweep). Gate generation and '
    'list-composition changes on the correct-answer rank-1 loss rate on that slice; it predicts gen20 negative if the rate '
    'reaches ~4%+. Unverified until that slice is run.', '']


def _key(s):
    """Metric identity: tautomer-canonical InChIKey first block (casmi.chem.score_key)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog('rdApp.*')
    global _T
    if '_T' not in globals():
        _T = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToInchiKey(_T.Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None


def keys_for(smiles):
    cache = pickle.load(open(CACHE, 'rb')) if os.path.exists(CACHE) else {}
    todo = sorted({s for s in smiles if isinstance(s, str) and s and s not in cache})
    if todo:
        print('canonicalising', len(todo), 'SMILES', flush=True)
        with Pool(4) as p:
            cache.update(zip(todo, p.map(_key, todo, chunksize=200)))
        pickle.dump(cache, open(CACHE, 'wb'))
    return cache


def read_lists(path):
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {m: [x for x in s.split(';') if x][:25] for m, s in zip(f.molecule_id, f.smiles)}


def rr_of(lists, akey, K):
    return pd.Series({m: next((1 / (i + 1) for i, s in enumerate(lists.get(m, [])) if K.get(s) == k), 0.0)
                      for m, k in akey.items()})


def jrr(path, *ks):
    d = json.load(open(path))
    for k in ks:
        d = d[k]
    return pd.Series(d['rr'], dtype=float)


# ---------------------------------------------------------------- data
def load():
    np_ans = pd.read_csv(f'{BD}/run2/bench_answers.csv', dtype=str, keep_default_na=False)
    un_ans = pd.read_csv(f'{BD}/unseen1/bench_answers.csv', dtype=str, keep_default_na=False)
    files = {'v22': f'{BD}/run2/submission.csv', 'gen20': f'{BD}/np_g20/submission_e2r2_final.csv',
             **{f'gen20+{v}': f'{BD}/np_g20/bench_sub_{v}.csv' for v in ('rel100', 'rel250', 'e2r3', 'no_e2r2')}}
    lists = {k: read_lists(p) for k, p in files.items()}
    ci = {r: json.load(open(f'{BD}/{r}/e004_candidate_inputs.json')) for r in ('run2', 'np_g20')}
    src = {}
    for r in ci:   # every candidate source V22 / gen20 had before the final list
        src[r] = {m: {'prefusion': ci[r]['prefusion'][m], 'base_tail': ci[r]['base_tail'][m],
                      'engine2': ci[r]['engine2'][m]['smiles'][:40], 'pubchem': ci[r]['pubchem'][m]['pc'][:25]}
                  for m in ci[r]['prefusion']}
    allsmi = list(np_ans.normalized_smiles) + [s for L in lists.values() for v in L.values() for s in v]
    allsmi += [s for r in src.values() for d in r.values() for v in d.values() for s in v]
    K = keys_for(allsmi)
    akey = dict(zip(np_ans.molecule_id, [K[s] for s in np_ans.normalized_smiles]))
    bad = sum(K[s] != mk for s, mk in zip(np_ans.normalized_smiles, np_ans.metric_key))
    rr = {k: rr_of(L, akey, K) for k, L in lists.items()}
    # cross-check own scoring against the bench's per-molecule rr, then fill what only the json has
    chk = {'v22': jrr(f'{BD}/run2/bench_results.json', 'main', 'submission.csv'),
           'gen20': jrr(f'{BD}/np_g20/bench_results.json', 'replay_v22'),
           **{f'gen20+{v}': jrr(f'{BD}/np_g20/bench_results.json', v) for v in ('rel100', 'rel250', 'e2r3', 'no_e2r2')}}
    check = {k: int((rr[k] - chk[k].reindex(rr[k].index)).abs().gt(1e-9).sum()) for k in rr}
    rr['v22_run3'] = jrr(f'{BD}/run3/bench_results.json', 'main', 'submission.csv')   # csv was overwritten
    npb = dict(cls=np_ans.set_index('molecule_id').cls.astype(int), rr=rr)
    un = {'v22': jrr(f'{BD}/unseen1/bench_results.json', 'main', 'submission.csv'),      # csv was overwritten
          'gen20': jrr(f'{BD}/unseen_g20/bench_json.json', 'main', 'submission.csv'),
          **{f'gen20+{v}': jrr(f'{BD}/unseen_g20/bench_json.json', v) for v in ('rel100', 'rel250', 'e2r3', 'no_e2r2')}}
    unb = dict(cls=un_ans.set_index('molecule_id').cls.astype(int), rr=un)
    gs = {s: json.load(open(f'{BD}/gensweep/out/gensweep_{s}.json')) for s in ('base', 'B20')}
    e1 = {s: pd.Series({m: (1 / v[0] if 0 < v[0] <= 25 else 0.0) for m, v in d.items()}) for s, d in gs.items()}
    e1b = dict(cls=npb['cls'].reindex(e1['base'].index), rr={'v22': e1['base'], 'gen20': e1['B20']})
    libmax = pd.Series({m: v[2] for m, v in gs['base'].items()})
    return dict(np=npb, unseen=unb, engine1=e1b, K=K, akey=akey, lists=lists, src=src, check=check, bad_akey=bad,
                libmax=libmax, np_ans=np_ans)


# ---------------------------------------------------------------- comparisons
def pair(bench, base, var):
    """{class: (base rr array, variant rr array)} on the molecules both have."""
    cls, rr = bench['cls'], bench['rr']
    out = {}
    for c in (1, 2, 3):
        ms = cls.index[cls == c]
        if len(ms):
            out[c] = (rr[base].reindex(ms).to_numpy(float), rr[var].reindex(ms).to_numpy(float))
    return out


def comparisons(D):
    C = {}
    for b in ('np', 'unseen'):
        C[('gen20', b)] = pair(D[b], 'v22', 'gen20')
        for v in ('rel100', 'rel250', 'e2r3', 'no_e2r2'):
            C[(v, b)] = pair(D[b], 'gen20', f'gen20+{v}')
    C[('gen20', 'engine1')] = pair(D['engine1'], 'v22', 'gen20')
    C[('v22 rerun', 'np')] = pair(D['np'], 'v22', 'v22_run3')
    for v in ('gen20', 'rel100', 'rel250', 'e2r3', 'no_e2r2'):     # honest: class 1 from np, classes 2/3 unseen
        C[(v, 'honest')] = {1: C[(v, 'np')][1], 2: C[(v, 'unseen')][2], 3: C[(v, 'unseen')][3]}
    return C


def boot(comp, B, seed=0):
    """B x 3 matrix of per-class mean deltas (stratified paired bootstrap); NaN for a missing class."""
    rng = np.random.default_rng(seed)
    M = np.full((B, 3), np.nan)
    for c, (a, b) in comp.items():
        d = b - a
        M[:, c - 1] = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return M


def at_mix(comp, w):
    w = np.asarray(w, float)
    have = np.array([c in comp for c in (1, 2, 3)])
    w = np.where(have, w, 0.0)
    return w / w.sum() if w.sum() > 0 else w


def summary_rows(C, B):
    rows = []
    for (v, b), comp in C.items():
        M = boot(comp, B)
        n = sum(len(a) for a, _ in comp.values())
        w_nat = at_mix(comp, [len(comp[c][0]) if c in comp else 0 for c in (1, 2, 3)])
        w_com = at_mix(comp, COMMUNITY_MIX)
        r = dict(variant=v, bench=b, n=n)
        for tag, w in (('nat', w_nat), ('mix', w_com)):
            base = sum(w[c - 1] * comp[c][0].mean() for c in comp)
            var = sum(w[c - 1] * comp[c][1].mean() for c in comp)
            dd = np.nansum(M * w, 1)
            r.update({f'{tag}_base': base, f'{tag}_var': var, f'{tag}_d': var - base,
                      f'{tag}_lo': np.percentile(dd, 2.5), f'{tag}_hi': np.percentile(dd, 97.5),
                      f'{tag}_p': (dd > 0).mean() + 0.5 * (dd == 0).mean()})
        for c in (1, 2, 3):
            r[f'c{c}'] = comp[c][1].mean() - comp[c][0].mean() if c in comp else np.nan
            r[f'c{c}_var'] = comp[c][1].mean() if c in comp else np.nan
        a_all = np.concatenate([comp[c][0] for c in comp]); b_all = np.concatenate([comp[c][1] for c in comp])
        r.update(top1=(b_all == 1).sum(), top1_base=(a_all == 1).sum(), hit=(b_all > 0).sum(), hit_base=(a_all > 0).sum(),
                 changed=int((a_all != b_all).sum()))
        base_lb = LB_BASE if v != 'v22 rerun' else None
        lbv = LB.get(v)
        r['lb_d'] = (lbv - base_lb) if (lbv is not None and base_lb) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def simplex(step):
    k = int(round(1 / step))
    return np.array([(i / k, j / k, (k - i - j) / k) for i in range(k + 1) for j in range(k + 1 - i)])


def sign_ok(d, v):
    return d < 0 if v in DECISIVE else d < TIE_TOL


def level(D, w):
    """V22's MRR at mix w on the honest slices (low: class 2/3 PubChem-only, harder than the test) and on the np
    bench (high: class 2/3 memorised by the models). A mix is level-plausible if the LB's 0.451 lies between them."""
    nc, uc = D['np']['cls'], D['unseen']['cls']
    lo = [D['np']['rr']['v22'][nc == 1].mean(), D['unseen']['rr']['v22'][uc == 2].mean(), D['unseen']['rr']['v22'][uc == 3].mean()]
    hi = [D['np']['rr']['v22'][nc == c].mean() for c in (1, 2, 3)]
    return np.dot(w, lo), np.dot(w, hi)


def mix_search(C, bench, step, B):
    vs = [v for v in ('gen20', 'rel100', 'rel250', 'e2r3') if (v, bench) in C]
    W = simplex(step)
    Ms = {v: boot(C[(v, bench)], B, seed=1) for v in vs}
    pts = {v: np.array([C[(v, bench)][c][1].mean() - C[(v, bench)][c][0].mean() if c in C[(v, bench)] else np.nan
                        for c in (1, 2, 3)]) for v in vs}
    have = ~np.isnan(pts[vs[0]])
    W = W[(W[:, ~have] == 0).all(1)] if (~have).any() else W
    rows = []
    for w in W:
        ok = {v: sign_ok(np.nansum(pts[v] * w), v) for v in vs}
        allb = np.ones(B, bool)
        for v in vs:
            allb &= sign_ok(np.nansum(Ms[v] * w, 1), v)
        rows.append(dict(w1=w[0], w2=w[1], w3=w[2], n_ok=sum(ok.values()), all_ok=all(ok.values()),
                         p_all=allb.mean(), **{f'ok_{v}': ok[v] for v in vs},
                         **{f'd_{v}': np.nansum(pts[v] * w) for v in vs}))
    return pd.DataFrame(rows), vs


def lb_sim(comp, w, lb_d, n=N_PUBLIC, sims=20000, seed=2):
    """P(a public split of n molecules at mix w, plus run noise, shows a delta <= the LB delta)."""
    rng = np.random.default_rng(seed)
    w = at_mix(comp, w)
    counts = rng.multinomial(n, w, size=sims)
    tot = np.zeros(sims)
    for c, (a, b) in comp.items():
        d = b - a
        for s in range(sims):   # ponytail: python loop is fine for 20k sims x 3 classes; vectorise if it gets slow
            k = counts[s, c - 1]
            if k:
                tot[s] += d[rng.integers(0, len(d), k)].sum()
    sim = tot / n + rng.normal(0, RUN_SD, sims)
    return (sim <= lb_d).mean(), np.percentile(sim, [5, 50, 95])


# ---------------------------------------------------------------- gen20 diagnosis (np bench, lists available)
def gen20_diag(D):
    K, akey, L, src = D['K'], D['akey'], D['lists'], D['src']
    cls = D['np']['cls']
    pool = set(pd.read_parquet(f'{BD}/pool/pool_meta.parquet', columns=['key']).key)
    ts = pd.read_parquet(f'{BD}/pool/train_structs.parquet', columns=['key', 'n_spec']).groupby('key').n_spec.sum()
    rows, kern = [], []
    for m in L['v22']:
        a = [K.get(s) for s in L['v22'][m]]
        g = [K.get(s) for s in L['gen20'][m]]
        v22src = {K.get(s) for d in src['run2'][m].values() for s in d}
        g20pc = {K.get(s) for s in src['np_g20'][m]['pubchem']}
        new = [k for k in g if k not in v22src]                      # only the wider generation can add these
        newgen = [k for k in new if k not in pool and k not in g20pc]
        pos_new = [i + 1 for i, k in enumerate(g) if k in set(new)]
        ak = akey[m]
        ra = a.index(ak) + 1 if ak in a else 0
        rg = g.index(ak) + 1 if ak in g else 0
        rows.append(dict(molecule_id=m, cls=cls[m], r_v22=ra, r_gen20=rg, n_new_top25=len(new), n_newgen_top25=len(newgen),
                         n_new_top5=sum(p <= 5 for p in pos_new), new_at1=int(1 in pos_new),
                         ans_in_v22_sources=ak in v22src, ans_in_pool=ak in pool, ans_nspec=int(ts.get(ak, 0)),
                         libmax=D['libmax'].get(m, np.nan)))
        for p, k in enumerate(a, 1):                                # every V22 candidate as a proxy answer
            q = g.index(k) + 1 if k in g else 0
            kern.append(dict(cls=cls[m], p=p, q=q, d_rr=(1 / q if q else 0.0) - 1 / p, is_answer=k == ak,
                             n_new_above=sum(1 for x in pos_new if x <= (q if q else 25))))
    ro = []   # does gen20 re-order the candidates V22 already had? (engine 1's features are list-relative)
    for part in ('base_tail', 'prefusion'):
        for m in src['run2']:
            a = [K.get(s) for s in src['run2'][m][part]]
            g = [K.get(s) for s in src['np_g20'][m][part]]
            sa, sg = set(a), set(g)
            ac, gc = [k for k in a if k in sg], [k for k in g if k in sa]
            ak = akey[m]
            ro.append(dict(part=part, cls=cls[m], top1_flip=bool(ac and gc and ac[0] != gc[0]), top5_reorder=ac[:5] != gc[:5],
                           new_at1=bool(g) and g[0] not in sa, n_new=len(sg - sa),
                           ans_lost_top1=(bool(a) and a[0] == ak and (not g or g[0] != ak)),
                           ans_gained_top1=(bool(g) and g[0] == ak and (not a or a[0] != ak))))
    return pd.DataFrame(rows), pd.DataFrame(kern), pd.DataFrame(ro), ts


# ---------------------------------------------------------------- report
def f(x, d=4):
    return '' if pd.isna(x) else f'{x:+.{d}f}' if isinstance(x, float) and d < 0 else f'{x:.{d}f}'


def sgn(x):
    return '' if pd.isna(x) else f'{x:+.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', type=int, default=4000)
    ap.add_argument('--step', type=float, default=0.02)
    args = ap.parse_args()
    D = load()
    C = comparisons(D)
    S = summary_rows(C, args.boot)
    o = ['# Bench calibration against the public LB (generated by `bench/calibrate.py`)', '']
    o += [f'Scoring check: own re-scoring of the np csvs vs the bench json rr, molecules that differ: {D["check"]}; '
          f'answer keys differing from bench metric_key: {D["bad_akey"]}. run3 and all unseen rows use the bench json '
          f'(their final csvs were overwritten by later stages).', '']
    o += ['## 1. Variant x bench (deltas vs the variant\'s own base; base for rel100/rel250/e2r3/no_e2r2 = gen20 replay)', '',
          'nat = the bench\'s own class mix; mix = community 28/20/52 (renormalised over the classes the bench has). '
          'P = paired stratified bootstrap P(variant better). LB delta = public score - 0.451 (noise about +-0.004 per run).', '',
          '| variant | bench | n | base MRR (nat) | var MRR (nat) | d nat [95% CI] | P nat | d mix [95% CI] | P mix | d c1 | d c2 | d c3 | top-1 base->var | hit@25 base->var | mols changed | LB d | sign ok (mix) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in S.itertuples():
        ok = '' if pd.isna(r.lb_d) else ('yes' if sign_ok(r.mix_d, r.variant) else 'NO')
        o.append(f'| {r.variant} | {r.bench} | {r.n} | {r.nat_base:.4f} | {r.nat_var:.4f} | {sgn(r.nat_d)} [{sgn(r.nat_lo)}, {sgn(r.nat_hi)}] | '
                 f'{r.nat_p:.2f} | {sgn(r.mix_d)} [{sgn(r.mix_lo)}, {sgn(r.mix_hi)}] | {r.mix_p:.2f} | {sgn(r.c1)} | {sgn(r.c2)} | {sgn(r.c3)} | '
                 f'{r.top1_base}->{r.top1} | {r.hit_base}->{r.hit} | {r.changed} | {sgn(r.lb_d)} | {ok} |')
    o += ['', '| depth7 | any | - | no per-molecule bench output exists (`d7out` holds only a 400-row visible-test csv, no answers) | | | | | | | | | | | | +0.0000 | untestable |', '']
    # absolute per-class levels for the table the caller asked for
    o += ['Per-class MRR of each scored list (variant rows of the table above use these):', '',
          '| bench | list | all | c1 | c2 | c3 | top-1 | hit@25 |', '|---|---|---|---|---|---|---|---|']
    for b in ('np', 'unseen', 'engine1'):
        cls = D[b]['cls']
        for k, s in D[b]['rr'].items():
            s = s.reindex(cls.index)
            per = [s[cls == c].mean() if (cls == c).any() else np.nan for c in (1, 2, 3)]
            o.append(f'| {b} | {k} | {s.mean():.4f} | ' + ' | '.join('' if pd.isna(x) else f'{x:.4f}' for x in per)
                     + f' | {(s == 1).sum()} | {(s > 0).sum()} |')
    # 2. mix search
    o += ['', '## 2. Class re-weighting: is there a test mix where every bench delta takes the LB sign?', '',
          f'Rule: gen20, rel100, rel250 must come out negative; E2R3 must not be a gain above {TIE_TOL}. Grid step {args.step} '
          'over (w1, w2, w3); p_all = bootstrap share of draws in which all signs are right at that mix.', '']
    for b in ('np', 'unseen', 'honest'):
        G, vs = mix_search(C, b, args.step, max(1000, args.boot // 4))
        good = G[G.all_ok]
        o.append(f'**{b}** ({len(G)} mixes): all signs right at {len(good)} mixes; '
                 + ', '.join(f'{v} right at {int(G[f"ok_{v}"].sum())}' for v in vs) + '.')
        if len(good):
            g = good.sort_values('p_all', ascending=False)
            o.append(f'  - region: w1 {good.w1.min():.2f}-{good.w1.max():.2f}, w2 {good.w2.min():.2f}-{good.w2.max():.2f}, '
                     f'w3 {good.w3.min():.2f}-{good.w3.max():.2f}; best p_all {g.p_all.iloc[0]:.2f} at '
                     f'({g.w1.iloc[0]:.2f}, {g.w2.iloc[0]:.2f}, {g.w3.iloc[0]:.2f}); mixes with p_all >= 0.8: {(G.p_all >= 0.8).sum()}')
            top = g.head(5)
            o.append('  - top mixes: ' + '; '.join(f'({t.w1:.2f},{t.w2:.2f},{t.w3:.2f}) p_all {t.p_all:.2f} '
                     + ' '.join(f'{v} {getattr(t, "d_" + v):+.4f}' for v in vs) for t in top.itertuples()))
        lv = np.array([level(D, w) for w in G[['w1', 'w2', 'w3']].to_numpy()])
        G['plaus'] = (lv[:, 0] <= LB_BASE + 0.03) & (lv[:, 1] >= LB_BASE - 0.03)
        o.append(f'  - level-plausible mixes (V22 honest-slice level <= 0.481 and np level >= 0.421): {int(G.plaus.sum())}; '
                 f'of them with all signs right: {int((G.plaus & G.all_ok).sum())}; best p_all among them: '
                 f'{G.p_all[G.plaus].max() if G.plaus.any() else float("nan"):.2f}')
        com = G.iloc[((G[['w1', 'w2', 'w3']] - at_mix({1: 0, 2: 0, 3: 0} if b != 'unseen' else {2: 0, 3: 0}, COMMUNITY_MIX)) ** 2).sum(axis=1).argmin()]
        o.append(f'  - at the community mix: ' + ', '.join(f'{v} {com["d_" + v]:+.4f} ({"ok" if com["ok_" + v] else "WRONG"})' for v in vs)
                 + f'; p_all {com.p_all:.2f}')
        o.append('')
    # 3. gen20 diagnosis
    R, Kn, RO, ts = gen20_diag(D)
    ch = R[R.r_v22 != R.r_gen20]
    o += ['## 3. gen20 diagnosis (np bench, final lists of run2 = V22 and np_g20 = V22 + gen20)', '',
          f'Molecules whose answer rank changed: {len(ch)} (class 1: {(ch.cls == 1).sum()}, class 2: {(ch.cls == 2).sum()}, '
          f'class 3: {(ch.cls == 3).sum()}). Class-3 newly found: {((ch.cls == 3) & (ch.r_v22 == 0) & (ch.r_gen20 > 0)).sum()}, '
          f'newly at rank 1: {((ch.cls == 3) & (ch.r_gen20 == 1) & (ch.r_v22 != 1)).sum()}.', '',
          '| molecule | class | V22 rank | gen20 rank | answer in V22 sources | new cands in top 25 | of them in top 5 | lib_max |',
          '|---|---|---|---|---|---|---|---|']
    for r in ch[ch.cls < 3].itertuples():
        o.append(f'| {r.molecule_id} | {r.cls} | {r.r_v22} | {r.r_gen20} | {r.ans_in_v22_sources} | {r.n_new_top25} | {r.n_new_top5} | {f(r.libmax, 2)} |')
    o += ['', 'Crowding (new candidates = in gen20\'s final 25 but in none of V22\'s candidate sources):', '',
          '| class | molecules | mean new in top 25 | mean new in top 5 | share with a new candidate at rank 1 | median n_spec of the answer (train) |',
          '|---|---|---|---|---|---|']
    for c, g in R.groupby('cls'):
        o.append(f'| {c} | {len(g)} | {g.n_new_top25.mean():.2f} | {g.n_new_top5.mean():.2f} | {g.new_at1.mean():.3f} | {g.ans_nspec.median():.0f} |')
    o += ['', f'For scale: a typical training structure has {ts.median():.0f} library spectra (median; 90th percentile '
          f'{ts.quantile(0.9):.0f}); the np-bench answers have {R.ans_nspec.median():.0f}.', '',
          'Re-ordering of candidates V22 already had (engine-1 list `base_tail` and the pre-fusion list), np bench:', '',
          '| list | class | top-1 among shared candidates flips | top-5 order changes | a new candidate takes rank 1 | mean new candidates | correct answer loses rank 1 | correct answer gains rank 1 |',
          '|---|---|---|---|---|---|---|---|']
    for (pt, c), g in RO.groupby(['part', 'cls']):
        o.append(f'| {pt} | {c} | {g.top1_flip.mean():.3f} | {g.top5_reorder.mean():.3f} | {g.new_at1.mean():.3f} | {g.n_new.mean():.1f} | '
                 f'{int(g.ans_lost_top1.sum())} | {int(g.ans_gained_top1.sum())} |')
    o += ['', 'Displacement kernel: every candidate V22 ranked at position p, followed into gen20\'s final list '
          '(proxy for a correct library / COCONUT answer sitting at p). d_rr = change in its reciprocal rank if it were the answer.', '',
          '| V22 position | class | n | share pushed down | share dropped from 25 | mean d_rr | mean d_rr, true answers only (n) |',
          '|---|---|---|---|---|---|---|']
    Kn['bucket'] = pd.cut(Kn.p, [0, 1, 3, 10, 25], labels=['1', '2-3', '4-10', '11-25'])
    for (bk, c), g in Kn.groupby(['bucket', 'cls'], observed=True):
        ga = g[g.is_answer]
        o.append(f'| {bk} | {c} | {len(g)} | {((g.q > g.p) | (g.q == 0)).mean():.3f} | {(g.q == 0).mean():.3f} | {g.d_rr.mean():+.4f} | '
                 + (f'{ga.d_rr.mean():+.4f} ({len(ga)})' if len(ga) else '-') + ' |')
    # what the test would need for -0.010
    k1 = Kn[Kn.cls < 3].groupby('bucket', observed=True).d_rr.mean()
    o += ['', '### LB consistency: how likely is -0.010 on ~124 public molecules given each bench\'s per-molecule deltas?', '',
          f'Simulation: draw {N_PUBLIC} molecules at the mix from the bench\'s paired deltas, add N(0, {RUN_SD}) run noise; '
          'P = share of draws at or below the LB delta.', '', '| variant | bench | mix | P(sim <= LB d) | sim 5/50/95% |', '|---|---|---|---|---|']
    for v in ('gen20', 'rel100', 'rel250', 'e2r3'):
        for b in ('np', 'honest', 'unseen'):
            for w in (COMMUNITY_MIX, (0.4, 0.3, 0.3)):
                p, q = lb_sim(C[(v, b)], w, LB[v] - LB_BASE)
                o.append(f'| {v} | {b} | {w} | {p:.3f} | {q[0]:+.4f} / {q[1]:+.4f} / {q[2]:+.4f} |')
    h = C[('gen20', 'honest')]
    g23 = COMMUNITY_MIX[1] * (h[2][1] - h[2][0]).mean() + COMMUNITY_MIX[2] * (h[3][1] - h[3][0]).mean()
    o += ['', f'What the test would need for -0.010 at 28/20/52: honest class-2/3 gains add {g23:+.4f}; a rank 1 -> 2 flip of a '
          f'correct answer costs 0.5, so class 1 alone would need a flip rate of {(0.010 + g23) / COMMUNITY_MIX[0] / 0.5:.1%}, '
          f'classes 1+2 together {(0.010 + g23) / sum(COMMUNITY_MIX[:2]) / 0.5:.1%}, or {0.010 / sum(COMMUNITY_MIX[:2]) / 0.5:.1%} '
          'if the test\'s class-3 gain is zero. Bench rates: see the re-ordering table (correct answer loses rank 1).']
    o += ['', f'Kernel means for class 1/2 proxies by V22 position: ' + ', '.join(f'{i}: {x:+.4f}' for i, x in k1.items()), '']
    o += READING
    open(OUT, 'w').write('\n'.join(o) + '\n')
    R.to_csv(os.path.join(HERE, 'results', 'calibration_gen20_molecules.csv'), index=False)
    print('\n'.join(o))


if __name__ == '__main__':
    main()
