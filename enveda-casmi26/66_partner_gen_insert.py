"""Class-3 partner generation, step 2: insertion policy on bench477 (rank 1 locked).

What: products from 65 (A's rank 1 transformed by the mass-matched biotransformation) are ordered by the untrained
MetFrag-lite fragment score on B's spectra and inserted into B's list after rank 1 (existing copies removed first).
Policies: P_a all products at rank 2+; P_b all products at rank 4+; P_c best product only at rank 2; P_d best product
at rank 2 only when B's rank 1 is not confident (library_max < 0.87); added after P_a-P_d all lost (stated before
running): P_e all products at rank 16+; P_f the same only when B's rank-1 library_max < 0.7.
Why: step 1 passed (6 truths generated that are missing from the lists); the question is whether insertion gains
more than it costs by pushing existing answers down.
Output: artifacts/66_partner_gen_insert.json; acceptance P(better) >= 0.9, no truth pushed out of the 25.
Usage: python 66_partner_gen_insert.py [bench_dir]
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'overlay'))
gen = importlib.import_module('65_partner_transform_gen')
B = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/user/work/out_bench477'); sys.path.insert(0, str(B))
import frag_rescore as fr
test = pd.read_parquet(B / 'bench_comp/test.parquet')
ans = pd.read_csv(B / 'bench_answers.csv'); tk = dict(zip(ans.molecule_id, ans.inchikey14)); cls = dict(zip(ans.molecule_id, ans.cls))
sub = pd.read_csv(B / 'submission.csv'); L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
ev = pd.read_parquet(B / 'frozen_candidate_evidence.parquet'); lib = {(m, x): v for m, x, v in zip(ev.molecule_id, ev.smiles, ev.library_max.fillna(0))}
t = test[test.adduct.isin(gen.ADD)].copy(); t['nm'] = t.precursor_mz - t.adduct.map(gen.ADD); NM = t.groupby('molecule_id').nm.median().to_dict()
nbr = gen.partners(test)
kc = {}
def k14(s):
    if s not in kc: m = Chem.MolFromSmiles(s); kc[s] = gen.key(m) if m is not None else None
    return kc[s]
def spectra(m):
    g = test[test.molecule_id == m]
    return [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float) / max(np.max(r.ms2_normalized_intensities), 1e-12),
                 adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1) for r in g.itertuples() if len(r.ms2_mzs)]
prods = {}
for b, As in nbr.items():
    if b not in NM: continue
    P = {}
    for a in As:
        if a not in NM: continue
        dm = NM[b] - NM[a]
        if abs(dm) < gen.TOL: continue
        for k, s in gen.generate(L[a][0], dm).items():
            if k != k14(L[b][0]): P.setdefault(k, s)
    if P:
        smis = list(P.values()); sc = fr.score_molecule(spectra(b), smis, cache={})
        prods[b] = [s for s, _ in sorted(zip(smis, sc), key=lambda z: -(z[1] or 0))]
print('molecules with products:', len(prods), '| products total:', sum(map(len, prods.values())))
def insert(l, ps, at):
    ks = {k14(p) for p in ps}; rest = [x for x in l if x == l[0] or k14(x) not in ks]
    return rest[:at] + list(ps) + rest[at:]
def policy(name):
    out = {}
    for m, l in L.items():
        ps = prods.get(m)
        if not ps: out[m] = l; continue
        if name == 'P_a': out[m] = insert(l, ps, 1)
        elif name == 'P_b': out[m] = insert(l, ps, 3)
        elif name == 'P_c': out[m] = insert(l, ps[:1], 1)
        elif name == 'P_d': out[m] = insert(l, ps[:1], 1) if lib.get((m, l[0]), 0) < 0.87 else l
        elif name == 'P_e': out[m] = insert(l, ps, 15)
        elif name == 'P_f': out[m] = insert(l, ps, 15) if lib.get((m, l[0]), 0) < 0.7 else l
    return out
def rr(D):
    o = []
    for m in L:
        ks = [k14(x) for x in D[m][:25]]; o.append(1 / (ks.index(tk[m]) + 1) if tk[m] in ks else 0.0)
    return np.array(o)
base = rr(L); res = {}
inl = np.array([tk[m] in {k14(x) for x in L[m][:25]} for m in L])
for p in ['P_a', 'P_b', 'P_c', 'P_d', 'P_e', 'P_f']:
    D = policy(p); assert all(D[m][0] == L[m][0] for m in L)
    d = rr(D) - base; rng = np.random.default_rng(0); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    newl = np.array([tk[m] in {k14(x) for x in D[m][:25]} for m in L])
    c3 = np.array([cls[m] == 3 for m in L])
    res[p] = dict(delta=round(float(d.mean()), 5), p_better=round(float((bs > 0).mean()), 3), up=int((d > 0).sum()), down=int((d < 0).sum()),
                  pushed_out=int((inl & ~newl).sum()), new_in=int((~inl & newl).sum()), class3_delta=round(float(d[c3].mean()), 4))
    print(p, res[p])
Path(HERE / 'artifacts').mkdir(exist_ok=True); (HERE / 'artifacts/66_partner_gen_insert.json').write_text(json.dumps(res, indent=1))
