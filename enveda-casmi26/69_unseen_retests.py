"""Re-test, on the calibrated leak-free unseen bench, the CASMI ideas rejected on the leaky bench477 (7-8 Nov).

What: base = the unseen bench's merged477 lists (0.477-era pipeline, production models, 320 Enveda-180 timsTOF
structures absent from train.parquet, classes 2/3). Every idea keeps rank 1 unless it says otherwise. Scored like
68_unseen_calibration.py (InChIKey-14 or metric key), paired bootstrap P(better), up/down, rank-1 changes.
Ideas: A loss necessity (64); B isomer exclusion R3 (overlay/iso_exclusion.py) and R1 rank-1 collision swap;
C formula prior (same-formula first behind a confident rank 1); D partner-transform generation (65) with the
insertion policies of 66 (P_a all at rank 2+, P_c best at rank 2, P_f all at 16+ when rank-1 lib < 0.7).
Acceptance (unchanged): P(better) >= 0.9, no truth pushed out of the 25, rank 1 only where the idea says.
Why: the calibration passed (68), so ideas rejected on the leaky bench may have been rejected for the wrong reason.
Output: artifacts/69_unseen_retests.json. Usage: python 69_unseen_retests.py [bench_dir]
"""
import json, sys, importlib
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'overlay'))
B = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/user/work/unseen_bench'); sys.path.insert(0, str(B))
ln = importlib.import_module('64_loss_necessity'); gen = importlib.import_module('65_partner_transform_gen')
import iso_exclusion as iso, frag_rescore as fr
test = pd.read_parquet(B / 'bench_comp/test.parquet')
ans = pd.read_csv(B / 'bench_answers.csv', dtype=str, keep_default_na=False)
keys = {m: {a, b} - {''} for m, a, b in zip(ans.molecule_id, ans.inchikey14, ans.metric_key)}
cls = dict(zip(ans.molecule_id, ans.cls.astype(int)))
sub = pd.read_csv(B / 'submission.csv', dtype=str, keep_default_na=False); L = dict(zip(sub.molecule_id, sub.smiles.str.split(';')))
ev = pd.read_parquet(B / 'frozen_candidate_evidence.parquet'); lib = {(m, x): v for m, x, v in zip(ev.molecule_id, ev.smiles, ev.library_max.fillna(0))}
kc, fc = {}, {}
def k14(s):
    if s not in kc: x = Chem.MolFromSmiles(s); kc[s] = Chem.MolToInchiKey(x)[:14] if x is not None else None
    return kc[s]
def form(s):
    if s not in fc: x = Chem.MolFromSmiles(s); fc[s] = CalcMolFormula(x) if x is not None else None
    return fc[s]
def rr(D):
    return np.array([next((1 / (i + 1) for i, x in enumerate(D[m][:25]) if k14(x) in keys[m]), 0.0) for m in ans.molecule_id])
def hit(D):
    return np.array([any(k14(x) in keys[m] for x in D[m][:25]) for m in ans.molecule_id])
base, base_hit = rr(L), hit(L)
c3 = np.array([cls[m] == 3 for m in ans.molecule_id])
def report(name, D):
    d = rr(D) - base; rng = np.random.default_rng(0); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    h = hit(D)
    r = dict(delta=round(float(d.mean()), 5), p_better=round(float((bs > 0).mean()), 3), up=int((d > 0).sum()), down=int((d < 0).sum()),
             rank1_changed=int(sum(D[m][0] != L[m][0] for m in L)), pushed_out=int((base_hit & ~h).sum()), new_in=int((~base_hit & h).sum()),
             class3_delta=round(float(d[c3].mean()), 4), class2_delta=round(float(d[~c3].mean()), 4))
    print(name, r, flush=True); return r
print('base merged477 MRR', round(base.mean(), 4), '| truths in 25:', int(base_hit.sum()), 'of', len(base), flush=True)
R = {}
# A loss necessity
loss = {m: ln.observed_losses([(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz) for r in g.itertuples()]) for m, g in test.groupby('molecule_id')}
DA, _ = ln.apply(L, loss); R['A_loss_necessity'] = report('A loss necessity', DA)
# B isomer exclusion
nb, st = iso.iso_exclude(sub, ev, test); R['B_iso_R3'] = report('B iso R3', dict(zip(nb.molecule_id, nb.smiles.str.split(';')))); R['B_iso_R3']['stats'] = st
G = iso.iso_groups(test); D1 = {m: list(l) for m, l in L.items()}
for a in D1:
    for b in G.get(a, ()):
        if a < b and k14(D1[a][0]) == k14(D1[b][0]) and len(D1[a]) > 1 and len(D1[b]) > 1:
            los = b if lib.get((a, D1[a][0]), 0) >= lib.get((b, D1[b][0]), 0) else a
            D1[los][0], D1[los][1] = D1[los][1], D1[los][0]
R['B_iso_R1'] = report('B iso R1 (rank-1 collision swap)', D1)
# C formula prior
for tau in (0.7, 0.87):
    DC = {}
    for m, l in L.items():
        if lib.get((m, l[0]), 0) >= tau:
            f0 = form(l[0]); h = l[:25]; DC[m] = [h[0]] + [x for x in h[1:] if form(x) == f0] + [x for x in h[1:] if form(x) != f0] + l[25:]
        else: DC[m] = l
    R[f'C_formula_tau{tau}'] = report(f'C formula prior tau {tau}', DC)
# D partner generation
ADD = gen.ADD; t = test[test.adduct.isin(ADD)].copy(); t['nm'] = t.precursor_mz - t.adduct.map(ADD); NM = t.groupby('molecule_id').nm.median().to_dict()
nbr = gen.partners(test); prods = {}
def spectra(m):
    g = test[test.molecule_id == m]
    return [dict(mz=np.asarray(r.ms2_mzs, float), it=np.asarray(r.ms2_normalized_intensities, float) / max(np.max(r.ms2_normalized_intensities), 1e-12),
                 adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1) for r in g.itertuples() if len(r.ms2_mzs)]
for b, As in nbr.items():
    if b not in NM or b not in L: continue
    P = {}
    for a in As:
        if a not in NM or a not in L: continue
        dm = NM[b] - NM[a]
        if abs(dm) < gen.TOL: continue
        for k, s in gen.generate(L[a][0], dm).items():
            if k != k14(L[b][0]): P.setdefault(k, s)
    if P:
        smis = list(P.values()); sc = fr.score_molecule(spectra(b), smis, cache={})
        prods[b] = [s for s, _ in sorted(zip(smis, sc), key=lambda z: -(z[1] or 0))]
gen_truth = sum(any(k14(p) in keys[m] for p in ps) for m, ps in prods.items())
print('D partner generation: molecules with products', len(prods), '| products', sum(map(len, prods.values())), '| truth generated for', gen_truth, flush=True)
def ins(l, ps, at):
    ks = {k14(p) for p in ps}; rest = [x for x in l if x == l[0] or k14(x) not in ks]; return rest[:at] + list(ps) + rest[at:]
for name, fn in [('D_P_a', lambda m, l, ps: ins(l, ps, 1)), ('D_P_c', lambda m, l, ps: ins(l, ps[:1], 1)),
                 ('D_P_f', lambda m, l, ps: ins(l, ps, 15) if lib.get((m, l[0]), 0) < 0.7 else l)]:
    R[name] = report(name, {m: (fn(m, l, prods[m]) if m in prods else l) for m, l in L.items()})
R['_meta'] = dict(base_mrr=round(float(base.mean()), 4), n=len(base), partner_molecules=len(prods), gen_truth=int(gen_truth))
(HERE / 'artifacts/69_unseen_retests.json').write_text(json.dumps(R, indent=1, default=str))
