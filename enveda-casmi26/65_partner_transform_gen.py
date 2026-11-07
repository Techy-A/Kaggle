"""Class-3 candidate generation from test-side spectral partners (step 1: recall diagnostic).

What: for each molecule B with a spectral partner A (xspec RULE: shift-aware cosine >= 0.7, >= 6 matched peaks, same
ion mode), if the neutral-mass difference M_B - M_A matches a biotransformation (within 0.005 Da), apply that
transformation to A's structure at every valid site and collect the products as candidates for B.
Why: earlier class-3 generation used libraries or generic rules (lever rows 5, 11, 13, 38: flat or negative); the test
itself holds analogues of the hard molecules (partners), which no earlier attempt used to generate structures.
Output: artifacts/65_partner_gen.json: oracle recall (A's true structure), realistic recall (A's rank-1, all and
library_max >= 0.87), product counts, and how many hits are NOT already in B's 25-list.
Gate (pre-registered, CASMI26.md 7 Nov 23:50): realistic recall finds >= 3 truths absent from their lists, <= 30
products per molecule.
Usage: python 65_partner_transform_gen.py [bench_dir]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, str(Path(__file__).parent / 'overlay'))
from xspec_boost import partners
ADD = {'[M+H]+': 1.007276, '[M-H]-': -1.007276, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033823, '[M+K]+': 38.963158,
       '[M+Cl]-': 34.969402, '[M+CH2O2-H]-': 44.998201, '[M-H2O+H]+': 1.007276 - 18.010565}
TOL = 0.005
# additions: name -> (mass delta, reaction SMARTS list)
ADDS = {
    '+O': (15.994915, ['[#6;!H0:1]>>[#6:1]O']),
    '+CH2': (14.015650, ['[O,N;!H0:1]>>[*:1]C']),
    '+H2': (2.015650, ['[C:1]=[C:2]>>[C:1]-[C:2]', '[C:1]=[O:2]>>[C:1]-[O:2]']),
    '-H2': (-2.015650, ['[C;!H0;!$(C=*):1]-[C;!H0;!$(C=*):2]>>[C:1]=[C:2]']),
    '+hexose': (162.052824, ['[OX2H1:1]>>[O:1]C1OC(CO)C(O)C(O)C1O']),
    '+acetyl': (42.010565, ['[O,N;!H0:1]>>[*:1]C(C)=O']),
    '+SO3': (79.956815, ['[OX2H1:1]>>[O:1]S(=O)(=O)O']),
    '+glucuronide': (176.032088, ['[OX2H1:1]>>[O:1]C1OC(C(=O)O)C(O)C(O)C1O']),
}
# deletions: name -> (mass delta, formula of the removed side after the cut, H-capped)
DELS = {'-O': (-15.994915, 'H2O'), '-CH2': (-14.015650, 'CH4'), '-hexose': (-162.052824, 'C6H12O6'),
        '-acetyl': (-42.010565, 'C2H4O'), '-SO3': (-79.956815, 'H2O3S'), '-glucuronide': (-176.032088, 'C6H10O7')}
RXN = {k: (dm, [AllChem.ReactionFromSmarts(s) for s in ss]) for k, (dm, ss) in ADDS.items()}


def key(m):
    try: return Chem.MolToInchiKey(m)[:14]
    except Exception: return None


def generate(smi, dm):
    m = Chem.MolFromSmiles(smi)
    if m is None: return {}
    out = {}
    for name, (d, rx) in RXN.items():
        if abs(d - dm) > TOL: continue
        for r in rx:
            for ps in r.RunReactants((m,)):
                p = ps[0]
                try:
                    Chem.SanitizeMol(p); k = key(p)
                    if k: out.setdefault(k, Chem.MolToSmiles(p))
                except Exception: pass
    for name, (d, f) in DELS.items():
        if abs(d - dm) > TOL: continue
        for b in m.GetBonds():
            if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE: continue
            rw = Chem.RWMol(m); rw.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            try:
                for a in rw.GetAtoms(): a.SetNoImplicit(False); a.SetNumExplicitHs(0) if a.GetIsAromatic() is False and a.GetSymbol() == 'C' else None
                Chem.SanitizeMol(rw)
                frags = Chem.GetMolFrags(rw, asMols=True)
            except Exception: continue
            if len(frags) != 2: continue
            for i in (0, 1):
                if CalcMolFormula(frags[i]) == f:
                    k = key(frags[1 - i])
                    if k: out.setdefault(k, Chem.MolToSmiles(frags[1 - i]))
    return out


if __name__ == '__main__':
    B = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/user/work/out_bench477')
    test = pd.read_parquet(B / 'bench_comp/test.parquet')
    ans = pd.read_csv(B / 'bench_answers.csv'); tk = dict(zip(ans.molecule_id, ans.inchikey14)); ts = dict(zip(ans.molecule_id, ans.normalized_smiles))
    cls = dict(zip(ans.molecule_id, ans.cls))
    sub = pd.read_csv(B / 'submission.csv'); L = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
    ev = pd.read_parquet(B / 'frozen_candidate_evidence.parquet'); lib = {(m, x): v for m, x, v in zip(ev.molecule_id, ev.smiles, ev.library_max.fillna(0))}
    t = test[test.adduct.isin(ADD)].copy(); t['nm'] = t.precursor_mz - t.adduct.map(ADD); NM = t.groupby('molecule_id').nm.median().to_dict()
    nbr = partners(test)
    kc = {}
    def k14(s):
        if s not in kc: m = Chem.MolFromSmiles(s); kc[s] = key(m) if m is not None else None
        return kc[s]
    rows = []
    for b, As in nbr.items():
        if b not in NM: continue
        for a in As:
            if a not in NM: continue
            dm = NM[b] - NM[a]
            if abs(dm) < TOL: continue
            names = [n for n, (d, _) in ADDS.items() if abs(d - dm) <= TOL] + [n for n, (d, _) in DELS.items() if abs(d - dm) <= TOL]
            if not names: continue
            in_list = tk[b] in {k14(x) for x in L[b][:25]}
            g_or = generate(ts[a], dm); g_rl = generate(L[a][0], dm)
            rows.append(dict(b=b, a=a, cls_b=int(cls[b]), dm=round(dm, 4), trans=names, truth_in_list=in_list,
                             a_rank1_right=k14(L[a][0]) == tk[a], a_conf=lib.get((a, L[a][0]), 0) >= 0.87,
                             n_oracle=len(g_or), hit_oracle=tk[b] in g_or, n_real=len(g_rl), hit_real=tk[b] in g_rl))
    f = pd.DataFrame(rows)
    print('partner pairs:', sum(len(v) for v in nbr.values()) // 2, '| molecules with partners:', len(nbr), '| pairs with a transform mass:', len(f))
    if len(f):
        print(f.to_string(index=False))
        new_real = f[f.hit_real & ~f.truth_in_list].b.nunique(); new_or = f[f.hit_oracle & ~f.truth_in_list].b.nunique()
        res = dict(pairs_with_transform=len(f), molecules=int(f.b.nunique()), oracle_hits=int(f[f.hit_oracle].b.nunique()),
                   oracle_new=int(new_or), real_hits=int(f[f.hit_real].b.nunique()), real_new=int(new_real),
                   real_new_conf=int(f[f.hit_real & ~f.truth_in_list & f.a_conf].b.nunique()),
                   max_products=int(f.n_real.max()), median_products=float(f.n_real.median()))
    else:
        res = dict(pairs_with_transform=0)
    print(json.dumps(res)); Path('artifacts').mkdir(exist_ok=True); Path('artifacts/65_partner_gen.json').write_text(json.dumps(res, indent=1))
