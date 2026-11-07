"""Neutral-loss necessity check on ranks 2-25 (lever 3, step A).

What: a strong observed neutral loss (precursor minus loss, relative intensity >= 0.3, in at least half of a
molecule's spectra) needs a group that can lose it. Candidates in ranks 2-25 lacking that group move to the end of the
25-list; rank 1 never changes.
Why: only physical necessity is used (no learned weights); script 55 moved rank 1 on loss evidence and lost.
Output: artifacts/64_loss_necessity.json (diagnostic: violation rate of true answers vs other candidates, MRR delta,
paired bootstrap, up/down counts) and a printed summary.
Usage: python 64_loss_necessity.py [bench_dir]   (default /home/user/work/out_bench477)
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
LOSSES = {  # name: (neutral mass, SMARTS the candidate must contain)
    'CO2': (43.98983, '[CX3](=O)[OX2]'),
    'HCOOH': (46.00548, '[CX3](=O)[OX2H1]'),
    'NH3': (17.02655, '[#7;!H0;!$([#7]=*)]'),
    'SO2': (63.96190, '[#16](=O)=O'),
    'CH3OH': (32.02621, '[CH3][OX2]'),
}
PAT = {k: Chem.MolFromSmarts(v[1]) for k, v in LOSSES.items()}
TOL, REL, FRAC = 0.005, 0.3, 0.5
ADDUCT_NEUTRAL_SHIFT = {}  # losses are taken from the precursor ion, so no adduct arithmetic is needed


def observed_losses(spectra):
    """spectra: list of (mzs, normalized intensities, precursor). Returns the set of losses seen in >= FRAC of them."""
    cnt = {k: 0 for k in LOSSES}
    for mz, it, prec in spectra:
        mz, it = np.asarray(mz, float), np.asarray(it, float)
        if not len(it) or it.max() <= 0:
            continue
        it = it / it.max()
        for k, (m, _) in LOSSES.items():
            sel = np.abs(mz - (prec - m)) <= TOL
            if sel.any() and it[sel].max() >= REL:
                cnt[k] += 1
    return {k for k, c in cnt.items() if c >= max(1, FRAC * len(spectra))}


def has_group(smi, cache={}):
    if smi not in cache:
        m = Chem.MolFromSmiles(smi)
        cache[smi] = {k: (m is not None and m.HasSubstructMatch(p)) for k, p in PAT.items()}
    return cache[smi]


def apply(lists, loss_by_mol):
    out, moved = {}, 0
    for m, l in lists.items():
        need = loss_by_mol.get(m, set())
        if not need or len(l) < 3:
            out[m] = l; continue
        head, tail = l[:25], l[25:]
        bad = [i for i in range(1, len(head)) if any(not has_group(head[i])[k] for k in need)]
        new = [head[0]] + [head[i] for i in range(1, len(head)) if i not in bad] + [head[i] for i in bad] + tail
        assert new[0] == l[0] and sorted(new) == sorted(l)
        moved += new != l; out[m] = new
    return out, moved


if __name__ == '__main__':
    B = Path(sys.argv[1] if len(sys.argv) > 1 else '/home/user/work/out_bench477')
    test = pd.read_parquet(B / 'bench_comp/test.parquet')
    ans = pd.read_csv(B / 'bench_answers.csv'); tsmi = dict(zip(ans.molecule_id, ans.normalized_smiles)); tk = dict(zip(ans.molecule_id, ans.inchikey14))
    sub = pd.read_csv(B / 'submission.csv'); lists = {m: s.split(';') for m, s in zip(sub.molecule_id, sub.smiles)}
    loss = {m: observed_losses([(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz) for r in g.itertuples()])
            for m, g in test.groupby('molecule_id')}
    key = lambda s: Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]
    # diagnostic: violation rate, true answers vs other candidates in ranks 2-25
    tv = tn = ov = on = 0
    per = {k: [0, 0, 0, 0] for k in LOSSES}
    for m, need in loss.items():
        for k in need:
            g = has_group(tsmi[m])[k]; per[k][0] += 1; per[k][1] += (not g); tv += (not g); tn += 1
            for s in lists[m][1:25]:
                if key(s) == tk[m]: continue
                gg = has_group(s)[k]; ov += (not gg); on += 1; per[k][2] += 1; per[k][3] += (not gg)
    print('molecules with an observed loss:', sum(1 for v in loss.values() if v), '| per-loss (n_true, true_viol, n_other, other_viol):', per)
    print(f'TRUE answer violation {tv}/{tn} = {tv/max(tn,1):.3f}   other candidates {ov}/{on} = {ov/max(on,1):.3f}')
    new, moved = apply(lists, loss)
    def rr(L):
        out = []
        for m, l in L.items():
            ks = [key(s) for s in l[:25]]; out.append(1 / (ks.index(tk[m]) + 1) if tk[m] in ks else 0.0)
        return np.array(out)
    a, b = rr(lists), rr(new); d = b - a
    rng = np.random.default_rng(0); bs = np.array([rng.choice(d, len(d)).mean() for _ in range(5000)])
    r1 = sum(lists[m][0] != new[m][0] for m in lists)
    res = dict(lists_reordered=int(moved), mrr_base=float(a.mean()), mrr_new=float(b.mean()), delta=float(d.mean()),
               p_better=float((bs > 0).mean()), up=int((d > 0).sum()), down=int((d < 0).sum()), rank1_changed=int(r1),
               true_violation=[int(tv), int(tn)], other_violation=[int(ov), int(on)])
    print(json.dumps(res)); Path('artifacts').mkdir(exist_ok=True); Path('artifacts/64_loss_necessity.json').write_text(json.dumps(res, indent=1))
