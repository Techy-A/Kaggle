"""58: cross-test-spectrum evidence (related test molecules lend evidence to each other), pre-check + bench replay.

What: (precheck) on the bench477 pseudo-test (250 enveda-np-examples molecules, 1,184 timsTOF spectra) compute, for
every pair of molecules, the best spectral similarity over their spectra (modified cosine; neutral-loss cosine; same
ion mode only) and the structural similarity of their true answers (RDKit Morgan r2 2048 Tanimoto; Murcko scaffold).
Reports the fraction of molecules with a related partner and how well spectral similarity predicts structural
relatedness. (boost) replays a guarded, rank-1-locked boost on the merged477 bench lists: when molecule A has a
confident rank 1 and A, B are spectral partners, candidates in B's ranks 2-25 that are Tanimoto-similar to A's rank 1
move up by a fixed number of places (rules in the RULE dict, pre-registered in the plan note before scoring).
Why: the host allows automated, unsupervised processing across test spectra (topic 752297); in a natural-product test
set, analog series may share scaffolds. Stop rule fixed in advance: drop if < 5% of molecules have a related partner.
Output: artifacts/58_summary.json, artifacts/58_pairs.csv (pairs above loose thresholds), artifacts/58_boost_per_list.csv.
CPU only, about 2-4 min.
Usage: python enveda-casmi26/58_cross_spectrum_boost.py precheck|boost [bench_dir] [merged477_csv]
  (merged477_csv: `tar xzf <bench_dir>/bench477_intermediates.tar.gz variants/submission_merged477.csv` into a temp dir)
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
PHASE = sys.argv[1] if len(sys.argv) > 1 else "precheck"
B = Path(sys.argv[2] if len(sys.argv) > 2 else "/home/user/work/out_bench477")
# merged477 lists live in bench477_intermediates.tar.gz (variants/submission_merged477.csv); extract it first.
LISTS = sys.argv[3] if len(sys.argv) > 3 else "variants/submission_merged477.csv"
ART = Path(__file__).resolve().parent / "artifacts"
TOL = 0.01  # Da, timsTOF
MIX = {1: 0.28, 2: 0.20, 3: 0.52}
# Pre-registered in "CASMI next steps plan.md" (section "Cross-test-spectrum boost") before the boost was scored.
RULE = dict(spec_cos=0.7, spec_match=6, conf_libmax=0.87, tani=0.5, shift=3, max_rank=25)

FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fp(smi):
    m = Chem.MolFromSmiles(smi)
    return FPG.GetFingerprint(m) if m is not None else None


def scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetRingInfo().NumRings() == 0:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))


def prep(mz, it, prec):
    """Peaks below precursor - 0.5, >= 1% of max, top 50, sqrt intensity, unit norm."""
    mz, it = np.asarray(mz, float), np.asarray(it, float)
    k = (mz < prec - 0.5) & (it >= 0.01 * it.max())
    mz, it = mz[k], it[k]
    o = np.argsort(-it)[:50]
    mz, it = mz[o], np.sqrt(it[o])
    return mz, it / (np.linalg.norm(it) + 1e-12)


def greedy(mza, ia, mzb, ib, shift=None):
    """Greedy (modified) cosine: each peak used once; matches by mz or by mz + precursor shift."""
    d = np.subtract.outer(mza, mzb)
    ok = np.abs(d) <= TOL
    if shift is not None and abs(shift) > TOL:
        ok |= np.abs(d - shift) <= TOL
    ai, bi = np.nonzero(ok)
    if len(ai) == 0:
        return 0.0, 0
    p = ia[ai] * ib[bi]
    o = np.argsort(-p)
    ua, ub, s, n = set(), set(), 0.0, 0
    for j in o:
        if ai[j] in ua or bi[j] in ub:
            continue
        ua.add(ai[j]); ub.add(bi[j]); s += p[j]; n += 1
    return s, n


def losses(mz, it, prec):
    """Neutral losses 12-250 Da of the prepared peaks, unit norm."""
    l = prec - mz
    k = (l >= 12) & (l <= 250)
    l, w = l[k], it[k]
    return l, w / (np.linalg.norm(w) + 1e-12)


def spectral_pairs(q):
    """Best modified cosine / NL cosine per molecule pair (same ion mode), with matched-peak counts."""
    S = []
    for r in q.itertuples():
        mz, it = prep(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz)
        lz, lw = losses(mz, it, r.precursor_mz)
        S.append((r.molecule_id, r.ionization_mode, r.precursor_mz, mz, it, lz, lw))
    best = {}
    for a, b in combinations(range(len(S)), 2):
        ma, mb = S[a], S[b]
        if ma[0] == mb[0] or ma[1] != mb[1] or len(ma[3]) == 0 or len(mb[3]) == 0:
            continue
        mc, mn = greedy(ma[3], ma[4], mb[3], mb[4], shift=ma[2] - mb[2])
        nc, nn = greedy(ma[5], ma[6], mb[5], mb[6]) if len(ma[5]) and len(mb[5]) else (0.0, 0)
        key = tuple(sorted((ma[0], mb[0])))
        cur = best.setdefault(key, dict(mod_cos=0.0, mod_n=0, mod6=0.0, nl_cos=0.0, nl_n=0, dmass=0.0))
        if mc > cur["mod_cos"]:
            cur.update(mod_cos=mc, mod_n=mn, dmass=round(ma[2] - mb[2], 4))
        if mn >= 6 and mc > cur["mod6"]:
            cur["mod6"] = mc
        if nc > cur["nl_cos"]:
            cur.update(nl_cos=nc, nl_n=nn)
    return pd.DataFrame([dict(a=k[0], b=k[1], **v) for k, v in best.items()])


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(y, score)), 3) if 0 < y.sum() < len(y) else None


def precheck():
    q = pd.read_parquet(B / "bench_comp/test.parquet")
    ans = pd.read_csv(B / "bench_answers.csv").set_index("molecule_id")
    pairs = spectral_pairs(q)
    fps = {m: fp(s) for m, s in ans.normalized_smiles.items()}
    scf = {m: scaffold(s) for m, s in ans.normalized_smiles.items()}
    ids = list(ans.index)
    allp = pd.DataFrame(list(combinations(sorted(ids), 2)), columns=["a", "b"])
    allp["tani"] = [DataStructs.TanimotoSimilarity(fps[a], fps[b]) for a, b in zip(allp.a, allp.b)]
    allp["same_scaffold"] = [scf[a] is not None and scf[a] == scf[b] and scf[a] != "c1ccccc1"
                             for a, b in zip(allp.a, allp.b)]
    P = allp.merge(pairs, on=["a", "b"], how="left").fillna(
        dict(mod_cos=0.0, mod_n=0, mod6=0.0, nl_cos=0.0, nl_n=0, dmass=0.0))
    P["cls_a"], P["cls_b"] = P.a.map(ans.cls), P.b.map(ans.cls)
    n = len(ids)

    def frac(mask):
        m = P[mask]
        return round(len(set(m.a) | set(m.b)) / n, 4)

    out = {"n_molecules": n, "n_pairs": len(P), "n_pairs_same_mode_scored": int((P.mod_n > 0).sum())}
    out["partner_fraction_structure"] = {f"tani>={t}": frac(P.tani >= t) for t in (0.4, 0.5, 0.6, 0.7)}
    out["partner_fraction_structure"]["same_scaffold"] = frac(P.same_scaffold)
    out["partner_fraction_spectral"] = {f"modcos>={c},n>=6": frac(P.mod6 >= c) for c in (0.6, 0.7, 0.8)}
    out["partner_fraction_spectral"]["nlcos>=0.6,n>=4"] = frac((P.nl_cos >= 0.6) & (P.nl_n >= 4))
    out["partner_fraction_both"] = {
        f"modcos>={c},n>=6 & tani>={t}": frac((P.mod6 >= c) & (P.tani >= t)) for c in (0.6, 0.7) for t in (0.4, 0.5)}
    out["partner_fraction_both"]["modcos>=0.7,n>=6 & same_scaffold"] = frac((P.mod6 >= 0.7) & P.same_scaffold)
    # predictiveness of spectral similarity for structural relatedness (pairs)
    y = (P.tani >= 0.5).astype(int)
    pred = {"base_rate_tani>=0.5": round(float(y.mean()), 4), "n_related_pairs": int(y.sum()),
            "auc_modcos_n6": auc(P.mod6, y), "auc_modcos": auc(P.mod_cos, y), "auc_nlcos": auc(P.nl_cos, y),
            "spearman_modcos_tani": round(float(P[["mod6", "tani"]].corr("spearman").iloc[0, 1]), 3)}
    for c in (0.6, 0.7, 0.8):
        s = P.mod6 >= c
        pred[f"modcos>={c},n>=6"] = {"pairs": int(s.sum()), "precision_tani>=0.5": round(float(y[s].mean()), 3)
                                     if s.any() else None, "recall": round(float(y[s].sum() / max(y.sum(), 1)), 3),
                                     "median_tani": round(float(P.tani[s].median()), 3) if s.any() else None}
    out["predictiveness"] = pred
    # headroom: molecules with a detectable related partner whose answer is at ranks 2-25 in merged477
    rr = pd.Series(json.load(open(B / "bench_results.json"))["merged477"]["rr"])
    rel = P[(P.mod6 >= RULE["spec_cos"]) & (P.tani >= RULE["tani"])]
    mol = sorted(set(rel.a) | set(rel.b))
    out["detectable_related_molecules"] = {
        "n": len(mol), "by_class": ans.loc[mol].cls.value_counts().sort_index().to_dict(),
        "answer_rank1": int((rr[mol] == 1).sum()), "answer_rank2_25": int(((rr[mol] > 0) & (rr[mol] < 1)).sum()),
        "answer_missing": int((rr[mol] == 0).sum())}
    # extrapolation to a 400-molecule test (independent-pairs approximation)
    p_pair = ((P.mod6 >= 0.7) & (P.tani >= 0.5)).mean()
    out["extrapolated_fraction_n400"] = round(float(1 - (1 - p_pair) ** 399), 4)
    loose = P[(P.mod_cos >= 0.5) | (P.tani >= 0.4) | (P.nl_cos >= 0.5)]
    loose.round(4).to_csv(ART / "58_pairs.csv", index=False)
    return out


def boost():
    """Pre-registered guarded boost on the merged477 bench lists (see RULE), plus two diagnostics."""
    P = pd.read_csv(ART / "58_pairs.csv")
    res = json.load(open(B / "bench_results.json"))
    cls = pd.Series(res["classes"]).astype(int)
    sub = pd.read_csv(LISTS)
    L = {m: s.split(";")[:RULE["max_rank"]] for m, s in zip(sub.molecule_id, sub.smiles)}
    ct = pd.read_parquet(B / "bench_candidate_truth.parquet")
    truth = ct[ct.is_answer].groupby("molecule_id").smiles.apply(set)
    ev = pd.read_parquet(B / "frozen_candidate_evidence.parquet").set_index(["molecule_id", "smiles"]).library_max
    ans = pd.read_csv(B / "bench_answers.csv").set_index("molecule_id").normalized_smiles
    fcache = {}

    def F(s):
        if s not in fcache:
            fcache[s] = fp(s)
        return fcache[s]

    sp = P[(P.mod6 >= RULE["spec_cos"])]
    nbr = {}
    for a, b in zip(sp.a, sp.b):
        nbr.setdefault(a, []).append(b); nbr.setdefault(b, []).append(a)
    conf = {m: float(ev.get((m, l[0]), 0.0)) >= RULE["conf_libmax"] for m, l in L.items()}

    def apply(anchor_of):
        new, log = {}, []
        for m, lst in L.items():
            anchors = [f for f in (anchor_of(o) for o in nbr.get(m, [])) if f is not None]
            boosted = []
            for i in range(1, len(lst)):
                f = F(lst[i])
                if anchors and f is not None:
                    t = DataStructs.BulkTanimotoSimilarity(f, anchors)
                    if any(RULE["tani"] <= x < 1.0 for x in t):
                        boosted.append(i)
            score = {i: float(i) for i in range(1, len(lst))}
            for i in boosted:
                score[i] = max(1.0, i - RULE["shift"]) - 0.5  # never reaches rank 1; ties go to the boosted one
            new[m] = [lst[0]] + [lst[i] for i in sorted(score, key=lambda i: (score[i], i))]
            log.append(dict(molecule_id=m, n_anchors=len(anchors), n_boosted=len(boosted)))
        return new, pd.DataFrame(log).set_index("molecule_id")

    def rr_of(lists):
        return pd.Series({m: next((1 / (i + 1) for i, x in enumerate(l) if x in truth.get(m, set())), 0.0)
                          for m, l in lists.items()})[cls.index]

    r0 = rr_of(L)
    assert abs(r0.mean() - pd.Series(res["merged477"]["rr"])[cls.index].mean()) < 1e-12

    def score(new, log):
        r1 = rr_of(new)
        d = r1 - r0
        rng = np.random.default_rng(0)
        bs = np.array([d.values[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
        o = {"mrr_base": round(r0.mean(), 4), "mrr_boost": round(r1.mean(), 4), "delta": round(d.mean(), 5),
             "p_better": round(float((bs > 0).mean()), 4),
             "ci95": [round(float(x), 5) for x in np.percentile(bs, [2.5, 97.5])],
             "lists_with_anchor": int((log.n_anchors > 0).sum()), "lists_touched": int((log.n_boosted > 0).sum()),
             "candidates_boosted": int(log.n_boosted.sum()), "lists_changed_rr": int((d != 0).sum()),
             "rank1_changes": int(sum(L[m][0] != new[m][0] for m in L))}
        pc = {}
        for c in (1, 2, 3):
            x, y = r0[cls == c], r1[cls == c]
            pc[c] = {"n": int(len(x)), "up": int((y > x).sum()), "down": int((y < x).sum()),
                     "down_from_rank2": int(((x == 0.5) & (y < x)).sum()), "delta": round(float((y - x).mean()), 4)}
        o["per_class"] = pc
        o["mix_delta"] = round(sum(MIX[c] * pc[c]["delta"] for c in MIX), 5)
        return o, r1

    out = {"rule": RULE, "n_confident_anchors": int(sum(conf.values()))}
    new, log = apply(lambda o: F(L[o][0]) if conf.get(o) else None)
    out["primary"], r1 = score(new, log)
    out["diag_no_conf_gate"], _ = score(*apply(lambda o: F(L[o][0]) if o in L else None))
    out["diag_oracle_anchor"], r_or = score(*apply(lambda o: F(ans[o]) if o in ans.index else None))
    log.assign(cls=cls, rr_base=r0, rr_boost=r1, rr_oracle=r_or).to_csv(ART / "58_boost_per_list.csv")
    return out


if __name__ == "__main__":
    f = ART / "58_summary.json"
    summ = json.load(open(f)) if f.exists() else {}
    summ[PHASE] = precheck() if PHASE == "precheck" else boost()
    json.dump(summ, open(f, "w"), indent=1, default=int)
    print(json.dumps(summ[PHASE], indent=1, default=int))
