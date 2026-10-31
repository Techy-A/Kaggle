"""56: diagnose why casmi26-bench477 orders the variants in reverse of the public leaderboard.

What: from the bench477 outputs (/home/user/work/out_bench477), (1) per-molecule and per-class rank changes along
baseline0465 -> V1 -> V2 -> CUSTOM -> merged477 (rank-1 gains/losses, gains/losses inside 2-25); (2) neural model
health on the bench (neural_ok, score distribution, where the answer sits in neural order vs baseline order);
(3) bench queries vs the hidden test (adduct, mode, instrument, spectra per molecule, precursor m/z);
(4) a resampling model: how often a 125-molecule public sample shows +0.012 for merged477 vs baseline when the
bench's own per-molecule change distribution (or a zero-mean version of it) is true.
Why: decide whether the public gains are real (bench unrepresentative/leaky) or public noise (bench closer to private).
Output: artifacts/56_summary.json, artifacts/56_per_molecule.csv. CPU only, seconds.
Usage: python enveda-casmi26/56_bench_reversal_diagnosis.py [bench_dir] [test_parquet]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/user/work/out_bench477")
TEST = sys.argv[2] if len(sys.argv) > 2 else "/home/user/enveda_data/test.parquet"
ART = Path(__file__).resolve().parent / "artifacts"
CHAIN = ["baseline0465", "V1", "V2", "CUSTOM", "physical_V2", "merged477"]
MIX = {1: 0.28, 2: 0.20, 3: 0.52}

res = json.load(open(B / "bench_results.json"))
cls = pd.Series(res["classes"])
rr = pd.DataFrame({v: pd.Series(res[v]["rr"]) for v in CHAIN}).loc[cls.index]
rr["cls"] = cls
out = {}


def moves(a, b):
    """Per-class counts of rank changes from variant a to b."""
    d = {}
    for c, g in rr.groupby("cls"):
        x, y = g[a], g[b]
        d[int(c)] = {
            "n": len(g), "changed": int((x != y).sum()),
            "top1_gain": int(((y == 1) & (x < 1)).sum()), "top1_loss": int(((x == 1) & (y < 1)).sum()),
            "in25_up": int(((y > x) & (y < 1)).sum()), "in25_down": int(((y < x) & (x < 1)).sum()),
            "delta_mrr": round(float((y - x).mean()), 4),
        }
    d["mix_delta"] = round(sum(MIX[c] * d[c]["delta_mrr"] for c in MIX), 4)
    d["all_delta"] = round(float((rr[b] - rr[a]).mean()), 4)
    return d


pairs = list(zip(CHAIN[:-1], CHAIN[1:])) + [("CUSTOM", "merged477"), ("baseline0465", "merged477")]
out["moves"] = {f"{a}->{b}": moves(a, b) for a, b in pairs}

# 2. neural health
nf = pd.read_parquet(B / "neural_candidate_features.parquet")
ans = pd.read_csv(B / "bench_answers.csv")[["molecule_id", "metric_key"]].rename(columns={"metric_key": "ans"})
nf = nf.merge(ans, on="molecule_id")
nf["is_ans"] = nf.metric_key == nf.ans
nf["nrank"] = nf.groupby("molecule_id").neural_score.rank(ascending=False, method="min")
a = nf[nf.is_ans].copy()
a["cls"] = a.molecule_id.map(cls)
z = nf.groupby("molecule_id").neural_score.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
nf["z"] = z
err = json.load(open(B / "neural_candidate_features.parquet.errors.json"))
out["neural"] = {
    "rows": len(nf), "neural_ok_rate": round(float(nf.neural_ok.mean()), 4), "errors": len(err),
    "score_quantiles": nf.neural_score.quantile([.05, .25, .5, .75, .95]).round(3).tolist(),
    "cos_quantiles": nf.neural_cos.quantile([.05, .5, .95]).round(3).tolist(),
    "answer_listed_per_class": a.groupby("cls").size().to_dict(),
    "answer_top1_baseline_vs_neural": {int(c): [int((g.baseline_rank == 1).sum()), int((g.nrank == 1).sum())]
                                       for c, g in a.groupby("cls")},
    "answer_mean_rank_baseline_vs_neural": {int(c): [round(g.baseline_rank.mean(), 2), round(g.nrank.mean(), 2)]
                                            for c, g in a.groupby("cls")},
    "answer_neural_z_median": {int(c): round(float(nf.loc[g.index, "z"].median()), 2) for c, g in a.groupby("cls")},
    "neural_better_worse_tied_vs_baseline": [int((a.nrank < a.baseline_rank).sum()),
                                             int((a.nrank > a.baseline_rank).sum()),
                                             int((a.nrank == a.baseline_rank).sum())],
}

# 3. representativeness
q = pd.read_parquet(B / "effective_query_spectra.parquet")
t = pd.read_parquet(TEST)


def profile(d):
    spm = d.groupby("molecule_id").size()
    return {"spectra": len(d), "molecules": int(d.molecule_id.nunique()),
            "adduct": d.adduct.value_counts(normalize=True).round(3).to_dict(),
            "mode": d.ionization_mode.value_counts(normalize=True).round(3).to_dict(),
            "instrument": d.instrument_type.value_counts(normalize=True).round(3).to_dict(),
            "spectra_per_mol_mean": round(float(spm.mean()), 2), "single_spectrum_mols": round(float((spm == 1).mean()), 3),
            "precursor_mz_q": d.precursor_mz.quantile([.1, .5, .9]).round(1).tolist()}


out["profile"] = {"bench": profile(q), "test_visible": profile(t)}

# 4. public noise model: 125-molecule draws of merged477 - baseline per-molecule deltas
rng = np.random.default_rng(56)
d = (rr["merged477"] - rr["baseline0465"]).values
w = rr.cls.map(MIX).values / rr.cls.map(rr.cls.value_counts(normalize=True)).values  # reweight to 28/20/52
p = w / w.sum()
sims = {}
for name, dd in {"bench_as_is": d, "zero_mean": d - np.average(d, weights=p)}.items():
    m = np.array([dd[rng.choice(len(dd), 125, p=p)].mean() for _ in range(20000)])
    sims[name] = {"mean": round(float(m.mean()), 4), "sd": round(float(m.std()), 4),
                  "P(>=+0.012)": round(float((m >= 0.012).mean()), 4), "P(>=+0.010)": round(float((m >= 0.010).mean()), 4)}
sims["changed_share"] = round(float(np.average(d != 0, weights=p)), 3)
out["public_noise_model"] = sims

ART.mkdir(exist_ok=True)
rr.assign(rank_base=1 / rr.baseline0465.replace(0, np.nan), rank_merged=1 / rr.merged477.replace(0, np.nan)) \
  .to_csv(ART / "56_per_molecule.csv", index_label="molecule_id")
json.dump(out, open(ART / "56_summary.json", "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
