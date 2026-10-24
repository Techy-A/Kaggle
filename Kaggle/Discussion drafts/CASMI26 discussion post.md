# Draft: CASMI26 discussion post

**Suggested title:** From 0.455 to 0.465 in a week: what we copied, what we changed, what failed (and what we think is left)

*Draft for review. Replace the bracketed links before posting. Nothing here has been posted.*

---

## TL;DR

- We went from 0.455 to **0.465** in a week. Almost all of it came from public notebooks (credited below). **Our own contribution is small and mostly negative results**, and we think those are worth sharing.
- Of three changes we tried on top of the 0.464 base: **depth 7 cost −0.008**, a **candidate-level PubChem gate + "features-frozen" generation scored 0.465 (+0.001)**. We can't separate the two halves of that stack yet.
- Public reruns of identical code were deterministic in 32 of 33 cases in dariushafshar's table, so a single leaderboard test is a fairly clean measurement.

## Our progression (public score of each submission)

| Date | Submission | Score |
|---|---|---|
| 18-19 Oct | exact copies of public 0.454 / 0.455 / 0.456 notebooks | 0.454 / 0.455 / 0.456 |
| 24 Oct | exact copy of dmitriigluzdov's *The Data Behind Spectra to Structures* (v20/v24) | 0.464 |
| 24 Oct | same + engine-2 `max_depth` 6 → 7 | **0.456** (−0.008) |
| 24 Oct | same + candidate-level PubChem gate + features-frozen gen20 | **0.465** |

## What the field is doing (from dariushafshar's public-chain table, 481 versions)

The public leaders are one lineage, and each step is one or two changes:

1. **Analog propagation baseline** (prvsiyan) and **Ahmed Berat Özer's engine** (engine 1: candidate generation + fingerprint/analog ranker; engine 2 rankers).
2. **Forward-model re-ranking** (GLACIER / ICEBERG spectrum simulators). The big repeatable steps (+0.008 each) all came from **giving the simulators evidence on more molecules**: N1A (rebuilding [M+NH4]+ and [M−H2O+H]+ spectra as [M+H]+ proxies) and the N1 challenger (keep proxy GLACIER scores where ICEBERG also scores and re-rank by both).
3. **PubChem promotion**: a PubChem proposal can be promoted to rank 1 on popularity and fingerprint margin; thresholds (rel > 250 → 150) bought +0.006 once.
4. **Popularity priors, LOTUS/NPAtlas/ChEBI/LMSD pools, PubChemLite CC0** as extra candidate sources.
5. Roughly 20 further tweaks (gate thresholds, a sixth PubChem slot, LOTUS weight, popularity weights) were **flat (±0.000)** once on the 0.464 base.

## What was inside our notebooks

Our uploaded notebooks are **private pinned copies** of public work, credited in a cell at the top. The only code we changed is below.

1. **Exact copy of the 0.464 base**: no code change.
2. **+ depth 7** (two lines: the engine-2 GBM `max_depth=6` → `7`). Earlier it had given +0.003 / +0.004 on two older bases. On this base it lost 0.008, so we think the rankers here are already tuned around depth 6.
3. **+ candidate-level PubChem gate.** Per dariushafshar's notes, the 0.465 frontier (sanpier's version) fixes a real inconsistency: after the ICEBERG/GLACIER re-order, the PubChem candidate at rank 1 can differ from the one whose channel-level S / popularity / margin the gate used. We recompute the gate on the **promoted candidate's own evidence**. Thresholds are unchanged. It also writes an audit CSV of which promotion decisions changed.
4. **+ features-frozen generation ("genfix").** Engine 1's ranker uses **list-relative features** (ranks, z-scores, gap to best, softmax, formula share). Adding generated candidates re-scores the originals, and in our measurement reshuffled their top 5 in roughly half of molecules. Genfix keeps the original engine-1 run exactly, runs a second wider generation (20 analog parents, 100 per parent, 500 total) and inserts only structures the first run lacks, scoring them on the first run's scale minus a penalty (δ = 2). The original candidates cannot be re-scored.

## What we measured (a bench caveat first)

On a "thin class-1" bench (class-1 answers keep only 1-5 library spectra) the old ungated gen20 pushed the correct answer off rank 1 in **18 of 270** molecules (6.7%); genfix did so in **0 of 270**. This is the first bench of ours that sees the failure that cost ungated gen20 −0.010 on the leaderboard. **None of our benches is calibrated to the leaderboard**, so we treat this as a safety check and not as a forecast.

## Would we reach gold?

Honest view: gold (about 0.471) is roughly one more molecule on the public set (one molecule at rank 1 ≈ +0.0076), but we expect to **fall short with what we have**. The remaining levers we can see, in order:

1. Forward-model evidence coverage: extend ICEBERG/GLACIER scores to the molecules that still have none. It is the only change type that repeated +0.008.
2. Retraining the engine-1 ranker with source flags; this is the real class-3 lever but needs the unpublished simulation (10-40 CPU hours per setting).
3. A neural re-rank inside the top 25 (sanpier's add-on).

We would put our odds of reaching 0.471 this month at **low** and would be glad to be wrong.

## Questions for others

- Has anyone separated the gate fix from other changes on the 0.465?
- Has anyone measured coverage of ICEBERG/GLACIER scores by class?
- Anyone see a depth change help on this base?

## Credits

prvsiyan (analog propagation), Ahmed Berat Özer (engine, v7c/v15), Igor Zharov and Seyit Kaan Güneş (E006, N1/N1A), Dmitrii Gluzdov (*The Data Behind Spectra to Structures*), Saner Turhaner / sanpier (candidate-level gate, 0.465), dariushafshar (the public-chain table). Licences: the attributed-assets dataset includes CC BY-NC-SA components; the host's 17 Oct note allows open-sourced non-commercial data.

*[Links: our notebooks, once made public]*
