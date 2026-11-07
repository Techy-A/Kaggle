---
tags: [kaggle, casmi26, competition]
competition: enveda-CASMI26-molecule-id-mass-spectra
metric: MRR@25 on InChIKey first block
deadline: 2026-12-14
merge_deadline: 2026-12-07
---
# Enveda CASMI 2026: molecule ID from MS/MS

Back to [[Kaggle Home]]. Full experiment log: `enveda-casmi26/README.md` in the repo.

## Leaderboard history
| Date | Run | Public LB |
|---|---|---|
| 11 Oct | Cosine library search | 0.150 |
| 11 Oct | Fingerprint MLP v1 + COCONUT | 0.176 |
| 12 Oct | v2 + popularity prior | 0.222 |
| 13 Oct | Fork of the 0.440 public pipeline | 0.437 |
| 14 Oct | Own engine: ranker + fragmentation + analogs | 0.256 |
| 14 Oct | Own engine with GPU v3 model | 0.272 |
| 14 Oct | **Fork + own-engine merge (v2)** | **0.438** (best) |
| 15 Oct | Fork + v3 merge (beta 0.5 / 1.0) | 0.436 / 0.430 |
| 15 Oct | Unpinned Dual-Band + merge (/ + generation) | 0.427 / 0.426 |
| 15 Oct | Kino: E007 + LambdaRank + tail fill | 0.440 |
| 16 Oct | Kino: unchanged V22 copy (rel>150, no E2R3) | 0.451 (confirms V22 is stable) |
| 16 Oct | **Exact pinned copy of Antriksh Arya's 0.443** | **0.443** (team best; inside the bronze zone, ~rank 253–308 of 3,397, bronze line 0.442 at rank 339, silver 0.446) |
| 16 Oct | Exact pinned copy of Ahmed's v7c | 0.441 |
| 16 Oct | 0.443 + agreement-only promotion | 0.439 |
| 16 Oct | **Exact pinned copy of easonyanyan's V22** | **0.451**: team best, ~rank 32–54 of 3,407; silver line 0.446 (rank 170), gold ~rank 16 |

## Public notebooks (real scores)
- 0.443: Antriksh Arya, v7c + engine-2 rank-2 insert (pinned image `2757e0…`)
- 0.443: saddamaditya, forensics-lab apex dossier
- 0.441: Ahmed v7c fraggap (pinned 3.12 image `37c64f…`), Dual-Band, MetFrag-lite, …

## Final picks (plan)
- 57257429: V22 copy, 0.451
- 57248900: 0.443 copy, a different lineage, as a hedge

## In flight
- 16 Oct: 0.443 base / + merge / + merge + generation / agreement-only promotion. See [[What Works]] for the agreement idea once it's measured.

## Lessons
See [[What Works]] and [[What Does Not Work]] (tag `#casmi26`).


## Standing, 16 Oct 17:00 UTC
- Rank 63/3427 at 0.451, inside **silver** (top 5% = rank 171, cutoff 0.447). Gold needs top 17 (0.456). Bronze cutoff is 0.443 (rank 342).
- The V22 variants (rel100, E2R3) haven't been submitted yet.

## Publishing rule (user, 16 Oct)
- Make the 0.451 V22 pinned copy public **only once the team is at 0.453+**. Never publish the current best.

## Council evidence, 17 Oct (see the council report when the verdict is in)
- **rel100 has negative public evidence:** seyitkaangunes's sweep on the E006 base gave rel>250 0.442, rel>150 0.446, rel>100 0.444. Held until the user decides.
- **E2R3:** no direct LB evidence. Engine 2 rescues only about 3 public molecules (antriksharya probes). Held.
- **Depth 7 is the lead:** the v14seeds/Gluzdov "seed bag" seeds are dead code, and only max_depth 6->7 in the engine-2 HGB rankers runs. It gained +0.003 (0.441->0.444) and +0.004 (0.446->0.450) on two bases, and no 0.451 notebook has it. Built as push_v22_depth7 (kernel aryanawasthiteykik/casmi26-v22-depth7, not pushed yet).
- **Noise:** the public LB is about 124 molecules (one rank-1 hit is about 0.008), and GPU reruns of the same code differ by about ±0.003.
- **Dead ends measured publicly:** ungated FRIGID rank-2 (-0.009), JESTR (-0.022), CFM-ID LOTUS (0 to -0.018), stepped-CE GLACIER (-0.004), MSEntropy (0), ChEBI+LIPID MAPS in another pipeline (-0.005).

## Council verdict, 17 Oct (judge after a 2-round debate; the user is the jury)
| Motion | Ruling | P(beats 0.451) |
|---|---|---|
| V22 + engine-2 max_depth 7 (single change) | **submit next** | 0.55 |
| V22 + rel100 | reject (public sweep peaks at 150) | 0.20 |
| E2R3 | later, only on a scored depth-7 base | 0.25 |
| Bigger NP catalogue (NPAtlas/LOTUS) | later; first a CPU count of new structures | 0.25 |
| Real seed bag | reject (seed models highly correlated) | – |
| Gated FRIGID class-3 channel | research track, offline only | 0.15 |
| Zenodo Enveda-180 extras, ICEBERG NIST atlas, GLACIER fine-tune | reject (no gain expected / NIST disqualification risk / weeks of work) | 0.15 |
- Score rule for depth 7: >= 0.452 means it becomes the new base and a final pick; 0.451 is a tie and the user decides; <= 0.450 means stop the depth line, no re-runs for noise.
- The local dummy test.parquet is all enveda-180 library hits, so it is useless as a validation set.

- **17 Oct: Kino's V22 + rel100 scored 0.445** (-0.006 vs V22). The council predicted it (20% chance to beat 0.451). The PubChem gate stays at 150. #lb-proven
- **17 Oct: Kino's V22 + rank-3 insert (E2R3) scored 0.450** (-0.001, a noise-level tie). The council predicted about 0 (25%). Shelved; retest only on a depth-7 base.

## Jury decisions (user, 17 Oct)
1. **GPU budget:** about 15 h of our 30 h a week go to research (bench + the novel-molecule track). The rest goes to safe runs and copies. Kino's 30 h covers NTL copies where possible.
2. **Strategy: play hard for a month, then play safe.**
   - Until about **17 Nov**: experimental submissions are allowed when they're backed by the bench or the council (still 3 team slots a day, 2 kept for Kino).
   - After 17 Nov: sure-shot only, and the picks get locked towards the mid-December deadline.
3. Still open: if depth 7 ties at 0.451, does it replace the 0.443 as final pick 2? The judge says yes.
- **17 Oct: V22 + depth 7 scored 0.451**, an exact tie with V22. The council gave it 55% to beat 0.451. Under the judge's rule a tie is the user's call on replacing the 0.443 as final pick 2 (the judge recommends replacing it). Depth 7 changes engine 2's top-1 on about 9% of molecules, so it is a genuinely different submission with the same public score: a useful hedge for the private split.
- **17 Oct: NPAtlas coverage check (council motion M4 gate).** NPAtlas has 36,454 natural products, 34,587 of them in the test mass range. Only **485** are missing from V22's v4 pool (train + COCONUT) by the metric's own identity: 6,175 differ by raw InChIKey, but nearly all of those are tautomer or representation duplicates. Some of the 485 are also reachable through the PubChem channel. **Verdict: an NPAtlas catalogue is not worth building.** Candidate *coverage from databases* is basically saturated, so the gap is molecules that are in no database at all.

## Bench run 2 (17 Oct): first error breakdown
V22 on 250 np-examples, answers hidden per class.

**MRR by stage:**

| Stage | MRR |
|---|---|
| after fusion | 0.525 |
| E006 control | 0.529 |
| E006 output | 0.528 |
| final (+E2R2) | **0.519** |

**Per class, final:**

| Class | MRR | Top-1 | In top 25 |
|---|---|---|---|
| 1 | 0.967 | 66/70 | 70/70 |
| 2 | 0.849 | 39/50 | 50/50 |
| 3 | **0.151** | **7/130** | 51/130 |

**Loss buckets:**
- *missing* is 79 molecules (all class 3): +0.316 MRR if all were fixed. 42 of them are near misses (nearest candidate at Tanimoto ≥ 0.7).
- *ranked* is 59 molecules: +0.165.
- *fusion* is 0.

**Class-3 recall by source:**
- engine 1 base: 46 of 130, nearly all *generated* derivatives, because the answers are hidden from the pool;
- engine 2: 6;
- PubChem channel: 0.

**Leak caveat:**
- Engine 1 alone puts the class-3 answer first 38 times (MRR 0.31 vs 0.15 final), and on class 2 it scores 0.95 vs 0.85 final. So on the bench, every later stage seems to *hurt*.
- That contradicts the LB, where each layer gained.
- Cause: **all 250 np structures have other-library spectra in train** (median 168 each). Engine 1's FPNet A, v4b ranker and fe models have seen them, while engine 2 was given leak-free models. The bench therefore favours engine 1.
- Engine comparisons and fusion-weight conclusions are **not valid** on this bench. Within-pipeline post-processing replays are still to be calibrated (bench v3).

**Fix in progress:** a leak-free pseudo-test from the Enveda-180 Zenodo release, using structures absent from train.parquet (about 800 timsTOF molecules). They're drug-like, not natural products, but no model has seen them.

## Near-miss analysis (17 Oct, bench run 2, structure-only, so it isn't affected by the model leak)
Of the 130 class-3 molecules:
- **100** are reachable by engine 1's *own* derive rulebook (≤2 steps) from one of the top-6 most similar training structures (94 have such a parent at Tanimoto ≥ 0.7);
- engine 1 actually generated only **46**.

Of the 84 it didn't generate:
- **55 are reachable from a better parent**. This is a *parent-selection* problem: engine 1 uses only its top-6 spectral analogs as parents (gen_n_analog=6, ≤60 products per parent, ≤150 in total);
- 29 are unreachable (rulebook gap or far parent; the formula deltas are scattered).

Of the 46 generated, 39 aren't ranked first.

**Next:** CPU-only Kaggle sweep (`aryanawasthiteykik/casmi26-gen-sweep`, no GPU quota) of gen_n_analog 6/12/20/30 with larger product caps. It measures generation recall and engine-1 rank per class.

## Generator sweep (17 Oct, CPU, engine 1 only, np pseudo-test)
| Setting | Class-3 hit@25 (of 130) | Class-3 top-1 | Class-3 MRR (engine 1) | Class 1 MRR | Class 2 MRR | CPU time |
|---|---|---|---|---|---|---|
| base (6 parents / 60 / 150) | 46 | 38 | 0.312 | 0.977 | 0.893 | 24 min |
| A12 (12 / 80 / 300) | 59 | 47 | 0.388 | 0.977 | 0.893 | 31 min |
| **B20 (20 / 100 / 500)** | **69** | **51** | **0.430** | 0.955 | 0.891 | 41 min |
| C30 (30 / 120 / 800, min_sim 0.2) | 72 | 52 | 0.440 | 0.955 | 0.890 | 54 min |

- Generation recall is structural, so it's trustworthy: **+23 novel answers generated with B20**.
- Top-1 counts are inflated by the engine-1 FPNet leak, so the real gain is smaller.
- Classes 1/2 cost almost nothing: one class-1 molecule drops from rank 1 to rank 2.
- B20 takes about 1.7× the engine-1 time, which is fine within 9 h.
- **Next:** V22 + B20 (`aryanawasthiteykik/casmi26-v22-gen20`, cell 19 only). The first push was blocked by the 2-GPU-session limit while the benches run.

## Bench v3 (17 Oct, done 8 pm IST)
- Main pass identical to run 2 (final 0.519; class 3 0.151, 7 top-1), so the bench is **deterministic**.
- The calibration replays failed: a V22 cell overwrote the replay loop's `_spec` global (name clash), and the snapshot pickle failed on a pandas itertuples row.
- Both are fixed in build_bench.py (`_bq_*` names, per-key pickling).
- Next: `casmi26-bench-unseen-g20` = unseen pseudo-test + gen20 + fixed replays. Its main pass compared with unseen v1 gives an honest gen20 vs V22 comparison, and its replays give the calibration.

## Unseen bench v1 (17 Oct, 8:45 pm IST): V22 unchanged on 320 Enveda-180 structures absent from train (leak-free)
| Class | n | MRR | Top-1 | Hit@25 | Answer in the PubChem list | Lost in fusion |
|---|---|---|---|---|---|---|
| 2 (reachable only through PubChem) | 128 | **0.257** | 24 | 55 | 66 | **11** |
| 3 (hidden everywhere) | 192 | **0.005** | 0 | 3 | – | – |

- **Class 2 has a concrete leak point.** The PubChem channel *has* the answer for 66 of 128, but only 55 survive into the final 25. The other 11 are dropped by the gated merge, about +0.034 on the bench. A further 62 aren't in PubChem's top 25 at all.
- **Class 3 is hopeless on drug-like molecules:** 3 of 192 in the list, nearest candidates mostly at Tanimoto < 0.55. Engine 1's natural-product edit rules don't fit synthetic library chemistry. The real test is natural products (host), so this set is out of domain for generation but the right control for "does a change hurt class 2".
- **Next:** unseen-g20 (V22 + gen20 + fixed replays) checks that gen20 doesn't hurt class 2 and gives the calibration replays.
- **17 Oct: Kino re-ran depth 7 from his account and got 0.447**, against our 0.451 for the same notebook code. Either his copy isn't pinned to the original image (unpinned copies cost about 0.014 before), or **run-to-run noise is about ±0.004**. Either way, public-LB differences of 0.001–0.004 between variants are inside the noise.
- **User decision (17 Oct, 9:50 pm IST): 'yes, conditional'.** Submit V22 + gen20 at 5:35 am IST on 18 Oct, once, only if all of these hold:
  - the gen20 run is COMPLETE;
  - the unseen-g20 bench shows class-2 MRR ≥ 0.25 and class 3 not below V22;
  - the team is under 3 CASMI submissions that UTC day;
  - it hasn't been submitted before.

## Unseen-g20 bench (17 Oct, done 11:40 pm IST): V22 + gen20 on the leak-free unseen set, plus calibration replays
| Final list | V22 (unseen v1) | **V22 + gen20** |
|---|---|---|
| Class 2 MRR | 0.256 | **0.260** |
| Class 2 hit@25 | 55 | 56 |
| Class 3 MRR | 0.005 | **0.013** |
| Class 3 top-1 / hit@25 | 0 / 3 | 1 / 6 |

**gen20 helps, or at least doesn't hurt, on molecules no model has seen.** Both submission conditions pass.

**Calibration** (replays of cells 25–37 from the snapshot, on the gen20 base). replay_v22 equals the main pass exactly, so the replay is deterministic.

| Variant | Bench all | Bench c2 | Public LB |
|---|---|---|---|
| rel100 | 0.1137 | 0.265 | 0.445 (worse) ✗ |
| no_e2r2 | 0.1132 | 0.262 | – |
| **V22 (rel150)** | 0.1119 | 0.260 | **0.451** |
| E2R3 | 0.1090 | 0.254 | 0.450 (worse) ✓ |
| rel250 | 0.1047 | 0.241 | 0.445 (worse) ✓ |

**Verdict: partially calibrated (2 of 3 directions right).**
- The bench gets E2R3 and rel250 right but rel100 wrong.
- Reason: its class 2 is *all PubChem-only*, so loosening the PubChem gate always helps here. In the real test, many answers sit in the pool (class 1 and COCONUT class 2), and promoted PubChem guesses displace them.
- **So don't trust this bench for PubChem-gate decisions.** Use it for engine/generation changes and as a class-2 'do no harm' check.
- The snapshot was saved this time (381 MB), so later replays can run without a GPU.
- **17 Oct, 11:53 pm IST: V22 + gen20 submitted (57321705).** All conditions passed: unseen-g20 bench class 2 0.260 vs 0.256 and class 3 0.013 vs 0.005, and the team was at 4 of 5 that UTC day.

## gen20 result (landed 18 Oct, by 4:38 am IST): 0.441, worse
- **V22 + gen20 (57321705): public 0.441**, against V22's 0.451. The forecast was 0.451–0.452. That is −0.010, 2.5 times the ±0.004 noise floor, so it's a real loss, not noise.
- **Standing:**
  - Our best stays 0.451, rank 102 of 3,500 (silver).
  - Medal lines: gold (top 17) 0.459, silver (top 175) 0.451, bronze (top 350) 0.451.
  - Silver and bronze both sit at 0.451, so a large group of teams is tied with the public notebook score. Our silver rests on tie order, and any private reshuffle of the tie group decides it.
- **Why the bench was wrong (most likely):**
  - The unseen bench has **no class-1 molecules** (answers with library spectra), and its class 2/3 gains were tiny (class 2 0.260 vs 0.256, class 3 0.013 vs 0.005 on 320 molecules).
  - gen20 generates up to 500 candidates instead of 150, and those can push correct library and COCONUT answers out of the top ranks in the real test.
  - So the bench measured a small gain where gen20 helps and was blind to where it hurts.
- **Lessons:**
  - (1) A bench that leaves out a test class can't approve a change that touches every class. Add a class-1 slice before trusting the bench for generation changes.
  - (2) Gains below about 0.005 on the bench are not evidence.
  - (3) Final picks stay V22 (0.451) and V22 + depth 7 (0.451). **gen20 is dropped.**

## Bench breakdown of V22 (18 Oct, CPU, `bench/breakdown.py`)
Honest slices only: class 1 from the np bench (70 molecules; the structure keeps its other library spectra, which is what class 1 is), classes 2–3 from the leak-free unseen bench (128 + 192).

| Class | MRR | Top 1 | Ranked 2–25 | Dropped by fusion | Never a candidate |
|---|---|---|---|---|---|
| 1 | **0.967** | 66 | 4 | 0 | 0 |
| 2 | 0.257 | 24 | 31 | 11 | 62 |
| 3 | 0.005 | 0 | 3 | 0 | 189 (23 near misses, Tc ≥ 0.7) |

**Share of all lost points** (weighted by the test mix 28/20/52):
- **class-3 never a candidate: 76%**;
- class-2 never a candidate: 14%;
- class-2 ranked 2–25 or dropped by fusion: 8%;
- class 1: 1%.

**Reading:**
- **Class 1 is essentially solved.** Every point there is already banked, so any change that touches class 1 can only lose. That is exactly how gen20 lost 0.010.
- The bench's absolute level (0.33 at the test mix) is below the LB's 0.451. Its class 2 is PubChem-only, which is harder than the real test, so compare variants, not levels.

**Next candidate: gated gen20.**
- Use the wider generation only where engine 1's best library match is weak, i.e. likely class 3.
- That keeps class 1 untouched while still chasing the 23 class-3 near misses.
- Bench-testable once the class-1 + gen20 run (casmi26-bench-g20) confirms where gen20 hurt.

## Steps 2–4 (18 Oct)
- **Step 2, gated gen20** (`gensweep` per-molecule outputs, 170 np molecules, engine-1 lists). gen20 everywhere: class 1 0.977 → **0.955**, class 3 0.312 → 0.430.
  - **Gating at best library match < 0.7 keeps class 1 at 0.977 and class 3 at 0.383**, about 60% of the gain.
  - The class-3 gain is inflated here (leaky np bench); on the honest bench gen20's class-3 gain was tiny. Realistic LB effect: +0.000 to +0.003. Waiting on the full-pipeline bench (casmi26-bench-g20) before deciding.
- **Step 4, dedupe by metric key: already done.** 0 duplicate metric keys in 570 bench final lists. No gain available.
- **Step 3, learned fusion** (`bench/fusion_ltr.py`, running): a LightGBM lambdarank over the five E004 fusion inputs, with source ranks, flags and PubChem scores, against V22's hand-written fusion, out-of-fold on 390 honest molecules.
- **Step 3 result: learned fusion REJECTED, despite the bench.** Out-of-fold on 390 honest molecules:
  - class 1 0.967 → 0.951 (−0.016, 5 worse);
  - class 2 0.257 → **0.352** (+0.094);
  - class 3 flat;
  - +0.015 at the test mix, P = 0.98.
  - **Why it's rejected anyway:** every class-2 answer the pipeline finds on this bench comes only from the PubChem list (66 of 128; 0 from engine 2). So the model learned "trust PubChem more", the rel100 trap (bench up, LB −0.006). Real-test class-2 answers often come from the pool and COCONUT through engine 2.
  - **Rule:** no fusion or PubChem-weighting change is trusted on this bench until it has a class-2 slice whose answers live in engine-2 sources.

## Bench calibration with class 1: gen20 on the np bench (casmi26-bench-g20, done ~2:20 pm IST, 18 Oct)
| np bench, final list | Class 1 | Class 2 | Class 3 | All |
|---|---|---|---|---|
| V22 (run2) | 0.967 | 0.849 | 0.151 | 0.519 |
| V22 + gen20 | 0.963 | 0.850 | 0.216 | 0.552 |

- **Class 1 barely moved:** one molecule shifted rank, about −0.001 overall. So class-1 damage does not explain gen20's LB −0.010, and this morning's explanation is wrong.
- **The bench still calls gen20 a win (+0.033), the LB says −0.010.** So the bench is not calibrated for generation changes either.
- **Open causes:**
  - (a) noise: −0.010 is about 2.5× the floor;
  - (b) runtime budgets on the bigger hidden test (gen20 is slower). This can't be checked: Kaggle exposes only the commit-run log, not the hidden-test rerun log;
  - (c) the bench's class difficulty differs from the test's.
- **Decision: gated gen20 gets no slot.** There is no trustworthy evidence it helps.
- **Rule:** generation changes need a leaderboard A/B, not a bench verdict. The bench stays useful for diagnosis (where points are lost), not for approving submissions.
- **Copies:** the CASMI 0.454 public copy (guhongbin v459) was submitted at 2:26 pm IST (57347521), forecast 0.454.


## The docker image is worth more than the code (18 Oct, ~5:50 pm IST)
- **guhongbin v459 "e006 defaultimg" (0.454) has the same code as his 0.446 notebook** (E006 + PubChem promotion at rel > 150). Only the docker image changed: `…f6b083ad2b2f` (newer, also used by the NTL notebooks) instead of `…bf3fc647d461`. That's +0.008, about 2× the noise floor (the same code has scored 0.451 and 0.447 on two runs).
- v461 adds CC BY ChEBI on the new image and scores the same 0.454, so additions don't simply stack on it.
- **seyitkaangunes 0.452** (E006 + rel150 + N1A NH4/water-loss GLACIER + CC BY ChEBI) and **our V22 + depth7 (0.451)** both still run on the old image.
- **A/B on the LB (the user approved all 4 remaining slots today; Kino is busy):**
  - A: seyit's 0.452 code on the new image;
  - B: our V22 + depth7 on the new image.
  - Both are pushed (private) with auto-submit watchers. B waits for a free GPU session (limit 2).
- **If either lands at 0.455 or above, the image is the lever,** and the next submissions should port our own changes onto it.
- **Plus C: v459 + depth7, and D: v459 + gated gen20** (subagent port, diffs verified). Gated gen20 reruns engine 1 at 20 analogs only where lib_max < 0.7, falling back to the 6-analog list on error. Runtime is about 2× engine 1, close to ungated gen20, which completed on the LB.
- **Licence flag for the final selection:** every E006/V22-lineage notebook (ours included) mounts Ahmed Berat Özer's datasets (fpnet, glacier, iceberg, pubchem-tier, v2-pool, v3/v4b models). v459's markdown says those are **published as non-commercial**. Decide before choosing our two final CASMI entries.
- **0.454 copy result (landed by 6:50 pm IST, after ~4 h of hidden-test scoring): 0.454, as forecast.** It is our new best.
  - **Rank 110 of 3,525** (tie order; 50 teams sit on 0.454, 65 above).
  - Medal lines: gold (top 17) **0.460–0.461**; silver (top 176) **0.451**; bronze (top 352) 0.451. Top public score: 0.495.
  - We are in the silver zone, but only on tie order at the line's neighbourhood. Today's A–D A/B (new image) is the shot at clearing it.
- **GPU quota (6:52 pm IST, 18 Oct): 26.9 of 30 h used this week** (refreshes 24 Oct, 5:30 am IST). C (depth7) and D (gated gen20) are **held, not pushed**. Order of priority: NTL 0.559 copy > CASMI A > CASMI B > NTL ATLAS. C and D need Kino's quota (push dirs `/home/user/work/push_v459_depth7`, `push_v459_gated_gen20`).
- **New public top, 7:49 pm IST: 0.455, ahmedberatozer "casmi26-v14-pubchemlite".** It is his v13 (0.451) plus PubChemLite CC0 candidates in engine 2, and it runs on the **old** image.
  - Queued, in priority order: **E**, v14 on the new image (forecast 0.459), then its **exact copy** (0.455), then B.
  - E and the copy are a clean same-code test of the image.
  - The order is enforced by `scratchpad/casmi_queue.sh` (pushes one at a time behind the 2-session limit).
- **8:50 pm IST: handed to Kino.** Our GPU runs sat in Kaggle's queue for 3 h+ (a CPU probe ran instantly, so only GPU sessions were stuck). Kino runs from his account: v14 on the latest image (E), the v14 exact copy, and seyit's 0.452 on the latest image (A). Our queued copies were deleted (no quota used). B (our V22 + depth7 on the new image) has not run; one CASMI slot is left today.
- **~1 am IST, 19 Oct: Kino's GPU runs are stuck in the queue too**, so this is a Kaggle-wide T4 shortage, not our account. The runs stay queued and will submit into the 19 Oct slots once they finish. Today's 4 unused CASMI slots are lost.
- **5:51 am IST, 19 Oct: new public 0.456, guhongbin "v465 v15 pop015"** (ahmedberatozer's v15 = v14 + N1A GLACIER proxies, old image). Exact pinned copy pushed (`casmi26-0-456-pinned-copy`); it ran immediately and auto-submits. Forecast 0.456. It uses one of our CASMI slots today; Kino keeps two.
- **8:04 am IST, 19 Oct:** Kino's CASMI runs are in, 2 submitted (5:53 and 7:02 am IST, no description), both still pending, alongside our 0.456 copy.
- **0.456 copy result (landed by 10:26 am IST, 19 Oct): 0.456, exact. Our new best.** **Rank 69 of 3,570: silver** (top 178 = 0.455). Medal lines: gold (top 17) **0.463**, silver **0.455**, bronze (top 357) 0.451. 40 teams are above 0.456 and 39 tie it.
- **Kino's first run (submitted 5:53 am IST): 0.455.** Which of his three it is (v14 on the latest image, v14 exact copy, or seyit 0.452 on the latest image) is not labelled. If it is the v14 latest-image run, the image effect is **zero** on v14. His second run (7:02 am IST) is still pending.
- **Kino's second run (submitted 7:02 am IST): 0.455.** Both of Kino's runs scored 0.455 (the exact v14 copy is 0.455 by construction), so whichever of them ran on the latest image gained nothing. **The image effect does not carry over to the v14/v15 code** (unless it was the seyit-0.452-on-latest-image run, which would be +0.003). Our 0.456 copy stays the best.

## 19 Oct: depth7 and gated gen20 ported onto v465 (the public 0.456)
- Built by `enveda-casmi26/variants/port_v465.py`. Each variant differs from v465 only in the same lines as the v459 ports (checked with diff).
- **v465 + depth7** (`push_v465_depth7`, kernel aryanawasthiteykik/casmi26-v465-depth7). Forecast 0.456–0.458, bias-adjusted 0.456. Depth 7 gave +0.003 and +0.004 on two bases and tied on V22.
- **v465 + gated gen20** (`push_v465_gated_gen20`). Low priority: ungated gen20 lost 0.010 on the LB, and the bench can't see that loss.
- Our GPU quota is about used up until 24 Oct, 05:30 IST, so these need Kino's quota or slots.

## 19 Oct: bench calibration against known LB results (CPU; `bench/calibrate.py`, `bench/results/calibration.md`)
- **No bench and no class mix reproduces the LB ordering.** gen20 breaks every one of them. It scored +0.033 on the np bench, +0.007 on unseen and +0.004 on the honest mix, against −0.010 on the LB.
  - All signs come out right only at class 1 ≥ 56%, but that mix predicts V22 at about 0.54, not 0.451.
  - **The benches stay diagnostic only; none of them can approve a change.**
- **Why gen20 lost (mechanism):** engine 1's ranker features are relative to the whole candidate list (ranks, z-scores, gap to the best, softmax, formula share). Adding up to 350 generated candidates re-scores the candidates V22 already had, changing their top-5 order in 48–60% of molecules.
  - It isn't crowding: a new candidate never takes rank 1.
  - Correct class-1 answers lost rank 1 in 2 of 70 lists.
  - The bench can't see this, because bench answers have about 180 library spectra each against a median of 6 in the test-like world.
- **Fix to the bench:** add a "thin class-1" slice (keep only 1–5 library spectra per answer) and gate any list-composition change on how often the correct answer loses rank 1.
- **Fix to the method (H&M lesson):** compute list-relative features over the original, non-generated candidates only, so generated additions can't re-score them, and give the ranker source flags.
- The probability of −0.010 under the honest bench plus noise is about 1%. So either the test has something the bench lacks, or the public draw was unlucky.

## 24 Oct, 05:40 IST: GPU reset push
- Scan first: the new public **0.465** (sanpier, candidate-aware PubChem promotion) is **not reproducible**. Its blank input is required: it asserts on casmi_spectrum_model_v1.zip and selected_policy.json.
- Pushed and auto-submitting: the 0.464 exact copy (dmitriigluzdov) and 0.464 + depth 7 (2 CASMI slots, approved by the user for today).
- Leaderboard as of 23 Oct: 3,884 teams. We are rank 526 at 0.456. Silver 0.464, gold 0.471 (rank 17), top 0.504; 376 teams are at 0.464 or better.

## 24 Oct results
- **0.464 exact copy: 0.464**, as forecast. **0.464 + depth 7: 0.456 (−0.008).** Depth 7 hurts on this base: on the earlier bases it gave +0.003, +0.004 and a tie. The engine-2 rankers in this base are probably already tuned around depth 6. **Drop depth 7.**
- Leaderboard after the scores: 3919 teams, us rank 426 at 0.464; gold 0.471 (rank 17), silver 0.464 (rank 195), bronze 0.464 (rank 391).

## 24 Oct: features-frozen generation ("genfix") on the 0.464, thin class-1 bench (CPU, no GPU used)
- **Where the list-relative features live:** not in the notebook text. Engine 1 is Ahmed Berat Özer's v4n, loaded from `casmi26-attributed-v4g-reproduction-assets/casmi26-v4b-models/code` (cell 41). Cell 42 copies it and patches one line of `casmi/engine.py`. In `Engine.run`: `_rank_norm` / `_z` / `*_gap` / `*_grp_max` / `fz_softmax` / `fz_grp_ent` / `formula_share` / `n_formula` / `n_cand` / `n_gen` / `agree_*`. In `fe_v4/fams/*`: the `*_rank`, `*_gap`, `*_vs_lead` and `*_grp_*` family features. Then the cell-54 pool-popularity z over the top-60 list.
- **Ranker:** `ranker_v4b.pkl` (4 lambdarank boosters, 500 rounds, 160 features), trained offline by Ahmed on his engine-1 simulation `sim1x`, which is not published. Redefining the features, or adding is_generated / rank-within-source / n_sources_hit, needs that simulation rebuilt. That means engine-1 runs over thousands of leave-out training queries, about 4–7 s each on CPU (≈ 10–40 CPU-h per setting), with no guarantee of matching the original training distribution. **Not done.** The ranker already has is_gen / n_gen / gen_rank_fz / gen_parent_*. New source features would have no consumer.
- **Fix instead (features frozen):** `variants/port_genfix.py`. It keeps the 0.464 engine-1 run and list exactly. A second engine-1 run at gen20 (20 / 100 / 500) contributes only its new generated structures. Their score is mapped onto run 1's scale by quantile matching over the shared candidates, then penalised by δ. Each is inserted behind the originals that score higher, and the originals keep their order.
- **Identity:** the patch only adds lines (cell 54). On Kaggle, GENFIX=False gives a pickle-identical BASE to the original cell on 16 molecules. The offline rebuild matches the notebook's GENFIX=True lists on 16 of 16.
- **Bench** (`bench/build_genfix_bench.py`, kernel casmi26-genfix-bench, CPU, 83 min loop): np pseudo-test (70 / 50 / 130) plus 200 extra class-1 structures (enveda-180 timsTOF queries). Class-1 answers are thinned to 1–5 library spectra (median 3, was 181 / 12). Leak-free FPNet A.
  Result: `bench/results/genfix_bench.md`, made by `genfix_analyze.py`.
  - **Ungated gen20 reproduces the LB damage:** the correct thin class-1 answer loses rank 1 in **18 of 270 (6.7%)**, class-1 MRR −0.038. That is inside the 4–11% band calibration.md said would explain −0.010. The original-candidate top-5 changes in 29–56% of molecules.
  - **genfix δ = 2.0: 0 of 270 rank-1 losses (0%)**, class-1 MRR −0.0006, class 2 ±0, class 3 +0.083 (38 → 44 top-1), top-5 of the originals is never changed. δ = 1.5: same 0 / 270, c3 +0.093.
  - Cross-fitted δ (1.0 / 1.5): 3 / 270 losses (1.1%), mix +0.048, P(better) 1.00.
  - Class-3 gains are leak-inflated (the ranker / fe models probably saw the np structures). On the honest unseen bench, gen20's class-3 gain was +0.008.
- **Cost:** genfix is 2.5× engine-1 time on CPU. On the GPU commit, engine 1 takes ~4 s/molecule, so about +45–60 min on 400 hidden molecules. ICE input grows by ~2 candidates per molecule.
- **Push dir:** `/home/user/work/push_casmi464_genfix` (kernel casmi26-0-464-genfix, private, image …bf3fc647d461 pinned original, T4 preflight assert kept, max_depth 6, δ 2.0). **Not pushed** (user: Sunday).
- **Forecast:** 0.464–0.467, point 0.465. Class 1 is safe on the bench. The real gain is the honest class-3 share, about +0.000 to +0.004, under ±0.004 run noise. A loss would point to runtime or downstream fusion, not to re-scoring.
- **Licence flag:** the 0.464's main asset dataset (`casmi26-attributed-v4g-reproduction-assets`, which holds Ahmed's engine) is CC BY-NC-SA 4.0. That applies to the base and to this variant alike.
- 0.464 + depth 7 scored 0.456 (−0.008), so depth 7 is dropped on this base.

## 24 Oct, 11:40 IST: gate + genfix stack pushed (3rd CASMI slot today; the user asked to push today)
- `push_casmi_gate_genfix` (aryanawasthiteykik/casmi26-gate-genfix) = the 0.464 copy plus two changes:
  - **the candidate-level PubChem promotion gate** (the research agent's implementation of the change behind sanpier's 0.465: promotion uses the leader candidate's own S, popularity and fingerprint margin; thresholds unchanged; writes pubchem_promotion_audit.csv);
  - **the features-frozen gen20** (δ 2.0, max 60 new).
- The diffs compose exactly: stack vs genfix = the 38 gate lines; stack vs gate = the 87 genfix lines.
- Forecast 0.466 (0.464-0.468).
- Source of the gate idea: dariushafshar's public-chain table. Every +0.008 frontier step came from GLACIER/ICEBERG evidence coverage (N1A proxies, N1 challenger); gate tweaks on 0.464 gave +0.000.

## 24 Oct: research on getting above 0.466 (see `Study Guide/research_notes/CASMI above 0.466.md`)
- **Correction:** the 0.465 *is* reproducible. sanpier's wrapper embeds the scored pipeline (sanpier data-behind v2, a fork of dmitriigluzdov v16) as base64. Its only change is a candidate-level PubChem promotion gate. The neural add-on is unscored.
- Built, not pushed: `/home/user/work/push_casmi465_exact` (exact copy) and `/home/user/work/push_casmi464_candgate` (the gate on our v24 base, `variants/port_candgate.py`).
- Chain data (481 versions): identical reruns score identically (32 of 33 groups), so the ±0.004 "noise" came from unpinned copies. Steps are quantised at one molecule = 0.0076. The only big repeatable lever is forward-model coverage (N1 challenger, +0.008 ×3). Wider generation lost every time (−0.008 to −0.012). Prior and threshold tweaks are flat.
- Host, 17 Oct (discussion 750330): non-commercial open licences are now allowed, so the CC BY-NC-SA flag no longer blocks the final picks. NIST is still banned.

## Notebook scan (25 Oct 11:2x IST; nothing built)
Public top is now **0.466**: sanpier/pubchem-neural025-lb-0-466 (70 votes, version 25 Oct 04:55 IST, "PubChem + Neural025") and gwey1025/casmi26-lb-0-466 (3 votes, 25 Oct 00:22 IST). 0.465 tier: dmitriigluzdov (170 votes, the base we copied), lavinwins, guhongbin v498/v501/v502, avazaku evidence lab, youyouwou. 0.464 tier: evgendvorkin (198), amrkhalid124, rajrajak99 Apex v800-v840, d1m4official, hailonglv. Our stack is 0.465. The 0.466 pair is new (not in the vault before); "Neural025" suggests a neural re-rank weight of 0.25 on the PubChem gate, matching lever (3) in our discussion draft. To do on go: read sanpier's source and diff against the 0.464 base.

## Discussion scan (25 Oct 11:4x IST; replies not read, only topic lists and opening posts)
- Our own post "Three changes on top of the 0.464 base..." (752356) is up since 24 Oct 20:52 IST: 0 votes, 0 comments.
- 752497/752498 cosmorin, 25 Oct 11:28 IST: "Team up, Silver zone (~Top 200), pushing for Gold" (0 comments).
- 752297 (24 Oct): rules question whether the notebook may use relationships across test spectra at inference (clustering test spectra, using a confidently identified molecule to rank similar ones). Directly relevant to any cross-spectrum re-rank; host answer not read. 2 comments, last 25 Oct 03:51 IST.
- 751373 (21 Oct, active 25 Oct 08:55 IST, 4 comments): which third-party pretrained models are OK (MS2DeepScore, FIORA, SIRIUS/CSI:FingerID, chemical language models); earlier host rule: weights need original training code plus permitted training data.
- 751300 (7 votes, 6 comments): class 3 scores: poster's local gains do not reach the LB, same as our experience.
- 748882 (13 comments): consolidated open questions before final selection (reproducibility of official checkpoints).
To do on go: read the host replies in 752297 and 751373 before building anything cross-spectrum or using a new pretrained model.

## 0.466 copy (25 Oct 12:37 IST)
Kernel casmi-pubchem-neural025-copy (exact copy of sanpier/pubchem-neural025-lb-0-466, original image pin, push dir /home/user/work/push_casmi466), pushed v1; running. Not yet submitted.

## Discussion replies read (25 Oct 13:0x IST)
- 751373, host David Healey (22 Oct): models trained only on public libraries (MS2DeepScore, FIORA-OS) allowed; SIRIUS/CSI:FingerID not allowed; chemical language/graph models allowed; the code that generates intermediates is needed too; PubChem Lite is fine. Follow-up 25 Oct asks whether regenerable code is enough or intermediates must be bit-for-bit reproducible (unanswered).
- 752297 (cross-test-spectrum use): no host answer yet. A participant says it has always been ok to use test data this way; a second asks the host for yes/no. Treat as unconfirmed.
- 751300 (class 3): others also see local class-3 gains not reaching the LB; one claims a +0.003-0.004 class-3 lever over weeks and a possible +0.007-0.010 one. Nothing concrete.

## Master/Grandmaster discussion read (25 Oct 13:3x IST; 41 of 80 topics fetched)
19 posts by Masters/Grandmasters, mostly in "Chasing medals with prize-ineligible sources" (NIST-based weights and enforcement): greySnow, yuanzhe zhou, hengck23 (GMs) and M Sato, PC Jimmmy (Masters) say enforcement against non-prize teams is unclear; nothing from the host. Patrick Chan (Master): CC BY-NC weights (FRIGID) are arguably fine since the data is CC BY-NC, but the hosts' earlier line favoured permissive licences; asks whether NPLIB1 is NIST-derived. Chen Ken (Master): does using NIST only to filter public spectra make downstream weights ineligible. PC Jimmmy: private Kaggle datasets are allowed; test-spectrum relations "always ok" (his opinion, not the host). Mark (Master) also finds low CV-LB correlation, excluding Enveda NP examples. No techniques or scores shared.

## Methods digest 25 Oct (13:30 IST; read-only, nothing built, nothing pushed or submitted)

### 1. What "Neural025" is (sanpier/pubchem-neural025-lb-0-466)
- **Structure:** a 6-cell wrapper. It embeds (base64) the frozen 0.465 notebook (sha aa1a43cd...) and runs it unchanged as a subprocess via `runner.py` (exec of its cells), exports the final 25 candidates per molecule, scores them with a small neural model on GPU (`neural_scores.py`), then re-orders each list with a fixed reciprocal-rank fusion (`submission.py`). The candidate set is asserted unchanged (`sorted(before)==sorted(after)`).
- **Fusion rule:** `score = 1/(3+baseline_rank) + w/(3+neural_rank)`, w fixed, offset 3. The report says `weight_is_validated=False`, "fixed; no validation or tuning". The pulled latest version (24 Oct 23:25 UTC = 25 Oct 04:55 IST) has **w = 0.15**; gwey1025's earlier copy (25 Oct 00:22 IST) has **w = 0.25** (title "Neural025"). Both are titled LB 0.466, so w between 0.15 and 0.25 does not move the displayed score. Which w scored 0.466 first is unknown (version history is 403 for us).
- **Rank-1 is protected at w = 0.15:** 1/4 - 1/5 = 0.050 > 0.15/4 = 0.0375, so the baseline top-1 can never be displaced. At w = 0.25 it flips only if the neural model ranks the baseline top-1 below about 17 of 25. The fusion therefore reorders ranks 2-25 only.
- **The model (`sanpier/casmi-spectrum-model-v1`):** a CLIP-style spectrum-vs-molecule encoder. Peak transformer (3 layers, width 192, 6 heads, 96 peaks, Fourier m/z features, adduct embedding, collision-energy and precursor metadata), molecule MLP on a 2048-bit Morgan fingerprint, temperature 0.1, 16 hard negatives per query from a 128-pool, auxiliary fingerprint head (weight 0.5). Score = cos/0.1 + 2.0 * (predicted-fp . candidate-fp / 2048), averaged over all query spectra of the molecule. Trained **from random weights on competition `train.parquet` only** (2,430,327 spectra, 273,833 molecules; no external data, no pretrained checkpoint). 16 epochs x 500 steps x ~17 s = about 5 min on a T4 (+ a refit stage on all identities for `final_model.pt`). Holdout diagnostic MRR@25 = 0.484 (random molecule holdout, 256 candidates, not scaffold-disjoint, so it flatters).
- **`sanpier/catalogue-experiment-saner`:** *not used* by the 0.466 notebook (it only needs the model bundle). It is the author's offline experiment: a learned combiner (`learned.py`, pickles) over 29 features (baseline/main/prefusion/engine2/pubchem ranks, library max, ICEBERG/GLACIER z, neural z, interactions) on a 120-molecule panel (60 train / 30 calibration / 30 test), with reference spectra of held-out identities excluded. Test-30 MRR@25: baseline 0.879, neural alone **0.387** (-0.49 vs baseline, CI -0.63..-0.34), combiner_all 0.923, combiner_no_neural 0.907; combiner_all vs no_neural +0.017 (CI 0.00..0.05, 1 win, 0 losses, 29 ties). Report: "exploratory, do not choose a submission from the winner". The author then shipped a fixed RRF and not the combiner.
- **Reading:** the neural model is a weak independent voter (MRR 0.39 vs 0.88), worth at most a few rank-2-to-25 swaps. Its only measured value is +0.017 MRR on 30 test molecules with a CI touching 0.

### 2. Diff against our stack
| | 0.466 frozen baseline | our `push_casmi464` | our `push_casmi_gate_genfix` |
|---|---|---|---|
| FPNet in PubChem channel | A+B + fold0 (1.5*0.5*(zs+zm)) | A+B + mean of 5 folds (v18-v20) | same as 464 |
| PubChem sixth slot (rank 24) | no | yes | yes |
| Candidate-level promotion gate | **yes** (`candidate_promotion_evidence`) | no | yes (our port) |
| Features-frozen gen20 | no | no | yes (delta 2.0, max 60) |
| Hygiene | `pc_lists.json` unlinked, `rr.check_returncode()` | - | - |
| Post-hoc neural RRF | **yes, w 0.15** | no | no |
Cells of the frozen notebook that differ from our gate_genfix: 16 (five-fold and sixth-slot lines absent), 24 (CORE/RUNNER fold0 rule + two hygiene lines), 29 (genfix absent), 44/45 (version string and slot list). Nothing else. So the 0.466 is "v16 fold0 base + candidate gate + neural rank-fusion"; ours is a different, larger base (five-fold, sixth slot, genfix) with the same gate and no neural step.
- **Runtime/GPU:** T4, no internet. The cost is the baseline pipeline (hours; engine 1 about 4 s/molecule plus ICEBERG/GLACIER); the neural step is minutes (tiny model, 25 candidates x 1-4 spectra per molecule, `torch.inference_mode`). It needs a GPU only because it asserts `cuda`.
- **Licences/provenance:** model bundle and experiment dataset are labelled "unknown" licence on Kaggle (no description); content is derived only from `train.parquet` plus the author's code (`train.py`, `prepare.py`, `features.py`, `model.py`, `inference.py` are in the dataset, so the weights are regenerable). Host: models trained on `train.parquet` and redistributed artifacts are fine (745841); weights need the original training code plus permitted data (748882, 751373); the code that generates intermediates is also needed (751373 point 4). The experiment dataset's pickles/parquet are not part of the scored path. The five dmitriigluzdov datasets are the same as for our base (assets dataset CC BY-NC-SA, now allowed: host 750330, 17 Oct, non-commercial open licences permitted; NIST still banned). SIRIUS is not used anywhere. Compliance risk: low, provided the final write-up cites `train.py` and the data hash; the wrapper hides an exec of base64 code, which is reproducible but hard to audit.

### 3. What plausibly explains +0.001 over 0.465
1. **One molecule moving rank 3 -> 2 is +0.0013** (1/132 x 0.167); a rank-1 gain is not even possible at w = 0.15. The step is 0.13 of the 0.0076 quantum.
2. Rerun noise: dariushafshar's chain table (481 versions): 30 of 33 identical-code groups give the same displayed score, the other three differ by 0.002, 0.004 and 0.006.
3. The hygiene lines do not change ranking. So the neural add-on is a plausible but unproven +0.000 to +0.002.
- **Can we reproduce/improve with free compute?** Reproduce: yes, already pushed as `casmi-pubchem-neural025-copy` (v1 running). The model can be retrained in ~10-15 min on a free T4 from `train.py`. Improve: only within the noise; see the ranked list below.

### 4. Methods digest from the forum (123 topics, all fetched; Masters/Grandmasters and >=5 votes read)
Techniques that repeat (with numbers):
- **The bottleneck is the ranker and isomer separation, not retrieval**: Adarsh (Master, 743974, 24 votes): CV MRR@25 0.584 in regime B, 0.846 in A; formula right in >95% of cases, isomers wrong (745468). Udam Liyanage (743254, 13 votes): wrong winner has the same formula 98% of the time; 8% of molecules have a tautomer copy under a different PubChem key (dedupe on the metric key); one test molecule = 0.0025 MRR; seed noise alone moves a GBDT holdout by up to +/-0.007.
- **Pipeline shape (haideptry, 741745, 19 votes, 0.339 at the time):** exact library match (entropy/cosine) + mass-shifted analog propagation (sum Sim^4 x Tanimoto) + MetFrag-lite + a spectrum-to-fingerprint transformer (6,930 bits; neural score = f_c . z, one matmul) -> 31-feature HistGBM x 4 seeds x 2 priors -> confidence gating. Window +/-8.5 ppm; blind PubChem isomer expansion hurt (LB 0.335 -> 0.205 in another report).
- **MetFrag-lite tuning (NeckBeard, in 743254, isomer panel of 241):** down-weight 2-bond cleavages x0.6: +0.030 (t 2.6); linear instead of sqrt intensity +0.017; tolerance 0.005 Da +0.005; H shifts -2..+3 +0.010; all four +0.065 (t 4.6, LB: flat for his own submission). No gain from bond-propensity weights, 3-bond cuts (-0.031), penalising multi-route fragments (-0.015). "Parsimony helps, hand-written chemistry does not".
- **CV does not transfer to the LB, and the gap is class mix** (Bull 748068/748524/751300, Adarsh 741597, Aryan 748068 thread, Mark 751356, Tra Crew): local class-3 gains do not reach the LB; a popularity prior is +0.50 MRR on enveda-np-examples-like CV but +0.075 on the LB; a LambdaRank + fragmentation gain of +0.14 CV was +0.03-0.04 LB; Bull's fitted class shares are about 28% / 20% / 52% (class 1/2/3); ICEBERG re-ranking +0.02 on class 2 drops to ~0 on structures not in MassSpecGym; PubChem fixed-rank insertion +0.01-0.02 local but LB 0.295 -> 0.288. Tra Crew (751300, 751356) predicts the last ~15 submissions with a calibrated mock scorer built from a held-out train subset; says a class-3 lever is worth only +0.003-0.004 for weeks, with a possible +0.007-0.010 one not yet shown.
- **Generation:** Antriksh Arya (748029): rule-based re-ranking has a low ceiling; class 3 needs candidate generation, "toughest, few LB hits"; Udam (743254, point 6): derivative-generator "novel" simulations overstate reach about 3x and generated candidates have not paid on the LB. Matches our gen20 history (-0.008 to -0.012 ungated).
- **Public chain (dariushafshar, notebook "The Public Chain, Version by Version", 23 Oct 19:31 UTC):** 208 notebooks, 481 scored versions. 0.465 first shown by sanpier v2 (candidate-level gate). Since 9 Oct the frontier rose 17 times (+0.001 to +0.005, 4.2 reciprocal-rank units in total). 69% of 470 steps are smaller than one molecule (0.0076), 95% smaller than one standard error (0.036-0.042 on ~132 public molecules). Repeatable levers: N1 challenger +0.008 on both lineages (forward-model coverage), `rel > 250 -> 150` promotion +0.003 to +0.006 on 4 parents; PubChemLite in engine 2: -0.003 and +0.004 (sign flips = below one molecule); non-reproducible 3rd engine (yjsshisfsf) +0.004-0.005.
- **pjmathematician `casmi26-e255-dg464-gpu` (10 votes):** all 73 cells are byte-identical to our `push_casmi464` (dmitriigluzdov v20, 0.464). It is a copy, not a new method.
- **gwey1025/casmi26-lb-0-466 (3 votes) is a copy of sanpier's notebook:** same embedded baseline (identical module cells and base64), same datasets list and the same model; the only differences are (a) w 0.25 instead of 0.15 (experiment label `fixed_neural_025`) and (b) a "low-GPU commit fast path" in cell 0: on the visible placeholder test (md5 signature of spectrum ids) it copies `sample_submission.csv` and skips the pipeline; on the hidden rerun (`KAGGLE_IS_COMPETITION_RERUN`) every cell runs. Its last line forces `FULL_RUN = True` ("r19 one-time full smoke commit"). No new technique.
- **Rules (hosts, David Healey / Marie Killian):** NIST and anything trained on it banned; non-commercial open data allowed since 17 Oct (750330); models trained on `train.parquet` fine; GLACIER/ICEBERG, DreaMS, FRIGID, TabPFN ok with a reference to original training code and data (748882); PubChem CID-PMID/Patent/SID counts and LOTUS/NPAtlas ok; MS2DeepScore 2.0/FIORA-OS ok, SIRIUS/CSI:FingerID not (751373); scorer is first-match and extra guesses after rank 1 cannot lower a row (748882, 748029); unreproducible weights cost prize eligibility, and the hosts reserve the right to remove teams (leaderboard effect left unanswered, asked at 748882 and 751373). Cross-test-spectrum use (752297): still unanswered.

### 5. Repeats / not tried / compute / licence
- **Repeating:** forward-model coverage (GLACIER/ICEBERG, N1 family); rank-1 protection with insertion at ranks 2+; candidate-aware gating instead of channel-level gating; PubChem used only as a gated minority channel (blind expansion loses); validation shortfall (every poster says local gains shrink 3-10x on the LB).
- **Not tried by us:** (i) neural RRF at ranks 2-25 on our stack; (ii) MetFrag-lite parsimony settings (2-bond x0.6, linear intensity, 0.005 Da, H -2..+3), CPU only, but LB transfer is expected at about a quarter of the offline size; (iii) test-spectrum relations (unconfirmed by hosts); (iv) a coverage audit of which test adducts still lack a GLACIER/ICEBERG score after N1A.
- **Costs compute:** anything that re-runs the full pipeline (genfix +45-60 min on GPU, the baseline hours); neural training about 5-15 min and scoring minutes; the offline bench is CPU (hours).
- **Licence checks:** all new items are train.parquet-derived or already approved; do not use SIRIUS/CSI:FingerID; keep `train.py` and the data hash in any write-up; the sanpier datasets carry an "unknown" licence label, so cite the code and the host's train.parquet ruling.

### 6. Next experiments (ranked; one molecule = 0.0076; our stack is 0.465)
1. **Neural RRF (w 0.15, offset 3, rank-1 protected) on our gate+genfix stack, after an honest holdout check.** Check first on CPU or T4: score our bench candidates with `best_validation_model.pt` on the validation identities only (the `final_model.pt` has seen everything), paired bootstrap over clips for w in {0, 0.15, 0.25}; run only if P(better) >= 0.9, else log the failure. Expected gain +0.001 (range 0.000 to +0.003). Forecast if pushed: 0.466 (0.464-0.467). Note the repo rule: this is a blend-composition change (ten public failures), so only the neural model itself, trained on train.parquet, is a "trained" change; weight tuning on the public score is not allowed as an honest result. Cost: neural step minutes, the pipeline run unchanged.
2. **Use the neural score to gate the genfix insertions** (generated candidates currently use a quantile-mapped engine score with a fixed delta 2.0; add the rule "insert only if neural fp-score rank is in the top-k of the augmented list"). This is the one place a score independent of the library bias can help class 3. Expected +0.000 to +0.003 (honest unseen-bench gen20 gain was +0.008 on class 3, times about 50% class share, minus losses); forecast 0.465-0.467, point 0.466. Cost: neural scoring of up to 60 extra candidates per molecule, plus a bench replay on the unseen set (CPU, hours).
3. **Coverage audit, then extend the N1 family if a gap exists** (CPU, no submission): count final-list candidates per adduct with no GLACIER/ICEBERG score on the bench and in the 466 run audit files; the only +0.008-class repeatable lever in the chain was forward-model coverage. Expected +0.000 if no gap, up to +0.008 (one molecule) if one adduct group is blind; probability of finding one is low (N1A and N1 challenger already cover [M+NH4]+ and [M-H2O+H]+). Forecast: 0.465 (no change) to 0.470.
- Not recommended: more weight/threshold tweaks on the public score (flat in the chain), depth 7 (-0.008), wide PubChem expansion, MetFrag-lite retuning as a main bet (CPU-cheap, but LB transfer 1/4 and our base already has the v7c fragment gap).

## Forward-model coverage audit (25 Oct, IST; CPU, offline; next experiment 3 of the digest)

**Forecast written before the conclusion.** Expect the molecule-level gap to be small (positive adducts are all handled by N1A / the Na fallback), so the rank-1 gain from completing coverage is +0.000 to +0.004, point +0.001, P(at least one public molecule, +0.0076) about 0.1. Script `enveda-casmi26/50_forward_model_coverage_audit.py`, output `enveda-casmi26/artifacts/coverage_audit_summary.json` and `coverage_audit_molecules.csv`.

**Method.** The 400 test molecules (1,213 spectra, all timsTOF) are assigned evidence by the coded rules: ICEBERG scores only positive [M+H]+ / [M+Na]+ spectra (`fuse.build_ice_input`, `covered_only=True`); GLACIER scores real [M+H]+ only, plus N1A proxies from [M+NH4]+ / [M-H2O+H]+ when a molecule has no [M+H]+ (Na, K, negative never proxied); the N1 challenger needs both.

| Group | Molecules | Share | Adducts |
|---|---|---|---|
| ICE + GL (full N1 possible) | 366 | 91.5% | all with a real [M+H]+ |
| ICE only (Na fallback, GL blind) | 5 | 1.25% | [M+Na]+ only |
| GL only (N1A proxy, no ICE) | 2 | 0.5% | [M+NH4]+ only |
| **No forward-model score at all** | **27** | **6.75%** | 25 [M-H]- only, 2 [M+CH2O2-H]- only |

Every positive-mode adduct is covered at molecule level; the only blind group is negative-mode-only molecules (27 = 6.75%). Not full N1: 34 = 8.5%. Test adduct counts per spectrum: [M+H]+ 959, [M-H]- 193, [M+CH2O2-H]- 31, [M+Na]+ 22, [M+NH4]+ 4, [M+K]+ 2, [M+Cl]- 2 (no [M-H2O+H]+ in test, so the water-loss proxy is idle).

**By class: not answerable locally.** True classes are hidden. A cosine class proxy against `train.parquet` is useless: all 1,213 test spectra occur verbatim in it (binned cosine 1.000), but the matched structure is in the pipeline's top 25 for only 12 of 400 molecules (older `d7out` baseline), so it is not a class or answer signal (flagged, left alone; worth a look at what those train rows are). With the fitted prior shares 28 / 20 / 52% (assuming adduct is independent of class): blind molecules about 7.6 / 5.4 / 14.0 (class 1 / 2 / 3) of 400; on the 132 public molecules about 8.9 (about 2.5 / 1.8 / 4.6).

**Not available locally:** candidate-level coverage (which final-list candidates lack an ICE/GL score), the real run's `n_mols_scored`, the N1 stats of the 0.464/0.466 runs. The bench logs (`benchdata/log464`, `genfix1`) are from the np-examples pseudo-test, are all positive [M+H]+, and print no coverage lines. The ICEBERG package and weights are not on disk either (only `fuse.py` / `gl_fuse.py`), so no extension can even be prototyped offline.

**Expected gain if coverage were completed.** Upper bound with perfect evidence on all blind molecules: 8.9 public molecules x 0.0076 = 0.068, but that assumes every one flips to rank 1, which is not credible: class 1 (about 2.5 public) is already about 0.91 MRR, and forward models gained mostly on isomer separation (chain: challenger +0.008 on one same public molecule, three times). Calibrating at the observed hit rate of the +0.008 steps (about one molecule per coverage step covering a few positive-adduct blind molecules), a negative-mode extension would net about 0.1 to 0.5 molecule: +0.001 to +0.004. The 5 Na-only molecules (about 1.7 public) are already the tested negative ([M+Na]+ to [M+H]+ GLACIER shift: -0.008, -0.008, -0.013). The 2 NH4-only molecules (0.7 public) already have N1A.

**Verdict (negative result for a buildable extension).** The gap exists but it is negative-mode only: no negative-mode ICEBERG/GLACIER is available (both are positive-mode models; flipping [M-H]- to a pseudo-[M+H]+ is chemically unsupported and untested), nothing on disk to run, and the headroom is below one molecule. Recommendation: do not build; do not spend a GPU run. Forecast stays 0.465/0.466 (no change); P(extension reaches +0.0076) about 0.1. Re-open only if a negative-mode forward model with a commercial licence appears in the forum or datasets.

**29 Oct (IST) pointer:** FIORA-OS (MIT, MSnLib-trained, host-allowed in 751373) has an [M-H]- mode (no Na/formate/NH4) and could cover 25 of the 27 blind molecules; research and integration plan in `Study Guide/CASMI isomer reranker research.md` §7. Not run; needs user approval for download/install and a local CPU bench.

## Scan 25 Oct 19:07 IST (notebooks and discussions; titles only, sources unread)
- Our copy casmi-pubchem-neural025-copy (exact copy of sanpier, w 0.15) scored **0.465**, not 0.466: no gain over our stack (0.465). Either sanpier's 0.466 used another weight (gwey1025's copy has 0.25) or the 0.001 is rerun noise.
- New top public: **0.467** alexchilton/casmi26-lb-0-467-one-knob-rrf-offset-3-to-1 (9 votes): the rank-fusion offset in the neural re-rank changed from 3 to 1, a one-parameter change on the 0.466 line. New 0.466 copies: borrun (pubchem-context-ranker), ou20040313 (orange rank lab), guhongbin v508 pc466w025; sanpier now 78 votes, lavinwins 0.466. Steps of +0.001 around 0.465-0.467 are inside rerun noise (3 of 33 identical-code groups differ by 0.002-0.006), so treat 0.467 as unproven until seen twice.
- New topic 752541 (LeeDongHyuk, 25 Oct 14:45 IST): eligibility of non-reproducible final submissions on the private LB and medals; no replies yet.
To do on go: read alexchilton's source and diff against sanpier's.

## 0.467 copy (25 Oct ~19:20 IST)
casmi-one-knob-rrf-copy: exact copy of alexchilton 0.467 on its original pinned image (push dir /home/user/work/push_casmi467), pushed v1; a detached watcher submits it when the kernel completes. Forecast written before submitting: 0.466 (0.464-0.467); our copy of sanpier's 0.466 scored 0.465, so a 0.467 reproduction is unlikely.

## Stopped, difference read (25 Oct ~19:30 IST)
Watchers stopped before submitting the 0.467 copy (kernel casmi-one-knob-rrf-copy still runs on Kaggle; not submitted). Difference 0.466 (sanpier, pulled) -> 0.467 (alexchilton): one constant, RRF_OFFSET 3.0 -> 1.0, with NEURAL_WEIGHT 0.25 (not the 0.15 of the pulled sanpier file), plus a fast path on the placeholder test and new markdown. Author's own sweep (each a full submission): w 0.25 / offset 3 -> 0.466; w 0.25 / offset 1 -> 0.467; w 0.25 / offset 10 -> 0.446; w 0.75 / offset 3 -> 0.419. So the neural re-rank is sensitive to its parameters (flatter or heavier lost 0.02-0.05) and the +0.001 is one point inside rerun noise. This also fits our 0.465 for the w 0.15 copy. Decision: wait for a larger jump before spending more submissions here.

## lavinwins copy (25 Oct ~20:00 IST)
casmi-lavinwins-top1-copy: exact copy of lavinwins/casmi-26-lb-top-1 (public 0.466, 43 votes), original image, push dir /home/user/work/push_casmi_lavin, pushed v1 on user request. Not submitted (hold until a larger jump).

## romanrozen 0.468 copy (26 Oct 17:19 IST)
New top public **0.468** romanrozen/0-468-casmi-from-peaks-to-structures ("From Peaks to Structures": leader-safe consensus re-ranking on the strongest public pipeline; keeps the proven rank-1 decision and re-ranks the rest of each list). Notebook is 947 lines against 95 for sanpier's 0.466, so a different method, not one knob; sources otherwise the same datasets (sanpier spectrum model, dmitriigluzdov assets). Copy casmi-romanrozen-0468-copy (exact, original image, push dir /home/user/work/push_casmi468) pushed v1 on user request; NOT submitted. Forecast written before submitting: 0.467 (0.465-0.468); our copies of 0.466 lines scored 0.465.

## romanrozen 0.468 copy scored (26 Oct 23:48 IST)
casmi-romanrozen-0468-copy (exact copy, submitted 18:47 IST, ref 57667630) scored **0.468** public: reproduces the public score and is our best (previous best 0.467 from the lavinwins copy, 0.465 our own stack). Forecast written before submitting was 0.467 (0.465-0.468): hit the top of the range. Time to score about 5 h. Lesson: the leader-safe consensus re-rank (keeps rank 1, re-ranks the rest) is a different method from the one-knob lines and it reproduced exactly, so copies of 0.46x lines are reproducible when the code is exact. Copy only; the licence check (sources the same as sanpier's lines) still stands; do not treat as an honest validated model.

## romantamrazov spectral-parallax copy (27 Oct 16:41 IST)
New top public **0.472** romantamrazov/casmi-2026-spectral-parallax (15 votes; latest version 27 Oct 08:31 UTC; user called it V17, but the versioned pull returned 403 so I pulled the latest, which is the version carrying the 0.472). Builds on kozykappa's lb-0-468-balanced-acquisition-evidence (V2 ensemble: neural RRF with different weights for rank 1 and ranks 2-25, plus conservative one-slot candidate recovery) and adds a "diagnostic-ion consensus" (physics-informed) and an outsiders module; keeps the 0.465 pipeline byte for byte. Other new tops: heliosli 0.471, youyouwou 0.470, kozykappa 0.468. Copy casmi-spectral-parallax-copy (exact, original image, push dir /home/user/work/push_parallax17) pushed v1 at 16:41 IST; submission by hand when the visible run completes (user asked to submit directly). Forecast written before submitting: 0.471 (0.469-0.472); our exact copies have reproduced the public score within 0.001 (0.468, 0.467), one was 0.001 below (0.465 vs 0.466). Gold line about 0.471-0.473, so this copy lands on it.

## spectral-parallax copy scored (27 Oct 22:18 IST)
casmi-spectral-parallax-copy (exact copy of the latest romantamrazov casmi-2026-spectral-parallax, submitted 17:10 IST, ref 57699799) scored **0.472** public: reproduced the public score exactly. Forecast was 0.471 (0.469-0.472): hit the top of the range. Time to score under 5 h 10 min. Our best CASMI (previous 0.468, our own stack 0.465); gold line about 0.471-0.473, so this lands on it. Copy only: not an honest validated model, rank-only for final selection; sources are the same sanpier/dmitriigluzdov datasets plus the author's own modules. GPU quota: the visible run cost about 0.4 h (5.54 -> 5.12 h left); hidden scoring not billed. Lesson: exact copies of public CASMI notebooks have reproduced the public score on 4 of 5 tries (0.468, 0.467, 0.472, 0.465 vs 0.466).

## busyaprime 0.477 copy (28 Oct 17:02 IST)
New top public **0.477** busyaprime/lb-0-477-two-0-472-fixes-merged-the-right-way (author's submission 57722897 read 28 Oct 08:48 UTC). Claim: two public changes to the same 0.468 leader-safe recipe each reach 0.472 (Roman Tamrazov's fragment check; alex chilton's heavier neural weight 0.75 on ranks 2-25); stacked the plain way they still score 0.472, because the fragment check's leader rule picks contenders from the order it is handed; run on the order it was built on, with ranks 2-25 from the 0.75 order, the two add up (+0.005). Heliosli's 0.475 needs a private dataset (the optional rr3 layer is skipped when its dataset is absent, so this notebook does not depend on it). Commit run 21-30 min (12 molecules); scoring run hours on T4, internet off. Same datasets as the 0.468/0.472 copies (sanpier, dmitriigluzdov).
Copy casmi-busyaprime-477-copy (exact, original image, push dir /home/user/work/push_c477) pushed v1 at 17:02 IST; submit by hand when the visible run completes (user said submit). Forecast written before submitting: 0.477 (0.474-0.478); exact CASMI copies reproduced 4 of 5 times. GPU quota 1.66 h left before the run (expected cost about 0.4-0.5 h).

## busyaprime 0.477 copy scored (28 Oct 22:56 IST)
casmi-busyaprime-477-copy (exact copy, submitted 17:39 IST, ref 57736039) scored **0.477** public: reproduced the public score exactly. Forecast was 0.477 (0.474-0.478): hit the point estimate. Time to score at most 5 h 17 min. Our best CASMI by 0.005 over the 0.472 copy and 0.012 over our own 0.465 stack; above the gold line (about 0.471-0.473) and 0.002 above heliosli's 0.475, which needs a private dataset. Exact CASMI copies have now reproduced the public score on 5 of 6 tries. Copy only: rank-only for final selection, same sanpier/dmitriigluzdov datasets plus the author's own fragment-check and merge order; not an honest validated model of ours. GPU quota cost of the visible run about 0.45 h.

## Scan 29 Oct 02:35 IST (titles and scores only)
Only one public notebook above 0.477: heliosli/happy-to-make-the-dataset-public-if-it-helps at **0.478** (51 votes; 0.475 on 28 Oct), which needs heliosli's private rr3 reranker dataset, so it cannot be copied (the optional layer is skipped without it and the notebook falls back to the 0.468 line). At 0.477: kunaldesale2408 "Enveda CASMI 2026 - MID" (13 votes), romantamrazov Spectral Parallax (96 votes; it was 0.472 on 27 Oct, so it has been updated, possibly with the fragment check merge), youyouwou v120rel150, rajrajak99 Apex v1300/v1320 PXM, pjmathematician e303-pub477-gpu, cngnguynhuuz fork m477, and busyaprime (64 votes, our copy). 0.472 tier: alexchilton, romanrozen, youyouwou v117. The 0.477 plateau now has many forks, i.e. the merge is reproducible; the next step needs new information (heliosli's dataset) or an independent lever.

## Scan 29 Oct 11:34 IST
Notebooks unchanged at the top: heliosli 0.478 (private rr4 dataset), 0.477 plateau (busyaprime, Spectral Parallax, kunaldesale2408 MID 25 votes, rajrajak99, pjmathematician, youyouwou).
Discussions: 753525 (Harshit Kumar): about 125 public molecules, rank 2 -> 1 is worth 0.004 MRR, so 0.472 -> 0.477 is about one molecule; 273 teams sit at exactly 0.472, only 16 are above 0.48; modelled standard error of mean MRR about 0.04 on 125 molecules (public differences of 0.005 are mostly noise; private may reorder). 753511: same theme. 752297 (cross-test-spectrum processing): the only "yes, allowed" is from a participant (Marie Killian), not the host; still unconfirmed. 752541: host David Healey "we reserve the right to remove submissions that are not reproducible from the leaderboard, regardless of ranking". 753400: negative result, rule-based biotransformation of spectral analogs recovers few class-3 answers. 752846 / 752735: visible split 0.99 vs LB 0.41, validation lessons. 753724: a 20M-candidate database starter (licence unchecked). 745072 (prize eligibility for code and weights in public datasets) now 24 comments.
Isomer reranker research agent stopped on a usage limit overnight; resumed 29 Oct 11:3x IST.

## Isomer reranker research (29 Oct ~12:00 IST; read-only)
See [[CASMI isomer reranker research]] (`Study Guide/`). heliosli's rr3/rr4 overlay = permutation-only rerank of non-train candidates by z(-log rank) + alpha z(reranker), train candidates frozen, lib_max >= 0.9 untouched, self-test gate. Recommended first experiment: CPU LightGBM lambdarank on fragment-match features (reuse `fragments.py`) over same-formula train lists, scaffold folds, then bench on 0.477 lists with paired bootstrap. Expected +0.000 to +0.003, ~15% to beat 0.477. Open: the test-spectra-in-train anomaly and rules on train-membership logic.

## Isomer reranker experiment (29 Oct 12:07 IST; CPU only, nothing pushed to Kaggle or submitted) - NEGATIVE
Script `enveda-casmi26/51_isomer_reranker.py` (outputs `artifacts/51_*`). Training: 3,000 [M+H]+ train structures (bench answers excluded), each with 10-24 same-formula isomers from other train structures (51,628 rows), 5 folds by Murcko scaffold. Features: fragments.py 1-cut and 2-cut matches separately (2-cut weight 0.6, linear intensity, 0.005 Da, H -2..+3), top-5 / base-peak coverage, fragment counts, each z-scored in list. LightGBM lambdarank depth 6, 300 rounds. No train-membership, no cross-test-spectrum features.
- **Held-out isomer MRR:** LightGBM 0.871 (top-1 0.790) vs fragment-score-only 0.862 (top-1 0.780) vs random 0.206. +0.009 MRR, every fold positive (+0.003 to +0.013). Random train isomers are easy; this is not the bench's isomer difficulty.
- **Bench** (genfix bench `fin1` base lists, 450 molecules; guards fixed before scoring: only slots with the rank-1 formula permuted, z(-log rank) + 1.0 z(model), rank 1 moves only with combined margin >= 1.0, lib_max >= 0.9 untouched (40 lists)): 24 lists changed. Natural mix d -0.0056 [-0.0157, +0.0045], **P(better) 0.14**; 28/20/52 mix d -0.0073, P(better) 0.02. Class 1: 0.888 -> 0.885, **5 rank-1 losses**, 4 gains (fails zero-loss rule). Class 2: -0.004, 0 rank-1 changes. Class 3: 0.312 -> 0.301, 1 loss, 0 gains. Caveat: 5.8 of 25 candidates per list lack SMILES locally (scored at the list minimum); query peaks are the answer's own train [M+H]+ spectra.
- **Forecast (written before concluding):** bench gain -0.006 / 3-10 = -0.0006 to -0.002 public on top of 0.477.
- **Verdict:** rejected; do not integrate into the 0.477. The fragment-match signal is mostly already in the 0.477 (Tamrazov fragment check, sanpier RRF), as the research note warned.
- **Test-in-train anomaly (cheap check):** 40 random test spectra, all 40 have an exact array match (m/z and intensities equal) in train.parquet, each to exactly one row, all with `ingest_lib = enveda-180`, one InChIKey14 each (e.g. m_347e26 -> AVPWXVRXPXHKRI). So the test spectra are planted verbatim in train as the enveda-180 library under a structure label; given the earlier finding (that structure is in our top 25 for only 12/400), the labels are likely decoys or scrambled. Not used in any model; treat any enveda-180 lookup as a trap until the host says otherwise.

**29 Oct 13:24 IST pointer:** FIORA-OS step 2 bench (`52_fiora_negmode_bench.py`, 60 [M-H]- QTOF train molecules, enveda-180 and MSnLib structures excluded): held-out 31, top-1/MRR FIORA 0.355/0.561 vs fragment 0.226/0.543 vs random 0.097/0.231, P(better) 0.62; z-sum FIORA+fragment 0.419/0.627, P 0.94; 0.037 s/prediction on 4 cores. Step 3 no-go until a ~250-molecule bench confirms. Details: research note §8.

**29 Oct 13:46 IST pointer:** FIORA step 2 enlarged (`53_fiora_negmode_bench250.py`): pool gave 193 new held-out [M-H]- molecules (not 250). FIORA 0.466/0.606 top-1/MRR vs fragment 0.218/0.495 (P 1.00), comb 0.446/0.608 (P 1.00 vs fragment, 0.54 vs FIORA). Guarded genfix-bench check on 214 [M-H]- lists failed: 19 class-1 rank-1 losses, MRR -0.163 (P 0.00); the base order already ranks isomers at 0.94 within slots. Step 3 no-go. Details: research note §9.

## Next steps plan (29 Oct 16:3x IST)
Lever table (38 levers, bench and LB), ranked plan and final-selection advice in [[CASMI next steps plan]] (`Study Guide/`). Final selection recommended: 0.477 copy (57736039) + Spectral Parallax copy 0.472 (57699799). Step 4 (0.477 fork diffs): only new signal is Spectral Parallax's neutral-loss isomer rescue, bench-only; see "Step 4 result" in [[CASMI next steps plan]].

## Discussion scan 29 Oct 16:23 IST
- 753775 (new, no host answer): are checkpoints that other participants published as Kaggle datasets (gradient-boosted rankers, fine-tuned fingerprint networks, precomputed feature tables) without training code or data treated like public pretrained models, or must a team be able to retrain them? Relevant to every copy we hold (the 0.477 and 0.472 lines load dmitriigluzdov's and sanpier's assets); together with 752541 (host may remove non-reproducible submissions) this is the main eligibility risk of copies, not of the score. Watch for the host's answer before final selection.

**29 Oct 16:32 IST pointer:** FIORA step 3, rank-preserving fusion (`54_fiora_rank_preserving.py`, pre-registered in research note §10 at 16:26): z(-log base rank) + 0.25 z(FIORA entropy CE10) within rank-1-formula slots, lib_max >= 0.9 untouched. 214 [M-H]- lists: 2 changed, MRR -0.0022 (P 0.23), 1 class-1 rank-1 loss; fresh 41 formate/Cl/2M-H lists: 1 changed, +0.0012 (P 0.63). Both gates fail: no-go, no Kaggle version.

**29 Oct 16:40 IST pointer:** Step 4, Spectral Parallax disagreement rescue (`55_neutral_loss_rescue.py`, pre-registered §11 at 16:40). Faithful replay not possible locally (bench has no physical/V2/CUSTOM lists or neural/cosine features); ablated replay with published ion + neutral-loss gates on 250 np lists: 47 firings, 0 rank-1 gains, 23 losses (11 class 1), MRR -0.047 (P 0.00). No-go. Details: research note §11.

## Scan 30 Oct 01:37 IST
Notebooks: heliosli 0.478 still alone above the 0.477 plateau; new alexchilton "0.477 | 0.472 Weights + Fragment Check". Discussion 753900 (new): request for generation code of widely reused public assets (megayak simulated ranker rows, prvsiyan ranker features/fingerprint caches, ...) for prize eligibility; with 753662 and 753775 still open, eligibility of copies that load those assets is unresolved.

## Comments on heliosli's 0.478 notebook (read 30 Oct 08:4x IST)
4 top-level comments: fanyuchen (Expert, 27 Oct 19:13 IST) asked him to publish the rr3 bundle (rr3_config.json, rr_deploy.py, rr_v3.py, checkpoint, self-test, training/preprocessing code and data provenance); heliosli (Master) replied "Yes, happy to make the dataset public if it helps!" (21:46 IST), which is the notebook's title. FOYSAL posted a screenshot captioned "Private Dataset"; heliosli replied "Let the bullets fly a little longer~", i.e. he is keeping it private for now. sghwr (Expert): sarcasm about leading the board and fishing for a medal. As of 30 Oct the reranker datasets are still private (two empty entries in the notebook's dataset_sources). Keep watching; the 0.83 h of GPU is reserved for a copy if they go public.

## kunaldesale 0.478 copy (31 Oct, IST)
kunaldesale2408/enveda-casmi-2026-mid moved to public **0.478** (83 votes) with no private datasets (same public sources plus metric/rdkit-2026-3-3-wheel). Changes vs the 0.477 line: weights 0.25/0.75 with rank 1 locked, a stricter fragment re-scorer (>=3 spectra, margin >= 0.095, votes >= 0.75, nmargin >= 0.23, depth <= 6), a PubChem documentation prior on ranks 2-25 (w_pop 0.14, w_spec 0.05; the author says it was tuned on the public benchmark panel, +0.0086 there), formula-gated isomer swaps on ranks 2-10 by fragment evidence, and an InChIKey-14 dedup. Copy casmi-kunal-478-copy (exact, original image, push dir /home/user/work/push_kunal478) pushed on the user's word; submit when the visible run completes. Forecast written before submitting: 0.477 (0.475-0.478); exact CASMI copies reproduced 5 of 6 times, and +0.001 over the 0.477 is a quarter of one molecule, inside noise. Rank-only copy.

## Morning 31 Oct 08:31 IST
- kunaldesale 0.478 copy (ref 57818217) scored **0.476** public: inside the forecast range (0.475-0.478) but below both the author's 0.478 and our 0.477; exact CASMI copies now reproduced 5 of 7 within 0.001 (this one -0.002). Our best stays 0.477. Likely rerun noise or a later notebook version than the one that scored; not worth chasing.
- heliosli is now 0.479 (76 votes), still needs the private reranker datasets. Nothing copyable above 0.477.
- Medal lines (4,420 teams): gold 0.482, silver 0.477, bronze 0.477, top 0.517; us 0.477 rank 173 (was 144 at 22:04 UTC yesterday; same score, others caught up).
- Bench casmi26-bench477 still RUNNING (pushed 05:42 IST); quota 27.19 h left.

**31 Oct 09:53 IST pointer (bench reversal diagnosis):** bench477 baseline->merged477: 0 rank-1 gains, 4 losses; neural ranks the answer 9-10 on average vs baseline 1-4.5 (leakage would favour neural, so not leakage); bench differs from test in adduct mix and mass range; P(A real public gain) ~0.4 vs P(B noise) ~0.6. Final selection: 0.477 copy + own 0.465 stack (57585472). Details: plan note "Bench reversal diagnosis", `56_bench_reversal_diagnosis.py`.

## baidalinadilzhan 0.479 copy (31 Oct 12:3x IST)
baidalinadilzhan/casmi26-midx-exact-dedup-fill, public **0.479** (11 votes), no private datasets (same public sources as kunaldesale's MID plus the rdkit wheel). Final layer on the MID line: documentation-prior weights on ranks 2-25 (0.10 / 0.05 / 0.02), spectral-library bonus only for molecules whose best library match reaches 0.87, the pipeline's old leader pinned at rank 2 after a fragment-check leader move, metric-exact InChIKey-14 dedup and empty slots filled with scored outside candidates. Copy casmi-midx-479-copy pushed and submitted on the user's standing instruction. Forecast written before submitting: 0.477 (0.474-0.479); the kunaldesale 0.478 copy scored 0.476, and our bench says this whole neural/re-rank line loses vs the 0.465 baseline (P(B) about 0.6), so this is an LB-chasing submission, not a final-selection candidate by itself.

**31 Oct 14:05 IST pointer (candidate database audit, `57_candidate_recall_audit.py`):** current sources already hold 99.6% of natural products first reported 2021-2024 that are absent from train (NPAtlas probe, 2,523) and 96.7% of Enveda-180 novel structures (all via the PubChem tier). COCONUT 2026-10 / LOTUS 2026-04 / NPAtlas 2024_09 add 7-64 neutral structures new to every source and reach 8-9 of the 9 missing NPs; class 3 reach ~0 by the host's definition. Upper bound +0.0002 public; dilution ~0. No database change recommended. Details: plan note "Candidate database audit".

## baidalinadilzhan 0.479 copy scored (31 Oct 19:20 IST)
casmi-midx-479-copy (ref 57830973, submitted 12:54 IST) scored **0.479** public: reproduced the author's score exactly; forecast 0.477 (0.474-0.479) hit its top. New best CASMI (was 0.477). Same LB-chasing line the bench calls risky for private; final selection still pairs the best of this line with our own 0.465 stack. New public at 0.479: flexonafft "CASMI26: Closing the Evidence Gap" (99 votes), not yet read.

## flexonafft "Closing the Evidence Gap" 0.479, read 31 Oct 19:3x IST
flexonafft/casmi26-closing-the-evidence-gap (99 votes, public 0.479), same public datasets, no private data. It is busyaprime's m1 0.477 kept byte-for-byte below the config, plus the same final layer as baidalinadilzhan's 0.479 (rank 1 fixed; CUSTOM leader pinned at rank 2 after a leader move; ranks 2-25 by the CUSTOM consensus plus the MIDX documentation prior 0.10/0.05 gated at library 0.87/0.02; metric-exact dedup; outsider fill without duplicate fallbacks). Its iteration v4 adds only outsider recall in the tail: outsiders.py called with 40 candidates and 150 identity checks per molecule (was 20/30), and ranks 21-25 opened to scored outside candidates by neural score (consensus.rescue_tail); ranks 1-20 equal its v1. Ranks 21-25 are worth at most 1/21-1/25 per molecule, so this is a tiny tail change. Same score as our 0.479 copy, same lineage: not copied, nothing new to gain. Disagreement rescue is switched off.

## Above the public line (31 Oct 23:xx IST; research only)
See [[Above the public line]] (`Study Guide/`). At 22:54 IST: 38 teams above 0.479, but only 10 at >= 0.485 (28 are within one molecule); gold (rank 18) 0.483; us 0.479 rank 73. The top 5 had independent pipelines from day one (0.49 on 16 Oct vs public 0.446) and climb slowly with 89-197 submissions; no public assets. heliosli's team is 0.486 (public 0.479 + private reranker). **Corrections:** the "yes" to cross-test-spectrum processing in 752297 (Marie Killian, 26 Oct 20:53 IST) is posted as HOST, so it is allowed; the enveda-180 "test spectra in train" are the visible placeholder (staff, 745715), not the hidden test. Next step proposed: CPU pre-check of a rank-1-protected cross-test-spectrum consistency boost on the np-examples bench; odds of gold about 10%.

## Cross-test-spectrum boost pre-check (1 Nov 00:3x IST; CPU only, nothing pushed or submitted) - NEGATIVE
Bench477: 44% of molecules have a Tanimoto >= 0.5 analog in the set, 28% a detectable one (modcos >= 0.7); modified
cosine predicts analogs well (AUC 0.92). Pre-registered rank-1-locked boost on merged477 lists: +0.0013 bench MRR,
P(better) 0.83 (gate 0.9); oracle ceiling +0.0047. Forecast public +0.0001 to +0.0004. No-go. Details: [[CASMI next steps
plan]] section "Cross-test-spectrum boost"; script `enveda-casmi26/58_cross_spectrum_boost.py`.

- 1 Nov: copying bgunshui/casmi26-480-neural-ensemble (public 0.479 Variant M + z-score ensemble of public and self-trained spectrum model, ranks 2-25 only; author reports 0.480). Forecast 0.479 (0.474-0.484): +1 molecule over 0.479 is within noise.
- 2 Nov 05:56 IST: bgunshui 0.480 neural ensemble copy (ref 57880522) scored **0.479** = forecast; ties best 0.479.

- 2 Nov scan (discussions): host Marie Killian 27 Oct: submissions that use third-party weights whose training code/permitted data cannot be reproduced can be removed from the leaderboard regardless of rank (medal eligibility affected); 30 Oct (David Healey): pre-built datasets fine if free/open and not built from proprietary data; weight-only datasets (fpnet etc) still unanswered (thread 753662, 2 Nov). Keep our own reproducible 0.465 stack as one of the two final selections.

- 3 Nov (CASMI, "what are we missing"): our 17 Oct decision to reject the ICEBERG PubChem atlas as NIST-risky is OUTDATED: host Marie Killian ruled 14 Oct and 28 Oct that atlas spectra are usable as an in-silico library (ship only the subset used as a private Kaggle dataset plus the retrieval code). Blockers: the atlas download (iceberg-ms.mit.edu/dav/, per-formula MGF/MSP) needs a WebUI login (account with email/password, user must create it), the hidden test spectra are unknown so a subset cannot be precomputed for the true candidates, and the full atlas (91M structures) is far beyond a Kaggle dataset and this box (2.9 GB free). Leaders per earlier audit gained +0.008 per step from more ICEBERG/GLACIER coverage, so coverage is the lever, but the atlas route is only usable for a bounded candidate set. Leaderboard 3 Nov 12:36 IST: gold ~0.484 (rank 19), silver 0.480 (rank 231), bronze 0.479 (rank 462), top 0.525/0.521/0.519.

- 3 Nov atlas step 1+2 (CASMI): test.parquet holds the real 400 test molecules, so candidates are bounded: 5.5M PubChem structures in +-10 ppm windows, 12,341 distinct formulas; the 27 negative-only ("blind") molecules need 1,353 formulas / ~308k structures (<1 GB). Public atlas download works without login (/download_mgf?formula=..&adduct=..). Validation (59_atlas_negmode_validation.py, 80 negative-only train molecules): truth in atlas for 65/80 (81
- 3 Nov atlas steps 1+2 (CASMI): test.parquet holds the real 400 test molecules, so candidates are bounded: 5.5M PubChem structures in +-10 ppm windows, 12,341 distinct formulas; the 27 negative-only ("blind") molecules need 1,353 formulas / ~308k structures (<1 GB). The public atlas download works without login (/download_mgf?formula=..&adduct=..). Validation (59_atlas_negmode_validation.py, 80 negative-only train molecules): truth in atlas for 65/80 (81%); among same-formula isomers atlas-only binned cosine gives **MRR 0.317, top1 20%, top5 48%** (random MRR 0.02). Comparable to the honest unseen MRR 0.26 of the stack, so real signal; no baseline yet on the same molecules. Next: step 3 bench (needs per-molecule stack baseline for negative-only bench molecules; paired bootstrap P>=0.9 before any submission).

- 3 Nov atlas step 3 (CASMI, 60_atlas_bench_negative.py): bench477 has **18 negative-only molecules of 250** (7.2%, test 6.75%): 13 class 3, 3 class 2, 2 class 1. The stack (0.465 baseline) already hits 7 of 18 in its 25-list (MRR 0.299; five at rank 1). The atlas holds the truth for only **3-4 of 18** (class 3 truths are absent, as the host said class 3 is not in PubChem), and 4 of 18 have only the formate adduct (no [M-H]- spectrum, atlas has none for formate). Atlas rank of those present: 1 (class 3 b_QKXGRPHIBLICSC, which the stack misses: needs candidate injection), 2, 2 (both already rank 1 in the stack, so a naive fusion would lose them). Net: at most +1 molecule, with two plausible losses; n=18 cannot reach P>=0.9. **Verdict: not worth building for the leaderboard**; the 81% atlas coverage of 59 applies to class 1/2 molecules only, and the negative-only slice is mostly class 3. Stop-list: ICEBERG atlas as extra evidence. Not submitted.

- 3 Nov class-3 review: bench477 class 3 (130 molecules): truth in the stack's 25-list for 51 (39%), but top-1 for only 7 (baseline ranks of the 51 hits: rank 1 x7, 2 x7, 3 x18, 4 x4 ...), MRR 0.150 (baseline 0.465) / 0.139 (merged 0.477). Reordering is the lever in principle (39% hit vs 5% top-1), but every reorder tried is on the stop-list (next-steps rows 5, 11, 13, 14 learned fusion rejected as "trust PubChem" trap, 32, 33, 38) and the atlas (rows 34, 59, 60) adds at most one molecule. No new CASMI lever identified today; CASMI stays at 0.479 with the reproducible 0.465 stack as hedge.

- 3 Nov: **why our 0.480 copy scored 0.479**: bgunshui's notebook looks for its second scorer (final_model.pt + config_used.json) under /kaggle/input; it is his private training output, not attached to the public notebook, so on our account it prints "OURS: private model not found, keeping the public bundle" and runs as the 0.479 recipe. Fix: train our own scorer with his public training kit (bgunshui/casmi26-scorer-train-public: sanpier/casmi-spectrum-model-v1 kit, 80 epochs, T4, his run 8,503 s = 2.4 GPU h: data prep 6,181 s, holdout 56 epochs x 19 s, refit 52 epochs x 18 s; holdout MRR@25 0.660). Pushed aryanawasthiteykik/casmi-scorer-train-s2 (seed 20261103, otherwise verbatim). Then attach its output to the 0.480 copy so the 0.5 z(public) + 0.5 z(ours) ensemble actually runs, and the second model is reproducible from public code. Forecast for that submission: 0.480 (0.476-0.484).
- 4 Nov 02:49 IST: casmi-scorer-train-s2 done (8,804 s = 2.45 GPU h), final_model.pt written. Pushing casmi-480-neural-ensemble-copy v2 with it attached as a kernel source. Forecast 0.480 (0.476-0.484).
- 4 Nov: v2 visible log confirms "OURS BUNDLE READY ... casmi-scorer-train-s2" and "ENSEMBLE neural = 0.5*z(public) + 0.5*z(ours)". Submitted v2.
- 4 Nov 08:09 IST: v2 (0.480 recipe + our own scorer s2 as second member, ref 57961190) scored **0.474**: below forecast range (0.476-0.484) and below our 0.479. Our scorer (holdout MRR 0.641 vs bgunshui 0.660) made the average worse by ~0.6 molecule; same pattern as every own-model swap. Do not train more seeds (plan week 8-14 Nov item dropped). Best stays 0.479.
- 4 Nov scan: alexchilton/casmi26-lb-0-480-neural-ensemble attaches his public trained second scorer (alexchilton/casmi26-scorer-train-v80) and scored 0.480 on his account. Copying it (with his scorer, not ours). Forecast 0.479 (0.475-0.483). heliosli/casmi26-0-482 needs two private datasets (blank sources), cannot be copied.
- 5 Nov 04:14 IST: **casmi-ac480-copy (alexchilton 0.480 with his scorer-train-v80) scored 0.480** (forecast 0.479, 0.475-0.483). New CASMI best 0.480 = silver line; first time out of the 0.479 tie.
- 5 Nov scan: no CASMI notebook shows a real score above 0.480. rajrajak99 v2140 "484" and its hailonglv clone: 484 is a version label, no scored claim in text. ozertuu "0.479 LB+" header says "0.500 Apex SOTA" with no scored run. heliosli 0.482 still needs two private datasets. Host: no new reproducibility answer; EnvedaDark was never published (Healey 4 Nov). New audit posts (755961/755962: fusion demoted correct top-1, 0.94 local MRR did not transfer) say what our bench found.
- 5 Nov: user-directed copy of rajrajak99/casmi26-apex-v2140-balanced-tri-force-484 (no scored claim in text; "484" is a version label). Forecast 0.479 (0.473-0.484).
- 5 Nov, 61_extlib_hits.py (heliosli-style newer-library overlay, rebuilt from public mirrors): MassBank 2026.03 open records + GNPS community libraries 2026-10-22 = 615,350 spectra / 65,909 structures, of which only 6,876 spectra / 2,273 structures are NOT in train.parquet. Against the 400 real test molecules the best match has cosine 0.19: **0 test molecules hit at any threshold** (tau 0.6-0.9, k 3-8; false-hit rate on 5,676 training molecules 0.2-0.3%). MassBank and GNPS add nothing. heliosli's extlib also has **MSnLib 2026-07**, not in these mirrors; that is the only part left untested. Stop-list: MassBank/GNPS newer releases.
- 5 Nov, 61_extlib_hits.py --msnlib: MSnLib v8 MS2 (Zenodo 23494858, CC BY 4.0, derived from the 2026-07 release heliosli cites): 560,017 spectra / 46,907 structures, only **508 spectra / 56 structures not in train.parquet** (MSnLib is already in train). Best test match cosine 0.21: **0 test molecules hit at any threshold** (false-hit rate 0.05-0.07%). The whole newer-library route is dead: MassBank, GNPS and MSnLib together add no test match. heliosli's 0.482 gain, if real, must come from his private rr3 reranker, not from libraries. Line closed.
- 5 Nov 15:35 IST: rajrajak99 v2140 "484" copy (with bgunshui v5 scorer, ref 58008759) scored **0.470** (forecast 0.479, 0.473-0.484): below range. The title number was not a score. Best stays 0.480.

- 5 Nov web research (CASMI highways): no public write-up from the top teams (competition open; leader 0.525). Findings: (1) SpecBridge (arXiv 2601.17204, Hassoun lab): DreaMS encoder fine-tuned into frozen ChemBERTa space, cosine retrieval; claims +20-25% relative top-1 over strong baselines on MassSpecGym, MSnLib and Spectraverse; adapter weights MIT on Hugging Face (YinkaiW/SpecBridge, checkpoints for MassSpecGym / MSnLib / Spectraverse); GitHub code repo returned 404. A genuinely different neural signal from the FPNet/CLIP scorers in the 0.480 stack. (2) "MassSpecGym in the Wild" (arXiv 2606.19624, Coley/Pluskal, ICML 2026): retrieval benchmarks inflated by canonicalisation and candidate-set shortcuts, so paper numbers overstate. (3) Mathys-Gouverneur repo: test is 100% timsTOF, public references mostly not (median peaks 230 test vs 20 non-timsTOF); "nobody near the top generates molecules"; score lives in separating ~34 same-formula isomers; tail below rank 3 worth <=0.002. (4) MIST/DiffMS/MADGEN de novo: ~2-4% top-1, not competitive. arxiv.org and huggingface.co are blocked in this session's network.
- 5 Nov SpecBridge check (Kaggle notebook with internet, specbridge-recon): Hugging Face repo YinkaiW/SpecBridge returns "Repository Not Found" (removed or private), HF search for SpecBridge returns nothing, GitHub HassounLab/SpecBridge does not exist (clone asks for credentials = no such public repo; GitHub search 0 results). Weights and code are not publicly available, so it cannot be used and would fail the reproducibility rule anyway. Closed.

- 5 Nov **DreaMS retriever plan** (user go). DreaMS (pluskal-lab, MIT, weights Zenodo 10997887; host-approved checkpoint, 748882 A-1: deliver checkpoint + reference + own fine-tuning code). Steps and gates:
  1. dreams-setup notebook (internet on): install from source (pip install from git fails on Python 3.13 numpy pin, so --no-deps), fetch checkpoint, measure embedding speed on T4. Gate: embeddings come out (1024-d).
  2. Embed train (molecule-level, all instruments) + test on Kaggle GPU; cost from step 1 speed (expect 1-3 GPU h).
  3. Train a small head DreaMS-embedding -> fingerprint (CPU/GPU), score on the scorer kit's own holdout (27,383 validation identities, validation_candidates.npz) with the same MRR@25 diagnostic. Gate: holdout MRR@25 >= the public scorer's 0.660 (ours was 0.641 and lost on the LB).
  4. Only then add it as a third z-score member on ranks 2-25 of the alexchilton 0.480 copy (rank 1 untouched), forecast, one submission.
- 5 Nov DreaMS step 1 done (jobs/dreams/dreams_setup.py, 10 runs): pip install from git fails on Python 3.13 (numpy pin) so install --no-deps plus a loop that installs each missing module; matchms pinned 0.27.0 (--ignore-requires-python; newer matchms lacks ModifiedCosine); pyopenms stubbed (only used for mzML files); msml from roman-bushuiev/msml_legacy_architectures (needed to unpickle the checkpoint); TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 for the official Zenodo checkpoint. Result: 2,000 train spectra -> (2000, 1024) embeddings in 33 s (16.5 s per 1k at batch 32, including model load). Step 2 pushed (jobs/dreams/dreams_embed.py): all 1,213 test spectra + up to 2 spectra per training molecule (most peaks first), mean-pooled per molecule, batch 256, 7.5 h budget guard.
- 6 Nov 04:07 IST: **DreaMS step 2 done** (dreams-embed v2, batch 32): test 1,213 spectra -> (1213, 1024) in 26 s; train 624,654 spectra (up to 2 per molecule, most peaks first) -> **275,810 molecules x 1024** (float16, mean-pooled) in **2.25 GPU h**. Downloaded to /home/user/work/out_dreams_embed (train_emb.npy 565 MB). Next: step 3, head DreaMS -> fingerprint, scored on the scorer kit holdout against the 0.660 gate.
- 6 Nov **DreaMS step 3 gate: FAILED** (62_dreams_head.py, CPU 7 min). Frozen DreaMS molecule embeddings -> MLP -> 2048-bit Morgan probabilities, cosine ranking, on the exact 1,000 held-out molecules x 256 candidates our s2 scorer used (excluded from training): **MRR@25 0.218 (top-1 11.9%) vs s2 0.641 (top-1 50.5%)**, paired diff -0.423, P(better) 0.00. Gate was >= 0.660. Rank agreement with s2 is low (corr 0.23), but the model is far too weak to help an average. Frozen general-purpose spectrum embeddings lose most of what the task-trained scorer captures; closing a 0.42 gap would need full DreaMS fine-tuning (days of GPU), with every earlier own-model swap having lost on the LB. **DreaMS route closed; nothing submitted.** Cost: 2.25 GPU h + setup runs.
- 6 Nov: tie insight: 437 teams sit at exactly 0.480 (largest block), mostly the same public notebook, so they share one private score and the block is ordered by submission time. Any private gain above that block, however small, lifts us over all of them. Re-opened the cross-spectrum boost (58: +0.00128, P 0.83, 0 rank-1 changes) under this rule; ported as overlay/xspec_boost.py (bench replay reproduces 58 exactly: +0.00128, P 0.828, 45 anchors, 24 lists touched, rank 1 unchanged on 250/250) and appended as the final cell of the alexchilton 0.480 copy (casmi-ac480-xspec). Forecast public 0.480 (0.478-0.482); purpose is the private score.
- 6 Nov scan: host (Healey, 753662, added 4 Nov): "Intermediates also need to be reproducible. All solutions need to be fully provenanced to their data"; public weights OK only if the code and data that produced them are available. The 0.480 second scorer (alexchilton/casmi26-scorer-train-v80, public training kernel) fits that. New "0482" notebooks (popwenman gold apex v22, andrewoscar 482 roman frozen = romanrozen v9: Gold Apex V22 + timsTOF mass prior + isomer reranker) show no scored run above 0.480; not copied.
- 6 Nov: second rank-1-safe overlay, **instrument mass-accuracy prior** (idea from romanrozen v9 "MASS"; own rule): centre = median signed ppm of all rank-1 candidates (-1.49 on the bench, no answers used); ranks 2-25 with |ppm - centre| > 5 move to the end of the 25. Bench: answers beyond 5 ppm 0.6% vs other candidates 7.4%; alone +0.0007 (P 0.95, 3 up / 0 down); **xspec + mass: +0.0020, P 0.935, 7 up / 3 down, rank 1 unchanged on 250/250**. casmi-ac480-xspec v2 = 0.480 recipe + both overlays. Forecast public 0.481 (0.478-0.483).
- 6 Nov: casmi-ac480-xspec v2 submitted (3 left today). Visible run log: centre_ppm -859044 because the visible run ranks only 12 molecules and fills the rest with placeholder SMILES (CCO), so the median is garbage there; 0 lists reordered. In the hidden rerun every list is real, so the centre should be about -1.5 ppm, but v2 has no guard. Added a guard to the module for later versions: skip the mass prior unless |centre| <= 5 ppm.
- 6 Nov **63_tail_reranker.py** (rank-1-locked LightGBM lambdarank on ranks 2-25 of the bench477 merged lists, 5-fold molecule-grouped CV x 5 seeds): **+0.0386 MRR, P 1.00, 45 up / 5 down** (class 1 +0.009 with 2 down, class 2 +0.018, class 3 +0.063); headroom with rank 1 locked is +0.056. Ablations: without the neural features still +0.0386 (so not neural-score leakage); without PubChem rank +0.037; **without the position / internal-rank features (pos, main_rank, prefusion_rank, engine2_rank, pubchem_rank) it is -0.033**. The gain comes from reordering ranks 2-25 back toward the pipeline's earlier internal orders, i.e. undoing the heavier neural weight on ranks 2-25 that gained +0.004 on the public LB (w_mid 0.75). That is the known bench reversal (bench prefers the 0.465 baseline order: 0.517 vs merged477 0.498). Not trusted as is; only a leaderboard test can tell, and a real effect of this size would move the public score by several molecules.
- 6 Nov: tail-reranker LB test built (overlay/tail_rr.py; final model artifacts/63_final_model.txt trained on all 250 bench molecules, no neural features; gain importance: prefusion_rank 0.33, pos 0.22, main_rank 0.20, engine2_rank 0.07, rest small). Kernel casmi-ac480-tailrr = 0.480 recipe (alexchilton copy) + this reranker only, rank 1 locked. Pre-registered decision: public >= 0.485 means real (keep, refine); 0.480 or below means bench artefact (drop for good). Forecast 0.478 (0.470-0.500; wide because the bench says +0.039 and the LB history says reversal).
- 7 Nov 01:42 IST: **casmi-ac480-xspec v2 (0.480 recipe + cross-spectrum boost + mass-accuracy prior, rank 1 locked) scored 0.481** (forecast 0.481, 0.478-0.483) = **new CASMI best, first score above the 437-team 0.480 block**. v1 (boost only) scored 0.480 (forecast 0.480). Consistent with the bench: mass prior adds the visible step (+0.0007 bench alone, +0.0020 together).
- 7 Nov 06:27 IST: **tail-reranker TEST scored 0.459** (forecast 0.478, range 0.470-0.500; base 0.480). Pre-registered rule: <= 0.480 means bench artefact, drop for good. Result -0.021, far below: the bench reversal confirmed again (reordering ranks 2-25 back toward the earlier internal orders loses about 2.6 public molecules). Stop-list: any learned reorder trained on bench477 lists; internal-rank features. Bench direction is only usable for rules with a physical reason (mass error, cross-spectrum) and small size.
- 7 Nov scan: **alexchilton/casmi26-lb-0-481-gold-apex-v22-mass-prior** (162 votes) is now public at **0.481**: popwenman's Gold Apex V22 run verbatim = romanrozen v9 (zz3t3rn41 Gold Apex V22 + timsTOF mass prior, isomer reranker off) with popwenman's public scorer v80. The 0.481 tie will grow and our tie-break position will sink. Response: this base + **our cross-spectrum boost only** (its own mass prior already there) = casmi-ga481-xspec. Bench evidence for the boost: +0.0013, P 0.83, rank 1 locked. Forecast 0.481 (0.479-0.483); purpose: differ from the coming 0.481 crowd with an expected-positive tail change.
- 7 Nov: casmi-ga481-xspec v2 submitted (visible run in smoke mode so the evidence file exists; boost applied cleanly, rank 1 unchanged). Forecast 0.481 (0.479-0.483). NTL: ntl-669-q8pool v3 (our member, forecast 0.670) and ntl-ar670-copy (forecast 0.669) submitted.
- 7 Nov 14:50 IST scan: real notebook scores still top out at 0.481 (alexchilton, popwenman V22, romanrozen v9, zz3t3rn41 v2, hoowoo/duongynhi repros); heliosli 0.482 still needs private rr3. New claims without a scored run: popwenman V32 "0.482+" (= V22 base, V8='MASS', i.e. our ga481 base) and **easonyanyan V45 "0.484+"** = V22 base with V8='BOTH' (rank-1 MASS_LEAD rescue: leader > 5 ppm off centre and library < 0.9 swapped with a rank 2-3 candidate within tolerance; plus romanrozen's RR8 same-formula isomer LightGBM reranker, model embedded base64, sha-checked) and final-layer weights SPEC_W 0.10->0.05, NP_W 0.12->0.02, LIB_HIT_GATE 0.78->0.87. All inputs public. Every isomer reranker tested so far lost on the board (NX -0.058, our tail reranker 0.459), and MASS_LEAD touches rank 1. Submitting a copy as a test (casmi-ev45-copy). **Forecast 0.477 (0.465-0.484).**
- 7 Nov 14:55 IST: casmi-ev45-copy v1 submitted (smoke commit passed). Pending: ga481-xspec v2, ev45.
- 7 Nov 16:00 IST, push-for-0.485 session (all CPU, no submission):
  - **Leaderboard (4,920 teams):** silver cut = rank 246 (5%), which sits inside the 0.481 block (ranks 86-297); bronze cut = rank 492 (0.480). Us: rank 289 at 0.481. **0.482 = rank <= 86 = silver**, so +1 molecule step is enough; 0.485 = rank ~33. Teams >= 0.49: 15.
  - **Isomer exclusion (overlay/iso_exclusion.py, rule R3): rejected.** Placeholder test has 45 same-neutral-mass pairs of distinct molecules (spectral cosine max 0.65, median 0.16 -> distinct compounds, so one structure can be right for at most one). Bench477 has only 8 same-formula groups (18 molecules); 2 rank-1 collisions, both wrong for both molecules; R1 (collision swap) and R3 (confident-anchor removal) leave MRR at 0.4875. Blocker: anchors with library_max >= 0.87 are right only 32/45 (71%); a wrong anchor's structure can be the partner's true answer, so removal is not safe. Note: the visible test.parquet is a placeholder (md5 signature check), the hidden test is different.
  - **MASS_LEAD-style rank-1 mass rescue: no value.** Bench true answers sit within 4.5 ppm of the -1.49 ppm centre for 99% (max 7.3); leaders > 5 ppm with library < 0.9 = 2 of 250, 1 of them right; swap delta 0.0000.
  - Public kernels' commit outputs are placeholders (fast path), so no full 0.481 answer file is available to audit.
  - Live shots: ga481-xspec v2 (pending) and the V45 test (RR8 + MASS_LEAD). If V45 >= 0.482, rebuild with our xspec on top.
- 7 Nov, **hard-work plan toward 0.5 (user: no lucky runs).** Diagnosis: no offline check we own predicts the LB (tail reranker bench +0.039 -> LB -0.021; long-trained scorer random-holdout 0.718 vs 0.484 -> LB -0.019). Train has 183k timsTOF molecules (enveda-180, 1.15M spectra), the test's instrument; the scorer kit's holdout is random across all libraries ("not scaffold-disjoint").
  - **Step 1, yardstick** (jobs/casmi_yardstick.py, kernel casmi-yardstick, one T4 run ~4 h): public kit (sanpier v1 code, unchanged training) with two holdouts out of training: TS = enveda-180 molecules held out by Murcko scaffold (~4,000, all molecules sharing those scaffolds removed), queried with timsTOF spectra only; R = random 3% (the kit's style). Trains short 16x500 (sanpier v1), v80 80x500 (patience 4), long 40x2000 (no early stop); MRR@25/top-1 on 1,500 molecules per holdout, 256 candidates (true + same-formula + nearby mass). CPU smoke on an 81k-row slice passed.
  - **Pre-registered gate:** R must prefer long over short (reproduces the old offline view). The yardstick passes only if TS does NOT prefer long (long <= short + 0.01 on TS), i.e. it sees what the LB saw. Pass -> step 2 (scorer variants judged on TS: timsTOF-weighted sampling, CE conditioning). Fail -> stop; scorer work cannot be validated offline.
- 7 Nov 19:05 IST: **casmi-ga481-xspec v2 (alexchilton 0.481 base + our cross-spectrum boost) scored 0.481** (forecast 0.481, 0.479-0.483) = forecast, no gain at three decimals. Rank 209 of 4,926 (silver cut rank 246 = 0.481, bronze rank 492 = 0.480): **inside silver on the public board**. Private effect of the boost still unknown. casmi-ev45-copy pending.
- 7 Nov 19:15 IST, **yardstick result (casmi-yardstick, 3,022 + 1,125 + 506 s GPU = ~1.5 h training after prep): GATE FAILED.** Holdouts: TS = 4,583 scaffold-held-out molecules (4,035 timsTOF), eval 1,500; R = 8,077 random, eval 1,500; 256 candidates each (kit style).
  | config | TS MRR@25 (top-1) | R MRR@25 (top-1) |
  |---|---|---|
  | short 16x500 | 0.527 (38.7%) | 0.487 (34.3%) |
  | v80 80x500 (stopped ep 46, best 42) | 0.684 (56.3%) | 0.623 (48.5%) |
  | long 40x2000 | **0.770 (66.5%)** | **0.709 (58.6%)** |
  Pre-registered gate: pass only if TS does NOT prefer long (long <= short + 0.01). TS long - short = **+0.242**. Both holdouts rank long > v80 > short, the opposite of the LB (long lost 0.019 as sole scorer). So the timsTOF scaffold holdout does not see what the LB saw; a known-candidate retrieval diagnostic does not predict LB movement, whatever the holdout. **Scorer work cannot be validated offline this way; stop it.** Likely reasons (untested): LB gain/loss depends on how a scorer behaves inside the fused pipeline at rank 1 (diversity vs the other engines), not on its stand-alone retrieval of known structures; test truths include novel structures (class 3) that no known-candidate diagnostic contains. Outputs: /home/user/work/out_yard (yard_results.json, per-molecule ranks).
- 7 Nov 22:15 IST: **casmi-ev45-copy (easonyanyan V45 "0.484+" = 0.481 base + MASS_LEAD + RR8 isomer reranker + final-layer weights SPEC_W 0.05 / NP_W 0.02 / LIB_HIT_GATE 0.87) scored 0.457** (forecast 0.477, 0.465-0.484): below range, -0.024 vs the 0.481 base. Headline "0.484+" was a target, not a score (no scored run existed). Confirms again: every isomer reranker (NX -0.058, ours 0.459, RR8 here) loses on the LB; plus the final-layer re-weighting. Stop-list: RR8, MASS_LEAD (bench value was 0 anyway), POP/SPEC/NP weight changes on this base. Best stays 0.481.
- 7 Nov 22:40 IST, **lever 3 step A, pre-registered (64_loss_necessity.py).** Idea: a strong neutral loss needs a group that can lose it: CO2 (43.9898) needs [CX3](=O)[OX2]; HCOOH (46.0055) needs a carboxylic acid; NH3 (17.0265) needs N-H on a non-amide-free N; SO2 (63.9619) needs S(=O)=O; CH3OH (32.0262) needs a methoxy/methyl ester. Rule: if a loss peak (|m/z - (precursor - loss)| <= 0.005, relative intensity >= 0.3) is present in >= half of the molecule's spectra, candidates in ranks 2-25 lacking the group move to the end of the 25. Rank 1 never changes. Differs from script 55 (which moved rank 1 on loss evidence and lost). Diagnostic first: violation rate of the TRUE answers vs other candidates on bench477 (mass-prior style; need true-answer rate ~0). Accept only if: true-answer violation <= 1%, P(better) >= 0.9, rank 1 unchanged 250/250. Forecast bench gain +0.000 to +0.002; LB = bench / 3-10, i.e. invisible at three decimals; purpose is the private score.
- 7 Nov 22:55 IST, **lever 3 step A result (64_loss_necessity.py): REJECTED on the pre-registered criterion.** Bench477: a strong loss is observed in only 16 of 250 molecules; TRUE-answer violation **4/18 = 22%** (needed <= 1%) vs other candidates 137/421 = 33%, so the "necessary group" does not separate answers from decoys (CO2: true 1/7 vs other 49/164; HCOOH: 3/8 vs 86/185, i.e. the 46.0055 peak is not specific to acids; NH3: 0/3 vs 2/72 but n too small). 12 lists reordered, MRR 0.48754 -> 0.48744 (-0.0001), P(better) 0.00, up 0 / down 1, rank 1 unchanged 250/250. Stop-list: functional-group necessity from neutral losses. Lesson for the next checks: demand a true-answer violation rate near 0 on enough molecules (mass prior had 0.6% over 250) BEFORE building any reorder; coverage under 10% of molecules cannot move the score.
- 7 Nov 23:30 IST, **lever 3 steps B-D (the three checks), all closed, nothing to submit.**
  - **Check 2, fragment formula / RDBE -> reduced to a formula-consistency prior and tested on bench477:** rank-1 formula equals the true formula for 245/250 (98.0%), 130/130 at library_max >= 0.7 and 45/45 at >= 0.87; truth in ranks 2-25 (57 molecules) has a different formula from rank 1 in **0/57** cases, yet 686 of 5,996 candidates (11.4%, in 100 molecules) in ranks 2-25 differ in formula. A rule "different-formula candidates go behind same-formula ones when rank 1 is confident" changes **0 lists** at tau 0.7 / 0.87 / 0.95: the pipeline already orders same-formula first, so there is no headroom (max gain 0.0000). RDBE of fragments cannot separate same-formula isomers, which are 89% of ranks 2-25.
  - **Check 1, fragment-explained intensity (MetFrag-lite):** not re-run, already covered by lever-table rows 25 (fragment check, in the 0.477+ base), 32 (our LightGBM isomer reranker with fragment features: real lists -0.006, P 0.14, 5 class-1 losses) and 37 (parsimony tuning flat). Dead as an extra overlay.
  - **Check 3, collision-energy trend:** only a candidate-dependent version is meaningful and that needs a fragment model per CE, i.e. a variant of check 1 / the public "stepped-CE GLACIER" dead end (-0.004). 149 of 250 bench molecules have >= 2 CE settings, so coverage would be partial. Not built.
  - **Lever 3 verdict:** loss-necessity (22% true violation), formula prior (zero headroom), fragment checks (already in the base) all fail; the two overlays that transferred (cross-spectrum boost, mass prior) are the only physical rank-1-safe rules found. No new overlay. Remaining CASMI routes: a stronger public base (heliosli's private rr3 data), or class-3 candidate generation (every attempt so far flat or negative).
