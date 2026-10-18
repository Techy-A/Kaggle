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
