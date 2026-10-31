---
tags: [kaggle, casmi26, plan]
date: 2026-10-29
---
# CASMI26: next steps plan (29 Oct 2026, IST)

Research and planning only. Nothing installed, downloaded, pushed or submitted for this note. Back to [[CASMI26]].

**Where we stand (29 Oct 16:15 IST).** Best public 0.477 = exact copy of busyaprime's "two 0.472 fixes merged the
right way" (`/home/user/work/push_c477`, ref 57736039). Public top 0.478 (heliosli; needs a private rr4 dataset, not
copyable). The 0.477 plateau has many forks. About 125 public molecules: one rank-2 -> 1 move = 0.004, one miss -> rank
1 = 0.008, modelled SE of the public mean about 0.04. Gold line about 0.471-0.473. GPU quota 0.83 h until 31 Oct
05:30 IST (one 0.477-sized run costs about 0.45 h visible).

## 1. Every lever tried (ours and the public's), with measured effect
LB = public leaderboard change against the base it was applied to. "Bench" = our offline benches (np, unseen, thin
class-1 genfix bench, isomer benches). Rule of thumb from all of it: **LB = bench / 3-10, and the sign is not reliable
for anything that changes list composition.**

| # | Lever | Who | Bench | LB | Verdict |
|---|---|---|---|---|---|
| 1 | Exact pinned copy of best public notebook | us | - | reproduced every time but once (0.443, 0.441, 0.454, 0.456, 0.464, 0.468, 0.472, 0.477; the miss: 0.465 vs 0.466) | **works**; unpinned copies lost 0.014 |
| 2 | Docker image swap (same code) | guhongbin v459 / Kino | - | +0.008 once (0.446 -> 0.454), 0 on v14 | not a lever on current code |
| 3 | Own engine merged by RRF into strong pipeline | us | - | +0.001 / -0.001 / -0.007 | dead |
| 4 | Fingerprint re-rank inside fork's final 25 | us | - | 0.000 | dead |
| 5 | Generated class-3 candidates at rank 2 | us | 42% answers generated offline | -0.001 | dead |
| 6 | Popularity prior | us | +0.50 MRR | +0.075 | worked, shrinks 7x |
| 7 | Agreement-only promotion | us | - | -0.004 | dead |
| 8 | PubChem gate rel 150 -> 100 / 250 | us, public | bench +0.002 (wrong sign) | -0.006 / -0.006 | keep 150 |
| 9 | Engine-2 rank-3 insert (E2R3) | us | -0.003 | -0.001 | dead |
| 10 | Engine-2 max_depth 7 | us | - | +0.003, +0.004, 0.000, **-0.008** | base-dependent, dead on current base |
| 11 | gen20 (wider generation, ungated) | us | +0.033 np / +0.004 honest | **-0.010** | dead; re-scores list-relative features (top-5 changed in 48-60%) |
| 12 | Gated gen20 (lib_max < 0.7) | us | keeps class 1 | not submitted | no evidence |
| 13 | Features-frozen gen20 (genfix δ 2.0) + candidate-level PubChem gate | us | 0/270 rank-1 losses, c3 +0.083 (leaky) | 0.465 (vs 0.464 base) | +0.001, noise |
| 14 | Learned fusion (LambdaRank over E004 inputs) | us | +0.015, P 0.98 | not submitted | rejected: learns "trust PubChem" (rel100 trap) |
| 15 | NPAtlas / bigger NP catalogue | us | 485 new structures of 36k | - | not worth it; DB coverage saturated |
| 16 | Dedupe by metric key | us | 0 duplicates | - | already done |
| 17 | Forward-model coverage (N1A proxies, N1 challenger) | public | - | **+0.008 x3** | the only big repeatable lever; positive adducts now fully covered |
| 18 | [M+Na]+ -> [M+H]+ GLACIER shift | public | - | -0.008, -0.008, -0.013 | dead |
| 19 | Candidate-level PubChem promotion gate | sanpier | - | +0.001 | noise-level |
| 20 | Neural CLIP model RRF (w 0.15-0.25, offset 3) | sanpier / our copy | holdout MRR 0.39 alone | +0.001 (our copy 0.465 = 0) | noise-level |
| 21 | RRF offset 3 -> 1 / 10; w 0.75 everywhere | alexchilton | - | +0.001 / -0.020 / -0.047 | parameters are sharp; heavier loses |
| 22 | Leader-safe consensus V2 (rank 1 at 0.25, ranks 2-25 at 0.50) | romanrozen | - | +0.002 (0.466 -> 0.468) | works (protects rank 1) |
| 23 | w_mid 0.75 (also 0.70, 0.80 all 0.472) | alexchilton forks | - | +0.004 | works, flat in 0.70-0.80 |
| 24 | Rank-1 neural weight 0 / 0.50 / 1.00 | public forks | - | 0.472 / 0.448 / 0.359 | **moving rank 1 with the neural model loses** |
| 25 | Tamrazov fragment check on V2 order | Spectral Parallax | - | +0.004 (0.468 -> 0.472) | works |
| 26 | Fragment check + w 0.75 stacked plainly | busyaprime e1 | - | 0.472 (no add) | contender window changes |
| 27 | Fragment check leader moves from V2 + ranks 2-25 from 0.75 (merge order) | busyaprime m1 | - | **0.477** | current best; about one net leader move |
| 28 | Provenance reranker | Tamrazov / busyaprime e2 | - | 0.000 | dead |
| 29 | Long-trained neural model (Ahmed) | busyaprime e3/e4 | holdout 0.718 vs 0.484 | **-0.019** | offline neural gains do not transfer |
| 30 | NX isomer reranker weight 2 | busyaprime e6 | - | **-0.058** | dead |
| 31 | heliosli rr3 / rr4 overlay (permutation of non-train candidates, z(-log rank) + α z(reranker), lib_max >= 0.9 untouched) | heliosli | - | +0.007 on 0.468 line, rr4 0.478 | needs private dataset |
| 32 | Our fragment-feature LightGBM isomer reranker | us (script 51) | isomer sets +0.009; real lists **-0.006**, P 0.14, 5 class-1 rank-1 losses | not submitted | dead (redundant with 25/20) |
| 33 | FIORA-OS [M-H]- reranker ignoring base rank | us (52/53) | random isomers +0.11 MRR P 1.00; real lists **-0.163**, 19 class-1 rank-1 losses | not submitted | dead as built; base order already RR 0.94 within slot vs FIORA 0.61 |
| 34 | Negative-mode forward-model gap | us (50) | 27 of 400 molecules ([M-H]- / formate only) | - | headroom < 1 public molecule |
| 35 | enveda-180 rows = verbatim test spectra with decoy-looking labels | us | label in top 25 for 12/400 | - | **trap**, never use |
| 36 | Public dead ends (FRIGID ungated, JESTR, CFM-ID LOTUS, stepped-CE GLACIER, MSEntropy, ChEBI+LIPID MAPS, blind PubChem isomer expansion) | public | - | -0.009, -0.022, 0 to -0.018, -0.004, 0, -0.005, -0.13 | dead |
| 37 | MetFrag-lite parsimony tuning (2-bond x0.6, linear int., 0.005 Da) | NeckBeard | +0.065 isomer panel | flat | dead as a main bet |
| 38 | Rule-based biotransformation of analogs for class 3 | forum 753400 | few hits | - | dead |

**Three lessons that matter most**
1. **Rank 1 is where points are won and lost, and the base pipeline's rank 1 is already very good.** Every gain since
   0.466 protects rank 1 (rows 22-27); every lever that let a new signal move rank 1 freely lost (rows 24, 29, 30, 32,
   33). Our two rerankers looked strong on random-isomer benches and lost on real lists because the base order already
   ranks isomers at RR 0.94. Any new signal must enter as a small term on top of the base rank, never as a replacement.
2. **Our benches cannot approve list-composition changes.** gen20 (+0.033 bench, -0.010 LB), rel100 and learned
   fusion all had the wrong sign. The benches are good at diagnosis and at catching class-1 rank-1 losses; the zero
   class-1-loss gate is the one filter that would have stopped every LB loss we caused.
3. **The public LB cannot separate the plateau.** One molecule is 0.004 against SE 0.04; 0.472 -> 0.477 is about one
   molecule. Copies are reproducible (5 of 6), so the 0.477 is a real public number, but its private advantage over
   0.472 is closer to a coin with a small tilt than to a sure thing.

## 2. What the top notebooks do, layer by layer (read from pulled sources)
- **Base (0.465, base64-embedded, SHA pinned):** dmitriigluzdov v16 + sanpier's candidate-level PubChem gate. Library
  search, PubChem channel, GLACIER + ICEBERG (N1 challenger), engine 1 (Ahmed v4b) + engine 2 fusion.
- **Neural RRF (sanpier, 0.466):** CLIP spectrum/fingerprint model trained only on train.parquet; score
  1/(3+b) + w/(3+n).
- **Leader-safe V2 (romanrozen, 0.468):** rank 1 at w 0.25, ranks 2-25 at w 0.50; optional tail rescue (V3).
- **0.75 order (alexchilton forks, 0.472):** ranks 2-25 at w_mid 0.75.
- **Fragment check (Tamrazov Spectral Parallax, 0.472):** untrained MetFrag-lite. Rank 1: if lib_max < 0.87 and >= 3
  spectra, a same-formula candidate from ranks 2-6 can lead if it explains clearly more intensity, is higher on the
  neural model and has engine top-5 or GLACIER/ICEBERG support. Plus up to 2 adjacent swaps in ranks 2-11 and one
  outside candidate at rank 25.
- **Merge order (busyaprime m1, 0.477):** run the fragment check on the V2 order; keep only its leader moves, put the
  old leader second, the rest from the 0.75 order. The gain is the merge order, not a new signal.
- **heliosli rr3/rr4 (0.475/0.478):** permutation-only overlay on non-train candidates, z(-log rank) + α z(GNN +
  transformer fragment scorer through five fold-safe FPNets); private dataset.

## 3. Ideas from the web and forum, against our constraints
| Idea | Expected public gain | Cost | Risk | Licence | Validation |
|---|---|---|---|---|---|
| A. Rank-preserving FIORA fusion on [M-H]- / formate-only molecules: z(-log base rank) + α z(FIORA), α fixed in advance (0.25), rank-1 swap only with margin and lib_max < 0.87 | +0.000 to +0.001 (affects <= ~8 public molecules) | CPU replay of existing 53 artifacts (minutes); later a Kaggle dataset + 0.45 h GPU | the same 214 lists were used for the no-go, so α must be fixed before looking; a fresh slice is needed | FIORA MIT, OS weights on MSnLib, host-allowed (751373) | genfix-bench [M-H]- lists, paired bootstrap P >= 0.9 **and** 0 class-1 rank-1 losses |
| B. Copy heliosli if the rr4 dataset goes public (title: "happy to make the dataset public if it helps") | +0.001 (0.478) | 0.45 h GPU | low; private gain unknown | check dataset licence | exact-copy rule (reproduced 5/6) |
| C. Ensembling distinct 0.477 forks (kunaldesale MID, rajrajak Apex PXM, Spectral Parallax current) | 0 to +0.004 | read diffs (CPU), one run | blend-composition changes failed 10x in the SHL project and RRF of near-identical lists is a no-op | as the forks | only if a fork adds a *new* signal; otherwise skip |
| D. Class-3 recall by de novo generation (DiffMS, ICML 2025, formula-constrained graph diffusion; test-time-tuned LMs, arXiv 2510.23746) | ~0 by the deadline (published top-1 on MassSpecGym is a few %) | days of GPU training on train.parquet | very high; list-composition changes lost every time | DiffMS MIT; pretraining data must be checked | would need a leaderboard A/B; not affordable |
| E. Cross-ion analog search (MS2DeepScore 2.0) for [M-H]- molecules | ~0 (DB coverage saturated, row 15) | install + embeddings | medium | Apache-2.0 (verify); host-allowed | coverage count first |
| F. Private-split robustness when choosing between near-identical submissions | protects ~0.004-0.008 private | none | none | - | reasoning below (multiple-comparisons-with-the-best view: the 0.472-0.477 group is statistically one winner set) |

## 4. Ranked plan (at most 5 items, step -> verify)
Honest odds first: P(any item below beats 0.477 on the public LB) is about 15-20% in total. The plateau is a
noise wall; build for the private split.

1. **Lock the final selection now (no compute).** -> verify: two submission IDs written under "Final selection"
   in [[CASMI26]]; re-check only if a new lineage (not a fork) reaches >= 0.480 twice. Expected gain: protects the
   0.477 against the private reshuffle; cost 0.
2. **Watch heliosli's dataset; copy if it becomes public.** -> verify: dataset visible with a licence that passes the
   host rules, exact copy scores 0.478 +- 0.001. Expected +0.001 public, P(dataset released before the deadline) about
   30%. Uses the 0.83 h quota only then.
3. **Rank-preserving FIORA fusion, pre-registered.** Write α = 0.25, margin and gates into the script header
   *before* running; replay on the 214 [M-H]- genfix-bench lists from `artifacts/53_*` (CPU, minutes). -> verify:
   P(better) >= 0.9 and zero class-1 rank-1 losses; if it passes, a second check on a slice not yet used (e.g. the 31
   lists of script 52 run through the guard). Expected +0.000 to +0.001 public, P(pass) about 20%. Needs user approval
   for any Kaggle dataset/push.
4. **Read the diffs of the other 0.477 forks (kunaldesale MID, rajrajak PXM, current Spectral Parallax).** CPU only.
   -> verify: a table of executed-line differences vs busyaprime m1; act only if one adds a new signal (not a weight).
   Expected 0 to +0.004, P(useful) about 10%.
5. **Stop list (no work):** class-3 generation in any form, wider PubChem, depth 7, learned fusion, any reranker that
   can move rank 1 without the base rank as a prior, enveda-180 lookups. -> verify: nothing in this list gets GPU time.

## 5. Final selection (recommendation)
Kaggle scores the better of the two selected submissions on the private split (~275 molecules), so the second slot
should be a strong submission that **fails differently**, not the next-best near-duplicate.
- **Selection 1: the 0.477 copy (57736039).** Highest public score, reproduced exactly, same base as everything else.
- **Selection 2: the Spectral Parallax copy, 0.472 (57699799).** It differs from the 0.477 exactly where the 0.477's
  extra +0.005 comes from (ranks 2-25 at 0.50 instead of 0.75, and its own swaps/tail), so if the 0.477 merge was a
  one-molecule public fluke the second slot still holds the 0.472 lineage. It shares the base and the leader-safe rank
  1, so its expected private score is close to the 0.477's.
- Do **not** use the slot for our own 0.465 stack or the 0.468 copy (lower and highly correlated), or for any untested
  variant. Swap selection 2 only for a different lineage that has scored >= 0.477 publicly (e.g. a public heliosli
  copy at 0.478, which would then take selection 1 and move the 0.477 to selection 2).
- Licence: both use the dmitriigluzdov assets (CC BY-NC-SA, allowed by host since 17 Oct) and sanpier's
  train.parquet-trained model; both are copies, rank-only, not honestly validated models of ours.

Sources: [DiffMS (arXiv 2502.09571)](https://arxiv.org/html/2502.09571v2), [test-time tuned LMs for MS/MS
(arXiv 2510.23746)](https://arxiv.org/abs/2510.23746v1), [FIORA (Nat Commun 2025, PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11889238/),
[MS2DeepScore 2.0 cross-ion preprint](https://www.biorxiv.org/content/10.1101/2024.03.25.586580.full.pdf),
[Leaderboard Lottery / MCB (ICML 2026)](https://icml.cc/virtual/2026/80012), [Reducing overfitting in challenge-based
competitions](https://arxiv.org/pdf/1607.00091), pulled notebooks under `/home/user/work/pull_*`.

## Step 4 result (2026-10-29 16:30 IST): diffs of the public 0.477 notebooks vs busyaprime
Pulled to `/home/user/work/pull_f477_*`, code cells diffed against `pull_c477` (busyaprime 0.477, variant M). All seven
use busyaprime's seven datasets unchanged (no private/empty `dataset_sources`, no models, no kernel sources); kunal adds
only `metric/rdkit-2026-3-3-wheel`, not referenced in code. No enveda-180 references anywhere. Six are forks of
busyaprime; Spectral Parallax is now also a busyaprime fork plus one layer.

| notebook | fork of | difference | class |
|---|---|---|---|
| pjmathematician/casmi26-e303-pub477-gpu | busyaprime | byte-identical code | none |
| cngnguynhuuz/casmi26-fork-m477 | busyaprime | byte-identical notebook (same as pj) | none |
| rajrajak99/apex-v1300 | busyaprime | `COMMIT_MODE='fast'` with dummy commit file, guards around stages; scoring path unchanged | none (plumbing) |
| rajrajak99/apex-v1320 | apex-v1300 | `w_top=0.0` instead of 0.25 | (a) tweak |
| youyouwou/v120rel150 | busyaprime | `IS_RERUN=True` forced; InChIKey-14 dedup + backfill from CUSTOM/V2 | (a) rule; no-op on our lists (lever 16: 0 duplicates) |
| kunaldesale2408/mid | busyaprime | `w_mid=0.80`, `rescue=1`, 25 outsiders; merge gate removed (leader move applied even if V[0]!=C[0] or sets differ, V[0] not kept at rank 2); same K14 dedup | (a) tweaks + gate loosening |
| romantamrazov/spectral-parallax (new) | busyaprime 0.477 | adds "disagreement rescue": where the fragment check moved rank 1 but the 0.477 merge discarded it, swap to the physical leader only if same formula, leader `library_max`<0.955, neural z-margin>=0.04, cosine margin>=-0.002, >=1 source rank<=15, >=3 acquisitions, and per-acquisition fragment-ion and **neutral-loss** margins pass (median/mean/fraction gates) | (b) new evidence source (neutral-loss matching), heavily gated |

Public score for all is 0.477 = base, so on the public split the extra layers changed 0 net molecules.

**Recommendation.** Only one candidate signal: the Parallax neutral-loss + multi-acquisition ion gate for isomer
leader swaps. Worth one CPU replay, nothing on Kaggle. Validation: port `choose_disagreement`/`neutral_loss_score`
(rewritten, not executed from the pull) to our bench; pre-register its gates as published; replay on the
leader-disagreement lists of the genfix bench (same 214 [M-H]- lists as step 3, plus [M+H]+ if available); accept only
if paired bootstrap over molecules gives P(better) >= 0.9 *and* zero class-1 rank-1 losses, then confirm on an unused
slice. Expect few or zero firings (it fired on no net public molecule). Ignore the K14 dedup (already 0 duplicates),
w_top/w_mid/rescue tweaks and kunal's gate removal (a looser version of a gate we chose deliberately). Final selection
in section 5 stands.

## Bench option 1: a faithful bench from the 0.477 pipeline (planned 30 Oct, IST)
**Why.** Every own CASMI test so far ran on lists from the old V22 pipeline (enveda-casmi26/bench, build_bench.py), not on the 0.477's. That made the Parallax replay impossible and likely distorted the reranker and FIORA tests. The bench must be the real pipeline on molecules with known answers.

**Reuse.** build_bench.py already turns a notebook into a bench: the 250 enveda-np-examples natural products (1,184 timsTOF spectra) with answers hidden by simulated class (class 1 28%: only the query spectra removed; class 2 20%: every spectrum of the structure removed; class 3 52%: also removed from every candidate source), leak controls (leak-free FPNets, GBMs fit without the query set; remaining exposure: PubChem-channel FPNets A+B, pv GBM rows), plus make_unseen.py's leak-free Enveda-180 Zenodo slice (structures absent from train) for classes 2/3. Never use the enveda-180 rows of train.parquet.

**Steps (step -> verify).**
1. CPU, before the reset: build `casmi26-bench477` from push_c477. The 0.477 runs the frozen 0.465 pipeline from an embedded base64 notebook (FROZEN_B64) in a subprocess: decode it, inject the BENCH cell (adapted from build_bench.py), re-encode, update any SHA-256 pin; make the notebook save every intermediate (final list per variant V0/V1/V2/CUSTOM and the merged 0.477 order, the fragment-check list, neural_score/neural_cos/neural_ok, source ranks, library_max, frozen_candidate_evidence.parquet). -> verify: the cells compile; the decoded and re-encoded frozen notebook differs from the original only by the BENCH cell; a dry run of the BENCH cell on CPU (dryrun_benchcell.py) hides the right rows per class.
2. GPU, after 31 Oct 05:30 IST (quota back to 30 h): push the private kernel (never submitted), full run, no fast path. Estimate 2-4 h (the hidden 400-molecule run takes several hours; the bench is ~250 molecules); hard stop at the 9 h limit. -> verify: run COMPLETE, bench_results.json and the intermediates present.
3. CPU: score every variant from the saved intermediates. **Calibration gate:** the bench must order the known public scores the same way: V0 (0.465) < V1 0.25/offset 3 (0.466) <= V2 0.50 (0.472) and 0.75 (0.472) < merged 0.477. If it does not, the bench is not trusted and we stop.
4. Only then use it for our own ideas, each pre-registered: FIORA only on lists with no simulator evidence (negative-mode-only molecules); the Parallax neutral-loss rescue with all its gates; the engine-1 ranker retrain with source flags. Acceptance as before: paired bootstrap P(better) >= 0.9 and zero class-1 rank-1 losses; LB forecast = bench gain / 3-10.

**Risks.** The SHA pin or subprocess layout may resist the injection (fall back to injecting into the frozen notebook run directly); runtime near the limit; leakage left in the PubChem-channel FPNets makes absolute scores read high (compare variants, not absolutes); the calibration gate may fail, which is still a useful answer.
**Cost.** About 2-4 GPU h of 30, one private kernel, no submissions.

### Step 1 result (2026-10-31 03:30 IST): bench477 built and verified on CPU, not pushed
**Built.** `enveda-casmi26/bench/build_bench477.py` -> `/home/user/work/push_bench477` (`casmi26-bench477.ipynb` + `kernel-metadata.json`: id `aryanawasthiteykik/casmi26-bench477`, title "CASMI26 Bench477", private, GPU T4, internet off, original image pin `sha256:9a6bcfda…`, the same 7 datasets + competition as push_c477; no extra source: the leak-free `ours_fpnet_*.pt` ship inside the v4g assets, and adding megayak's dataset would double that glob).

**Changed in the frozen 0.465 pipeline** (decoded from FROZEN_B64, re-encoded as compact JSON, which reproduces the original bytes exactly; 54 of 59 cells byte-identical, cell count unchanged):
- cell 30 (SMOKE switch), +76 lines at the end: build_bench.py's BENCH block (overrides COMP, IS_RERUN = True, ICE_BUDGET 5400, preflight ids) plus `bench_dropspec.json`. Appended to cell 30 instead of a new cell so the outer stage labels (keyed on frozen cell numbers) stay right.
- cell 32 (engine 2): V22 patches (class-3 drop from COCONUT/bio, class-2 re-add, ranker rows without np, leak-free FPNets). The class-3 drop now runs after PubChemLite: this engine adds PubChemLite after ChEBI/LMSD, so the V22 anchor would have let PubChemLite re-add the answers.
- cell 34 (PubChem channel): class-3 answers count as pool structures; the gate never sees them (V22 patches; the first one needed escaped quotes here).
- cell 36 (new leak control): the external reference spectra (MassBank/MoNA/GNPS/MSnLib linked to train SIDs) of class-2/3 answers and twins are hidden.
- cell 37: engine 1 uses FPNet A instead of fpnet_full1; class-3 answers hidden from the candidate window.

**Pins.** Only `FROZEN_SHA256` (outer cell 9, also passed to runner.py) pins the frozen source: `d1e757c2…` -> `fa7450f7…`. MODULE_SHA256, MODEL_PINS, the fragment-check source hashes and the frozen notebook's own hashes (fold-safe checkpoint, prior helper, N1 embedded source, METRIC_SOURCE) are untouched and still hold.

**Outer notebook** (23 -> 27 cells): title cell; run cell asserts COMMIT_MODE != 'fast' and that BENCH ran, then points COMP/SAMPLE_IDS at `bench_comp` (the fragment check reads COMP/test.parquet); copies of the fragment-check list (`variants/submission_physical_V2.csv`) and the merged list (`variants/submission_merged477.csv`); a final cell scoring baseline0465, public0466, V0-V4, CUSTOM (0.75), physical_V2 and merged477 into `bench_results.json` (MRR, per class, top-1, hit@25, per-molecule rr), `bench_candidate_truth.parquet` (is_answer for every listed and outside candidate) and `bench477_intermediates.tar.gz` (variant lists, neural features with neural_score/neural_cos/neural_ok, frozen_candidate_evidence with source ranks and library_max, outsider features, reports, e004_candidate_inputs.json), also printed as `BENCH_BLOB` lines; `extract_log.py` now decodes them.

**Verified (CPU).** All 16 outer, 46 frozen and the embedded module code strings compile. The dry run of the BENCH block on the real train.parquet (311 s): answers 70/50/130 (class 1/2/3); pseudo-test 1,184 spectra / 250 molecules in the test schema; 0 np rows and 0 class-2/3 spectra left (42,134 rows removed, 189 ik14 incl. 9 twins); all 70 class-1 structures keep their 14,823 other-library spectra; answers all from enveda-np-examples, none has an enveda-180 spectrum and no enveda-180 row is touched; 130/130 class-3 answers in the v4 pool, all hidden from the window; 189 train SIDs hidden from the external references, class 1 kept. The scoring cell gives the hand-computed MRR on mock lists; the blob round-trips through extract_log.py.

**Runtime estimate.** V22 bench (gen 20) reached the end of its main pass at about 3.4 h on the same 250 molecules; this pipeline adds the N1 challenger (1,200 s cap), external references, neural scorer, outside candidates and the fragment check (1,500 s cap). Estimate 3.5-5 h, ICEBERG capped at 5,400 s; the notebook's own guards skip the optional stages after 7.5 h, and Kaggle stops at 9 h.

**Calibration targets for step 3** (public LB): V0 0.465 < V1 0.466 <= V2 0.472 and CUSTOM 0.75 0.472 < merged477 0.477; public0466 must equal V1 (the notebook asserts it). If the bench does not order them so, stop.

**Risks.** Remaining exposure: PubChem-channel FPNets A+B, the pv GBM rows, and train_structs' n_spec/libmask (row-aligned pool features built from the full train) can favour class-1 answers, so compare variants, not absolutes. The fragment check's 1,500 s cap may cut it short on 250 molecules (it then keeps the V2 list; check `time_budget_exceeded` in the report). The external-reference pins (train_structs, pool_meta hashes) were not checkable locally; if they fail, the pipeline drops the extension as in any run. An outer failure after the pipeline loses the scoring cell; the per-stage files stay in /kaggle/working.

**Step 2** (after 31 Oct 05:30 IST, user's go): `kaggle kernels push -p /home/user/work/push_bench477`, never submit; then `kaggle kernels status aryanawasthiteykik/casmi26-bench477` until COMPLETE, `kaggle kernels output aryanawasthiteykik/casmi26-bench477 -p <dir>`, and `python enveda-casmi26/bench/extract_log.py <log> <txt> <json>` if outputs are blocked.

## Step 2 (31 Oct 05:42 IST)
GPU quota reset (30 h left, next reset 7 Nov 05:30 IST). Pushed the private bench kernel aryanawasthiteykik/casmi26-bench477 v1 from /home/user/work/push_bench477 (never to be submitted). A tracked background wait downloads the outputs to /home/user/work/out_bench477 when it completes; expected 3.5-5 h, so about 09:15-10:45 IST. The CASMI 0.478 copy (casmi-kunal-478-copy) was submitted 31 Oct 03:53 IST (ref 57818217), pending.

## Step 3 result: calibration gate FAILED (31 Oct 09:4x IST)
casmi26-bench477 v1 completed (pushed 05:42 IST, about 3.8 h; fragment-check time budget not exceeded). Outputs in /home/user/work/out_bench477; bench_results.json copied to enveda-casmi26/artifacts/bench477_results.json. 250 molecules (70 class 1, 50 class 2, 130 class 3).

| Variant | Public LB | Bench MRR (all) | c1 | c2 | c3 |
|---|---|---|---|---|---|
| baseline0465 / V0 | 0.465 | 0.5169 | 0.948 | 0.868 | 0.150 |
| public0466 / V1 | 0.466 | 0.5147 | 0.943 | 0.860 | 0.151 |
| V2 (0.50) | 0.472 | 0.5113 | 0.937 | 0.858 | 0.149 |
| CUSTOM (0.75) | 0.472 | 0.5045 | 0.932 | 0.857 | 0.139 |
| physical_V2 (fragment check) | - | 0.5035 | 0.930 | 0.829 | 0.149 |
| merged477 | 0.477 | 0.4978 | 0.925 | 0.834 | 0.139 |

Paired bootstrap over molecules: merged477 vs baseline -0.019 (P(better) 0.00), vs V2 -0.0135 (0.00), vs CUSTOM -0.0067 (0.00); V2 vs V1 -0.0034 (0.06); V1 vs baseline -0.0022 (0.18).
The bench orders the variants exactly in REVERSE of the public leaderboard: every step that gained publicly (neural weight up, fragment check, merge) loses on this bench. Pre-registered rule: if the bench does not reproduce the LB order, it is not trusted and we stop. So no own idea is tested on it.
Likely reasons (not verified): (1) the bench pseudo-test (enveda-np-examples, class mix simulated by hiding spectra) differs from the hidden test in exactly the way the public gains exploit (e.g. the neural spectrum model and fragment check gain on hidden-test molecules that are different from these natural products); (2) leakage left in the PubChem-channel FPNets and pool features favours the baseline's own ranking on these molecules; (3) the public gains are about one molecule each (0.004) on ~125 public molecules, so the public order itself may be noise, and the bench may simply be right about private. Either way, neither this bench nor the old V22 bench can approve or reject changes to the 0.477 line.
Cost: about 3.8 GPU h of 30 (27.19 h left at 08:31 IST, roughly 26.2 h after this run).

## Bench reversal diagnosis (31 Oct 09:53 IST; CPU, no submission)
Question: is the bench477 reversal (baseline0465 0.517 > V1 > V2 > CUSTOM > merged477 0.498, public LB the other way) a bad bench (A: public gains real) or public noise (B: bench closer to private)? Script `enveda-casmi26/56_bench_reversal_diagnosis.py`, output `enveda-casmi26/artifacts/56_summary.json`, `56_per_molecule.csv`.

**1. Where the losses come from** (baseline -> merged477, 250 molecules): 54 lists change; **0 rank-1 gains anywhere, 4 rank-1 losses** (1 class 1, 3 class 2); inside 2-25 14 up vs 36 down. Mix-weighted delta -0.019 (c1 -0.023, c2 -0.035, c3 -0.011). Every neural step loses on its own: V0->V1 -0.002, V1->V2 -0.003, V2->CUSTOM -0.007 (class 3: 6 up / 22 down). The fragment check (CUSTOM->physical_V2) is mixed: class 3 +0.010 (22 up / 7 down) but class 2 -0.028 (2 rank-1 losses), net -0.001. The merge (CUSTOM->merged477) is 3 rank-1 losses (1 c1, 2 c2) and nothing else: its 7 leader moves give 0 wins.
**2. Neural health:** neural_ok 100%, 0 errors, 6,246 rows; score median 4.65 (5-95%: 1.3-7.3), cos median 0.47. The neural order is weak: the answer is neural rank 1 in 7/70 (c1), 5/50 (c2), 2/51 (c3 listed) vs baseline rank 1 in 63/40/7; mean answer rank neural 9.6/8.6/10.5 vs baseline 1.1/1.6/4.5; neural better than baseline for 13 answers, worse for 145. Median answer z within its list only +0.4 to +0.7. This matches sanpier's own panel (neural alone 0.387 vs baseline 0.879). Leakage check: the model (casmi-spectrum-model-v1) is not on disk; per the vault notes it was trained on all of train.parquet with a refit on all identities, which includes the enveda-np-examples rows the bench hides from the pipeline only. So the model has (most likely) seen these exact query spectra and answers; leakage would favour the neural variants, yet they lose. Leakage therefore does not explain the reversal (if anything the bench flatters neural).
**3. Representativeness:** both timsTOF only. Bench vs visible test: [M+H]+ 48% vs 79%, negative mode 33% vs 19%, [M+NH4]+/[M+Na]+ 15% vs 2%, spectra/molecule 4.7 vs 3.0, precursor m/z 10/50/90% 223/388/677 vs 280/329/379 (test is a much narrower mass band). Classes are simulated by hiding spectra (no real novelty), and baseline class 1/2 MRR 0.95/0.87 reads high (remaining FPNet/pool-feature exposure). So the bench is NOT a faithful copy of the test: more negative-mode and wider-mass molecules, an easier class-1/2 regime where any re-rank can only hurt.
**4. Public noise:** resampling 125 molecules (28/20/52 mix) from the bench's per-molecule merged-baseline deltas: SD of the public delta 0.0078 (22% of lists change). If the bench were exactly right, +0.012 public would have P ~ 0 (4 SD), so the bench is at least partly unrepresentative. With a zero-mean version, P(>= +0.012) = 0.054 per try, P(>= +0.010) = 0.094; but the 0.477 is the best of a long public search (chain: 481 versions; 273 teams at exactly 0.472, only 16 above 0.48; 753525/753511 SE ~0.04), so with ~10-20 effectively independent tries a +0.012 maximum is expected under zero true gain (winner's curse).
**Verdict:** P(A) ~0.4, P(B) ~0.6: the public +0.012 is most likely a mix of a small real gain on the test's [M+H]+-heavy narrow-mass molecules and selection noise; the bench overstates the loss but its direction (0 rank-1 wins, neural answer rank 9 vs 1) is a real warning. Expected private: 0.477 line about baseline +0.000 to +0.005, with real downside.
**Final selection recommendation:** two submissions that disagree on this question, since Kaggle scores the better one on private: (1) the 0.477 copy (casmi-busyaprime-477-copy, ref 57736039) and (2) a conservative baseline without the neural/fragment/merge layers: our own 0.465 stack (ref 57585472), preferred over the 0.464/0.465 public copies because it is the honest, own-validated model. Not two neural-line copies (0.472 and 0.477 share the same risk).
Negatives: no leakage explanation found; the neural model itself could not be inspected (not local).

## Candidate database audit (31 Oct 14:05 IST; CPU only, nothing pushed or submitted)
Question: can a database change (which structures can enter the 25-lists) win +0.005 or more public, mainly on class 3?
Script `enveda-casmi26/57_candidate_recall_audit.py`; outputs `artifacts/57_summary.json`, `57_probe_recall.csv`,
`57_additions.csv`. Downloads (data only, new dir `/home/user/benchdata/dbaudit`): COCONUT 2026-10 lite CSV (199 MB zip,
coconut.s3.uni-jena.de), Kaggle xianyu3/casmi26-licensed-coconut-candidates-202610, xjoannax88/casmi-pubchemlite-candidates
(proxy for the pipeline's franciscoangulo PubChemLite), deadsignalslab/casmi26-chebi-lmsd-regenerated,
dmitriigluzdov/casmi26-natural-product-knowledge-table (LOTUS 2026-04-13 + NPAtlas 2024_09). HMDB: download blocked (403).

**1. Current sources (0.477 busyaprime and 0.479 midx; identical datasets, no new database in either).**
- Engine 1 (Ahmed v4n): v4 pool = train.parquet structures U prvsiyan COCONUT (710,701 metric keys), +-10 ppm (30 ppm
  fallback), no formula filter, plus generated derivatives of strong analogs.
- Engine 2 (two-ranker): prvsiyan COCONUT 436,389 + ChEBI/LMSD 98,433 (sergiomalv) + PubChemLite CC0 + train, +-10 ppm
  (30 fallback), no formula filter, top 25 into the RRF fusion.
- PubChem channel: tier of 105.9 M structures, 150-1250 Da (+ delta/growth supplements of the popularity dataset), +-10 ppm,
  pass 1 top 5,000 by f.z plus the 200 most documented, 25 non-pool structures by z(f.z) + 0.15 popularity, merged only
  through the gate (rel > 150, candidate-level).
- Final lists (bench477, 250 lists): 24.98 entries on average (min 21). Origin of final candidates: v4 pool 55%, engine-2
  extras 10%, PubChem tier only 19%, in no database (generated) 16%. Rank 1: v4 pool 89%, tier only 8%, engine-2 extras 1%,
  generated 1%.

**2. Recall for structures absent from train.parquet** (present = raw ik14, parent ik14 or metric key matches):

| Probe | n | v4 pool / COCONUT | engine-2 pool | PubChem tier | current union | missing from all |
|---|---|---|---|---|---|---|
| P1 Enveda-180 Zenodo, not in train (synthetic, Enamine) | 3,717 | 0.0% | 0.2% | 96.7% | 96.7% | 121 |
| P2 NPAtlas first reported 2021-2024, not in train | 2,523 | 99.4% | 99.6% | 99.5% | 99.64% | 9 |
| P3 LOTUS/NPAtlas, not in train (random 5k) | 4,815 | 99.1% | 99.5% | 97.8% | 99.56% | 21 |

Even the newest published natural products are already in the current sources (99.6%). The 121 Enveda-180 misses are
synthetic screening compounds in no NP database.

**3. Additions: new structures and dilution** (150-1250 Da, largest fragment + uncharged parent):

| Addition | Structures | new to engine-2 pool | of which neutral | neutral, new to every source | P2 newly reached | extra decoys per +-10 ppm window (engine-2 window ~100-180) |
|---|---|---|---|---|---|---|
| COCONUT 2026-10 full (CC0 archive) | 479,690 | 4,545 | 52 | 45 | 8 of 9 | 0.000-0.001 |
| COCONUT 2026-10 ChEBI+Wikidata subset (CC BY 4.0 / CC0) | 149,002 | 273 | 9 | 7 | 8 | 0.000-0.001 |
| LOTUS 2026-04 (CC BY 4.0) | 151,885 | 438 | 137 | 52 | 8 | 0.011-0.018 |
| NPAtlas 2024_09 (CC BY-NC 4.0, allowed since 17 Oct) | 33,497 | 121 | 106 | 64 | 9 | 0.005-0.043 |

Almost every COCONUT-new row is a salt, mixture or permanently charged cation (4,297 cations with no neutral form; the test
has no [M]+ spectra in the visible file). The same handful (8-9) of 2021-2023 NPAtlas compounds is what any addition adds.

**4. Gain vs dilution.** Class 3: by the host's definition its answers are absent from PubChem and COCONUT, and the audit
finds only 7-64 neutral structures in all public NP databases that are outside our current sources: reach ~0. Class 2
(20% of the test): newly reachable rate 0.3-0.4% (P2); at an honest RR of 0.26 for newly reachable answers the upper bound
is **+0.0002 MRR (0.03 public molecule)**; dilution is also ~0 (<0.05 extra decoys per list). Neither recall nor dilution
moves: the database is saturated (confirms lever 15). The class-3 gap (79/130 answers missing on the bench) is a
generation gap, not a database gap; the class-2 gap is ranking inside the PubChem window (median 5-15k structures per
+-10 ppm; unseen1: PubChem-channel top-25 recall 52% for PubChem-only answers), which the widening attempts (rel 100/250,
blind expansion) already lost on the LB.

**5. Recommendation.** No database change is worth a GPU run or a submission (expected public gain +0.0000 to +0.0002, far
below 0.004 for one molecule). If a run happens anyway for another reason, the only defensible add-on is the licensed
COCONUT 2026-10 subset (xianyu3, CC BY 4.0 / CC0) into engine 2's pool, as a guarded insertion: only metric keys absent
from the current list, appended at ranks >= 20, never rank 1; verify ranks 1-19 byte-identical to the base on all lists
and that the recall-only test (57) still reads P2 99.9%. Do not add NPAtlas/LOTUS/COCONUT cations, HMDB or a wider
PubChem window. Negative result logged; stop list item 5 extended with "database additions".
