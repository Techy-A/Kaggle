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
