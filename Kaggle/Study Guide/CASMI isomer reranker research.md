---
tags: [kaggle, casmi26, research, ranking, isomers]
date: 2026-10-29
---
# CASMI isomer reranker research (29 Oct 2026, IST)

Back to [[CASMI26]]. Research only: nothing built, pushed or submitted, no GPU used. Base: our exact copy of
busyaprime's 0.477 (`/home/user/work/push_c477`). Goal (user step 3): our own same-formula isomer reranker.

Why isomers: formula is right in >95% of misses and the wrong winner is an isomer 98% of the time (forum 745468, 743254).
Scale (thread 753525, 29 Oct): about 125 public molecules, rank 2 -> 1 on one molecule = +0.004, standard error of the
mean MRR about 0.04; 273 teams sit exactly at 0.472, only 16 above 0.48. Top public is still heliosli 0.478 (private rr4 dataset).

## 1. What heliosli's layer does (read from source)
Sources: `/home/user/work/pull_h478` cell 25 (overlay v4, 0.478) and `/home/user/work/pull_helios` cell 26 (overlay v3,
0.475; cell 25 there is a probe setting `RR3_OVERRIDE = dict(alpha=2.0)`). The scorer code (`rr_deploy4.py`, `rr_v3.py`)
lives in the private dataset, so the exact score formula below is reconstructed from the cell comments.

**Position.** A post-processing cell after the pinned 0.468-line pipeline has written `submission.csv`. It never adds or
removes candidates; it only permutes each 25-list. Backup `submission_pre_rr4.csv`, audit `rr4_overlay_audit.json`.

**Consumes, per molecule:**
- the final list (`;`-joined SMILES, up to 25) from `submission.csv`;
- all test spectra of the molecule (`ms2_mzs`, intensities, precursor m/z, adduct, mode, mean collision energy);
- `library_max` from the pipeline export `frozen_candidate_evidence.parquet` (best library match);
- `train.parquet` InChIKey14 set (to flag training-set candidates);
- the v4b `casmi` engine code (for helpers, `V4CODE`).

**Scorer.** "Peaks matched to each candidate's 1-2-cut fragments, graph network + transformer": for each candidate,
enumerate 1- and 2-bond-cleavage fragments, match measured peaks to fragment masses, encode the candidate graph (GNN)
and the matched peak set (transformer) into one score per candidate. v4 adds the query view through dmitriigluzdov's
five fold-safe FPNets (`casmi26-fold-safe-fpnet`, public): each query goes through all five, scores are averaged; in
training each query was routed to the one encoder that never saw its structure (mimicking unseen test molecules).
v3 used one encoder (`cfg["encoder"]`).

**Rule (FREEZE).** The reranker scores training-set structures about 0.34 SD below the other candidates (a bias).
So: candidates whose InChIKey14 is in train keep their slot; only the non-train candidates are reordered among the
remaining slots by `z(-log rank) + alpha * z(reranker)` (plus a `delta` term, meaning not visible). Gates: skip if
`library_max >= lib_tau` (default 0.9), skip lists shorter than 2, skip after `budget_s` (default 3000 s), any error
leaves the list unchanged.

**Self-test.** Three bench molecules stored with offline scores; if the deployed scorer's correlation with them is
below 0.999 the whole overlay is skipped. Encoders are SHA-256 checked against the config.

**Config structure** (`rr4_config.json`; keys read by the cell): `encoders` (file names), `encoder_sha256`,
`encoder_dataset`, `ckpt` (reranker weights), `selftest` (json with cases: molecule_id, spectra, smiles, scores),
`mode` (e.g. FREEZE), `alpha`, `delta`, `lib_tau`, `budget_s`. v3: single `encoder` instead of the three encoder keys.

**What it gained:** 0.468-line + rr3 = 0.475; + rr4 = 0.478. Over 0.477 that is +0.001, a quarter of one molecule.
alpha 1 -> 2 was probed as a separate submission (result in their history, not ours).

## 2. Literature and methods
| Method | Idea | Licence (code / weights) | Train data need | Cost on free T4 | Kaggle offline port |
|---|---|---|---|---|---|
| MetFrag-lite (ours: `enveda-casmi26/fragments.py`) | 1-2 bond cuts, match peaks +-H shifts, explained intensity | MetFrag LGPL-2.1; ours own code | none | CPU, ~0.1-1 s per candidate | yes, already in repo |
| Learned fragment-match ranker (heliosli style) | features from matched fragments -> GNN+transformer or GBDT, trained on same-formula lists | own | train.parquet same-formula lists | GBDT CPU hours; small GNN 1-3 h T4 | yes |
| CFM-ID 4 | probabilistic fragmentation, predicted spectrum vs query | LGPL-2.1; weights partly NIST/METLIN-trained (verify) | retrain costly (C++, days) | slow (s-min per candidate) | hard; LOTUS lists tested -0.018 publicly |
| FIORA / FIORA-OS | GNN over bond cleavages predicting fragment intensities | MIT (repo, R pkg); OS weights host-approved (751373) | retrain on train.parquet possible | hours on T4 | yes (pip wheel + weights in a dataset) |
| ICEBERG / GLACIER / SCARF (ms-pred) | fragment DAG generation + intensity GNN | MIT code; NIST20 weights banned, GLACIER variants in use are allowed with reference | retrain on train.parquet: days | heavy | already inside the 0.477 (N1 family) |
| MIST | spectrum transformer over formula-annotated peaks -> fingerprint, contrastive retrieval | MIT; needs subformula annotation (SIRIUS for trees is banned; own annotator OK) | train.parquet | ~hours T4 | yes, if annotator is own |
| MassFormer | graph transformer forward model | BSD-2; NIST weights banned | retrain | hours-day | possible, low value |
| Contrastive retrievers (JESTR, sanpier CLIP model, MVP, DreaMS) | joint embedding spectrum/structure, hard negatives = isomers | MIT mostly; DreaMS MIT, GeMS pretraining (host lists DreaMS ok) | train.parquet | sanpier's: 5-15 min T4 | yes; JESTR tested -0.022, sanpier's RRF is already in 0.477 |
| SIRIUS / CSI:FingerID / fragmentation trees | trees, kernels | banned by host (751373) | - | - | ideas only (parsimony, explained intensity) |
| MassSpecGym (benchmark) | retrieval with formula-matched candidate sets (exactly our isomer problem), MCES-split | data MIT (HF card) | - | - | protocol idea: formula-bonus split |

Lessons from the literature and forum: on MassSpecGym "formula bonus" retrieval, fragment-aware and contrastive models
beat fingerprint retrieval by a few points of hit@1; gains on held-out scaffolds are much smaller than on random splits.
NeckBeard's MetFrag-lite tuning (2-bond x0.6, linear intensity, 0.005 Da, H -2..+3) gave +0.065 on an isomer panel and
flat on the LB. Every public poster reports 3-10x shrinkage from local to LB.

## 3. Our feasibility plan (step -> verify)
1. **Rules first** -> verify: written answers for (a) using train-structure membership at inference (FREEZE logic),
   (b) the train/test spectrum overlap below, (c) cross-test-spectrum use (752297: only a participant, Marie Killian,
   said "allowed"; not the host, treat as unconfirmed; do not use). Host 752541: non-reproducible submissions may be
   removed, so training code + data hash must ship.
2. **Data subset** -> verify: from `train.parquet` (2.54 M spectra; read columns lazily, disk has ~1 GB free, no new
   downloads), pick ~3,000 molecules with [M+H]+ timsTOF/Orbitrap spectra; for each build a same-formula decoy list
   (10-24 isomers) from other train structures with the same `molecular_formula`; split by Murcko scaffold, 5 folds.
   Check: every list has the answer plus >= 5 isomers.
3. **Features** (reuse `fragments.py`) -> per candidate: explained intensity / peak count for 1-cut and 2-cut
   (separately), NeckBeard settings, top-k peak coverage, unexplained base peak flag, neutral-loss matches, list-relative
   z-scores of each, plus the 0.477 base rank. Verify: the feature for the true answer beats a random isomer > 50%.
4. **Model** -> LightGBM lambdarank (CPU, ~30 features, depth 6, 300 rounds). Optional step 2: a small GNN only if
   GBDT shows signal. Verify: OOF hit@1 / MRR vs base-rank-only on scaffold-held-out folds.
5. **Bench protocol on the pipeline lists** -> apply to the 0.477's own final lists on the honest benches
   (`bench/results/unseen_*`, the thin class-1 slice from genfix bench), under the heliosli guardrails: rank 1 untouched
   unless the margin is large, lib_max >= 0.9 untouched. Paired bootstrap over molecules (10k), accept only P(better) >=
   0.9 AND zero thin-class-1 rank-1 losses. Divide the bench gain by 3-10 for the LB forecast and write it down first.

**Anomaly to flag before any training-structure logic:** the coverage audit (25 Oct) found all 1,213 test spectra occur
verbatim in train.parquet (binned cosine 1.000), yet the matched structure is in the top 25 for only 12 of 400. Either
the test spectra are planted in train under different (decoy?) structures, or the matching was a binning artefact.
Check: exact array equality on a sample, the `ingest_lib` and InChIKey of the matched rows, and the competition rules
/ data page on test-in-train. Until explained, a FREEZE-style "train membership" feature may be a leak or a trap.

## 4. Risks and rules to check
- Host: NIST and NIST-trained weights banned; SIRIUS/CSI:FingerID banned; train.parquet-trained models fine with code.
- Reproducibility: ship training script + seed + data hash; host may remove non-reproducible entries (752541).
- Cross-test-spectrum features: unconfirmed, avoid.
- Validation: our benches never predicted LB sign for list changes (gen20). A permutation-only rerank at ranks 2-25 is
  safer, but rank 2 -> 1 swaps are where the value is, and they are what can lose a class-1 point.
- Overlap with the 0.477: sanpier's neural RRF and Tamrazov's fragment check already encode spectrum-structure and
  fragment evidence; a new fragment feature may be largely redundant.

## 5. Ranked recommendation
1. **GBDT fragment-feature isomer reranker, ranks 2-25 plus guarded rank-1 swap, FREEZE-like gates** (CPU, ~1-2 days
   work). Expected on top of 0.477: **+0.000 to +0.003**, P(beating 0.477 on public) ~15%.
2. Same features fed through FIORA-OS predicted-spectrum similarity as an extra column (T4 hours). +0.000 to +0.002.
3. Small GNN+transformer like heliosli's: most cost, heliosli's own gain over 0.477-equivalent was ~+0.001. Not first.
Honest summary: one molecule is 0.004 against a standard error of 0.04, so any public result is mostly noise; build it
for the private split, not for the public number.

## 6. Result of the recommended first experiment (29 Oct 12:07 IST) - negative
`enveda-casmi26/51_isomer_reranker.py`. Scaffold-held-out isomer MRR: LightGBM 0.871 vs fragment score 0.862 vs random 0.206 (+0.009). On the genfix bench base lists with the guards fixed up front (formula-of-rank-1 slots only, alpha 1, rank-1 margin 1.0, lib_max >= 0.9 skipped): natural d -0.0056, P(better) 0.14; class 1 has 5 rank-1 losses vs 4 gains. Forecast -0.0006 to -0.002 on public. Rejected; not integrated. Anomaly: 40/40 sampled test spectra equal a train row exactly, all `enveda-180`, one structure each (labels apparently not the answers); not used. Next only if new information: a stronger isomer signal (forward-model spectra, e.g. FIORA-OS), not more fragment features.
