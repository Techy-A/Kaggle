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

## 7. FIORA / FIORA-OS as a negative-mode forward model (29 Oct 2026, IST; research only, nothing run)

Sources read: github.com/BAMeScience/fiora (README, `pyproject.toml`, `requirements.txt`, the shipped
`fiora_OS_v0.1.0_params.json` / `fiora_OS_v1.0.0_params.json`, `examples/example_input.csv`); search abstracts of
Nowatzky et al., Nat Commun 16, 2298 (2025). The full paper (nature.com, PMC, arXiv 2304.13136) could not be fetched
here (hosts not allowed by the session network), so the paper numbers below are marked unverified.

**What it is.** A GNN (RGCNConv, depth 10 in OS v1.0.0) that scores single bond cleavages (each side, up to 4 H shifts)
and outputs a fragment-intensity MS/MS spectrum; RT/CCS heads exist but `rt_supported`/`ccs_supported` are false in the OS
models. CLI `fiora-predict -i in.csv -o out.mgf`, columns Name, SMILES, Precursor_type, CE, Instrument_type.

**Adducts / modes (from the OS params files).** `precursor_mode` in {[M+H]+, [M-H]-, [M]+, [M]-}; `instrument` in {HCD}
only; inputs also collision energy, molecular weight and element composition. So:
- covers the 25 [M-H]--only test molecules (our main blind group);
- does **not** cover [M+CH2O2-H]- (2 molecules; a formate adduct usually fragments as [M-H]- after formic loss, so
  scoring them as [M-H]- is a plausible but untested proxy), [M+Na]+ (5 Na-only), [M+NH4]+ or [M+K]+;
- timsTOF is not a category: we would pass `HCD` and the spectrum's CE. Beam-type CID in a timsTOF collision cell is
  close to HCD, but the CE scale differs (Orbitrap NCE vs eV); treat CE as a tuned constant (try file CE, and 20/40 eV).

**Licence / rules.** Code MIT (repo and `pyproject.toml`). OS weights ship inside the MIT repo, trained only on MSnLib
(v1.0 for OS v0.1.0, v7 for OS v1.0.0; public Zenodo library, Orbitrap HCD; licence believed CC BY 4.0, verify on
Zenodo record 16984129 before use). Training code is public (`fiora-train`, MSnLib download/preprocess scripts). Host
in thread 751373 named FIORA-OS as allowed. No NIST in the OS models (the paper's NIST-trained model is not the one
shipped). Allowed, commercial-use compatible.

**Dependencies / offline install.** Python >= 3.10.8; torch (dev pin 2.6.0), torch_geometric 2.6.x, lightning-fabric
2.6.x, torchmetrics, rdkit (pin 2024.9.6), spectrum_utils 0.4.2, treelib, regex, dill, seaborn, scikit-learn.
torch, rdkit, sklearn are usually in the Kaggle image; torch_geometric 2.6 is pure Python (RGCNConv needs no
torch-scatter/sparse wheels). Realistic offline route: a Kaggle dataset holding the fiora source tree (with weights) plus
wheels for torch_geometric, lightning-fabric, spectrum_utils, treelib, then `pip install --no-index --no-deps`.
Risk: the 0.477 kernel's pinned image torch/numpy versions; check before building.

**Speed.** README: the 10-row example "typically completes within a few seconds" on CPU (includes model load); GPU via
`--dev cuda:0`. Paper abstract: large GPU speed-up, CFM-ID much slower. No per-spectrum figure seen; estimate
10-100 ms per (candidate, CE) on CPU, to be measured in step A2.

**Accuracy (unverified, abstract only).** Claims better prediction quality than ICEBERG and CFM-ID on NIST-2017,
MS-DIAL and CASMI 2016/2022; negative-mode numbers not seen. The OS model (MSnLib) may be weaker than the paper model.

**Public CASMI26 notebooks.** None of the 200 scanned Code-tab titles (`scan_casmi/kernels.json`) mentions FIORA.

### Integration plan for the 0.477 (guarded extra signal)
Stage 1 (gap only): for molecules whose spectra are all [M-H]- (25) or [M+CH2O2-H]- (2, as [M-H]- proxy), predict FIORA
spectra for the 25 final candidates, score cosine (sqrt intensities, 0.01 Da, precursor removed) against each real
spectrum, mean over spectra; re-rank with `final = base_rank_score + w * z(fiora_cos)` within formula-of-rank-1 slots,
w fixed in advance (w in {0.5, 1}), rank-1 swap only when the FIORA margin > m and lib_max < 0.9 (same guards as §6).
Na-only (5) stays out: FIORA has no [M+Na]+ mode. Stage 2 (optional): same column at ranks 2-25 for all molecules,
alongside ICEBERG/GLACIER, only if stage 1 passes.

Bench protocol: build a negative-mode pseudo-test from `train.parquet` [M-H]- spectra (excluding `enveda-180`, never
use the test-in-train rows), candidate lists from the same genfix pipeline; paired bootstrap over molecules,
accept only with P(better) >= 0.9 **and** zero rank-1 losses on the class-1 (library-hit) subset.

Runtime vs 9 h: stage 1 = 27 molecules x 25 candidates x <= 3 CE x mean ~1.2 spectra ~ 2.4k predictions, ~1-5 min CPU.
Stage 2 = 400 x 25 x ~3 = 30k predictions, ~10-50 min CPU (seconds on GPU). Install from dataset < 2 min. Fits easily.

Steps needing user approval (in order):
1. Download the fiora repo (MIT) with the OS v1.0.0 weights and pip-install it plus torch_geometric etc. into a local
   venv (network + install of external code).
2. A2: local CPU test on 30-60 negative-mode bench molecules: run the 10-row example and compare with
   `expected_output.mgf`, time per spectrum, then the bench above (no GPU, no Kaggle).
3. Only if the bench passes: upload a Kaggle dataset (fiora source + weights + wheels), push a kernel = 0.477 copy +
   stage-1 block, and submit (each a separate approval).

Forecast (written before any run): offline bench gain divided by 3-10 for public transfer. Ceiling: about 8.3 public
molecules in the covered gap (25/27 of ~8.9), most class 1 already right; earlier coverage steps gave ~+0.008 (one
molecule = ~0.004-0.008 on ~125-132 public molecules). Expected public +0.000 to +0.004, point +0.001, P(>= +0.004)
about 0.15; risk of -0.004 if the guard misfires once. Recommendation: **go for steps 1-2 only** (cheap, CPU, decides
it); no-go for a submission unless the bench passes the gates.

## 8. Step 2 (A2) result: FIORA-OS negative-mode isomer bench (29 Oct 13:24 IST) - weak, not yet a go
`enveda-casmi26/52_fiora_negmode_bench.py`, outputs `artifacts/52_*` (summary in `52_summary.json`). CPU only, nothing
pushed or submitted.
- **Data.** 60 [M-H]- queries from `train.parquet`, one spectrum each: instrument_type contains "tof" (LC-ESI-QTOF 53,
  ESI-QTOF 3, LC-ESI-TOF 3, qtof 1; no timsTOF label exists in train), single CE in eV (6-50, median 20), >= 5 peaks;
  libs riken 22, massbank 22, mona 11, msdial 4, spectraverse 1. Excluded: every `enveda-180` row (also as isomer
  source), genfix bench answers, and any structure with a `pluskal_ms2` row (MSnLib = FIORA-OS training data). One
  molecule per formula; 10-24 same-formula isomers from other train structures (mean 16.1 candidates). Split by a fixed
  random draw into tune 29 / test 31 (formula-disjoint).
- **FIORA.** [M-H]-, HCD, CE grid {10, 20, 30, 40} fixed in advance plus the grid mean; 3,856 predictions, 0 missing.
  Score: cosine (sqrt intensity) and unweighted entropy similarity, 0.01 Da greedy matching, peaks >= precursor - 1
  removed. Ties: expected reciprocal rank under random tie-break.
- **Tune half (MRR):** cos 10/20/30/40/mean 0.654/0.644/0.645/0.646/0.651; entropy 0.662/0.611/0.625/0.605/0.629.
  Chosen: entropy at CE 10.
- **Held-out half (31 molecules), top-1 / MRR:** FIORA entropy-CE10 0.355 / 0.561; fragment score
  (`fragments.explain_scores`, negative mode) 0.226 / 0.543; random 0.097 / 0.231 (expected 0.214).
  FIORA vs fragment: d MRR +0.019, 95% CI [-0.102, +0.144], **P(better) 0.62**. FIORA vs random P 1.00.
  Pre-fixed combination z(frag) + z(FIORA) within the list: top-1 0.419, MRR 0.627, vs fragment d +0.084
  [-0.019, +0.197], **P(better) 0.94**. Post hoc (not used for decisions): cos_mean on test 0.587, ent_mean 0.578; the
  tune-half CE choice is noise-level (spread 0.01-0.06 MRR on 29 molecules). All 60: FIORA 0.610 vs fragment 0.540.
- **Speed.** 4 shards x 1 thread (OMP_NUM_THREADS=1): 143 s wall for 3,856 = 0.037 s/prediction wall, ~0.15 s per
  core. Default threading in one process was 2.45 s/prediction (232-row test; thread oversubscription), so the Kaggle
  run must set OMP_NUM_THREADS=1 and shard. Stage 1 (~2.4k predictions) would take ~1.5 min on 4 cores.
- **Verdict.** FIORA alone is not clearly better than the fragment score (P 0.62). Its value is complementary: the
  combination passes P >= 0.9 nominally (0.94) but on 31 molecules, and the class-1 zero-rank-1-loss gate is untested
  (no base lists here). Forecast if integrated as stage 1: bench gain +0.08 MRR within isomer sets on <= ~8 affected
  public molecules = ~0.6 molecule-equivalents x 0.004 = +0.0027 before transfer; /3-10 -> **+0.0003 to +0.001**,
  risk -0.004 per misfired rank-1 swap. **Step 3: no-go for now.** Next (cheap, CPU, ~10 min): enlarge the bench to
  ~250 molecules with the same frozen settings (entropy CE10 + z-sum) and require P >= 0.9 there, then a guarded
  run on genfix-bench [M-H]- base lists for the class-1 gate, before any Kaggle dataset.

## 9. Step 2 enlarged: FIORA-OS on ~250 new held-out [M-H]- molecules + guarded genfix-bench check (29 Oct, IST)
`enveda-casmi26/53_fiora_negmode_bench250.py`, outputs `artifacts/53_*` (summary `53_summary.json`). CPU only, local;
nothing installed, pushed to Kaggle or submitted.
- **Forecast, written before any 53 result was read.** Frozen settings (entropy CE10, z(frag)+z(FIORA)). If the
  combination holds its 31-molecule gain (~+0.08 MRR within isomer sets) and the guarded bench shows a gain, the public
  effect is bench gain / 3-10 on at most ~8 affected public molecules, one molecule = 0.004: roughly **+0.0003 to
  +0.001 public**, downside -0.004 per misfired rank-1 swap. A result below ~+0.0005 is not measurable on the public LB.
