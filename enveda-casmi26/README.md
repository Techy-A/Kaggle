# Enveda CASMI 2026: molecule identification from MS/MS spectra

My work for the Kaggle competition [Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra): given the tandem mass spectra of an unknown molecule, rank up to 25 candidate structures (SMILES). The metric is MRR@25 on the first InChIKey block.

| | |
|---|---|
| **Public leaderboard** | own pipeline **0.272** (GPU v3 + ranker with fragmentation + analogs); fork of the 0.440 public notebook 0.437 |
| **Notebook 1** | [CASMI26: Validation That Behaves Like the Test](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test) |
| **Notebook 2** | CASMI26: Finding Molecules Not in Any Library (spectrum → fingerprint model) |
| **Notebook 4** | CASMI26: The Popularity-Prior Trap (why priors look perfect on library-molecule validation) |

## Findings so far

1. **The visible `test.parquet` is copied from `enveda-180`.** That library is mostly drug-like synthetic compounds that are already in the training set, while the hidden test is natural-product-like. Scores on the visible file don't predict the leaderboard.
2. **`enveda-np-examples` is the right validation set.** It holds 250 natural products measured on timsTOF (the test instrument), and all of them have public spectra elsewhere. Removing those spectra from the reference library simulates the novelty classes:

   | Simulated class | Molecules | Cosine search MRR@25 |
   |---|---|---|
   | 1: has public spectra | 125 | 0.939 |
   | 2/3: no public spectra | 125 | 0.000 |

3. **Only about 16% of the public test is class 1** (LB 0.150 / class-1 MRR 0.939). Most of the available score is in molecules without reference spectra.

4. **A spectrum → fingerprint model finds the molecules that have no reference spectra.** A CPU-trained MLP predicts 2,048 Morgan bits; candidates from COCONUT ∪ training structures (median 58 within 10 ppm) are ranked by fingerprint likelihood. Confident cosine matches (≥ 0.7) go first:

   | Method | Class 1 | Class 2/3 | Expected at 16% class 1 |
   |---|---|---|---|
   | Cosine only | 0.939 | 0.051 | 0.19 |
   | Fingerprint only | 0.338 | 0.371 | 0.365 |
   | Gate: cosine ≥ 0.7, then fingerprint | 0.785 | 0.337 | 0.409 |

5. **Leaderboard: 0.176, not the ~0.41 the validation predicts.** In the simulation the answer is always in the candidate pool. Working back from the score, only about 18% of hidden class-2/3 answers are in COCONUT ∪ train. Candidate coverage (PubChem) is the next lever, and the validation needs a split where the answer is only reachable through PubChem.

6. **PubChem + popularity prior: also 0.176 on the LB.** On the stricter split (answers reachable only through PubChem) PubChem covers 97% of class-2/3 answers, but with ~7,000 candidates per molecule the fingerprint model alone ranks them poorly (class 2/3 MRR 0.11). A popularity / natural-product prior lifts the *validation* score to 0.61, yet the leaderboard didn't move. The validation molecules come from public spectral libraries, so they are well studied, and popularity favours them in a way it doesn't favour the hidden molecules. A prior can't be tuned on this split.

7. **Why the prior fooled the validation.** Ranked by PubChem popularity alone, the answer is #1 in its 10 ppm window for 67% of `enveda-np-examples` molecules, 18% of public-library molecules, and ~0% of `enveda-180` molecules without public spectra (top 25: 99% / 53% / 0.7%). Class-2 molecules have no public spectra by definition, so they look like the last group.

8. **v2 fingerprint model (Morgan + MACCS) doubles PubChem-only MRR offline but scored 0.146 on the LB** (v2, PubChem + COCONUT + train, cosine gate 0.7, *no* prior). Offline, on 490 held-out `enveda-180` structures without public spectra: MRR 0.157 → 0.318. **Correction to finding 6 / notebook 4:** "0.176 with and without the prior" compared runs that also differed in candidates (COCONUT-only vs + PubChem), so it doesn't isolate the prior. A v2 + prior run is the clean comparison.

9. **v2 + prior: 0.221 (best).** Same model and candidates as finding 8, with the prior (0.25 × popularity + 1 × natural-product flag on z-scored fingerprint likelihood): 0.146 → 0.221. So the prior is worth +0.075 on the LB, much less than the +0.50 the library-based validation suggested, but real. Notebook 4 and its conclusions were corrected accordingly.

| Run | Model | Candidates | Prior | Public LB |
|---|---|---|---|---|
| 1 | cosine | library | – | 0.150 |
| 2 | v1 fingerprint | COCONUT + train | – | 0.176 |
| 3 | v1 fingerprint | + PubChem | yes | 0.176 |
| 4 | v2 fingerprint | + PubChem | no | 0.146 |
| 5 | v2 fingerprint | + PubChem | yes | **0.221** |

10. **Prior weights tuned on both PubChem-only splits** (`tune_prior.py`, v2 held-out model). S1 (library natural products) flatters popularity and S2 (no public spectra) penalises it, so the choice is made on the average and the worse split: w_pop 0.10 / w_np 1.0 gives S1 0.729, S2 0.273 (average 0.501), against 0.800 / 0.178 (0.489) for the 0.25 / 1.0 behind LB 0.221. Submitted as a probe.

11. **0.440 public fork + v2 fingerprint re-rank (β 0.3): 0.437.** Inside the source's own run-to-run range (0.436–0.440), so no measurable gain. Variants queued: β 0 (clean baseline), β 0.15, and top-guess-protected (only ranks 2–25 re-ranked). The official metric (read in full) scores the first correct guess only, so re-ordering cannot "poison" a row.

12. **Re-rank ablation on the fork: no effect.** β 0 (re-rank off) 0.437, top-guess-protected 0.437, β 0.15 0.437, β 0.3 0.437. The fork itself scores 0.437 in this account (the source reports 0.436–0.440; GPU stages are not bit-reproducible). A fingerprint re-rank inside the final 25 does not move any public-LB answer, so it is dropped; v2 (plain fork) is the safe final pick.

13. **Learned ranker (LightGBM LambdaRank, `learned_ranker.py`).** Research first: the 0.440 public pipeline already has learned rankers (four LightGBM boosters, plus two in a second engine) and a same-formula GLACIER/ICEBERG re-rank, and its published ranker rows carry no candidate identities, so our fingerprint feature can't be added to it. The ranker is therefore built on our own pipeline: one row per InChIKey14, features from fingerprint fit (z, gap, rank), popularity (raw, z, gap, rank), natural-product flag and window size; 5-fold CV grouped by molecule on 740 PubChem-only validation molecules. OOF MRR@25: S1 0.779 / S2 0.313, against 0.666 / 0.299 for the hand prior and 0.256 / 0.318 for fingerprint only. Submitted as a probe against 0.222.

14. **Fragmentation features (`fragments.py`, `frag_features.py`): the biggest single gain so far.** Every 1- and 2-bond cut of a candidate's heavy-atom graph gives fragment masses; matched as ions (+-2 H shifts, 10 ppm) against the molecule's merged peaks, they give an explained-intensity score. Computed for the strongest 50 candidates per molecule (chosen by features, not labels). Ranker OOF MRR@25: S1 0.783 -> 0.816, **S2 0.307 -> 0.467** (the split popularity cannot help). The fragmentation gap is now the most important feature.

15. **Both ranker probes scored below the hand prior on the LB:** learned ranker 0.172, ranker + fragmentation 0.199 (hand prior 0.222). Likely cause: the ranker was trained on features from the held-out model (`fpnet_v2h`) but inference used the full-data model (`fpnet_v2_full`), whose likelihoods sit on a different scale, so the absolute gap features (`fp_gap`, `pop_gap`) shift. Two fixes are queued: the ranker with `fpnet_v2h` at inference, and a scale-free alternative.
16. **Hand prior + fragmentation z-score (`pipeline.frag_z`, `w_frag`).** Fragmentation is scored for the top 40 candidates by fit plus the top 10 by popularity, z-scored within the molecule (unscored candidates get -3) and added to the prior. Validation MRR@25 (S1 / S2):

| Prior | No fragmentation | + 0.5 x frag z |
|---|---|---|
| pop 0.25, np 1.0 | 0.770 / 0.220 | **0.795 / 0.448** |

Because every term is z-scored within the molecule, a model swap can't shift it the way it shifts the ranker's gap features.

17. **Leaderboard, 14 Oct: the model fix and analogs both carry over.** Same candidates, gate 0.7:

| Run | Public LB |
|---|---|
| hand prior (0.25 pop / 1 np) | 0.222 |
| ranker + fragmentation, mismatched model (finding 15) | 0.199 |
| ranker + fragmentation, `fpnet_v2h` at inference | 0.216 |
| hand prior + 0.5 x fragmentation z | 0.245 |
| ranker + fragmentation + analogs, `fpnet_v2h` | 0.256 |
| **GPU v3 model (3 x 4,096, 64 spectra/structure, 25 epochs) + ranker retrained on v3 features** | **0.272** |

The v3 model and its ranker come from the GPU training notebook (`build_notebook_train.py`, `train_gpu.py`); the submission notebook only runs inference.

18. **Fork + fusion with our engine: 0.438** (plain fork 0.437). `fork_fusion_cell.py` runs our pipeline (v2 engine, ranker with fragmentation + analogs) in a time-guarded subprocess after the fork and merges the lists by reciprocal-rank fusion over metric identities (beta 0.5, K 3, the fork's first guess protected). A small gain, about one partial rank change on ~130 public molecules. Next: the same fusion with the GPU v3 engine (0.272 alone), and beta 1.0.

19. **GPU v3 model, offline (training notebook log, T4, 2.8 h end to end).** 3 x 4,096 hidden, up to 64 spectra per structure (1.81M spectra), 25 epochs, ~82 s per epoch; GPU featurisation matches the CPU one to 1.2e-7. Fingerprint likelihood alone, answers only via PubChem:

| Split | v2 MRR@25 | **v3 MRR@25** | v2 -> v3 top-1 | v2 -> v3 top-25 |
|---|---|---|---|---|
| S1 (250 natural products) | 0.262 | **0.439** | 17% -> 32% | 53% -> 71% |
| S2 (490 without public spectra) | 0.318 | **0.514** | 21% -> 39% | 64% -> 80% |

Ranker OOF MRR@25 with v3 features (S1 / S2): learned ranker 0.818 / 0.502, + fragmentation 0.847 / 0.627, **+ analogs 0.854 / 0.660** (v2: 0.842 / 0.521). The LB moved less (0.256 -> 0.272), so the remaining public-LB gap is mostly outside the ranking these splits measure (candidate coverage, class-1 handling).

20. **Analog-seeded generation (`generate.py`), feasibility.** For each validation molecule (its own spectra removed from the library), the 20 structures behind the most similar library spectra are mutated with 22 common natural-product transformations (O/N-(de)methylation, hydroxylation, (de)glycosylation, (de)acetylation, reduction/desaturation, ester hydrolysis, +-CH2, sulfation, prenylation), up to two steps; only products within 10 ppm of the molecule's mass are kept:

| Split | Answer generated | Median products | Median time |
|---|---|---|---|
| S1 (250 natural products) | **42%** | 16 | 3.3 s |
| S2 (490 synthetic `enveda-180`) | 1.8% | 4 | 2.5 s |

The transformations are natural-product chemistry, so they rarely reach the synthetic `enveda-180` compounds, but for natural products (the hidden test's kind) the answer is among a handful of generated structures 42% of the time. Caveat: library relatives of these molecules may be closer than those of truly novel ones. Next: rank generated candidates end to end in a class-3 simulation (answer removed from every pool).

21. **LB, 15 Oct.** A copy of the Dual-Band 0.441 notebook *without its pinned Docker image* (it ran on Python 3.13) scored 0.427 with our v3 merge, and 0.426 with merge + generated candidates at rank 2. So generation added nothing measurable on the public molecules: one hit at rank 2 would be worth about +0.004, and the -0.001 is consistent with one answer pushed from rank 2 to 3. On the 0.437 fork, the v3 merge scored 0.436 at beta 0.5 and 0.430 at beta 1.0 (the v2 merge scored 0.438). The merge is neutral at best, and more weight on the weaker engine hurts. Next: exact copies of the public 0.441 (v7c) and 0.443 (Antriksh Arya, v7c + engine-2 rank-2 insert) notebooks on their pinned images, with before/after-merge/after-generation outputs submitted from the same version.

## Layout

| File | Purpose |
|---|---|
| `casmi.py` | Mass helpers, peak cleaning, cosine search, library index, MRR@25, novelty-class split |
| `validate.py` | Scores the baseline per simulated class: `python validate.py data/train.parquet` |
| `build_notebook.py` | Generates `notebooks/notebook_validation.ipynb` from `casmi.py` plus the analysis cells |
| `fingerprints.py` | Morgan fingerprints for the candidate pool (training structures + COCONUT) |
| `fpmodel.py` | Spectrum featurisation, the MLP, training (`prep` / `train`), fingerprint likelihood, cosine gate |
| `evaluate.py` | Per-class comparison of cosine, fingerprint and gated ranking |
| `learned_ranker.py` | LightGBM LambdaRank over fingerprint / popularity features, grouped CV, saved booster |
| `fragments.py`, `frag_features.py` | In-silico 1/2-bond fragmentation and explained-intensity features |
| `predict.py` | Writes `submission.csv` with the gated ranking |
| `build_notebook2.py` | Generates `notebooks/notebook_fingerprint.ipynb` |
| `kernel-metadata*.json` | Kaggle notebook metadata |

The competition data (CC BY-NC 4.0) is not redistributed here. Download it from Kaggle into `data/`.
