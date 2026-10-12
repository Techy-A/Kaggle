# Enveda CASMI 2026: molecule identification from MS/MS spectra

My work for the Kaggle competition [Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra): given the tandem mass spectra of an unknown molecule, rank up to 25 candidate structures (SMILES). The metric is MRR@25 on the first InChIKey block.

| | |
|---|---|
| **Public leaderboard** | **0.221** (fingerprint v2 + PubChem + prior); 0.176 (v1); 0.150 (cosine only) |
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

## Layout

| File | Purpose |
|---|---|
| `casmi.py` | Mass helpers, peak cleaning, cosine search, library index, MRR@25, novelty-class split |
| `validate.py` | Scores the baseline per simulated class: `python validate.py data/train.parquet` |
| `build_notebook.py` | Generates `notebooks/notebook_validation.ipynb` from `casmi.py` plus the analysis cells |
| `fingerprints.py` | Morgan fingerprints for the candidate pool (training structures + COCONUT) |
| `fpmodel.py` | Spectrum featurisation, the MLP, training (`prep` / `train`), fingerprint likelihood, cosine gate |
| `evaluate.py` | Per-class comparison of cosine, fingerprint and gated ranking |
| `predict.py` | Writes `submission.csv` with the gated ranking |
| `build_notebook2.py` | Generates `notebooks/notebook_fingerprint.ipynb` |
| `kernel-metadata*.json` | Kaggle notebook metadata |

The competition data (CC BY-NC 4.0) is not redistributed here. Download it from Kaggle into `data/`.
