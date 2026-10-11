# Enveda CASMI 2026: molecule identification from MS/MS spectra

My work for the Kaggle competition [Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra): given the tandem mass spectra of an unknown molecule, rank up to 25 candidate structures (SMILES). The metric is MRR@25 on the first InChIKey block.

| | |
|---|---|
| **Public leaderboard** | 0.176 (fingerprint model + cosine gate); 0.150 (cosine only) |
| **Notebook 1** | [CASMI26: Validation That Behaves Like the Test](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test) |
| **Notebook 2** | CASMI26: Finding Molecules Not in Any Library (spectrum → fingerprint model) |

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
