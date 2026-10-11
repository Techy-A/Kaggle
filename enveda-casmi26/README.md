# Enveda CASMI 2026: molecule identification from MS/MS spectra

My work for the Kaggle competition [Enveda CASMI 2026 – Molecule ID From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra): given the tandem mass spectra of an unknown molecule, rank up to 25 candidate structures (SMILES). The metric is MRR@25 on the first InChIKey block.

| | |
|---|---|
| **Public leaderboard** | 0.150 (cosine library-search baseline) |
| **Notebook** | [CASMI26: Validation That Behaves Like the Test](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test) |

## Findings so far

1. **The visible `test.parquet` is copied from `enveda-180`.** That library is mostly drug-like synthetic compounds that are already in the training set, while the hidden test is natural-product-like. Scores on the visible file don't predict the leaderboard.
2. **`enveda-np-examples` is the right validation set.** It holds 250 natural products measured on timsTOF (the test instrument), and all of them have public spectra elsewhere. Removing those spectra from the reference library simulates the novelty classes:

   | Simulated class | Molecules | Cosine search MRR@25 |
   |---|---|---|
   | 1: has public spectra | 125 | 0.939 |
   | 2/3: no public spectra | 125 | 0.000 |

3. **Only about 16% of the public test is class 1** (LB 0.150 / class-1 MRR 0.939). Most of the available score is in molecules without reference spectra.

## Layout

| File | Purpose |
|---|---|
| `casmi.py` | Mass helpers, peak cleaning, cosine search, library index, MRR@25, novelty-class split |
| `validate.py` | Scores the baseline per simulated class: `python validate.py data/train.parquet` |
| `build_notebook.py` | Generates `notebooks/notebook_validation.ipynb` from `casmi.py` plus the analysis cells |
| `kernel-metadata.json` | Kaggle notebook metadata |

The competition data (CC BY-NC 4.0) is not redistributed here. Download it from Kaggle into `data/`.
