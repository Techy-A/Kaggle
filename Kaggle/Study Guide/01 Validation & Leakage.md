---
tags: [study, validation]
---
# 01 Validation & leakage

Back to [[00 Study Guide Home]] · see also [[Validation]].

> **Core idea:** a validation score is only useful if it predicts the hidden test score. That requires a validation set that is new to the model **in the same way** the test is.

## 1. Grouped cross-validation
Rows that share a hidden factor must stay in the same fold. Examples: the same prompt, the same molecule, the same patient, the same day.

![[grouped_cv.png]]

- **Random split (left):** the model has seen other answers to the same prompt. It learns the prompt, not the author, so CV comes out too high.
- **Grouped split (right):** whole prompts are held out, the way the hidden test holds them out.
- Tools: `GroupKFold`, `StratifiedGroupKFold` (sklearn); for time series, `TimeSeriesSplit` with a gap.

**Check:** for each test row, ask what the model has seen that is "the same thing". Group on that.

## 2. Leakage, train–test contamination
The answer reaches the model through a side door.

![[leakage.png]]

| Kind | Example | Fix |
|---|---|---|
| Target leakage | a feature computed using the label | build every feature inside the fold |
| Group leakage | the same molecule in train and validation | grouped CV |
| Pre-training contamination | our np pseudo-test molecules were already in RIKEN/GNPS train | an **unseen** bench built from molecules absent from train |
| Preprocessing leakage | a scaler or TF-IDF fit on all rows | fit inside a `Pipeline` per fold |
| Temporal leakage | the future used to predict the past | split by time, with a gap |

## 3. The noise floor
A public LB is a sample. Two runs of the same code can differ.

![[noise_band.png]]

- With n test items and a per-item score of sd σ, the LB's sd is about σ/√n.
- Our measured floor: **±0.004** on CASMI (same code scored 0.451 and 0.447).
- Rule: a single-submission difference below the floor is **not a result**.

## 4. The winner's curse
Choose the best of many noisy scores and you choose the luckiest one, not the best model.

![[winners_curse.png]]

- 30 submissions of true quality 0.451 with noise sd 0.003: the best one shows about 0.457.
- That is why our forecasts for new changes ran about **0.002 too optimistic**: see [[Predictions vs Results]].
- Defences:
  - pick finals by the offline bench plus the LB together, not the LB alone;
  - prefer changes that win on **two** independent checks;
  - shrink every forecast by the running bias.

## 5. Adversarial validation
Train a classifier to tell train from test.
- AUC around 0.5: same distribution.
- High AUC: the features it uses are the ones that shift. Drop them, re-weight, or build validation from the most "test-like" train rows.

## In our comps
- **CASMI:**
  - The first bench (np pseudo-test) was leaky.
  - The unseen bench (Enveda-180 Zenodo molecules absent from train; 320 used) is the honest one.
  - Bench and LB agree on 2 of 3 calls, so we don't tune the PubChem gate on the bench.
- **NTL:** the hidden test has about 2–3 passages per prompt, so validation must group by prompt.
- **Both:** the noise floor and the bias go into every forecast.

## Exercises
1. From the two runs (0.451, 0.447), estimate the LB sd. How many runs would you need to tell apart a gap of 0.002?
2. Run adversarial validation on NTL train vs test using char-n-gram TF-IDF. What is the AUC?
3. In `bench/build_bench.py`, list every place the answer is hidden (4 of them). What would leak if one were missing?

## Resources
- Kaggle Learn: *Intermediate ML → Data Leakage*.
- Kaufman et al., *Leakage in Data Mining* (KDD 2011).
- sklearn user guide: "Cross-validation iterators for grouped data".
