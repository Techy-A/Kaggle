---
tags: [study, calibration]
---
# 06 Calibration & label shift

Back to [[00 Study Guide Home]].

> **Core idea:** a probability of 0.7 should come true 70% of the time. When the test's class mix differs from what the model expects, adjust the predictions to the **expected** mix rather than retraining.

## 1. Calibration
![[calibration.png]]

- **Reliability diagram:** bin the predictions and plot predicted vs actual frequency. Below the diagonal means over-confident; neural nets usually are.
- **Measures:** ECE (expected calibration error), Brier score, log-loss.
- **Fixes, fitted on OOF or validation data only:**
  - **Temperature scaling:** softmax(z / T), one parameter. It keeps the argmax, so accuracy doesn't change.
  - **Platt scaling:** a logistic model on the score, for binary tasks.
  - **Isotonic regression:** flexible but needs lots of data.
- **Why calibrate when the metric is accuracy?**
  - Thresholds, gates and ensembles all assume comparable probabilities.
  - Averaging an over-confident model with a calibrated one lets the over-confident one dominate.

## 2. Label shift and balancing
The metric is **balanced accuracy**, or the test is known to be close to class-balanced, but the raw argmax over-predicts a few classes.

![[sinkhorn.png]]

**Sinkhorn balancing:** treat the test's probability matrix P (passages × classes) and repeat a few times:
1. Scale the columns so each class's total matches the target, e.g. uniform.
2. Scale the rows so each passage still sums to 1.

Then take the argmax.
- **α (partial balancing):** blends between raw (α = 0) and fully balanced (α = 1). Check each notebook's exact definition; some use it as an exponent on the column scaling.
- Full balancing is right only if the test really is balanced **at the batch size you balance over**.
- With about 2–3 passages per prompt in the hidden test, the balance holds globally, not per prompt.

**Other tools:**
- EM prior re-estimation (Saerens et al. 2002): estimate the test prior from the predictions.
- Optimal-transport assignment: Hungarian matching when the counts are exact.

## In our comps
- **NTL:**
  - `BALANCE_ALPHA` 0.5 is the public setting; α 0.25 scored 0.555 publicly. Our 0.556 + α 0.25 variant is predicted 0.556 after the bias adjustment.
  - **Final-pick hedge:** keep one final at α 0.5. The α optimum on 30% public data is a winner's-curse risk: see [[01 Validation & Leakage]].
- **CASMI:**
  - The PubChem gate (rel 150) is a threshold on a score that isn't calibrated.
  - That is part of why the bench predicts the gate's LB effect poorly (2 of 3 calls).

## Exercises
1. Implement Sinkhorn in NumPy in ≤ 20 lines. Check that the column sums converge on a random 300 × 18 matrix.
2. On NTL OOF predictions, plot balanced accuracy against α from 0 to 1 in steps of 0.1. Is the curve flat near the top? A flat top means α doesn't matter much and the hedge is cheap.
3. Fit a temperature on OOF predictions and compare ECE before and after.

## Resources
- Guo et al., *On Calibration of Modern Neural Networks* (2017).
- Saerens, Latinne & Decaestecker, *Adjusting the Outputs of a Classifier to New a Priori Probabilities* (2002).
- Cuturi, *Sinkhorn Distances* (2013).
