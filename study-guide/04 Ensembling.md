---
tags: [study, ensembling]
---
# 04 Ensembling

Back to [[00 Study Guide Home]].

> **Core idea:** models that make **different** mistakes average them out. The gain comes from diversity, not from adding more of the same model.

## 1. The methods
| Method | Combines | When |
|---|---|---|
| Mean of probabilities | calibrated probabilities | models of similar quality |
| Weighted mean | weights tuned on OOF predictions | quality varies; keep weights coarse |
| Rank averaging | per-model ranks | the score scales differ |
| Stacking | a meta-model trained on OOF predictions | many models and enough data |

![[rrf.png]]

**RRF:** score(item) = Σ over engines of 1 / (k + rank).
- It is robust to unscaled scores.
- k (typically 60, smaller for short lists) controls how much the top ranks dominate.
- An item ranked well by **both** engines beats one ranked first by only one.

## 2. Rules that keep ensembles honest
1. Tune weights on **out-of-fold** predictions only; never on the public LB.
2. Use coarse weights (0.25 steps). Fine weights fit noise: see the winner's curse in [[01 Validation & Leakage]].
3. Check correlation first. Two models with OOF correlation above 0.98 add little.
4. **One** change per submission. If you merge two notebooks and the score drops, you can't tell which part caused it.

## Exercises
1. Compute RRF by hand for k = 60 on the figure's lists. Does the order change from k = 3?

## Resources
- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion* (SIGIR 2009).
- *The Kaggle Book* (Banachewicz & Massaron), chapter on ensembling.
- MLWave, *Kaggle Ensembling Guide*.
