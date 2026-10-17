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
| **Reciprocal rank fusion (RRF)** | ranked lists | retrieval / candidate lists, as in CASMI |
| Stacking | a meta-model trained on OOF predictions | many models and enough data |
| Gating / routing | one model decides which expert to trust | as in the NTL Claude gates |

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

## In our comps
- **CASMI:** V22 fuses engine 1 (v4n + derive generation), engine 2 (HGB rankers), the PubChem channel (rel gate 150), ICEBERG/GLACIER, E006, v7c and E2R2.
  - The open class-2 loss (11 molecules on the unseen bench) is a **fusion** problem: points are lost where the PubChem channel's candidates are merged into the list. Check the per-molecule ranks before and after fusion to see how.
- **NTL:** our merge of TabPFN with the Claude gates and punctuation rules is predicted below the 0.556 line (0.549–0.551).
  - Lesson: merging two good notebooks doesn't add up when they share the same base signal.
  - **Never pick the TabPFN merge as a final:** the TabPFN licence is non-commercial. See [[10 Competition Playbook]].

## Exercises
1. Compute RRF by hand for k = 60 on the figure's lists. Does the order change from k = 3?
2. Compute the OOF correlation between two NTL public models. Is it above 0.98?

## Resources
- Cormack, Clarke & Buettcher, *Reciprocal Rank Fusion* (SIGIR 2009).
- *The Kaggle Book* (Banachewicz & Massaron), chapter on ensembling.
- MLWave, *Kaggle Ensembling Guide*.
