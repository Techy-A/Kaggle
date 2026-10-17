---
tags: [study, gbdt, ranking]
---
# 02 Gradient boosting & learning-to-rank

Back to [[00 Study Guide Home]].

> **Core idea:** add small trees one at a time. Each tree fits the gradient of the loss, i.e. the current mistakes. Stop when validation stops improving.

## 1. How boosting works
- Prediction: F_m(x) = F_{m−1}(x) + η · tree_m(x).
- Each tree fits the negative gradient. For squared error that is the residual; for log-loss, y − p.
- Main controls:
  - learning rate η (0.02–0.1);
  - number of rounds;
  - `num_leaves` / `max_depth`;
  - `min_child_samples`;
  - feature and row subsampling;
  - L1/L2 regularisation.

![[boosting_early_stop.png]]

**Early stopping:** watch the validation loss and keep the round where it bottoms out.
- The stopping round must come from the **inner** fold. If you pick it on the scoring fold, it leaks.
- For the final model trained on all data, use about 1.1 × the mean best round.

## 2. The libraries
| Library | Strength |
|---|---|
| LightGBM | fastest; leaf-wise trees; `lambdarank` objective |
| XGBoost | robust; `rank:ndcg` / `rank:pairwise` |
| CatBoost | best with categoricals; ordered boosting resists target leakage |
| sklearn `HistGradientBoosting` | no extra dependency; what engine 2 uses |

## 3. Learning-to-rank
In ranking only the **order within a query** matters: for us, within one molecule's candidate list.

| Approach | Loss looks at | Example |
|---|---|---|
| Pointwise | each candidate alone (is it the answer?) | HGB classifier, engine 2 today |
| Pairwise | pairs (answer above non-answer?) | RankNet, `rank:pairwise` |
| Listwise | the whole list's metric | LambdaMART (`lambdarank`) optimises NDCG / MRR directly |

- **Pointwise** is simple but wastes effort on easy negatives.
- **Listwise** puts effort where rank 1 vs rank 2 is decided. That matches MRR: see [[05 Ranking & Retrieval]].
- Data format: `group=` sizes, i.e. candidates per query, with all of a query's rows in the same fold.

## 4. Features for rankers
- **Query-relative features** often win: score − best score in the list, rank in list, z-score within the list.
- Agreement features: how many engines or sources proposed this candidate.

## In our comps
- **CASMI:** engine 2 is two HGB rankers (pointwise).
  - A LambdaMART version is the obvious offline experiment for **week 4**.
  - It must win on the unseen bench **and** survive the noise floor.
- **NTL:** the gate models are small boosted or linear models on top of LLM features. Watch overfitting: 18 classes and few prompts.

## Exercises
1. Train LightGBM with `objective='lambdarank'` on bench candidates grouped by molecule. Compare MRR@25 with pointwise HGB.
2. Add "score minus the list's best score" as a feature. Does the pointwise model close the gap?

## Resources
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine* (2001).
- Burges, *From RankNet to LambdaRank to LambdaMART: An Overview* (2010).
- LightGBM docs: "Parameters Tuning" and "lambdarank".
