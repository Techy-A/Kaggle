---
tags: [study, ranking, retrieval]
---
# 05 Ranking & retrieval

Back to [[00 Study Guide Home]].

> **Core idea:** retrieval has two stages. Stage 1 finds candidates and needs **recall**. Stage 2 orders them and needs **precision at the top**. Stage 1 is the ceiling: an answer that is never a candidate scores 0.

![[two_stage_retrieval.png]]

## 1. The metrics
| Metric | Formula | Rewards |
|---|---|---|
| Hit@k / Recall@k | is the answer in the top k? | stage 1 |
| MRR@k | mean of 1/rank of the first correct answer (0 if rank > k) | getting the answer to rank 1 |
| NDCG@k | Σ gain / log₂(rank+1), normalised | graded relevance |
| MAP | mean of precision at each correct answer | several correct answers |

![[mrr.png]]

**MRR is very top-heavy:**
- Rank 1 → 2 costs 0.5 of a molecule.
- Rank 10 → 11 costs about 0.009.
- So:
  - effort spent at ranks 2–5 pays the most;
  - a new candidate that lands at rank 20 is worth only 0.05;
  - every hit@25 gain still counts, but much less than a top-1 gain.

## 2. Diagnosing the losses
Split the lost score by cause:
1. **Not generated:** a stage-1 recall problem. Fix it with candidate generation or bigger pools; this is what gen20 did.
2. **Generated, but ranked below 25:** a stage-2 problem.
3. **Ranked 2–25:** stage-2 precision. Use listwise rankers or better features: see [[02 Gradient Boosting & Learning-to-Rank]].

For a class-3 miss, `bench/near_miss.py` checks whether the rulebook could reach the answer from the oracle parent.
- If yes, it's a **parent-selection** problem.
- If no, it's a **rulebook gap**.

## 3. Candidate generation tricks
- Analog search: take the nearest library structures by Tanimoto, then apply edit rules (`derive`) to reach the target mass and formula.
- Widen the search (more parents, more products per parent) and keep the re-ranker strict.
- Budget caps (`gen_max_total`) keep the runtime inside the Kaggle limit.

## In our comps
- **CASMI:** score is MRR@25 on the tautomer-canonical, stereo-free InChIKey14 ([[08 Cheminformatics & MS]]).
  - Class 3 (novel structures, about 39–52% of the test) holds most of the lost points.
  - **gen20** (`gen_n_analog=20, gen_max_per_parent=100, gen_max_total=500`) raised the novel answers generated from 46 to 69 of 130 on the bench.
  - It was the first lever we found ourselves and checked on both benches.
- **NTL:** this is classification, not ranking. But balanced accuracy has the same "fix the decision at the top" logic: only the argmax counts.

## Exercises
1. For the gen20 bench, split the 69 generated class-3 answers by final rank (1, 2–5, 6–25, beyond 25). Where is the next MRR point?
2. Show that a uniform improvement of every rank by 1 is worth more for molecules at ranks 2–3 than at ranks 10–20.

## Resources
- Manning, Raghavan & Schütze, *Introduction to Information Retrieval*, chapter 8 (evaluation).
- Pinecone / Weaviate blog posts on two-stage retrieval and re-ranking.
