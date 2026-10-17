---
updated: 2026-10-18
---
# Predictions vs results

Back to [[Kaggle Home]].

**Rules:**
- Write the predicted public score down **before** the result lands.
- After each result, update the running bias.
- Adjust the next predictions by the running bias.

**Running bias, as of 18 Oct (02:35 IST):**
- New changes: **about 0.002 too optimistic** (mean of rel100, E2R3, depth 7 and the NTL merge: −0.004, −0.001, −0.002, −0.001).
- Exact copies: unbiased (3 of 3 within ±0.001).
- Bias-adjusted ranges: 1 of 1 hit (the merge).

| Date | Comp | Submission | Type | Predicted | Actual | Miss (actual − predicted) |
|---|---|---|---|---|---|---|
| 16 Oct | NTL | 0.531 public copy (Kino) | exact copy | 0.531 | 0.532 | +0.001 |
| 17 Oct | NTL | 0.552 public copy | exact copy | 0.552 | 0.552 | 0.000 |
| 17 Oct | CASMI | V22 + rel100 (Kino) | new change | 0.449 (council: −0.002) | 0.445 | −0.004 |
| 17 Oct | CASMI | V22 + E2R3 (Kino) | new change | 0.451 (council: ~0) | 0.450 | −0.001 |
| 17 Oct | CASMI | V22 + depth 7 | new change | 0.453 (council: +0.002) | 0.451 | −0.002 |
| 17 Oct | CASMI | V22 + depth 7, rerun by Kino | same code | 0.451 | 0.447 | −0.004 (noise or pinning) |
| 17 Oct | NTL | 0.556 public copy | exact copy | 0.556 | **0.556** | 0.000 |
| 17 Oct | NTL | merge (TabPFN + Claude gates, with punctuation) | new combination | 0.551 (bias-adjusted 0.549–0.551) | 0.550 | −0.001 (inside the adjusted range) |
| 17 Oct | CASMI | V22 + gen20 | our own change | 0.453, **bias-adjusted 0.451–0.452** | pending | |
| 18 Oct | NTL | 0.556 + Sinkhorn α 0.25 | combination of two public tricks | 0.557, **bias-adjusted 0.556** | pending | |

**Noise floor:** the same CASMI code scored 0.451 and 0.447 on two runs. Treat any single-submission difference under about 0.004 as noise.
