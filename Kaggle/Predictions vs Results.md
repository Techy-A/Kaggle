---
updated: 2026-10-18
---
# Predictions vs results

Back to [[Kaggle Home]].

**Rules:**
- Write the predicted public score down **before** the result lands.
- After each result, update the running bias.
- Adjust the next predictions by the running bias.

**Running bias, as of 18 Oct (4:40 am IST):**
- New changes: **about 0.005 too optimistic** on average, median 0.002 (6 changes; the stylegram-add miss was −0.010). Misses: rel100 −0.004, E2R3 −0.001, depth 7 −0.002, NTL merge −0.001, gen20 −0.012. gen20 is the outlier: a bench-backed change that lost 0.010 on the LB.
- Exact copies: unbiased (4 of 4 within ±0.001).
- Bias-adjusted ranges: 1 of 2 hit (the merge hit; gen20 missed by 0.010).

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
| 17 Oct | CASMI | V22 + gen20 | our own change | 0.453, **bias-adjusted 0.451–0.452** | **0.441** | **−0.012** (−0.010 vs the adjusted range): far outside the noise floor |
| 18 Oct | NTL | 0.556 + Sinkhorn α 0.25 | combination of two public tricks | 0.557, **bias-adjusted 0.556** | 0.556 | −0.001 (0.000 vs adjusted) |
| 18 Oct | NTL | 0.556 + centered stylegram evidence added (not replaced), β 1 | our own change (lab simulation) | 0.560, **bias-adjusted 0.556–0.560** (stylegram alone +0.05 vs replace in the 2–3/prompt sim; the blend dilutes it) | **0.550** | **−0.010** (−0.006 vs the adjusted range): the sim scored the stylegram alone; in the blend, adding doubled its weight |
| 18 Oct | NTL | 0.556 + GLM+Claude gate (GLM-4-9B features, λ 1) | our own change (lab simulation) | 0.558, **bias-adjusted 0.554–0.558** (gate on the full posterior, same scale as the proven Claude gate; the sim gain was measured on the stylegram only) | 0.555 | −0.003 (inside the adjusted range) |
| 18 Oct | NTL | 0.557 public copy (lightningv08, TabPFN inside) | exact copy | 0.557 | 0.557 | 0.000 |
| 18 Oct | CASMI | 0.454 public copy (guhongbin v459 e006 defaultimg) | exact copy | 0.454 | 0.454 | 0.000 |
| 18 Oct | NTL | 0.559 public copy (obstacledeveloper adaptive centered stylegram), run by Kino | exact copy | 0.559 | 0.559 | 0.000 |
| 19 Oct | NTL | 0.563 public copy (vnton37nguyn ntl-lora-instyle) | exact copy | 0.563 | **0.561** | **−0.002**: the first exact copy to miss (earlier copies 4/4 exact). Likely GPU nondeterminism in the LoRA members, or newer dataset versions mounted than the author's run |
| 19 Oct | CASMI | 0.456 public copy (guhongbin v465 v15 pop015, old image) | exact copy | 0.456 | pending | |
| 19 Oct | NTL | 0.569 public copy (prvsiyan the-prompt-is-not-the-author) | exact copy (the last copy missed by −0.002) | 0.569, **range 0.567–0.569** | pending | |
| 18 Oct | CASMI | A: seyitkaangunes 0.452 code (E006 + rel150 + N1A + CC BY ChEBI) on v459's newer docker image | image swap (v459 = the 0.446 code on this image) | 0.457 (0.452 + part of the +0.008 image effect), **range 0.452–0.460** | pending | |
| 18 Oct | CASMI | B: our V22 + depth7 (0.451) on v459's newer docker image | image swap | 0.455, **range 0.450–0.459**; may fail outright if V22's code needs the old image | pending | |
| 18 Oct | CASMI | E: ahmedberatozer v14 pubchemlite (0.455, old image) on the newer image | image swap on the best code | 0.459, **range 0.453–0.463** | pending | |
| 18 Oct | CASMI | 0.455 public copy (ahmedberatozer v14 pubchemlite, old image) | exact copy; with E, a clean same-code image A/B | 0.455 | pending | |
| 18 Oct | CASMI | C: v459 (0.454) + engine-2 max_depth 7 | our own change (LB-neutral on V22) | 0.454, **range 0.450–0.457** | held (GPU quota) | |
| 18 Oct | CASMI | D: v459 (0.454) + gated gen20 (lib_max < 0.7) | our own change, LB A/B (the bench can't judge generation changes) | 0.452, **range 0.444–0.456**; ungated gen20 lost 0.010 on V22 | held (GPU quota) | |

**Noise floor:** the same CASMI code scored 0.451 and 0.447 on two runs. Treat any single-submission difference under about 0.004 as noise.
