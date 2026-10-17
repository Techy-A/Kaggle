---
tags: [study, playbook, casmi26, ntl]
updated: 2026-10-17
---
# 10 Competition playbook

Back to [[00 Study Guide Home]].

**The decision rules, one page.** Each rule cites the study note it comes from. Read this before every submission and before choosing finals.

## Season
- **Until about 17 Nov: play hard.** New experiments are OK if they are **bench-backed or council-backed**.
- **After that: play safe.** Sure-shot entries only: exact copies, pinned reruns and hedges.
- **GPU:** about 15 of the 30 weekly hours go to research. Keep 2 sessions' headroom for Kino when they ask.

## Before an experiment
1. Search [[What Does Not Work]].
2. Write down what the change does to **stage 1 (recall)** or **stage 2 (order)**: [[05 Ranking & Retrieval]].
3. Make **one** change per variant (a single config field): [[04 Ensembling]], [[09 ML Engineering & Reproducibility]].

## Before a submission
1. **Offline check:**
   - CASMI: the unseen bench (honest). The np bench is leaky: [[01 Validation & Leakage]].
   - NTL: grouped-by-prompt CV.
2. **Forecast:** write it into [[Predictions vs Results]], minus the running bias (about **0.002** for new changes).
3. **Expected gain below the noise floor (±0.004)?** Submit only if it is cheap and the season allows.
4. **Verify** the submission in the competition's submissions list.

## Reading a result
- A difference under 0.004 between single submissions is **noise**, not a finding.
- A change that wins on LB but loses on the bench: suspect luck, i.e. the winner's curse.
- A change that wins on **both**: adopt it, and log it in [[What Works]] with both numbers.
- Update the bias table every time.

## Choosing finals
| Comp | Final A (safe) | Final B (upside) | Never |
|---|---|---|---|
| CASMI | V22 + depth 7 (0.451) | gen20 if it scores ≥ 0.452 (it won on both benches) | anything tuned only on the PubChem gate via the bench |
| NTL | best pinned public-line copy at α 0.5 (hedge) | 0.556 line + α 0.25 if it holds | the TabPFN merge (non-commercial licence) |

**Why hedge α:** the α optimum on public data is a winner's-curse pick, while α 0.5 is the principled setting. Two finals cover both: [[06 Calibration & Label Shift]].

## Standing team rules
- Make the 0.451 CASMI notebook public only once we reach **0.453+**.
- Give all times in **IST**.

## Open levers, ranked
1. **CASMI class-3 parent selection:** spectrum-aware parent ranking ([[08 Cheminformatics & MS]], `near_miss.py`).
2. **CASMI class-2 fusion loss:** 11 molecules on the unseen bench ([[04 Ensembling]]).
3. **CASMI listwise ranker** for engine 2 (LambdaMART): [[02 Gradient Boosting & Learning-to-Rank]].
4. **NTL size-weighted centering** for 2–3-passage prompts: [[07 NLP Stylometry]].
5. **NTL model-family gates** (Gemini, GLM).
