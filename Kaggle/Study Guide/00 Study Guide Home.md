---
tags: [study, index]
updated: 2026-10-17
---
# Advanced DS study guide

Back to [[Kaggle Home]].

**Purpose:** learn the topics that actually move our leaderboard scores.
- Every note ends with **"In our comps"**: where the topic already decided something in [[Competitions/CASMI26|CASMI26]] or [[Competitions/Name That LLM|Name That LLM]].
- Read [[10 Competition Playbook]] before any submission decision.

![[two_stage_retrieval.png]]

## Topics
| # | Note | Why it matters for us | Figures |
|---|---|---|---|
| 01 | [[01 Validation & Leakage]] | the bench, grouped CV, the noise floor, the winner's curse | grouped_cv, leakage, winners_curse, noise_band |
| 02 | [[02 Gradient Boosting & Learning-to-Rank]] | engine 2's HGB rankers, the NTL gates | boosting_early_stop |
| 03 | [[03 Transformers & LoRA]] | the NTL lora3 base | lora |
| 04 | [[04 Ensembling]] | V22's fusion of engines, the NTL merges | rrf |
| 05 | [[05 Ranking & Retrieval]] | MRR@25, candidate generation (gen20) | two_stage_retrieval, mrr |
| 06 | [[06 Calibration & Label Shift]] | Sinkhorn α, gate thresholds | calibration, sinkhorn |
| 07 | [[07 NLP Stylometry]] | stylegram centering, punctuation rules | none |
| 08 | [[08 Cheminformatics & MS]] | InChIKey14, tautomers, derive rules | tautomers |
| 09 | [[09 ML Engineering & Reproducibility]] | pinning, the 0.451 vs 0.447 rerun, Kaggle limits | none |
| 10 | [[10 Competition Playbook]] | **the decision rules**, one page | none |

## Eight-week plan (weeks start Monday)
This plan fits the [[Daily Routine]]: one Kaggle Learn lesson and 10 O'Reilly pages a day, plus **one study block of about 45 min** taken from this guide.

The order follows the season:
- **Weeks 1–4 (play-hard month, until about 17 Nov):** the topics that feed live experiments.
- **Weeks 5–8 (play safe):** depth and engineering.

| Week | Dates | Topic | Hands-on deliverable |
|---|---|---|---|
| 1 | 19–25 Oct | [[01 Validation & Leakage]] | Re-derive our noise floor from the 0.451 / 0.447 pair. Write the "grouped split" check for NTL. |
| 2 | 26 Oct – 1 Nov | [[05 Ranking & Retrieval]] + [[08 Cheminformatics & MS]] | Read `near_miss.py`; count the class-3 parent-selection misses on the unseen bench. |
| 3 | 2–8 Nov | [[06 Calibration & Label Shift]] | Implement Sinkhorn balancing from scratch (20 lines) and match the public notebook's output. |
| 4 | 9–15 Nov | [[04 Ensembling]] + [[02 Gradient Boosting & Learning-to-Rank]] | Try a LambdaMART (LightGBM `lambdarank`) version of engine 2 offline on the bench. |
| 5 | 16–22 Nov | [[03 Transformers & LoRA]] | Fine-tune a small LoRA (r = 8) on NTL train, using grouped CV. |
| 6 | 23–29 Nov | [[07 NLP Stylometry]] | Build a centered char-n-gram stylegram; compare grouped CV with and without centering. |
| 7 | 30 Nov – 6 Dec | [[09 ML Engineering & Reproducibility]] | Pin every package in the final picks; rerun them twice and confirm identical outputs. |
| 8 | 7–13 Dec | Review + [[10 Competition Playbook]] | Write the post-mortems for both comps into [[What Works]] / [[What Does Not Work]]. |

**Rules for the plan:**
- If a live competition needs something, it wins over the plan for that day.
- Any finding goes into [[What Works]] or [[What Does Not Work]], with a number.
- Each forecast goes into [[Predictions vs Results]] **before** the result lands.

## Regenerating the figures
Run `python make_figures.py` in this folder; it writes `img/*.png`. Edit the numbers there if a real result changes, for example the noise-band chart.
