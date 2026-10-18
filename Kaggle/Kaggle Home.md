---
tags: [kaggle, index]
---
# Kaggle Playbook

Goal: **Kaggle Expert**. Rule: every mistake gets written down once and never repeated.

## Notes
- [[What Works]] · [[Notebook Lineages]] · [[Predictions vs Results]]: proven by a leaderboard score or a clean A/B
- [[What Does Not Work]]: tried, measured, failed (with numbers)
- [[Code Competition Mechanics]]: submissions, runtimes, environments, gotchas
- [[Validation]]: building CV that predicts the LB
- [[Networking]]: discussions, comments, teaming
- [[Daily Routine]]
- [[Study Guide/00 Study Guide Home|Study Guide]]: advanced-DS topics with figures, an 8-week plan and the [[Study Guide/10 Competition Playbook|competition playbook]]
- [[Study Guide/reports/Kaggle grandmaster handbook|Kaggle Grandmaster Handbook]]: the full path to GM, with hacks, compute and plans for our competitions

## Competitions
- [[Competitions/CASMI26|Enveda CASMI 2026]]: molecule ID from MS/MS (MRR@25)
- [[Competitions/Name That LLM|Name That LLM]]: which of 18 LLMs wrote a paragraph (balanced accuracy)

## How to use this vault
1. Before any new experiment, search [[What Does Not Work]] first.
2. After every leaderboard result, add one line to [[What Works]] or [[What Does Not Work]] with the number.
3. Tag entries with the competition (`#casmi26`) and the type (`#lb-proven`, `#offline-only`, `#mistake`).

## Preferences
- **Compute is free only:** Kaggle GPU (30 h/week per account), Kaggle CPU sessions, free Colab T4 as overflow (`colab/colab_runner.ipynb` runs any of our Kaggle job kernels unchanged and uploads the outputs to a private dataset). No paid GPUs.
- **Always give times in IST** (UTC+5:30), never UTC.
- **Don't use the term "final pick" unless the user uses it first.** Talk about submissions and what each one is for instead.
