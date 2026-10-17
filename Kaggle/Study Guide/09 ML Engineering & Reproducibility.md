---
tags: [study, engineering]
---
# 09 ML engineering & reproducibility

Back to [[00 Study Guide Home]] · see also [[Code Competition Mechanics]].

> **Core idea:** the score you submit has to be the score you can reproduce. Unpinned code drifts, and drift looks just like a real change.

## 1. Pinning and determinism
- **Pin:** package versions, model and dataset versions (Kaggle dataset version numbers) and the notebook's Docker image (Settings → Environment → "Pin to original").
- **Seeds:** `random`, `numpy`, `torch` (including `torch.cuda.manual_seed_all`) and `PYTHONHASHSEED`.
- **GPU non-determinism:** atomics in cuDNN and scatter ops. Use `torch.use_deterministic_algorithms(True)` where possible, or accept the noise and **measure** it.
- **External services** (PubChem, APIs) change over time. Cache their responses as a dataset for the final runs.

## 2. Code competition hygiene
- Write a small dry run (e.g. 50 test rows) that exercises every code path before the full run.
- Time-box every stage with explicit budgets: candidate caps, batch sizes. Log the stage times.
- Print machine-readable result lines (like `BENCH_JSON`) so logs can be parsed automatically.
- Use one config object (e.g. `EngineCfg`); a variant means changing **one** field.
- Keep a lineage of which notebook came from which, with each one's score: [[Notebook Lineages]].

## 3. Kaggle resource limits (ours)
- 2 GPU sessions at a time; **30 GPU h per week** (~15 h of it for research).
- 5 submissions per UTC day; the reset is at **05:30 IST**.
- The kernel slug is derived from the title, so a renamed title means a new slug. Keep `kernel-metadata.json` `id` in sync.
- Always verify a submission went in through the competition's submissions list, not the CLI text.

## 4. Experiment tracking
- For every run: config diff, git commit, seed, runtime, offline score, LB score, and the forecast from [[Predictions vs Results]].
- MLflow / W&B are fine; a markdown table in this vault is enough at our scale.

## In our comps
- **CASMI:** the same depth-7 code scored 0.451 and 0.447. We don't know whether that was pinning drift (e.g. a PubChem response change) or LB noise; either way it sets our ±0.004 floor.
  - Final picks must be pinned and rerun once to confirm.
- **NTL:** copies of public notebooks must pin the **dataset versions** of the LoRA adapters. A silent update of a public dataset changes the score.

## Exercises
1. Rerun the V22 final on CPU-only stages twice and diff the outputs. Which stage is non-deterministic?
2. Write a 10-line checker that compares two `submission.csv` files and reports the share of rows whose top-1 changed.

## Resources
- Chip Huyen, *Designing Machine Learning Systems* (O'Reilly).
- PyTorch docs: "Reproducibility".
