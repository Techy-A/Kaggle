---
tags: [kaggle, casmi26, competition]
competition: enveda-CASMI26-molecule-id-mass-spectra
metric: MRR@25 on InChIKey first block
deadline: 2026-12-14
merge_deadline: 2026-12-07
---
# Enveda CASMI 2026: molecule ID from MS/MS

Back to [[Home]]. Full experiment log: `enveda-casmi26/README.md` in the repo.

## Leaderboard history
| Date | Run | Public LB |
|---|---|---|
| 11 Oct | Cosine library search | 0.150 |
| 11 Oct | Fingerprint MLP v1 + COCONUT | 0.176 |
| 12 Oct | v2 + popularity prior | 0.222 |
| 13 Oct | Fork of the 0.440 public pipeline | 0.437 |
| 14 Oct | Own engine: ranker + fragmentation + analogs | 0.256 |
| 14 Oct | Own engine with GPU v3 model | 0.272 |
| 14 Oct | **Fork + own-engine merge (v2)** | **0.438** (best) |
| 15 Oct | Fork + v3 merge (beta 0.5 / 1.0) | 0.436 / 0.430 |
| 15 Oct | Unpinned Dual-Band + merge (/ + generation) | 0.427 / 0.426 |

## Public notebooks (real scores)
- 0.443: Antriksh Arya, v7c + engine-2 rank-2 insert (pinned image `2757e0…`)
- 0.443: saddamaditya, forensics-lab apex dossier
- 0.441: Ahmed v7c fraggap (pinned 3.12 image `37c64f…`), Dual-Band, MetFrag-lite, …

## In flight
- 16 Oct: 0.443 base / + merge / + merge + generation / agreement-only promotion. See [[What Works]] for the agreement idea once it's measured.

## Lessons
See [[What Works]] and [[What Does Not Work]] (tag `#casmi26`).
