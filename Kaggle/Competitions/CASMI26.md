---
tags: [kaggle, casmi26, competition]
competition: enveda-CASMI26-molecule-id-mass-spectra
metric: MRR@25 on InChIKey first block
deadline: 2026-12-14
merge_deadline: 2026-12-07
---
# Enveda CASMI 2026: molecule ID from MS/MS

Back to [[Kaggle Home]]. Full experiment log: `enveda-casmi26/README.md` in the repo.

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
| 15 Oct | Kino: E007 + LambdaRank + tail fill | 0.440 |
| 16 Oct | Kino: unchanged V22 copy (rel>150, no E2R3) | 0.451 (confirms V22 is stable) |
| 16 Oct | **Exact pinned copy of Antriksh Arya's 0.443** | **0.443** (team best; inside the bronze zone, ~rank 253–308 of 3,397, bronze line 0.442 at rank 339, silver 0.446) |
| 16 Oct | Exact pinned copy of Ahmed's v7c | 0.441 |
| 16 Oct | 0.443 + agreement-only promotion | 0.439 |
| 16 Oct | **Exact pinned copy of easonyanyan's V22** | **0.451**: team best, ~rank 32–54 of 3,407; silver line 0.446 (rank 170), gold ~rank 16 |

## Public notebooks (real scores)
- 0.443: Antriksh Arya, v7c + engine-2 rank-2 insert (pinned image `2757e0…`)
- 0.443: saddamaditya, forensics-lab apex dossier
- 0.441: Ahmed v7c fraggap (pinned 3.12 image `37c64f…`), Dual-Band, MetFrag-lite, …

## Final picks (plan)
- 57257429: V22 copy, 0.451
- 57248900: 0.443 copy, a different lineage, as a hedge

## In flight
- 16 Oct: 0.443 base / + merge / + merge + generation / agreement-only promotion. See [[What Works]] for the agreement idea once it's measured.

## Lessons
See [[What Works]] and [[What Does Not Work]] (tag `#casmi26`).


## Standing, 16 Oct 17:00 UTC
- Rank 63/3427 at 0.451, inside **silver** (top 5% = rank 171, cutoff 0.447). Gold needs top 17 (0.456). Bronze cutoff is 0.443 (rank 342).
- The V22 variants (rel100, E2R3) haven't been submitted yet.

## Publishing rule (user, 16 Oct)
- Make the 0.451 V22 pinned copy public **only once the team is at 0.453+**. Never publish the current best.

## Council evidence, 17 Oct (see the council report when the verdict is in)
- **rel100 has negative public evidence:** seyitkaangunes's sweep on the E006 base gave rel>250 0.442, rel>150 0.446, rel>100 0.444. Held until the user decides.
- **E2R3:** no direct LB evidence. Engine 2 rescues only about 3 public molecules (antriksharya probes). Held.
- **Depth 7 is the lead:** the v14seeds/Gluzdov "seed bag" seeds are dead code, and only max_depth 6->7 in the engine-2 HGB rankers runs. It gained +0.003 (0.441->0.444) and +0.004 (0.446->0.450) on two bases, and no 0.451 notebook has it. Built as push_v22_depth7 (kernel aryanawasthiteykik/casmi26-v22-depth7, not pushed yet).
- **Noise:** the public LB is about 124 molecules (one rank-1 hit is about 0.008), and GPU reruns of the same code differ by about ±0.003.
- **Dead ends measured publicly:** ungated FRIGID rank-2 (-0.009), JESTR (-0.022), CFM-ID LOTUS (0 to -0.018), stepped-CE GLACIER (-0.004), MSEntropy (0), ChEBI+LIPID MAPS in another pipeline (-0.005).

## Council verdict, 17 Oct (judge after a 2-round debate; the user is the jury)
| Motion | Ruling | P(beats 0.451) |
|---|---|---|
| V22 + engine-2 max_depth 7 (single change) | **submit next** | 0.55 |
| V22 + rel100 | reject (public sweep peaks at 150) | 0.20 |
| E2R3 | later, only on a scored depth-7 base | 0.25 |
| Bigger NP catalogue (NPAtlas/LOTUS) | later; first a CPU count of new structures | 0.25 |
| Real seed bag | reject (seed models highly correlated) | – |
| Gated FRIGID class-3 channel | research track, offline only | 0.15 |
| Zenodo Enveda-180 extras, ICEBERG NIST atlas, GLACIER fine-tune | reject (no gain expected / NIST disqualification risk / weeks of work) | 0.15 |
- Score rule for depth 7: >= 0.452 means it becomes the new base and a final pick; 0.451 is a tie and the user decides; <= 0.450 means stop the depth line, no re-runs for noise.
- The local dummy test.parquet is all enveda-180 library hits, so it is useless as a validation set.

- **17 Oct: Kino's V22 + rel100 scored 0.445** (-0.006 vs V22). The council predicted it (20% chance to beat 0.451). The PubChem gate stays at 150. #lb-proven
