---
tags: [kaggle, playbook]
---
# What Works

Back to [[Kaggle Home]]. Every entry has evidence. *LB* = public leaderboard.

## Strategy
- **Exact, pinned reproductions of the best public notebooks reproduce their scores**: Antriksh's 0.443 → 0.443, Ahmed's v7c 0.441 → 0.441 (same Docker image, same datasets, no edits). Unpinned copies lost 0.014. #lb-proven #casmi26
- **Start from the strongest public notebook, reproduced exactly.** The best public notebook (0.441–0.443) beat everything we built from scratch (0.272) by a wide margin. Fork first, then improve. #lb-proven #casmi26
- **Check a notebook's *real* score, not its title.** Titles overclaim. Real scores come from Kaggle's kernel list (`bestPublicScore` from `kernels.KernelsService/ListKernels`). #casmi26
- **Read the official metric code in full.** It told us that reordering after the first correct guess can't hurt, and that identity = InChIKey first block (stereo removed, tautomer-canonical). #casmi26
- **Small, targeted inserts beat big re-weighting.** Antriksh Arya: inserting a strong second engine's top-1 at **rank 2** took 0.441 → 0.443. #lb-proven #casmi26
- **One change per submission, against a known baseline.** Otherwise you can't tell what helped. See the confound in [[What Does Not Work]]. #mistake

## Modelling
- **Train on GPU in a separate notebook; the submission notebook only runs inference.** GPU training of the fingerprint MLP (3 x 4096, 25 epochs, 2.8 h on a free T4) took S2 top-1 from 21% to 39%. Pass the outputs in via `kernel_sources`. #lb-proven (0.256 → 0.272) #casmi26
- **Featurise on the GPU, and verify it bit-for-bit against the CPU path used at inference** (`train_gpu.selftest`). #casmi26
- **Fragmentation as a relative feature** (explained-intensity gap to the best candidate in the same molecule) was the top ranker feature: S2 0.502 → 0.627 offline. #casmi26
- **Analog propagation features**: binned spectral search over the whole library at any precursor mass, then Tanimoto to the hit structures. Ranker +0.04 on the LB (0.216 → 0.256). #lb-proven #casmi26
- **Use the same model for inference as for ranker training.** Fixing that mismatch took the LB from 0.199 to 0.216. #lb-proven #casmi26
- **z-scored features survive model swaps; absolute "gap" features don't.** #casmi26

## Engineering
- **Time-guard every add-on step**: run it in a subprocess with a deadline, and on any error keep the source submission. Nothing we added ever broke a run. #casmi26
- **Write several output files from one run** (`submission_before_fusion.csv`, `submission_fusion_only.csv`, ...) and submit each with `-f`, for a clean A/B. #casmi26
- **Smoke-test locally before spending GPU hours**: a tiny model, a few molecules, and a mock `/kaggle/input`. #casmi26
- **Keep memory flat**: process molecules in batches, `.copy()` slices of big arrays, free big objects *before* forking `multiprocessing.Pool` workers. #casmi26

## Community
- **Data-backed comments get replies from top players.** Our validation-trap numbers started conversations with 57th-place Antriksh Arya. See [[Networking]].

- **Read the diff, not the title (CASMI26, 17 Oct).** The public 'v14seeds' notebooks claimed a seed bag, but only their max_depth 6->7 change actually runs. Trace public LB gains to the lines that execute.
- **A council (researcher + evidence presenter, then advocate vs skeptic, then a judge) is worth running before spending slots.** It killed rel100 using public evidence before we wasted a slot.
- **Error analysis before tuning (CASMI26, 17 Oct).** The bench showed class 3 holds almost all the lost points. The near-miss analysis showed the bottleneck is *parent selection* (6 analogs). One config knob (20 analogs) raised generated novel answers from 46 to 69 of 130 on the bench.
- **Calibrate a bench against known LB results before trusting it.** The unseen bench got 2 of 3 known variant directions right; it fails exactly where its class mix differs from the test (PubChem-only class 2).
