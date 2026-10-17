# CASMI26 local test bench

`build_bench.py` turns the V22 pinned copy into a bench notebook (kernel `aryanawasthiteykik/casmi26-bench`, private, never submitted).

**Pseudo-test.** The 250 `enveda-np-examples` natural products (1,184 timsTOF spectra), with the answers hidden per simulated class:

| Class | Share | What is removed |
|---|---|---|
| 1 | 28%, 70 molecules | only the query spectra |
| 2 | 20%, 50 molecules | every spectrum of the structure and its tautomer twins; the structure stays a candidate (re-added to engine 2's pool) |
| 3 | 52%, 130 molecules | additionally hidden from engine 1's candidate window (its own `drop_pid` mechanism, extended to a set), from the PubChem gate's best-pool score, from COCONUT + ChEBI/LMSD (engine 2) and from the PubChem channel |

Generated candidates are not filtered, because generating the answer is a real skill.

**Leak controls** (`--leakfree 1`, the default):
- engine 1 uses FPNet A instead of fpnet_full1, which was trained on the panel;
- engine 2 uses megayak's leak-free FPNets;
- engine 2's "ours" GBMs are fit without the np query set.

Still exposed: the PubChem-channel FPNets A+B (they saw other-library spectra of these structures) and the pv GBM rows (roster unknown). So absolute bench scores read high; compare variants against each other.

**Outputs** (`/kaggle/working`):
- `bench_results.json`: MRR@25 overall and per class, top-1 and hit@25 counts, and per-molecule reciprocal ranks, for every stage snapshot.
- Replayed variants of the CPU-only tail (cells 25–37 run again from a pre-25 snapshot): `replay_v22` (determinism check), `rel100`, `rel250`, `e2r3`.
- `bench_sub_<variant>.csv` for each variant.

**Calibration targets** (known public LB, base 0.451):

| Variant | Public LB |
|---|---|
| rel150 | 0.451 |
| rel100 | 0.445 |
| rel250 | 0.445 (omerfarukmerey) |
| E2R3 | 0.450 |
| depth 7 | pending |

The bench is trusted only if it ranks these the same way.

**Engine variants:** `--depth 7 --tag d7` builds the depth-7 bench.

`dryrun_benchcell.py` runs the setup cell locally against the real train.parquet and a mock pool. Result on 17 Oct:
- 250 answers (130/50/70);
- 0 np rows and 0 class-2/3 spectra left;
- all 70 class-1 structures keep their other spectra;
- all 130 class-3 answers are in the v4 pool;
- frag/fp re-indexing verified;
- about 7 min on 4 CPUs.

**Run 1 (17 Oct)** failed in cell 19. The fe_v4 feature caches (`res/frag2`, `analog_struct`, `fragnet`) are row-aligned to the v4 pool, so the pool files can't be filtered.
- Fix: pool files unchanged; the class-3 answers are hidden from `P.window` instead.
- Engine 2 and the PubChem channel ran correctly before the crash:
  - 126 of 130 class-3 answers dropped from COCONUT/bio;
  - 2 class-2 answers re-added (the other 48 were already in COCONUT/bio);
  - leak-free FPNets loaded;
  - 355,926 ranker rows without the np panel.
- Timings: engine 2 about 43 min, PubChem channel about 36 min.

Kernel outputs (kaggleusercontent.com) are blocked from the sandbox, so the bench prints `BENCH_JSON` lines into the run log and `extract_log.py` collects them.
