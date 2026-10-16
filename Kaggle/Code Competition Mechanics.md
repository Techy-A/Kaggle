---
tags: [kaggle, mechanics]
---
# Code Competition Mechanics

Back to [[Kaggle Home]].

## Submissions
- **5 per day**, reset at **00:00 UTC = 05:30 IST**. Plan them the evening before.
- A submission **re-runs the notebook on the hidden test**. The fork-based pipelines take **6–8 h to score**.
- `competitions submit -k OWNER/KERNEL -v VERSION -f submission.csv -m MSG`. **Some competitions (CASMI26) only accept a file named `submission.csv`**, so an A/B needs one kernel version per variant.
- At most **2 GPU notebook runs at a time** per account.
- GPU stages aren't bit-reproducible: the same code scored 0.437 and 0.440. On ~130 public molecules one molecule ≈ **0.008**. Don't chase ±0.002.
- Public "best scores" are a **maximum over many runs**. Expect a reproduction to land a little lower.
- **Teams share the 5/day.** The merge deadline is usually a week before the end.

## Team rules (CASMI26, from 16 Oct)
- Submit only entries with a **sure-shot 0.441+** expectation: exact pinned copies of proven public notebooks, or zero-risk tweaks on them. Experiments go through validation first.
- Use at most **3 of the 5 daily team slots**; **2 are reserved for Kino**.

## Runtime
- Submission notebook ≤ 9 h. Training notebooks (not submitted) can run up to 12 h on GPU (30 GPU-h per week).
- The commit run of the fork lineage uses a **smoke switch** (12 molecules when the visible test is detected), so commits are quick; the full run happens only when scoring.
- Time-guard add-on cells with `psutil.Process(os.getpid()).create_time()`. `T0` gets redefined inside the forks.

## Environment
- **Pin the Docker image** of any notebook you reproduce (`docker_image` + `docker_image_pinning_type: original`).
- `machine_shape: NvidiaTeslaT4` (recent PyTorch dropped P100 support).
- Internet off: wheels come from datasets (`metric/rdkit-2026-3-3-wheel`).

## Workflow that works
1. `kernels pull -m` the source and read its metadata.
2. Copy it unchanged, with the same image and the same datasets, plus a credits cell on top.
3. Append our cell at the end, guarded, writing **extra** output files.
4. Smoke-test our cell locally against a mock `/kaggle/input`.
5. Push, wait for COMPLETE, then submit each output file with `-f`.
