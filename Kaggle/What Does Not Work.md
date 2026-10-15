---
tags: [kaggle, playbook, mistakes]
---
# What Does Not Work

Back to [[Kaggle Home]]. Read this before every new experiment.

## Mistakes that cost submissions #mistake
| Mistake | Cost | Rule from now on |
|---|---|---|
| Re-ran a public notebook **without its pinned Docker image** (it ran on Python 3.13 instead of 3.12) | 0.441 notebook scored **0.427** | Always copy `docker_image` from `kernels pull -m` metadata into our `kernel-metadata.json`, with `docker_image_pinning_type: original` |
| Trained the ranker on held-out-model features, ran it with the full model | LB 0.216 → **0.199** | Inference model = training-feature model |
| Compared two runs that differed in more than one thing ("prior did nothing, 0.176 vs 0.176") | A wrong conclusion that we published | One change per comparison |
| Pushed a new kernel version while the previous one was running | Run cancelled | Wait for COMPLETE before pushing |
| Title longer than 50 characters in `kernel-metadata.json` | 400 Bad Request | Keep titles ≤ 50 characters |
| Kernel `id` that didn't match the slug Kaggle created | 409 Conflict | Use the slug Kaggle prints after the first push |
| Unquoted heredoc containing backticks | Shell executed them; text lost | Use `<<'EOF'` |

## Ideas that didn't move the LB
- **Weak engine merged into a strong pipeline** (reciprocal-rank fusion): v2 engine +0.001, v3 engine −0.001 at beta 0.5, **−0.007 at beta 1.0**. A 0.27 engine voting next to a 0.44 pipeline is noise. #lb-proven #casmi26
- **Fingerprint re-rank inside the fork's final 25**: 0.437 under every setting. #lb-proven #casmi26
- **Generated class-3 candidates at rank 2** (mutating the closest library analogs): 42% of answers generated offline, but **−0.001** on the LB. The test's novel molecules seem to be further from the libraries than the validation relatives were. #lb-proven #casmi26
- **Popularity prior tuned on library molecules**: +0.50 MRR offline, only +0.075 on the LB. Library validation molecules are famous by construction. #casmi26

## Offline gains that shrink ~4x on the LB
- Ranker + fragmentation + analogs: +0.14 offline on S2 vs **+0.016–0.04** on the LB. The public LB is ~130 molecules, about half of them class 3 (unreachable by re-ranking). #casmi26

## Environment traps
- `api.kaggle.com` is blocked from the cloud container: use the `kaggle_www.py` wrapper (endpoint `https://www.kaggle.com/api`).
- `kaggleusercontent.com` is blocked, so kernel outputs and logs can't be downloaded. Ask the user to paste the log, or put the key numbers into a public notebook.
- Publishing or first-forking other people's notebooks via the API may be blocked: the user does *Copy & Edit* in the UI.
