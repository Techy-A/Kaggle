---
tags: [kaggle, ntl, competition]
competition: name-that-llm
metric: balanced accuracy (18 classes)
deadline: 2027-01-13
---
# Name That LLM: which of 18 models wrote this paragraph?

Back to [[Kaggle Home]]. Code in the repo: `name-that-llm/`. Team: with Kino Lutor (merge separately per competition).

## Data
- train: 2,699 passages = 150 prompts x 18 models (exactly one per model per prompt); 36–274 words.
- test: ~4,000 passages on **unseen prompts**; per-model counts are unequal and the mix differs between public and private.

## Leaderboard history
| Date | Run | CV (prompt-grouped 5-fold) | Public LB |
|---|---|---|---|
| 15 Oct | TF-IDF char 1–5 + word 1–2, logistic regression | 0.342 | **0.345** |
| 15 Oct | + prompt-group Hungarian assignment (one model per prompt) | 0.433 | 0.335 |
| 15–16 Oct | Kino: name-that-llm-v5 (6 versions) | – | 0.461–0.492 |
| 16 Oct | Kino: exact copy of lightningv08 0.531 Qwen-LoRA ensemble (pinned env) | – | **0.532** (rank 110/735) |

## Lessons
- **Prompt-grouped CV tracks the LB for the classifier** (0.342 vs 0.345). Trust it.
- **The one-per-model-per-prompt assignment does NOT transfer** (−0.010 on the LB): the test doesn't keep that structure. #lb-proven
- LB on day 2: top 0.531, #10 0.499, #100 0.460.

## Next
- 16 Oct: public best is **0.541** (`lightningv08/0-541-lora-adaptive-centering`, Qwen3 8B/4B + Qwen2.5 7B LoRA + encoder kernels, same pinned image `2757e0...`). An exact copy puts us around rank ~77: inside bronze (top 100 for 250–999 teams). Silver (top 50) needs **> 0.541**, because 65 teams are tied at 0.541 and ties go to the earlier submission.
- Fine-tune a small transformer (DeBERTa-v3 / ModernBERT) on GPU; ensemble with TF-IDF.
