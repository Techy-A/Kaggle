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
| 16 Oct | Kino: exact copy of lightningv08 0.531 Qwen-LoRA ensemble (pinned env) | – | 0.532 (rank 110/735) |
| 17 Oct | Us: exact pinned copy of zbigniewkirchner/ntl-v6 | – | **0.552** (rank 33/1,032, silver zone) |

## Lessons
- **Prompt-grouped CV tracks the LB for the classifier** (0.342 vs 0.345). Trust it.
- **The one-per-model-per-prompt assignment does NOT transfer** (−0.010 on the LB): the test doesn't keep that structure. #lb-proven
- LB on day 2: top 0.531, #10 0.499, #100 0.460.

## Next
- 16 Oct: public best is **0.541** (`lightningv08/0-541-lora-adaptive-centering`, Qwen3 8B/4B + Qwen2.5 7B LoRA + encoder kernels, same pinned image `2757e0...`). An exact copy puts us around rank ~77: inside bronze (top 100 for 250–999 teams). Silver (top 50) needs **> 0.541**, because 65 teams are tied at 0.541 and ties go to the earlier submission.
- Fine-tune a small transformer (DeBERTa-v3 / ModernBERT) on GPU; ensemble with TF-IDF.

## 17 Oct
- Public notebooks moved overnight: zbigniewkirchner/ntl-v6 **0.552**, lightningv08 0-551 (TabPFN), kirilltym 0.550, lucasboesen 0.549. The planned 0.541 copy is obsolete.
- Built an exact pinned copy of ntl-v6 (aryanawasthiteykik/ntl-0-552-pinned-copy, image 2757e0..., all inputs public; uses the public Claude-API passages dataset, which the rules allow as free public external data). Pushed from our account after the weekly GPU quota reset. A watcher submits it once with `submission.parquet` when the run completes.
- Its trick: a third Qwen3-8B LoRA with a 5-way within-Claude head, plus "Claude gates" (when the blend predicts a Claude model, Claude-only models choose which one).

- **17 Oct: 0.552 copy scored 0.552, rank 33 of 1,032.** With more than 1,000 teams the medal lines are percentages: bronze top 10% (rank 103, 0.547 now), silver top 5% (rank 51, 0.550 now), gold top 12 (0.555 now). 14 teams are above us and 23 are tied at 0.552.

## 17 Oct night: merge experiment
- zbigniewkirchner ntl-v6 (0.552) and lightningv08 0-551 (0.551) both extend Lucas v3 (0.549) with non-overlapping additions:
  - zb: lora3 (Qwen3-8B LoRA with a Claude head) plus within-Claude gates, about 50 lines;
  - lv: v3 run as a script, then a TabPFN complement (0.1 weight) and punctuation rules.
- The merge (`aryanawasthiteykik/ntl-merge-tabpfn-claude-gates`, built by name-that-llm/merge/build_merge.py):
  - lv shell, with lucas_reference.py replaced by the zb main code;
  - the zb code saves its Claude-gate scores;
  - the gate is re-applied last, after the TabPFN mix and punctuation rules.
- Datasets are the union of both (all public).
- Submitted once when its run completes (the user asked for a few NTL submissions).
- Hoped-for result: the two gains stack, about +0.002–0.004, which would be the silver line (0.554).
- Note: Lucas's own latest version already adds family gates and more proxy-generation datasets. Its score is unknown (bestPublicScore is still 0.549).

## NTL research (17 Oct night, research agent)
**Public notebooks above 0.552**, all forks of zbigniewkirchner v6:

| Notebook | Public LB | Change |
|---|---|---|
| obstacledeveloper centered stylegram | **0.556** | prompt-centered LinearSVC twin of the stylegram, *replacing* it on clusters of 2–18 |
| liurouhan v11 | **0.555** | BALANCE_ALPHA 0.5 → 0.25 only |
| lightningv08 | 0.554 | centered stylegram + α 0.25 + TabPFN + punctuation |

**No gain:**
- punctuation veto (0.552; Lucas says −0.003 to −0.005);
- DeBERTa complement;
- uncertainty-adaptive Claude gate;
- Hungarian matching;
- GPT gate (−0.001).

**Hints from the discussions:**
- The hidden test has about **2–3 passages per prompt** (not 18). Size-weighted centering `(n-1)/(n+1)` beat hard centering, 0.460 vs 0.427.
- Use extra Claude data only to choose *which* Claude.
- About half of all errors stay within the same lab (Gemini Flash versions, GLM ↔ Claude-opus, GPT versions).

**Ranked ideas:**
1. Centered stylegram + α 0.25 (unpublished combination; est. 0.556–0.559).
2. Size-weighted centering.
3. prvsiyan's extended 18-model dataset (new, unused, licence unchecked).
4. Gemini-Flash / GLM gates.
5. Token log-prob (Binoculars-style) features.

TabPFN weights are under a non-commercial licence, which is a winner-licence risk.

**Actions (night of 17 Oct):**
- exact pinned copy of the 0.556 (`ntl-0-556-pinned-copy`), auto-submit;
- `ntl-556-sinkhorn-alpha-025` (idea 1), pushed when a GPU slot frees, then submitted;
- the merge experiment (with punctuation, expected weak) is still auto-submitting as an information point.

**Results (landed 18 Oct, by 02:35 IST):**
- 0.556 exact pinned copy: **0.556**, our new best (was 0.552).
  - Rank **53 of 1356**: back in **silver**.
  - Medal lines: gold (top 12) 0.560, silver (top 67) 0.555, bronze (top 135) 0.552.
  - 59 teams sit at 0.556 or above; 30 are above it.
  - 26 teams sit at 0.555, just under the silver line. One more step up by the field pushes us out.
- Merge (TabPFN/punctuation + lora3/Claude gates): **0.550**, below both parents (0.551, 0.552).
  - Merging two notebooks that share the same base signal doesn't add.
  - It is out of the finals anyway (TabPFN licence).
- Next: α 0.25 on the 0.556 line submits at 5:37 am IST (forecast 0.556 after the bias adjustment).
- **18 Oct, 5:38 am IST: 0.556 + Sinkhorn α 0.25 submitted (57330793).** Forecast 0.556 after the bias adjustment (raw 0.557). It's the first NTL submission of the UTC day.
- **α 0.25 result (landed by 7:55 am IST, 18 Oct): 0.556, the same as α 0.5.** Lowering the Sinkhorn α doesn't move the public score on this line.
- **The field moved overnight:**
  - We're rank 65 of 1,403. The medal lines are gold (top 12) **0.565**, silver (top 70) **0.556**, bronze (top 140) 0.553.
  - 46 teams are now above 0.556, up from 30, and the leader is at 0.616. Our silver hangs on tie order at the line.
  - The next real gain has to come from our own work (lab simulation first), not from α.

## Lab: where the stylegram's errors are (18 Oct, CPU, `name-that-llm/lab/02_confusion_gates.py`)
The test is simulated (held-out prompts, 2–3 passages per prompt, the notebook's clustering); the base is plain + centered evidence.

![[ntl_confusion.png]]

- **Errors:** 52% stay inside the right lab, 48% cross labs. Lab-level accuracy is 0.71.
- **Weakest classes:** claude-fable-5 0.14, gpt-5.6-sol 0.16, glm-5.3-flash 0.18, kimi-k3 0.20, glm-5.3 0.23.
- **Cross-lab confusions:** GLM ↔ Claude (fable-5, sonnet-5), and Kimi/Qwen → gpt-5.5.
- **Within-lab blur:** the Gemini 3.6/3.7/3.8 flashes, and the GPT 5.6 luna/sol/terra trio.
- **Within-lab gates on the same stylegram features: no gain** (Claude/Gemini/GLM/GPT all within ±0.002). This matches the public GPT gate's −0.001. A gate needs different information (LLM/LoRA features), not the same features retrained on a subset.

