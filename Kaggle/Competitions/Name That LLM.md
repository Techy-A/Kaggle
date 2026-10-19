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

## Lab: LLM features against the GLM ↔ Claude blur (18 Oct, `lab/03_llm_gate.py`, features cached by `lab/jobs/ntl_featcache`)
Same simulated test (2–3 passages per prompt). Base: the stylegram, plain + centered evidence, 0.389.
- **GLM+Claude 7-way gate on GLM-4-9B L8+L3: +0.035** (λ 1, 5/5). On Qwen3-8B: +0.025. On both feature sets together: +0.033.
- The full 18-way GLM-4-9B head added as a member: **+0.10** (weight 0.5, 5/5).
  - In the 0.556 notebook the plain glm9b head has **weight 0**; only its centered twin counts.
  - Those weights were fitted on full 18-passage prompts, not test-shaped ones, so this is a possible lever to revisit.
- **Caveat:** the real blend already carries LLM information (LoRAs, token attention), so blend-level gains will be smaller than these stylegram-level ones.
- **Built, not yet pushed:** `variants/build_glm_claude_gate.py` adds the gate on the balanced posterior, before the within-Claude gate. It reuses the glm9b features the notebook already computes, so there is no GPU cost. It goes on whichever base wins: stylegram-add (if its score improves) or 0.556.

- **Stylegram-add result (landed by 10:56 am IST, 18 Oct): 0.550, worse than 0.556.**
  - **Why:** the sim scored the stylegram on its own. In the blend, adding the centered evidence to the plain head (both at weight 1.516) doubled the stylegram's say on clustered rows, against 0.03–0.19 for every other member, so it drowned the LoRAs and the encoders.
  - **Lesson:** a single-member sim can't judge a change that alters that member's effective blend weight. Keep total weight fixed, e.g. 0.5 × plain + 0.5 × centered, or simulate the blend.
- **Standing (10:56 am IST): rank 88 of 1,448, bronze.** Medal lines: gold (top 12) **0.570**, silver (top 72) **0.556**, bronze (top 144) 0.555. We're tied at 0.556 but below silver on tie order; anything above 0.556 gets silver back.
- **Next: 0.556 + GLM+Claude gate** (forecast 0.554–0.558). The gate sits on the full posterior at λ 1, the same scale as the existing Claude gate (+0.003 on LB), so the stylegram-add weight problem doesn't apply.
- **18 Oct, 1:2x pm IST:** exact pinned copy of lightningv08/0-557-lora-smollm3-tabpfn (0.557) pushed. The user approved a 4th NTL slot today to win silver back. It contains TabPFN (non-commercial licence), so use it for rank only: never one of the two finally selected entries.
- **GLM+Claude gate result (landed 1:37 pm IST): 0.555 vs 0.556, a wash** (inside the forecast range). The stylegram-level +0.035 didn't transfer: the full blend (LoRAs, token attention) already separates GLM from Claude about as well. **Lab lesson:** both of today's sim-backed changes were measured against the stylegram alone and neither transferred. Next NTL lab work must simulate the blend, which needs out-of-fold predictions of the LLM members.
- **0.557 copy result (landed by 3:52 pm IST): 0.557** as forecast; our new best.
  - **The field moved again:** rank **82 of 1,508**. Medal lines: gold (top 13) **0.572**, silver (top 75) **0.557**, bronze (top 150) 0.556. We're tied with the silver line but below it on tie order, so still bronze.
  - **Takeaway:** copying public notebooks only keeps pace, because the silver line rises about 0.001 a day. Getting clear of the tie needs our own gain over the best public line.


## Lab: blend-level simulation, calibrated against the LB (18 Oct, `lab/04_blend_sim.py`)
Rebuilds most of the 0.556 blend from honest out-of-fold predictions: the stylegram (plain + centered twin), lora and lora2, the two token-attention members, the debv3l/electl heads, the q3_8b/glm9b/debv3l/lora centered twins, mixglm clustering, Sinkhorn balancing, and the within-Claude gate. Left out because no OOF exists: ultrabert, desk, lora3, style_lgb, the q25_7b/style4b twins, and the LoRA Claude head. It is calibrated against five known LB results. Paired deltas vs base over 10 draws:

| Variant | LB | 2–3/prompt | 4–6 | 8–10 | 14–17 | **4–6 + Claude gate** |
|---|---|---|---|---|---|---|
| no_center | −0.004 | **+0.007** ✗ | −0.003 | −0.006 | −0.006 | **−0.003** |
| add (stylegram) | −0.006 | **+0.006** ✗ | −0.003 | −0.005 | −0.009 | **−0.003** |
| alpha 0.25 | 0.000 | +0.001 | −0.001 | +0.001 | −0.002 | **−0.001** |
| GLM+Claude gate | −0.001 | −0.009 | −0.008 | −0.010 | −0.012 | **−0.003** |

- **The 2–3 passages-per-prompt assumption was wrong.** At 2–3, centering looks harmful, the opposite of the LB. From 4 passages per prompt upward, every LB sign is reproduced. **The hidden test most likely has ~4–10 passages per prompt.** That also explains why the stylegram-level sims (steps 1–3, all at 2–3) misled.
- **The GLM-gate mismatch was a missing piece of the sim, not the gate.** The notebook's within-Claude gate re-decides WHICH Claude after the GLM gate, overwriting the GLM gate's within-Claude errors. With it in the sim, the GLM gate goes from −0.008 to −0.003 (LB −0.001).
- **Calibrated sim = 4–6 per prompt + Claude gate.** Use deltas only: the absolute level is ~0.045 above the LB because members are missing.
- **Noise:** losers of −0.003 still beat base in 3–4 of 10 draws. To approve a change, use ≥30 draws and P(better) ≥ 0.9, and check it at 8–10 per prompt too.
- **Speed:** fitting the 5 folds takes ~12 min alone. Running 3 at once without thread limits made each ~3× slower; set OMP_NUM_THREADS when running in parallel.

## Public scan, 18 Oct 5:31 pm IST
- **New top: 0.559, obstacledeveloper "Local candidate: adaptive centered stylegram"** (35 votes). The code is our 0.556 base plus five changes; its markdown only describes the first:
  1. Sinkhorn α 0.25 (LB wash for us).
  2. **Orthographic shield** after Sinkhorn: curly quotes give Claude and GLM −2.5; ≥2 em-dashes give Gemini −2.5. Checked on train: Claude 0/749 and GLM 0/300 passages have curly quotes; Gemini em-dash ≥2 in 0.2%. On prvsiyan's extended data, Claude has curly quotes in 0.2% and GLM in 0%. The Gemini rule is weaker there (7.8%).
  3. Within-family gates for GPT, Gemini and GLM (λ 0.25), plus a retrained Claude gate. These are balanced LinearSVCs trained on train rows plus external rows at weight 0.25: the Claude API passages, and **prvsiyan's extended dataset (CC BY 4.0, 4,584 clean passages, all 18 models)**.
  4. An uncertainty-adaptive Claude gate: the gate is shrunk when the blend is sure.
  5. Its markdown cites Kaggle's data page: **the hidden rerun has ~4,000 passages**.
- **0.558: josephadamski91 "Apex v4 + 10% ATLAS".** It mixes in an ATLAS attribution model (Qwen3-8B with per-source LoRA, trained on train only) at 10%, before Sinkhorn: 0.549 → 0.551 and 0.556 → 0.558. **It could stack on 0.559.**
- **Done:** a private exact copy of 0.559 is pushed (`aryanawasthiteykik/ntl-0-559-pinned-copy`), ready for tomorrow's slots. Its sources are Apache 2.0 notebooks and CC BY data (no TabPFN), so it is OK as a finally selected entry.
- **CASMI:** nothing new above 0.454.

## Lab: screening on the calibrated blend sim (18 Oct, 30 draws each, `04_blend_sim.py --screen --v559`)
Paired delta vs the 0.556 base; P = bootstrap P(better) over draws.

| Variant | 4–6/prompt | 8–10/prompt | Verdict |
|---|---|---|---|
| plain glm9b head w 0.1 / 0.2 / 0.4 | +0.000 / −0.001 / −0.005 | −0.001 / −0.003 / −0.009 | ✗ (its weight 0 is right; q3_8b worse) |
| stylegram ×1.25 | +0.0008 (P 0.96) | −0.0002 (P 0.35) | ✗ not robust across test shapes |
| stylegram ×0.8 | −0.001 | −0.001 | ✗ |
| twins ×1.5 | −0.004 | +0.002 (P 0.99) | ✗ flips with group size |
| twins ×0.67 | −0.002 | −0.008 | ✗ |
| half plain/centered stylegram | −0.002 | −0.002 | ✗ |
| 0.559: orthographic shield | **+0.0009 (P 1.00)** | **+0.0007 (P 1.00)** | ✓ robust |
| 0.559: family gates | +0.0008 (P 0.88) | −0.0001 | ~ |
| 0.559: adaptive Claude gate | −0.0002 | +0.0003 (P 0.89) | ~ |
| 0.559: all four | +0.0014 (P 0.94) | +0.0001 | LB +0.003 |

- **None of today's three own ideas passes.** The 0.556 weights are already near the optimum for any test shape.
- **The shield is the one robust piece:** a hard rule of the "physical invariant" kind. Our next own idea is to **mine more invariants** (characters or patterns a lab never uses in train) and verify them on prvsiyan's data, plus tune the −2.5 strength. All of it can be tested in the sim.
- **The 4–6 calibration still fits the LB best** (8–10 overstates the GLM-gate loss and the α 0.25 loss).

## Lab: more orthographic invariants (18 Oct, `lab/05_invariants.py`, sim `--v559 --ref all559`)
- **Mined per model, kept only if ≤ 1% in train AND in prvsiyan's data:**
  - straight apostrophe → not gpt-5.6-sol / gpt-5.6-luna (fires on 32% of other passages);
  - ≥3 em-dashes → not gemini-3.1-pro / 3.6-flash / 3.8-flash / qwen3.8-max (fires on 7%).
  - At lab level, only the 0.559 rules survive.
- **On top of the full 0.559 recipe (30 draws):** apostrophe rule +0.0000 (4–6) / +0.0001 (8–10, P 0.98); em-dash ≥3 rule 0.0000; shield ×2 0.0000.
- **Dead end.** The rules are true, but the blend already obeys them: it almost never predicts sol/luna on a straight-apostrophe passage. A hard rule only pays where the model breaks it, and that was only the curly quote → Claude/GLM case.
- **Next NTL candidate: 0.559 + 10% ATLAS** (`variants/build_atlas_mix.py`, push dir `push_ntl559_atlas`). Public evidence: +0.002 on two bases. ATLAS's adapter licence is "other" with no terms, so it is **for rank only**, like TabPFN.
- **8:50 pm IST: the 0.559 copy is handed to Kino** (ours sat in Kaggle's GPU queue for 3 h+ and was deleted, with no quota used). Forecast 0.559.
- **11:34 pm IST standing: rank 132 of 1,605 at 0.557 (bronze).** The field moved fast: 112 teams are above 0.557. Medal lines: gold (top 13) **0.581**, silver (top 80) **0.559**, bronze (top 160) 0.556. Kino's 0.559 copy would only tie the silver line, so we need our own gain (0.559 + ATLAS is next). Kino had not submitted yet.
- **3:08 am IST, 19 Oct: new public 0.563, vnton37nguyn "NTL lora instyle"** (v16). It is the 0.559 lineage plus prvsiyan blocks, family gates, in-style extra passages in the stylegram and an in-style Qwen2.5-7B LoRA. Its author found the shield and α 0.25 neutral or negative in their CV and leaves them out. No TabPFN or ATLAS. Exact copy pushed (`ntl-0-563-pinned-copy`); it started running immediately, so the GPU queue had cleared. Forecast 0.563. It supersedes Kino's 0.559 copy.
- **0.563 copy result (landed by 6:00 am IST, 19 Oct): 0.561, our new best (−0.002 vs forecast).** **Rank 79 of 1,685: silver** (top 84). Medal lines: gold (top 13) **0.581**, silver (top 84) **0.561**, bronze (top 168) 0.557. We are on the silver line itself (30 teams at 0.561, 58 above), so any small gain secures it. A further NTL submission with no description was made at 5:54 am IST (probably Kino's 0.559 copy) and is pending.
- **6:10 am IST, 19 Oct: new public 0.569, prvsiyan "The Prompt Is Not the Author".** It re-implements the 0.563 notebook from scratch and adds **shared-prompt Gaussian group heads**, which model the prompt effect and integrate it out instead of subtracting a finite-sample group mean. That is the ingredient worth learning from for our own work. No TabPFN or ATLAS; same inputs as the 0.563. Exact copy pushed (`ntl-0-569-pinned-copy`), running, auto-submits. Forecast 0.567–0.569.
- **Kino's 0.559 copy (submitted 5:54 am IST, 19 Oct): 0.559, exact.** Below our 0.561; the 0.569 copy is still scoring.
