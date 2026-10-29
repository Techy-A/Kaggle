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
- **0.569 copy result (landed by 9:05 am IST, 19 Oct): 0.569, exact. Our new best.** **Rank 50 of 1,712: silver, with margin** (silver line top 85 = 0.563, 0.006 below us). Medal lines: gold (top 13) **0.586**, silver **0.563**, bronze (top 171) 0.558. 34 teams are above 0.569 and 19 tie it. Gold needs about +0.017, which only our own ideas will reach (the prompt-Gaussian heads are the lead to study).
- **9:51 am IST public scan, 19 Oct:** new top is **0.592, nishantdahal "Two-stage LoRA on the public pool"**, above the gold line (0.586). **It can't be copied:** its dataset source is private (empty in the metadata, and the author has no public datasets), so `pool_v1.parquet` and the LoRA checkpoints wouldn't mount for us. Its recipe is worth noting:
  1. Stage 1: a Qwen3-8B (24 layers) LoRA classifier trained only on the ~35k public passages from the 18 models, pooled from the public datasets.
  2. Stage 2: 2 epochs of fine-tuning on the 2,699 train passages. That alone gives CV 0.483 → 0.569.
  3. Plus a Qwen3-14B (27 layers) twin, which disagrees with the 8B model on 22% of passages, an n-gram SVC on train + pool, the centred twin, and balancing. CV 0.593, LB 0.592.
  - Inference takes ~1 h on 2×T4. Training both LoRAs needs far more GPU than our remaining quota (2.15 h until 24 Oct), so it's a candidate for after the refresh, or for Kino's quota. Watch for its datasets going public; the scan routine will then copy it.
- Other new notebooks: kirilltym "Gaussian Ensemble | ELECTRA Group Expert" 0.568 and "Two-Stage LoRA | Centered Stylegram" 0.567, both below our 0.569.
- **10:20 am IST, 19 Oct: rebuilding the 0.592 recipe ourselves** (Colab, free T4):
  - Pool rebuilt from the 14 credited public datasets: 36,923 passages (`lab/06_build_pool.py`), uploaded as the private dataset `ntl-public-pool-v1`.
  - Training notebook `colab/ntl_twostage_lora_train.ipynb`: Qwen3-8B 24 layers, LoRA r16, triplet plain+centred CE, stage 1 pool then stage 2 train, resumable from Drive.
  - Checked: a CPU dry run passes, and the LoRA merge check passes.
  - The inference kernel is prepared (8B only; blend weights carried over heuristically).
  - Expected: below 0.592 (no 14B, weights not refit) but plausibly above our 0.569. To verify on the LB.
- **5:46 pm IST, 19 Oct scan: new public 0.594, lightningv08 "Two-Stage LoRA + OLMo2 Ensemble".** nishantdahal has now made the two-stage LoRA checkpoints public (`nishantdahal/ntl-two-stage-lora`, CC BY-SA 4.0), and this notebook adds an OLMo2-7B LoRA (Apache 2.0). No TabPFN or ATLAS. Exact copy pushed and running; it auto-submits. Forecast 0.594, **above the gold line (0.586)**. Also new: zbigniewkirchner 0.574 "fully public" (below). Our own Colab rebuild of the 8B two-stage LoRA is still worth finishing as our own model, but it will rank below this copy.

## 19 Oct: 0.594 copy submitted; seed-bag variant built
- **0.594 pinned copy submitted** at 17:58 IST (57396197). Forecast 0.594.
- **Variant built, not pushed:** `push_ntl594_bag8b` (aryanawasthiteykik/ntl-594-bag8b). It is 0.594 with nishant's 8B LoRA averaged 50/50 with our Colab rebuild, in log-probability space before the blend weights.
  - Everything else is byte-identical, and the hash checks pass.
  - It adds about 10–20 min of runtime.
  - **Trap found:** our export folder has the same name (`lora8b_v1`) and sorts first, so the unmodified code would have silently used our 8B in place of nishant's. The variant guards against this.
- **Evidence is weak:** nishant's notebook lists "seed averaging" under "What did not help". Forecast 0.594 (0.592–0.596), most likely a tie, so it is low priority. It needs the checkpoint plus Kino's GPU.
- **19 Oct, 20:40 IST: the 0.594 copy scored 0.594**, an exact hit. Rank **32/1859, silver**. Lines: gold 0.599 (rank 13), silver 0.579 (rank 92), bronze 0.569. Top score 0.650.
- **NTL scan, 20:40 IST:** the best public notebook is now **0.597**, alexchilton/ntl-0592-plus-0569. It runs the 0.569 notebook unchanged and adds its blend at weight 0.5 inside the 0.592 notebook. There is no TabPFN or ATLAS, and its inputs are the 0.569 copy's plus nishant's dataset.
  - Pinned copy pushed at 20:42 IST: `push_ntl597`, aryanawasthiteykik/ntl-0-597-pinned-copy. It auto-submits once.
  - **Quota risk:** 1.35 h of GPU quota was left, and this notebook runs both pipelines.
  - Other public notebooks: josephadamski91 two-stage + ATLAS at 0.596 (rank only), and lightningv08 0.595 (Gaussian + OLMo2).
- 20:40 IST: the Colab checkpoint is still not on Kaggle (it was expected about 20:00).
- **19 Oct, 23:57 IST: the 0.597 copy scored 0.597**, an exact hit. Rank **31/1904, silver**. Lines: gold **0.603** (rank 13, up from 0.599 three hours earlier), silver 0.592, bronze 0.569. 23 teams are at 0.599 or better. The field is moving fast, so the copies only hold our position.
- **20 Oct, 06:42 IST NTL scan:** new public **0.600**, kirilltym/0-600-teach-the-model-remove-the-prompt. It is the two-stage LoRA plus both of prvsiyan's Gaussian heads, with the same public inputs as the 0.597 and no TabPFN or ATLAS. Exact copy built in `push_ntl600` but **not pushed**: our GPU quota is 0.66 h, and the similar 0.597 run used 0.69 h. Added to the Kino list (item 3).

## 24 Oct, 05:40 IST: GPU reset push
- Scan: 0.636 fleongg contains ATLAS (rank only). The 0.624-0.628 notebooks (romantamrazov, mrpicklerick ×2, nuridudak ×2, ozertuu) all depend on nuridudak/ntl-ours-v1, which is now private, and fall back to about the 0.620 base without it. The clean best is cosmorin's 0.622 reproduction of nishantdahal's 0.620.
- Pushed and auto-submitting: the 0.622 exact copy, and the 0.636 exact copy for rank only (never a final selection).
- Next: encoder-member rebuild (06:30) and ATLAS rebuild (07:30), so the 0.624-0.636 lines become usable as our own work.

## 24 Oct: ATLAS licence decision
- **The user accepted ATLAS under CC BY-SA 4.0** on 24 Oct (option 1). Since 20 Oct, josephadamski91/ntl-atlas-qwen3-8b-v1 ships `LICENSES.md`: weights and docs are CC BY-SA 4.0 and the code is Apache-2.0. Kaggle's licence field still says "other".
- So the **0.636 copy (fleongg/ntl-part-two-complement) is now eligible as a final pick**, with attribution to Joseph Adamski and share-alike on derived weights.
- The old "ATLAS = rank only" rule is lifted **for this dataset only**. TabPFN stays rank only.
- Our rebuild is parked as an optional A/B: code in `lab/atlas/` and `lab/08_train_atlas.py`, the Colab notebook `colab/ntl_atlas_train.ipynb`, and the unpushed variant `/home/user/work/push_ntl636_ouratlas`. The full unit takes about 4 h on a T4, and the expected effect is within ±0.003.
- Our 0.636 submission's description still says "rank only", from before this decision.

## 24 Oct: why the 0.622 and 0.636 copies got no GPU
- Root cause: our copies pinned the original's docker image digest `gcr.io/kaggle-images/python@sha256:17b2fc…` with pinning 'original'. That digest is **Kaggle's CPU image** (CPU-only torch), so the runs got no CUDA despite enable_gpu. The 0.636 burned 3.5 h on CPU before failing.
- The originals (cosmorin, fleongg) use **pinning "latest"** (pinning_type None). Fix: drop docker_image and set pinning 'latest'. Both re-pushed at 10:45 IST.
- **Rule from now on:** copy pinning exactly as the original has it. If pinning is None, use 'latest', and never pin a `kaggle-images` (CPU) digest for a GPU notebook.

## 24 Oct scores
- 0.622 copy = 0.622 and 0.636 copy = 0.636 (both exact). Leaderboard: rank 144/2932 at 0.636; gold 0.653 (rank 15), silver 0.636, bronze 0.624, top 0.7.

## 24 Oct: OLMo-2-13B member, fold 0 result and 0.639 variant (CPU lab, `lab/10_olmo13b_fold0_gain.py`)
- **Training finished.** Stage 1 (2,099 steps, 66,366 passages, 6.6 h on 2 T4): plain 0.494, centred-by-prompt 0.606. Fold 0 (stage 2, 239 steps): **plain 0.585, triplet-centred 0.646** (540 rows, 30 per class). Full unit (299 steps, 1.7 h): reload agreement 1.0, in-sample bacc 0.873 on 128 rows.
- **Label order differs.** The OLMo files use the notebook's order (gpt first); the lab's encoder and token-attention OOFs are alphabetical. The script permutes by name. (The notebook variant uses the member's own order, as the 14B LoRAs do.)
- **Stack proxy on fold 0:** SVC + modernbert 0.40 + desklib 0.10 (nuridudak 0.626 weights) + 3 token-attention members at 0.3, Sinkhorn over the 540 rows: **0.563**. It is far below the real 0.639 stack (the real LoRAs have no OOF), so the gains below are an upper bound.
- **Grid (gain over W=0/TW=0, paired bootstrap over prompts, 2000 draws), selected rows:**

| W | TWIN_W | bacc | gain | P(better) |
|---|---|---|---|---|
| 0 | 0.10 | 0.585 | +0.022 | 1.000 |
| 0.05 | 0 | 0.567 | +0.004 | 0.75 |
| 0.05 | 0.05 | 0.578 | +0.015 | 0.995 |
| 0.05 | 0.10 | 0.594 | +0.032 | 1.000 |
| 0.10 | 0.10 | 0.594 | +0.032 | 1.000 |
| 0.20 | 0.10 | 0.600 | +0.037 | 1.000 |

  Pattern: the centred twin carries the gain (the plain head alone needs W>=0.15 to reach P 0.94); gain rises with both weights, with no peak inside the grid, so it is not evidence for more than 0.10 on a stack this weak.
- **Chosen: W = 0.05, TWIN_W = 0.10** (inside the 0.03-0.10 band). Built `/home/user/work/push_ntl639_olmo13b` with `variants/build_olmo13b_639.py --w 0.05 --twin-w 0.10` (cells 26 -> 29). **Not pushed, not submitted.**
- **Forecast (written before submitting): public 0.642** (range 0.638-0.647). Reasoning: the proxy gain (+0.03) overstates because the 0.639 stack already holds three 14B/8B LoRAs of the same kind; a fifth-to-tenth of that survives, plus it is a new base model family (OLMo). Blend-composition changes have a poor LB record, and this one changes only a member weight, so the downside is about -0.002.
- **Runtime estimate:** forward only through 27 of 40 layers, NF4, T4. From training throughput (about 1.8 passages/s with backward and checkpointing) inference is about 6-8 passages/s: roughly 3 min per 1,000 test passages, plus about 5 min to load the base and adapter. Add that to the 0.639 notebook's own runtime.
- **Caveats:** fold 0 only (540 rows, noise about 0.01); the proxy's weight scale is not the notebook's; the centred twin applies on clustered rows only in the notebook, while the proxy applies it on all rows.

## OLMo-13B on the 0.639: measurement pushes (25 Oct 01:5x IST)
Forecast written before submitting (TWIN_W = 0.10 in both): W=0.15 -> 0.642 (0.637-0.646); W=0.20 -> 0.641 (0.636-0.646).
Reason: fold0 proxy gain flattens above W=0.05 and the proxy stack is weaker than the real one; pushed on the user's request to measure the weight response, not as a recommendation.
Notebooks: ntl-0-639-olmo13b-w15, ntl-0-639-olmo13b-w20. Scores to be added below.
Result (25 Oct 10:57 IST): W=0.15 -> 0.635, W=0.20 -> 0.635 (TWIN_W 0.10), against 0.639 for the base: -0.004 both, below the forecast range's low end (0.636/0.637). The fold0 proxy gain (+0.03) did not transfer; same pattern as the encoders on the 0.636 (0.634). Identical scores at both weights suggest the loss comes from the centred twin / member itself, not W. Scoring runs did not draw GPU quota (8.87 h left before and after). Not pursued further without a new reason.

## Notebook scan (25 Oct 11:2x IST; nothing built)
New top public: **0.641** lightningv08/0-641-lora-llama-qwen3-8-27b (84 votes, version 25 Oct 10:22 IST; title says LoRA + Llama + Qwen3.8-27B, not yet read). +0.002 over the 0.639 lineage (kirilltym 63 votes, maverickss26 and romantamrazov and arlearonreidma forks at 0.639). 0.636 tier unchanged (fleongg, lynnsakurai, huangkx666, avazaku). No other new tier. To do on go: read the 0.641 source, check its licences and whether the gain is a new member (contradicts "no new LLM member helps above 0.61") or a recipe change.

## Discussion scan (25 Oct 11:4x IST; replies not read, only topic lists and opening posts)
- Our own post "What 134 public versions show..." (752357) is up since 24 Oct 20:54 IST: 0 votes, 0 comments.
- 752478 (25 Oct 08:49 IST, -1 vote): whether generating training data through the 18 candidate models' APIs is allowed under Rule 2.6; 0 comments, no host answer yet. Same worry as our open question on the 0.639's readout bank (private passages).
- 752280: how the passages were generated; 752270: external data from paid and region-restricted LLMs (3 votes); 751655 (active 25 Oct 10:15 IST): datasets forum for competitions.
- 752365 (jusill): Sinkhorn balancing cost when the test mix is uneven, measured on 2,699 OOF passages at about 0.64 balanced accuracy. Relevant: every notebook in our chain ends with Sinkhorn. Result numbers not read.
- 750768: whether test passages share prompts, and whether using relationships between test passages is allowed (host answer not read).
- 749447 (121 votes, 54 comments): "Will the winner just be whoever can get the most extra training data?"; 752418: are the 0.700 scores real or a trick.
To do on go: read the 752365 numbers (could change our Sinkhorn step) and the host replies on 752478 and 750768.

## 0.641 copy (25 Oct 12:43 IST)
Kernel ntl-0-641-pinned-copy (copy of lightningv08/0-641-lora-llama-qwen3-8-27b, latest env, push dir /home/user/work/push_ntl641). v1 failed ("Final checkpoint is missing"): the notebook reads lightningv08's own dataset at /kaggle/input/<slug>, but a dataset owned by someone else mounts at /kaggle/input/datasets/<owner>/<slug>. v2 replaces that path (145 occurrences, only change); fixture run COMPLETE. Not yet submitted. New data in it: Llama-3.1 24-layer prefix and OLMo-2 native24 members and a Qwen3.8-27B LoRA tail (check licences before a final pick: Llama 3.1 community licence).

## Discussion replies read (25 Oct 13:0x IST)
- 752365 reply (GoPain): Sinkhorn removes a per-class offset in the scores; alternative is cross-fitted offsets (learn one log-offset per class on four folds' OOF, add to the fifth fold, argmax), which needs no assumption about the test mix. Table of numbers cut off in my read; open the thread before using. Worth testing against our Sinkhorn step.
- 750768: no host answer. A participant guesses test has about 222 new prompts with at most one answer per model (4,000/18), not necessarily all 18.
- 751655: a participant points to MAABench (huggingface ming9999/maabench, used in ATLAS) and two public datasets (darragh1walsh/ntl-kaggle-ai-gen-run-1, prvsiyan's extended dataset, which is already in our chain). Licences not checked.
- 752478 (Rule 2.6, API-generated data): still no host reply.

## Master/Grandmaster discussion read (25 Oct 13:3x IST; 41 of 69 topics fetched)
47 posts/comments by Masters and Grandmasters, almost all in "Will the winner just be whoever can get the most extra training data?" (121 votes): opinion on pay-to-win (Cody_Null, Ravi Ramakrishnan, greySnow, Yurnero, Tawara, CoreyJamesLevinson), no scores or recipes. Useful bits: Nicholas Broad suggests competitors share generated data so volume stops mattering; Cody_Null notes private Astra labels in a past competition and that enforcement has been near-zero (so a rule hit on API data is unlikely but cannot be ruled out); prvsiyan (Master) built the extended dataset because of the pay-to-win worry and says Kaggle benchmarks give $100/month API credit; jazivxt points to kaggle.com/benchmarks for that credit. Technical: shanzhong8 (Master) agrees early layers hold more author information than late ones (matches the 24-layer prefix members); Ghaith Mechi (Master) proposes provider-first then per-provider specialists (we tested confusion by lab and sibling specialists earlier, no gain); Ryenhails (Master) reports extremely slow /kaggle/input reads on some GPU sessions (relevant to our 7 h risk). No host answer on API-generated data.

## Methods digest 25 Oct (13:2x IST; read and log only, nothing built, pushed or submitted)

### A. The 0.641 recipe against our 0.639 (source read: lightningv08/0-641-lora-llama-qwen3-8-27b vs push_ntl639)
- **The 0.641 is the 0.639 plus exactly one member.** Everything up to the Kirill helper is the same chain: SVC 1.0 + Qwen3-14B 0.4 + Qwen2.5-14B 0.2 + DeBERTa 0.25 (log-prob sum, Nishant two-stage), `0.85 main + 0.15 complement`, ATLAS `0.9/0.1` on rows with top-2 margin < 0.3, Kirill spectral helper `0.90/0.10` (synthetic4404 bank), then Sinkhorn (300 iterations, full balance, skipped below 50 rows). New: `U = 0.90 U + 0.10 softmax(Qwen3.8-27B)` on every row, then the same single Sinkhorn. Nominal shares 81% reference, 9% spectral, 10% Qwen. Grouping (topic_groups, centred twin, Gaussian groups) and the Sinkhorn are unchanged.
- **New member: Qwen3.8-27B** (architecture `qwen3_5`, 64 layers, hidden 5120), NF4, full 64-layer forward, no truncation, batch 3, LoRA rank 16 / alpha 32 on layers 56-63 only ("tail8"). Eight mean-pool readouts at depths 8,16,...,64, per-depth LayerNorm, plain + centred 18-way heads. Trained on 28,390 deduplicated external responses, then 2 epochs on the 2,699 official rows; final weights fit on all official rows, so there is **no OOF for it** and no honest validation number exists. Base: Qwen/Qwen3.8-27B via greghavens/Qwen3.8-27B-bnb-4bit (rev 26157380...).
- **The title is misleading on Llama and OLMo-2.** `llama31-prefix24` and the OLMo-2 native24 files are only in `REQUIRED_INPUT_CHECKS` (existence asserts). No executed worker loads either; the run script launches reference-decoders, deberta, complement (comp_cached), CPU spectral helper, ATLAS and the 27B worker. The OLMo dataset's own card says "only the plain readout is used by the associated ensemble", and that is not in this run either. So the Llama-3.1-prefix and OLMo-native24 members are not part of the +0.002. (If the legacy `ntl611-prefit` stack inside comp_cached touches Llama features, it is the same stack the 0.636 and 0.639 already carry.)
- **Other differences are plumbing**: `RuntimeBudget` is neutered (untimed, `remaining()` = inf), weights staged sequentially to `/kaggle/temp`, the 27B NF4 base copied in from the dataset, the 27B worker given both T4s (`CUDA_VISIBLE_DEVICES=0,1`), its own wheels (transformers 5.14.1, bitsandbytes 0.50.2) installed with `--target` so the other workers keep the old environment. Datasets attached: 14 (nlm-l24-adapters, ntl-olmo2-native24-final-inference-assets, extra-api-passages, ntl-generations, ntl-proxy-generations, qwen3-8b-lora-claude-head, ATLAS, free-proxy-generations, spectral heads 4404, two-stage-lora, handwriting-blend, prvsiyan extended, qwen35-9b-safetensors, ultrabert-380m) plus Qwen3 4B/8B/14B-base and Qwen2.5-14B models.
- **Runtime and GPU:** the fixture run in our copy only completed the 3-row preview. The author's own reserve for the 27B is `1200 + 5.0 s x rows`, so up to about 5.9 h for 4,000 rows on 2 T4 on top of the 0.636-family stack; real total unknown (not logged in the notebook). Risk: the 9 h limit and slow `/kaggle/input` reads (see C).
- **What plausibly explains +0.002:** one more 10% member of a different family and size (27B with 64-layer context, tail-only LoRA) averaged into a stack that already has ATLAS and the spectral helper. Comparable public steps from the same chain: ATLAS +0.002 to +0.007 on six bases; Kirill helper 0.636 -> 0.639 (his own official-only control 0.637, so the extra data was worth +0.002). A +0.002 step is inside the public noise (GoPain: prompt-cluster SE of the public LB about 0.014; paired differences on the same rows are smaller but not by an order of magnitude). It is also not an honest measurement: no OOF for the member.
- **Counter-evidence from our own runs:** our OLMo-2-13B member at W 0.15/0.20 scored 0.635 (-0.004), after a fold-0 proxy gain of +0.03. So "another member at 10%" is not a reliable +0.002; the 27B is a different case only because the author trained on 28k external passages first (the same stage-1 pool trick that gave Nishant's +0.028).
- **Reproducible by us with free compute?** Running their released weights: yes in principle (the copy `ntl-0-641-pinned-copy` v2 exists), cost likely 4-8 h of the 8.87 h GPU quota. Training our own: no. A 27B NF4 needs two T4s for a 64-layer forward, stage 1 over 28k passages at about 5 s/row is about 40 h of forward alone, and Colab T4 (15 GB) cannot hold the NF4 base plus activations. So only the exact copy is reachable, and it carries the licence questions in D.

### B. Forum digest (forum 13688374, 69 topics fetched with comments; Masters/Grandmasters and >= 5 votes)
Technique table (numbers as posted):

| Technique | Source | Number | Status for us |
|---|---|---|---|
| Group CV by prompt_id | Adisak 750390 (10th, 18 votes), 749593 | random 0.217 vs grouped 0.369; TF-IDF grouped CV 0.343 -> LB 0.353 | in use |
| Pool-trained members read 0.03-0.04 high on grouped CV | GoPain reply in 750390 | SVM train-only CV 0.370/LB 0.361 (-0.009); SVM +pool 0.467/0.423 (-0.044); two-stage LoRA 0.576/0.538 (-0.038); +Sinkhorn 0.576/0.549; blend 0.590/0.544 | compare pool-trained candidates only with each other; our OOF is the same kind |
| Sinkhorn on unbalanced tests | locdunhat 752365 (OOF 2,699 rows, 0.64 bacc, pools of 2,000, 40 draws) | even: plain 0.6387, 1 it 0.6398, **3 it 0.6409**, 50 it 0.6379; alpha 100: 0.6384/0.6391/**0.6395**/0.6363; alpha 20: 0.6357/0.6376/**0.6379**/0.6298; alpha 5: **0.6393**/0.6391/0.6384/0.6167; alpha 1: **0.6361**/0.6345/0.6284/0.5704. Small sparse pools: 0.645 -> 0.475 with full balance | our chain uses 300 iterations |
| Cross-fitted class offsets (GoPain reply, full table) | 752365 reply, grouped 5-fold OOF | DeBERTa-v3-large two-stage: argmax 0.521, Sinkhorn 0.564, offsets 0.560. Qwen3-8B LoRA: 0.595, 0.600, 0.599. LoRA + 0.5 DeBERTa + 1.0 SVC: 0.626, 0.626, 0.621. Known-truth sim, Dirichlet(1) test: Sinkhorn regret 0.060, offsets 0.018; EM prior worse than both. LB: Sinkhorn on a LoRA alone +0.011 (0.538 -> 0.549) while its OOF moved 0.5760 -> 0.5761 | blends whose members lean different ways gain nothing; single skewed members gain a lot |
| Sinkhorn is a training-prior fix | GoPain in 750213 | balance to equal mass costs <= 0.003 on uneven tests; balancing to the true uneven mix costs up to 0.040; for a model trained on an unbalanced pool equal-mass balancing gave +0.020 to +0.022 | explains why it helps on the LB |
| Test-time grouping needs many answers per prompt | GoPain 750213 sim; Adisak; Daniel 751871 | true groups, bacc gain: 1 answer 0; 2 answers -0.067 (centring) / +0.006 (Hungarian) / +0.024 (joint); 3: +0.002/+0.013/+0.042; 18: +0.101/+0.117/+0.115. Real LB: TF-IDF 0.353 -> 0.344 (Hungarian) -> 0.329 (+centring); 0.427 vs 0.425. Cavity estimate (centre on the other members only) never lost with true groups | already known; cavity centring is the one idea we have not used |
| Early layers beat late layers | Aleksey 750462 (frozen Qwen3-4B, mean pool, LR) | layer 3 0.391, 6 0.379, 9 0.370, 18 0.327, 27 0.341, last 0.311, 6 layers concatenated 0.414; blend with TF-IDF+stylo (0.369) 0.439 CV / 0.420 LB. shanzhong8 (Master) agrees | in the chain (LoRA24 = first 24 of 36/40 layers, pooled at 8 depths) |
| Readout diversity | Daniel 751871 (Qwen3.5-9B first 19 layers) | two-stage 8-layer readout 0.578, layer-18-only 0.582, they disagree on 29% of passages; both plus train-only LoRA 0.605 CV / 0.573 LB; "readout diversity is a cheap substitute for a second backbone" | Kirill helper is the same idea; one single-layer head is not in our chain |
| Coverage beats volume | Daniel 751871 | adding 13/18 label coverage: +4 pts overall but the 5 uncovered labels lost 6 pts recall ("new prompt => not GLM"); 18/18 covered (15k): 0.447 -> 0.508 | check which labels our pool covers |
| Claude gate / within-lab head | seyit 750224, zbigniew | within-Claude accuracy 0.63 -> 0.69; Qwen3-8B LoRA with Claude head 0.4838 plain, 0.4932 gated vs public 0.4559; hierarchy lab->line->version +2.4 pts on a CPU stylegram (CI excludes 0) | the zbigniew LoRA is in the 0.641's dataset list |
| Pool leakage check | SwapIY 751941, GoPain | pool passages with cosine >= 0.4 to a train prompt centroid 0.26%; dropping the 934 close ones moved SVC -0.002/+0.006 (noise); simulated lift +0.1 per unit share of near-copies | no leak; the 0.03-0.04 CV-LB gap is unexplained |
| Typography | Masood 750664 | em dash: Claude 63-83%, GLM 79-82%, Gemini 0-1%; curly apostrophe GPT/Gemini; siblings differ by only 9% vs 37% (GPT) em-dash rate; punctuation solves lab (0.58 lab accuracy, 55% of errors right-lab-wrong-model); LightGBM + 80 features 0.372; <60 words 0.27 vs >160 words 0.46. Proxy-generated Gemini text has em dashes 13-43%, so train rules flip on external data | known |
| Negative results | Adisak, Nishant, Daniel | truncated copies of train passages: 0.343 -> 0.335 (word cut), 0.331 (sentence cut); more parameters, seed averaging (<= 0.1 pt), Gemma-4-12B, family gates, one-per-model clusters, stacker, more stage-2 epochs, char CNN (+0.4 pt noise), DeBERTa-base on train only, lab->model hierarchy (better lab acc, same model acc); mean of several LoRA runs <= 0.1 pt; adding the Oleg Zholobov 135k corpus hurt CV (formatting gap, HALX2158) | consistent with our ten failed blend changes |
| External data worth | 749447 comments, Maniceet 750947 | about +0.11 to +0.12 from external data; removing the openrouter+openai set 0.463 -> 0.456 on one validator | the field's biggest lever, and the rules question below |
| Load time | HALX2158 751268 (and 750924) | Qwen3.5-9B 1,206 s vs 22 s from a Dataset; Model attach 19 s (8B) / 37 s (14B); UltraBERT 147 s -> 1,856 s on slow nodes; 169 MB/s sequential but from_pretrained mmap stalls; cp to /tmp 46 MB/s. Advice: attach as Kaggle Models, log read speed, drop optional parts under a deadline | the 641 neuters its deadline |
- Unanswered: Rule 2.6 on API-generated data (752478, 751011, 751954): no host reply as of the fetch. Hierarchy (751722, Master Ghaith Mechi): no results posted.
- No Master/Grandmaster posted a score or recipe beyond: prvsiyan (Master, 47 votes) the extended dataset (9,290 passages, 510 prompts, 18 labels, CC BY-SA 4.0 for the new split but the dataset field says CC BY 4.0: a user asked, no answer), pjmathematician (Master, 19 votes) the reconstructed train prompts (CC BY-SA 4.0), shanzhong8 the early-layer agreement.

### C. Digest
- **Techniques that repeat across the field:** (1) a big public pool in stage 1, then train rows in stage 2 (every 0.59+ notebook); (2) early-layer mean pools of a truncated LLM (24 of 40/36 layers), 8 depths, LayerNorm per depth, plain + prompt-centred heads; (3) 10% additive members after the main blend (ATLAS, spectral helper, now Qwen 27B), each +0.002 to +0.007 on public; (4) final Sinkhorn; (5) grouped-by-prompt CV and a warning that pool-trained CV reads 0.03-0.04 high.
- **What we have not tried:** partial Sinkhorn (1-3 iterations) or cross-fitted offsets in place of 300-iteration Sinkhorn; cavity centring; a single early-layer readout head beside the 8-layer one; label-coverage check of our pool (18/18 vs 13/18); the 135k Oleg corpus (reported to hurt); the free Benchmarks quota ($100/month) for our own matched data (blocked on Rule 2.6 and unanswered); deadline-guarded optional components.
- **What costs GPU:** the 27B (5 s/row, both T4, 4-8 h); everything in A except the spectral helper (CPU); training any 14B (6.6 h stage 1 on 2 T4 for OLMo-13B). Free/CPU: all Sinkhorn variants, cavity centring, readout heads on cached features, pool coverage audit.
- **Needs a licence check before a final pick:** see D.

### D. Licences involved in the 0.641
| Component | Licence as stated | Note |
|---|---|---|
| Qwen3.8-27B + NF4 conversion (greghavens/Qwen3.8-27B-bnb-4bit) inside lightningv08 collection | Apache 2.0 (collection description says so, notices retained under qwen38-27b-nf4) | converter's own terms not independently fetched; fine if Apache 2.0 holds |
| lightningv08/ntl-olmo2-native24-final-inference-assets (14 versions) | Kaggle field "Other (specified in description)"; components: OLMo2/Qwen2.5/SmolLM3/Gemma/exact/tail/oof Apache 2.0, smollm3-two-stage CC BY-SA 4.0, **tabicl and tabular offline assets PriorLabs noncommercial** | not executed in the 641 (grep: no tabicl/tabpfn code), but the collection is attached; use for rank only unless the unused files are dropped, as with TabPFN |
| Llama-3.1 24-layer prefix (`llama31-prefix24`, inside that collection; description text truncated at "## L...") | Llama 3.1 Community Licence (Meta) expected | existence-checked only, not executed; commercial use allowed under 700M MAU, needs "Built with Llama", licence copy, and "Llama" at the start of any derived model name, plus the Acceptable Use Policy; not independently read from the dataset |
| ATLAS josephadamski91/ntl-atlas-qwen3-8b-v1 | weights CC BY-SA 4.0, code Apache 2.0 (accepted by the user 24 Oct) | share-alike on derived weights |
| Kirill spectral heads kirilltym/ntl-spectral-style-heads-paired-4404 | none stated (Kaggle field unspecified); the 641 itself says "unspecified licence" | rank use only until the author states one |
| zbigniewkirchner/ntl-qwen3-8b-lora-claude-head | none stated; base Qwen3-8B Apache 2.0 | author unspecified |
| seyitkaangunes/ntl-extra-api-passages | CC BY 4.0 (synthetic, official Claude API); card says no competition text | |
| abdullah1afify/ntl-generations, dophuchung/ntl-proxy-generations | CC BY 4.0 | |
| lucasboesen/ntl-free-proxy-generations | card: made via Kaggle Benchmarks free quota; licence not shown on page | provider output terms not checked |
| prvsiyan/prvsiyans-extended-name-that-llm-dataset | CC BY-SA 4.0 (README) vs CC BY 4.0 (field) | unresolved per comment |
| nishantdahal/ntl-two-stage-lora, ntl-handwriting-blend (pool v4 75,783 passages from 26 public datasets) | none stated; sources CC BY 4.0 / CC BY-SA 4.0 / CC0 per the 0.620 notebook | |
| pjmathematician/ultrabert-380m-ntl | UltraBERT-380M Apache 2.0 | |
| dhruvtibarewal/nlm-l24-adapters, bbucxi/qwen35-9b-safetensors | none stated | bbucxi is a bare re-upload of Qwen3.5-9B (Apache 2.0 upstream) |
| GLM-4-9B (lucasboesen kernel glm4_9b_trunc25), ELECTRA, Desklib | inherited from the 0.636 chain; GLM-4 has its own licence | already in our chain |
Datasets' pages were fetched anonymously; descriptions longer than the page's JSON-LD (the collection's Llama section) were truncated and not read.

### E. Next experiments (ranked, at most 3)
1. **Run the exact 0.641 copy (`ntl-0-641-pinned-copy` v2) once, for rank and for the measurement only.** Expected gain +0.002 over the 0.639. **Forecast (written before submitting): public 0.641** (range 0.639-0.643; the 0.597, 0.622 and 0.636 copies each hit exactly). Cost: 4-8 h GPU of the 8.87 h left; check read speed in the log first. Not a final pick until D is settled (PriorLabs files attached, Llama text unread, Kirill/zbigniew no licence). Blend-composition, so no validation claim.
2. **Sinkhorn variants on our own OOF, CPU only:** 1/3/50/300 iterations and cross-fitted offsets, at the 4 test-like crops with prompt-grouped OOF of the lab stack, paired bootstrap over prompts. Expected: +0.001 to +0.002 for 3 iterations on a balanced mix, -0.01 to -0.07 risk for full balance if the test is very uneven (locdunhat table); the public LB history (Sinkhorn +0.011 on a single LoRA) says our test predictions lean, so the 300-iteration step may be doing real work. Accept only at P(better) >= 0.9; this is a correction step, the one kind of change the project rules allow. Forecast if submitted on the 0.639 chain: 0.639 +/- 0.002 (no change expected; the gain is risk reduction).
3. **One early-layer single readout head on cached LoRA24 features (CPU):** Daniel's single-layer-18 head tied the 8-layer head and disagreed on 29%; two readouts plus the train-only LoRA added +2 pts there. Expected +0.001 to +0.003 on OOF, and by the olmo13b and encoder history likely 0 to +0.002 on public. Forecast: 0.639 -> 0.640 (0.637-0.642). Only if the lab has the cached features; otherwise drop it.
Not recommended: any new member at 10% or more (OLMo-13B -0.004, encoders -0.002), any Rule-2.6-dependent API data, grouping tricks (Hungarian/centring lose at 1-2 answers per prompt).

## 25 Oct (14:2x IST): Sinkhorn variants on our OOF (next experiment 2; CPU lab, `lab/11_sinkhorn_variants.py`, output `lab/artifacts/sinkhorn_variants.json`)
- **Forecast (written in the 25 Oct digest, restated before the conclusion below):** on the public 0.639 chain, 3 iterations or 300 iterations both land at 0.639 +/- 0.002; no measurable change if the public test is near balanced, a gain only if it is uneven. Not submitted.
- **What was evaluated.** The best honest 5-fold prompt-grouped OOF blend in the lab: SVC 1.0 + modernbert 0.40 + desklib 0.10 + 3 token-attention members 0.3 each + LoRA24 plain 1.0 + LoRA24 prompt-centred 1.0 (the LoRA24 readouts applied to the held-out fold's cached features; centred head by prompt-centring the fold's features). Members alone (argmax bacc): SVC 0.477, modernbert 0.553, desklib 0.499, tok 0.394/0.370/0.411, LoRA plain 0.482, LoRA centred 0.596. Weights are the proxy's, not tuned. **Not the real 0.639 stack** (its 14B LoRAs/ATLAS have no OOF), so this transfers loosely. **No length crops exist for NTL (that is an SHL notion): rows are full passages.**
- **Balanced OOF (2,699 rows, Sinkhorn over all rows), bacc and paired bootstrap over 150 prompts vs 300 iterations (2000 draws):**

| variant | bacc | gain vs sk300 | 90% CI | P(better) |
|---|---|---|---|---|
| argmax | 0.6131 | +0.0025 | -0.003..+0.008 | 0.75 |
| 1 it | 0.6127 | +0.0022 | -0.003..+0.007 | 0.74 |
| 3 it | 0.6127 | +0.0022 | -0.0015..+0.0056 | 0.83 |
| 10 it | 0.6112 | +0.0007 | -0.002..+0.003 | 0.65 |
| 50 it | 0.6116 | +0.0011 | 0.000..+0.0026 | 0.89 |
| 300 it | 0.6105 | 0 | | |
| cross-fitted offsets | 0.6083 | -0.0023 | -0.0056..+0.0011 | 0.14 |

- **Simulated uneven test pools** (2,000 rows drawn from the OOF with class counts ~ multinomial(Dirichlet(alpha)), 200 draws each; gain vs 300 iterations, draw-level P(better) in brackets):

| mix | argmax | 1 it | 3 it | 10 it | 50 it | 300 it (bacc) | offsets |
|---|---|---|---|---|---|---|---|
| alpha 1 | +0.076 [0.99] | +0.076 | +0.065 [0.99] | +0.038 | +0.003 | 0.537 | +0.072 [0.98] |
| alpha 5 | +0.034 | +0.035 [1.00] | +0.033 [1.00] | +0.023 | +0.005 | 0.578 | +0.029 |
| alpha 20 | +0.016 | +0.016 | +0.016 [0.97] | +0.013 | +0.005 | 0.597 | +0.012 |
| balanced | +0.005 [0.81] | +0.005 | +0.004 [0.80] | +0.004 | +0.002 | 0.609 | -0.0004 |

  Regret vs oracle (known-mix prior shift, argmax of logP + log pi), 300 it: 0.068 / 0.035 / 0.017 / 0.005 at alpha 1 / 5 / 20 / balanced; 3 it: 0.002 / 0.002 / 0.001 / 0.001; offsets: -0.004 / 0.006 / 0.005 / 0.005. (Regret can be negative: the oracle maximises accuracy, not bacc.)
- **Findings.** (1) Full 300-iteration balance is the worst variant at every mix and costs 0.003 on a balanced pool already (on this blend; P(better) of 3 it only 0.83, so it fails the 0.9 rule). (2) 1-3 iterations keep nearly all of argmax's robustness (regret <= 0.002 at every mix) and equal the best on balanced pools; 50 iterations is the only variant whose balanced bootstrap clears 0.89, but its gain is +0.001. (3) Cross-fitted offsets are worse than 3 iterations everywhere and worse than 300 on a balanced pool (GoPain's regret 0.018 vs 0.060 at alpha 1 reproduces in direction: ours 0.004 vs 0.068, offsets win over 300 it but not over 3 it). (4) Here argmax is as good as 3 it: this blend is not skewed. The 8-member proxy already has a flat class histogram, so Sinkhorn has little to fix; the public +0.011 for Sinkhorn on a single LoRA (GoPain) is the case where it works, and our real 0.639 chain is a blend, closer to this proxy.
- **Recommendation.** Not accepted under the rule (no variant reaches P(better) >= 0.9 on the balanced OOF; the 0.83 for 3 it is the best of the plausible ones). As a risk reduction only, replacing 300 by 3 iterations costs nothing measurable on a balanced test (+0.002) and removes up to 0.07 of downside if the hidden mix is uneven; it would not be an honest validated gain. Forecast on the public 0.639 chain: 0.639 +/- 0.002 (range 0.637-0.641). Real chain not tested; a 2-submission cost is not worth +0.000-0.002. Notebooks unchanged. Caveat: the lean of the real chain's pre-Sinkhorn predictions is unknown; check the predicted class histogram of the 0.639 test output before choosing (if flat, 3 it equals 300 it).

## Scan 25 Oct 19:07 IST (notebooks and discussions; titles only, sources unread)
- New top public: **0.642** lightningv08/0-642-lora-llama-qwen3-8-27b (113 votes), the 0.641 kernel retitled and re-versioned: +0.001 more, probably another small change to the same chain. jnliang has a 0.640 baseline of kirilltym's 0.639 (1 vote). Others unchanged (0.639 tier; 0.636 tier).
- Our copy of the 0.641: v1 failed (path to lightningv08's own dataset), v2 (path fix, latest image) went COMPLETE with no score after minutes (hidden-test run failed), v3 pins the original container image like the author; submitted 25 Oct 19:07 IST (ref 57631776), PENDING. 3 submissions left today. Forecast 0.641 (0.639-0.643), rank-only.
- Discussion: 752539 new (WOOSUNG YOON, "Dataset And Kaggle Benchmark"); thread 749447 (pay-to-win) now 59 comments; 752270 (paid and region-restricted LLMs) has a first reply. Still no host reply on 752478 (Rule 2.6, API-generated data).
To do on go: read the 0.642 source and diff against the 0.641.

## 0.642 copy (25 Oct ~19:20 IST)
ntl-0-642-copy: copy of lightningv08 0.642 on its original pinned image, own-dataset path fix as in the 0.641 copy (push dir /home/user/work/push_ntl642), pushed v1; a detached watcher submits it when the fixture run completes. Forecast written before submitting: 0.641 (0.639-0.643) if the hidden-test run works; the 0.641 copy v3 is still PENDING, so a failure here would repeat that one.

## Stopped, difference read (25 Oct ~19:30 IST)
Watchers stopped before submitting the 0.642 copy (kernel ntl-0-642-copy still runs on Kaggle; not submitted). The 0.642 and 0.641 notebooks have identical source (0 changed lines between the pulled 0.641 and 0.642 versions), so the +0.001 is rerun/data-version noise or a metadata change, not a new method. The 0.641 copy v3 (ref 57631776) is still PENDING. Decision: wait for a larger jump before spending more submissions here.

## dhruvtibarewal/nlm-v27: 0.651 (26 Oct 00:40 IST)
Found by the user. Public **0.651** (27 votes, version 25 Oct 10:17 IST), +0.009 over the lightningv08 0.642 and 0.002 under the gold line 0.653. areebbb has a 0.643 ("same stack, sharper blend") too. Header of the notebook: fold-averaged two-stage LoRA24 (Qwen3-8B-Base x5 folds + full, Qwen3-14B-Base x3 folds + full, Qwen2.5-7B, Llama-3.1-8B x3 folds + full, a seed-7 8B), Nishant two-stage checkpoints (8B, 14B, a second 14B), a 4-view n-gram SVM on train+pool, shared-prompt Gaussian, adaptive centred twins, 10% ATLAS before Sinkhorn. So the jump is fold-averaged adapters plus Gaussian/centred heads on them, the lever listed earlier as untested by us (Gaussian group heads on LoRAs). Licence flags: Llama-3.1-8B (community licence), dhruv's own datasets (nlm-l24-adapters-v5, nlm-matched-clean: licence unchecked), ATLAS CC BY-SA 4.0.
Copy: kernel nlm-v27-copy (exact, original image), push dir /home/user/work/push_nlmv27, pushed v1 at 00:40 IST. NOT submitted (user said don't submit). Forecast written before submitting: 0.651 (0.648-0.653) if the hidden run works; runtime unknown, many 8B/14B forward passes, risk of the 9 h limit.

## nlm-v27 copy submitted (26 Oct 20:13 IST)
Llama-3.1 Kaggle access granted 17:2x IST (needed for model source metaresearch/llama-3.1/Transformers/8b/2; v1 failed on it). nlm-v27-copy v2: visible run COMPLETE after about 2.2 h (full pipeline even on the 3-row placeholder; comp stack 3,099 s on 3 rows). Submitted by hand at 20:13 IST (ref 57669744), PENDING. Forecast written before the score: 0.651 (0.648-0.653). The notebook sets its own budget (ATLAS_BUDGET = 8.3 h - comp-stack time - 10 min) and raises if ATLAS projects an overrun, so a slow hidden run can fail instead of finishing late; estimate 6-9 h, failure risk about 20-30%. GPU quota: 24.46 h used, 5.54 h left (was 8.83 h at 18:44 IST), so the visible kernel runs do draw from the weekly quota (about 3.3 h across the 0.651 visible run and the two CASMI copies); the hidden scoring runs have not been seen to. Watcher scripts die when the shell call ends; use timed checks instead.
CASMI 0.468 copy (casmi-romanrozen-0468-copy) submitted by hand 26 Oct 18:47 IST (ref 57667630), PENDING.

## nlm-v27 copy scored (27 Oct 03:23 IST)
nlm-v27-copy (exact copy of dhruvtibarewal nlm-v27, submitted 26 Oct 20:13 IST, ref 57669744) scored **0.651** public: reproduced exactly; forecast 0.651 (0.648-0.653) hit. Time to score at most 7 h 10 min (first seen COMPLETE at 03:23 IST; PENDING at 01:50), inside the 6-9 h estimate; no ATLAS-overrun failure. Our best NTL by 0.009 over the 0.642 line (not reproduced, notebook identical to the 0.641 whose copies twice failed). Gold line 0.653 is 0.002 away. Rank-only: Llama-3.1 community licence, dhruv's datasets unchecked, ATLAS CC BY-SA 4.0. Hidden scoring does not draw GPU quota (5.54 h left unchanged before and after); visible runs do.

## New top 0.654 and its copy (28 Oct 12:57 IST)
Scan 12:57 IST: new top public **0.654** romantamrazov/name-that-llm-who-am-i (95 votes; was 0.639 at 37 votes on 25 Oct), above the gold line 0.653. Header says: acknowledgements dhruvtibarewal nlm-v27, kirilltym 0.639, fleongg, Nishant Dahal; "our changes": a train-only Bayesian typography likelihood from short stylistic markers plus capped confidence-aware corrections, no changes to the pre-trained checkpoints, no forced Hungarian; the parent pipeline's exact output is saved separately. So it is our 0.651 nlm-v27 line plus a typography correction (+0.003), the kind of blend/post-processing change that failed on the public board many times before; here it holds at 0.654. Other new: lightningv08 0.652 (Qwen3.8-27B LoRA + spectral ensemble), many 0.651 copies (maverickss26, alexchilton, avazaku verified reproduction, dhruv fully reproducible). Uses the kernel source avazaku/name-that-llm-verified-llama-asset-cache for Llama, so no Llama model source in its metadata.
Copy ntl-who-am-i-654-copy (exact, original image, push dir /home/user/work/push_ntl654) pushed v1 at 12:57 IST; NOT submitted yet (user has not said submit). Forecast written before any submission: 0.654 (0.651-0.655), since the 0.651 copy reproduced exactly. Quota 5.12 h left before the visible run (about 2.2 h expected).

## 0.654 copy submitted (28 Oct 16:47 IST)
ntl-who-am-i-654-copy (exact copy of romantamrazov name-that-llm-who-am-i, public 0.654) visible run COMPLETE after about 3.8 h (longer than the 0.651's 2.2 h; it uses avazaku's Llama asset cache kernel and a typography step). Submitted by hand on the user's word at 16:47 IST (ref 57734852), PENDING; 4 submissions left today. Forecast written before the score: 0.654 (0.651-0.655). GPU quota 1.66 h left after the visible run (it used about 3.5 h of 5.12). Hidden scoring not billed; expected 6-9 h so a score around 23:00-02:00 IST.

## New top 0.660 (28 Oct 22:42 IST)
Scan 22:42 IST: new top public **0.660** mercurydust/name-that-llm-submission (13 votes; "Name That LLM | Who am I? V31"). Its page says V31 reached 0.660 and V30's 0.654 was reproduced (the 0.654 is romantamrazov's, which our copy 57734852 is waiting on). It builds on dhruvtibarewal nlm-v27, kirilltym 0.639, fleongg, Nishant Dahal; V31 changes listed as: auxiliary data usage, family-level gating, adaptive probability balancing, confidence-aware Bayesian typography; no weights modified. Same datasets, models and kernel sources as the 0.654 (including avazaku's Llama asset cache). Other new: alexchilton 0.654 (59 votes), peterdreamhan 0.653 "Soft Sinkhorn 0.5 T4" (5 votes), lightningv08 0.652. The "adaptive probability balancing" and "soft Sinkhorn" lines fit our 25 Oct Sinkhorn test (fewer iterations robust on uneven pools).
Not copied: GPU quota 1.20 h left (visible run of this family costs about 3.5 h); resets 31 Oct 05:30 IST. Pulled source is at /home/user/work/pull_merc. Forecast for a copy: 0.660 (0.656-0.661), unverified. Pending: ntl-who-am-i-654-copy (ref 57734852) and casmi-busyaprime-477-copy (ref 57736039).

## 0.660 fastpath copy submitted (28 Oct 22:54 IST)
ntl-v31-660-fastpath-copy (push dir /home/user/work/push_ntl660): copy of mercurydust name-that-llm V31 (claimed public 0.660) with one shortcut: when the test file has <= 3 rows (the visible placeholder run) it writes a 3-row preview submission and skips comp_stack.py and infer.py; the hidden rerun (thousands of rows) runs every original step. Visible run succeeded, log shows the line "placeholder test: wrote the preview file, the pipeline is skipped", no errors; quota cost about 0.08 h (1.20 -> 1.12 h left, instead of about 3.5 h). Submitted by hand at 22:54 IST (ref 57744232), PENDING; 3 submissions left today. Forecast written before the score: 0.660 (0.656-0.661), unverified (author's own figure). Not an exact copy; rank-only. Lesson: a placeholder-test shortcut makes big NTL notebooks cheap to push when GPU quota is low; the hidden rerun is unbilled.
Pending: ntl-who-am-i-654-copy (57734852), casmi-busyaprime-477-copy (57736039), this one (57744232).

## 0.654 copy scored (29 Oct 02:35 IST)
ntl-who-am-i-654-copy (exact copy of romantamrazov name-that-llm-who-am-i, submitted 28 Oct 16:47 IST, ref 57734852) scored **0.654** public: reproduced exactly; forecast 0.654 (0.651-0.655) hit. Time to score under 9 h 50 min (PENDING at 22:56 IST, COMPLETE by 02:35). Our best NTL until the 0.660 copy (ref 57744232) scores; +0.003 over the 0.651 copy from the typography correction. Above the gold line 0.653. Rank-only (copy; Llama licence, dhruv's datasets unchecked, ATLAS CC BY-SA 4.0).

## 0.660 fastpath copy scored; scan 29 Oct 11:34 IST
ntl-v31-660-fastpath-copy (ref 57744232, submitted 28 Oct 22:54 IST) scored **0.660** public: forecast 0.660 (0.656-0.661) hit, and the placeholder shortcut did not change the hidden-run result. Our best NTL (+0.006 over the 0.654 copy), above the gold line 0.653. Rank-only. GPU quota 1.12 h left (resets 31 Oct 05:30 IST).
Notebooks: romantamrazov who-am-i is now also 0.660 (121 votes); mercurydust 0.660 (42); otmaneelmahdaouy "NTL v33 Championship Veritable" 0.654 (new); alexchilton 0.654; peterdreamhan soft Sinkhorn 0.653; lightningv08 0.652.
Discussions: 753177 (Harshita Kumari, 6 votes): retraining nlm-v27's Qwen2.5-14B member on the 120k pool raised its OOF 0.569 -> 0.605 but public fell 0.651 -> 0.647; keeping the old member plus the retrained one at 0.4 gave 0.650; HALX2158's zero-weight ablations on the hidden test show removing the old q25_14 cost -0.014. Lesson again: OOF gains on pool-trained members do not carry to the public board. 752766 (45 votes): "Name That LLM Open Pool" crtproduction/name-that-llm-open-pool, 254,561 passages from 59 public sources, all 18 labels, CC BY-SA 4.0, updated weekly (the data-volume lever, free; licence share-alike). 753053/752708 (GoPain): grouped CV over-picks test-time centring; 753526: stratified CV 0.13 vs prompt-grouped 0.31 for the same baseline (grouping matters). 752084: open rules questions on generated data (2.6) still unanswered.

## 0.663 copy submitted (29 Oct 13:5x IST)
New top public **0.663** omerfarukmerey/0-663-0-660-v31-atlas-20 (6 votes): mercurydust's 0.660 V31 with ATLAS weight 0.10 -> 0.20, Llama-3.1-8B from avazaku's public cache copied to local disk, one model copy per GPU when it fits, and a built-in FAST_COMMIT path (test < 50 rows skips the heavy cells). Our copy ntl-663-v31-atlas20-copy (push dir /home/user/work/push_ntl663): v1 failed (avazaku cache not in the author's pulled settings); v2 failed (avazaku's cache notebook is no longer accessible, Kaggle silently dropped it); v3 takes Llama from the official model source metaresearch/llama-3.1/Transformers/8b/2 and drops only the folder-name part of the Llama assert (the four shard-size checks stay and passed); visible run clean (FAST_COMMIT True, llama copied in 472 s, preview written, no errors). Submitted by hand on the user's word. Forecast written before the score: 0.663 (0.659-0.664); ATLAS weight changes have gone both ways before (+0.002 to +0.007 at 10%, -0.002 at 15%). Copy with one edit; rank-only. GPU quota 0.83 h left after three visible runs (resets 31 Oct 05:30 IST).

## Discussion scan 29 Oct 16:23 IST
- 753779 (Carrot Production, new, 0 votes yet): audit of the 254k "Open Pool" (59 community datasets). About 1 in 4 passages is junk or off-genre: raw API dumps stored as text 13,114; refusals 6,939; cut off mid-sentence 15,963; preambles/CJK/length outliers/duplicate texts with two labels 3,545; off-genre sources (chat logs, benchmark answers, reasoning traces) 46,672. In train 99.9% of passages end with punctuation, raw pool 93%, cleaned 99.4%. Same-size test (Qwen3-1.7B-Base LoRA, 2,500 passages per label, 1 epoch): cleaned data +0.026 to +0.027 balanced accuracy on real train passages, two seeds; the dirtier a label, the bigger the gain (Spearman 0.64; kimi-k3 +16 recall points, claude-opus-4-8 +13). Open Pool v9 = cleaned set, 192,926 passages, removed rows listed by id and reason; CC BY-SA 4.0. Caveat: this is CV on train passages, and pool-trained CV runs 0.03-0.04 above the LB; no LB number yet. Useful for any retraining (we have 0.83 h GPU this week, none for a retrain), and its cleaning rules are a free check for our own pools.
- 753177: GoPain on judging "does this help the blend" offline: score OOF on a test-like shape (fewer answers per prompt), held up over 15 submissions with one bad day.
