---
tags: [kaggle, name-that-llm, research]
date: 2026-10-24
---
# Name That LLM: the public chain, version by version

This note does for NTL what dariushafshar's CASMI notebook ["The Public Chain, Version by Version"](https://www.kaggle.com/code/dariushafshar/casmi26-the-public-chain-version-by-version)
does for CASMI: it records which change bought which public score. It is research only. I made no submissions and no
kernel pushes. I read all notebook code as text and never ran it.

**Read:** 24 Oct 2026, about 10:15 UTC (leaderboard snapshot 10:16 UTC).

**Coverage:**
- 55 public notebooks with 134 scored versions: every public NTL notebook with a version at 0.600 or above, plus
  their ancestors (lucasboesen 0.541, prvsiyan 0.569, nishantdahal 0.592 and others), with every scored version of
  each, including early low ones (lowest 0.127).
- 82 of the 134 steps carry a change line I wrote by hand from the code diff. The rest carry a mechanical
  description.

**Leaderboard right now:**
- Public top: **0.700** (Hod Carriers).
- Gold line (rank 15): 0.655. Silver line (rank 146): 0.636. Bronze line (rank 293): 0.624.
- We are **rank 148 of 2,932 at 0.636**, tied with the silver line but below it on tie order.
- 152 teams are at 0.636 or above, and 61 at 0.639 or above.

**News since this morning:**
- The public frontier is **0.639**, not 0.636: [kirilltym/0-639-teach-more-keep-the-signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature)
  (01:06 UTC).
- An independent refactor of it, [maverickss26/kaggle-name-the-llm-v2 v4](https://www.kaggle.com/code/maverickss26/kaggle-name-the-llm-v2)
  (06:36 UTC), also scored 0.639.

**Data:** all CSVs are in `name-that-llm/lab/chain/`:

| file | one row is |
|---|---|
| `versions.csv` | a scored version: date, score, parent, step, change type, change line, datasets, URL |
| `notebooks.csv` | a notebook: author, first scored version, best, Kaggle fork parent, parent and how known |
| `lineage_edges.csv` | a parent → child link (61), with how it is known: fork 11, credit 11, copy 8, code 31 |
| `change_types.csv` | one measured step of a change type, with base, child, delta and whether it was confounded |
| `inputs_licences.csv` | an input dataset or model: visibility, licence, who uses it |

**How it was read** (the scripts are in the scratchpad, `ntlchain_scripts/`):
- **Notebook list:** `kernels.KernelsService/ListKernels` with `readMask: kernels.forkParentInfo`, which also gives
  Kaggle's fork record.
- **Version list:** `ListKernelVersions`.
- **Per-version public score:** `LegacyKernelsService/GetKernelViewModel` with `versionNumber`, from
  `submission.scoreFormatted`. This is the number the version page shows. 62 complete versions have no submission;
  they are quick saves, rechecked twice.
- **Source:** `GetKernelSessionSource`.
- **Lineage and diffs:**
  - Embedded scripts (`Path(...).write_text('...')`, module tables) were extracted with `ast` and never executed.
  - Lineage comes from normalized-line `SequenceMatcher` on comment-free code, plus fork records and markdown credits.

## 1. What a step is worth

- **Public share.** The hidden rerun has about 4,000 passages (Kaggle data page, quoted in the 0.559 notebook). The
  public LB is said to use about 40% of them (karthikeya18,
  [750213](https://www.kaggle.com/competitions/name-that-llm/discussion/750213)), so roughly 1,600 passages, about 89
  per class.
- **One passage.** One passage fixed in one class moves balanced accuracy by about 1/(18 × 89) ≈ **0.0006**. A
  displayed step of 0.001 is one or two passages.
- **Re-runs.** 14 exact or near-exact copies are in the table (12 public, plus our 0.622 and 0.636 copies today):
  - 11 land on exactly the same displayed score.
  - 3 land 0.001 higher (two copies of kirilltym's 0.600 scored 0.601, and saddamaditya's near copy of kiza's 0.603
    scored 0.604).
  - **So the re-run spread is at most 0.001**, and anything up to ±0.001 is noise.

## 2. The frontier over time

The versions that raised the best public score:

| run (UTC) | score | gain | notebook | what it did |
|---|---|---|---|---|
| 15 Oct 18:47 | 0.522 | | ahmedberatozer/name-that-llm-d | frozen-feature log-prob blend (root of the kirilltym line) |
| 16 Oct 05:47 | 0.541 | +0.016 | [lucasboesen/lora-x-adaptive-centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) v1 | adaptive prompt centring and centred twin heads, plus a LoRA readout |
| 16 Oct 14:43–19:46 | 0.547 → 0.549 | +0.008 | lucasboesen v2–v3 | own second LoRA seed and token-attention heads |
| 17–18 Oct | 0.550 → 0.557 | +0.008 | kirilltym v12, lightningv08 v5/v8 | cohesion centring; TabPFN + centred stylegram; SmolLM3 |
| 18 Oct 14:23 | 0.563 | +0.006 | [vnton37nguyn/ntl-lora-instyle-0-563](https://www.kaggle.com/code/vnton37nguyn/ntl-lora-instyle-0-563) | in-style external passages and an in-style Qwen2.5-7B LoRA |
| 18 Oct 18:17 | 0.569 | +0.006 | [prvsiyan/the-prompt-is-not-the-author](https://www.kaggle.com/code/prvsiyan/the-prompt-is-not-the-author) v1 | shared-prompt Gaussian group heads |
| 19 Oct 01:50 | 0.592 | +0.023 | [nishantdahal/two-stage-lora-on-the-public-pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool) v2 | two-stage LoRA (stage 1 on 35k public passages, stage 2 on train), Qwen3-8B + 14B |
| 19 Oct 09:59 | 0.596 | +0.004 | josephadamski91/two-stage-lora-atlas-and-early-layers | 0.592 + ATLAS + a Qwen3-4B early-layer classifier |
| 19 Oct 11:19 | 0.597 | +0.001 | alexchilton/ntl-0592-plus-0569 | the 0.569 blend mixed at 0.5 into the 0.592 |
| 19 Oct 14:24 | 0.600 | +0.003 | [kirilltym/0-600-teach-the-model-remove-the-prompt](https://www.kaggle.com/code/kirilltym/0-600-teach-the-model-remove-the-prompt) v20 | two-stage 0.85 + Gaussian stack 0.15 (v19, 0.594), then a Gaussian head on the 8B LoRA (+0.006) |
| 19 Oct 17:14 | 0.604 | +0.004 | [sangam21/two-stage-lora-gaussian-full-hungarian-0-604](https://www.kaggle.com/code/sangam21/two-stage-lora-gaussian-full-hungarian-0-604) | a Gaussian head on the 14B too, plus a spaced em-dash guard and Hungarian assignment in clusters of 12–18 |
| 19 Oct 22:20 | 0.607 | +0.003 | [josephadamski91/teach-the-model-10-atlas](https://www.kaggle.com/code/josephadamski91/teach-the-model-10-atlas) | 0.600 + 10% ATLAS (+0.007 on its own base) |
| 20 Oct 19:37 | 0.611 | +0.004 | [dhruvtibarewal/nlm-v21](https://www.kaggle.com/code/dhruvtibarewal/nlm-v21) | own fold LoRA24s (Qwen3-8B/14B, Llama-3.1-8B) + nishant checkpoints + **comp stack 0.15 + ATLAS 10% on margin < 0.3** |
| 23 Oct 02:45 | 0.620 | +0.009 | [nishantdahal/two-stage-lora-part-two](https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two) | pool 35k → 84k, Qwen2.5-14B + DeBERTa-v3-large, EOS-id fix |
| 23 Oct 05:31 | **0.636** | +0.016 | [fleongg/ntl-part-two-complement](https://www.kaggle.com/code/fleongg/ntl-part-two-complement) | the nlm-v21 recipe (comp 0.15 + ATLAS 10%) applied on top of the 0.620 |
| 24 Oct 01:06 | **0.639** | +0.003 | [kirilltym/0-639-teach-more-keep-the-signature](https://www.kaggle.com/code/kirilltym/0-639-teach-more-keep-the-signature) | 0.636 + 10% readout bank on the LoRA24 features, fit on train + 4,404 private synthetic passages |

The two biggest jumps since 19 Oct were **data plus training recipe** (two-stage LoRA +0.023, then the bigger pool
+0.028 against its own base) and **one blend structure**: main 0.85 + complementary stack 0.15 + 10% ATLAS on
low-margin rows. That structure was invented in nlm-v21 and gave +0.016 when fleongg moved it onto the 0.620.

## 3. Who built on whom (lineage summary)

```
lucasboesen 0.541 (adaptive centring, twins, LoRA24)
 ├─ kirilltym 0.550 → … → 0.568 (Gaussian heads) ─ v19 0.594 (two-stage .85 + stack .15) ─ v20 0.600 (+Gaussian on 8B LoRA)
 │     ├─ sangam21 0.604 (+Gaussian 14B, em-dash guard, Hungarian) → kiza 0.604 / 0.603, ozertuu 0.604, flexonafft 0.604, saddamaditya 0.604
 │     ├─ josephadamski91 0.607 (+10% ATLAS) → kirilltym 0.637 v1 0.605 (ATLAS 15%)
 │     └─ copies 0.601 (zhangyiji v15, otmaneelmahdaouy)
 ├─ vnton37nguyn 0.563 → prvsiyan 0.569 (Gaussian) → v2–v5 0.593–0.604 (+two-stage branch .85, own LoRAs)
 │     ├─ zbigniewkirchner 0.574 (in-style 8B LoRA) ─ alexchilton 0.597 ─ zhangyiji v17 0.601 (+ATLAS, −0.002)
 │     └─ dhruvtibarewal nlm-v21 0.611 (comp stack .15 + ATLAS 10% on margin < .3)
 │           ├─ dazhijiang 0.612 (in-style 8B swap)   ├─ mrpicklerick 0.600 (comp off, post off)
 │           └─ lightningv08 v14–v18 0.602–0.610 (+Qwen3.8-27B tail)
 └─ nishantdahal 0.592 (two-stage LoRA on the public pool) → josephadamski91 0.596, huangkx666 0.596–0.597
       └─ nishantdahal part two 0.620
             ├─ cosmorin 0.622 (SVC prefit cache) → emmanuelmirekumensah 0.622 (copy)
             ├─ exact copies 0.620: bartwoz, emmanuelmirekumensah, huangkx666 v24, saamhm v22
             ├─ nuridudak 0.624 (+ModernBERT .25, desklib .10, hard guards) → 0.626 (MB .40); ozertuu 0.624 (copy); romantamrazov v8 0.624
             │     └─ mrpicklerick 0.625 (guards off) → 0.627 (MB .30 / desk .05) → romantamrazov v9 0.628 (SVC topic centring)
             │           ├─ adityaakbar 0.631 (+10% UltraBERT/ELECTRA, public encoder dataset)
             │           └─ amanatar 0.632 (6-view SVC, kNN centring, soft guards …) → v2 0.621 (reweights)
             └─ fleongg 0.636 (0.85 × 0.620 + 0.15 × comp stack, 10% ATLAS, from nlm-v21) → mtoshidesu 0.636 (copy)
                   ├─ kirilltym 0.637 v3 (+readout bank, train only) / 0.639 (bank + 4,404 synthetic) → maverickss26 v4 0.639
                   └─ lightningv08 v20 0.636 (+full-text Qwen3.8-27B 10% on low-margin half; tie)
```

**How each link is known:**
- **Kaggle fork records:** kirilltym 0.600 ← ahmedberatozer; kirilltym 0.637 ← josephadamski91;
  lightningv08 0.636 ← lightningv08 0.595; sangam21, thechi ← kirilltym 0.600; cosmorin, bartwoz ← nishant 0.620;
  mtoshidesu ← fleongg.
- **Markdown credits:** fleongg ← nishant v2 (plus "our scored 0.612 method", which is nlm-v21's recipe);
  kirilltym 0.639 ← fleongg V1; nuridudak ← nishant; adityaakbar ← mrpicklerick v2; flexonafft v13 ← sangam21;
  romantamrazov v11 ← fleongg.
- **Everything else is code similarity.** The 0.624 → 0.627 steps are one-constant diffs.

## 4. What the 0.636, 0.628, 0.627 and 0.626 add on top of nishantdahal's 0.620

All of these keep nishant's members exactly: n-gram SVC 1.0, Qwen3-14B LoRA 0.4, Qwen2.5-14B LoRA 0.2, centred twins
0.1 each, DeBERTa-v3-large 0.25, Sinkhorn with 300 iterations.

| notebook | score | added on top of the 0.620 | inputs it needs |
|---|---|---|---|
| cosmorin public620-reproduction | 0.622 | nothing in the model; the SVC is loaded from a CPU-prefit, sha-pinned joblib cache instead of being refit in the run. **Deterministic:** our copy also scored 0.622 | `cosmorin/ntl620-svc-cpu-prefit-v1` output |
| nuridudak 0-624 | 0.624 | ModernBERT-large 0.25 + desklib DeBERTa-v3-large 0.10 (both two-stage: 122k external, then train) + 3 hard typography guards after Sinkhorn | `nuridudak/ntl-ours-v1`, **now private** |
| nuridudak 0-626 | 0.626 | ModernBERT 0.25 → **0.40** (one constant) | same, private |
| mrpicklerick 0-624-base-without-guards | 0.625 | guards **off** (+0.001) | same, private |
| mrpicklerick 0-627 | 0.627 | guards off, ModernBERT **0.30**, desklib **0.05** | same, private |
| romantamrazov who-am-i v9/v10 | 0.628 | the 0.627 + SVC topic centring: centred SVC scores in content clusters of 8–18, added at 0.62 × a gate on low blend margins (+0.001) | same, private |
| romantamrazov v11/v12 | 0.621 | without ntl-ours-v1, so the encoders are skipped: **−0.007** | |
| adityaakbar 0627-small-encoder | 0.631 | the 0.627 on the **public** `nuridudak/ntl-two-stage-encoders` (Apache-2.0) + 10% probability mix of UltraBERT 5-fold + an ELECTRA-large readout, before Sinkhorn (+0.004) | public, plus an `adityaakbar/ntl-0627-encoder-cache` output |
| amanatar ultimate-sota | 0.632 | the 0.627 members and weights + 6-view SVC on a class-balanced pool, consensus clustering with bisection, kNN centring, soft guard penalties, cavity repulsion, partial Sinkhorn α 0.82 (many changes at once, +0.005) | public |
| **fleongg part-two-complement** | **0.636** | `P = 0.85·softmax(LOGIT_0.620) + 0.15·softmax(comp)`, then on rows whose top-2 margin < 0.3, `P = 0.9·P + 0.1·ATLAS`, then Sinkhorn. **comp** is the 0.563–0.611 frozen-feature stack, run as a subprocess (see below) | all public; licences in §6 |

**The comp stack contains:**
- the stylegram SVC at 1.516 and stylometry LightGBM;
- UltraBERT 0.095;
- frozen DeBERTa, ELECTRA and desklib heads (0.05 / 0.032 / 0.132);
- LoRA24 public / seed7 / Claude heads (0.177 / 0.113 / 0.177);
- the in-style Qwen2.5 LoRA 0.17;
- token-attention Qwen3.5 / DeBERTa (0.153 / 0.185);
- centred twins on Qwen3-8B, Qwen2.5-7B, Qwen3-4B, GLM-4-9B and DeBERTa;
- shared-prompt Gaussian heads on four frozen views;
- Claude and family gates;
- balancing α 0.5.

**Why "encoders on the 0.636 gave 0.634"** (fleongg's report, as the user relayed it; I could not read notebook
comments without signing in):
- desklib, UltraBERT, ELECTRA and DeBERTa are **already inside the comp stack**. Adding desklib or the small encoders
  again mostly double-counts them.
- **ModernBERT is the only encoder that is new to the 0.636.** On the 0.620 its main effect was about
  +0.004 to +0.007 (the romantamrazov v11 loss shows the size).
- Inside a blend that already carries 15% of other encoders, the expected marginal gain is +0.001–0.002 at best,
  which is consistent with a −0.002 draw.

**The 0.637 and 0.639** (kirilltym, built on fleongg V1):
- **The addition:** a fixed 10% "night24" readout bank, applied after ATLAS and before Sinkhorn. It reads the exact
  public LoRA24 features with late-layer Fourier, PCA-Fourier, word/character prompt residuals, Laplace, kNN and
  calibrated RBF readouts.
- **0.637:** the bank is fit on official train only.
- **0.639:** the bank is fit on train plus **4,404 privately generated passages** (12 models, weight 0.25).
- **Where it lives:** `kirilltym/ntl-spectral-style-heads-paired-4404` is public, licence "unknown". It holds only the
  fitted banks, not the passages.

## 5. Change types: what they gave, and whether it repeated

Full rows are in `change_types.csv`. A "confounded" step changed other things too.

| change type | measured steps (base → child) | verdict |
|---|---|---|
| **Complementary stack** (main 0.85 + the 0.563-era stack 0.15) | +0.002 (0.592 → 0.594), +0.001 (prvsiyan v2), +0.005 (0.592 + 0.569 at 0.5), **+0.016** (0.620 → 0.636, with ATLAS); removing it: −0.006 (flexonafft v15), −0.011 (mrpicklerick 0.600, with post off) | **Repeated: positive 5 times out of 5, and removal hurt twice.** The strongest blend-level lever, because the two halves disagree (frozen-feature stack vs fine-tuned LoRAs) |
| **ATLAS share** (10% before balancing) | +0.002, +0.002 (author's two bases), **+0.007** (0.600 → 0.607), +0.004 (with early layers on 0.592); **−0.002** on prvsiyan's 0.603; 10% → 15%: −0.002 | Mostly positive at 10%, and more is worse. Margin-gated (< 0.3) in every ≥0.611 notebook |
| **Encoder members** (ModernBERT, desklib) | +0.004 (0.620 → 0.624, with guards), +0.002 (MB 0.40), +0.002 (MB 0.30 / desk 0.05), loss when absent −0.007; **on the 0.636: −0.002** (reported) | Repeated on the 0.620 base; **did not stack on the 0.636** (overlap with the comp stack) |
| Small encoders (UltraBERT, ELECTRA, 10% probability mix) | +0.004 (0.627 → 0.631); UltraBERT 0.15 log-space + reweights −0.011 (amanatar v2) | One clean gain, and it is already inside the comp stack |
| **Gaussian group heads** (shared-prompt) | +0.006 (prvsiyan v1), +0.006 (kirilltym v18), **+0.006 (on the two-stage 8B LoRA, kirilltym v20)**, +0.005 (prvsiyan v4, all LoRAs), +0.004 (sangam21, 14B plus guard and Hungarian), +0.001 (lightningv08) | **Repeated, about +0.005 each time at the 0.56–0.60 level.** Never applied to the 0.620 main's 14B LoRAs (the comp stack has them on frozen views only) |
| New LLM members | in-style 8B LoRA +0.005; own two-stage LoRAs +0.006 (prvsiyan v3); in-style swap +0.001; OLMo2 +0.001; Qwen3-4B early layers −0.001; Qwen3.8-27B tail at 0.2 / 0.1 / 0.05: −0.009 / −0.001 / −0.001; full-text 27B 10% on the 0.636 reference: 0.000 | Early on, new diverse readers paid. Above 0.61 a new big backbone has not helped |
| **Pool and data** | pool 35k → 84k (+0.028 with two members; the author says the pool is most of it: 14B CV 0.569 → 0.611); in-style share 0.35 → 0.60 +0.003; +4,404 synthetic passages in the readout bank +0.002 (0.637 → 0.639) | **Repeated, and the largest gains.** Every data step has been positive |
| Stage-1/stage-2 recipe | two-stage LoRA +0.023 at its introduction; EOS-id fix part of the 0.620; nishant lists more stage-2 epochs, pool in stage 2, short windows and seed averaging as not helping | One big structural gain; tuning it further was flat |
| Centring and twin heads | +0.016 (introduction, lucasboesen v1), +0.001 (cohesion), +0.005 (twin = plain weight, with an extra member), SVC topic centring +0.001 | Big once; small after |
| Typography guards | hard guards in nuridudak (confounded), **guards off +0.001**, curly-quote guard + Hungarian 9 −0.001, soft penalties in amanatar (confounded) | **Neutral (±0.001).** The blend already obeys the rules; this matches our 18 Oct lab finding |
| SVC changes | prefit cache +0.002 (deterministic); 6-view balanced pool (confounded +0.005); SVC 1.0 → 0.7 (in the −0.011) | Small; the cache step is real but is a refit difference, not a recipe change |
| Sinkhorn and balancing | α changes inside confounded steps only (romantamrazov v4 off, amanatar α 0.82) | No clean evidence; everyone keeps full Sinkhorn at 300 iterations |
| Hungarian assignment | part of sangam21 +0.004; −0.001 (kiza, from size 9); forum: 0.427 → 0.425 (danielandresmiranda); nishant says worse on the LB; entropic OT ≈ 0 (lynnsakurai) | **Neutral to negative.** It only works when the test has many passages per prompt, which it does not seem to |
| Readout bank | +0.001 (train only), +0.002 more with synthetic data | New; one notebook, but reproduced by an independent refactor (maverickss26 v4 = 0.639) |
| Blend weights | amanatar v2 −0.011, mrpicklerick 0.600 −0.011 (mixed), ATLAS 15% −0.002, encoder weights +0.002 twice | **Pure reweights above 0.62 were flat or negative**, except the encoder-weight tweaks on the 0.624 |
| TabPFN | only in the 0.551–0.557 era (lightningv08, kirilltym v14) | Not in any ≥0.600 notebook |

## 6. Inputs that are private, deleted, or carry licence questions

| input | status | licence | affects |
|---|---|---|---|
| `nuridudak/ntl-ours-v1` | **private now** (403) | — | 0.624, 0.625, 0.626, 0.627, 0.628 cannot run as published; romantamrazov v11/v12 fell to 0.621 without it |
| `nuridudak/ntl-two-stage-encoders` | public replacement | **Apache-2.0** | adityaakbar 0.631 and amanatar 0.632 run the 0.627 recipe on it |
| `josephadamski91/ntl-atlas-qwen3-8b-v1` | public | Kaggle "other"; `LICENSES.md`: weights CC BY-SA 4.0, code Apache-2.0 | every ATLAS notebook (0.607, 0.611, 0.612, 0.636–0.639); **accepted by the user under CC BY-SA on 24 Oct** |
| `kirilltym/ntl-spectral-style-heads-paired-4404` | public | **unknown** | 0.637 and 0.639. The 0.639 bank is fit on 4,404 privately generated passages that are not released. For a final pick that is a licence question and a §2.6 external-data question. It is fine for rank |
| `lightningv08/ntl-olmo2-native24-final-inference-assets` | public, but re-versioned | "other" (README: components Apache-2.0 / CC BY-SA 4.0) | the v20 0.636 runtime folders (`ntl636-fulltext-qwen-external3200-runtime`, `ntl627-own-fulltext`) are **not in the current version**, so v20 cannot be rerun as published. It also bundles a Llama-3.1 prefix (Llama Community Licence) and an NF4 Qwen3.8-27B |
| `fleongg/ntl-quality-selected-public-svc-pool`, `fleongg/ntl-legacy-qwen24-recovery` | public | "other", no terms | 0.636, 0.637, 0.639. Derived from public CC BY / CC BY-SA pools and the public LoRA24 |
| `nishantdahal/ntl-handwriting-blend`, `nishantdahal/ntl-two-stage-lora` | public | CC BY-SA 4.0 | every ≥0.620 notebook |
| `bbucxi/qwen35-9b-safetensors` | public | "unknown" on Kaggle (Qwen weights are Apache-2.0 upstream) | the comp stack's token-attention member |
| `lucasboesen/name-that-llm-glm4-9b-trunc25` (notebook output) | public | GLM-4-9B weights: GLM-4 licence (commercial use after registration) | the comp stack's GLM-9B twin and Gaussian head, so in 0.611–0.639 |
| desklib/ai-text-detector-v1.01 (base of desklib members) | public | to check | comp stack (desk 0.132), nuridudak encoders |
| TabPFN | — | non-commercial | none of the ≥0.600 notebooks |
| `private-data-source` | private | — | romantamrazov v12 |

## 7. Forum: what the top-10 teams say

**Who.** Top 10 at 10:16 UTC:

| rank | team | score |
|---|---|---|
| 1 | Hod Carriers (tononnh) | 0.700 |
| 2 | Bootstrapping (wujianming) | 0.688 |
| 3 | vishakkbhat | 0.673 |
| 4 | sorenravn | 0.673 |
| 5 | Le Viet (sleepyssr) | 0.672 |
| 6 | takumashiga | 0.666 |
| 7 | adisak | 0.666 |
| 8 | horikitasaku | 0.664 |
| 9 | mutesacedric | 0.664 |
| 10 | kulshekhar | 0.662 |

I read all 65 topics and 223 posts through the API. Seven posts are from top-10 accounts, and none describes a model.
What they do say:

- **tononnh (#1)**, in [749447](https://www.kaggle.com/competitions/name-that-llm/discussion/749447) "Will the winner
  just be whoever can get the most extra training data?": "My best LB score before adding external data was 0.583, but
  I suspect part of that is LB probing." So the #1 team's lead came with external data.
- **vishakkbhat (#3)**, in [750947](https://www.kaggle.com/competitions/name-that-llm/discussion/750947) (OpenRouter +
  OpenAI passages): "Removing this dataset … reduced the score from 0.463 to 0.456 on my validation set."
- **adisak (#7):**
  - [750390](https://www.kaggle.com/competitions/name-that-llm/discussion/750390): group folds by `prompt_id` (random
    folds 0.217 vs grouped 0.369, LB 0.355); truncated-copy augmentation does not help (0.343 → 0.335 / 0.331).
  - [751011](https://www.kaggle.com/competitions/name-that-llm/discussion/751011): asks the hosts whether
    self-generated API passages are allowed under §2.6 without publishing them, and what cost counts as reasonable.
    No host answer yet.

**Other threads that matter for the chain:**
- **danielandresmiranda**, [751871](https://www.kaggle.com/competitions/name-that-llm/discussion/751871):
  - test-time grouping with Hungarian "did not transfer" (0.427 → 0.425);
  - coverage of all 18 labels matters more than volume;
  - two readouts of one backbone disagree on 29% of passages and add about 2 points.

  That last point is the same idea as kirilltym's readout bank.
- **seyitkaangunes**, [750224](https://www.kaggle.com/competitions/name-that-llm/discussion/750224): extra Claude
  passages help as a within-Claude gate, not as 18-way training rows.
- **swapiy**, [751941](https://www.kaggle.com/competitions/name-that-llm/discussion/751941): the public in-style pool
  does not leak train prompts (only 0.26% of pool passages are above cosine 0.4 to a train-prompt centroid). So pool
  gains in CV are real.
- **josephadamski91**, [750447](https://www.kaggle.com/competitions/name-that-llm/discussion/750447): 10% ATLAS took
  0.556 → 0.558 and 0.549 → 0.551.

**Reading:** the 0.06 gap between the best public notebook (0.639) and the top (0.700) is most likely **data**. The
#1 team says so implicitly, the #3 team measures it, and in the public chain every data step was positive and the
biggest. The rules question about privately generated data (§2.6) is still open.

## 8. Is there a public path above 0.636?

**Yes, and it already exists: 0.639**, built from public inputs only (kirilltym, and an independent refactor by
maverickss26). Its caveat is the readout bank's "unknown" licence and its private training passages.

Can non-overlapping changes go further? The table below shows what is already inside the 0.639 and what is not:

| change | in 0.639? | overlaps with |
|---|---|---|
| nishant 0.620 members, twins, Sinkhorn | yes | — |
| comp stack 0.15 (stylegram, LoRA24 ×3, in-style Q2.5, token-attention, UltraBERT, DeBERTa/ELECTRA/desklib heads, Gaussian heads on frozen views, gates) | yes | encoders, small encoders, Gaussian on frozen views |
| ATLAS 10% on margin < 0.3 | yes | ATLAS share |
| readout bank 10% | yes | — |
| **Gaussian head on the main's Qwen3-14B / Qwen2.5-14B LoRA features** | **no** | none: the comp stack's Gaussian heads read frozen views, not these LoRAs |
| ModernBERT member | no | partly: other encoders are in the comp stack |
| cosmorin SVC prefit cache | no | none (it replaces the main's SVC fit) |
| SVC topic centring (romantamrazov) | no | none |
| typography guards, Hungarian | no | — (both neutral) |
| Qwen3.8-27B full text | no | none, but it tied |

## 9. Ranked: combinations likely to beat 0.636

Each forecast uses the observed step sizes; ±0.001 is re-run noise. **For final picks:** items 1 and 3 need a decision
on the "unknown" readout-bank licence. Items 2, 4 and 5 are our own training or correction changes on top of whatever
base we keep.

1. **Exact copy of kirilltym/0-639-teach-more-keep-the-signature (or the official-only 0.637).**
   - **Forecast:** 0.639 (0.638–0.640).
   - **Evidence:** two notebooks with different code both scored 0.639. Every input is public and our 0.636 copy
     already mounts nearly all of them; the only new one is `kirilltym/ntl-spectral-style-heads-paired-4404`.
   - **Cost:** the same GPU time as our 0.636 run plus a few minutes for the bank.
   - **Use:** rank only until the bank licence is clarified. The 0.637 bank has no private data in its fit.
2. **0.639 (or 0.636) + shared-prompt Gaussian heads on the main's two 14B LoRAs.**
   - **Forecast:** +0.002 to +0.005 (0.641–0.644 on the 0.639).
   - **Evidence:** the change type repeated about +0.005 four times. The exact form, a Gaussian head on two-stage LoRA
     features, gave +0.006 at kirilltym v20 and +0.004 (with extras) at sangam21. It is not present anywhere in the
     0.620/0.636/0.639 main, so nothing overlaps.
   - **Risk:**
     - the head is fit on train features of LoRAs that were themselves fine-tuned on train, so it is overconfident in
       sample. kirilltym's +0.006 had the same issue and still paid;
     - its weight must stay small (0.10–0.15 as in v20) because it enters before the 0.85/0.15 mix.
   - **Cost:** about 15–20 min more of T4 for the 2,699 train passages through both 14B trunks (nishant's inference
     code). The fit itself is on CPU.
   - **Validation:** our blend sim can check it at 4–6 and 8–10 per prompt before any slot is used.
3. **Licence-clean readout bank: refit kirilltym's bank on official + public external passages instead of the 4,404
   private ones.**
   - **Forecast:** 0.637 → 0.638–0.640.
   - **Evidence:** the private passages added +0.002 over the train-only bank. Data steps have been positive every time
     in this chain, and the top teams point at data. The public CC BY pools (seyit, abdullah1afify, dophuchung, lucas,
     prvsiyan, nishant's pool_v4) cover all 18 labels.
   - **Cost:** LoRA24 features for the extra passages, about 1 GPU-hour per 10k on T4. The bank fit runs on CPU with
     its public `readouts.py`.
   - **Use:** this makes the 0.639 idea usable as an honest final pick.
4. **Swap the main's runtime SVC fit for cosmorin's prefit cache, inside the 0.639.**
   - **Forecast:** +0.000 to +0.002.
   - **Evidence:** +0.002 on the 0.620, deterministic (our copy matched), and orthogonal to everything added later.
     Inside the 0.85 main its effect shrinks.
   - **Cost:** almost free; it also saves the SVC fit time.
   - **Caveat:** it is a different fit, not a better recipe, so it falls under the CLAUDE.md rule against selection by
     public score. Report it as public-tuned.
5. **ModernBERT only (no desklib) as a 10% probability-space mix after the 0.85/0.15 step, the way adityaakbar added
   the small encoders.**
   - **Forecast:** −0.002 to +0.002.
   - **Evidence:**
     - ModernBERT is the one encoder that is not already in the comp stack, and it was worth +0.004–0.007 on the 0.620;
     - fleongg's reported 0.634 shows the naive version fails;
     - the probability-space mix kept the +0.004 in adityaakbar's case.
   - **Uses:** the public `nuridudak/ntl-two-stage-encoders` (Apache-2.0).

**Not recommended**, on this evidence:
- re-weighting the main/comp/ATLAS shares (every pure reweight above 0.62 was flat or negative; ATLAS 15% was −0.002);
- adding Qwen3.8-27B (tie at best);
- typography guards and Hungarian assignment (neutral or negative);
- desklib, UltraBERT or ELECTRA again (already in the comp stack).

## Sources

- **Scratchpad** (session scratchpad,
  `/tmp/claude-0/-home-user/b6319532-f3ee-57bf-b6c7-f5215a48473a/scratchpad/`):
  - `ntlchain/`: `meta.json` (all versions and sources), `src/` (raw sources), `code/`, `md/`, `norm/`, `emb/`
    (extracted embedded scripts), `forum/` (65 topics, `posts.json`), `lb/` (leaderboard snapshot), `dsmeta/`
    (dataset metadata).
  - `ntlchain_scripts/`: `collect.py`, `recheck.py`, `totext.py`, `embedded.py`, `lineage.py`, `stepdiff.py`,
    `build_csv.py`.
- **Reference for the format:** the dariushafshar CASMI chain notebook and its dataset (CC BY 4.0).
- **Our vault:** [[Name That LLM]] (24 Oct scores, the ATLAS licence decision) and [[Notebook Lineages]].
