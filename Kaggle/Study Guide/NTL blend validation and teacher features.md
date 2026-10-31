---
tags: [kaggle, ntl, research, validation, teacher]
date: 2026-11-01
---
# NTL: judging a member inside a blend without OOF, and teacher / likelihood features

Research only (1 Nov 2026, 02:2x IST). Sources: forum threads 753177, 753053, 752708, 754084, 750213, 751871, 752197,
752199 (fetched with comments); alexchilton's "0.662 All Public + What Failed" notebook (markdown cells, kernels pull
API); web search for GLM-5.3-Flash / Kimi K3 facts. arxiv.org is not reachable from this session (DNS blocked), so the
ATLAS paper and the detector papers are cited from our own notes and from memory, not re-read. Nothing built, pushed or
submitted. Back to [[Name That LLM]], [[Above the public line]], [[NTL public chain]].

## A. Validation recipe for a member inside a blend

### What GoPain (locdunhat) actually did
The comment authors show as anonymous in the API; the recipe is the reply in 753177 and the posts 752708, 753053, 754084
(code: `locdunhat/group-centring-when-test-topics-must-be-found`, `locdunhat/validating-test-time-centring-for-the-test-batch`,
`locdunhat/which-offline-score-tracks-the-lb`, dataset `locdunhat/proxy-selection-simulations`, CC0).

- **Folds:** 5-fold GroupKFold on `prompt_id` (about 30 prompts, 540 passages per fold), all members out of fold.
- **Test-like shape:** inside each validation fold, cut every prompt to **m random answers, m in {18, 6, 3, 2}**,
  **15 draws per m** (5 in 752708), the same draws for every configuration. The class mix is whatever the random cut
  leaves (roughly uniform); no uneven mix was simulated.
- **Rebuild test-side groups**, not true prompt ids: stop-word-free TF-IDF cosine, average linkage, cut 0.85, fitted on
  the fold; centring/twin heads applied inside those clusters (2-18 passages).
- **Score the whole pipeline**, balanced accuracy. Sinkhorn: **off** in the 752708 LB check ("no Sinkhorn"); the 15-sub
  blend family is not stated. 751871 found Sinkhorn -0.003 on OOF while LB ablations give it +0.002 (753177 reply), so
  OOF does not see Sinkhorn's value: keep it on and identical in both arms, do not judge it.
- **Tracking:** 15 submissions of one blend family (LB 0.593-0.621). Pearson/Spearman with the LB: m=18 0.13/-0.17,
  m=6 0.45/-0.09, m=3 0.70/0.25, **m=2 0.79/0.58** (0.91/0.82 on the first 10). Without one outlier m=2 Spearman 0.49.
- **Failure cases:** (1) the next 5 picked by m=2 from a 162-config grid: Spearman 0.05 (selection inflation; 754084's
  simulation says this collapse is expected even for the true best proxy). (2) Small cuts **under-rate a centred twin**
  (+0.002/+0.007 at cut 2/3 vs +0.017 on the LB, nearer the m=18 +0.024). (3) Full-prompt OOF over-picks centring weight
  (t 1.5-2.0 all lost to 1.0; centring 0.5 best on m=18 was worst on LB). Cause (753053): a fold has ~30 prompts, the
  test a few hundred, so test clusters are less pure.
- **His current rule (754084):** accept a change only if it **improves every cut** (18/6/3/2), rank by the **mean over
  cuts**, ignore the order of the top few submissions. Best (unrun) idea: embed the validation passages into the
  unlabelled test batch, cluster together, score only the validation passages (worst regret 0.014 vs 0.030-0.061).

### Other LB facts that bound what OOF can say
- Member OOF is a poor guide to LB value: retrained q25 OOF 0.569 -> 0.605 gave LB 0.651 -> 0.647 (swap), keep+add at
  0.4: 0.650. Zero-weight ablations: old q25_14 (OOF 0.569, w 0.2) worth 0.014; SVC (OOF 0.473) 0.011; both twins 0.017;
  ATLAS 10% 0.009; Sinkhorn 0.002. **Diversity of training data is what pays**, not member OOF.
- alexchilton: identical code re-scores within **0.009** (0.645/0.654, 0.643/0.651); his own 5-backbone blend was
  nested-CV 0.6865 and LB 0.56 (true 18-passage groups offline). Adding LoRA members to a strong blend moved +-0.004.

### What we can apply to our line
The 0.662 line's members (nishant/dhruv LoRA24 8B/14B, ll8v5, Qwen3.8-27B, DeBERTa, comp stack, ATLAS) have no OOF, so
"the blend with vs without the member" can only be approximated by a **proxy blend we can rebuild out of fold**:

| member | OOF reproducible? | cost |
|---|---|---|
| n-gram SVC (4 views, pool) | yes, have it (`lab/artifacts/svc12_*.npz`) | CPU |
| LoRA24 Qwen3-8B plain + centred | yes, have it (`lora24_oof.npz`) | done |
| ModernBERT / desklib two-stage, 3 token-attention | yes (lab 07/08) | done |
| q8pool (training now) | fold0 only (540 rows); 5 folds = +4 stage-2 runs | ~1 GPU h each |
| ATLAS | rebuild exists (`lab/atlas/`), 5-fold training not run | ~5 x 0.6 h GPU |
| comp stack (frozen features + heads) | in principle (frozen features on 2,699 + heads), not done | ~1-2 GPU h |
| nishant/dhruv 14B, ll8v5, 27B, q25_14 | **no** (pool stage 1 would have to be re-run; 7-13 GPU h each) | out of reach |
| Gaussian head, typography likelihood, Sinkhorn | yes, deterministic on top of OOF | CPU |

**Recipe for us (one script, CPU):** proxy blend = SVC + LoRA24 plain/centred + encoders (+ ATLAS if rebuilt), weights
fixed from the notebook (not tuned); m in {18,6,3,2} x 15 draws per fold, groups from the notebook's own clustering;
the new member added at the weight it would get in the notebook (member count excluded from `wsum`, as in our q8pool
variant); Sinkhorn on in both arms; paired bootstrap over prompts per m. **Accept only if the delta is >0 at every m
and P(better) >= 0.9 at the mean over cuts.** Then read the forecast as "+delta x (shrink 0.3-0.5)" because the real
blend's members overlap more than the proxy's (our 08 note: proxy gain is an upper bound). Limit: on fold0-only members
(540 rows, ~30 prompts) the bootstrap is wide; a +0.005 delta will not reach 0.9.

## B. Teacher / likelihood features from candidate models

### Method family
| method | idea | needs | fit for us |
|---|---|---|---|
| Per-model log-likelihood / rank / entropy (GLTR-style stats) | text from model X has higher logp, lower rank under X | X's weights (or same-family proxy), one forward pass | the direct "generator likelihood" Lian's slices aim at |
| Likelihood ratio between candidates | log p_X - log p_Y separates X from Y, cancels text difficulty | 2+ candidate LMs | strongest for GLM vs Claude if GLM is one side |
| Binoculars (Hans et al. 2024) | ppl under performer / cross-ppl observer->performer; zero-shot, normalises for prompt difficulty | two LMs sharing a tokenizer (base + instruct of one family) | built for human-vs-machine; per-family signal only if the pair is that family |
| DNA-GPT (Yang et al. 2023) | truncate, regenerate with candidate, compare n-gram overlap | generation with each candidate: k samples x passages | far too slow for 4,000 x 18 in 9 h; no |
| Ghostbuster (Verma et al. 2024) | unigram/trigram + weaker-LM token probs, combined by feature search, then logistic regression | only small LMs (originally ada/davinci probs) | the template for cheap stats + LR head; yes |
| Fast-DetectGPT | conditional probability curvature, one pass | one LM | binary detector; weak for 18-way |
| ATLAS (arXiv 2610.19583; already 10-20% in our line) | frozen Qwen3-8B, shared LoRA A + per-source B[s] + prefix; class = source with highest likelihood | Qwen3-8B fp16, 18 passes per text | already in the line: same idea with a *learned* proxy per source |

ATLAS is the learned version of "likelihood under each candidate"; a true-generator teacher adds what it cannot learn
from 150 prompts: the real model's distribution. That only exists for the three open labels.

### The open-weight labels (facts, 1 Nov)
| model | size | licence | Kaggle | fits 2xT4 (32 GB VRAM, ~29 GB RAM)? |
|---|---|---|---|---|
| Kimi K3 | 2.8T MoE, ~104B active | Kimi K3 licence, MIT-style (per Lian's dataset) | unsloth UD-IQ2_XXS as 4 layer slices (lianmatsuo, 130-180 GB each); MXFP4 mirror | **no**; needs layer-range passes over ~0.7 TB from /kaggle/input |
| GLM-5.3 | 744B MoE, 40B active | bespoke commercial licence, revenue-gated review (check before use) | UD-IQ2_M GGUF shards | **no** |
| GLM-5.3-Flash | 320B MoE, 18B active | **MIT** (Z.ai card, per secondary sources) | rahim3 UD-IQ3_XXS GGUF, 2 parts, 120 GB | **no in memory**; only by layer-sliced passes from disk |
| no small Kimi-K3 / GLM-5.x checkpoint | - | - | - | search found none |

**GLM-5.3-Flash cost estimate.** 4,000 passages x ~350 tokens = 1.4 M tokens. 120 GB does not fit 61 GB (VRAM+RAM), so
it must stream from /kaggle/input by layer range (Lian's design): ~4 slices, each a full pass storing 4,000 x 350 x
hidden activations (fp16 ~ 25 GB per boundary, over the 20 GB /kaggle/working limit unless chunked by passage). MoE
prefill on T4 at 3 bits with CPU-side experts: optimistic 200-500 tok/s -> 0.8-2 h per slice set, plus 120 GB reads
at ~200-400 MB/s (~5-10 min per pass, repeated per chunk). Realistic 3-8 h of the 9 h budget **before** the 0.662 line
runs (which already uses most of it). Plus training the head needs the same on 2,699 train + pool (~10-30 GPU h).
**P(works inside the hidden run, in time) about 0.05-0.1.** Not recommended.

### Recommended feature: same-family small-proxy likelihood stats (Ghostbuster-style), small weight
- **Proxy LMs:** GLM-4-9B (base and chat; GLM-4 licence, commercial use after registration, already a frozen member
  in the comp stack via `lucasboesen/name-that-llm-glm4-9b-trunc25`) as the GLM-family proxy; for Kimi, Moonshot's
  Moonlight-16B-A3B (MIT; Kaggle availability unverified, it would need a public upload) or skip Kimi. Plus Qwen3-8B
  base already loaded for ATLAS/LoRA (free logits).
- **Features per passage (~40):** mean/std/quantiles of token logp, mean log-rank, entropy, under each LM; pairwise
  logp differences (GLM - Qwen, chat - base); Binoculars ratio within GLM-4 (base observer, chat performer).
  18-way logistic regression (or LightGBM) on these + nothing else; class log-probs into the blend at **w 0.05-0.10**
  before Sinkhorn, outside `wsum`.
- **Cost:** GLM-4-9B fp16 split across 2 T4 (or 8-bit on one), ~2-3k tok/s prefill: hidden test 1.4 M tokens x 2 LMs
  ~ 20-30 min (fits next to the line only if ATLAS's budget is trimmed). Training: score train (2,699) + a 30k pool
  subset (~11 M tokens) ~ 2-3 GPU h, once; head on CPU.
- **Validation (A's recipe):** honest 5-fold OOF of the head on train (pool rows only in training folds), add to the
  proxy blend at each m in {18,6,3,2}, Sinkhorn on, paired bootstrap; accept only if positive at every m and
  P >= 0.9; also report glm-5.3 / glm-5.3-flash / kimi-k3 / claude-fable-5 recall deltas. It is a new member trained
  on its own (a modelling change, eligible), and it is trained on different signal than every LoRA (the diversity that
  753177 says pays).
- **Expected effect:** low-dimensional likelihood stats separate families more than siblings; GLM-4 is not GLM-5.3,
  so expect lab-level gain (GLM vs Claude confusion) more than GLM-5.3 vs -Flash. Weak classes +0.02 to +0.05 recall
  alone; blend +0.000 to +0.004 public. **P(passes the 0.9 bar) about 0.15; P(public > 0.664 from it) about 0.1.**

## Next step
1. CPU first (no GPU): write the multi-cut proxy-blend scorer (m 18/6/3/2 x 15 draws, notebook clustering, Sinkhorn
   both arms, every-cut rule) on the OOF we have, and rerun it on q8pool fold0 once stage 2 lands. That also makes the
   q8pool decision honest.
2. Only then, if GPU is left: GLM-4-9B + Qwen3-8B likelihood-stat head (~3 GPU h), validated with the same scorer.
   Leave GLM-5.3-Flash / Kimi K3 teachers alone unless a public notebook shows them running inside 9 h.
