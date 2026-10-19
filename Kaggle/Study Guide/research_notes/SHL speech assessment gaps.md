# SHL speech assessment: what our research has not yet covered

Written 19 Oct 2026 for SHL Hiring Assessment 2026 (`shl-hiring-assessment-2026`). Metric: 0.6·RMSE + 0.4·(1 − Pearson), lower is better.
Where we stand: #3 on public at 0.3062 (a public-tuned mix). The honest main model scores 0.3178. #1 is at 0.2913 and #2 at 0.2962.
Scope: ideas that are **not** already in `shl-grammar-scoring/README.md` §3–12, `submissions/README.md`, the `src/` docstrings, the handbook (Ch. 4, Ch. 7 SHL section), `What Works.md` or `What Does Not Work.md`.
Rules that apply to every idea (from CLAUDE.md):
- Models must have commercial-use licences.
- No external training data.
- An idea is accepted only if it changes how a model is trained or corrected, and wins on speaker-grouped OOF at test-like crop lengths with paired-bootstrap P(better) ≥ 0.9.

Evidence tiers:
- **A**: two or more independent sources with numbers.
- **B**: one source with a number.
- **C**: plausible, but there is no number, or the number came from a leaky CV.

---

## 0. Access limits

- **The Kaggle forum and notebooks were not reachable.** The competition is private. Kaggle's internal API returns `competitions.get … PERMISSION_DENIED` without a participant login. The unauthenticated `/discussion` page returns 404, and this container has no Kaggle credentials. Web search finds no indexed forum threads.
- **What I used instead:** about 45 public GitHub write-ups by participants (search `shl grammar 2026`). Several of them report their own public scores. These are self-reported and unverified.
- **Earlier SHL competitions:** the "SHL Intern Hiring Assessment 2025" write-ups on GitHub use MFCC + LightGBM / wav2vec2 baselines. There is nothing in them we have not already surpassed.

---

## 1. What our existing research already covers (do not re-propose)

- **Validation and corrections that transferred:**
  - speaker-grouped folds (WavLM-base-plus-SV clusters);
  - training on 4 test-length crops, with crop transcripts;
  - the −0.118 test bias fix, found by shift probes;
  - a ×1.15 stretch on full-length clips only;
  - the decoded metric;
  - the averaging identity, which forecasts mixes to within 0.0005.
- **Models in the main pipeline:**
  - verbatim-prompt Whisper transcripts feeding a DeBERTa-v3-large ridge and a DeBERTa fine-tune (`--crops --drop-zero`);
  - a ridge on Voxtral-Mini-3B audio states plus WavLM / WavLM-large / Whisper-encoder features;
  - a WavLM-large SVR;
  - an attention-pooling head on cached Voxtral frames, trained on a fresh random test-length prefix every epoch;
  - a score-0 gate;
  - separate blend weights for the 45.06 s batch.
- **Failed on the public LB (composition changes):**
  - extra frozen encoders (Qwen3-4B, w2v-BERT 2.0, Qwen2-Audio, Phi-4);
  - kernel ridge, three times;
  - training without the noise clips;
  - Voxtral rubric-state features;
  - a fourth blend;
  - clipping to [2, 5];
  - a batch offset;
  - RoBERTa instead of DeBERTa;
  - nine-seed averages.
- **Rejected by validation:**
  - LLM judges (Qwen3-8B, Gemini), LanguageTool, T5 edit rate, CoLA, CEFR classifier;
  - Voxtral asked for the score directly; Voxtral QLoRA (fold 0 got worse);
  - window TTA, co-training, topic-direction removal;
  - CrisperWhisper (also CC-BY-NC);
  - Voxtral-Small-24B, Phi-4 states, Phi-4 transcripts, Qwen2.5-7B LoRA;
  - TAPT;
  - attention heads on Phi-4 and WavLM-large frames;
  - head with transcript conditioning, EMA or dropout 0 (all within noise);
  - importance weighting;
  - soft pseudo-labels for all ridge/SVR members (worse; text ridges only gave a small gain; step 22, 19 Oct).
- **Calibration is exhausted.** The optimal shift is −0.1179 and the best rescale is 0.98. The handbook's "a·p+b calibration" is effectively done.
- **Handbook (Ch. 4 and 7) procedure hacks still untried:**
  - AWP in the DeBERTa fine-tune. It was neutral on the small CommonLit-2 and AES2 regression sets.
  - Full-data refits blended over seeds. These cannot be validated directly on OOF.
  - Self-distillation of trained heads. Only pseudo-labelling of the ridges has been done.
  - A max-length sweep. This is moot: transcripts of 45–60 s of speech fit in 256 tokens.
  - Backbone dropout 0 for the DeBERTa fine-tune. It was tested only on the attention head.
- **Unlogged gap.** `src/07_finetune_audio.py` (WavLM-base-plus fine-tune, random 15 s crops) exists, but no result for it appears in any log or artifact. **End-to-end fine-tuning of a speech encoder has never been evaluated here.**

---

## 2. What competitors' public write-ups show

| Repo (self-reported public score) | Method notes relevant to us |
|---|---|
| [prblydv/shl-grammar-scoring](https://github.com/prblydv/shl-grammar-scoring) (**0.2962, "1st" on its 19 Oct snapshot**; documented path ends at 0.3141 on 15 Oct) | AST (AudioSet, BSD-3) spectrogram branch + DeBERTa + 11 SSL layer summaries in one gated fusion network, trained on rented RTX 4090/5090. 14 base OOF predictions feed a quadratic/multiscale-RBF **residual meta-learner** and a learned **hurdle** (P(y>0) × positive regressor). The final 60/40 weights were "chosen after public feedback". The meta-learner's development score was *worse* for the public-best weight. Folds are grouped only by clean/noisy source pairs, not by speaker, so its 0.4776 OOF RMSE is optimistic. **The step from 0.3141 to 0.2962 is not documented anywhere in the repo** ([ENSEMBLES.md](https://github.com/prblydv/shl-grammar-scoring/blob/main/docs/ENSEMBLES.md), [winner_manifest.json](https://github.com/prblydv/shl-grammar-scoring/blob/main/reports/winner_manifest.json)). |
| [AMITABHSHUKLA/spoken-grammar-scoring](https://github.com/AMITABHSHUKLA/spoken-grammar-scoring) (0.3064, #1 around 13 Oct) | The "public solution" we already know: Voxtral-Mini L12–14 ridge, WavLM-large layer-20 SVR, and LightGBM on 123 transcript features, including a Qwen3.5-9B AWQ probability-weighted rubric score and an LLM edit rate. A wav2vec2-CTC "any speech?" gate. NNLS blend. Speaker-grouped CV RMSE 0.505 / r 0.867 on 732 clips. |
| [nakulgrover118/grammar-scoring-engine](https://github.com/nakulgrover118/grammar-scoring-engine) (0.3211) | **Pools a speaker's own *test* clips**, validated on speaker-grouped CV. 54 base regressors in 4 NNLS stacks. A "similar-voice adjustment" that the author shows is noise on train but kept because it helped public. Details in §3, idea 1. |
| [PRINCE2-AI/shl-grammar-scoring-engine](https://github.com/PRINCE2-AI/shl-grammar-scoring-engine) (0.3262) | Same family as ours. A separate stacker for the 45 s batch took it from 0.3390 to 0.3262 (we already do this). A Qwen2.5-3B LoRA regressor was "better OOF, worse public". |
| [zeelrathi19/shl-grammar-scoring](https://github.com/zeelrathi19/shl-grammar-scoring) (no LB score) | CPU only. **Parakeet-TDT-0.6B encoder, layers 6+18 mean-pooled, into an SVR: 0.481 CV RMSE alone, carrying stack weight 0.76.** Whisper-small adds a little. Caveat: CV is `RepeatedStratifiedKFold`, *not speaker-grouped*. Its topic-held-out stack scores 0.488. |
| [sim2126/SHL-Kaggle](https://github.com/sim2126/SHL-Kaggle) (0.3316 on 3 Oct) | A LoRA-tuned Voxtral-Mini regressor, Voxtral-Small-24B, and real-audio-token pooling were each **tried and dropped on CV**. This independently matches our Voxtral QLoRA failure. |

**What #1 and #2 are probably doing (inference, weak evidence).** None of the public material explains a score near 0.29. The only documented route above 0.31 is the one we already use: meta-learners and mix weights selected with public feedback. In our own log that route was worth −0.0116 (0.3178 → 0.3062). A further −0.015 to −0.02 within a day or two on about 130 public clips is far more consistent with more aggressive public-fitted mixing than with a new model. If that is right, the private split should pull these scores back toward the honest models' range (about 0.315–0.33). Treat this as a hypothesis: #1 (0.2913) has no public write-up.

---

## 3. New ideas, ranked

Gains are in the leaderboard metric. A rule of thumb for converting: −0.01 RMSE ≈ −0.006 metric, and +0.01 Pearson ≈ −0.016 metric once calibrated.

### 1. Pool predictions across a speaker's own test clips (a correction step; CPU only) — evidence B, near-certain data fact

- **The data fact.** Same-speaker clips in train score almost identically:
  - 66% of same-speaker pairs are identical, with a mean |Δ| of 0.25, against 1.15 for unrelated pairs ([nakul notebook §6.2](https://github.com/nakulgrover118/grammar-scoring-engine)).
  - Spread is 0.21 within a speaker against 1.01 overall ([AMITABHSHUKLA](https://github.com/AMITABHSHUKLA/spoken-grammar-scoring)).
  - Test speakers are not in train (ECAPA2 test→train max cosine 0.78), but **55 of 216 test clips belong to speakers with 2–3 test clips**.
- **The method.** Move each prediction a fraction λ toward the mean prediction of its speaker's clips in the set being scored: `p + λ·(mean_spk(p) − p)`. No labels are involved.
- **Their CV.** On multi-clip speakers, RMSE went 0.5043 (λ=0) → 0.4981 (0.25) → **0.4972 (0.5)** → 0.4992 (0.65) → 0.5112 (1.0). It helps 3-clip speakers, leaves pairs about unchanged, and leaves singletons untouched.
- **Our forecast.** About 25% of test clips are affected, so roughly −0.002 RMSE plus a small Pearson gain: **about −0.001 to −0.003 metric**.
- **Why it fits our rules.** It encodes a fact about how the data were generated (handbook: "post-processing that encodes a fact … transfers"). It can be validated exactly on our OOF, because speaker-grouped folds keep a speaker's clips together.
  - Pick λ on fold seeds s0–s2.
  - Report the bootstrap with the λ search nested.
- **Licence trap.** The competitor's ECAPA2 is **CC-BY-NC-4.0**, which is not allowed. Use the WavLM-base-plus-SV model we already have, or `speechbrain/spkrec-ecapa-voxceleb` (Apache-2.0). Use a tight same-speaker threshold, not the loose 0.12 fold-clustering distance, and check it on train pairs first.
- **Do not copy** the "similar-voice adjustment". Its author shows errors of different speakers are uncorrelated (−0.04), against +0.58 for the same speaker, so the adjustment is noise.

### 2. Fine-tune a speech encoder end-to-end with learned layer weights and attention pooling (a training change; GPU) — evidence A for the technique, C at our data size

- **Literature:**
  - Fine-tuned wav2vec2 beat BERT graders on small L2 sets. On TLT-school (B1 holistic), PCC was **0.927 vs 0.857** for BERT on manual transcripts, with 12 epochs, lr 5e-5, batch 4×2, frozen CNN. It also won on the "formal correctness" sub-score. ICNALE (3,898 training answers): accuracy 77.9% vs 53.5% ([Bannò & Matassoni, SLT 2022](https://arxiv.org/abs/2210.13168)).
  - Speak & Improve 2025 (244 h of training audio): a fine-tuned wav2vec2 grader got RMSE **0.394** and a fine-tuned Whisper-small encoder got **0.384**, against 0.440–0.445 for the BERT baseline ([NTNU](https://arxiv.org/abs/2506.05121), [Aalto](https://arxiv.org/abs/2507.17918), [challenge overview](https://www.isca-archive.org/slate_2025/qian25_slate.pdf)).
  - The S&I winner adapted Whisper-medium.en with lightweight trainable modules (adapters) rather than full fine-tuning.
- **Against:**
  - Our data is 10–300× smaller than these sets.
  - LoRA on audio LLMs failed for us (Voxtral QLoRA) and for sim2126.
  - A Qwen2.5-3B LoRA regressor was "better OOF, worse public" for PRINCE2.
- **Recipe sketch.** WavLM-large (MIT, already licensed in our pipeline) or w2v-BERT 2.0 (MIT):
  - frozen CNN;
  - learned softmax over layers;
  - our attention-pooling head;
  - a fresh random test-length crop each epoch, as in `20_attn_pool.py`;
  - LLRD, about 8–12 epochs, 3 seeds;
  - judged as a *new member* by its correlation with the blend's residuals (the head's 0.078 was what made it useful).
- **Start cheap.** Run the existing `07_finetune_audio.py` once on a free T4 and log the result. That closes the unlogged gap above.
- **Expected gain:** 0 to −0.006. GPU: about 1–2 T4-hours per seed for 5 folds at 15–30 s crops in fp16 (our estimate).

### 3. A correlation-aware loss for the trainable models (attention head, DeBERTa fine-tune) — evidence B

- **Literature.** Adding a batch-wise ranking (ListNet) loss to the MSE regression loss raised ASAP average QWK from **0.768 (regression only) to 0.794** (R2BERT, [Yang et al., Findings EMNLP 2020](https://web.comp.polyu.edu.hk/labimcl/paper/Yang-2020-FindingsEMNLP-AES.pdf)). Ranking only scored 0.756.
- **Why it fits us.** Our metric puts 0.4 weight on 1 − Pearson, and our models are compressed toward the mean (README §12, "Extremes").
- **Variants:** a batch `1 − corr(pred, y)` term, or a pairwise margin-ranking term, added to MSE with weight 0.1–0.5. It is a one-flag change in `20_attn_pool.py`, so it trains in minutes.
- **Validate on 3 retrains, not 1.** On 18 Oct, a CPU retrain alone gave P = 0.92, so a single retrain cannot show a real gain.
- **Expected gain:** 0 to −0.003.

### 4. Parakeet-TDT (CC-BY-4.0) as a new source of frames for the attention head, or of verbatim transcripts — evidence B/C (risky under our composition rule)

- **For:**
  - zeelrathi19: the Parakeet-TDT-0.6B encoder (layers 6+18) was the strongest single block. Removing it costs 0.035 RMSE in their stack, and it is "least hurt when whole prompt topics are held out". Their CV is *not* speaker-grouped, so the absolute numbers are inflated.
  - The S&I 2025 open-track winner moved from 0.368 to **0.364** by training its BERT grader on more verbatim transcripts from **parakeet-tdt-1.1b**, enhanced with phoneme alignments to capture disfluencies and fillers ([overview](https://www.isca-archive.org/slate_2025/qian25_slate.pdf)). Note that this winner fine-tuned Parakeet on S&I data, which we cannot do.
- **Against.** Attention heads on Phi-4 and WavLM-large frames made our blend worse, and Phi-4 transcripts correlated 0.95 with Whisper's.
- **Rule status.** This is a new-signal member. That kind of change transferred once for us (Voxtral) and failed often. Submit only if the OOF gain comes with a residual correlation at or above the head's 0.078.
- **Expected gain:** 0 to −0.004. GPU: about 1 T4-hour to cache frames.

### 5. C-Mixup on the attention head, pairing clips with similar labels across topic clusters and recording batches — evidence B (general), C (speech)

- **C-Mixup** samples mixing partners by label similarity. Across 11 regression datasets it beats the best prior method by 6.56% in-distribution, 4.76% on task generalisation and 5.82% on OOD robustness. It "mitigates the effects of domain-associated information" ([Yao et al., NeurIPS 2022](https://arxiv.org/abs/2210.05775)). That targets our main shift: new prompts and the 45.06 s batch.
- **Speech evidence is weak.** Aalto's S&I "swap sampling" (building samples from different speakers' parts, picked within 0.5 of the score) left RMSE unchanged (0.383 vs 0.384) and helped only the edge levels ([Aalto](https://arxiv.org/abs/2507.17918)).
- **How to apply it.** Mix after pooling (manifold mixup) inside `20_attn_pool.py`. Cost: minutes.
- **Expected gain:** 0 to −0.003.

### 6. Speaker-aware target denoising in training — evidence C

- **The method.** Train on `(1−β)·y + β·mean_speaker(y)` for multi-clip speakers. Score OOF against the raw labels.
- **Why it might work.** If the 0.25–0.35 within-speaker spread is mostly rater noise, the targets get cleaner without losing any signal.
- **Why it might not.** Test labels are noisy too, so the gain is bounded by how much variance the noisier targets add to the fit.
- **Validation.** Folds already keep speakers together, so leave-speaker-out OOF validates it with no leakage.
- **Expected gain:** 0 to −0.002. Cost: minutes.

### 7. The remaining handbook procedure hacks for the DeBERTa fine-tune: AWP from epoch 2, backbone dropout 0, 3-seed self-distillation from its own OOF — evidence A in written-text competitions, neutral on small regression sets

- See handbook Ch. 4.
- The fine-tune's 9-seed average lost publicly (0.3196 vs 0.3175). Any change here must be compared on 3-seed averages on both sides.
- **Expected gain:** 0 to −0.002. GPU: about 1–2 T4-hours per variant.

### 8. An ordinal or distributional head on the 0.5 grid with expected-value decoding (CORN, or Gaussian soft labels) — evidence C for our metric

- An S&I closed-track team used CORN loss with posterior calibration and augmentation and reached 0.411 against the 0.440 baseline. The effect of CORN alone is not separated out ([overview](https://www.isca-archive.org/slate_2025/qian25_slate.pdf)).
- An ordinal margin loss helped F1 on unseen prompts (TEEMI, ~8.2k responses: +1.02% overall F1 and +4.42% on language use; [arXiv 2509.03372](https://arxiv.org/abs/2509.03372)). Those are classification metrics, not RMSE.
- Our labels are probably rater averages (rounding hurt by 0.02), so hard classes are wrong. Soft labels with expected-value decoding are the only safe form.
- **Expected gain:** about 0.

### Considered and set aside

- **Learned hurdle / soft zero gate** (prblydv, 0.3166 → 0.3141 with public-chosen weights). Our gate already separates all 37 noise clips perfectly, and the hurdle's development score was worse.
- **Topic-adversarial (DANN-style) heads.** These come from the cross-prompt essay-scoring literature ([PAES](https://arxiv.org/abs/2008.01441), [ACL 2023 prompt-aware domain generalisation](https://aclanthology.org/2023.acl-long.696.pdf), [GAPS 2025](https://arxiv.org/abs/2502.08450)), with no speech evidence. Our topic-direction removal already failed.
- **Prototype or metric-based classification for imbalance** ([Lo et al., NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.86.pdf)). It is a CEFR classification setting.
- **LLM-generated responses + TTS augmentation** ([arXiv 2506.04077](https://arxiv.org/abs/2506.04077)). This is synthetic external data, so it is **probably against the no-external-data rule**.
- **Phi-4-MM LoRA graders** (NTNU, 0.389 on 244 h). This is the same family that failed for us at 769 clips.
- **AST spectrogram branch** (prblydv). AudioSet sound-event features have no reported standalone value for grammar. Its use there was public-informed.

---

## 4. Domain shift: new prompts and the 45.06 s batch

- **Every source agrees the batch is the hard part:**
  - It makes up 47% of test against 14% of train (AMITABHSHUKLA, PRINCE2).
  - It is over-predicted by 0.12–0.18 in CV (zeelrathi19).
  - Its within-batch correlation is lower: 0.74 against 0.86 elsewhere (ours).
- **Fixes that worked for others** (a separate stacker, an is45 flag) are already in our pipeline as batch weights. Fixes that failed for us (a batch offset) failed for PRINCE2 as well.
- **Literature.** Acoustic traits ("delivery") transfer best to unseen tasks, and content and language-use aspects transfer worst ([arXiv 2508.12591](https://arxiv.org/abs/2508.12591), TEEMI). This fits our finding that the WavLM SVR collapses on the batch while text models and the Voxtral ridge hold up. The new levers are therefore:
  - training-time changes that remove domain cues: ideas 3 and 5;
  - test-time pooling of evidence within a speaker: idea 1.
  More feature views are not one of them.

---

## Sources

Competitor write-ups (self-reported, unverified):
- [prblydv](https://github.com/prblydv/shl-grammar-scoring)
- [AMITABHSHUKLA](https://github.com/AMITABHSHUKLA/spoken-grammar-scoring)
- [nakulgrover118](https://github.com/nakulgrover118/grammar-scoring-engine)
- [PRINCE2-AI](https://github.com/PRINCE2-AI/shl-grammar-scoring-engine)
- [zeelrathi19](https://github.com/zeelrathi19/shl-grammar-scoring)
- [sim2126](https://github.com/sim2126/SHL-Kaggle)
- [devyanshbatra](https://github.com/devyanshbatra/shl-grammar-scoring-2026)
- [starkhushi](https://github.com/starkhushi/shl-2026-grammar-scoring)

Papers:
- [Bannò & Matassoni 2022](https://arxiv.org/abs/2210.13168)
- [Bannò et al. 2022, Linguaskill](https://arxiv.org/abs/2211.08849)
- [S&I 2025 overview](https://www.isca-archive.org/slate_2025/qian25_slate.pdf)
- [S&I 2025 tasks and baselines](https://arxiv.org/abs/2412.11985)
- [NTNU S&I](https://arxiv.org/abs/2506.05121)
- [Aalto S&I](https://arxiv.org/abs/2507.17918)
- [R2BERT](https://web.comp.polyu.edu.hk/labimcl/paper/Yang-2020-FindingsEMNLP-AES.pdf)
- [C-Mixup](https://arxiv.org/abs/2210.05775)
- [MMO ordinal ASA](https://arxiv.org/abs/2509.03372)
- [Lo et al. 2024](https://aclanthology.org/2024.findings-naacl.86.pdf)
- [Unified MLLM ASA](https://arxiv.org/abs/2508.12591)
- [Kim et al. 2022](https://arxiv.org/abs/2204.03863)

Licences, from the Hugging Face API on 19 Oct 2026:
- `nvidia/parakeet-tdt-0.6b-v2` and `nvidia/parakeet-tdt-1.1b`: CC-BY-4.0
- `Jenthe/ECAPA2`: **CC-BY-NC-4.0 (not allowed)**
- `speechbrain/spkrec-ecapa-voxceleb`: Apache-2.0
- `facebook/w2v-bert-2.0`: MIT
- `MIT/ast-finetuned-audioset-10-10-0.4593`: BSD-3
- `openai/whisper-medium.en`: Apache-2.0
