# Holistic graders: fusion, pooling, speech LLMs, losses, label noise (with the Speak & Improve 2025 challenge as the benchmark)

Part of the paper notes for [[11 Speech Assessment]]. The evidence tags ([A], [B], [C]) are defined in `01 Speech encoders and fine-tuning.md`.

**Why S&I 2025 matters here.** It is the only recent public benchmark with:
- spontaneous L2 English answers up to one minute, scored holistically;
- many teams on one held-out evaluation set;
- a closed track ("only the released data").

It is the nearest thing to a controlled comparison for our task. Its corpus licence is **non-commercial research only** ([corpus paper](https://www.isca-archive.org/slate_2025/knill25_slate.pdf)), so it is useless to us as data. It is useful only as evidence.

---

## A. The benchmark

### Qian et al.: S&I Challenge 2025 tasks and baselines (arXiv 2412.11985; SLaTE 2025) [A]
- [arXiv 2412.11985](https://arxiv.org/abs/2412.11985).
- **Data:**
  - Train: 6,640 test submissions, about 244 h.
  - Dev: 438 submissions. Eval: 442.
  - Four open-speaking parts per submission. Parts 3 and 4 are one-minute long turns.
  - Part scores run 2 (A2) to 5.5 (C1+) in half points. The overall score is their average.
- **Baseline:**
  - Whisper-small ASR, then BERT-base word embeddings, 4 self-attention heads and a regression head.
  - One grader per part.
  - Dev RMSE 0.445, PCC 0.750. Eval RMSE 0.440, PCC 0.726.

### Final eval leaderboard, open track (as printed in the NTNU paper's Table 2)

| Rank | Team | RMSE | PCC |
|---|---|---|---|
| 1 | Perezoso (Duolingo) | 0.364 | 0.826 |
| 2 | NTNU SMIL V | 0.375 | 0.820 |
| 3 | nhanphan (Aalto) | 0.384 | 0.801 |
| 25 | Baseline | 0.440 | 0.726 |

---

## B. The top systems

### 1. Cai, Madnani & Yancey (Duolingo): Team Perezoso, 1st place (SLaTE 2025) [B]
- [ISCA PDF](https://www.isca-archive.org/slate_2025/cai25_slate.pdf).
- **Speech grader, almost a pure "frozen encoder + small adapter head" design:**
  - Whisper **medium.en encoder, frozen**.
  - The **last six layers** each go through their own small MLP adapter (1024 → 128, two linear layers with layer norm and a nonlinearity).
  - The six adapter outputs are concatenated, projected to 128, then pooled with **mean + standard-deviation pooling**, then a linear output.
  - About 1M trainable parameters, MSE loss, Adam lr 1e-3 with cosine decay, 10 epochs.
  - Audio augmented with MUSAN noise and room impulse responses.
- **Text grader:** the same adapter recipe on the last 8 layers of a frozen BERT-base (1.3M trainable parameters).
- **Final model:** a linear regression over 11 inputs:
  - the speech grade and two BERT grades;
  - ASR confidence (the geometric mean of Whisper token posteriors);
  - words per second, duration;
  - counts of characters, words and sentences;
  - mean word length, mean sentence length.
- **Results on the dev set** (RMSE / PCC):

  | Component | RMSE | PCC |
  |---|---|---|
  | Whisper speech grader alone | 0.363 | 0.839 |
  | Their BERT grader | 0.396 | 0.805 |
  | Official BERT baseline | 0.443 | 0.747 |
  | ASR confidence alone | 0.451 | 0.737 |
  | Number of words alone | 0.492 | 0.676 |
  | Final linear stack | 0.349 | 0.853 |

  Eval: 0.364 RMSE, 1st.
- **Caveat:** no ablations. We do not know what the multi-layer adapters or the noise augmentation contributed.
- **Transfers:**
  - The winning *speech* model never fine-tuned the encoder. It learned a per-layer adapter head on frozen frames, which is close to our attention head.
  - The differences from ours: several layers kept separate with learned adapters (ours: one PCA-256 of averaged lower-middle Voxtral layers), mean + std pooling, and noise augmentation.
  - MUSAN is external data, so it is not allowed for us.

### 2. Lin et al. (NTNU): wav2vec2 + Phi-4-multimodal fusion, 2nd place (SLaTE 2025) [B]
- [arXiv 2506.05121](https://arxiv.org/abs/2506.05121).
- **Speech grader:**
  - wav2vec2-base, fine-tuned for 30 epochs at lr 1e-4.
  - **Attention pooling** over frames, plus a "prototype" module: one learnable vector per CEFR level, initialised at the class mean, with cosine similarities fed to the head.
- **LLM grader:** Phi-4-multimodal fine-tuned with LoRA (rank 320) on audio + ASR transcript + question text, per part.
- **Results** (eval RMSE / PCC):

  | System | RMSE | PCC |
  |---|---|---|
  | BERT (their replication) | 0.445 | 0.727 |
  | wav2vec2 grader | 0.394 | 0.790 |
  | Phi-4 one model for all parts, audio + text | 0.412 | 0.796 |
  | Phi-4 audio only | 0.428 | 0.793 |
  | Phi-4 text only | 0.463 | 0.759 |
  | Phi-4 per part | 0.389 | 0.808 |
  | **wav2vec2 + Phi-4 per part** | **0.375** | **0.820** |

  The fusion weight was grid-searched on dev separately for each of 8 score intervals. That is risky on small data.
- **Transfers:** audio input beats text-only input even inside an LLM (0.428 against 0.463 RMSE), and the speech encoder and the LLM are complementary. Our Voxtral + WavLM blend already exploits both. A prototype-similarity layer is cheap, but it has no ablation.

### 3. Phan et al. (Aalto): "One Whisper to Grade Them All", 3rd place (SLaTE 2025) [B]
- [ISCA PDF](https://www.isca-archive.org/slate_2025/phan25_slate.pdf).
- **Method:**
  - One Whisper-small encoder over all four parts, cut into 30 s chunks and time-averaged per chunk.
  - The aggregator is either a mean (AVG) or a 2-layer transformer (TF).
  - Loss: RMSE (MSE underflowed in fp16). lr 5e-5, 15 epochs. The paper does not say clearly whether the encoder weights were updated.
- **Data tricks:**
  - "Swap sampling": build new samples from parts of different speakers whose part scores are within 0.5 of the target.
  - "OE": oversample the rare edges, scores ≤ 2.75 or ≥ 4.75.
- **Results** (eval RMSE; RMSE_e = on the 7% edge cases):

  | Model | RMSE | RMSE_e |
  |---|---|---|
  | AVG | 0.384 | 0.698 |
  | AVG + 4× swap sampling | 0.383 | 0.717 |
  | AVG + edge oversampling | 0.387 | **0.613** |
  | TF | 0.372 | 0.637 |
  | TF + edge oversampling, OE dev set | 0.383 | **0.597** |

  - TF was best but unstable across epochs, so AVG was submitted.
  - A content check showed the AVG grader ignores content: swapping parts changed predictions by 0.000.
- **Transfers:**
  1. Edge oversampling cuts edge error by about 12% at roughly zero overall cost. Pearson rewards the edges, so this matters for our metric.
  2. More synthetic recombinations did not help.
  3. Simple averaging is more stable across checkpoints than a transformer aggregator, which is useful when you cannot see the test set.

### 4. Porwal et al. (Aalto): ordinal classification, closed track (SLaTE 2025) [A for the loss, B for the rest]
- [ISCA PDF](https://www.isca-archive.org/slate_2025/porwal25_slate.pdf).
- **Method:** the same architecture as the BERT baseline. Only the objective and the data change.
- **Results:**

  | Variant | Dev RMSE / PCC | **Eval** RMSE / PCC |
  |---|---|---|
  | Regression baseline | 0.445 / 0.746 | 0.440 / 0.726 |
  | Cross-entropy classes | 0.419 / 0.801 | 0.468 / 0.731 |
  | **CORN ordinal loss** | 0.402 / 0.807 | **0.411 / 0.782** |
  | CORN + BERT-MLM text augmentation | 0.392 / 0.813 | 0.419 / 0.765 |
  | CORN + sentence-swapping | 0.392 / 0.816 | 0.422 / 0.768 |
  | CORN + part-wise best MLM | 0.383 / 0.819 | 0.417 / 0.771 |
  | CORN + per-class thresholds tuned on dev | 0.364 / 0.839 | 0.423 / 0.766 |

- **Lessons:**
  - CORN is the only change that held on eval: RMSE -0.029, PCC +0.056 over regression.
  - Every augmentation and the dev-tuned thresholds improved dev and *worsened* eval relative to plain CORN.
  - Plain cross-entropy was worse than regression on eval.
- **Transfers:** this is the same pattern our log shows (a dev gain that does not transfer), seen independently on a 442-submission eval set. A change to the *loss* transferred; data and threshold tweaks chosen on dev did not.

### 5. Wang et al. (NTNU): LLM + voice-cloning data augmentation (SLaTE 2025) [C]
- [ISCA PDF](https://www.isca-archive.org/slate_2025/wang25_slate.pdf).
- **Data:** GEPT opinion task, 1,200 responses over 4 prompt types. One type (300) is held out as unseen. The rest is split 8:1:1, so the seen test has 90 responses.
- **Method:**
  - An LLM writes level-conditioned answers.
  - A multi-speaker TTS voices them, using speaker embeddings from real learners.
  - A "dynamic importance" loss reweights synthetic examples.
  - Phi-4-multimodal with LoRA does the scoring.
- **Results:**

  | | Seen accuracy | Unseen accuracy |
  |---|---|---|
  | Phi-4 trained on real data only | 73.33 | 63.33 |
  | Plus synthetic data | 76.67 | 64.00 |

  The unseen gain is 0.67 points, about 2 responses out of 300.
- **Transfers:** none. Synthetic data from an external LLM and TTS conflicts with our no-external-data rule, and the gain on new prompts is nil.

---

## C. Speech LLMs as graders

### 6. Ma, Qian, Tang, Bannò, Knill & Gales: L2 oral proficiency with speech LLMs (Interspeech 2025) [A]
- [arXiv 2505.21148](https://arxiv.org/abs/2505.21148).
- **Data:**
  - Linguaskill: 31,476 training submissions (3,465 h). Tests LinGen (1,049) and LinBus (712).
  - S&I: 6,640 training submissions.
  - Scores 1-6 with half points.
- **Model:**
  - Qwen2-Audio-7B-Instruct with LoRA (rank 16) in both encoder and decoder (10M parameters), 2 epochs, lr 1e-4.
  - The one-minute parts are split into two 30 s chunks and their scores averaged.
  - Calibrated linearly on dev.
- **Loss comparison** (LinGen part 5, RMSE / PCC):

  | Loss and decoding | RMSE | PCC |
  |---|---|---|
  | Zero-shot, hard | 1.227 | 0.178 |
  | Zero-shot, soft | 1.147 | 0.371 |
  | Cross-entropy, hard | 0.615 | 0.879 |
  | Cross-entropy, soft | 0.563 | 0.890 |
  | **"Fair average" (FA), soft** | **0.559** | **0.892** |
  | Regression head | 0.565 | 0.890 |

  FA means: softmax over the grade letters, take the expected score Σ p_c·score(c), and train with MSE on that expectation. CE and FA "converge faster than the regression model".
- **Submission level, LinGen** (RMSE / PCC):

  | System | RMSE | PCC |
  |---|---|---|
  | wav2vec2 with attention pooling | 0.383 | 0.934 |
  | BERT | 0.356 | 0.942 |
  | Qwen2-Audio zero-shot | 0.807 | 0.640 |
  | Qwen2-Audio FA | 0.323 | 0.954 |

  Trained on S&I and tested on S&I dev: PCC 0.833 against 0.753 for BERT.
- **Transfers:**
  - Fine-tuned speech LLMs are state of the art at 6.6k-31k submissions. Zero-shot they are poor, which matches our Voxtral-asked-for-the-score result.
  - With 769 clips, LoRA on a 7B audio LLM is far outside the regime shown here (our Voxtral QLoRA folds were 0.764 and 0.789).
  - The **FA loss is the transferable piece**. It is a distribution-aware output that still gives continuous predictions, so it is compatible with the finding that rounding hurts. Note the gain over a regression head here is small: 0.006 RMSE.

### 7. Bannò, Ma, Qian, Tang, Knill & Gales: natural-language-based assessment (SLaTE 2025) [B]
- [arXiv 2507.10200](https://arxiv.org/abs/2507.10200).
- **Method:** zero-shot Qwen2.5-72B (4-bit) reads Whisper-small transcripts plus CEFR *analytic* can-do descriptors (10 aspects, including grammatical accuracy). Option order is shuffled to fight position bias, and probability-weighted scores are averaged.
- **Results** (PCC, S&I dev / eval):

  | System | Dev | Eval |
  |---|---|---|
  | BERT fine-tuned | 0.753 | 0.727 |
  | Zero-shot LLM + descriptors | 0.806 | 0.761 |
  | Qwen2-Audio trained on read-aloud | 0.740 | 0.750 |
  | Qwen2-Audio trained on S&I spontaneous speech | 0.833 | 0.821 |

  The text systems drop more from dev to eval.
- **Transfers:** a zero-shot rubric LLM on transcripts can beat a small fine-tuned text model. That fits our LLM judge (r = 0.62 alone). It still adds nothing when the ensemble already has a strong text model. The grammatical-accuracy descriptor was among the most position-sensitive.

---

## D. Pooling long answers and fusing views

### 8. McKnight et al.: conversational test auto-marking (SLaTE 2023) [C, skimmed]
- [ISCA PDF](https://www.isca-archive.org/slate_2023/mcknight23_slate.pdf).
- **Design:** the wav2vec2 grader replaces mean pooling with four intra-segment attention heads, then four cross-segment attention heads over multiple audio segments. It is pre-trained on Linguaskill, then fine-tuned on the Occupational English Test. A Longformer text ensemble is added and the two are combined.
- **Caveat:** no pooling ablation is reported.
- **Transfers:** hierarchical attention (within segment, then across segments) is the Cambridge design for long answers. Our attention head is the single-level version.

### 9. Fusion results already covered elsewhere
- Bannò & Matassoni 2022 (note 01): shallow average 0.885 against deep fusion 0.883 PCC.
- Bannò 2023 (note 01): a per-part OLS combination helps; a submission-level average does not.
- NTNU (above): score fusion of an encoder and an LLM.

In every case, late (score-level) fusion is as good as or better than deep fusion at these data sizes. That is consistent with our transcript-conditioned head being worse (+0.0034).

---

## E. Multi-task, ordinal and noise-aware training

### 10. Gong et al.: GOPT multi-aspect, multi-granularity pronunciation (ICASSP 2022) [A]
- [arXiv 2205.03432](https://arxiv.org/abs/2205.03432).
- **Data:** Speechocean762, 5,000 utterances, half for training.
- **Ablation** (utterance total-score PCC):

  | Variant | PCC |
  |---|---|
  | Utterance-only training | 0.736 ± 0.011 |
  | Joint phoneme + word + utterance multi-task | 0.742 ± 0.005 |
  | Embedding size 24 (27K parameters) | 0.742 |
  | Embedding size 96 (355K parameters) | 0.725 |
  | 3 transformer layers | 0.742 |
  | 6 transformer layers | 0.731 |

- **Transfers:**
  - Multi-task auxiliary targets moved the utterance score by about half a standard deviation of seed noise. That is weak.
  - **Tiny heads win at a few thousand items.** Keep our heads small.

### 11. Craighead et al. 2020: auxiliary objectives (note 02) [A]
- With BERT, only L1 prediction helped (0.921 → 0.908 RMSE). POS, UD and LM auxiliary tasks gave nothing.

### 12. Saeki et al.: UTMOS, VoiceMOS 2022 winner (Interspeech 2022) [A]
- [arXiv 2204.02152](https://arxiv.org/abs/2204.02152).
- **Data:**
  - Main track: 4,974 labelled training clips.
  - Out-of-domain (OOD) track: only 136 labelled clips plus 540 unlabelled.
- **Method:** a fine-tuned wav2vec2 with a BLSTM giving **frame-level scores averaged at inference**. Loss = clipped MSE + **pairwise contrastive (margin ranking) loss on score differences**, plus a listener-ID embedding, a phoneme encoder, and speed/pitch augmentation. The final model is a stack of strong and weak learners (ridge, SVR, GP on SSL means).
- **Ablation** (utterance-level MSE / LCC; the full system first):

  | Variant | Main track | OOD track |
  |---|---|---|
  | Full system | 0.276 / 0.883 | 0.378 / 0.891 |
  | Without contrastive loss | 0.241 / 0.881 | 0.407 / 0.870 |
  | Without listener ID | 0.307 / 0.880 | 0.636 / 0.847 |
  | **Without augmentation** | **0.226 / 0.885** | **0.322 / 0.887** |

  Augmentation made MSE worse on both tracks. The contrastive loss helped correlation only in the tiny OOD track.
- **Transfers:**
  - A ranking term is cheap, aimed at correlation, and weakly evidenced.
  - Speed and pitch augmentation is not a free win for regression error.
  - Modelling rater identity was the strongest component. We have no rater IDs.

### 13. Shi, Cao & Raschka: CORN rank-consistent ordinal regression (arXiv 2021, revised 2023) [A, skimmed]
- [arXiv 2111.08851](https://arxiv.org/abs/2111.08851).
- **Method:** K-1 conditional binary tasks P(y > k | y > k-1), trained on conditional subsets. Rank consistency comes from the chain rule, with no weight sharing, and it beats CORAL.
- **Transfers:** a drop-in output layer for any head. Decode by the expected value, Σ_k P(y > k) × step + minimum, *not* by thresholding, because rounding hurts our metric.

### 14. Lo, Chen, Sung & Chen: multi-margin ordinal loss and non-uniform intervals (arXiv 2025) [C]
- [arXiv 2509.03372](https://arxiv.org/abs/2509.03372).
- **Data:** TEEMI corpus, 8,214 responses.
- **Results:** cross-entropy + multi-margin ordinal loss gave holistic macro-F1 35.55%, and +1.02 F1 on an unseen prompt.
- **Transfers:** weak. Classification metrics at about 35-39% accuracy, no RMSE, no seeds.

### 15. Malinin, Ragni, Knill & Gales: uncertainty in spoken language assessment (ACL 2017) [B, skimmed]
- [ACL Anthology](https://aclanthology.org/P17-2008).
- **Method:** a Deep Density Network predicts a mean and a variance per candidate. It is trained to give high variance on artificial out-of-distribution inputs, so uncertain candidates can be sent to humans. On BULATS it beats GP and MC-dropout at rejection while grading equally well.
- **Transfers:** a heteroscedastic (Gaussian NLL) head is the principled way to soften noisy or atypical clips during training. There is no direct evidence that it improves RMSE.

---

## Summary for SHL
1. **The winning 2025 speech grader was a frozen ASR encoder with a ~1M-parameter multi-layer adapter head** (Perezoso). It was not fine-tuned and not an LLM.
2. **The only change that transferred from dev to the S&I eval set was a loss change** (CORN). Data augmentation, thresholds and score-interval fusion weights tuned on dev did not transfer. This mirrors our own "procedure changes transfer" rule, with an extra nuance: a procedure change tuned on dev can still overfit dev.
3. **Distribution-aware outputs (CORN with expected-value decoding, or "fair average") are the cleanest untried training change for our trainable heads.** The evidence: -0.029 RMSE / +0.056 PCC (Porwal, 6.6k submissions) and -0.006 RMSE (Ma, 31k).
4. **Edge oversampling fixes the extremes** at no overall cost (Phan: edge RMSE 0.698 → 0.613).
5. **Speech LLM fine-tuning is SOTA only at 6.6k-31k submissions.** Zero-shot it is weak (PCC 0.64).
6. **Keep heads small and fuse late** (GOPT, Bannò 2022, our transcript-conditioned head).
