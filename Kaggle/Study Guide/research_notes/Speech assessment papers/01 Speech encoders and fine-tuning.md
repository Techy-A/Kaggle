# Speech encoders for scoring: what the layers hold, frozen vs fine-tuned, small-data recipes

Part of the paper notes for [[11 Speech Assessment]]. Each entry gives the task and data (with sizes), the method, the exact gain as printed, and what transfers to SHL (769 labelled clips of 45-60 s, about 11.7 hours; new questions in test).

**Method.** I downloaded every paper below as a PDF (arXiv, ISCA Archive, ACL Anthology, ERIC) and read the method and results sections. Numbers are copied from the tables. If I only skimmed a paper, its entry says so.

**Evidence tags:**
- **[A]** a controlled ablation in the paper, on data of at least a few thousand items, or replicated by an independent paper;
- **[B]** a single paper with a clear comparison, but a small test set, a weak baseline or no seeds;
- **[C]** stated without an ablation, or the comparison is confounded.

---

## 1. SUPERB: the frozen, weighted-layer-sum protocol (Yang et al., Interspeech 2021) [A]
- [arXiv 2105.01051](https://arxiv.org/abs/2105.01051).
- **Task:** a benchmark of 10 speech tasks with one frozen, shared SSL upstream and light task heads.
- **Method:** the head reads a learnable softmax-weighted sum of all hidden layers. Upstream hyperparameters are not tuned.
- **Finding:** frozen representations with a weighted layer sum are competitive across tasks. This is the standard "frozen" baseline the rest of this note compares against.
- **Transfers:** the protocol itself. A learned layer weighting is a cheap way to avoid hand-picking layers. Our pipeline averages fixed layer ranges (for example WavLM-large layers 8-16) instead.

## 2. WavLM (Chen et al., IEEE JSTSP 2022) [A, skimmed]
- [arXiv 2110.13900](https://arxiv.org/abs/2110.13900).
- **Finding:** the learned SUPERB layer weights show that bottom layers matter for speaker tasks and top layers for content and ASR. Denoising and overlap simulation in pre-training mostly helps speaker diarisation.
- **Transfers:** speaker identity sits in the lower layers. That is consistent with our speaker-twin leakage being strongest for audio models. It also means lower layers are where a model can memorise voices.

## 3. Pasad, Chou & Livescu: layer-wise analysis of wav2vec 2.0 (ASRU 2021) [A]
- [arXiv 2107.04734](https://arxiv.org/abs/2107.04734).
- **Finding:** wav2vec 2.0 behaves like an autoencoder. The middle layers hold the most phonetic and word information. The last 1-2 layers drift back toward the input and change the most during fine-tuning.
- **Gain from re-initialising the top layers before ASR fine-tuning** (WER, base model, best n chosen on dev-clean):

  | Training data | Layers re-initialised | test-clean | test-other |
  |---|---|---|---|
  | 10 min | 3 | 49.0 → 44.1 | 56.7 → 51.8 |
  | 1 h | 1 | 20.3 → 19.8 | 29.8 → 29.3 |
  | 10 h | 1 | 11.3 → 10.9 | 20.6 → 19.4 |

  The gain is large at 10 minutes and minor at 1-10 h.
- **Transfers:** if we ever fine-tune a wav2vec2-style encoder, dropping or re-initialising its top layers is a cheap recipe step. For frozen features, the top layers of wav2vec2-style models are the worst place to read from.

## 4. Pasad, Shi & Livescu: comparative layer-wise analysis of 11 SSL models (ICASSP 2023) [A]
- [arXiv 2211.03929](https://arxiv.org/abs/2211.03929).
- **Findings:**
  - Models trained to predict cluster IDs (HuBERT, WavLM) put phonetic and word content in their higher layers.
  - wav2vec2-style models peak in the middle layers.
  - CCA scores against phone and word labels predict which single layer is best for ASR and spoken-language-understanding tasks.
  - The best single layer often matches or beats a weighted sum of all layers.
- **Transfers:** this explains our log entry where WavLM-large layer 20 alone beat the 8-16 average (SVR 0.585 against 0.597). The upper WavLM layers are content layers, so they overlap with Voxtral. That is why layer 20 then hurt in the blend while the acoustic middle layers added information.

## 5. Chen, Fu, Liu, Li & Lee: efficient tuning of SSL speech models (SLT 2022) [A]
- [arXiv 2210.06175](https://arxiv.org/abs/2210.06175).
- **Task and data:** SUPERB tasks. A low-resource study on 1 h, 10 h and 100 h subsets of Libri-Light for ASR and phone recognition.
- **Findings:**
  - Houlsby adapters reach parity with full fine-tuning using more than 90% fewer trainable parameters. The weighted sum is "a powerful and efficient method".
  - At 100 h, full fine-tuning beats adapters for phone recognition. At 10 h and 1 h, adapters win.
  - Adapters are more stable across seeds and tolerate a much wider learning-rate range: fine-tuning "does not work on larger learning rates".
  - LoRA "performs worst among efficient methods and even worse than Baseline in some tasks" in their speech setting.
- **Transfers:** our audio is about 11.7 h, inside the regime where adapters beat fine-tuning. But those tasks have dense frame labels, and we have one scalar per clip, which is far weaker supervision. This argues for adapters over full fine-tuning, and for low expectations.

## 6. Feng & Narayanan: PEFT-SER (ACII 2023) [B]
- [arXiv 2306.05350](https://arxiv.org/abs/2306.05350).
- **Task and data:** speech emotion recognition on 4 datasets: IEMOCAP 5,531 utterances, CREMA-D 4,798, MSP-Improv, MSP-Podcast. Metric: UAR, averaged over the datasets.
- **Results** (Table V, mean across datasets; my reading of a table that lost its layout in text extraction, matched against the paper's prose; LoRA rank hardly mattered between 8, 16 and 32):

  | Encoder | Frozen encoder + downstream head | LoRA |
  |---|---|---|
  | WavLM Base+ | 65.28 | 67.28 (best overall) |
  | wav2vec2 Base | 62.27 | 63.30 |
  | Whisper Small | 66.53 | 66.54 |
  | Whisper Base | 65.19 | 66.41 |
  | Whisper Tiny | 64.21 | 63.42 |

  - The across-dataset standard deviation is about 6-9 UAR.
  - The authors: on Whisper, LoRA "maintains relatively the same performance or even underperforms". Adapters and prompt tuning on Whisper were below the frozen head.
- **Transfers:**
  - About +2 UAR for LoRA on WavLM, on about 5k utterances.
  - Inconsistent on Whisper-family encoders (-0.8 to +1.2). Voxtral's audio encoder is Whisper-derived, so this is consistent with our failed Voxtral QLoRA.

## 7. Kim, Jeon, Seo & Kim: SSL for pronunciation assessment (arXiv 2022) [B]
- [arXiv 2204.03863](https://arxiv.org/abs/2204.03863).
- **Data:**
  - KESL: 17,800 utterances from 300 Korean children, holistic pronunciation scored 1-5 by 5 experts and averaged.
  - Speechocean762: 5,000 utterances, split 50/50.
- **Method:** fine-tune wav2vec2 or HuBERT with CTC on the target data's transcripts, freeze it, then run a BLSTM over the average of all layers. Speaker-grouped 10-fold CV.
- **Results** (PCC):

  | | KESL holistic | Speechocean762 fluency | Speechocean762 prosody |
  |---|---|---|---|
  | HuBERT-Large frozen | 0.75 | 0.75 | 0.74 |
  | HuBERT-Large CTC-fine-tuned | 0.82 | 0.78 | 0.77 |
  | Best hand-crafted + GOP baseline | 0.64 | 0.67 | 0.66 |

  - Layers 11-22 are much better than layers 1-9; the best single layer is 20.
  - The all-layer average (0.82) is slightly above layer 20 (0.81).
  - On top of the encoder, a BLSTM (0.82) beats an MLP (0.80) and linear regression (0.78).
- **Transfers:**
  - The fine-tuning signal is ASR on matched transcripts, not the score. We have no manual transcripts.
  - Averaging layers is a small, real gain.
  - A sequence head beats pooling plus linear regression, which is the same lesson as our attention head.

## 8. Bannò & Matassoni: wav2vec 2.0 for L2 proficiency on small data (SLT 2022) [B]
- [arXiv 2210.13168](https://arxiv.org/abs/2210.13168).
- **Data:**
  - ICNALE monologues: 4,332 answers of 36-69 s (train 3,898, test 217), classification into 5 levels.
  - TLT-school: Italian pupils, 7 small-talk questions, B1, holistic score 0-12. Train 322, dev 85, test 87. **This is our data size**. ASR WER is 41%, and each answer was scored by one rater.
- **Method:**
  - Fine-tune `wav2vec2-base` with only the CNN feature encoder frozen, then mean-pool, Dense 768, dropout, linear output.
  - TLT holistic: AdamW, lr 5e-5, batch 4 with gradient accumulation 2, 12 epochs.
  - Baseline: frozen BERT on manual or ASR transcripts, with a deep MLP head.
- **Results:**

  | | PCC | MSE |
  |---|---|---|
  | TLT holistic, wav2vec2 | 0.927 | 2.30 |
  | TLT holistic, BERT on manual transcripts | 0.857 | 6.11 |
  | TLT holistic, BERT on ASR transcripts | 0.749 | 9.88 |
  | TLT lexical sub-score, BERT | 0.846 | 0.217 |
  | TLT lexical sub-score, wav2vec2 | 0.863 | 0.178 |
  | TLT lexical sub-score, average of both | 0.885 | 0.164 |
  | TLT lexical sub-score, deep fusion | 0.883 | 0.166 |

  - ICNALE accuracy: 77.9% against 53.5%.
  - wav2vec2 also beat BERT on the "formal correctness" (grammar) sub-score.
- **Caveats:**
  - The test set is only 87 clips, a single split, with no confidence intervals.
  - The baseline BERT was frozen and weak.
  - Test and train share the same 7 questions.
- **Transfers:** fine-tuning a base-size SSL encoder on about 300 clips is not hopeless. The averaging result also matches ours: a simple average of two views did as well as deep fusion.

## 9. Bannò, Knill, Matassoni, Raina & Gales: SSL grader on Linguaskill (SLaTE 2023) [A]
- [arXiv 2211.08849](https://arxiv.org/abs/2211.08849).
- **Data:** Linguaskill, 31,475 training speakers and 1,033 dev/calibration speakers. Tests: LinGen (1,049 speakers) and LinBus (712). CEFR 1-6. The ASR WER is about 20%.
- **Method:** fine-tuned wav2vec2-base per part with mean pooling, CNN frozen. 1-3 epochs, lr 1e-6 to 5e-5. Ensembles. Calibrated and combined by OLS on the dev set.
- **Per-part RMSE on LinGen** (P3 and P4 are one-minute free-speaking answers, the closest to SHL):

  | Part | Hand-crafted | BERT | wav2vec2 |
  |---|---|---|---|
  | P1 (short answers) | 0.625 | 0.628 | 0.601 |
  | P3 | 0.671 | 0.681 | **0.845** |
  | P4 | 0.686 | 0.694 | **0.845** |
  | P5 (short answers) | 0.633 | 0.629 | 0.674 |

  The authors blame mean pooling, which "may be giving too compressed a representation for longer utterances".
- **Submission level** (LinGen): RMSE 0.383 hand-crafted, 0.395 BERT, 0.455 wav2vec2. The per-part linear combination of all three gives 0.356 (PCC 0.943).
- **Other findings:**
  - The wav2vec2 grader never predicts above 4.6. It cannot separate the top levels, which depend on "message construction" (what is said).
  - Averaging the three systems at submission level gave no significant gain. Only the per-part regression did.
- **Transfers:**
  - A mean-pooled fine-tuned encoder is bad on one-minute answers. Our attention pooling is the fix this paper asks for.
  - Audio models compress the top of the scale, the same as our errors at score 5.

## 10. Park & Choi: the cold-start problem for unseen questions (arXiv 2023) [A, the key counter-evidence]
- [arXiv 2306.14310](https://arxiv.org/abs/2306.14310).
- **Data:** TOEIC Speaking, 1,879 test-takers, 20,669 responses, 12 test forms. Three forms are held out entirely as "unknown content".
- **Results with a frozen encoder** (average PCC, known content → unknown content, no prompt embeddings):

  | Encoder | Known | Unknown |
  |---|---|---|
  | wav2vec2-base | 0.713 | 0.527 |
  | HuBERT-base | 0.732 | 0.571 |
  | data2vec-base | 0.714 | 0.597 |
  | Whisper-base | 0.800 | **0.724** |

  - Question-type and question-text embeddings (BERT or CLIP) recover part of the drop for the SSL encoders: wav2vec2 unknown-content PCC goes from 0.527 to 0.605.
- **Unfreezing the encoder hurt everywhere, and most on new questions** (Table 4):

  | Encoder | Known | Unknown |
  |---|---|---|
  | wav2vec2 | 0.758 → 0.522 | 0.605 → 0.412 |
  | Whisper-base | 0.800 → 0.705 | 0.724 → 0.550 |

- **Caveat:** the fine-tuning recipe is not described (no LLRD, no adapters, batch size 4, RAdam). A careful recipe might lose less.
- **Transfers:** this is the closest published analogue to our test set, which answers new questions. It says:
  1. ASR-trained encoders (Whisper, and therefore Voxtral) generalise to new questions much better than SSL-only encoders;
  2. naive fine-tuning can destroy exactly that generalisation.

  Any encoder fine-tuning for SHL must be judged on topic-grouped folds.

## 11. Nguyen et al.: ASR-based wav2vec2 for speech-disorder assessment (arXiv 2024) [B]
- [arXiv 2410.08250](https://arxiv.org/abs/2410.08250).
- **Data:** C2SI, 106 speakers (82 patients), a read passage, intelligibility and severity scored 0-10 by 6 experts. A tiny dataset.
- **Method:** full fine-tuning of wav2vec2-large with stats pooling. The encoder is started either from an SSL checkpoint or from one already fine-tuned for ASR.
- **Results** (MSE, 3K-hour model):

  | Start from | Intelligibility | Severity |
  |---|---|---|
  | SSL checkpoint | 1.65 | 2.1 |
  | ASR checkpoint | 0.73 | 1.15 |

  - In layer-wise training, freezing and fine-tuning perform the same for early layers. Intelligibility needs fine-tuning from layer 8 upward; severity mostly does not.
- **Transfers:** in a tiny-data regime, the starting checkpoint (ASR-trained) mattered far more than the tuning method. This points the same way as Park & Choi: Whisper or Voxtral encoders are the right family.

## 12. Chapariniya et al.: BLEMORE challenge, frozen vs fine-tuned wav2vec2 (arXiv 2026) [C]
- [arXiv 2603.23650](https://arxiv.org/abs/2603.23650).
- **Data:** 2,456 clips of blended non-verbal emotion.
- **Result:** frozen layers 6-12 averaged scored 0.207 against 0.161 for end-to-end fine-tuning. The authors' reading: "Finetuning 95M parameters on only 2,456 samples leads to severe overfitting".
- **Caveat:** a challenge paper with a single comparison and a non-speech-content task.
- **Transfers:** one more data point that full fine-tuning at a few thousand clips can lose to frozen middle layers.

## 13. Liu et al.: ASR-free fluency scoring with SSL units (arXiv 2023) [B]
- [arXiv 2302.09928](https://arxiv.org/abs/2302.09928).
- **Data:** ByteQA, open responses from 269 Mandarin L1 adults (train 4,404, dev 1,395, test 1,451 utterances, mean 20 s), fluency scored 0-4.
- **Method:**
  - Take frozen wav2vec2-large layer-15 frames.
  - Fit K-means with K = 50, so each frame gets a pseudo-phone cluster ID.
  - A BLSTM reads the frames plus the cluster-ID embeddings. No ASR, no transcript.
- **Results** (PCC, open response):

  | System | PCC |
  |---|---|
  | Hand-crafted fluency features | 0.690 |
  | Deep ASR features with forced alignment | 0.759 |
  | **SSL + clusters, no ASR** | **0.797** |
  | SSL with ASR alignment | 0.791 |
  | SSL with human-transcript alignment | 0.812 |

  The cluster IDs co-occur with phonemes, and silence spreads over many clusters.
- **Caveat:** no ablation of the cluster IDs alone.
- **Transfers:** a fully unsupervised, own-data way to give a frame head explicit timing and unit structure (pauses, rate, rhythm), with no transcript errors.

## 14. Cooper, Huang, Toda & Yamagishi: generalisation of MOS predictors (ICASSP 2022) [B, skimmed]
- [arXiv 2110.02635](https://arxiv.org/abs/2110.02635).
- **Findings:**
  - Fine-tuned wav2vec2 with mean pooling and a linear layer generalises best zero-shot to new listening tests.
  - Fine-tuning on a small in-domain set improves it further.
  - Unseen systems remain hardest.
  - Utterances with a single rating were noticeably harder even after fine-tuning.
- **Transfers:** label noise from single raters caps achievable correlation. Two-stage fine-tuning, broad first and then in-domain, is the standard small-data route, but we have no broad labelled stage (no external data).

## 15. Gong et al.: Whisper-AT (Interspeech 2023) [A]
- [arXiv 2307.03183](https://arxiv.org/abs/2307.03183).
- **Finding:** Whisper's representation is "not noise-invariant, but is instead highly correlated to non-speech sounds". Linear probes on Whisper layers give the best ESC-50 sound classification of the ASR models compared. Unlike the other models, Whisper "encodes background sound information even in its deepest layer".
- **Transfers:** Voxtral's audio tower is Whisper-derived, so its states carry the recording channel. This is a mechanism for our train-vs-test AUC of 0.85 and for the 45.06 s recording batch behaving differently.
  - Label-preserving channel augmentation could help.
  - Removing channel directions naively would also remove whatever genuinely differs in that batch.

---

## Cross-paper summary for the ~800-clip regime
1. **Encoder family beats tuning method.** ASR-trained encoders (Whisper and its descendants) are best on new questions (Park & Choi) and in tiny-data clinical scoring (Nguyen). We already use them.
2. **Frozen + small trainable head is the robust default.**
   - Full fine-tuning wins only with matched-domain data, and even then the evidence is a single 87-clip test (Bannò & Matassoni).
   - It loses badly on unseen questions (Park & Choi) and at 2.4k clips (BLEMORE).
3. **If we fine-tune, the recipe should be:**
   - an SSL-style encoder (WavLM, wav2vec2) rather than Whisper (PEFT-SER found PEFT gains on Whisper inconsistent);
   - adapters rather than full fine-tuning (Chen 2022);
   - the top 1-3 layers re-initialised or dropped (Pasad 2021, where the gain is clear only at very small data);
   - the CNN front-end frozen (every paper here);
   - judged on topic-grouped folds.
4. **Layer choice matters, and all-layer mixing is a small gain** (Kim: 0.82 against 0.81). A learned per-layer mix inside a trained head is the cheap version.
5. **Mean pooling fails on one-minute answers** (Bannò 2023: P3/P4 RMSE 0.845 against 0.67). Attention or sequence heads are needed; Kim: BLSTM > MLP > linear regression.
