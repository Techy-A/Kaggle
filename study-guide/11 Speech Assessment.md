---
---
# 11 Speech assessment

Back to [[00 Study Guide Home]].

> **Core idea:**
> - A "grammar score" for spontaneous speech is a human judgment that mixes *what* was said with *how fluently* it was built.
> - Speech recognisers quietly erase the errors you are trying to measure.
> - With a few hundred clips, the winning recipe is a strong frozen ASR-trained encoder, a small trainable head, the right loss, and validation that looks like the test (new speakers *and* new questions).

The paper-by-paper notes, with data sizes and exact numbers, are in `research_notes/Speech assessment papers/`:
- `01` speech encoders and fine-tuning;
- `02` grammar and the ASR filter;
- `03` holistic graders, losses and the Speak & Improve 2025 challenge;
- `04` robustness and industry systems.

## 1. The task family in plain words
- **Automated speaking assessment (ASA)** predicts a human score from a recording.
  - *Holistic* scores rate the whole performance, for example on CEFR levels A1-C2.
  - *Analytic* scores rate one aspect: delivery (fluency and pronunciation), language use (grammar and vocabulary), or topic development.
- **Read-aloud vs free speaking.**
  - In read-aloud tasks the system knows the words, so it can compare against them.
- **There are two kinds of models:**
  - *Cascaded:* ASR → transcript → text model, plus timing features (words per second, pauses).
  - *End-to-end:* audio → encoder → score, with no transcript.

  Every strong system since 2022 combines both, usually with a simple linear stack.
- **Know the ceiling.** Two trained humans scoring the *same* one-minute TOEFL-type answer agree at only about r = .59 (ETS SpeechRater 5.0 report). Averaged over 6 answers, they reach about .88.

## 2. What raters actually score: grammar is tangled with fluency
- **The ETS evidence.** In ETS's study of analytic speaking scores (Xi & Mollaun 2006), *delivery* and *language use* correlate at r = .78-.90 observed, and **.89-.98 after correcting for rater noise**.
  - Raters said that when judging language use they look at how well the speaker builds sentences "on the fly", which is automaticity. That is a fluency cue.
- **Why it works this way.** Building grammatical sentences in real time is a skill. A speaker who has it pauses less, restarts less and sounds smoother. The rater hears the smoothness, so the "grammar" score inherits fluency.
- **Consequences:**
  1. Audio models that never see a word can predict a grammar score well. That is exactly what we found: frozen audio encoders were the strongest signal.
  2. A model that only counts grammar *errors* is weak. A Cambridge grader that sees only GEC edit sequences was one of the two weakest of six graders (RMSE 0.547, against 0.400 for the full-text grader).
     - ETS's operational grammar features measure *complexity* (clause counts, POS-pattern similarity to each score level), not accuracy, and reach only about r = .3-.4.
  3. This is why LanguageTool counts, T5-GEC edit rates and CoLA added nothing for us. The error signal is a small slice of what raters reward, and the text encoder already has it.

## 3. ASR is a lossy filter, and it fails exactly on the errors
- **Recognisers are trained to produce clean text.** Whisper adds punctuation, writes "$20" for "twenty dollar" and drops "um", repetitions and false starts (Ma et al. 2023). Each of these is grammar or fluency evidence.
- **The error tokens are recognised worst.** In Cambridge's BULATS data, the word error rate on words carrying a grammatical error was **33.1%, against 22.7% overall** (Knill et al. 2018).
- **The loss is large even with good ASR.** Grammatical error detection reaches F0.5 42.5 on manual transcripts but only 29.4 on the best ASR (Lu et al. 2019).
- **Fixes, from cheapest to most expensive:**
  1. *Verbatim prompting* (what we do): give Whisper a prompt full of fillers and mistakes so it keeps them.
  2. *Confidence masking:* trust low-confidence words less (Lu 2019).
  3. *Soft-prompt or LoRA tuning of the recogniser* toward verbatim output. This needs transcribed audio (Ma 2023; Lin 2025: explicit "um/uh" tokens improved WER by 11% relative). We have none.
  4. *ASR-free audio models*, which hear the disfluency directly. This is our Voxtral and WavLM side.
- **Graders built on fluency features are robust to WER**, because they do not read the words. Holistic PCC stayed at 0.854 when WER fell from 47.5% to 30.4% (Knill 2018). Text-structure features need good ASR.

## 4. Speech encoders: what is inside, frozen vs fine-tuned
### 4.1 Two families
- **Self-supervised (SSL)** models (wav2vec2, HuBERT, WavLM) learn by predicting masked audio. They capture acoustics, phones and some words.
- **ASR-trained** models (Whisper, and audio-LLM towers such as Voxtral's) are trained to map audio to text, so their states carry word and sentence information.
- **For scoring, the ASR-trained family wins, most of all on new questions.**
  - Frozen encoders on TOEIC answers to *unseen* test forms (Park & Choi 2023): Whisper-base kept PCC 0.724, while wav2vec2 fell to 0.527.
  - In a 106-speaker clinical task, starting from an ASR-fine-tuned checkpoint more than halved the error (MSE 1.65 → 0.73; Nguyen et al. 2024).

### 4.2 What the layers hold
- **Lower layers** hold the speaker and the channel; **middle and upper layers** hold phones and words (WavLM's layer-weight analysis; Pasad et al. 2021, 2023).
  - wav2vec2 behaves like an autoencoder: its last layers drift back toward the input.
  - HuBERT and WavLM put phonetic and word content high up.
- **The layer choice matters, and it depends on the job.** On its own, WavLM-large layer 20 beat our layer 8-16 average. In the blend it hurt, because it duplicates the word-level content Voxtral already has.
- **Whisper is not noise-invariant.** It encodes the *type of background sound* even in its last layer (Whisper-AT). Its features therefore carry the recording channel, which helps explain why a classifier separates our test audio from train (AUC 0.85).
- **Mixing layers** is a small, reliable gain: a learned weighted sum (SUPERB), or an average of all layers (Kim 2022: 0.82 against 0.81 for the best single layer).

### 4.3 Frozen + small head, or fine-tuning?
- **Frozen encoder + small trainable head** is the default for small data, and it is what won Speak & Improve 2025.
  - Team Perezoso (1st) froze Whisper-medium's encoder and trained about 1M parameters: one small adapter per layer for the last six layers, then mean + std pooling.
  - Alone it reached PCC 0.839, against 0.747 for the official text baseline.
- **Fine-tuning can win in-domain:**
  - wav2vec2-base fine-tuned on only 322 Italian school answers reached PCC 0.927, against 0.857 for BERT on manual transcripts (Bannò & Matassoni 2022). But that was one 87-clip test set, with the same 7 questions as training.
  - CTC fine-tuning on target transcripts gave +0.03 to +0.07 PCC for pronunciation (Kim 2022).
- **Fine-tuning can collapse on new questions:**
  - Unfreezing Whisper-base on TOEIC dropped unseen-question PCC from 0.724 to 0.550 (Park & Choi).
  - A 2.4k-clip emotion challenge found frozen middle layers beat fine-tuning (0.207 against 0.161).
- **If you do fine-tune at small N, the recipe is:**
  - freeze the CNN front end;
  - prefer *adapters* to full fine-tuning (Chen et al. 2022: adapters beat full fine-tuning at 1-10 h of audio and tolerate far more learning rates);
  - LoRA works on WavLM (+2.0 UAR in Feng & Narayanan 2023) but is inconsistent on Whisper (-0.8 to +1.2 across sizes);
  - re-initialise or drop the top 1-3 layers of a wav2vec2-style model (Pasad 2021; a big gain at 10 min of data, minor at 1-10 h);
  - use a low learning rate and few epochs;
  - **validate on unseen questions**.

## 5. Long answers need better pooling than an average
- **Mean pooling breaks on one-minute answers.** Cambridge's fine-tuned, mean-pooled wav2vec2 grader matched text graders on short answers (P1 RMSE 0.601). On one-minute answers it was far worse (P3 and P4 0.845, against 0.67-0.69), and it never predicted above 4.6 (Bannò et al. 2023).
  - The averaging washes out the few moments that matter, and the top levels depend on *what* is said.
- **The fixes:**
  - *attention pooling*: a learned weight per frame (NTNU, McKnight 2023, and our Voxtral head);
  - *sequence heads*: BLSTM > MLP > linear on pooled vectors (Kim 2022: 0.82 / 0.80 / 0.78);
  - *chunk-then-aggregate* (Aalto: 30 s chunks, then a mean or a 2-layer transformer).
- **Small heads generalise better.** GOPT's 27K-parameter head beat bigger ones (PCC 0.742 against 0.725). The Aalto transformer aggregator was better on average but unstable across epochs.

## 6. Speech LLMs as graders
- **Fine-tuned, they are the state of the art.** Qwen2-Audio-7B with LoRA, trained on 31k Linguaskill submissions, reached PCC 0.954, against 0.942 for BERT (Ma et al. 2025). On the 6.6k-submission S&I set it gave 0.833 against 0.753.
- **Zero-shot, they are weak:** PCC 0.640. A text LLM reading CEFR descriptors zero-shot beat a fine-tuned BERT on S&I (0.761 against 0.727 on eval) but not the fine-tuned audio LLM.
- **Using their hidden states as features** is the cheap middle road, and our best single view. Fine-tuning a 3-7B audio LLM on 769 clips is far below the data scale where it has been shown to work. Our Voxtral QLoRA failed in its first folds.

## 7. Losses and labels
- **Regression (MSE)** is the default. It is fine, but it treats 4.5 vs 5 the same everywhere and pulls predictions to the mean.
- **Plain classification** throws away the order, and hard decoding (taking the argmax class) is worse:
  - Ma 2025: RMSE 0.615 hard against 0.563 soft;
  - Porwal 2025: on eval, cross-entropy was *worse* than regression.
- **Ordinal and distribution-aware outputs** keep the order and give calibrated continuous predictions:
  - **CORN** trains K-1 conditional "is the score above k?" classifiers. On S&I eval it changed RMSE 0.440 → 0.411 and PCC 0.726 → 0.782 with the same BERT architecture (Porwal 2025). It was the *only* change in that paper that survived from dev to eval.
  - **"Fair average"** takes a softmax over score bins and trains MSE on the expected value. On 31k submissions it was slightly better than a regression head (RMSE 0.559 against 0.565; Ma 2025) and trained more smoothly.
  - Decode by the **expected value**, never by rounding to a bin.
- **Ranking terms.** A pairwise margin loss on score differences targets correlation directly (UTMOS). The evidence is weak: in the UTMOS ablation, removing it lowered correlation 0.891 → 0.870 on the small out-of-domain track and left the main track unchanged (0.883 → 0.881).
- **Rater noise:**
  - modelling the rater's identity was the largest component in UTMOS, but needs rater IDs;
  - heteroscedastic heads predict a variance and can send doubtful cases to humans (Malinin 2017);
  - auxiliary tasks barely help a pre-trained encoder (Craighead 2020: only L1 prediction, 0.921 → 0.908 RMSE; GOPT: +0.006 PCC, inside seed noise).

## 8. Robustness: new questions, channel, augmentation, test-time adaptation
- **New questions.** Validation must hold out *questions*, not just speakers (Park & Choi). ASR-trained audio encoders and audio LLMs are the most prompt-robust components. Text models lose more from dev to eval.
- **Augmentation must be label-preserving:**
  - pitch shift and moderate noise are plausibly safe;
  - *speed perturbation is not*, because it changes speech rate and pauses, which drive the grammar score (section 2);
  - UTMOS's own ablation found speed/pitch augmentation *raised* MSE;
  - the S&I winner used noise and reverberation corpora (MUSAN, RIR) without an ablation. Those are external data, forbidden for us.
- **Test-time adaptation for regression** (SSA, ICLR 2025) aligns test feature statistics to train inside the subspace the regressor uses. It assumes the labels have not shifted. Our hard 45.06 s batch is lower-scoring *and* answers new questions, so aligning it would erase real signal. The failed batch-offset probe is the same lesson.

## 9. How industry does it
- **ETS SpeechRater:**
  1. ASR;
  2. about 100 interpretable features (fluency, pronunciation, rhythm, vocabulary, grammatical complexity, content, discourse);
  3. filter models that send non-scorable responses (silence, non-English, off-topic) to humans;
  4. a Lasso-then-OLS linear model with 20 features, chosen together with experts.
- **Pearson Versant** gets reliable grammar ("Sentence Mastery", 30% of the score) by using constrained tasks (repeat, build a sentence) instead of open answers. Test-retest r = 0.97.
- **Cambridge (Linguaskill):**
  - several complementary graders (text, wav2vec2, hand-crafted features, recently a speech LLM);
  - combined by per-part linear regression and calibrated on a held-out dev set;
  - uncertainty is used to hand doubtful candidates to examiners.
- **Duolingo-style (S&I winner):** frozen-encoder graders plus ASR confidence, words per second and length counts, stacked by linear regression.

Our system already has the same shape:
- graders of different kinds (audio encoders, Voxtral, DeBERTa);
- a non-scorable filter (the score-0 gate);
- a linear, non-negative stack;
- calibration (the bias and stretch corrections).

## 10. Evidence at a glance (data sizes matter)
| Study | Data | Key number | Tier |
|---|---|---|---|
| Bannò & Matassoni 2022 | 322 train / 87 test answers | fine-tuned wav2vec2 PCC 0.927 vs BERT 0.857 | B (tiny test) |
| Park & Choi 2023 | 20,669 responses | unfreezing: unseen-question PCC 0.724 → 0.550 | A |
| Perezoso (S&I 1st) | 6,640 submissions | frozen Whisper + 1M-parameter adapters PCC 0.839 alone | B (no ablation) |
| Porwal 2025 | 6,640 / 442 eval | CORN eval RMSE 0.440 → 0.411 | A |
| Ma 2025 | 31k + 6.6k | speech-LLM PCC 0.954; fair-average loss 0.559 vs regression 0.565 | A |
| Phan 2025 | about 1,700 speakers | edge oversampling: edge RMSE 0.698 → 0.613, overall ±0 | B |
| Knill 2018 | 994 train / 226 eval speakers | WER on grammar-error words 33.1% vs 22.7% | A |
| Xi & Mollaun 2006 | 140 examinees × 6 tasks | delivery vs language use disattenuated r .89-.98 | A |
| Chen 2022 (adapters) | 1-100 h ASR | adapters > full fine-tuning at ≤ 10 h | A |
| UTMOS 2022 | 4,974 / 136 clips | speed/pitch augmentation raised MSE | A |

## Exercises
1. On our OOF predictions, compute the RMSE separately for labels ≤ 2, 2.5-4 and ≥ 4.5. How much of the total squared error do the edges carry? (This sizes idea 3.)
2. Swap the attention head's MSE output for a 9-bin softmax with an expected-value MSE loss. Report the OOF metric over 3 retrains, against the noise reference of 0.0003.
3. Correlate per-layer Voxtral pooled features with the label. Plot r by layer. Which layers would you give separate adapters?
4. Write down which of our augmentations (crops, and any audio perturbation) are label-preserving for a *fluency-entangled* grammar score, and why.

## Resources
- Zechner & Evanini (eds.), *Automated Speaking Assessment: Using Language Technologies to Score Spontaneous Speech* (Routledge, 2019). The standard book.
- Chen, Zechner et al., *SpeechRater 5.0* (ETS RR-18-10, 2018), [ERIC](https://files.eric.ed.gov/fulltext/EJ1202795.pdf).
- Qian et al., *Speak & Improve Challenge 2025* ([arXiv 2412.11985](https://arxiv.org/abs/2412.11985)) and the [SLaTE 2025 proceedings](https://www.isca-archive.org/slate_2025/index.html).
- Ma et al., *Assessment of L2 oral proficiency using speech LLMs* ([arXiv 2505.21148](https://arxiv.org/abs/2505.21148)).
- Park & Choi, *Cold start problem for end-to-end speech scoring* ([arXiv 2306.14310](https://arxiv.org/abs/2306.14310)).
- Pasad, Shi & Livescu, *Comparative layer-wise analysis of SSL speech models* ([arXiv 2211.03929](https://arxiv.org/abs/2211.03929)).
- Xi & Mollaun, *Analytic scoring for the TOEFL Academic Speaking Test* ([ERIC](https://files.eric.ed.gov/fulltext/EJ1111433.pdf)).
