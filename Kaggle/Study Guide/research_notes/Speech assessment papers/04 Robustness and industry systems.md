# Robustness (new prompts, channel shift, augmentation, test-time adaptation, calibration) and industry systems

Part of the paper notes for [[11 Speech Assessment]]. The evidence tags ([A], [B], [C]) are defined in `01 Speech encoders and fine-tuning.md`.

---

## A. Industry systems

### 1. ETS SpeechRater 5.0 (Chen, Zechner, Yoon, Evanini et al., ETS RR-18-10, 2018) [A]
- [ERIC full text](https://files.eric.ed.gov/fulltext/EJ1202795.pdf).
- **Use:** operational for TOEFL Practice Online (TPO) Speaking as the sole score, and as a contributory score next to human raters for TOEFL iBT Speaking (per the report and ETS pages).
- **Pipeline:**
  1. Non-native ASR.
  2. Feature modules, about 100 features grouped by construct (below).
  3. **Filtering models** that flag non-scorable responses (no speech, non-English via language ID, off-topic or copied answers) and send them to humans.
  4. A linear regression scoring model.
- **Feature groups** (the correlation with human holistic score is quoted where printed):
  - **Fluency and disfluency:** speech-unit structure and edit disfluencies, for example `longSilRatio` (r = -.36).
  - **Pronunciation:** acoustic-model scores and vowel-duration deviation from native norms; `amscore` r = .404, the new features about .4.
  - **Rhythm:** for example `rpvis` (r = .357 in SR5; -.44 in Lai et al. 2013).
  - **Vocabulary:** frequency and diversity.
  - **Grammar:**
    - POS-bigram similarity to score-level learner corpora;
    - clause and phrase counts from parses;
    - "features … focus, for the most part, on grammatical **complexity** rather than on grammatical accuracy", because accuracy is too hard to measure through ASR, disfluencies and missing punctuation;
    - the best grammar features are around r = .3-.4.
  - Content and discourse coherence.
- **Model building:** experts pick the features relevant to the task, Lasso selects, experts adjust, then OLS. For TPO v5, 20 of 102 features were kept.
- **Agreement on TPO** (evaluation set):

  | Level | Human-human r | System-human r |
  |---|---|---|
  | Item (about 1,756 responses) | .591 | .557 |
  | Speaker (6 items aggregated, 263 speakers) | about .87-.88 | .77 |

  The standardised mean difference stays within ±.06.
- **Transfers:**
  1. **The human-agreement ceiling.** One rater on one response agrees with another at only r ≈ .6. Our OOF Pearson of 0.915 implies SHL labels are far more reliable than single ratings, which is consistent with averaged, off-grid test labels.
  2. Industry deliberately measures grammar through complexity proxies, not error counts.
  3. A separate non-scorable filter is standard practice. Our score-0 gate is that filter.

### 2. Pearson Versant (Cheng, Bernstein, Pado & Suzuki, BEA 2009, Versant Arabic Test) [B]
- [BEA PDF](https://www.cs.rochester.edu/u/tetreaul/bea4/Cheng-BEA4.pdf).
- **Design:** 69 *constrained* items per test: reading aloud, sentence repeats, short answers, **sentence building** (reorder three phrases) and passage retelling. This keeps ASR tractable.
- **Subscores and weights:**
  - Sentence Mastery 30%: the grammar-and-memory construct, scored from response fidelity on repeats and builds.
  - Vocabulary 20%.
  - Fluency 30%.
  - Pronunciation 20%.
- **Scoring:** trained on 246 h of transcribed native and learner responses, plus about 7,500 human judgments per fluency and pronunciation subscore. Human raters agree with each other at r = 0.79 (pronunciation) and 0.83 (fluency).
- **Validity:**
  - Test-retest r = 0.97.
  - r = 0.87 with the human ILR oral interview.
  - The English and Spanish versions correlate 0.88-0.90 with CEFR judgments.
- **Free-speech contrast:** the paper cites Zechner & Xi at r = 0.50 (opinion task) and r = 0.69 (picture description) for machine-human agreement on open responses.
- **Transfers:** Pearson gets reliable *grammar* scores by designing tasks where the expected words are known. Open answers like SHL's are the hard case, where even industry systems sit far lower.

### 3. Cambridge (Linguaskill / BULATS auto-marker)
Known through the research papers in notes 01-03; I could not open the Cambridge validity PDF (it returned 404):
- Knill 2018: a Gaussian-process grader on fluency and audio features, robust to WER.
- Malinin 2017: deep density networks with uncertainty-based rejection to humans.
- Bannò 2022, 2023: view-specific graders and wav2vec2 graders, combined by per-part OLS **calibrated on a held-out dev set**.
- Ma 2025: a speech-LLM grader with linear calibration on dev.
- The recurring design is: several complementary graders, linear combination and calibration on dev, and an uncertainty measure that hands doubtful candidates to humans.

### 4. Duolingo-style engineering (Team Perezoso, note 03)
The S&I winner's final model is:
- end-to-end graders;
- plus ASR confidence (alone r = 0.737 PCC);
- plus words per second, duration and length counts;
- all combined in a single linear regression.

This is the modern version of the SpeechRater recipe: interpretable features and deep graders, stacked linearly.

---

## B. Robustness

### 5. New prompts or questions
- **Park & Choi 2023** (note 01, [arXiv 2306.14310](https://arxiv.org/abs/2306.14310)):
  - Frozen SSL encoders lose about 0.13-0.19 PCC on unseen test forms. Whisper-base loses 0.08 (0.800 → 0.724).
  - Fine-tuning the encoder roughly doubles the loss.
  - Question-text embeddings recover a little.
- **Ma et al. 2025** (note 03): speech-LLM graders generalise across task types far better than wav2vec2 or BERT. Trained only on the read-aloud part, the overall PCC is 0.911, against 0.847 (wav2vec2) and 0.888 (BERT).
- **Bannò et al. 2025** (note 03): text-only systems drop more from dev to eval than the audio LLM.
- **Takeaway for SHL** (34% of test clips have a same-topic train neighbour, against 70% within train):
  - ASR-trained audio encoders and audio LLMs are the most prompt-robust components.
  - Text models and any fine-tuned encoder must be validated on topic-grouped folds.

### 6. Recording channel and batch shift
- **Whisper-AT** ([arXiv 2307.03183](https://arxiv.org/abs/2307.03183)): Whisper-family features encode the background-sound type even in the last layer. A test-vs-train classifier will find the channel easily, as ours does at AUC 0.85.
- **Perezoso** used additive noise and reverberation augmentation (MUSAN and RIR corpora) without an ablation. Those corpora are external data, which our rules forbid.
- **UTMOS** ablation (note 03): speed and pitch augmentation *raised* MSE (OOD 0.322 → 0.378 when added; main 0.226 → 0.276). Augmentation is not automatically good for regression error.
- **The label-preservation rule.** An augmentation is valid only if a rater would give the same score:
  - pitch shift, mild loudness changes and added noise at moderate SNR are plausibly label-preserving;
  - **speed or tempo perturbation is not**, because it changes speech rate and pausing, which are the fluency cues that drive the grammar score (Xi & Mollaun, note 02);
  - cropping is label-preserving only if the score does not depend on length. Our crop-trained models assume the human score of a clip also applies to its prefix, which the leaderboard has supported.

### 7. Test-time adaptation for regression (Adachi, Yamaguchi, Kumagai & Hamagami, ICLR 2025) [B, skimmed]
- [arXiv 2410.03263](https://arxiv.org/abs/2410.03263).
- **Method:**
  - Naive feature alignment (match target feature statistics to the source) fails or diverges in regression. Regression features live in a small subspace, and most dimensions do not affect the output.
  - SSA instead aligns, by a symmetric KL between diagonal Gaussians, only the PCA subspace that matters to the regressor, weighted by each direction's effect on the output. Only the feature extractor is updated.
- **Results:** beats BN-adapt and other baselines on UTKFace with corruptions, Biwi Kinect and California Housing.
- **The critical assumption:** the shift is *covariate shift*, where p(y|x) is unchanged and the target label distribution is similar. If the target batch really has different labels, which is label shift, then aligning features erases real signal.
- **Transfers:** for SHL's 45.06 s batch the training batch scores lower, and the test batch answers new questions. Our probes showed a plain score offset for the batch failed. Feature-space alignment is the same bet in a more complicated form. **Not recommended** without a validation design that simulates the batch shift, for example leaving out the training batch.

### 8. Calibration to the score scale
- **Industry practice:** fit a linear calibration on a held-out dev set (Cambridge papers), and report the standardised mean difference against humans (SpeechRater: within ±.15).
- **Distribution-aware outputs:**
  - "Fair-average" expected-value decoding (Ma 2025: 0.559 against 0.563 for cross-entropy with soft decoding, 0.615 hard) gives continuous, calibrated predictions from a classifier.
  - Hard class decoding is clearly worse, which matches SHL's rounding failure.
- **Dev-tuned per-class thresholds overfit dev** (Porwal 2025: dev RMSE 0.402 → 0.364, eval 0.411 → 0.423).

---

## Summary
1. Open-response grammar scoring is the hard case, even for industry:
   - item-level human-human r is about .6 for TOEFL-type items;
   - Versant reaches high reliability only through constrained tasks;
   - SpeechRater scores grammar through complexity, not accuracy.
2. Robustness to new questions comes mostly from the **encoder family** (ASR-trained, or an audio LLM), not from tricks. Fine-tuning reduces it.
3. Channel information is in Whisper-family features by design. Augmentation must be label-preserving and is not reliably helpful; speed perturbation is not label-preserving for fluency-entangled scores.
4. Test-time feature alignment assumes covariate shift. SHL's hard batch looks like covariate *and* label shift, so it is a poor fit.
