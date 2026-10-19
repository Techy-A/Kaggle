# Spoken grammatical competence: what raters measure, what ASR destroys, what grammar features add

Part of the paper notes for [[11 Speech Assessment]]. The evidence tags ([A], [B], [C]) are defined in `01 Speech encoders and fine-tuning.md`. All papers were read as full PDFs unless an entry is marked "skimmed".

---

## 1. Xi & Mollaun: analytic scoring of the TOEFL Academic Speaking Test (ETS RR-06-07, 2006) [A]
- [ERIC full text](https://files.eric.ed.gov/fulltext/EJ1111433.pdf).
- **Data:** 140 examinees, 6 tasks each. Responses were double-rated on three analytic dimensions: delivery (fluency and pronunciation), language use (grammar and vocabulary) and topic development. The study used G-theory.
- **Finding:** language use and delivery are almost the same thing to raters.

  | Task | Delivery vs language use, observed r | Disattenuated r |
  |---|---|---|
  | 1 | .84 | .96 |
  | 2 | .79 | .89 |
  | 3 | .78 | .92 |
  | 4 | .90 | .98 |

  - Delivery vs topic development was lower.
  - Averaged over two or more tasks, the analytic profiles become flat.
- **Why:** raters said they judged language use by how well the speaker put words together "on the fly". "Automaticity" is in the language-use descriptor and also shows up as fluency in delivery. Raters disagreed (2.9 of 5) with "when rating language use I … rarely thought about automaticity".
- **Transfers:** this is the clearest explanation of SHL's main empirical fact, that frozen audio (fluency and rhythm) models predict the *grammar* score better than transcript models. A human "grammar" score is partly a fluency score. Two consequences:
  1. Do not expect explicit grammar-error features to add much.
  2. Do not throw away delivery information when modelling grammar.

## 2. Bannò, Balusu, Gales, Knill & Kyriakopoulos: view-specific assessment (Interspeech 2022) [A]
- [ISCA PDF](https://www.isca-archive.org/interspeech_2022/banno22_interspeech.pdf).
- **Data:** Linguaskill. 31,475 training and 1,033 dev speakers. Tests: LinGen (1,049) and LinBus (712). The ASR WER is 19.5%.
- **Method:** each grader is trained on the holistic score but can only see one "view". The grammatical-accuracy grader reads only GEC edit sequences (ERRANT edits between the ASR text and its automatic correction, GEC trained on CLC + BEA-19). The parts-of-speech grader reads only POS sequences. Others see only pronunciation, rhythm, or the full text through BERT.
- **Results** (LinGen RMSE, ensemble of 10):

  | Grader | RMSE |
  |---|---|
  | Full text (BERT + attention) | 0.400 |
  | Hand-crafted all-view baseline | 0.412 |
  | Pronunciation | 0.452 |
  | Rhythm | 0.508 |
  | Grammar edits only | 0.547 |
  | POS only | 0.550 |

  The OLS combination of views beats each component, so the views are complementary.
- **Transfers:** a grader that sees *only* grammatical errors is, with the POS-only grader, the weakest grader of holistic proficiency. That matches our finding that LanguageTool counts and a T5 GEC edit rate added nothing. The error signal is already inside the full-text view.

## 3. Knill et al.: impact of ASR performance on free-speaking assessment (Interspeech 2018) [A]
- [ISCA PDF](https://www.isca-archive.org/interspeech_2018/knill18_interspeech.pdf).
- **Data:** BULATS. A Gaussian-process grader trained on 994 speakers, evaluated on 226 speakers.
- **Results:**

  | | ASR WER | Grader PCC |
  |---|---|---|
  | ASR system 1 | 47.5% | 0.854 |
  | ASR system 2 | 30.4% | 0.854 |
  | Standard human examiners | | 0.848 |

  - A 17-point drop in WER changed nothing for the audio-and-fluency feature grader.
  - POS-tag features help only with the better ASR. Baseline + POS: PCC 0.847 with ASR 1 but 0.865 with ASR 2. POS alone: 0.792 against 0.833.
  - **ASR is worst exactly where the grammar errors are.** With ASR 2, the WER on words carrying a grammatical error is 33.1%, against 22.7% on all words. Words with pronunciation errors are at 29.0%.
- **Transfers:** quantitative support for the README's opening example. A recogniser fails disproportionately on the very tokens a grammar score depends on. Holistic graders built on fluency features are robust to WER, but grammar-sensitive text features need good ASR.

## 4. Lu, Gales, Knill, Manakul, Wang & Wang: ASR impact on spoken grammatical error detection (Interspeech 2019) [A]
- [ISCA PDF](https://www.isca-archive.org/interspeech_2019/lu19b_interspeech.pdf).
- **Data:** BULATS free speaking, with manual grammatical-error annotation, plus NICT-JLE. A GED model trained on the written CLC.
- **Results:**

  | ASR WER | GED F0.5, feedback scoring | With ASR confidence |
  |---|---|---|
  | 25.5% | 25.7 | 27.2 |
  | 21.3% | 27.8 | 29.9 |
  | 19.5% | 29.4 | 31.2 |
  | Manual transcripts | 42.5 | |

  - Treating words with ASR confidence below a threshold (0.4 best) as correct reduces false alarms.
  - Fine-tuning the written-text GED system on speech transcripts gives significant further gains.
- **Transfers:** even with good ASR, about a third of the detectable grammar signal is lost to transcription. ASR confidence is a useful mask. Our pipeline already extracts Whisper confidence; using it as a per-token signal inside the text model, not as one clip statistic, is untested.

## 5. Ma, Qian, Gales & Knill: adapting Whisper for spoken language assessment (SLaTE 2023) [B]
- [arXiv 2307.09378](https://arxiv.org/abs/2307.09378).
- **Finding:** Whisper adds punctuation, applies inverse text normalisation and **skips hesitations, repetitions and false starts**. In their example, "it costs um twenty dollar p- per week" becomes "It costs $20 per week": the plural error and the disfluency are gone.
- **Two fixes, each trained on less than 20 h of verbatim transcripts:**
  - full fine-tuning;
  - soft-prompt tuning: 20 learned decoder-input embeddings, 15 KB of parameters, about as good as fine-tuning.
- **Results** (LibriSpeech, 10 h of training data): Speech WER 4.0 → 3.2 (test-clean) and 7.9 → 6.8 (test-other). Recall on hesitations, numbers, abbreviations and repetitions rose by about 30 points absolute.
- **Transfers:**
  - Our verbatim *text* prompt is the zero-label cousin of this soft prompt.
  - A learned soft prompt needs verbatim reference transcripts, which we do not have.
  - The "$20" example shows why digit normalisation is itself grammar-destroying.

## 6. Bannò, Ma, Qian, Knill & Gales: towards end-to-end spoken GEC (ICASSP 2024) [B, skimmed]
- [arXiv 2311.05550](https://arxiv.org/abs/2311.05550).
- **Finding:**
  - Whisper can be fine-tuned to output disfluency-removed text directly (end-to-end disfluency detection), and this beats a cascade.
  - End-to-end *GEC* is limited by the small amount of spoken GEC data. A cascade that uses large text-GEC corpora still wins on correction.
- **Transfers:** none directly, since we have no error annotations. It confirms that disfluency structure is learnable from audio, so audio models can "see" disfluencies that a normalising transcript drops.

## 7. Craighead, Caines, Buttery & Yannakoudakis: auxiliary objectives for grading speech transcriptions (ACL 2020) [A]
- [ACL Anthology](https://aclanthology.org/2020.acl-main.206).
- **Data:** BULATS, 1,213 candidates with up to 7 one-minute spontaneous answers each, from 322 unique prompts. Split 55/25/20 by candidate. Scores 0-6 in half points. ASR WER 19.5%.
- **Results** (10 repetitions; RMSE, PCC):

  | Model | RMSE | PCC |
  |---|---|---|
  | Feature-rich linear baseline | 1.086 | 0.685 |
  | LSTM scorer | 1.022 | 0.681 |
  | LSTM + POS + UD + L1 auxiliary tasks | 1.005 | 0.690 |
  | BERT scorer | 0.921 | 0.762 |
  | BERT + L1 prediction | 0.908 | 0.769 |
  | BERT + LM auxiliary | 0.910 | |
  | BERT + POS auxiliary | 0.918 | |
  | BERT + UD auxiliary | 0.920 | |

  - Only BERT + L1 prediction was significant. "No combinations of auxiliary objectives led to any improvement for the BERT encoder."
  - Removing filled pauses from the transcripts made BERT worse (0.926 against 0.921), so hesitations carry signal.
- **Transfers:** with a pre-trained encoder, linguistic auxiliary tasks give at most 1-2% RMSE, and only from a target the model does not already know (L1). We have no L1 labels. Keeping fillers in the transcript is right; our verbatim prompt does this.

## 8. Bhat, Xue & Yoon: shallow syntactic-complexity measure for speech scoring (ACL 2014) [B, skimmed]
- [ACL PDF](https://mirror.aclweb.org/acl2014/P14-1/pdf/P14-1123.pdf).
- **Data:** TOEFL-type 45-60 s spontaneous responses. 47,227 responses to train the measures and 2,950 to evaluate the scoring model. The score distribution is 84-85% at score points 2-3.
- **Method:** a MaxEnt model over POS bigrams predicts the score level a response's grammar resembles.
- **Results:**
  - PCC with proficiency: 0.57 on manual transcripts and 0.52 on ASR, against 0.48 and 0.43 for the earlier vector-space measure.
  - Deep-parse complexity features drop by more than 0.40 in r from manual to ASR. Shallow POS features drop by about 0.05.
  - Adding the measure to the operational model is statistically significant (Steiger's Z).
- **Transfers:** shallow features survive ASR errors and deep parses do not. A fine-tuned DeBERTa on verbatim transcripts subsumes POS-bigram statistics, and our transcript statistics add less than 0.001.

## 9. Lu, Lin, Lin, Wang & Chen: relevance and grammar information for speaking assessment (SLaTE 2025) [C]
- [ISCA PDF](https://www.isca-archive.org/slate_2025/lu25_slate.pdf).
- **Data:** GEPT picture description, 1,199 responses over 4 question sets. One set is held out as "unknown". Scores 1-5 from two raters, averaged and floored.
- **Method:**
  - Phi-4 GEC, fine-tuned on Speak & Improve and prompted with few-shot examples.
  - SERRANT error-type frequencies per word.
  - Fused by cross-aspect attention with relevance and delivery features.
- **Results** (holistic accuracy, unknown test): 0.717 with grammar features, 0.693 without, 0.687 for the SAMAD baseline. Binary accuracy 0.797 against 0.780.
- **Caveats:** tiny test sets (about 90 seen and 300 unseen), no seeds, and the GEC was trained on an external corpus.
- **Transfers:** at most a weak signal. It would violate our no-external-data rule, and our own GEC-feature tests found nothing.

## 10. Lin, Lu, Wang, Lin & Chen: hesitation tagging for verbatim ASR (SLaTE 2025) [B]
- [ISCA PDF](https://www.isca-archive.org/slate_2025/lin25b_slate.pdf).
- **Data:** Speak & Improve 2025. 315 h of audio, of which 55 h have manual verbatim transcripts. About 3,900 training utterances in the post-challenge experiments.
- **Results:**
  - LoRA (rsLoRA r = 32) fine-tuning of Whisper large-v3 cut the closed-track WER from 10.84% to 6.47%.
  - Training on targets with explicit "um"/"uh" tokens improved WER by 11.3% relative (6.2% → 5.5%, large-v3-turbo), even though fillers are deleted before scoring.
  - Off-the-shelf WER on S&I: large-v3 9.5%, large-v3-turbo 9.6%, small 10.7%.
- **Transfers:** keeping fillers in the decoding target helps a recogniser keep content words too. That is indirect support for our filler-rich Whisper prompt. Fine-tuning needs verbatim transcripts, which we lack.

---

## What the grammar literature says, in one paragraph
Human grammar or "language use" judgments of spontaneous speech are dominated by how fluently and automatically the speaker builds sentences:
- delivery vs language use has a disattenuated r of 0.89-0.98 (Xi & Mollaun);
- an errors-only grader is among the weakest views of proficiency (Bannò 2022);
- the operational SpeechRater measures grammatical *complexity*, not accuracy, because accuracy is too hard to detect through ASR (see note 04).

ASR is a filter that removes precisely the evidence a grammar score needs:
- words with grammatical errors have 1.46 times the WER of all words (Knill 2018);
- grammatical error detection on ASR output reaches 29.4 F0.5 against 42.5 on manual transcripts (Lu 2019);
- Whisper normalises "twenty dollar" to "$20" and drops disfluencies (Ma 2023).

The two levers that work are:
1. verbatim-preserving transcription, through prompts or fine-tuning;
2. ASR-free audio models, which hear the disfluencies directly.

Explicit grammar-error features rarely add to a strong text encoder.
