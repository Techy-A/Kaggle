# Data-Centric AI: Building, Auditing, Documenting and Cleaning Datasets, and Famous Dataset-Quality Failures

> Research note. WebFetch could not resolve any host during this session (DNS failures on arxiv.org, proceedings.neurips.cc and others). Every number below therefore comes from search-result abstracts and snippets of the primary sources linked, not from reading the full PDFs. Where snippets disagreed, both versions are given. Items marked "(prior knowledge, not re-verified)" come from my background knowledge, are kept out of Cited Findings, and appear only under Inferences.

---

## 1. Label-error detection: confident learning, cleanlab, test-set errors, cartography, AUM, influence functions, Data Shapley

### Takeaway
Real benchmarks carry a few percent of label noise, even in their test sets: at least 3.3% on average across 10 major datasets, and about 6% of ImageNet val. That is enough to flip which model looks best. Cheap, model-agnostic detectors are available: confident learning/cleanlab (needs only out-of-fold predicted probabilities), training-dynamics methods (cartography, AUM) and data valuation (Data Shapley, influence functions). Treat their output as a ranked list for human review, not as an automatic delete list.

### Cited Findings
**Confident learning (CL) and cleanlab**
- Northcutt, Jiang & Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels", JAIR 70 (2021), pp. 1373–1411, DOI 10.1613/jair.1.12125, published April 2021 — [JAIR](https://www.jair.org/index.php/jair/article/download/12125/26676/26691)
- How it works: CL builds the "confident joint" to estimate the joint distribution of noisy (given) and true labels, then prunes noisy examples. It needs only noisy labels and predicted probabilities, so it works with any model. The authors prove realistic sufficient conditions under which CL exactly finds label errors and exactly estimates the joint distribution — [JAIR](https://www.jair.org/index.php/jair/article/download/12125/26676/26691)
- Reported results: higher ResNet accuracy on a cleaned ImageNet, and better than 7 recent learning-with-noisy-labels methods on CIFAR — [JAIR](https://www.jair.org/index.php/jair/article/download/12125/26676/26691)
- cleanlab is the reference implementation. The core call is `label_issues = cl.find_label_issues(data, labels)`, with `CleanLearning` wrapping any sklearn-compatible classifier — [cleanlab GitHub](https://github.com/cleanlab/cleanlab)

**"Pervasive Label Errors in Test Sets Destabilize ML Benchmarks" (Northcutt, Athalye & Mueller, NeurIPS 2021 Datasets & Benchmarks, oral)**
- Covers the test sets of 10 widely used vision, NLP and audio datasets. Estimated average error rate is at least 3.3% (one ResearchGate preprint copy says 3.4%, so the figure depends on the version) — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749); [NeurIPS D&B PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/f2217062e9a397a1dca429e7d70bc6ca-Paper-round1.pdf)
- ImageNet validation: 2,916 errors (about 6%). QuickDraw: an estimated 5 million or more errors (about 10%). Candidates were flagged by CL and then validated by human (crowdsourced) review — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749)
- Ranking flip: on ImageNet with corrected labels, ResNet-18 outperforms ResNet-50 if the prevalence of originally mislabeled test examples rises by just 6%. Conclusion: on noisy real-world data, lower-capacity models can sometimes be more useful than higher-capacity ones — [arXiv 2103.14749](https://arxiv.org/abs/2103.14749); [OpenReview](https://openreview.net/forum?id=XccDXrDNLek)

**Dataset cartography (Swayamdipta et al., EMNLP 2020)**
- Each training example gets two scores across epochs: *confidence* (mean probability of the gold label) and *variability* (standard deviation of that probability). Three regions result:
  - **easy-to-learn** (high confidence, low variability): the most frequent group;
  - **hard-to-learn** (low confidence, low variability): often mislabeled;
  - **ambiguous** (high variability): contributes most to out-of-distribution generalization.
  — [ACL Anthology](https://aclanthology.org/2020.emnlp-main.746/)

**Area Under the Margin, AUM (Pleiss et al., NeurIPS 2020)**
- AUM is the margin (logit of the assigned class minus the largest other logit), averaged over training epochs. A negative AUM suggests a wrong label. The method adds an extra fake class of deliberately mislabeled "threshold samples" to learn a cutoff, and removes samples whose AUM falls below it — [NeurIPS 2020 PDF](https://proceedings.neurips.cc/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf); [author blog](https://geoffpleiss.com/blog/aum.html); [code](https://github.com/asappresearch/aum)
- WebVision50: removing 17% of the training data improved test error by 2.6 points (absolute) — [NeurIPS 2020 PDF](https://proceedings.neurips.cc/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf)
- Caveat: on NLP datasets AUM filtering also removes a significant number of correctly labeled points — [BlackboxNLP 2021](https://aclanthology.org/2021.blackboxnlp-1.22v1.pdf)

**Data valuation and influence**
- Data Shapley (Ghorbani & Zou, ICML 2019) gives each training point an equitable value for its contribution to predictor performance. Low-value points tend to be mislabeled examples or outliers — [Semantic Scholar](https://www.semanticscholar.org/paper/Data-Shapley:-Equitable-Valuation-of-Data-for-Ghorbani-Zou/b7a717233ec3ff37385ab1b06816d0ca375f5bb3); [Morning Paper summary](https://blog.acolyer.org/2019/07/15/data-shapley/)
- Data Shapley has limits for data selection. A 2024 paper, "Rethinking Data Shapley for Data Selection Tasks: Misleads and Merits", argues it can mislead — [arXiv 2405.03875](https://arxiv.org/pdf/2405.03875)
- Influence functions (Koh & Liang, ICML 2017, pp. 1885–1894): later work describes them as tracing each training example's influence on a test loss. Dropping or downweighting harmful training examples can improve test loss, and mislabeled examples can have outsized influence — [NeurIPS 2022 source-aware influence paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/1343edb2739a61a6e20bd8764e814b50-Paper-Conference.pdf)

### Inferences
- Kaggle recipe:
  1. Get K-fold out-of-fold probabilities.
  2. Run `cleanlab.filter.find_label_issues`.
  3. Inspect the top-ranked suspects by hand.
  4. Pick one: (a) drop, (b) relabel, or (c) downweight. Decide by CV, and make sure the CV labels themselves are not "cleaned" with information from the model being scored.
- A noisy *test or LB* set means small leaderboard gaps between models may be noise (the ResNet-18 vs ResNet-50 flip). Prefer robust, lower-variance models and do not over-tune to the public LB.
- Cartography and AUM come almost for free: log per-sample gold-label probability and margin at every epoch. For the 18-class authorship task, "hard-to-learn" passages may be mislabeled or genuinely indistinguishable between sibling models. "Ambiguous" ones are where better features help most.
- (Prior knowledge, not re-verified) cleanlab also has `Datalab` for outliers, near-duplicates and non-IID checks, plus multi-annotator support.

### Gaps
- I could not get per-dataset error rates for the other 8 datasets in the Pervasive Label Errors paper (MNIST, CIFAR-10/100, Caltech-256, IMDB, 20news, Amazon Reviews, AudioSet), the human-validation counts, or the VGG-11 vs VGG-19 flip details on CIFAR-10, because full-text fetch failed.
- I could not verify the "mislabeled example fixing" experiment in the original Koh & Liang paper.
- I could not check the exact cleanlab API parameters, such as `filter_by='prune_by_noise_rate'`, against the current docs.

---

## 2. Dataset documentation: Datasheets, Data Statements, Data Cards, Model Cards

### Takeaway
Documentation frameworks are question checklists that bring hidden assumptions to the surface: why the data exists, what it contains, how it was collected and labelled, preprocessing, intended and out-of-scope uses, and maintenance. For a competitor, the questions are a ready-made audit script for any dataset you are handed.

### Cited Findings
- Datasheets for Datasets (Gebru et al.): first arXiv preprint 2018, later published in Communications of the ACM. It proposes a short document to accompany every dataset, so creators and users communicate better. Its questions cover motivation, composition, collection process and other areas — [arXiv 1803.09010](https://arxiv.org/pdf/1803.09010); [CACM](https://cacm.acm.org/research/datasheets-for-datasets/)
- A CSCW study reports that datasheets help ML engineers notice and understand ethical issues in training data — [Boyd, CSCW](http://karenboyd.org/Datasheets_Help_CSCW.pdf)
- Data Statements for NLP (Bender & Friedman, 2018) are documentation aimed specifically at language datasets. They were updated in 2021, and their questions can be folded into a datasheet — [Hugging Face documentation landscape](https://huggingface.co/docs/hub/en/model-card-landscape-analysis)
- Data Cards (Pushkarna, Zaldivar & Kjartansson, FAccT '22, Google) are structured summaries of dataset facts for stakeholders across the dataset lifecycle. Google pairs them with the "Data Cards Playbook" — [ACM DL](https://dl.acm.org/doi/10.1145/3531146.3533231); [HF landscape](https://huggingface.co/docs/hub/en/model-card-landscape-analysis)
- Model Cards (Mitchell et al., 2019) document trained models: evaluation, intended use and related issues. They drew on both Data Statements and Datasheets — [HF landscape](https://huggingface.co/docs/hub/en/model-card-landscape-analysis)
- Reusable documentation templates are in use in practice (the Hugging Face and GEM data/model cards) — [arXiv 2108.07374](https://arxiv.org/pdf/2108.07374)

### Inferences
- (Prior knowledge, not re-verified) Datasheets are organized into seven sections: Motivation, Composition, Collection Process, Preprocessing/Cleaning/Labeling, Uses, Distribution, Maintenance. Composition questions that matter most for leakage:
  - What does each instance consist of?
  - Is there a label?
  - Are there recommended splits?
  - Are there errors, sources of noise or redundancies?
  - Is it self-contained?
  - Are there relationships between instances?
- (Prior knowledge, not re-verified) Model Cards ask for model details, intended use, factors, metrics, evaluation data, training data, quantitative analyses *disaggregated by factor/slice*, ethical considerations, and caveats.
- Practical use: before modelling, write a one-page "datasheet" for the competition data. For MS/MS:
  - instrument and collision energy;
  - adduct;
  - source library;
  - how candidates were generated;
  - how "novel molecules" were defined.

  For authorship:
  - which models and versions produced the text;
  - prompt set;
  - passages per prompt;
  - decoding settings;
  - post-processing such as truncation, whitespace or encoding.

  Each "relationship between instances" answer (same molecule, same prompt) defines a grouping variable for CV.

### Gaps
- I could not retrieve the exact question count of Datasheets (I believe it is about 57) or the Data Cards "OFTEn" framework details from primary text.

---

## 3. Data audits and slicing: weak supervision, slice discovery, the Data-Centric AI competition, duplicates

### Takeaway
Improving data under a fixed model can add 20+ points (Ng's competition). Weak supervision can stand in for weeks of labelling. Automated slice discovery finds coherent failure subgroups. Deduplication improves models and cuts memorization roughly 10x. All of these are cheap, high-yield audits.

### Cited Findings
**Weak supervision (Snorkel)**
- Snorkel (Ratner et al., PVLDB 11(3):269–282, Nov 2017): users write labeling functions (heuristics with unknown accuracies and correlations), and Snorkel models and combines their outputs instead of hand-labelling — [ACM DL](https://dl.acm.org/doi/10.14778/3157794.3157797)
- In the user study, subject-matter experts built models 2.8x faster, with an average 45.5% better predictive performance than after seven hours of hand labelling. Snorkel gave a 132% average improvement over prior heuristic approaches and came within an average 3.60% of large hand-curated training sets — [PMC copy](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7075849/); [Stanford DAWN](https://dawn.cs.stanford.edu/publications/snorkel/snorkel-rapid-training-data-creation-weak-supervision)

**Slice discovery**
- Domino (Eyuboglu et al., ICLR 2022 oral) uses cross-modal embeddings (CLIP, ConVIRT, MIMIC-CLIP, EEG-CLIP) and an error-aware mixture model over embeddings, labels and predictions. It beats prior slice-discovery methods by double digits in its evaluation framework and generates natural-language slice descriptions — [arXiv 2203.14960](https://arxiv.org/html/2203.14960v3); [OpenReview](https://openreview.net/pdf?id=FPCMqjI0jXN)
- Domino's slicing is inspired by Spotlight (d'Eon et al.), which searches embedding space for high-loss regions. A later critique says relying on high loss gives insufficient discrimination between different failure modes — [arXiv 2203.14960](https://arxiv.org/html/2203.14960v3); [CB-SLICE](https://arxiv.org/html/2605.29836)

**Andrew Ng's Data-Centric AI Competition (2021)**
- Setup:
  - fixed model, a cut-off ResNet50 trained for 100 epochs;
  - about 1,500 handwritten Roman numerals (i–x);
  - participants could only change the data and the train/validation split, with the dataset capped at 10,000 images.
  — [The Batch](https://www.deeplearning.ai/the-batch/congrats-to-the-data-centric-competition-winners-category-deeplearning-ai-news)
- Results: baseline 64.4%, and many entrants beat it by more than 20 points. Best-Performance winners scored about 86.0–86.4%. 489 entrants submitted 2,458 unique datasets — [The Batch](https://www.deeplearning.ai/the-batch/congrats-to-the-data-centric-competition-winners-category-deeplearning-ai-news)
- Winners: Best Performance went to Divakar Roy, Innotescus and Synaptic-AnN. Most Innovative went to Mohammad Motamedi, Johnson Kuan and GoDataDriven — [Andrew Ng on X](https://x.com/AndrewYNg/status/1445433969470631938)
- Synaptic-AnN describe their entry as "3rd place (85.45% on the leaderboard)", which conflicts slightly with the 86.0–86.4% range above (likely a different leaderboard snapshot) — [deeplearning.ai blog](https://www.deeplearning.ai/blog/data-centric-ai-competition-synaptic-ann); [Motamedi write-up](https://www.deeplearning.ai/blog/data-centric-ai-competition-mohammad-motamedi); [GoDataDriven write-up](https://www.deeplearning.ai/blog/data-centric-ai-competition-godatadriven)

**Deduplication (Lee et al., ACL 2022; arXiv 2107.06499, 2021)**
- A single 61-word English sentence appears 61,036 times verbatim in C4 train and 61 times in C4 validation (about 0.02% of samples in each) — [ACL Anthology PDF](https://aclanthology.org/2022.acl-long.577.pdf)
- Models trained on undeduplicated C4 emit more than 1% of tokens unprompted as part of memorized sequences. Deduplicated training emits memorized text about 10x less often and reaches the same or better accuracy in fewer training steps — [ACL Anthology PDF](https://aclanthology.org/2022.acl-long.577.pdf); [arXiv](https://arxiv.org/abs/2107.06499)

### Inferences
- Ng's competition shows that fixing the data (relabelling, removing junk, targeted augmentation, a good validation split) is a first-class lever. Its "fixed model, vary data" design is a useful *ablation discipline*: freeze a baseline model and measure each data change by CV.
- Dedup checklist for Kaggle:
  - exact hashes;
  - MinHash/LSH near-duplicates for text;
  - perceptual hashes or embedding-cosine for images;
  - InChIKey/skeleton plus fingerprint similarity for molecules;
  - for MS/MS, spectral cosine for repeated spectra of the same compound.

  Duplicates *within train* bias the loss. Duplicates *across train/val* inflate CV.
- (Prior knowledge, not re-verified) Lee et al.'s two tools are ExactSubstr (suffix array, 50-token threshold) and NearDup (MinHash). They also found that a few percent of standard validation-set examples overlap the training data.
- Slice audit for authorship: break CV down by prompt, length bucket, source model family and topic. For MS/MS: by adduct, collision energy, instrument, mass range, compound class and number of candidates.

### Gaps
- Exact ExactSubstr and NearDup parameters and the train/val overlap percentages were not confirmed this session.
- I found no quantitative Spotlight results.

---

## 4. Train–test contamination and leakage in benchmarks

### Takeaway
Leakage is endemic. Near-duplicates sit in 3.3% of CIFAR-10 and 10% of CIFAR-100 test sets. Leakage was documented in 294 papers across 17 fields, with ML's advantage vanishing once it was fixed (civil-war prediction). LLM benchmarks show up to about 8-point drops on fresh look-alike tests (GSM1k).

### Cited Findings
**CIFAR near-duplicates and ciFAIR (Barz & Denzler)**
- 3.3% of CIFAR-10 and 10% of CIFAR-100 test images have near-duplicates in training — [arXiv 1902.00423](https://arxiv.org/abs/1902.00423); [J. Imaging 6(6):41](http://dx.doi.org/10.3390/jimaging6060041)
- ciFAIR-10/100 replace the duplicate test images with new Tiny Images and keep the training sets unchanged. Data is at cvjena.github.io/cifair — [arXiv 1902.00423](https://arxiv.org/abs/1902.00423)
- On the clean test sets, accuracy fell by a relative 9–14% according to one summary, and up to 12% according to another. Architecture rankings were mostly stable: unchanged on CIFAR-100, and only Wide ResNet and DenseNet swapped on CIFAR-10 — [Barz & Denzler PDF](https://pub.inf-cv.uni-jena.de/pdf/barz2020train.pdf)

**Kapoor & Narayanan, "Leakage and the reproducibility crisis in ML-based science" (Patterns, 2023)**
- Leakage was found in 17 fields, affecting 294 papers, in some cases with "wildly overoptimistic conclusions". The July 2022 arXiv preprint said 329 papers from 20 reviews and called it a lower bound — [Patterns/Cell](https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10499856/); [arXiv 2207.07048](https://arxiv.org/pdf/2207.07048)
- The paper gives a fine-grained taxonomy of 8 types of leakage, "from textbook errors to open research problems", and proposes **model info sheets** as the remedy — [Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9)
- Civil-war prediction case study: once leakage is fixed, the claimed advantage of complex ML models over older logistic-regression models disappears — [Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9)

**LLM benchmark contamination**
- GSM1k (Zhang et al., 2024, Scale AI) has 1,205 new problems mirroring GSM8k. Some models dropped up to 8 points, and several families (Phi, Mistral, some Llama models) showed systematic overfitting while Gemini, GPT and Claude showed little to none. Secondary reports give 10–13 points for some models, depending on the table — [arXiv 2405.00332](https://arxiv.org/pdf/2405.00332); [Scale](https://scale.com/research/llm-performance-grade-school-arithmetic)
- The authors note the overfitting may be indirect (collecting similar data, or selecting checkpoints by benchmark score), not verbatim exposure — [arXiv 2405.00332](https://arxiv.org/pdf/2405.00332)
- A critique: a gap on a "matched" fresh set is also consistent with imperfect matching, or with overfitting to surface conventions — [arXiv 2407.12220](https://arxiv.org/pdf/2407.12220)

### Inferences
- (Prior knowledge, not re-verified) Kapoor & Narayanan's 8 leakage types:
  - L1.1 no train/test split;
  - L1.2 preprocessing on train+test (e.g., normalization or imputation fitted on all data);
  - L1.3 feature selection on train+test;
  - L1.4 duplicates in the dataset;
  - L2 model uses illegitimate features (proxies of the target);
  - L3.1 temporal leakage;
  - L3.2 non-independence between train and test (e.g., same patient or same group);
  - L3.3 sampling bias in the test distribution.

  The two most relevant here are L3.2 (the same molecule or the same prompt on both sides of the split) and L1.4 (duplicates).
- The ciFAIR lesson: duplicates mainly inflate *absolute* scores, and may or may not change rankings. In a retrieval competition with *novel* test molecules, any CV that lets similar molecules cross folds will overstate MRR and may also mis-rank methods that memorize (e.g., nearest-neighbour library lookup) above methods that generalize.
- Practical contamination checks:
  - nearest-neighbour search from each val item to train (embedding cosine, n-gram or MinHash for text, Tanimoto/MCES for molecules);
  - plot CV score against the neighbour-distance bucket. If score collapses at large distance, your CV is leaky relative to a novel-molecule test.

### Gaps
- I could not fetch the Kapoor & Narayanan taxonomy table to confirm the exact labels (listed above from prior knowledge).
- No n-gram-overlap decontamination survey numbers were retrieved.

---

## 5. Famous dataset failures and lessons (ImageNet people, COVID X-rays, Kaggle leaks, WILDS)

### Takeaway
Models exploit whatever correlates with the label: hospital markers, row construction, image tiling, synthetic rows. The classic failures all come from not asking how the data was produced and how it differs between sources or splits.

### Cited Findings
**ImageNet person subtree**
- Yang et al. (FAT* 2020), "Towards Fairer Datasets", name three problem sources in ImageNet's person subtree: the stagnant WordNet vocabulary, the attempt to illustrate every category with images, and unequal representation within concepts — [Princeton](https://gfx.cs.princeton.edu/pubs/Yang_2020_TFD/index.php); [image-net.org](https://www.image-net.org/filtering-and-balancing/)
- 1,593 of the 2,832 person categories were flagged as potentially offensive or unsafe and filtered. ScienceDaily gives the denominator as 2,932 (about 54%); the paper uses 2,832 — [ScienceDaily](https://www.sciencedaily.com/releases/2020/02/200214105246.htm); [arXiv 2206.04179](https://arxiv.org/pdf/2206.04179)

**COVID-19 imaging models**
- Roberts et al., Nature Machine Intelligence 3, 199–217 (15 March 2021): 2,212 studies were found, 415 passed initial screening, and 62 were included after quality screening. **None** of the 62 models was of potential clinical use, owing to methodological flaws and biases — [Nature MI](https://www.nature.com/articles/s42256-021-00307-0); [Cambridge news](https://www.cam.ac.uk/research/news/machine-learning-models-for-diagnosing-covid-19-are-not-yet-suitable-for-clinical-use)
- DeGrave, Janizek & Lee, Nature Machine Intelligence 3(7):610–619 (2021), "AI for radiographic COVID-19 detection selects shortcuts over signal":
  - saliency maps highlighted laterality markers (text tokens burned into radiographs);
  - swapping markers between COVID-positive and COVID-negative images shifted predictions significantly more than swapping random patches;
  - the models exploited systematic differences between data sources (COVID-positive and COVID-negative came from different datasets) in markers and patient positioning;
  - they "appear accurate, but fail when tested in new hospitals".
  — [Nature MI](https://www.nature.com/articles/s42256-021-00338-7); [PMC preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC7523163/)
- A suggested mitigation is cropping to the central 75% of the radiograph, but shortcut learning was "reduced but not resolved" — [Radiology 2023](https://pubs.rsna.org/radiology/doi/full/10.1148/radiol.230379)

**Kaggle leaks**
- Santander Customer Transaction Prediction (2019), the "magic":
  - YaG320 showed that 100,000 of the 200,000 test rows were synthetic: a row is real if at least one of its feature values is unique, otherwise it is fake;
  - the test set had fewer unique values per column than train;
  - the winning-style trick was frequency-encoding each of the 200 variables (or adding a count==1 "unique" flag), computed after removing the fake test rows. One poster reported moving to 0.913 public LB.
  — [Kaggle discussion 88888](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88888); [13th place write-up](https://www.linkedin.com/pulse/winning-13th-place-kaggles-magic-competition-corey-levinson); [21 Magics write-up](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/writeups/vasily-ryazanov-21-magics)
- Santander Value Prediction (2018): Kagglers found a leak tied to how the anonymized data was built (columns and rows were lagged time series). Top scores required exploiting it — [kweonwooj repo](https://github.com/kweonwooj/kaggle_santander_value_prediction_challenge); [Kaggle discussion](https://www.kaggle.com/c/santander-value-prediction-challenge/discussion/61288)
- Airbus Ship Detection (2018): test images were transformed (translated and cropped) versions of overlapping training tiles, because location information was not considered in the split. Organizers released the test masks, moved the old test into train, uploaded new test data and reset the leaderboard — [Kaggle discussion 64388](https://www.kaggle.com/c/airbus-ship-detection/discussion/64388); [Medium lessons](https://medium.com/data-science/lessons-learned-from-kaggles-airbus-challenge-252e25c5efac)
- Even Titanic has a leak (shared ticket/family groups across train/test) — [Recio blog](https://david-recio.com/2022/04/11/titanic-leak.html)

**Dataset shift: WILDS (Koh et al., ICML 2021)**
- Camelyon17-WILDS (train hospitals → unseen hospital): ERM gets about 93.2% in-distribution vs about 70.3% out-of-distribution average accuracy — [ICML PDF](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf); [appendix](https://proceedings.mlr.press/v139/koh21a/koh21a-supp.pdf)
- iWildCam2020-WILDS (new camera traps): macro-F1 falls from 47.0 (in-distribution) to 31.0 (out-of-distribution) — [arXiv 2012.07421](https://arxiv.org/html/2012.07421v3)
- The OOD gap is not because the OOD test is intrinsically harder; the paper checked this with a control. CivilComments is the subpopulation-shift example — [arXiv 2012.07421](https://arxiv.org/html/2012.07421v3)

### Inferences
- Common pattern: the label correlates with the *source or process* (hospital, row-synthesis procedure, tile position) rather than the content. Audits to run:
  1. Train a classifier to predict the *source*, or train-vs-test (adversarial validation).
  2. Inspect saliency and feature importance for non-semantic features.
  3. Check how test rows were generated.
- In the authorship task, the "source" confound is each generator's *prompt/topic mix* or *formatting pipeline*. In MS/MS, it is *instrument, library or adduct* differences between train and test.
- Leaks like Santander "magic" are legitimate on Kaggle when the rules allow them, but the lesson for honest CV is to model test-generation artifacts explicitly.
- WILDS numbers set expectations: a 20+ point ID→OOD drop is normal under real shift. The "novel molecules" test in CASMI 2026 is exactly such a shift.

### Gaps
- No reliable source was found on a Two Sigma or Instacart leak; I am not reporting them.
- I could not retrieve Birhane & Prabhu's "Large image datasets: a pyrrhic win" details.

---

## 6. Molecular data specifics: splits, duplicates, canonicalisation, MassSpecGym

### Takeaway
For molecules, random splits leak through near-identical structures. Scaffold splits help but still leak similar molecules. Canonical-SMILES dedup misses tautomer and stereo duplicates. MassSpecGym's answer is an MCES-distance-based clustering split, plus formula- or mass-matched candidate sets capped at 256. A 2026 audit still found evaluation problems in 17 of the 26 papers that reported MassSpecGym results.

### Cited Findings
**Splits**
- A 2026 survey says random splits often give overly optimistic estimates, and that MoleculeNet scaffold-split results may partly reflect leakage and metric inflation — [arXiv 2604.16586](https://arxiv.org/html/2604.16586v1)
- "Scaffold Splits Overestimate Virtual Screening Performance" (2024): molecules with different Bemis–Murcko scaffolds are often still highly similar, so scaffold splits remain optimistic — [arXiv 2406.00873](https://arxiv.org/pdf/2406.00873)
- Counterpoint: a 2025 JCIM study found classical ML and GNN models perform similarly on scaffold and random splits on its datasets — [JCIM 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00475)

**Duplicates and canonicalisation**
- Different tautomers usually get distinct canonical SMILES, so chemically identical compounds escape dedup. Canonical-SMILES filtering can remove one stereoisomer while others remain, which creates leakage. Both points come from an audit of top-ranked TDC ADMET leaderboard models — [JCIM 2026](https://doi.org/10.1021/acs.jcim.6c00819)
- LIT-PCBA: stereoisomers recur across splits, almost always with the same label, so stereo-agnostic memorization looks like generalization — [arXiv 2507.21404](https://arxiv.org/html/2507.21404v1)
- B3DB: canonical-SMILES dedup found only 3 duplicates, while InChIKey-skeleton dedup found 1,490 duplicated skeletons (non-peer-reviewed preprint) — [Preprints.org](https://www.preprints.org/frontend/manuscript/7c64c5572f4d6075669a7ee9103277bc/download_pub)
- RDKit's canonical tautomer choice can depend on stereochemistry elsewhere in the molecule — [RDKit issue #7499](https://github.com/rdkit/rdkit/issues/7499); [Pat Walters, "The Trouble With Tautomers"](https://patwalters.github.io/The-Trouble-With-Tautomers/)

**MassSpecGym (Bushuiev et al., NeurIPS 2024 Datasets & Benchmarks)**
- Covers about 29k unique molecules and three tasks: de novo generation, retrieval and spectrum simulation — [arXiv 2410.23326](https://arxiv.org/html/2410.23326v1)
- Split: agglomerative clustering of molecules using MCES (maximum common edge subgraph edit distance) as the metric, with a linkage threshold so that molecules in different folds are at least MCES distance 10 apart — [arXiv 2410.23326](https://arxiv.org/html/2410.23326v1)
- The supplement describes a refinement that excludes molecules with MCES < 2 from any test-fold molecule. A Hugging Face note says the released set is strictly disjoint at MCES ≤ 2. This differs from the main-text threshold of 10, so verify against the code — [Supplement](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Supplemental-Datasets_and_Benchmarks_Track.pdf); [HF dataset](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym)
- Motivation: the earlier standard split by 2D InChIKey lets near-duplicate test molecules (Tanimoto > 0.85) leak from train — [arXiv 2410.23326](https://arxiv.org/html/2410.23326v1)
- Retrieval candidates come from PubChem (118M molecules, downloaded 31 May 2024), in two variants, "equal mass" and "equal formula", each capped at 256 candidates per spectrum. The metric is hit rate@k — [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Paper-Datasets_and_Benchmarks_Track.pdf); [arXiv 2602.16507](https://arxiv.org/pdf/2602.16507)
- Candidate-set difficulty matters a great deal. An audit reports tuned MIST top-1 hit rate of 0.194 on the official formula candidates (n=16,042) vs 0.873 with random candidates — [coleygroup Audit_ML_MS_analysis](https://github.com/coleygroup/Audit_ML_MS_analysis/blob/main/DETAILED_RESULTS.md)
- MSAlign reports R@1 of 16.2% on the MCES split vs 53.8% on the formula split of MassSpecGym — [arXiv 2605.19752](https://arxiv.org/pdf/2605.19752)
- "MassSpecGym in the Wild" (2026):
  - found evaluation issues in at least 17 of the 26 papers reporting MassSpecGym results in its first year, in three classes: (i) data leakage, (ii) shortcut learning, (iii) implementation bugs and metric divergence;
  - inconsistent SMILES canonicalization and biased candidate sets let models reach near-perfect recall without learning spectrum–molecule relationships;
  - leakage also enters via external pretraining data, e.g., fingerprint-decoder pretraining that excluded test compounds only by exact 2D InChIKey;
  - the authors released MassSpecGym v1.5 with fixes.
  — [arXiv 2606.19624](https://arxiv.org/abs/2606.19624)
- "What are we optimizing for?" (2026; finding from a secondary summary): there is a Pareto trade-off where models that best reconstruct fingerprints (high Tanimoto) can retrieve worse, and listwise contrastive losses did best for retrieval — [arXiv 2602.16507](https://arxiv.org/html/2602.16507)

### Inferences
**For CASMI 2026 (MRR@25, novel molecules absent from train):**
1. **Standardize first.** Strip salts, neutralize, canonicalize tautomers, decide on a stereo policy, and do it identically for training molecules, candidates and any lookup library. Dedupe at the 2D-InChIKey first block (skeleton). Otherwise the same compound appears as "different" molecules and inflates CV.
2. **Make CV mimic "novel".** Use group K-fold by 2D InChIKey *at minimum*. Better, cluster by MCES or Tanimoto (e.g., Butina at about 0.6–0.7, or MassSpecGym's MCES clustering) and split by cluster.
3. **Report CV broken down by nearest-train-neighbour similarity.** The test likely sits in the low-similarity bucket.
4. **Candidate-set realism.** Build validation candidates the way the competition does (same formula or mass window from PubChem/COCONUT etc.). Random-candidate evaluation can overstate top-1 by about 4x (0.873 vs 0.194).
5. **Audit for shortcuts.** Candidate metadata (e.g., PubChem CID magnitude, popularity, presence in common libraries, SMILES canonical form) can predict the true answer without the spectrum. Test a "candidate-only" baseline that ranks candidates without looking at the spectrum. This is the MS analogue of a hypothesis-only baseline.
6. **Check external pretraining data** (e.g., fingerprint→structure decoders, spectral libraries such as GNPS/MoNA/NIST) for overlap with the test compounds beyond exact-InChIKey matching.
7. Repeated spectra of the same compound at different collision energies or adducts must stay in the same fold.

### Gaps
- I could not resolve whether MassSpecGym's main split threshold is MCES 10, MCES 2 or both (cluster at 10, then exclude < 2 neighbours). Read the repo or code to settle it.
- I could not confirm MoleculeNet's original split definitions or numbers from the 2018 paper.
- I could not obtain total spectrum count or baseline numbers from the MassSpecGym paper (fetch failed). (Prior knowledge, not re-verified: about 231k spectra.)

---

## 7. Text/LLM-authorship data specifics: artifacts, hypothesis-only baselines, prompt leakage, topic confounds

### Takeaway
Crowdsourced and generated text carries label-revealing artifacts that a trivial partial-input model can exploit: hypothesis-only NLI scored about 67% on SNLI. AI-text detectors latch onto prompt-specific styles, formatting and extraction artifacts. For multi-passage-per-prompt authorship data, always split CV by prompt and run "partial-input" and "metadata-only" baselines.

### Cited Findings
- Gururangan, Swayamdipta, Levy, Schwartz, Bowman & Smith, "Annotation Artifacts in NLI Data" (NAACL 2018):
  - a hypothesis-only text classifier labels about 67% of SNLI and about 53% of MultiNLI correctly (three-way task, chance about 33%);
  - negation and vagueness words correlate strongly with particular classes;
  - progress on NLI was likely overestimated.
  — [ACL Anthology](https://aclanthology.org/N18-2017/); [PDF](https://aclanthology.org/N18-2017.pdf)
- A clinical-NLI replication reports 67.0 / 53.9 F1 for the hypothesis-only baseline on SNLI / MultiNLI — [arXiv 1808.06752](https://arxiv.org/pdf/1808.06752)
- Prompt-specific shortcuts (Park et al., 2024):
  - AI-text detectors trained on a narrow set of prompts rely on spurious stylistic cues tied to those instructions;
  - the FAILOpt attack exploits this to evade detectors;
  - adding AI text generated from *other* prompts to training alleviates the bias and improves robustness.
  — [arXiv 2406.16275](https://arxiv.org/pdf/2406.16275)
- Formatting artifacts (2026): in one benchmark the strongest detector cue was the PDF-extraction artifact. Human texts kept their line breaks, while generated texts were a single prose block. Artifact exploitability is tokenizer-dependent, so scores are not comparable across tokenizer families — [arXiv 2610.13583](https://arxiv.org/html/2610.13583)
- MAGE (ACL 2024): entity tags such as "ORDINAL"/"DATE" can act as detection shortcuts — [MAGE](https://aclanthology.org/2024.acl-long.3.pdf)
- A regex baseline's top features include em-dashes, formal connectives and prompt-leakage templates — [SV-Detect](https://arxiv.org/pdf/2606.07313)
- The GenAI Content Detection Task 3 shared task flags possible test leakage (public human-written sources) and discusses confounding factors — [arXiv 2501.08913](https://arxiv.org/pdf/2501.08913)
- Generation artifacts relate mainly to token co-occurrence — [LREC 2022](https://aclanthology.org/2022.lrec-1.744.pdf)

**Kaggle "LLM – Detect AI Generated Text" (2023–24) lessons**
- 1st place: fine-tuned many LLMs with a causal-LM objective on the PERSUADE corpus to *generate* student-like essays (building a training distribution matched to the test), plus a contrastively trained embedding model with kNN neighbours for each test essay — [rbiswasfc GitHub](https://github.com/rbiswasfc/llm-detect-ai)
- 3rd place: TF-IDF pipeline plus 12 DeBERTa-v3-large models; a deobfuscator corrected texts only when they had more than 15 errors (the test had injected character noise); hard-example mining; per-prompt post-processing with UMAP distances — [awsaf49 GitHub](https://github.com/awsaf49/detect-fake-text)
- 19th place: never found CV/LB correlation when generating data resembling the host's, so CV was unreliable — [19th place write-up](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/writeups/darek-k-eczek-19th-place-solution)
- 34th place: focused on balancing the label distribution with extra data — [Kaggle discussion 470353](https://www.kaggle.com/c/llm-detect-ai-generated-text/discussion/470353)

### Inferences
**For the 18-class LLM-authorship competition (several passages per prompt):**
1. **GroupKFold by prompt** (and by topic cluster if prompts are paraphrased). Passages from the same prompt share topic, entities and length targets. Random K-fold lets the model learn "prompt → which models answered it" or topic cues, inflating CV. This is Kapoor & Narayanan's non-independence leakage (L3.2, prior knowledge).
2. **Partial-input and metadata baselines**, the analogue of hypothesis-only:
   - length, punctuation and whitespace stats only;
   - formatting only (markdown, line breaks, em-dashes, list bullets, Unicode quotes);
   - first or last 20 tokens only;
   - a prompt-ID-only classifier.

   If any of these scores well above chance (1/18 ≈ 5.6%), that signal is an artifact, which may be exploitable on the LB, but check whether it holds across prompts.
3. **Check the prompt-model design.** If the label distribution differs by prompt (e.g., some models answered only certain prompts), a topic model alone predicts the label. Compute label|prompt entropy and train a TF-IDF-on-topic-words baseline.
4. **Text normalization ablation.** Strip or normalize whitespace, Unicode, refusal boilerplate ("As an AI…"), and echoed prompt text (*prompt leakage*: generations often restate the prompt). Measure the CV change. Big drops mean you relied on the artifact.
5. **Use training dynamics and cleanlab on the 18-way OOF probabilities** to find passages that look mislabeled or are genuinely confusable between sibling models (e.g., two versions of the same family). Consider hierarchical labels (family → version) and inspect the confusion matrix by prompt.
6. **Match training data to the test distribution**, as the Detect-AI winners did. If the test prompts are new, CV must hold out prompts. If test passages come from the same prompts as train, a prompt-aware feature (e.g., per-prompt normalization, or kNN within a prompt) may be legitimately useful. Check the competition rules and data description.

### Gaps
- I found no paper that directly quantifies inflation from random vs prompt-grouped splits in authorship attribution; the recommendation above is inferred from the leakage literature.
- The organizers' 18-class competition design (which models, how prompts are assigned) is unknown to me and should be documented from the competition page.
