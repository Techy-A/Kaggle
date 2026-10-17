---
tags: [study, report, introspection, error-analysis, data-centric]
updated: 2026-10-17
---
# Read the failures, then distrust every shortcut

Top data scientists and Kaggle grandmasters rarely start introspection with attribution plots. They start by reading about a hundred failures by hand and sorting them into categories. Each category's share of errors caps what fixing it can gain. They then change one thing at a time against a validation split built to reproduce the test's novelty, and they treat every high-scoring feature as a suspected shortcut until a partial-input baseline or an ablation clears it. Interpretability tools do earn a place as hypothesis generators: held-out permutation importance over clusters of correlated features, TreeSHAP for explaining why one item outranks another, ALE instead of PDP when features are correlated, and input ablation instead of attention maps. Each of these answers a narrower question than it seems to, and several popular saliency methods fail basic sanity checks. The data side matters just as much. Benchmark test sets carry at least 3.3% label errors, leakage distorted results in 294 papers across 17 fields, and a 2026 audit found evaluation problems in 17 of 26 papers reporting MassSpecGym results, where baselines that never look at the spectrum reached 50–99% HR@1. Kaggle winners turn all this into a recognisable routine. They choose the CV group key that makes CV move with the leaderboard, track retriever recall separately from the end metric, correct explicitly for the expected seen/novel prior, and spend one final pick on a CV-backed hypothesis about the private test.

For [[Competitions/CASMI26|CASMI 2026]], the priority is a per-query funnel decomposition (formula, then candidate recall, then rank) on the unseen-molecule bench, with PubChem-popularity features isolated and ablated for class 3. For [[Competitions/Name That LLM|Name That LLM]], the priorities are a prompt-grouped CV, an 18×18 confusion matrix sorted by lab, a transformation-ladder audit of the LoRA model, and a joint per-prompt and Sinkhorn assignment that is validated on train before anyone trusts it.

## A hundred hand-read errors set the ceiling on every fix

Every respected source on error analysis converges on the same manual procedure, and they are surprisingly specific about it. Andrew Ng's recipe is to gather about **100 misclassified dev examples** and tag each in a spreadsheet. Columns are failure categories that you invent as you look, plus a free-text comment, and one example can tick several columns. Each category's share is a hard **ceiling**: if only 5% of errors are dogs, a perfect dog fix takes a 90%-accurate system to at most 90.5% ([Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)). Two refinements matter in competitions. The first is to split the dev data into an "Eyeball" set you read and a "Blackbox" set you only score. If Eyeball accuracy rises much faster than Blackbox, you have overfit your own error analysis. The second is that the importance of mislabelled dev data depends on the accuracy level: the same 0.6% of label noise is 6% of errors at 90% accuracy but **30% of errors at 98%**, and only at that point is cleaning the labels worth the effort ([Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)). For pipelines, Ng's "error analysis by parts" swaps in a perfect output for one stage at a time and assigns the error to whichever stage, once fixed, makes the final answer correct. This is exactly the experiment a two-stage retrieval system needs.

Hamel Husain's LLM-era version adds a vocabulary for the same discipline. **Open coding** means writing free-form notes on at least 30 traces, noting the *first* failure in each. **Axial coding** means grouping those notes into a failure taxonomy and counting, and he calls it "the most important step". You stop at about 100 traces, once new categories stop appearing ([Husain & Shankar, Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)). In his NurtureBoss case study, three issues accounted for over 60% of problems, and fixing date handling alone raised success **from 33% to 95%** ([Husain, Field Guide](https://hamel.dev/blog/posts/field-guide/)). Karpathy's recipe supplies the checks to run before error analysis even makes sense:

- fix the seed;
- verify the loss at initialisation (−ln(1/n), which is **ln 18 ≈ 2.89** for an 18-way head);
- beat an input-independent baseline;
- overfit one batch;
- visualise the exact tensor entering the model;
- "complexify only one at a time", confirming each added signal gives the boost you expected ([Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)).

Google's Rules of ML translates this to ranking:

- An error is "a pair where a positive was ranked lower than a negative" (Rule #26).
- Successive models are compared by the position-weighted symmetric difference of their result lists, and you read the queries where that difference is large (Rule #24).
- A change that improves log loss but hurts the ranking metric should be rejected (Rule #25).
- Document-only features that boost popular items for every query are a known trap (Rule #35).

All four are in [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml).

Ablation discipline is the statistical half of the same habit, and the literature shows that single-run comparisons are routinely wrong. Bouthillier et al. find that variance from data sampling, initialisation and hyperparameters "impact markedly the results". They propose declaring A better than B only when **P(A>B) ≥ 0.75 across paired splits or seeds**, a threshold that held up across all their case studies, and they found that a single-point comparison gave roughly a 10% false-positive rate in simulation ([arXiv 2103.03098](https://arxiv.org/pdf/2103.03098v1)). Dodge et al. found published comparisons whose conclusion would flip under a different hyperparameter-search budget ([arXiv 1909.03004](https://arxiv.org/abs/1909.03004)). Lipton and Steinhardt list "gains actually stem from hyper-parameter tuning" as a recurring failure of attribution in papers ([arXiv 1807.03341](https://arxiv.org/abs/1807.03341)).

Grandmasters describe the same loop in plainer words. CPMP treats modelling as "an experimental science, just like physics", where a tested change "becomes your new baseline" only if it wins ([SED 1817](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)). Chris Deotte says that after 80 competitions, "I'll look at its output. It'll always be getting something wrong. I oftentimes know exactly where to look" ([SED 1817](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)). The NVIDIA KGMoN team puts the throughput argument bluntly: "The biggest lever we have … is the number of high-quality experiments we can run" ([NVIDIA Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)).

In practice, the protocol for both our competitions follows directly from these sources. Fix the grouped folds. Run every variant on the same folds, with 3–5 seeds for neural models. Accept a change only if it wins at least **12 of 15 fold×seed pairs** (5×3 at γ = 0.75). Keep a "didn't work" log alongside the ablation table, as described in [[10 Competition Playbook]] and [[01 Validation & Leakage]]. For MRR@25, a paired bootstrap over queries is the cheap significance test. That is our extension of the paired-comparison logic; no source here validates it specifically for MRR.

## Attribution tools explain the model, never the world

### Permutation importance beats gain, but correlated features fool both

The first rule of tabular introspection is that built-in tree importances measure what training used, not what generalises. scikit-learn warns that impurity importance is "strongly biased" toward high-cardinality features and is computed on training statistics. In its Titanic demo, a purely random numeric column ranks highly ([sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html); [sklearn example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)). Strobl et al. traced the same bias in random forests to biased split selection and bootstrap sampling ([Strobl et al. 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25)). LightGBM's default `"split"` importance and its `"gain"` alternative are both training-set measures of this kind ([LightGBM API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html)). Their one honest use is spotting leaks: a single feature that dominates absurdly is the signature of label information leaking into the inputs.

Held-out permutation importance is the better default. The sklearn recipe shuffles each column with `n_repeats=30` and keeps only features whose mean importance minus two standard deviations is still above zero. It carries three warnings. Importance is model-specific, it depends on the metric, and **correlated features split or hide each other's importance**. On the breast-cancer data, a naive reading concludes that every feature is unimportant ([sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html)). Hooker, Mentch and Zhou go further. Permute-and-predict methods (permutation importance, PDP, ICE) force the model to extrapolate into impossible rows, so they "can vastly over-emphasize correlated features". They recommend refit or drop-column importance instead ([arXiv 1905.03151](https://arxiv.org/abs/1905.03151)). PDPs share the independence assumption and can hide heterogeneity that ICE curves reveal ([sklearn 5.1](https://scikit-learn.org/stable/modules/partial_dependence.html)). ALE plots avoid both extrapolation and omitted-variable bias at lower cost ([Apley & Zhu, arXiv 1612.08468](https://arxiv.org/abs/1612.08468)).

SHAP is the strongest local tool for GBDTs. TreeSHAP is exact in O(TLD²) time and is built into XGBoost and LightGBM ([Lundberg et al., arXiv 1802.03888](https://arxiv.org/abs/1802.03888)). It has two traps. The first is that the explanation's meaning depends on the background: with a background dataset, TreeExplainer defaults to "interventional" (100–1,000 samples recommended), and without one it uses "tree_path_dependent" ([SHAP docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)). The path-dependent variant can give non-zero credit to features the model never uses ([Molnar, IML: SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)). The second is that SHAP is not causal. In the SHAP authors' own simulation, "Bugs reported" receives positive attribution despite having zero causal effect ([SHAP docs: causal caveats](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html)).

| Tool | Question it actually answers | Main failure mode | Use it for |
|---|---|---|---|
| Split/gain (MDI) | How much did training use this feature? | Cardinality bias; training-set only | Spotting a dominant leaked feature |
| Held-out permutation | How much does held-out score drop when this column is destroyed, given the rest? | Correlated features hide each other; extrapolation | Feature triage, using clustered groups and the true metric |
| Drop-column / refit | Is the information unique and needed? | Compute cost | Final keep/drop decisions |
| TreeSHAP | How did the model allocate this one prediction? | Background-dependent; not causal | Pairwise rank flips, interaction discovery |
| PDP / ICE | What is the average (or per-row) effect of moving one feature? | Impossible rows under correlation | Spotting heterogeneity (ICE) |
| ALE | What is the main effect when features are correlated? | Uneven quantile bins | Similarity-score shape checks |

### For neural models, ablate the inputs instead of reading attention

Attention weights are weak evidence. Jain and Wallace showed that attention is often uncorrelated with gradient and leave-one-out importance, and that permuting it barely changes outputs ([arXiv 1902.10186](https://arxiv.org/abs/1902.10186)). Wiegreffe and Pinter's rebuttal turns on definitions and adds a seed-variance baseline ([arXiv 1908.04626](https://arxiv.org/abs/1908.04626)). Both papers studied RNNs, not modern decoders, so how well either result transfers to a LoRA-tuned LLM is unknown.

Probes need **control tasks**. Hewitt and Liang assign random labels by word type, and a probe counts as meaningful only if it is *selective*: high accuracy on the real task and low on the control ([arXiv 1909.03368](https://arxiv.org/abs/1909.03368)). Even then, a probe shows that information is decodable, not that the model uses it ([Belinkov, arXiv 2102.12452](https://arxiv.org/abs/2102.12452)).

Saliency fares worst. Adebayo et al. found methods whose maps stay the same when the model's weights are randomised, which makes them useless for debugging ([arXiv 1810.03292](https://arxiv.org/abs/1810.03292)). Under ROAR, many methods were "not better than a random designation" ([arXiv 1806.10758](https://arxiv.org/abs/1806.10758)). Integrated gradients rests on sound axioms ([arXiv 1703.01365](https://arxiv.org/abs/1703.01365)), but it still has to pass the same randomisation check.

The logit lens and the more reliable tuned lens give cheap views of what each layer predicts ([LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens); [arXiv 2303.08112](https://arxiv.org/abs/2303.08112)). For a classification head, a per-layer linear probe is the practical analogue. Sparse autoencoders are research tooling: Anthropic's monosemanticity study trained them on 8 billion activation samples from a one-layer model ([transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features/index.html)), which is not a one-week competition tool.

For practitioners, the most trustworthy introspection is behavioural and causal: perturb or remove an input property and measure how the output changes.

### Shortcut detection is a partial-input baseline plus an insertion test

Geirhos et al. define shortcuts as rules that "perform well on standard benchmarks but fail to transfer". Their examples include cows that go unrecognised on beaches and a pneumonia model that read the hospital token instead of the lung ([arXiv 2004.07780](https://arxiv.org/abs/2004.07780)).

The two canonical detection moves both have famous proofs.

The first is the **partial-input baseline**. A classifier that sees only the NLI hypothesis scores about **67% on SNLI** against 33% chance ([Gururangan et al., arXiv 1803.02324](https://arxiv.org/abs/1803.02324)). BERT-era models collapse on HANS, a set constructed so that the lexical-overlap heuristic fails ([arXiv 1902.01007](https://arxiv.org/abs/1902.01007)).

The second is the **insertion/removal test**. A PASCAL VOC "horse" classifier relied on a source tag present in about a fifth of horse images. Removing the tag killed the prediction, and pasting it onto a car turned the car into a horse ([Lapuschkin et al., arXiv 1902.10178](https://arxiv.org/abs/1902.10178)). Their SpRAy method found this by spectral clustering of explanations across the whole dataset. DeGrave et al. ran the same swap test on COVID radiographs: exchanging laterality markers moved predictions more than exchanging random patches ([Nature MI](https://www.nature.com/articles/s42256-021-00338-7)).

Simplicity bias explains why this keeps happening. Networks can rely exclusively on the simplest feature even when it is less predictive, and ensembles do not reliably fix this ([arXiv 2006.07710](https://arxiv.org/abs/2006.07710)).

## Datasets fail quietly, and the same five failures recur

### Label noise is common enough to reorder models

Northcutt, Athalye and Mueller estimate **at least 3.3% label errors** across the test sets of 10 major benchmarks, including **2,916 errors (about 6%) in ImageNet val**. With corrected labels, ResNet-18 overtakes ResNet-50 if the share of originally mislabelled test examples rises by just 6% ([arXiv 2103.14749](https://arxiv.org/abs/2103.14749)).

Detection tools are cheap and model-agnostic:

- **Confident learning (cleanlab)** needs only out-of-fold probabilities ([JAIR](https://www.jair.org/index.php/jair/article/download/12125/26676/26691); [cleanlab](https://github.com/cleanlab/cleanlab)).
- **Dataset cartography** places each example by its confidence and variability across epochs. "Hard-to-learn" examples are often mislabelled, and "ambiguous" ones drive out-of-distribution generalisation ([ACL Anthology](https://aclanthology.org/2020.emnlp-main.746/)).
- **AUM**, the margin averaged over training, let Pleiss et al. remove 17% of WebVision50 and cut test error by 2.6 points ([NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/c6102b3727b2a7d8b1bb6981147081ef-Paper.pdf)). On NLP data it also removes many correct labels ([BlackboxNLP 2021](https://aclanthology.org/2021.blackboxnlp-1.22v1.pdf)).
- **Data Shapley** surfaces low-value points ([Semantic Scholar](https://www.semanticscholar.org/paper/Data-Shapley:-Equitable-Valuation-of-Data-for-Ghorbani-Zou/b7a717233ec3ff37385ab1b06816d0ca375f5bb3)), though it can mislead when used to select data ([arXiv 2405.03875](https://arxiv.org/pdf/2405.03875)).

These tools produce a ranked list for human review, not a delete list.

Data-centric work also pays off on its own. In Ng's Data-Centric AI Competition the model was frozen and only the data could change. Many of the 489 entrants beat the **64.4% baseline by more than 20 points**, and the winners reached about 86% ([The Batch](https://www.deeplearning.ai/the-batch/congrats-to-the-data-centric-competition-winners-category-deeplearning-ai-news)). Snorkel's labelling functions let experts build models 2.8× faster than hand labelling ([PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7075849/)). Domino finds coherent failure slices automatically with cross-modal embeddings ([arXiv 2203.14960](https://arxiv.org/html/2203.14960v3)).

### Documentation questions double as an audit script

Datasheets for Datasets ([arXiv 1803.09010](https://arxiv.org/pdf/1803.09010)), Data Statements, Data Cards and Model Cards ([HF landscape](https://huggingface.co/docs/hub/en/model-card-landscape-analysis)) are question checklists. Their best competition use is to answer "what are the relationships between instances?" before modelling, because every answer (same molecule, same prompt, same patient) defines a CV grouping variable.

### Duplicates and non-independence leak across splits

Lee et al. found one 61-word sentence repeated **61,036 times** in C4. Deduplicated training emitted memorised text about 10× less often, with equal or better accuracy ([ACL 2022](https://aclanthology.org/2022.acl-long.577.pdf)). In CIFAR, **3.3% of CIFAR-10 and 10% of CIFAR-100 test images** have near-duplicates in train ([arXiv 1902.00423](https://arxiv.org/abs/1902.00423)). Kapoor and Narayanan documented leakage in **294 papers across 17 fields**, and in their civil-war case study ML's advantage over logistic regression disappeared once the leakage was fixed ([Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(23)00159-9)).

### Labels correlate with the data's source, not its content

The famous failures share one cause: the label was correlated with *how or where the data was produced* rather than with its content.

- **COVID-19 models.** Of 62 models that passed quality screening, **none** was of potential clinical use ([Roberts et al., Nature MI](https://www.nature.com/articles/s42256-021-00307-0)). Positive and negative images came from different sources, and models learned those source differences ([DeGrave et al.](https://www.nature.com/articles/s42256-021-00338-7)).
- **ImageNet.** 1,593 of 2,832 person categories were flagged as offensive or unsafe and filtered ([Princeton](https://gfx.cs.princeton.edu/pubs/Yang_2020_TFD/index.php)).
- **Santander Customer Transaction (Kaggle).** Half the test rows were synthetic and could be identified by unique-value counts ([Kaggle 88888](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88888)).
- **Airbus Ship Detection (Kaggle).** Test tiles were shifted crops of training tiles, and the leaderboard was reset ([Kaggle 64388](https://www.kaggle.com/c/airbus-ship-detection/discussion/64388)).

### Real distribution shift costs 20 points or more

WILDS sets realistic expectations. Camelyon17 drops from about **93.2% in distribution to 70.3%** at an unseen hospital ([ICML](https://proceedings.mlr.press/v139/koh21a/koh21a.pdf)). LLM contamination shows the same pattern: on the fresh GSM1k problems, some models lost up to 8 points relative to GSM8k ([arXiv 2405.00332](https://arxiv.org/pdf/2405.00332)).

### Molecules leak through near-duplicate structures

Molecular data has its own versions of these failures. Scaffold splits remain optimistic because molecules with different Bemis–Murcko scaffolds are often still near-identical ([arXiv 2406.00873](https://arxiv.org/pdf/2406.00873)). One 2025 JCIM study found no difference between scaffold and random splits on its datasets ([JCIM 2025](https://pubs.acs.org/doi/10.1021/acs.jcim.5c00475)). Tautomers get distinct canonical SMILES and escape deduplication ([JCIM 2026](https://doi.org/10.1021/acs.jcim.6c00819)). RDKit's canonical tautomer choice can even depend on stereochemistry elsewhere in the molecule ([RDKit #7499](https://github.com/rdkit/rdkit/issues/7499)).

MassSpecGym's answer is MCES-based cluster splitting. The main text describes a distance-10 threshold, while the supplement and Hugging Face card describe a stricter MCES ≤ 2 exclusion. Check the code before relying on either ([arXiv 2410.23326](https://arxiv.org/html/2410.23326v1); [Supplement](https://proceedings.neurips.cc/paper_files/paper/2024/file/c6c31413d5c53b7d1c343c1498734b0f-Supplemental-Datasets_and_Benchmarks_Track.pdf)).

### Generated text carries prompt and formatting artifacts

Text data repeats the pattern in its own form. AI-text detectors trained on narrow prompt sets learn prompt-specific stylistic cues ([arXiv 2406.16275](https://arxiv.org/pdf/2406.16275)). In one 2026 benchmark the strongest detector cue was a PDF-extraction line-break artifact ([arXiv 2610.13583](https://arxiv.org/html/2610.13583)).

### The five recurring failures

Distilled, five failures recur:

1. label noise;
2. duplicates across splits;
3. group non-independence (the same entity on both sides of the split);
4. source confounds;
5. artifacts of how the test set was constructed.

Each has a cheap audit, covered in [[01 Validation & Leakage]].

## Kaggle winners reproduce the test's novelty, then bet one pick on it

### First, make CV move with the leaderboard

Nearly every first-place write-up begins by showing that local CV tracks the leaderboard. Folds and seeds alone do not get them there; the deciding choice is the **group key**.

Raja Biswas's Eedi win is the template. Folds keyed by `QuestionId` were too optimistic and folds keyed by `SubjectId` too pessimistic. GroupKFold on `ConstructId` "yielded a narrow Validation/LB gap and good correlation" ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)). A constant offset is acceptable. Ahmet Erdem's LECR validation sat at 0.764 against a public LB of 0.727, but the two moved together, and his validation score turned out to be his private score ([LECR 1st](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812)). Team Hydrogen never judged a model on a single seed; every check used blends of 3 seeds ([Hydrogen 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)).

Adversarial validation is the standard shift check. Label train rows against test rows and train a classifier: an AUC near 0.5 means no shift, and when the AUC is high, the train rows that look most like test become the validation set ([FastML via KDnuggets](https://www.kdnuggets.com/2016/10/adversarial-validation-explained.html)). Grandmasters run the visual version too. David Austin's embedding UMAPs revealed real versus synthetic images ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)), and a BELKA competitor's chemical-space plot showed that the test library shared nothing with train ([Leash Bio](https://leashbio.substack.com/p/belka-results-suggest-computers-can)).

### Then gain score from data and distillation, and keep models by what they add to the blend

With CV trusted, the gains come from four places.

**Leak-free, multi-round, soft pseudo-labels.** These need k separate pseudo-label sets under k-fold, so validation never sees labels from a model trained on itself ([NVIDIA Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)).

**Distilling large teachers into the inference budget.** LMSYS 1st distilled 70B teachers into Gemma2-9B ([LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)).

**Ensembles judged by blend contribution rather than single-model score.** Hydrogen found single-model scores "did not correlate well" with blend value ([Hydrogen 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)).

**Post-processing tuned to the test's prior.**

- Hydrogen rescaled predictions so each column's mean matched the target mean ([Hydrogen 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)).
- LECR kept unseen contents within 5% of a topic's best probability and treated test matching as an assignment problem ([LECR 1st](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812)).
- Eedi 3rd probed the leaderboard to estimate the test's seen-to-unseen ratio at about 1:3. He then scaled unseen candidates' probabilities until 75% of top-1 predictions were unseen, moving **private 0.564 → 0.600** ([Eedi 3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498)). That prior correction was worth more than most model upgrades.

### For novel test targets, validate the novelty itself and protect recall

On retrieval tasks with novel targets, winners validate the novelty directly and protect recall.

- Eedi 4th scored unseen-misconception subsets separately ([Eedi 4th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559)).
- BELKA 1st held out 3% of building blocks so that validation contained non-shared chemistry ([BELKA 1st](https://www.kaggle.com/competitions/leash-BELKA/discussion/519020)).
- Eedi 1st prioritised `recall@32` over `map@25` and deliberately shipped the higher-recall retriever, because hard negatives raised MAP without raising recall ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)).
- LECR 1st trained a second LightGBM that excluded features "which will potentially have different distribution on real test set". It was a deliberately shift-aware ensemble member ([LECR 1st](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812)).

For text with unknown generators, DAIGT's winners credited the data mix rather than the architecture. They had multiple single models above 0.970 "due to the quality of the dataset", which plugged the blind spots of earlier models in each iteration ([DAIGT 1st](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295)). James Day added test-time short-context students that "look for dataset-specific wording quirks" ([DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)).

### Write the solution up as a staged ablation table

Winning write-ups share a recognisable shape: tl;dr, validation scheme, pipeline, then a staged ablation table that credits each change with a private-LB delta, then "what didn't work". Eedi 1st is the canonical example. Its steps run 0.495 → 0.531 (few-shot) → 0.575 (pseudo-labelled synthetic data) → 0.596 (more negatives) → 0.615 (CoT) → 0.625 → 0.638 (larger rerankers). It also lists failed ideas, among them iterative hard-negative mining and LoRA merging ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)).

### Final picks: one safe, one hypothesis

Final-pick strategy is where discipline pays off or fails.

- Eedi 2nd trusted CV on a 72B model that the public LB disliked. They spent their third pick on it and won a prize ([Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)).
- ELL 1st did not select its best-CV submission "fearing overfitting", and that submission had the best private score ([ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)).
- DAIGT 5th left its best submission out of the final three and dropped from 3rd to 5th ([DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)).
- LMSYS 1st lost one pick because the submission referenced a model the author had deleted ([LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)).
- BELKA is the winner's-curse warning. Three of the top five teams used the same tutorial notebook with minimal changes, and the winner called his result "pure luck and randomness" ([Leash Bio](https://leashbio.substack.com/p/belka-results-suggest-computers-can); [BELKA 1st](https://www.kaggle.com/competitions/leash-BELKA/discussion/519020)).

The defensible default follows. Pick A is the best seed-averaged CV ensemble. Pick B carries a stated, CV-backed hypothesis about the private test. Every artifact for both picks is frozen. Before reading anything into a leaderboard gap, use CV folds to simulate the public/private split ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)).

## What to change in our two live competitions

### CASMI 2026: decompose the funnel and isolate the popularity prior

The MS/MS literature makes two facts central for us.

**First, getting the formula right is easy and getting the 2D structure right is hard.** In CASMI 2022 the best formula accuracy was **94%**, but the best 2D-structure accuracy was only **26%** ([CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)). Rerankers therefore cannot help when the truth is missing from the candidate set.

**Second, popularity priors are powerful and dangerous.** In CASMI 2016, ChemSpider reference count alone outperformed all fragmenters combined, and the organisers warned that metadata "cannot assist in the identification of true unknowns" ([CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)). The 2026 audit makes the same point more sharply. Ranking candidates by **PubChem's default order gives 49.98% HR@1 without looking at the spectrum**. A SMILES-only classifier reached 99.66% HR@1 because true answers kept repository formatting while decoys were RDKit-canonical ([MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)).

The audit documents three more leakage routes:

- **Simulator augmentation** that touches test molecules lifts R@1 from 53.76% to 67.75%.
- **A formula annotator trained with test spectra** lifts HR@1 from 26.64% to 35.15%.
- **Pretraining decoders** that exclude only exact InChIKey matches memorise close analogues ([MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)).

Random splits are mostly lookups: **95.3%** of MassSpecGym-random test structures appear in training ([Coley lab audit](https://github.com/coleygroup/Audit_ML_MS_analysis)).

On the constructive side:

- CASMI 2016 deduplicated stereoisomers by the first InChIKey block, keeping the highest-scoring isomer ([CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)).
- An equal-weight fusion of six fragmenters beat every single entry ([CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)).
- Retrieval-level confidence predicts correctness better than per-bit fingerprint uncertainty ([arXiv 2603.10950](https://arxiv.org/abs/2603.10950)).

The table below turns this evidence into tests on our unseen-molecule bench, in priority order. The decision rules are our inferences from the cited evidence, not prescriptions from the sources. See also [[05 Ranking & Retrieval]], [[08 Cheminformatics & MS]] and [[02 Gradient Boosting & Learning-to-Rank]].

| Priority | Test or change | Decision rule | Evidence |
|---|---|---|---|
| 1 | **Per-query funnel**, reported per test class (1/2/3): adduct correct → formula correct → true InChIKey14 present in candidate union (by channel: library, analogue, PubChem) → rank bucket (1, 2–5, 6–25, >25) | Oracle MRR (= candidate recall) is the reranker's ceiling. If class-3 recall is low, spend effort on analogue generation, not the reranker | [Ng, error analysis by parts](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf); [CASMI 2022](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results); [Eedi 1st recall@32](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688) |
| 2 | **Spectrum-blind and constant-spectrum baselines**: (a) a reranker on candidate-only features; (b) the full reranker with spectra replaced by a constant | Any class-3 MRR well above random from (a), or little drop under (b), means a shortcut | [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624); [Gururangan](https://arxiv.org/abs/1803.02324) |
| 3 | **Prior isolation**: ablate PubChem-popularity features (CID order, reference counts) separately on seen and novel bench queries; consider a class-gated or novelty-interaction model | Keep prior features only where they win ≥ 75% of paired folds *on the novel slice* | [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1); [Rule #35](https://developers.google.com/machine-learning/guides/rules-of-ml); [LECR shift-aware model](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812) |
| 4 | **Channel-provenance audit**: canonicalise every candidate identically (tautomer, stereo-strip, InChIKey14) and train a classifier to predict candidate *source channel* from reranker features | High source-predictability means the reranker can learn "which pipeline produced this" | [SMILES-format shortcut](https://arxiv.org/abs/2606.19624); [RDKit #7499](https://github.com/rdkit/rdkit/issues/7499) |
| 5 | **Dedupe by InChIKey14 before ranking**; keep the best-scoring isomer and tautomer | Duplicate skeletons waste top-25 slots | [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1) |
| 6 | **Similarity-stratified MRR**: bin bench queries by max Tanimoto (ECFP4 2048) or MCES of the truth to the nearest library/training molecule | Class 3 should sit in the low bins; bench composition should match a 40–50% novel share | [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624); [Coley audit](https://github.com/coleygroup/Audit_ML_MS_analysis) |
| 7 | **Component-leak check**: confirm that no embedding model, simulator or formula tool behind our features was trained on bench molecules or near analogues | Exclude analogues above a Tanimoto threshold, not just exact keys | [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624) |
| 8 | **Reranker introspection**: MRR@25 permutation importance within query groups, with correlated similarity features permuted as clusters; TreeSHAP difference between the truth and the item ranked just above it for rank-2/3 queries | A rank 2→1 promotion is worth +0.5 per query, so explain those flips first | [sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html); [Rule #24/#26](https://developers.google.com/machine-learning/guides/rules-of-ml) |
| 9 | **Novel-prior correction**: tune an explicit boost for analogue/novel candidates on a bench whose novel share matches the test | Eedi 3rd's analogue of this was worth +0.036 private | [Eedi 3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498) |
| 10 | **Confidence features**: rank-1 score, rank-1 vs rank-2 margin, fraction of intensity explained by subformulae | These transfer to novel molecules better than priors | [Selective prediction](https://arxiv.org/abs/2603.10950); [MassSpecGym peak-explanation filter](https://arxiv.org/abs/2410.23326) |

Report every bench number with bootstrap confidence intervals; the audit uses 99.9% CIs from 20,000 resamples ([MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)). For final picks:

- **Pick A** is the best paired-CV reranker ensemble.
- **Pick B** carries the class-3 hypothesis: weaker prior features plus a novel-candidate boost tuned on the bench.

The test-class mix and the share of class-3 truths absent from PubChem are unknown. The candidate-recall-by-channel numbers from step 1 are therefore the most important missing measurement.

### Name That LLM: the errors live inside each lab

The attribution literature predicts precisely where our balanced-accuracy loss lives.

**Cross-family attribution is nearly solved; within-family attribution is not.** Sun et al. reach **97.1%** on 5-way classification across chat APIs, but only **59.8%** on four Qwen2.5 sizes, with the 14B-vs-32B pair hardest at 63.1% ([Sun et al. 2025](https://arxiv.org/abs/2502.12150)). M4GT-Bench found its two OpenAI siblings the hardest generators to separate ([arXiv 2402.11175](https://arxiv.org/abs/2402.11175)). A JavaScript attribution study found that most errors came from confusing close variants of the same release, such as GPT-4o versus GPT-4o-mini ([arXiv 2510.10493](https://arxiv.org/abs/2510.10493)).

**The transformation ladder shows where the signal lives** ([Sun et al. 2025](https://arxiv.org/abs/2502.12150)):

| Transformation applied | Accuracy (chat APIs) |
|---|---|
| None (original) | 97.1% |
| Words shuffled | 88.9% |
| Letters shuffled | 39.1% |
| Markdown skeleton only | 73.1% |
| Paraphrased | 91.4% |
| Summarised | 58.1% |

So the signal sits mostly in word choice and content, with a strong formatting component.

**Other properties of the signal** ([Sun et al. 2025](https://arxiv.org/abs/2502.12150)):

- It is front-loaded: the *first token* alone gives about 50% accuracy, and accuracy saturates past about 256 tokens.
- Decoding settings are nearly invisible.
- A TF-IDF bigram logistic regression reaches 85.5% and exposes the phrases doing the work.
- Leave-one-model-out shows lineage: held-out Grok outputs were labelled ChatGPT **82.8%** of the time.

**Two warnings matter for our features.** Repetition penalty can move detector accuracy by up to 38 points ([RAID](https://arxiv.org/abs/2405.07940)). Prompt content shifts perplexity regardless of who wrote the text ([Binoculars](https://arxiv.org/abs/2401.12070)), which is the published rationale for normalising stylometric features within each prompt, as our stylometric centering does ([[07 NLP Stylometry]]).

**On Sinkhorn balancing.** Balanced accuracy is the mean of per-class recall. Sinkhorn balancing is therefore a prior correction of the same kind as Hydrogen's mean matching and Eedi 3rd's prior scaling ([[06 Calibration & Label Shift]]). That reasoning is ours, not a cited result. It means α must be tuned on grouped OOF predictions, not on the public LB.

| Priority | Test or change | Decision rule | Evidence |
|---|---|---|---|
| 1 | **StratifiedGroupKFold by prompt** for every OOF number; also compare a random split against the prompt-grouped one | If the random split scores well above the grouped one, topic is leaking | [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685); [Eedi group-key sweep](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688) |
| 2 | **18×18 confusion matrix sorted by lab**; per-class recall vector; label each off-diagonal pair's share of balanced-accuracy loss as its ceiling | Work on the largest within-lab block first | [Ng ceiling](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf); [Sun et al.](https://arxiv.org/abs/2502.12150) |
| 3 | **Hand-read about 100 errors from the top 3 confused pairs** using open and axial coding | Name the discriminating cue before engineering a feature for it | [Husain Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) |
| 4 | **Partial-input baselines**: GBDTs on length only, markdown/format only, first-20-tokens only, and prompt-ID only | Any score far above 1/18 ≈ 5.6% marks an artifact; check whether it holds across prompt groups | [Gururangan](https://arxiv.org/abs/1803.02324); [Sun markdown-only 73–78%](https://arxiv.org/abs/2502.12150) |
| 5 | **Transformation ladder on the LoRA model's OOF predictions** (strip markdown, shuffle words, crop the opening, paraphrase) | Large drops from formatting removal mean fragility if the test formatting differs | [Sun et al.](https://arxiv.org/abs/2502.12150); [Lapuschkin insertion test](https://arxiv.org/abs/1902.10178) |
| 6 | **Pair-specialist TF-IDF LR** for each confused sibling pair; read its coefficients, then add them as features or a pair head (hierarchical lab → model) | Keep if it wins ≥ 12 of 15 fold×seed pairs with the LLM logits held fixed | [Sun et al.](https://arxiv.org/abs/2502.12150); [Bouthillier](https://arxiv.org/pdf/2103.03098v1) |
| 7 | **Joint per-prompt assignment**: on train, measure how often passages that share a prompt come from distinct models; if near 100%, add a "no duplicate model within a prompt" constraint alongside Sinkhorn's global marginals | Adopt only if grouped OOF balanced accuracy improves on paired folds | [LECR assignment framing](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812) |
| 8 | **Training-dynamics and cleanlab audit** on 18-way OOF probabilities | "Hard-to-learn" sibling passages may be indistinguishable rather than mislabelled; do not delete them blindly | [Cartography](https://aclanthology.org/2020.emnlp-main.746/); [cleanlab](https://github.com/cleanlab/cleanlab) |
| 9 | **Layer probe with a control task** (random labels per prompt) on frozen hidden states | Pick the pooling layer or LoRA target layers by selectivity, not raw probe accuracy | [Hewitt & Liang](https://arxiv.org/abs/1909.03368) |
| 10 | **Sanity checks**: loss at init ≈ 2.89; seed-averaged LoRA comparisons; random-crop augmentation if test passages are excerpts | Prevents silent failures and over-reliance on first-token cues | [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/); [Sun length curve](https://arxiv.org/abs/2502.12150) |

Rows 7 and 10 depend on two facts we have not verified: how passages are assigned to models within a prompt, and whether test passages are whole responses or excerpts. Both should be measured on train before either change ships. For final picks:

- **Pick A** is the seed-averaged LoRA plus stylometric ensemble with Sinkhorn α tuned on grouped OOF predictions.
- **Pick B** adds the joint per-prompt assignment, or a pair-specialist stage, if that change wins on paired grouped folds.

No source here measures how much random splits inflate authorship CV compared with prompt-grouped splits, so the row 1 comparison is a measurement we have to make ourselves.

## Conclusion

Both competitions are bounded more by structure than by model capacity. In CASMI the bound is candidate recall on novel molecules, and in Name That LLM it is the within-lab confusion blocks. Introspection should therefore be organised around those two partitions, not around global feature importances.

The less obvious lesson is that the strongest levers are also the most fragile. The PubChem prior that lifts classes 1 and 2, the Sinkhorn marginals that lift balanced accuracy, and the formatting cues that separate labs all succeed by encoding an assumption about how the test set is composed. Winners do not avoid such levers. Eedi 3rd's prior boost was one of the biggest single gains in that competition. What winners do is make each assumption explicit, give it a parameter, tune it on a bench whose composition matches the test, and give it its own final pick. The same tools also tell us when to stop: BELKA's luck-dominated podium and Bouthillier's 10% single-run false-positive rate say that once paired CV can no longer separate two candidates, more leaderboard probing adds noise rather than information.
