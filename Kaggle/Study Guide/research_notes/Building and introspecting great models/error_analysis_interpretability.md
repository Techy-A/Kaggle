# How Expert Data Scientists Debug and Introspect Trained Models: Error Analysis Workflows and Interpretability Methods

Research notes, compiled 2026-10-17. Primary sources were fetched and read directly (Karpathy blog, Google Rules of ML, Ng's *Machine Learning Yearning* draft PDF, Hamel Husain's blog, scikit-learn docs, SHAP docs, Molnar's *Interpretable ML* book, arXiv abstracts and PDFs). Where only secondary sources were reachable, this is noted.

Two target competitions are referenced throughout:
- **(C1)** molecule-retrieval ranking, scored by MRR@25, with gradient-boosted (GBDT / LambdaMART-style) rerankers;
- **(C2)** 18-class LLM-authorship text classification, scored by balanced accuracy, with LoRA-tuned LLMs plus stylometric features.

---

## Q1. Which structured error-analysis workflows do respected practitioners recommend (Ng, Karpathy, Google Rules of ML, Hamel Husain), and what are the concrete steps?

### Takeaway
Every respected source says the same thing: before building anything, look by hand at a sample of failures, about 100 of them. Tag each failure in a spreadsheet with categories you make up as you go, then count the categories. Each category's share of errors is a ceiling on what fixing it can gain. Put your effort where the count is large and a fix is plausible. For pipelines, swap in a "perfect" output for one component at a time to see which component causes each error. Generic dashboards and metrics are no substitute for reading examples.

### Cited Findings

**Andrew Ng, *Machine Learning Yearning* (draft, c. 2018)**
- The core procedure: "1. Gather a sample of 100 dev set examples that your system misclassified… 2. Look at these examples manually, and count what fraction of them are dog images. The process of looking at misclassified examples is called error analysis." — [Ng, MLY draft (GitHub mirror of official PDF)](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- The ceiling logic: if only 5% of errors are dogs, "5% is a 'ceiling'… Thus, if your overall system is currently 90% accurate (10% error), this improvement is likely to result in at best 90.5% accuracy." If 50% of errors are dogs, accuracy could go from 90% to 95%. — [Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- On cost: "Manually examining 100 examples does not take long. Even if you take one minute per image, you'd be done in under two hours." — [Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- The spreadsheet method: "I usually create a spreadsheet and fill it out while looking through ~100 misclassified dev set images." Each row is an error, each column is a category (Dog, Great cat, Blurry, Instagram, Mislabeled), plus a free-text Comments column. One example can tick several columns, so the percentages need not sum to 100%. New categories emerge while you look ("you will probably be inspired to propose new error categories"). "The most helpful error categories will be ones that you have an idea for improving." — [Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- Mislabeled dev data: track it as its own column. At 90% accuracy, with 0.6% of errors due to mislabels (6% of all errors), fixing labels is not crucial. At 98% accuracy, the same 0.6% is 30% of errors, and "it is now worthwhile to improve the quality of the labels in the dev set." — [Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- Eyeball vs Blackbox dev split, and how big each should be:
  - Sizing by number of mistakes in the Eyeball set: ~10 mistakes is "very small"; ~20 gives "a rough sense"; ~50 gives "a good sense"; ~100 gives "a very good sense of the major sources of errors"; some people analyze up to 500.
  - At a 5% error rate you need about 2,000 Eyeball examples to get ~100 errors.
  - A Blackbox set of 1,000–10,000 examples is usually enough to tune hyperparameters.
  - Overfitting check: "If you see the performance on the Eyeball dev set improving much more rapidly than the performance on the Blackbox dev set, you have overfit the Eyeball dev set."
  - The guidance applies to tasks humans do well; for tasks humans cannot do, eyeballing is less useful.
  — [Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- Error analysis by parts (pipeline attribution): "Replace the cat detector output with a hand-labeled bounding box… If the cat breed classifier still misclassifies it, attribute the error to the cat breed classifier. Otherwise, attribute the error to the cat detector." For a pipeline A→B→C, make A "perfect" and rerun B and C; if the result is now correct, attribute the error to A; then repeat for B. — [Ng, MLY ch. 54–55](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)

**Andrej Karpathy, "A Recipe for Training Neural Networks" (Apr 25, 2019)**
- Two premises: "Neural net training is a leaky abstraction" and "Neural net training fails silently… Most of the time it will train but silently work a bit worse." — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)
- Step 1, "Become one with the data": "I like to spend copious amount of time (measured in units of hours) scanning through thousands of examples." He found duplicates and corrupted images and labels this way. Write code to "search/filter/sort by whatever you can think of… The outliers especially almost always uncover some bugs in data quality or preprocessing." — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)
- Step 2, skeleton and dumb baselines. His checks, verbatim headings with short notes:
  - "fix random seed"
  - "add significant digits to your eval" (evaluate on the whole test set, not smoothed batch losses)
  - "verify loss @ init" (expect −log(1/n_classes) for softmax; **for C2 with 18 classes this is ln 18 ≈ 2.89**)
  - "init well" (set the final-layer bias to match class priors)
  - "human baseline"
  - "input-independent baseline" (zero out the inputs; does the real model beat this?)
  - "overfit one batch" (as few as two examples, down to zero loss)
  - "visualize just before the net" (decode the exact tensor fed into `model(x)`)
  - "visualize prediction dynamics" (watch a fixed test batch over training)
  - "use backprop to chart dependencies" (gradient of example i's output should be non-zero only on input i; this catches batch-mixing bugs and causal-mask leaks)
  — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)
- "complexify only one at a time. If you have multiple signals to plug into your classifier I would advise that you plug them in one by one and every time ensure that you get a performance boost you'd expect." — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)
- Other numbers: "Adam with a learning rate of 3e-4" as a safe baseline; ensembles are "a pretty much guaranteed way to gain 2% of accuracy on anything" but gains top out "after ~5 models"; prefer random search over grid search. — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)

**Google, "Rules of Machine Learning" (Martin Zinkevich)**
- Rule #26, "Look for patterns in the measured errors, and create new features": "If you give the model a feature that allows it to fix the error, the model will try to use it… if you try to create a feature based upon examples the system doesn't see as mistakes, the feature will be ignored." For ranking, an error is "a pair where a positive was ranked lower than a negative." On adding features: "Don't be too specific… just add a dozen features and let the model figure out what to do with them." — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rule #24, "Measure the delta between models": for ranking, run both models on a sample of queries and compute "the size of the symmetric difference of the results (weighted by ranking position)." Read the queries where the difference is high. Sanity check: "a model when compared with itself has a low (ideally zero) symmetric difference." — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rule #25: "if there is some change that improves log loss but degrades the performance of the system, look for another feature." In other words, judge rankers by the ranking metric, not by the surrogate loss. — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rule #27: "measure first, optimize second". Turn gripes into numbers using human-labeled samples. — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rule #23: "You are not a typical end user." Rule #14: "Starting with an interpretable model makes debugging easier." Rule #10: "Watch for silent failures." Rule #33: train on data up to Jan 5, test on data from Jan 6 onward (a temporal holdout). — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Rule #35, "Beware of the inherent skew in ranking problems": "Don't have document-only features… even if a given app is a popular download regardless of what the query was, you don't want to show it everywhere." Also: put higher regularization on features that cover more queries. — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)

**Hamel Husain, error analysis for LLM products (blog, 2025)**
- "Error analysis — the single most valuable activity in AI development and consistently the highest-ROI activity." "Generic metrics are worse than useless." — [Husain, A Field Guide to Rapidly Improving AI Products](https://hamel.dev/blog/posts/field-guide/)
- Case study (NurtureBoss):
  - Method: a spreadsheet with one row per conversation; open-ended notes on bad behavior; an LLM drafts a taxonomy; each row is mapped to labels; frequencies are counted.
  - Result: "just three issues accounted for over 60% of all problems."
  - Date handling failed 66% of the time; fixing it raised success "from 33% to 95%."
  — [Husain, Field Guide](https://hamel.dev/blog/posts/field-guide/)
- Data viewer: "Teams with thoughtfully designed data viewers iterate 10x faster than those without them." This is his claim, not a measured study. — [Husain, Field Guide](https://hamel.dev/blog/posts/field-guide/)
- Evals FAQ method:
  - **Open coding**: write free-form notes per trace. "Start by annotating at least 30 traces yourself." Focus on "the first failure observed in a trace, as upstream errors can cause downstream issues."
  - **Axial coding**: group the notes into a "failure taxonomy… Axial coding is the most important step. At the end, count the number of failures in each category."
  - Keep going until "theoretical saturation": "We recommend reviewing at least 100 traces."
  - "Spend 30 minutes manually reviewing 20-50 LLM outputs whenever you make significant changes."
  - "We've spent 60-80% of our development time on error analysis and evaluation."
  — [Husain & Shankar, LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)
- Judgments: favor binary pass/fail over 1–5 Likert scales, paired with written critiques. Critiques used as few-shot examples gave "15-20% higher agreement rates between human and LLM evaluations" (his anecdotal number). Plan to label 100–200 examples per failure mode for an LLM judge. — [Husain, Field Guide](https://hamel.dev/blog/posts/field-guide/); [Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)

### Inferences
- **Recipe a Kaggle competitor can run this week:**
  1. Take out-of-fold (OOF) predictions.
  2. Sample about 100 errors, stratified by class or query type.
  3. Fill in a Ng-style spreadsheet: free-text notes first, then categories (Husain's open coding, then axial coding).
  4. Count the categories; each share is that category's ceiling.
  5. Work on the largest category you have an idea for.
  6. Keep an "Eyeball" fold separate from a "Blackbox" fold so you don't overfit your own error analysis.
- **C2 (18-class balanced accuracy):**
  - Balanced accuracy is the mean of per-class recall, so the per-class recall vector is the error-analysis target.
  - Mine the 18×18 confusion matrix for the top off-diagonal pairs, which are probably sibling models from the same LLM family.
  - Sample errors from each of those pairs for the spreadsheet.
  - Each pair's share of total error, weighted by 1/18 per class, is the ceiling.
  - Check that loss at init is ≈ ln 18 ≈ 2.89 for the classification head.
- **C1 (MRR@25):**
  - MRR is dominated by where the first relevant item lands.
  - An error is a query whose first positive is at rank > 1. Bucket queries by that rank: 2–5, 6–25, >25 (zero contribution), and "not in candidate set."
  - Rule #26's definition of an error (positive ranked below a negative) maps directly onto LambdaMART pairs.
  - Rule #24's symmetric-difference check between two rerankers shows which queries actually changed.

### Gaps
- I could not fetch Chip Huyen's *Designing Machine Learning Systems* (O'Reilly, 2022) chapter on slice-based evaluation; her huyenchip.com TOC page returned almost no text. Her specific slice-evaluation guidance (e.g., Simpson's-paradox examples, heuristic and error-based slicing) is therefore not cited here.
- The Ng quotes come from a GitHub mirror of the free MLY draft PDF, not from deeplearning.ai directly. The content matches the book, but the mirror is not the official host.
- I did not find the classic Ng "ceiling analysis" table for photo-OCR pipelines (Coursera ML, 2011) in a primary text. The MLY "error analysis by parts" chapter is the documented equivalent.

---

## Q2. How do strong teams design ablations: one change at a time, ablation tables, seeds and variance, significance with few runs?

### Takeaway
Change one thing at a time, state a hypothesis before each run, and report variance across seeds and data splits. Variance from splits, initialization and hyperparameter choices is large enough that a single-run comparison often reaches the wrong conclusion. A practical criterion is that A beats B in at least 75% of paired runs (P(A>B) ≥ 0.75), not merely that A has a higher mean. Many claimed gains turn out to come from hyperparameter tuning rather than the stated idea.

### Cited Findings
- Karpathy's recipe is ablation-driven: "perform a series of ablation experiments with explicit hypotheses along the way", fix seeds, and "complexify only one at a time." — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)
- Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" (MLSys 2021, arXiv Mar 2021):
  - "Variance due to data sampling, parameter initialization and hyperparameter choice impact markedly the results."
  - Recommendations: "1) randomize as many sources of variations as possible in the performance estimation; 2) prefer multiple random splits to fixed test sets; 3) account for the resulting variance when concluding on the benefit of an algorithm over another."
  - Randomizing more sources of variation in an imperfect estimator gets closer to the ideal estimator "at a 51 times reduction in compute cost."
  — [arXiv 2103.03098](https://arxiv.org/abs/2103.03098)
- Bouthillier et al. propose the probability of outperforming, P(A>B) = (1/k)·Σ 1[R_A,i > R_B,i] over k paired splits or seeds, and declare A better if P(A>B) ≥ γ, with **γ = 0.75** "robust across all case studies." The simulation used k = 50. In their simulation the single-point comparison had a false-positive rate of about 10%. — [arXiv 2103.03098 (PDF)](https://arxiv.org/pdf/2103.03098v1)
- Dodge et al., "Show Your Work" (EMNLP 2019): "test-set performance scores alone are insufficient." Report expected validation performance as a function of the hyperparameter-search budget; "we find multiple recent model comparisons where authors would have reached a different conclusion if they had used more (or less) computation." — [arXiv 1909.03004](https://arxiv.org/abs/1909.03004)
- Lipton & Steinhardt, "Troubling Trends in ML Scholarship" (2018), list as a trend the "failure to identify the sources of empirical gains, e.g., emphasizing unnecessary modifications to neural architectures when gains actually stem from hyper-parameter tuning." — [arXiv 1807.03341](https://arxiv.org/abs/1807.03341)
- Wiegreffe & Pinter (2019) used "a variance calibration based on multiple random seed runs" as a baseline before interpreting attention differences. Seed variance is the null distribution for any "is this difference real?" claim, including interpretability claims. — [arXiv 1908.04626](https://arxiv.org/abs/1908.04626)

### Inferences
- **Ablation protocol for a Kaggle team:**
  1. Fix a K-fold split.
  2. For each variant, run the same folds (paired), with 3–5 seeds if it is a neural model.
  3. Build an ablation table with columns: change, hypothesis, mean CV, std, P(better) across fold×seed pairs, and public-LB delta (treat the LB with caution).
  4. Accept a change only if it wins on most paired folds. With 5 folds × 3 seeds = 15 pairs, γ = 0.75 means winning on at least 12 of 15.
- **C1:** MRR@25 is noisy per query, so a paired bootstrap over queries is the cheap significance test. This follows by analogy from the paired-comparison logic above; I did not find a source specific to MRR.
- **C2:** LoRA fine-tunes can vary noticeably across seeds. An ablation of a single stylometric feature group should hold the LLM logits fixed (precompute OOF logits) so that LLM seed noise does not swamp the feature's effect.

### Gaps
- I did not retrieve a primary source on minimum seed counts for LLM fine-tuning ablations (e.g., Dodge et al. 2020, "Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping"). It is relevant but not verified here.
- I did not fetch Dror et al.'s "Hitchhiker's Guide to Testing Statistical Significance in NLP" (ACL 2018) for the specific choice of test (e.g., paired bootstrap, permutation test).

---

## Q3. Interpretability for tabular and GBDT models: SHAP/TreeSHAP, permutation vs impurity importance, PDP/ICE, ALE, and their failure modes

### Takeaway
- **Feature importance:** use permutation importance on held-out data with repeats. Do not use MDI or split-count importance for decisions; it is biased toward high-cardinality features and computed on training statistics.
- **Correlated features** break both permutation importance and PDP/ICE, because permuting creates impossible rows. Cluster correlated features, or use ALE or conditional or grouped variants.
- **SHAP:** TreeSHAP is exact and fast for GBDTs and gives local attributions plus interaction values. Remember that SHAP explains the model, not the world. It describes correlations, and the choice of background data ("interventional" vs "tree_path_dependent") changes what it means.

### Cited Findings

**Impurity (MDI) importance and its biases**
- scikit-learn: "impurity-based feature importance for trees is strongly biased and favor high cardinality features (typically numerical features) over low cardinality features such as binary features or categorical variables with a small number of possible categories." Also: "impurity-based importances are computed on training set statistics and therefore do not reflect the ability of feature to be useful to make predictions that generalize to the test set." In their Titanic demo, a purely random high-cardinality `random_num` feature gets high MDI importance. — [sklearn User Guide 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html); [sklearn example: Permutation Importance vs MDI](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html)
- Strobl, Boulesteix, Zeileis & Hothorn (BMC Bioinformatics, 2007): random forest variable importance measures "are not reliable in situations where potential predictor variables vary in their scale of measurement or their number of categories." There are two mechanisms: "biased variable selection in the individual classification trees… and effects induced by bootstrap sampling with replacement." The fix is unbiased trees (cforest) with "subsampling without replacement." — [Strobl et al. 2007](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25)
- Lundberg, Erion & Lee (2018): "popular feature attribution methods are inconsistent, meaning they can lower a feature's assigned importance when the true impact of that feature actually increases." This motivates TreeSHAP. The TreeSHAP algorithm "has also been merged into XGBoost and LightGBM." — [arXiv 1802.03888](https://arxiv.org/abs/1802.03888)
- LightGBM's `feature_importance(importance_type=...)` defaults to `"split"` ("numbers of times the feature is used in a model"); the alternative `"gain"` gives "total gains of splits which use the feature." Both are training-set, impurity-style measures. — [LightGBM Booster API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html)

**Permutation importance: what it measures and when it misleads**
- Algorithm: importance i_j = s − (1/K)·Σ_k s_{k,j}, where s is the reference score and s_{k,j} is the score after shuffling column j on repeat k. `n_repeats` gives a distribution. The sklearn example uses `n_repeats=30` and keeps only features with mean − 2·std > 0. — [sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html)
- "Features that are deemed of low importance for a bad model (low cross-validation score) could be very important for a good model… Permutation importance does not reflect the intrinsic predictive value of a feature by itself but how important this feature is for a particular model." — [sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html)
- "Features that are important on the training set but not on the held-out set might cause the model to overfit." Compute it on both and compare. — [sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html)
- Results depend on the metric: "different metrics might lead to significantly different feature importances, in particular for models trained for imbalanced classification problems." — [sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html)
- Correlated features: "When two features are correlated and one of the features is permuted, the model still has access to the latter through its correlated feature. This results in a lower reported importance value for both features." On the breast-cancer dataset "a naive interpretation would suggest that all features are unimportant." The remedy: "cluster features that are correlated and only keep one feature from each cluster" (the sklearn example uses hierarchical clustering on Spearman rank correlation). — [sklearn 5.2](https://scikit-learn.org/stable/modules/permutation_importance.html)
- Hooker, Mentch & Zhou (2019/2021), "Please stop permuting features" / "Unrestricted Permutation forces Extrapolation": permute-and-predict methods (permutation variable importance, PDP, ICE) "can produce diagnostics that are highly misleading, particularly when there is strong dependence among features… forcing the original model to extrapolate." They "can vastly over-emphasize correlated features." They recommend "measuring the change in model performance after muting the effects of the features" by retraining, i.e., drop-column or refit importance. — [arXiv 1905.03151](https://arxiv.org/abs/1905.03151)
- Molnar adds three points:
  - Adding a correlated duplicate "can decrease the importance of the associated feature by splitting the importance between both features."
  - Permutation importance includes interaction effects.
  - Conditional or subgroup permutation importance (permute within strata of the correlated feature) keeps the marginal interpretation. "Strongly dependent features usually have very low conditional importance, even when they are used by the model."
  — [Molnar, IML: Permutation Feature Importance](https://christophm.github.io/interpretable-ml-book/feature-importance.html)

**PDP, ICE and ALE**
- sklearn: "Both PDPs and ICEs assume that the input features of interest are independent from the complement features, and this assumption is often violated in practice. Thus, in the case of correlated features, we will create absurd data points." PD curves "can obscure a heterogeneous relationship created by interactions… When interactions are present the ICE plot will provide many more insights." Use `kind='individual'` or `'both'`. — [sklearn 5.1 PDP/ICE](https://scikit-learn.org/stable/modules/partial_dependence.html)
- Molnar on PDP: a PDP can be flat while half the data has a positive effect and half a negative one, so heterogeneity is hidden. PDP "is causal for the model… but not necessarily for the real world" (citing Zhao & Hastie 2019). — [Molnar, IML: PDP](https://christophm.github.io/interpretable-ml-book/pdp.html)
- Apley & Zhu, ALE (arXiv 2016; JRSS-B 2020): PD plots "require extrapolation… far outside the multivariate envelope of the training data." Marginal (M) plots "produce substantially biased and misleading results when the predictors are dependent, analogous to the omitted variable bias." ALE plots "do not require extrapolation" and "are not biased by the omitted variable phenomenon," and are "far less computationally expensive than PD plots." — [arXiv 1612.08468](https://arxiv.org/abs/1612.08468)
- Molnar on how ALE works: it averages differences in prediction within quantile intervals (using the conditional distribution), accumulates them, and centers the curve. A caveat: quantile intervals can have very different lengths, which "can lead to some weird ALE plots." — [Molnar, IML: ALE](https://christophm.github.io/interpretable-ml-book/ale.html)

**SHAP and TreeSHAP**
- Lundberg & Lee (NeurIPS 2017): SHAP is "a unique solution in this class [additive feature attributions] with a set of desirable properties," and it unifies six existing methods, including LIME. — [arXiv 1705.07874](https://arxiv.org/abs/1705.07874)
- TreeSHAP lowers the complexity from O(TL·2^M) to O(TLD²), where T is trees, L is leaves, D is depth and M is the number of features. KernelSHAP is slow and "the shap Python package no longer uses it" by default. There are two TreeSHAP variants, "tree-path dependent" (approximately conditional) and "interventional", and the shap package now defaults to interventional. — [Molnar, IML: SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
- `shap.TreeExplainer(feature_perturbation=...)`: with "auto" (the default), "the 'interventional' approach will be used when a background is provided, otherwise the 'tree_path_dependent' approach." Interventional requires a background dataset, "anywhere from 100 to 1000 random background" samples. — [SHAP TreeExplainer docs](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)
- Conditional (path-dependent) SHAP "may give non-zero Shapley values to unused features" when they are correlated with used features (Sundararajan & Najmi 2020; Janzing et al. 2020). KernelSHAP and interventional SHAP sample from the marginal distribution, so they share the permutation extrapolation problem. — [Molnar, IML: SHAP](https://christophm.github.io/interpretable-ml-book/shap.html)
- SHAP interaction values extend SHAP to pairwise effects. Lundberg et al. also propose "supervised" clustering of samples on their SHAP vectors. — [arXiv 1802.03888](https://arxiv.org/abs/1802.03888)
- SHAP is not causal (Lundberg, Dillon, LaRiviere, Roth & Syrgkanis, Microsoft; SHAP docs):
  - "SHAP makes transparent the correlations picked up by predictive ML models. But making correlations transparent does not make them causal!"
  - In their simulated retention example, "Bugs reported" gets positive SHAP although its causal effect is zero, because of unobserved confounding by "product need."
  - "Discount" shows a negative relationship despite a small positive causal effect.
  - "Ad Spend" gets a positive effect with no causal effect at all.
  - The bar plot includes "feature redundancy clustering" to show which features are interchangeable.
  — [SHAP docs: Be careful when interpreting predictive models in search of causal insights](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html)

### Inferences
- **What each tool tells you:**
  - **MDI, split or gain importance:** "how much did training use this feature?" Good for spotting a leak (one feature absurdly dominant), bad for selecting features.
  - **Held-out permutation importance:** "how much does held-out score drop if this feature's information is destroyed, given the other features remain?"
  - **Drop-column or refit importance:** "is the information unique and needed?"
  - **SHAP:** "how did the model allocate this prediction?" SHAP dependence plots, colored by the strongest interaction, are a good way to find features worth engineering.
  - **ALE:** "what is the main effect when features are correlated?"
  - **ICE:** "is the effect heterogeneous?"
- **C1 (GBDT reranker, MRR@25):**
  - Compute permutation importance with a custom scorer equal to MRR@25 computed per query group. Permute within query groups so a molecule's features aren't swapped across queries.
  - Fingerprint, similarity and embedding-distance features are typically highly correlated. Cluster them (Spearman plus hierarchical clustering) and permute whole clusters. This avoids the "everything looks unimportant" artifact sklearn warns about.
  - Expect gain and split importance to inflate continuous similarity scores relative to binary flags (the MDI cardinality bias).
  - Rule #35's warning about document-only features maps to molecule-only features that boost popular molecules for every query. Check this by ablating them.
- **C2 (stylometric features in a GBDT or linear stacker):**
  - Stylometric features such as length, punctuation rates and type-token ratio are mutually correlated, so the same clustering caveat applies.
  - Look at SHAP per class (18 outputs), which shows which stylometric cue separates each LLM pair.
  - Watch for features that "work" only through a dataset artifact such as length or truncation (see Q5).

### Gaps
- I did not fetch the sklearn example "Permutation Importance with Multicollinear or Correlated Features" itself. The Spearman plus hierarchical clustering detail is from my recollection of that example, and the user guide confirms only that the strategy is to "cluster features that are correlated."
- I found no primary source specifically on SHAP for LambdaMART (ranking objectives). TreeSHAP explains the raw score per document, which is what drives within-query ordering, but I did not verify any documented pitfalls specific to ranking.

---

## Q4. Interpretability for neural and NLP models: the attention-as-explanation debate, probing and control tasks, saliency sanity checks and integrated gradients, logit lens and tuned lens, sparse autoencoders

### Takeaway
- **Attention weights** are at best weak evidence of importance.
- **Probes** must be compared against control tasks, otherwise high probe accuracy may only mean the probe learned the task.
- **Saliency methods** must pass model-randomization and label-randomization sanity checks; several popular ones fail. Integrated gradients is principled but still needs a baseline choice and the same checks.
- **Logit lens and tuned lens** are cheap, useful views of layer-by-layer predictions.
- **Sparse autoencoders** are a research-grade tool, rarely worth it in a competition.
- **For practitioners**, the most trustworthy introspection is behavioral and causal: ablate or perturb the input and measure the change in output.

### Cited Findings

**The attention-as-explanation debate**
- Jain & Wallace, "Attention is not Explanation" (NAACL 2019):
  - "Learned attention weights are frequently uncorrelated with gradient-based measures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions."
  - "Randomly permuting attention weights often induces only minimal changes in output."
  - The tests: Kendall τ between attention and gradient importance, and between attention and leave-one-out importance (total variation distance, TVD, of the output).
  — [arXiv 1902.10186](https://arxiv.org/abs/1902.10186); [PDF](https://arxiv.org/pdf/1902.10186v3)
- Wiegreffe & Pinter, "Attention is not not Explanation" (EMNLP 2019): the claim "depends on one's definition of explanation." They propose four tests:
  1. a uniform-weights baseline;
  2. variance calibration across random seeds;
  3. a diagnostic that uses frozen pretrained attention weights;
  4. end-to-end adversarial attention training.
  
  "Even when reliable adversarial distributions can be found, they don't perform well on the simple diagnostic." Both papers study RNN or BiLSTM attention, not modern transformers. — [arXiv 1908.04626](https://arxiv.org/abs/1908.04626)

**Probing classifiers**
- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks" (EMNLP 2019):
  - Control tasks "associate word types with random outputs… can only be learned by the probe itself."
  - A good probe is **selective**: high accuracy on the real task and low accuracy on the control task.
  - "Dropout… is ineffective for improving selectivity of MLPs, but other forms of regularization are effective."
  - ELMo layer 1 gives slightly better POS accuracy, but layer 2 is "substantially more selective."
  — [arXiv 1909.03368](https://arxiv.org/abs/1909.03368)
- Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances" (Computational Linguistics, 2021/22) reviews the methodological limits. The core caveat is that probe accuracy shows information is decodable, not that the model uses it. — [arXiv 2102.12452](https://arxiv.org/abs/2102.12452)

**Saliency, integrated gradients and sanity checks**
- Adebayo et al., "Sanity Checks for Saliency Maps" (NeurIPS 2018):
  - "Some existing saliency methods are independent both of the model and of the data generating process."
  - The **model parameter randomization test** compares saliency on the trained model with saliency on a randomly initialized network of the same architecture. Similar maps mean the method ignores the learned weights.
  - The **data randomization test** trains on permuted labels.
  - Their Figure 1 shows Guided BackProp and Guided GradCAM maps look like an edge detector's output. Methods that fail are "inadequate for… finding outliers in the data, explaining the relationship between inputs and outputs that the model learned, and debugging the model."
  — [arXiv 1810.03292](https://arxiv.org/abs/1810.03292); [PDF](https://arxiv.org/pdf/1810.03292v3)
- Sundararajan, Taly & Yan, Integrated Gradients (ICML 2017): it is built on two axioms, Sensitivity and Implementation Invariance, that "are not satisfied by most known attribution methods." It "requires no modification to the original network… just needs a few calls to the standard gradient operator." — [arXiv 1703.01365](https://arxiv.org/abs/1703.01365)
- Hooker, Erhan, Kindermans & Kim, ROAR (NeurIPS 2019): "many popular interpretability methods produce estimates of feature importance that are not better than a random designation." Only the ensembled methods VarGrad and SmoothGrad-Squared beat random. — [arXiv 1806.10758](https://arxiv.org/abs/1806.10758)
- Leavitt & Morcos (2020) argue interpretability research relies too much on intuition and that this "risk[s]—and in some cases ha[s] caused—illusory progress." They call for falsifiable hypotheses. — [arXiv 2010.12016](https://arxiv.org/abs/2010.12016)

**Logit lens and tuned lens**
- nostalgebraist, "interpreting GPT: the logit lens" (LessWrong, 2020): "GPT's probabilistic predictions are a linear function of the activations in its final layer. If one applies the same function to the activations of intermediate GPT layers, the resulting distributions make intuitive sense." By layer depth, predictions move from "nonsense" to "shallow guesses (right part of speech / register)" to "better guesses," and they converge "long before the end." — [LessWrong](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)
- Belrose et al., Tuned Lens (2023): it trains "an affine probe for each block in a frozen pretrained model." It is "more predictive, reliable and unbiased than the logit lens," which "is often brittle," and was tested on models up to 20B parameters. "The trajectory of latent predictions can be used to detect malicious inputs with high accuracy." — [arXiv 2303.08112](https://arxiv.org/abs/2303.08112)

**Sparse autoencoders and mechanistic interpretability**
- Cunningham et al. (2023): SAEs on LM activations learn "sparsely activating features that are more interpretable and monosemantic than directions identified by alternative approaches." The method targets superposition and polysemantic neurons. — [arXiv 2309.08600](https://arxiv.org/abs/2309.08600)
- Anthropic, "Towards Monosemanticity" (2023): a one-layer transformer with a 512-neuron MLP, with SAEs trained on "8 billion data points" of MLP activations. Expansion factors ran from 1× (512 features) to 256× (131,072 features), with detailed analysis of 4,096 features. Larger dictionaries show "feature splitting." — [transformer-circuits.pub](https://transformer-circuits.pub/2023/monosemantic-features/index.html)

### Inferences
- **Practitioner ranking for C2 (LoRA LLM authorship classifier), from cheapest and most reliable down:**
  1. **Behavioral perturbation tests.** Truncate texts, strip formatting or whitespace, normalize punctuation, paraphrase, and measure the change in per-class recall. This is the "muting" or ablation logic of Hooker et al. applied to text.
  2. **Occlusion or leave-one-out at the token or sentence level**, which is exactly Jain & Wallace's τ_loo quantity.
  3. **Integrated gradients on the embedding layer** (e.g., Captum), but run Adebayo's randomization check first: compare IG maps from the trained LoRA model with those from the base model plus a random head.
  4. **A linear probe on frozen hidden states per layer**, with a Hewitt–Liang control task (random labels per text-ID or per prompt), to find which layer carries authorship signal. Use it to choose the layer for pooling or for LoRA targeting.
  5. **Logit lens or tuned lens** are mostly useful for generative next-token analysis. For a classification head, a per-layer linear probe is the analog.
  6. **SAEs**: skip them unless you already have a pretrained SAE for your base model. They are not a one-week tool.
- **Attention maps are not evidence.** Do not use them to claim "the model looks at X." Ablate X instead.

### Gaps
- I did not verify how well the Jain & Wallace findings transfer to fine-tuned decoder LLMs. Both debate papers studied RNN or BiLSTM attention. Later transformer-specific work exists (e.g., attention rollout and flow, Abnar & Zuidema 2020) but was not fetched.
- I did not verify current tooling details or versions for Captum, TransformerLens or the tuned-lens package.

---

## Q5. Shortcut learning and "Clever Hans" detection: how practitioners find out a model is using a spurious cue

### Takeaway
Shortcuts are decision rules that do well on i.i.d. test data and fail under shift. Several detection methods work:
- inspect explanations at dataset scale, by clustering attributions;
- train a deliberately crippled model (e.g., hypothesis-only or input-subset) and see whether it scores well above chance;
- insert or remove the suspected cue and measure the flip;
- evaluate on controlled out-of-distribution or contrast sets where the heuristic fails.

### Cited Findings
- Geirhos et al., "Shortcut Learning in Deep Neural Networks" (Nature Machine Intelligence 2020; arXiv v5 2023):
  - Definition: "Shortcuts are decision rules that perform well on standard benchmarks but fail to transfer to more challenging testing conditions."
  - Examples: cows unrecognized on a beach ("grass" as a shortcut for "cow"); "a lush hilly landscape without any animal at all might be labelled as a 'herd of grazing sheep'"; a pneumonia classifier that "Looks at hospital token, not lung" and "fails on scans from new hospitals"; a toy model that learned object location instead of shape (stars vs moons).
  - Recommended steps: interpret results carefully ("exercise great care before attributing high-level abilities"), and make o.o.d. generalization tests standard practice, because the i.i.d. assumption "has been called 'the big lie in machine learning.'"
  — [arXiv 2004.07780](https://arxiv.org/abs/2004.07780); [PDF v5](https://arxiv.org/pdf/2004.07780v5)
- Lapuschkin et al., "Unmasking Clever Hans Predictors" (Nature Communications 2019):
  - A Fisher-vector classifier on PASCAL VOC 2007 had excellent "horse" accuracy but relied on a copyright or source tag "present in about one fifth of the horse figures."
  - "Removing the tag also removes the ability to classify the picture as a horse. Furthermore, inserting the tag on a car image changes the classification from car to horse."
  - The artifact had gone unnoticed in a dataset of 9,963 samples "that typically humans never look through exhaustively."
  - Their **SpRAy** (Spectral Relevance Analysis) method applies "spectral clustering on a dataset of LRP explanations" to surface "typical as well as atypical decision behaviors."
  - "Standard performance evaluation metrics can be oblivious to distinguishing these diverse problem solving behaviors."
  — [arXiv 1902.10178](https://arxiv.org/abs/1902.10178); [PDF](https://arxiv.org/pdf/1902.10178)
- Gururangan et al., "Annotation Artifacts in NLI Data" (NAACL 2018): a text classifier that sees only the hypothesis gets "about 67% of SNLI and 53% of MultiNLI" right. "Negation and vagueness are highly correlated with certain inference classes." This is the canonical partial-input baseline for detecting shortcuts. — [arXiv 1803.02324](https://arxiv.org/abs/1803.02324)
- McCoy, Pavlick & Linzen, HANS (ACL 2019): MNLI-trained models, BERT included, adopt "lexical overlap," "subsequence," and "constituent" heuristics, and "perform very poorly on HANS," a controlled set where those heuristics fail. — [arXiv 1902.01007](https://arxiv.org/abs/1902.01007)
- Shah et al., "Pitfalls of Simplicity Bias" (NeurIPS 2020): "neural networks can exclusively rely on the simplest feature and remain invariant to all predictive complex features." Simplicity bias "persists even when the simplest feature has less predictive power," and "ensembles and adversarial training… can fail in mitigating" it. — [arXiv 2006.07710](https://arxiv.org/abs/2006.07710)
- Karpathy's "input-independent baseline" is the degenerate case of the partial-input check: zero the inputs and confirm the model does worse. — [Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)
- Ribeiro et al., LIME (KDD 2016), show explanations used for "identifying why a classifier should not be trusted." Their known example is a husky-vs-wolf classifier using snow; that example is from the paper body and was not verified in the abstract fetched. — [arXiv 1602.04938](https://arxiv.org/abs/1602.04938)

### Inferences
- **Shortcut checklist for C2 (LLM authorship):**
  - Train "partial-input" baselines on cheap features alone: length or token count, whether the text was truncated at a max-token limit, leading or trailing whitespace, markdown and bullet usage, the first or last N characters only, and the prompt or topic if available.
  - If a length-only or format-only GBDT reaches well above the 1/18 ≈ 5.6% chance balanced accuracy, the dataset has shortcut opportunities. Decide whether they will hold on the private test set (same generation pipeline → maybe; different pipeline → dangerous).
  - Run insertion and removal tests on the trained LoRA model, Lapuschkin-style. Strip trailing boilerplate (e.g., "I hope this helps!") or add it to a different class's text, and measure the prediction flip rate.
  - Cluster per-example attribution vectors or hidden states (SpRAy-style). Small, tight clusters of correct predictions with odd attributions are where to look.
- **C1 (molecule retrieval):** the analog of a shortcut is a query-independent molecule prior: popularity, molecular weight, or how often an ID appears as a positive in training. Train a reranker on molecule-only features. If it gets substantial MRR, the model may be learning "popular positives" rather than query-molecule matching. This mirrors Rule #35's warning against document-only features.

### Gaps
- I found no primary source specific to shortcut learning in LLM-authorship detection. The suggestions above are extrapolations from the NLI artifact literature.
- I did not fetch Geirhos et al.'s "designing good o.o.d. tests" subsection in full. Its text was cut off in my extraction.

---

## Q6. Introspecting ranking models: debugging learning-to-rank and retrieval (recall@k vs rerank failures, per-query analysis, feature importance in LambdaMART)

### Takeaway
In a two-stage system, the first stage's recall@k is a hard ceiling on the reranker. Decompose every query's failure into one of three cases:
1. no positive among the candidates (a retrieval failure);
2. a positive among the candidates but ranked low (a reranker failure);
3. ranked correctly.

Then analyze per query, by query slice and by rank bucket. Compare rerankers using the position-weighted symmetric difference of their result lists (Rules of ML #24). Judge features with a ranking-metric permutation importance, not with split counts or surrogate loss.

### Cited Findings
- Recall is the ceiling: "reranker recall is upper bounded by retriever's Recall@k." One recent paper also warns that some LLM-rerank evaluations use an inject-then-rank "oracle protocol" (they insert all positives into the candidates), which "corresponds to a strictly larger upper envelope" than what is actually achievable. — [The Recall Ceiling of LLM Recommendation Reranking (arXiv 2609.27953, via search result; full text not verified)](https://arxiv.org/pdf/2609.27953)
- Practitioner debugging order: "Run the same labeled queries through first-stage-only and reranked pipelines… Record Recall@k before reranking, nDCG or MRR after reranking." Then "Increase candidate depth for recall failures… and bypass reranking for query classes whose ordering is already reliable," and "Slice the results by exact identifiers, paraphrases, long documents." — [ZeroEntropy: First-Pass Retrieval (vendor blog, via search)](https://www.zeroentropy.dev/concepts/first-pass-retrieval/); [ZeroEntropy: recall at k](https://www.zeroentropy.dev/concepts/recall-at-k/)
- Google Rules of ML:
  - #26 defines a ranking error as "a pair where a positive was ranked lower than a negative" and says to add features for patterns in those errors.
  - #24 measures the position-weighted symmetric difference between models' result lists and reads the queries where it is high.
  - #25 says to prefer the ranking metric over log loss.
  - #35 warns against document-only features.
  — [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- Ng's error analysis by parts generalizes directly to two stages. Give the reranker a "perfect" candidate set (inject the true positive). If it now ranks the positive first, attribute the error to retrieval; otherwise attribute it to the reranker. — [Ng, MLY ch. 54–55](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)
- LambdaMART in LightGBM: built-in importance is `"split"` (the default) or `"gain"`. Both are training-statistics measures with the MDI biases described in Q3. — [LightGBM Booster API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html); [sklearn 5.2 on MDI bias](https://scikit-learn.org/stable/modules/permutation_importance.html)

### Inferences
- **C1 recipe (MRR@25):**
  1. **Decompose per query.** For each validation query compute r* = rank of the first positive in (a) retrieval order and (b) reranked order. Report:
     - the fraction of queries with no positive in the candidate set, which gives MRR loss that is unrecoverable by any reranker;
     - the fraction with a positive in the candidates but r* > 25 after reranking, which scores 0;
     - the distribution of 1/r* for the rest.
     - The **oracle reranker MRR**, where every query with a positive in the candidates gets reciprocal rank 1, equals candidate recall. That is the Ng-style ceiling for reranker work.
  2. **Find the marginal value of candidate depth.** Plot oracle MRR against candidate depth k (e.g., 50/100/200/500) to see where adding candidates stops raising the ceiling.
  3. **Bucket and read examples.** Bucket reranker failures by query properties (molecule size, scaffold rarity, number of positives, whether the query molecule appears in training) and read ~100 failures (Ng/Husain).
  4. **Watch rank 2–3 promotions.** The gap between rank 1 and rank 2 matters most for MRR; a rank-2→1 promotion is worth +0.5 per query. Inspect queries where the positive is at ranks 2–3, using SHAP on the positive and on the doc ranked above it, to see which feature differences decide the order. The SHAP difference between the two documents explains the pairwise flip.
  5. **Permutation importance with an MRR@25 scorer**, permuting within query groups, with correlated feature clusters permuted together.
  6. **Model deltas.** Compute the Rule #24 position-weighted symmetric difference between successive rerankers to see which queries changed and whether the change helped.
- **Train/validation leakage:** in GBDT rerankers, features computed using the label, such as target-encoded "how often this molecule is a positive", must be computed out of fold. Otherwise gain importance will show them dominating. That pattern is the leak signature mentioned in Q3.

### Gaps
- The recall-ceiling sources here come from a search summary: a September 2026 arXiv paper and vendor blogs. I did not fetch their full text, so treat the exact quotes as secondary-source paraphrases.
- I found no primary, peer-reviewed source specifically on SHAP or permutation importance for LambdaMART. I also did not verify the classic LTR references (e.g., Burges 2010, "From RankNet to LambdaRank to LambdaMART") in this session.
- I found no source on debugging molecule retrieval specifically (e.g., scaffold-split analysis for retrieval). The recommendations above are inferences.
