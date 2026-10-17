# Post-processing, Metric-Optimization and Ensembling Hacks: Small Code, Large Leaderboard Gains

Scope note: this file goes deeper than `Building and introspecting great models/kaggle_grandmaster_practice.md`. That file already covers Eedi 3rd's unseen-candidate boost (0.564 → 0.600 private), LECR 1st's relative-probability post-processing, DAIGT 1st's rank averaging, and KGMoN hill climbing at a high level. Those items appear here only as one-line cross-references. Everything else here comes from primary write-ups and notebooks fetched through Kaggle's internal discussion API (`GetForumTopicById`) and the public kernels-pull API, on 2026-10-17.

Evidence tiers used throughout:
- **A**: a winner or top-5 write-up with a before/after number for this exact trick.
- **B**: a top write-up that uses the trick with no isolated number, or a public notebook with a measured CV/LB delta.
- **C**: forum or tutorial advice, or theory with no Kaggle-measured gain.

---

## Q1. Metric-specific optimization: thresholds, expected-metric decoding, prior/label-shift correction, Sinkhorn/Hungarian balancing, group constraints, calibration, clipping, MAP/MRR tricks

### Takeaway
The largest and most reliable "free" gains come from three sources:
1. Decoding predictions to match the metric's structure: QUEST's discretization (+0.027-0.030 Spearman), Feedback Prize's span-level second stage (+0.036), Shopee's min-2 rule (+0.019), and expected-F1 decoding in Instacart.
2. Hard constraints known from the test design: Recursion's per-plate Linear Sum Assignment, Optiver's zero-weighted-mean outputs, and HPA's label-ratio matching.
3. Prior/label-shift correction when train and test class priors differ: Quora's closed-form rescale, and Sinkhorn marginal balancing in Name That LLM.

Threshold tricks tuned on a small or shifted public LB are the main source of shake-up risk. ICR (thresholding hurt private) and Home Credit (a gamble on the metric hack) are the cautionary cases.

### Cited Findings

**1. Prior / label-shift correction (closed form, binary) — Tier A/B**
- What it is (Quora Question Pairs, 2017): train had 37% positives and test about 16.5%. The proposed conversion is `a = 0.165/0.37`, `b = (1-0.165)/(1-0.37)`, `f(x) = a*x / (a*x + b*(1-x))`. "Alternatively, you can train use a weighted version of the log-loss function." The author argues oversampling is "morally the same" but less exact. — [Quora: Statistically valid way to convert training predictions to test predictions](https://www.kaggle.com/competitions/quora-question-pairs/discussion/31179)
- Gain (Quora 1st place): they found that train/test bias differs across 3 "perimeters" defined by question-frequency counts (qid1_count = qid2_count = 1; min = 1 and max > 1; min > 1). The global public rescale was "not strong enough" deep in the stack, and per-perimeter rescaling was "too strong". Optimizing a rescale halfway between the two "helped to gain ~0.001 comparing to public rescale". The final layer was Ridge by perimeter plus Lasso, then a 55/45 blend "based on public LB score". — [Quora 1st place](https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355)
- Theory: Black Box Shift Estimation (BBSE) estimates the test label marginal p(y) under label shift (p(x|y) fixed) using any black-box predictor's confusion matrix. "BBSE works even when predictors are biased, inaccurate, or uncalibrated, so long as their confusion matrices are invertible." — [Lipton et al. 2018, arXiv:1802.03916](https://arxiv.org/abs/1802.03916)
- Theory: Saerens et al.'s EM (maximum-likelihood) prior adjustment needs calibrated p(y|x). With bias-corrected calibration first, EM is "hard-to-beat" against BBSL/RLLS. — [Alexandari et al. 2020, arXiv:1901.06852](https://arxiv.org/abs/1901.06852)
- Minimal recipe (multiclass): `p_adj ∝ p(y|x) * π_test(y)/π_train(y)`, then renormalize. If π_test is unknown, estimate it by EM: start with π_test = π_train, repeat {adjust; π_test = mean(p_adj)} until it converges. Calibrate first (temperature scaling on OOF).
- Risk: a wrong π_test, or covariate shift that is not pure label shift. Quora 1st found a single global rescale was not right for every subgroup.

**2. Sinkhorn / optimal-transport marginal balancing (known or assumed test class mass) — Tier B**
- What it is: Sinkhorn-Knopp matrix scaling is the solver for entropy-regularized optimal transport. — [Cuturi 2013, arXiv:1306.0895](https://arxiv.org/abs/1306.0895)
- Current Name That LLM public line: "Sinkhorn column rescaling of the test posterior toward uniform class mass (alpha = partial strength). CV: never negative (in-domain +0.002..+0.007, also with unequal class counts); under genre shift (external data) predictions collapsed onto one class and balancing took balanced accuracy from 0.12 to 0.19." — [LoRA x Adaptive Centering (Lucas Boesen)](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- Exact public implementation (50 iterations, partial target):
  ```python
  target = (1 - alpha) * P.sum(0) + alpha * np.full(K, len(P) / K)
  for _ in range(50):
      P *= (target / (P.sum(0) + 1e-12))[None, :]
      P /= P.sum(1, keepdims=True) + 1e-12
  pred = P.argmax(1)
  ```
  The base notebook uses `BALANCE_ALPHA = 0.5`. The fork "NTL v11 Sinkhorn 0.25" writes submissions for α ∈ {0, 0.1, 0.25, 0.5} and submits α = 0.25 ("weaker Sinkhorn alpha=0.25, no punctuation rules"). It states public LB 0.552 for its blend, with the 0.549 base. — [NTL v11 Sinkhorn 0.25 (liurouhan)](https://www.kaggle.com/code/liurouhan/name-that-llm-ntl-v11-sinkhorn-0-25); [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- Risk: α tunes how far you trust "test is balanced". If test classes are not balanced, α = 1 forces wrong marginals. Small test sets (few rows per class) make the column sums noisy. Sinkhorn is transductive, so it behaves differently on the hidden private split if that split's class mix differs.

**3. Hungarian / Linear Sum Assignment when each group holds distinct labels — Tier A (strongest group-constraint precedent)**
- Recursion Cellular Image Classification (2019) 1st place. Post-processing was: (1) average logits over sites and TTA; (2) ignore control classes; (3) mark "831 classes that can't be on the given plate" as impossible (each plate holds a known subset of 277 siRNAs, each appearing once); (4) "Linear Sum Assignment (LSA) is applied". Single model with this post-processing: 0.98997 private / 0.95802 public. Progressive pseudo-labeling with the same constraint gave 0.99700 private, and the 3-model ensemble 0.99763 private. During PL "I use greedy-like approach instead of LSA", but "Using LSA increases score because examples that were added later are more difficult". The write-up gives no isolated with/without-LSA number. — [Recursion 1st place](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)
- Minimal recipe: `from scipy.optimize import linear_sum_assignment; r, c = linear_sum_assignment(-logp_group)`. Here `logp_group` is (items in group × allowed labels) and impossible labels are set to -inf. With fewer items than labels the assignment is rectangular and still works.
- Risk: none if the constraint is guaranteed by the data design. If the constraint is only usually true, one violation can cascade into several mistakes inside the group.

**4. Threshold / rounding optimization for ordinal and rank-correlation metrics — Tier A**
- Google QUEST Q&A (2020) 1st: "discretization of predictions for some challenging targets led to better spearman corr. So we used no thresholds but discretized such columns based on their distribution in the training set. That simple heuristics boosted score by 0.027-0.030 for almost every single model (up to 0.448 bert-base, 0.454 bart)." — [QUEST 1st place](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840)
- Data Science Bowl 2019 1st (weighted QWK): they regressed, then used "[Optimizer Rounder](https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved) to optimize thresholds for weighted qwk". Because "qwk is still not so stable on our local cv", they validated on weighted RMSE and ignored QWK. Final private QWK 0.568, public 0.563. — [DSB 2019 1st place](https://www.kaggle.com/competitions/data-science-bowl-2019/discussion/127469)
- Minimal recipe (OptimizedRounder): regress the target, then `scipy.optimize.minimize(lambda t: -qwk(y_oof, np.digitize(p_oof, np.sort(t))), t0=[0.5,1.5,2.5,3.5], method='Nelder-Mead')`. Fit on OOF only; to reduce noise, average thresholds across folds or bootstrap samples.
- Risk: QWK thresholds are high-variance on small or shifted test sets. DSB 1st chose a smoother proxy (RMSE) for model selection for this reason.

**5. Expected-F1 decoding (per-instance variable threshold) — Tier B**
- Instacart Market Basket (2017): a fixed global threshold (0.19-0.21) is "the simplest, but not optimal". "Different order should have different threshold". Example: case 1 with p(A) = 0.9, p(B) = 0.3 gives E[F1({A})] = 0.81 vs E[F1({A,B})] = 0.71; case 2 with p(A) = 0.3, p(B) = 0.2 gives E[F1({A})] = 0.28 vs E[F1({A,B})] = 0.313. Better "None" handling gave "0.003 percentage point lift"; one route is P(None) = ∏(1 - p_i). The post links the "F1-Score Expectation Maximization" kernel (kaggle.com/mmueller/f1-score-expectation-maximization-in-o-n) and papers [arXiv:1206.4625](https://arxiv.org/abs/1206.4625) and [arXiv:1402.1892](https://arxiv.org/abs/1402.1892). — [Instacart: from Bronze to Silver](https://www.kaggle.com/competitions/basket-analysis/discussion/37016)
- Minimal recipe: sort items by p, and for each k compute E[F1 | top-k] under independent Bernoullis (exact DP, or Monte Carlo with 1000 draws). Pick the argmax k. Add "None" as a pseudo-item with p = ∏(1 - p_i).
- Risk: assumes calibrated, independent probabilities. Calibrate first.

**6. Label-ratio matching (multi-label F1) — Tier A (qualitative)**
- Human Protein Atlas (2019) 1st (bestfitting): "F1 is not a good metric as it's sensitive to the threshold and the threshold is depend on the distribution." Final post-processing kept the predicted label ratios equal to the public test set's (rare classes at the train ratio), with a second submission using the average of train and public ratios. "Although I tried to add or reduce the count of rare classes by 2-5 samples, the public LB can improve, but this was a dangerous way. I just only used it to evaluate the possible shakeup." — [HPA 1st place](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/discussion/78109)
- Recipe: for each class c, set the threshold to the (1 - r_c) quantile of test scores, where r_c is the target positive rate. This is a quantile threshold rather than a probability threshold.
- Risk: r_c taken from public LB probing can overfit. bestfitting hedged with two submissions.

**7. Metric-structure constraints in regression (sum-to-zero / weighted-mean removal) — Tier A**
- Optiver Trading at the Close (2023) 1st (hyd): "Subtract weighted-mean is better than average-mean since metric already told." `target -= (target*stock_weights).sum()/stock_weights.sum()`. Validation without → with PP: CatBoost 5.8287 → 5.8240; ensemble 5.8142 → 5.8117. The transformer also used `out = out - out.mean(1, keepdim=True)` inside the model. — [Optiver 2023 1st place](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)
- Risk: very low, because the target is defined relative to an index.

**8. Span/sequence decoding as a learned second stage (NER-style metrics) — Tier A**
- Feedback Prize 2021 1st: token-model ensemble 0.712 CV / 0.706 LB with rule post-processing. A LightGBM "sentence prediction" stage over about 3M over-recalled candidate spans (recall 0.895-0.974 per class), with about 170 features, reached 0.748 CV / 0.742 LB: "lgb sentence prediction: cv increase 0.036, lb increase 0.036". One sub-trick ("select sample with high boundary threshold and choice 65% length with the highest probability") added 0.008. On a 5-fold longformer, swapping rule PP for the LGB stage moved LB 0.697 → 0.727. — [Feedback Prize 2021 1st](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)
- Recipe: lower the thresholds to over-recall candidates, then train a GBDT on candidate-level features (mean/min/max token probabilities, length, position, neighbour-class probabilities) to predict "this candidate matches a gold span".

**9. Minimum-cardinality rule ("at least k answers") — Tier A**
- Shopee Product Matching 1st: the competition guarantees at least 2 matches per query (self plus one). The "min2" rule rejects the second-closest match only if its distance exceeds a high min2 threshold. Public LB 0.724 → 0.743. Union of image, text and combined-embedding matches plus threshold tuning: 0.757 → 0.776. "At last we only had to tune those 2 numbers against LB" (stage-2/3 thresholds). — [Shopee 1st place](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)

**10. Clipping / support constraints — Tier C**
- Playground S4E4 (abalone rings, RMSLE): train has unique classes 1-29 and no 28, so "clipping the predictions between these thresholds and post-processing for class 28 will do a world of good" (`np.clip(ypred, 1, 29)`; map 27.5-29 to 29). No measured number. — [PS S4E4 Common pitfalls and tips](https://www.kaggle.com/competitions/playground-series-s4e4/discussion/488093)
- ICR (balanced log loss) is the counterexample. Chris Deotte's per-fold threshold study under sliding-time CV found "during some time periods, thresholding works and during others it does not". Even a "very conservative PP threshold (`p[p<0.1]=0`) hurt my private LB score." His time-CV model scored CV 0.27, public 0.17, private 0.39. — [ICR silver write-up (Chris Deotte)](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/431067)

**11. MAP@K / ranking-specific tricks — Tier A/B**
- Eedi (MAP@25), private 7th / public 2nd: a "Voting ensemble of each pipeline". Each pipeline's 25 predictions get reciprocal-rank weights `1/i` (i = 1..25). They are joined across pipelines, summed per candidate, and sorted by `sum_rank`, with ties broken by individual ranks. This is a reciprocal-rank-fusion variant. — [Eedi 7th (public 2nd)](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551388)
- Eedi 1st's 25-slot construction (ranks 1-5 from the 72B listwise model, 6-8 from 32B, 9-25 from 14B) and Eedi 3rd's unseen-prior boost are in the existing notes (cross-reference only).
- LECR 1st's relative-to-best-probability rule for unseen candidates (keep if within 5% of the topic's best) is also in the existing notes.

### Inferences
- Under a fixed classifier, balanced accuracy (mean per-class recall) rewards predicting argmax_y p(y|x)/π̂(y), where π̂ is the class mass the model expects in test. That is exactly prior correction to a uniform prior. So for balanced accuracy, Sinkhorn balancing toward uniform and Saerens-style prior division are two routes to the same goal. Sinkhorn also uses the test set's own column sums (transductive), which is why it helps more under shift (0.12 → 0.19 in the external-data test).
- A pattern holds across tiers: constraint-based post-processing (Recursion LSA, Optiver zero-mean, Shopee min2, QUEST discretization to the train distribution) has low risk and large gains. Threshold values tuned against the public LB (HPA rare-class counts, ICR thresholds, Home Credit REDUCE) carry the shake-up risk.

### Gaps
- No isolated with/without-LSA number in the Recursion write-up. I found no other Kaggle winner write-up that uses Sinkhorn by name (the Kaggle search for "sinkhorn" returned only Santa and image-matching topics).
- I could not retrieve Faron's/mmueller's Instacart expected-F1 kernel itself, or an Instacart winner's isolated gain from expected-F1 decoding.
- I found no Kaggle write-up with a measured log-loss gain from probability clipping in the 2020s. The classic advice (clip to [1e-15, 1-1e-15], or to [0.025, 0.975] for noisy labels) remains folklore without a primary number here.

---

## Q2. Ensembling hacks: hill climbing, Caruana selection, rank/geometric/power averaging, simple stacking, folds/seeds, blend of blends

### Takeaway
Greedy forward selection with replacement over saved OOF predictions (Caruana 2004 and its Kaggle form, "hill climbing") is the best-evidenced ensembling method. It beat every other combiner on 10/10 metrics in the original paper, and Chris Deotte won Playground S5E5 with it. The combiner should match the metric: rank or power average for AUC, the log1p space for RMSLE, log-probabilities for log loss and accuracy. The main risk is overfitting the selection set, which Caruana's three fixes (replacement, sorted initialization, bagging) address.

### Cited Findings

**Caruana et al. 2004, "Ensemble Selection from Libraries of Models" — Tier A (academic)**
- About 2000 models per problem (KNN, ANN, SVM, DT, bagged/boosted trees with varied parameters), 7 problems, 10 metrics, and a 1000-point hillclimb/validation set. Forward stepwise selection repeatedly adds "the model in the library that maximizes the ensemble's performance to the error metric on a hillclimb (validation) set." — [Caruana et al. 2004 (PDF)](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- Three anti-overfitting fixes:
  1. **Selection with replacement**. Without it, performance "peaks, and then quickly declines"; with replacement the curve flattens so much that the stopping point matters little.
  2. **Sorted initialization**. Start from the N best models, "typically adds 5-25 of the best models", chosen on the hillclimb set.
  3. **Bagged selection**. Draw p = 0.5 of the library 20 times, run selection in each bag, and average the 20 ensembles, which "also is a simple weighted average". — [Caruana 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- Results, as mean normalized score across 10 metrics:
  - Ensemble selection 0.969, "the clear winner", outperforming the other ensemble methods and the best individual models "on all 10 of the 10 metrics (significant at p = 0.001)".
  - Bayesian model averaging 0.948.
  - Best single algorithms: SVM 0.905, ANN 0.902, bagged trees 0.888.
  - Stacking with logistic regression "performs poorly because regression overfits dramatically when there are 2000 highly correlated input models and only 1k points". An unweighted average of everything is worse than picking the best single model. — [Caruana 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)

**Kaggle hill climbing (Deotte and others) — Tier A**
- Chris Deotte, Playground S5E5 1st (RMSLE): hundreds of GBDT, NN and cuML models fed into a GPU hill-climbing notebook, which "automatically selects models". It picked 7 models with weights 1/12, 1/12, 1/12, 1/4, 1/6, 1/6, 1/6. Final CV 0.05880, public 0.05677, private 0.05841. "Diversity is more important than single model CV score": three XGBoost models with target-encoding features have poor CV (0.06XX) yet moved the ensemble from CV 0.05890 to 0.05880 and private 0.05847 to 0.05841. Residual models ("NN over LinearRegression", "XGB over NN") were added as diverse members. — [PS S5E5 1st: GPU Hill Climbing](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611); notebook [cdeotte/gpu-hill-climbing-cv-0-05930](https://www.kaggle.com/code/cdeotte/gpu-hill-climbing-cv-0-05930)
- Deotte's "icing on the cake", used "in all my Kaggle competitions": find weights on 5-fold OOF, then retrain each model on 100% of train with a fixed iteration count 25% (= 1/(K-1)) above the average early-stopping iterations. Apply the OOF-derived weights to the full-data test predictions, and also average K seeds. "This gives a nice boost in every Kaggle competition!" (no number). — [PS S5E5 1st](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611)
- For RMSLE: transform OOF and test predictions to `log1p`, hill-climb or ridge with RMSE, then `expm1`. Mean is optimal for MSE and median for MAE. — [How To Ensemble with RMSLE (Deotte)](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/576111)
- Oscar Aguilar's Playground S3E23 (AUC) version sorts models by OOF AUC, starts from the best, and at each step tries every remaining model with weight `w ∈ np.arange(-0.5, 0.51, 0.01)` (negative weights allowed), as `(1-w)*ens + w*model`. It keeps the best step and drops that model (no replacement). — [Boost Your Ensemble with Hill Climbing](https://www.kaggle.com/competitions/playground-series-s3e23/discussion/444784)
- Overfitting warning (follow-up thread): "Hill Climbing can lead to overfit: adding KA2 and RF1 below to the ensemble don't generalize well on validation data. Maybe, we should add an early stopping criteria." The proposed fix is to run hill climbing inside CV (fit weights on K-1 folds of OOF, score on the held-out fold). — [Ensemble / Hill Climbing again / In Parallel / Cross Validation](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450827)
- LANL Earthquake 1st: "Our final submit is a hillclimber blend of three types of models: (i) LGB, (ii) SVR, (iii) NN." Their public kernel blending NN and LGB by geometric mean scored 2.25993 private. "Just blending LGB and NN would have produced the best private LB score (2.25909). Adding SVR did improve CV though." This shows CV-chosen extra members can cost a little on private. — [LANL 1st place](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/discussion/94390)

**Rank / power / geometric averaging — Tier A (usage) / C (claims)**
- ISIC 2024 1st (pAUC above 80% TPR): 150 GBDTs (CatBoost/LGBM/XGBoost × 5 folds × 10 seeds), "predictions were ranked (.rank(pct=True)) and averaged with equal weights". 150 vs 45 models "didn't provide any significant improvement". — [ISIC 2024 1st](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533196)
- ISIC 2024 4th: a geometric mean across LGB/CB/XGB slightly beat each member, e.g. 10-model-feature pAUC 0.18650/0.18669/0.18647 → Gmean 0.18703. For each model, 40 seed combinations were run and the top 5 checkpoints by CV blended. "I have a submission that reached private LB = 0.1732 (1st place) by changing the selection of image models, but it wasn't my best CV score." — [ISIC 2024 4th](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/532760)
- Jigsaw Toxic Severity (2022) 1st (Guanshuo Xu): "Public LB looks misleading so I focused on the validation performance only … I only considered linear models build on validation data". Final submission "is a weighted rank average of 15 models"; ensemble of all 15: public 0.7879 / private 0.8139. — [Jigsaw Rate Severity 1st](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/306274)
- The menu, from a TPS Nov-2021 guide: simple/weighted/rank averaging; stretch (min-max) averaging; power averaging `mean(pred^p)` with p = 2-16, "to be used only when all the models are highly correlated, otherwise your score may become worse"; logit-space averaging; OOF stacking with LR on ranks. It notes SIIM-ISIC 2020's 1st and 2nd places "seem to just have used rank averages". — [How to trick AUC (TPS Nov 2021)](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/discussion/284507)

**Stacking with simple / regularized linear models — Tier A**
- Mercari 1st: 12 sparse-MLP predictions blended with weights "tuned … using a Lasso model" on 1% of data held out (5% locally). "Lasso says: meh, I don't want your new models, and sets their weights to zero." Diversity came cheaply: "it's better to achieve diversity in the ensemble by training the same model on different datasets", plus binarized TF-IDF inputs ("like getting an extra dataset … for free"). Final 0.37758 private (MXNet) vs 0.38006 (TF). — [Mercari 1st](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50256)
- Jigsaw Toxic (2018) 1st: "a weighted mean of arithmetic averaging and stacking, which worked marginally better (~.0001) than either approach alone". The stacker was LightGBM with "small trees with low depth and strong l1 regularization", DART + GBDT bagged over 6 seeds. A model was kept only if it improved CV log loss, CV AUC and public LB together. Stacking moved LB 0.9885 → 0.9890. — [Jigsaw Toxic 1st](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557)
- QUEST 1st: "stacking. Instead, linear blends worked unreasonably well" (listed under what didn't work). — [QUEST 1st](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840)
- Quora 1st: a 4-layer stack (about 300 L1 models, about 150 L2, two linear L3 models, then a 55/45 L4 blend by public LB) is the extreme "blend of blends". — [Quora 1st](https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355)
- Name That LLM public line: "Blend weights are fit with L-BFGS on 5-fold prompt-grouped out-of-fold predictions" in log-probability space. "Its predicted gains have matched our public-LB gains to within about 0.002-0.004 per version." — [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)

**Folds × seeds; checkpoint averaging — Tier B**
- LMSYS 1st averaged the 5 fold LoRAs into one model (existing notes). Optiver 2021 1st: "trained 10 models with different seeds and cherry-picked the top 5 models with the best validation scores". — [Optiver RV 1st](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- Santander 1st: "10 fold Stratified cross validation with multiple seeds for final blend"; final blend 2.1 NN : 1 LGBM. — [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)

### Inferences
- Minimal hill-climbing recipe (Caruana-safe):
  ```python
  # O: (n_oof, M) OOF preds; T: (n_test, M); score(y, p) higher-better
  order = np.argsort([-score(y, O[:,m]) for m in range(M)])
  ens = list(order[:N_init])              # sorted init, N_init 3-10
  while True:
      cur = score(y, O[:, ens].mean(1))
      gains = [score(y, O[:, ens+[m]].mean(1)) - cur for m in range(M)]  # WITH replacement
      m = int(np.argmax(gains))
      if gains[m] <= 1e-6: break
      ens.append(m)
  w = np.bincount(ens, minlength=M) / len(ens)
  ```
  Bag this 20× over random 50% subsets of models and average the w's. Run it inside an outer CV to estimate the true gain. Choose the averaging space to match the metric: log1p for RMSLE, logit/log-prob for log loss and argmax metrics, rank for AUC and MRR.
- Positive-weight, with-replacement selection (Caruana) is safer than Aguilar's negative-weight variant when the selection set is small.

### Gaps
- I did not retrieve a Kaggle write-up that reports Caruana's bagged selection beating plain hill climbing on a private LB.
- The power-averaging claims (p = 2-16 for AUC) come from a guide and a Medium post, not from a measured winner ablation.

---

## Q3. Leaderboard and test-structure exploitation (legitimate vs not)

### Takeaway
Most of the famous "magic" wins came from exploiting how the test set was built:
- Santander's fake test rows plus unique-value counts.
- Optiver 2021's recovered time-id order.
- Ventilator's PID-controller inversion.
- Instant Gratification's GMM cluster structure.
- Quora's question-frequency graph.
- Recursion's plate leak plus per-plate assignment.

These are legal when they use only the provided data. Metric hacks (Home Credit 2024) and public-LB probing (ICR, Eedi 3rd) are high variance. Using test unlabeled data transductively (pseudo-labels, test-set standardization, MLM or tokenizers on test, clustering test into prompt groups) is routine and usually rewarded.

### Cited Findings

**Test-structure "magic" (Tier A)**
- Santander Customer Transaction (2019) 1st. The magic: "count of values, especially the fact that some are unique". The YaG320 kernel "List of Fake Samples and Public/Private LB split" made it possible to compute uniqueness over train + real test only. Progression with "has one" uniqueness features: LB .910 → .914 (adding "not unique" features) → .921 (with fake rows removed). The NN reached .92687 public / 0.92546 private. Pseudo-labels: 5000 highest / 3000 lowest test points (NN), which moved private 0.92497 → 0.92546. — [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
- Optiver Realized Volatility (2021) 1st (nyanp): real prices were recovered via "tick size", and "by compressing the time-id x stock-id price matrix to one dimension using t-SNE, I was able to recover the order of the time-id". That enabled time-series CV and nearest-neighbour aggregation features ("boost from 0.21 to 0.19"). Time-drifting features were converted to ranks within time-id "to reduce the risk of shakedown". — [Optiver RV 1st](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970)
- Ventilator Pressure (2021) 4th, "Hacking the PID control": `u_in` is set by a PID controller linear in the target pressure. "I reversed this mapping and get the exact values for about 30% of the data." The author notes the technique "overlaps a lot with the existing 1st-place solution". — [Ventilator 4th](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285278)
- Instant Gratification (2019) 1st: QDA → pseudo-label test plus flip mismatched train labels → GMM with 6 clusters per batch mapped to classes → repeat. 0.9754 CV / 0.9744 public / 0.97598 private. — [Instant Gratification 1st](https://www.kaggle.com/competitions/instant-gratification/discussion/96549)
- Quora 1st: graph "density features from the graph built from the edges between pairs of questions inside train and test datasets concatenated": neighbour counts, k-core-like higher-order neighbours, connected-component stats, and graphs weighted by model predictions. — [Quora 1st](https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355)
- Recursion 1st: "HUVEC-18 is moved to the training set (known leak)", plus the plate-assignment constraint (Q1). — [Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)
- LANL 1st: a transductive train subsample. They chose 10 of 17 train earthquakes whose feature distributions minimized the mean KS statistic against the full test set (sampled up to 10k times), kept only features with KS p > 0.05 train vs test, and checked that OOF vs test prediction distributions were not rejected by KS. Their LGB kernel alone scored 2.279 private MAE, "1st place". — [LANL 1st](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/discussion/94390)

**Metric hacks (Tier A, ethically contested)**
- Home Credit Credit Risk Model Stability (2024). The stability metric's `88 * min(0, a)` slope term let you "make your model worse for the first weeks and improve your score a lot". Lowering scores by 0.02 in the first half of WEEK_NUM "improves my score by almost 0.03". — [Problem with competition metric (at7459)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/476449)
- Later, the test dates proved recoverable ("the dates in the test could be accurately restored"), which revived the hack. — [Metric hack again, sorry (Evgeny Patekha)](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/497167)
- 1st place (yuuniee), "My Betting Strategy": the best no-hack private score in his pool was "close to about 0.53". The pure-model gap among top teams was "0.00X" while the hack's tuning moved scores "0.0X", so "it became gambling". He picked the median of the expected optimum (DEVIDE = 1/2, REDUCE = 0.03; public-optimal REDUCE was 0.03, private probably 0.02-0.04) and submitted one no-hack and one hack submission. — [Home Credit 1st](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337)

**LB probing (Tier A, risky)**
- ICR (Chris Deotte): of three final candidates, he picked two that used LB-probed pseudo-labels and a PP threshold, "Go Big or Go Home!". "Using probed positive targets from public LB actually hurt my private LB score", and the conservative threshold also hurt. The private test turned out to hold more of the hard "Alpha = D" class (probing notebook). — [ICR write-up](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/431067)
- ISIC 2024 1st tested the noise std (0.02/0.05/0.08/0.12) added to image-model features "via leaderboard probing (one of few things tested without cv)". In the final weeks he lowered his significance bar to p = 0.2 and "mainly relying on the Public leaderboard". He still won. — [ISIC 2024 1st](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533196)
- Eedi 3rd's single-class probes (seen-only 0.154 vs unseen-only 0.444) → about 1:3 prior → unseen boost: see existing notes.

**Transductive use of unlabeled test (Tier A)**
- Jigsaw Toxic 1st, "rough-bore" PL: label test with the best ensemble, add, train to convergence. LB 0.9880 → 0.9885; "PL helps with" the train-test distribution difference. — [Jigsaw Toxic 1st](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557)
- Recursion progressive PL (0.98997 → 0.99700 private, single model). — [Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)
- HPA 1st: metric-learning nearest-neighbour label replacement for about 1000 test images ("almost the same score as replacing 1300") gave "0.03+, which was a huge improvement". It also computed normalization mean/std on train + test. — [HPA 1st](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/discussion/78109)
- QUEST 1st: a leakage trap in pseudo-labeling. Using all folds' models to pseudo-label external data inflates CV, because "4 out of 5 models used to generate pseudolabels have 'seen' the current validation set". Per-fold pseudo-labels from models trained only on that fold's train part gave a smaller but honest gain (0.414 → 0.422 vs the "leaked" 0.445). — [QUEST 1st](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840)
- Name That LLM public line, transductive steps "(no labels used)":
  - Test-set standardization of LM-head features (`(Ete - Ete.mean(0)) / Ete.std(0)` when there are at least 2 test rows).
  - Topic clustering of test passages (TF-IDF plus GLM-4-9B layer-24 cosine, average linkage, cut at 0.60-0.62). At a 0.65 cut, "same-prompt pair recall 0.89 vs 0.78 for TF-IDF alone, precision 0.97".
  - Prompt-centered "twin" heads applied inside clusters of 2-18 passages. — [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- Measured adaptive-centering gains (pooled 150-prompt simulation, plain vs centered). By test regime of 1-2 / 1-5 / 3-8 / all-18 passages per prompt:
  - v8 "replace" mode: +0.000 / +0.019 / +0.034 / +0.056.
  - v9 "add" mode (twins as extra evidence from cluster size 2): +0.024 / +0.042 / +0.043 / +0.052.
  - "Never centers singletons (v1's failure)."
  - Public LB v8 0.499 → v13 0.532 → 0.541 → 0.547 → 0.549 → 0.550. — [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)

### Inferences
- Legitimate vs not (Kaggle norms as reflected in these threads):
  - Generally accepted: exploiting structure inside the provided data (row/time order, duplicates, group sizes, fake rows), transductive fitting on unlabeled test (PL, standardization, clustering, tokenizers/MLM), and constraint-based assignment.
  - Contested: metric hacks that make the model worse on purpose (Home Credit; both whistle-blowers called it unrelated to data science).
  - Outright risky: hand-labelling, and probing the public LB for labels (ICR shows it can backfire).
- A generalizable defence: build a CV that mimics how the private split was made (Deotte's sliding-time CV; LANL's KS-matched subsample; NTL's pooled simulation with 1-8 passages per prompt). Then judge any "magic" or PP by its distribution across folds, not by its best fold.

### Gaps
- I could not fetch the Ventilator 1st-place write-up (only the 4th's reference to it), or a numeric PID-inversion gain.
- Home Credit 1st's final private score and the effect of the hack in isolation are not stated in the write-up.

---

## Q4. Test-time augmentation and its text/tabular analogues; output smoothing; rule overrides

### Takeaway
For pairwise text tasks, TTA by swapping order is worth about 0.007 log loss (LMSYS 3rd). WSDM 1st recovered most of the benefit by applying it to only the 33% most uncertain rows. Translation TTA gave Jigsaw +0.0003 alone and +0.0012 when combined with train-time augmentation. Image TTA (flips/rotations) is standard, but ISIC 2024 1st found it gave nothing. Rule overrides are cheap but fragile: NTL's punctuation veto helped OOF by +0.0007 yet cost forks 0.003-0.005 on the LB. The family "gates" (re-deciding within a family with a specialist model) held up.

### Cited Findings
- LMSYS 2024 3rd: "TTA (by flipping response_a and response_b) provided a boost depending on the backbone of ~0.007" (log loss). Disabling Gemma-2 soft-capping improved CV and LB "by roughly 0.002". — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
- WSDM Cup Multilingual Chatbot Arena (2025) 1st: two passes, "first using the merged model on all samples, then second time with response order swap using the original model … only on 33% of samples with the most uncertain predictions". Using a 72B model for the second pass meant "only 5% of samples could be processed with TTA, which ended up looking significantly worse on LB". — [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
- Jigsaw Toxic 1st: back-translations through FR/DE/ES as train and test augmentation. "Vanilla Bi-GRU 0.9862; w/ train-time aug 0.9867; w/ test-time aug 0.9865; w/ both 0.9874". "A single TTA-ed model was beating the majority of teams' … ensembled submissions." Translations stayed on the same side of the CV split as the original comment. — [Jigsaw Toxic 1st](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557)
- Recursion 1st: TTA = flips plus 90° rotations, averaging logits over sites and TTA. Extra crop-scale TTA "didn't give me any boost on the private test set" but helped public. — [Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)
- ISIC 2024 1st: "Averaging model predictions for several variations of augmented samples also did not produce any positive results." He also added Gaussian noise (std 0.1) to image-model predictions before feeding them as GBDT features, "since models tend to slightly overfit … due to early stopping", and used patient-relative ratio features (prediction / patient mean). — [ISIC 2024 1st](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533196)
- Name That LLM rule overrides:
  - **Punctuation veto**: in train, curly quotes never occur in Claude/GLM text and 2+ em-dashes almost never in Gemini, so those classes get −2 on log P. OOF +0.0007 (+0.0018 together with the Claude gate). Then: "Version 4's veto scored only +0.001 over version 3 … and two public forks with the same rule lost 0.003-0.005 … Train-only punctuation invariants … do not hold on free-proxy text, and probably not fully on the test either." It was removed in v5.
  - **Family gates**: when the blend predicts a Claude (or GPT) label, a specialist 5-way model re-decides within the family, using `argmax_c log P(c) + λ log P_spec(c)` with λ = 1.5. The specialists are trained on extra passages whose prompts carry all 5 family labels, "so topic cannot act as a label shortcut". GPT gate plus second view: OOF +0.013 over v4. Without the extra passages the gate did not help (OOF −0.0004). — [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
  - The v11 fork's own LoRA with a within-Claude head: balanced accuracy 0.4559 (public LoRA) → 0.4838 → 0.4932 with the gate. Public LB went 0.541 → 0.544 (gate) → 0.547 → 0.552. — [NTL v11 Sinkhorn 0.25](https://www.kaggle.com/code/liurouhan/name-that-llm-ntl-v11-sinkhorn-0-25)
- Output smoothing via neighbourhoods: Shopee 1st's Iterative Neighbourhood Blending replaces each embedding with a similarity-weighted sum of its thresholded neighbours, re-normalizes, and iterates over 3 stages. With diverse text models: public 0.776 → 0.784, and using all three embeddings at INB stage 1 → 0.793. — [Shopee 1st](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)

### Inferences
- "Selective TTA" (only on uncertain rows) is the budget-aware form. It is cheap to add wherever the inference budget is binding.
- Rules mined from train ("class X never contains token Y") should be validated on out-of-distribution text before use. A soft log penalty is safer than a hard veto, and the NTL evidence says even that can hurt.

### Gaps
- No isolated TTA number for LMSYS 1st (length-2000 TTA, in the existing notes).
- No isolated number for passage-chunk TTA in authorship tasks.

---

## Q5. Evidence-ranked catalog (case studies with numbers)

### Takeaway
Ranked by evidence strength, then by gain size. The tricks with an isolated, winner-reported gain are:
- Feedback Prize's span second stage (+0.036).
- QUEST's discretization (+0.027-0.030).
- HPA's NN-label replacement (+0.03).
- Shopee's min2 (+0.019) and union/thresholds (+0.019).
- Recursion's constraints plus progressive PL (+0.007 private).
- Eedi 3rd's unseen boost (+0.036 private, in the existing notes).
- Jigsaw's TTA plus PL plus stacking (+0.0013 in total, on a saturated AUC).
- Optiver's weighted-mean removal (−0.0025 to −0.005 MAE).
- Quora's per-perimeter rescale (−0.001 log loss over the public rescale).
- PS S5E5 hill climbing (weak diverse members −0.00006 RMSLE private).

### Cited Findings
| Rank | Trick | Competition, place, year | Reported gain | Risk | Source |
|---|---|---|---|---|---|
| A1 | Learned span-level 2nd stage (LGBM over over-recalled candidates) | Feedback Prize 2021, 1st, 2022 | +0.036 CV and LB (0.706 → 0.742); 65%-length sub-trick +0.008 | Low (CV = LB) | [link](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177) |
| A2 | Discretize predictions to train distribution (Spearman) | Google QUEST, 1st, 2020 | +0.027-0.030 per single model | Low-medium | [link](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840) |
| A3 | Unseen-candidate prior boost (calibrate to expected novel fraction) | Eedi, 3rd, 2024 | private 0.564 → 0.600 | Medium (prior from LB probing) | existing notes / [link](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498) |
| A4 | Metric-learning NN label replacement on test | HPA, 1st, 2019 | "+0.03" | Low (insensitive to threshold) | [link](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/discussion/78109) |
| A5 | min-2 matches rule; union of modality matches + 2 tuned thresholds; INB | Shopee, 1st, 2021 | 0.724 → 0.743; 0.757 → 0.776; 0.776 → 0.793 | Medium (thresholds tuned on LB) | [link](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136) |
| A6 | Impossible-class masking + LSA per plate + progressive PL | Recursion, 1st, 2019 | single model 0.98997 → 0.99700 private (PL); ensemble 0.99763 | Low (hard constraint) | [link](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543) |
| A7 | Uniqueness counts on real-only test (fake-row removal) | Santander, 1st, 2019 | LB .914 → .921; final .92546 private | Low once the structure is understood | [link](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003) |
| A8 | Time-id order recovery → NN features | Optiver RV, 1st, 2021 | 0.21 → 0.19 (RMSPE) | Low-medium | [link](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/discussion/274970) |
| A9 | Online learning + weighted-mean removal | Optiver Close, 1st, 2023 | PP: 5.8142 → 5.8117 val; OL: 5.4438 → 5.4030 test | Low | [link](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446) |
| A10 | Swap-order TTA (pairwise preference) | LMSYS, 3rd, 2024 | ~0.007 log loss | Low | [link](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766) |
| A11 | Back-translation TTA + train aug; rough PL; LGBM stack | Jigsaw Toxic, 1st, 2018 | 0.9862 → 0.9874 (aug); 0.9880 → 0.9885 (PL); → 0.9890 (stack) | Low | [link](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557) |
| A12 | Metric hack (degrade early weeks) | Home Credit, 1st, 2024 | ~0.03 metric (vs 0.00X model gaps) | Very high (gamble; ethically contested) | [link](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337) |
| A13 | Per-subgroup prior rescale | Quora, 1st, 2017 | ~0.001 log loss over public rescale | Medium | [link](https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355) |
| A14 | Hill climbing w/ diverse weak members; 100%-retrain trick | PS S5E5, 1st, 2025 | CV 0.05890 → 0.05880; private 0.05847 → 0.05841 | Low-medium (selection overfit) | [link](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611) |
| A15 | KS-matched train subsample (transductive covariate matching) | LANL, 1st, 2019 | single LGB 2.279 private = 1st | Medium | [link](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/discussion/94390) |
| B1 | Reciprocal-rank voting across pipelines | Eedi, 7th private / 2nd public, 2024 | not isolated | Low | [link](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551388) |
| B2 | Uncertain-only TTA | WSDM Arena, 1st, 2025 | not isolated (5%-coverage TTA was "significantly worse") | Low | [link](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902) |
| B3 | OptimizedRounder for QWK, model-select on RMSE | DSB 2019, 1st | not isolated; 0.568 private | Medium | [link](https://www.kaggle.com/competitions/data-science-bowl-2019/discussion/127469) |
| B4 | Rank-pct averaging of 150 GBDTs; noise on stacked features | ISIC 2024, 1st | not isolated | Low | [link](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533196) |
| B5 | Weighted rank average on validation only, ignoring public LB | Jigsaw Severity, 1st, 2022 | 0.8139 private (15-model) | Low | [link](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/306274) |
| B6 | Sinkhorn partial balancing (α) | Name That LLM, public notebooks, 2026 | CV +0.002-0.007 in-domain; 0.12 → 0.19 under shift | Medium (α, test balance) | [link](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) |
| B7 | Adaptive prompt-centering on test clusters | Name That LLM, public notebooks, 2026 | sim +0.024 (1-2/prompt) to +0.052 (all-18) | Medium (clustering errors) | [link](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) |
| B8 | Expected-F1 per-instance thresholding | Instacart, 2017 (forum) | None-handling +0.003 | Low if calibrated | [link](https://www.kaggle.com/competitions/basket-analysis/discussion/37016) |
| C1 | Punctuation veto rules | Name That LLM, 2026 | OOF +0.0007, LB −0.003 to −0.005 in forks | High | [link](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering) |
| C2 | Hard clipping to label support | PS S4E4, 2024 | not measured | Low | [link](https://www.kaggle.com/competitions/playground-series-s4e4/discussion/488093) |
| C3 | Power averaging (AUC) | TPS Nov 2021 guide | not measured | Medium | [link](https://www.kaggle.com/competitions/tabular-playground-series-nov-2021/discussion/284507) |

Counter-evidence (PP that hurt):
- ICR: a conservative `p[p<0.1]=0` and LB-probed pseudo-labels both hurt private. — [ICR](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/431067)
- LANL: the SVR improved CV, but "just blending LGB and NN would have produced the best private LB score (2.25909)". The final 3-model blend's private score is not stated. — [LANL](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/discussion/94390)
- Recursion: extra crop TTA helped public only. — [Recursion](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)

### Inferences
- A rule of thumb across these examples: post-processing that encodes a fact about the data-generating process (constraint, prior, discreteness, group structure) transfers to private. Post-processing whose parameters were tuned to the public LB gives a coin flip unless a CV that mimics the private split confirms it.

### Gaps
- I did not retrieve Mercari/Jigsaw-era "blend of blends" ablations with private-LB deltas. I also did not find Kaggle write-ups for DAIGT post-processing beyond rank averaging (DAIGT 2nd's write-up covers data and pretraining, not PP).

---

## Q6. Transfer to (1) CASMI 2026 (MRR@25, multi-engine candidate lists, InChIKey14 identity, ~40-50% novel) and (2) Name That LLM (18-way balanced accuracy, ~2-3 passages per prompt, Sinkhorn α)

### Takeaway
For CASMI, the transferable tricks are:
- Reciprocal-rank fusion across engines (Eedi 7th).
- Dedup on the identity key before scoring.
- A calibrated novelty prior (Eedi 3rd).
- Hill climbing directly on MRR over OOF challenges.
- Rank/percentile normalization per query before blending (ISIC/DAIGT).

For Name That LLM, the transferable tricks are:
- Sinkhorn balancing as prior correction (tune α on a pooled simulation, not the LB).
- Within-cluster group constraints if the data guarantee distinct labels per prompt (Recursion-style LSA).
- Family-level gates.
- Selective TTA.
- Not trusting train-only lexical vetoes.

### Cited Findings
- RRF-style voting (1/rank, summed across pipelines) won Eedi private 7th / public 2nd on a MAP@25 retrieval task. — [Eedi 7th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551388)
- The novel-fraction prior boost (scale unseen candidates' scores until the top-1 unseen share matches the estimated test share): +0.036 private in Eedi 3rd. — [Eedi 3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498)
- Relative-to-best-score acceptance for unseen candidates (LECR 1st), and a cascade where 1/rank-weighted top slots go to the strongest reranker (Eedi 1st): existing notes.
- Hill climbing optimizes any metric directly, including ranking metrics ("Ensemble selection allows ensembles to be optimized to performance metric such as accuracy, cross entropy, mean precision, or ROC Area"). — [Caruana 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)
- NTL Sinkhorn: the current code and its CV evidence; the v11 fork tries α ∈ {0, 0.1, 0.25, 0.5} and submits 0.25. — [NTL v11](https://www.kaggle.com/code/liurouhan/name-that-llm-ntl-v11-sinkhorn-0-25); [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- NTL clustering quality: pair precision 0.97 and recall 0.89 at the 0.65 cut with mixed TF-IDF and GLM distance; false-positive pair rate 0.012-0.059 depending on the cut at m = 150 prompts. Centering singletons failed (v1). — [LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)
- Group constraint with LSA: Recursion 1st (Q1). — [Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)

### Inferences

**CASMI 2026 (MRR@25)**
1. *Identity dedup first.* Collapse every engine's list to unique InChIKey14 (first 14 chars, the connectivity layer), after tautomer canonicalization. Take max (or log-sum-exp) of the scores within a key, then re-rank. Duplicate tautomers or stereoisomers in a list push true answers down without adding any chance of a hit, so this is pure metric gain at zero risk (the Shopee/LECR "dedup equivalent answers" analogue).
2. *Fuse engines by rank, not raw score.* Engine scores are on incomparable scales. Use `score_k = Σ_engines w_e / (k0 + rank_e)` (RRF; k0 ≈ 0-60; Eedi 7th used k0 = 0). Alternatively, use per-query percentile ranks averaged as in ISIC 2024 1st. Fit `w_e` (and k0) by Caruana hill climbing on MRR@25 over held-out challenges, with replacement and bagged, because the challenge count is small.
3. *Novelty prior.* With ~40-50% novel test molecules, candidates absent from training or reference libraries are probably under-scored by library-match engines. Apply Eedi 3rd's recipe: multiply novel-candidate scores by C, tuned on a CV that holds out novel molecules at the expected 40-50% rate, so that the top-1 novel share matches the expected share. Do not tune C on LB probes.
4. *MRR weighting.* MRR gives 1, 0.5, 0.33 … for ranks 1, 2, 3, so nearly all value is in the top 3-5. Spend expensive rerankers (and any TTA, such as several fragmentation-prediction settings) on the top 5-10 only (WSDM's uncertain-only TTA logic). Fill ranks 10-25 with diversity, i.e. candidates from different formulas or scaffolds or from other engines' top hits, rather than near-duplicates of rank 1. This is Eedi 1st's slot construction applied to MRR.
5. *Constraints.* If the challenge design guarantees that distinct challenges have distinct answers, or that the formula or adduct is known, use them as hard masks (Recursion's "impossible classes") or as an assignment across challenges. This is speculative; check the CASMI rules.

**Name That LLM (18-way balanced accuracy)**
1. *Sinkhorn α is prior correction.* Balanced accuracy is mean recall, so the Bayes-optimal decision divides by the expected class mass. Sinkhorn toward uniform with partial α is that correction plus a transductive estimate. Choose α on the pooled simulation, varying the test size and the class imbalance of the simulated test. The base notebook reports CV "never negative" at α = 0.5 in-domain, so 0.25 vs 0.5 is a variance-vs-bias choice. Hedge with the two final submissions (bestfitting/HPA and Home Credit 1st both did this).
2. *Per-prompt group constraint (biggest potential upside, unverified).* With ~2-3 passages per prompt, check in train whether passages sharing a `prompt_id` ever share a label. If they never do, solve LSA inside each detected test cluster on `-log P` (Recursion 1st). Use it only on high-precision clusters: the clustering's 3-6% false-positive pair rate means a hard constraint could also cause errors. A soft version (penalize assigning the same label twice within a cluster) is safer.
3. *Order of operations.* Blend in log-prob space (L-BFGS on prompt-grouped OOF) → Sinkhorn → family gates → (optional) group assignment. Gates should operate after balancing, as in the public line.
4. *Avoid train-only lexical vetoes.* The punctuation rule is the ICR lesson for this competition: an OOF gain did not survive the LB.
5. *Selective TTA for text.* Average predictions over 2-3 overlapping chunks or truncations of each passage, only for rows whose top-2 margin is small (WSDM's 33% rule). Jigsaw shows text TTA helps most when matched by the same augmentation at train time.
6. *Hill climbing vs L-BFGS.* With 15+ members and OOF on only a few thousand rows, a Caruana-style with-replacement, bagged selection is a cheap robustness check on the L-BFGS weights.

### Gaps
- I do not know CASMI 2026's exact rules (whether formula or adduct are given, or whether answers can repeat across challenges), or NTL's train structure (whether same-prompt passages always have distinct labels, and the exact test class balance). Both determine whether the assignment tricks apply. Check before implementing.
- The v11 fork's notebook header does not state the public LB of its α = 0.25 output separately from the 0.552 blend figure.
