---
tags: [study, report, handbook, progression, grandmaster, compute, ensembling, kaggle]
updated: 2026-10-17
related: ["[[reports/Building and introspecting great models]]", "[[10 Competition Playbook]]", "[[Competitions/CASMI26]]", "[[Competitions/Name That LLM]]"]
---
# Bank team golds, then win one solo

> [!warning] Unverified: does SHL award medals?
> SHL Hiring Assessment 2026 is a hiring competition. It does not show up in our list of medal competitions, and its rules page couldn't be read from here. Hiring and Community competitions normally award **no medals**. Everything below that counts an SHL gold toward Master assumes that it does. Check the competition's overview page first. If it awards no medals, the route to Master runs through Name That LLM and CASMI, and SHL's value is the hiring result and the write-up, not a rank. The honest-pick advice for SHL holds either way.

To become a Competitions Grandmaster you need **five gold medals, at least one of them won solo**. For an Expert on a two-person team the proven route has three parts: collect team golds while building one reusable pipeline, take the solo gold in a domain-heavy competition where insight beats headcount, and run a validation discipline strict enough that private leaderboards stop surprising you. The rules have not changed since 2016, and Kaggle explicitly left them alone in its July 2025 overhaul. Gold is still a near-fixed top 10–20 teams in almost every competition. Among Grandmasters in the current top 100, the median time from Expert to Grandmaster was **2.3 years**. Our next milestone is Master (1 gold + 2 silver), and it is reachable this season. SHL is #2 on public (gold-zone *if* it awards medals; unverified), and Name That LLM and CASMI both sit in silver. Our SHL position is also the most inflated of the three, though: 0.3116 comes from a mix fitted to public scores, while the honestly validated model scores 0.3178. So the most valuable single decision in this handbook is to keep a final pick that does not depend on public-fitted weights. Across about 65 winning write-ups, the cheap hacks that survive the private leaderboard fall into two kinds. The first changes *how a model learns*: input formatting, dropout off for regression, soft pseudo-labels and self-distillation, full-data refits and seed averaging. The second encodes a *guaranteed* fact about the test. The hacks that fail encode an assumption the CV cannot check, and that is exactly the pattern behind our own failures (per-prompt Hungarian assignment, the punctuation veto, SHL blend reshuffles). Compute is not the bottleneck. Linking two Colab Pro accounts at about $20 a month adds 30 Kaggle GPU-hours a week, and 50 hours on a rented RTX 4090 cost about $17. The real constraints are 15 research hours a week and a noise floor of roughly ±0.004 per submission.

This handbook stands on its own, but it extends [[reports/Building and introspecting great models]] and the study-guide chapters [[01 Validation & Leakage]], [[04 Ensembling]], [[05 Ranking & Retrieval]], [[06 Calibration & Label Shift]], [[09 ML Engineering & Reproducibility]] and [[10 Competition Playbook]]. Lessons we have proven on the leaderboard live in [[What Works]] and [[What Does Not Work]].

## Chapter 1. Five golds and one solo: the rules, the cutoffs and the clock

### The ladder is frozen, and the 2025 overhaul raised its relative value

Kaggle's Progression page sets three Competitions tiers. **Expert needs 2 bronze**, **Master needs 1 gold + 2 silver**, and **Grandmaster needs 5 gold including 1 solo gold**. The Datasets and Code tracks have separate, vote-driven ladders ([Kaggle Progression](https://www.kaggle.com/progression)). Kaggle explains why the Competitions track needs fewer medals: they "represent months of work" and "are limited to only the top participants in each competition" ([Kaggle Progression](https://www.kaggle.com/progression)).

The July 2025 Progression Update said plainly, "We are not making any changes to the Competitions progression track." Everything around that track did change:

- The Discussion track was retired.
- The Novice and Contributor tiers became a single "Kaggler" tier.
- Only votes from Experts and above now count toward Code and Dataset medals.
- Forking now counts as an upvote.
- Grandmaster Levels were added ([Kaggle Progression Update, 2025-07-07](https://www.kaggle.com/discussions/product-announcements/588704)).

Under the new levels, repeating the Competitions Grandmaster requirement raises your level, while repeating the other tracks does not. At launch Giba stood at **13x Grandmaster with 64 competition golds** ([Kaggle Progression Update](https://www.kaggle.com/discussions/product-announcements/588704)). After these changes, Competitions is the only track you can climb purely through your own results. Its prestige has therefore risen relative to the others.

The solo-gold rule exists so that nobody reaches Grandmaster just by joining strong teams ([TDS: How to become a Kaggle Competitions Grandmaster](https://towardsdatascience.com/how-to-become-a-kaggle-competitions-grandmaster-9d77431c5b7d/)). Two practical details are *not* stated on Kaggle's current pages. One is that a team gold counts as a full gold for every member. The other is that "solo" means a team of one at the deadline. The long-standing community understanding is yes on both, but treat that as convention rather than written rule. Every member of a medal-winning team of up to five receives the medal, so for the Master step it does not matter which of us wrote the winning code.

### Gold means a near-fixed top 10–20 teams almost everywhere

The cutoff table on the Progression page is the most important table in competitive Kaggle:

| Medal | 0–99 teams | 100–249 teams | 250–999 teams | 1000+ teams |
|---|---|---|---|---|
| Bronze | Top 40% | Top 40% | Top 100 | Top 10% |
| Silver | Top 20% | Top 20% | Top 50 | Top 5% |
| Gold | Top 10% | Top 10 | Top 10 + 0.2% | Top 10 + 0.2% |

"Top 10 + 0.2%" means one extra gold for every 500 teams. A 500-team competition awards 11 golds and a 5,000-team competition awards 20. Percentages are rounded down ([Kaggle Progression](https://www.kaggle.com/progression)). Getting Started, Playground and Community competitions typically award no medals. Each competition's overview tab states whether it does ([Kaggle Progression](https://www.kaggle.com/progression); [Kaggle Docs: Competitions](https://www.kaggle.com/docs/competitions)).

The table has a sharp consequence. **Silver scales with the size of the field but gold barely does.** A 3,500-team competition hands out 174 silvers and only 16 golds. Large competitions are good places to farm silvers, and gold is a fight against the same dozen or so strong teams whatever the size of the field. Medals and tiers never decay, while ranking points do. Kaggle has not changed the points formula since 13 May 2015 ([Kaggle Progression](https://www.kaggle.com/progression)).

### Our medal arithmetic today

Applying the table to the live team counts from 17 October 2026 gives this picture:

| Competition | Teams | Gold | Silver | Bronze | Where we stand | Medal now |
|---|---|---|---|---|---|---|
| Enveda CASMI 2026 (MRR@25) | 3,495 | top 16 | top 174 | top 349 | about rank 100, best public 0.451 | Silver |
| Name That LLM (balanced accuracy) | 1,363 | top 12 | top 68 | top 136 | rank about 53–56 at 0.556; gold line about 0.560 | Silver |
| SHL Hiring Assessment 2026 (0.6·RMSE + 0.4·(1−Pearson)) | not recorded here | top 10 (100–249 teams) to top 10 + 0.2% | top 20% / top 50 / top 5% | – | #2 public at 0.3116 vs #1 0.3064; honest main model 0.3178 | Gold (public only) |

Team counts and deadlines come from the competition pages ([Kaggle: Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra); [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)). CASMI's team-merger deadline is **7 December 2026** and its final deadline is **14 December**. Name That LLM's are **6 January** and **14 January 2027**. Both run 9-hour notebook-only reruns with a maximum team size of five. Both fields are still growing, so all the cutoffs will drift upward.

Three readings follow:

- **Name That LLM.** Our silver margin is only 12–15 places, with three months of new entries still to come. The gold gap of about 0.004 is small in score but large in rank.
- **CASMI.** Gold needs a jump of about 84 places. Our own log shows that the public LB covers only about 124 molecules, so a single extra rank-1 hit is worth about 0.008 ([[Competitions/CASMI26]]).
- **SHL.** On public we are already inside gold at any plausible field size. *If SHL awards medals* (unverified, see the warning above), that gold survives the private split and the two silvers hold, **we become Masters at the end of this season**. If it doesn't, Master needs an NTL or CASMI gold.

### Grandmasters take about two years from Expert, and the fifth gold is the hard one

As of 17 October 2026 Kaggle lists **407 Competitions Grandmasters**, 2,366 Masters and 12,448 Experts among ranked users ([Kaggle Rankings](https://www.kaggle.com/rankings); [Grandmasters page](https://www.kaggle.com/rankings/grandmasters)). That is up from the roughly 240 Dieter cited when he earned his solo gold in early 2020 ([Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)).

The research notes include an analysis of the tier-milestone dates on the ranking-history panels of the Competition GMs currently in the top 100 ([Kaggle Rankings](https://www.kaggle.com/rankings/competitions)). Among those who reached GM after the progression system launched, the medians were **2.26 years from Expert to GM (IQR 1.65–3.82)** and **1.85 years from Master to GM (IQR 1.17–3.05)**. The pace is accelerating: **15 of those 58 became GM in 2026 alone**. Recent GMs also tend to sit right at the threshold, with 5 to 9 golds. This sample is survivor-biased, so the typical path is slower.

| Grandmaster | Joined → Expert → Master → GM | Golds | What got them there |
|---|---|---|---|
| Chris Deotte ([profile](https://www.kaggle.com/cdeotte)) | 2018-03 → 2019-06 → 2019-10 → **2020-06** | 26 | Notebooks GM first through sharing; robust local validation, GPU-accelerated experiments, train/test difference analysis, metric understanding ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)) |
| Christof Henkel "Dieter" ([profile](https://www.kaggle.com/christofhenkel)) | 2017-11 → 2018-06 → 2019-01 → **2020-03** | 54 | Six team golds in a year, then a solo 5th of 2,000+ in Bengali.AI on home 3×GTX 1080 Ti after relabelling data from Bengali script rules ([Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)) |
| Theo Viel ([profile](https://www.kaggle.com/theoviel)) | 2018-07 → 2019-02 → 2019-06 → **2020-09** | 24 | About 26 months from joining |
| Psi / Philipp Singer ([profile](https://www.kaggle.com/philippsinger)) | Expert 2019-04 → Master 2019-06 → **GM 2020-06** | 38 | About 14 months from Expert |
| CPMP ([profile](https://www.kaggle.com/cpmpml)) | Expert 2016-08 → Master 2017-06 → **GM 2018-08** | 29 | Fifth gold won solo "specifically using post-processing" ([NVIDIA GTC Q&A](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)) |
| Bojan Tunguz ([profile](https://www.kaggle.com/tunguz)) | Master 2018-02 → **GM 2021-04** | 9 | About 3 years stuck at Master |
| hengck23 ([profile](https://www.kaggle.com/hengck23)) | Master 2017-12 → **GM 2023-06** | – | 5.5 years at Master |
| Jeroen Cottaar ([profile](https://www.kaggle.com/jeroencottaar)) | 2023-05 → 2025-02 → 2025-04 → **2026-02** | 5 | 2nd in Geophysical Waveform Inversion for about $700 of cloud compute ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)) |
| tomoon33 ([profile](https://www.kaggle.com/tomoon33)) | 2023-05 → 2024-09 → 2024-10 → **2026-01** | 5 | About 15 months from Master |
| James Day ([profile](https://www.kaggle.com/jsday96)) | 2019-07 → 2022-08 → 2024-01 → **2026-01** | 6 | 5th in LLM-Detect AI Text on a home 2×RTX 3090 + 4090 ([DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)) |
| sayoulala ([profile](https://www.kaggle.com/sayoulala)) | **GM 2024-12** | – | First solo gold by winning LMSYS Chatbot Arena outright ([LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)) |

The common thread across these journeys is sequence. Dieter, Deotte and Theo banked team golds first. Dieter entered more competitions per year, joining late but bringing a standardised baseline: "more and more parts of the work became standardized" ([Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)). The solo gold then came from a competition chosen to suit the person. CPMP took his with post-processing, Dieter with domain research, and sayoulala with an LLM-distillation pipeline that he had already proven elsewhere.

The stuck cases show where the time goes. Bojan's three years and hengck23's five and a half years at Master are the cost of a missing fifth or solo gold, not of missing skill.

The base rates are better than they look. In 2025 **more than half of winning teams were solo for the third year running, and over half were first-time winners** ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)).

### A realistic timeline for us

If SHL awards medals and its gold and both silvers hold, we reach Master once Name That LLM finalises in mid-January 2027. From there the survivor-biased median of 1.85 years from Master to GM points to late 2028. The recent fast cases took 10–15 months from Master, which points to early 2028. Either way we still need four more golds, including one solo.

Our working plan is three phases. Through 2027, enter about four to six medal competitions a year as a team, with one reusable pipeline per problem family (retrieve-and-rerank, LoRA text classification, audio regression). Then each of us takes one solo attempt in a niche where we have a domain edge. MS/MS-style chemistry is the obvious candidate. Plan for 2029 and treat 2028 as the upside case. This is our projection from the cited medians, not a sourced forecast. The notes contain no published figure for the median number of competitions entered before reaching GM.

## Chapter 2. Pick fights where gold is cheap and shake-ups are survivable

### Medal arithmetic favours mid-sized, domain-heavy competitions

Two structural facts should drive competition choice. The first is the fixed cutoffs for mid-sized fields: with 250–999 teams, silver is a fixed **top 50** and bronze a fixed **top 100**, while gold stays at about 10–11 ([Kaggle Progression](https://www.kaggle.com/progression)). That band is the friendliest for silvers.

The second is that domain-heavy and data-heavy competitions draw fewer casual entrants. Grandmaster advice notes that "computer vision competitions often attract significantly fewer competitors, mostly due to large data sizes" ([TDS](https://towardsdatascience.com/how-to-become-a-kaggle-competitions-grandmaster-9d77431c5b7d/)). Dieter's criterion for a solo attempt is the sharpest version of this. Pick a competition "interesting enough to provide fun during weeks of working alone, but also… challenging enough to leave room for unique ideas" ([Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)).

Supply is limited. Kaggle ran 68 competitions in 2025, and 34 of them were Community competitions, which typically carry no medals ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)). Each medal-bearing slot is therefore worth planning for.

Before entering, read what the host writes and the top solutions of similar recent competitions. NVIDIA's grandmasters note that "some competitions are won by models that are publicly shared from previous competitions and adapted… People do not read enough." Bo Liu's filter is equally useful: "If I do not have any interesting ideas, then I do not join" ([NVIDIA GTC Q&A](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)).

Code competitions carry one extra selection risk. Notebooks rerun on hidden data, and Kaggle's docs single out notebooks that are not robust to an unseen stage-2 test as a common pitfall ([Kaggle Docs: Competitions](https://www.kaggle.com/docs/competitions)). Join late only if you already own a pipeline for that problem family. Even then, Dieter's late-entry pattern needs three to four weeks for a medal attempt.

### Teammates should be close on the leaderboard and far apart in method

Kaggle's own advice is to approach "people who are close to your leaderboard position", and only after doing real work yourself. "In every single team-up, I learned stuff" ([NVIDIA GTC Q&A](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)). Teams gain "a significant blending effect" even when members work independently and only combine at the end, which is exactly what a solo competitor must beat ([Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)).

The mechanics are simple and binding:

- Everyone competes as a team, even a team of one.
- Merges are allowed until the merger deadline, unless the combined team would exceed the size limit or the historical submission limit.
- "Submission limits (usually 5 per day) apply to the whole team" ([Kaggle Docs: Competitions](https://www.kaggle.com/docs/competitions)).

The usual statement of the historical limit (combined submissions no more than the daily maximum times days elapsed) is secondhand, from AIcrowd's identical rule ([AIcrowd discourse](https://discourse.aicrowd.com/t/is-there-a-limit-on-the-total-number-of-submissions-when-merging-teams/7825)). Check the competition's Rules page before planning a merge.

Our own leaderboard has already shown that **merging correlated work adds nothing**. We merged two Name That LLM public notebooks that share a base, and the result scored 0.550, below both parents ([[What Does Not Work]]). The literature predicts this. Team Hydrogen found that "single model scores did not correlate well… with their ability to blend… diversity really mattered" ([FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)), and Mercari's Lasso blender simply zeroed out redundant models ([Mercari 1st](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50256)).

So before agreeing to any merge, ask the other team for out-of-fold predictions on a shared fold split and measure the blend gain. Merge only if their pipeline is independent of ours. That is our rule, derived from these sources.

Inside a team, split the work along the pipeline, not by model. In LMSYS 3rd one member generated the 500k pseudo-labelled set while another trained. In LMSYS 2nd one member found the full-swap augmentation while another ran the last ten days of experiments on 4×A100. LMSYS 3rd's final ensemble used one Gemma-2 per teammate "for diversity" ([LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766); [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)).

For us the solo-gold requirement has a planning consequence. **At some point each of us must enter a medal competition alone.** That is simplest in a competition the other is not entering, and it should be scheduled deliberately, not left to chance.

### Shake-ups punish public-LB tuning, not bold ideas

Shake-ups are where Master-level teams lose golds. The documented causes are consistent:

- **Luck at the top.** In BELKA, three of the top five teams ran the same tutorial notebook with minimal changes, and the winner called his result "pure luck and randomness" ([Leash Bio](https://leashbio.substack.com/p/belka-results-suggest-computers-can); [BELKA 1st](https://www.kaggle.com/competitions/leash-BELKA/discussion/519020)). The LLM 20 Questions winner said "any competitor in the gold zone could have potentially secured the win" ([LLM 20Q 1st](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106)).
- **Public-LB probing.** Chris Deotte's ICR write-up admits that "using probed positive targets from public LB actually hurt my private LB score", and that even a conservative threshold hurt ([ICR](https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/431067)).
- **Tricks that only move public.** CLRP 3rd's chain of head tricks took public from 0.462 to 0.453 while "Private LB stayed at ~0.461" ([CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)). Recursion's extra crop TTA likewise helped public only ([Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)).

The same evidence shows who *gains* in shake-ups. Teams with robust, validated signals win, such as DAIGT 8th, who rose sharply on perplexity features ([DAIGT 8th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470224)).

Home Credit 2024's winner quantified the boundary. CV differences of about **0.001–0.005 "had a low correlation with LB"**, differences above 0.01 had a high one, and gains from feature engineering correlated with LB far better than gains from parameter tuning ([Home Credit 1st](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337)). Our SHL experience fits that pattern exactly. **Procedure changes transferred to the leaderboard** (random crops, a bias fix, attention pooling on random crops). **Changes to blend composition did not.**

The defence comes from Ruchi Bhatia: "simulating the public/private LB split using multiple CV folds to estimate potential leaderboard shakeups" ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)). Judge every trick by its spread across folds, not by its best fold or its public delta.

## Chapter 3. Trust a CV that copies the test, then pick one safe and one bet

### The group key, and the way the test is assembled, decide whether CV means anything

Nearly every first-place write-up begins by showing that CV tracks the leaderboard. Folds and seeds alone don't get them there. The deciding choice is the grouping. Raja Biswas's Eedi sweep is the template. Folds by `QuestionId` "produced overly optimistic estimates", folds by `SubjectId` were "overly pessimistic", and GroupKFold on `ConstructId` "yielded a narrow Validation/LB gap and good correlation" ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)).

A constant offset between CV and LB is harmless. Ahmet Erdem's LECR validation sat at 0.764 against a public 0.727, yet it moved with the LB and matched his private score ([LECR 1st](https://www.kaggle.com/competitions/learning-equality-curriculum-recommendations/discussion/394812)).

Adversarial validation is the standard shift check. Train a classifier to tell train rows from test rows; an AUC near 0.5 means no shift ([FastML via KDnuggets](https://www.kdnuggets.com/2016/10/adversarial-validation-explained.html)). When test targets are novel, validate the novelty itself:

- BELKA 1st held out 3% of building blocks ([BELKA 1st](https://www.kaggle.com/competitions/leash-BELKA/discussion/519020)).
- Eedi 4th scored unseen-misconception subsets separately ([Eedi 4th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559)).

Our Name That LLM failure adds a second requirement beyond grouping. **The CV must reproduce how the hidden test is assembled.** Our prompt-grouped folds were correctly grouped, but they kept train's structure of exactly one passage per model per prompt. Inside that structure per-prompt Hungarian assignment looked like a gift, taking CV from 0.342 to 0.433. On the leaderboard it took 0.345 down to 0.335, because the hidden test has unequal per-model counts and only two or three passages per prompt ([[Competitions/Name That LLM]]).

The public notebook line in that competition shows the right template. It evaluates in a pooled simulation that varies the test regime (1–2, 1–5, 3–8 or all 18 passages per prompt) and reports each change's gain per regime ([LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)).

### Error analysis puts a ceiling on every idea

Before tuning anything, read the failures. Andrew Ng's recipe is to tag about **100 misclassified dev examples** in a spreadsheet. Each category's share of errors caps what fixing it can gain. Split the dev data into an "Eyeball" set you read and a "Blackbox" set you only score, and when the system has stages, use "error analysis by parts" ([Ng, MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf)). Hamel Husain's open/axial coding is the LLM-era version. In his NurtureBoss case, three issues accounted for over 60% of problems ([Husain & Shankar, Evals FAQ](https://hamel.dev/blog/posts/evals-faq/); [Husain, Field Guide](https://hamel.dev/blog/posts/field-guide/)).

Karpathy's checks come first: fix the seed, check the loss at initialisation (ln 18 ≈ 2.89 for an 18-way head), beat an input-independent baseline, and overfit one batch ([Karpathy 2019](https://karpathy.github.io/2019/04/25/recipe/)). For retrieval, track recall@K separately from the end metric. Eedi 1st "prioritized optimizing for `recall@32`" and shipped the higher-recall retriever even though another encoder had better MAP ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)).

Our CASMI bench shows how much this buys. On 250 natural-product examples, class 3 scored MRR 0.151 against 0.967 for class 1. Seventy-nine class-3 answers were simply missing from the candidate list, worth +0.316 MRR if all were fixed. Error analysis by parts then traced most of them to *parent selection* in analogue generation, not to the reranker ([[Competitions/CASMI26]]).

### Ablations need seeds and paired folds, and our forecasts need a haircut

Single-run comparisons are routinely wrong. Bouthillier et al. propose declaring A better than B only when **P(A>B) ≥ 0.75** across paired splits or seeds. Single-point comparisons gave roughly a 10% false-positive rate in their simulation ([arXiv 2103.03098](https://arxiv.org/pdf/2103.03098v1)). Psi's habit is the practitioner's version. He trains three seeds per experiment and "only if the average of the three seeds improves" promotes the change from one fold to all five ([ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)). AES2's winner averaged three seeds of everything to cut variance ([AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)). Our operational rule is that a change must win **at least 12 of 15 fold×seed pairs** (5 folds × 3 seeds at γ = 0.75). We keep a "didn't work" log next to the ablation table.

Iteration speed then decides how many honest ablations fit into a week. The fast-loop ladder has four rungs:

1. Smoke-test on 1–5% of the data in a CPU session. Otto 1st experimented by "training with 5% of the data and evaluating with other 10%" ([Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022)).
2. Develop on one fixed fold. Eedi 1st "always trained my models on folds 1-4 and validated on fold 0" ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)).
3. Run full k-fold only for winners on that fold.
4. Run a full fit plus seeds only for submission candidates.

Write a hypothesis list up front, as Eedi 1st's "Meta Thoughts" did, with CPMP's estimate of the time each idea takes to code and run ([SED 1817](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)). Timebox any idea that has shown no fold-0 signal after two attempts. A teammate code check is the cheapest debugging step: LMSYS 2nd's stalled Gemma-2-27B was fixed only by comparing batch size and gradient-clipping settings with a teammate's code ([LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)).

We have measured two facts of our own that change how forecasts should be read:

- **Noise floor.** CASMI's per-submission noise floor is about **±0.004**. Run-to-run reruns of the same notebook differ by roughly that much.
- **Forecast bias.** Our forecasts for new changes have run about **0.002 too optimistic**.

The bias is the winner's curse in miniature. We ship the changes whose estimates look best, so the shipped estimates are biased upward. Subtract 0.002 from every forecast before deciding. If the corrected forecast is below the 0.004 floor, one public submission cannot confirm the change. Decide it on paired CV, or bundle it with other changes.

The same logic covers CASMI's ~4× shrinkage from offline gains to the LB, which we logged in [[What Does Not Work]]. Calibrate a bench against changes whose LB result you already know before trusting it. Our unseen-molecule bench got 2 of 3 known directions right, and it failed exactly where its class mix differs from the test.

### Spend submissions on calibration, not discovery

CPMP submits early "to see if my cross-validation correlates with the score. If it does, then I don't need to submit too much work" ([SED 1817](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)). LMSYS 3rd's correlation was so tight that "0.001 improvements in CV would show up on the LB" ([LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)).

Our budget is 5 submissions a day per competition with two often left for the teammate. That gives about 3 a day for our own work: roughly 170 more in CASMI before 14 December and roughly 270 in Name That LLM before 14 January. **Submissions are not our scarce resource. GPU-hours and the 15 research hours a week are.** Use the slots in the first weeks for a CV-versus-LB calibration scatter of 3–5 deliberately diverse models. After that, mostly confirm changes that have already won on paired CV.

Log every submission: notebook slug and version, dataset versions, config id, fold-0 score, full CV, public LB, runtime, and the free-text `-m` message from `kaggle competitions submit -k <user>/<slug> -v <version> -m "..."` ([Kaggle API tutorials](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/tutorials.md)). Eedi 2nd also kept inference time in its per-model table and used it to allocate the runtime budget ([Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)).

### Final picks: one safe, one hypothesis, everything frozen

The best of your selected submissions is what counts, so two picks are a portfolio, not a vote. The write-ups show both sides of the lesson:

| Team | What they did with the picks | Outcome |
|---|---|---|
| Eedi 2nd | Spent the third pick on a 72B model the public LB disliked but CV trusted | That model gave "outstanding Private Leaderboard scores" ([Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)) |
| ELL 1st | Skipped its best-CV submission "fearing overfitting" | That submission had the best private score ([ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)) |
| DAIGT 5th | Left its best submission out of the final three | Dropped from 3rd to 5th ([DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)) |
| HPA 1st (bestfitting) | Hedged a label-ratio assumption with a second submission using averaged ratios | Won ([HPA 1st](https://www.kaggle.com/competitions/human-protein-atlas-image-classification/discussion/78109)) |
| Home Credit 1st | Submitted one metric-hack and one no-hack pick | Won ([Home Credit 1st](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337)) |
| Jigsaw Severity 1st | Ignored a misleading public LB and selected on validation only | Won ([Jigsaw Severity 1st](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/discussion/306274)) |

The defensible default follows. **Pick A** is the best seed-averaged CV ensemble whose CV-to-LB relationship looks normal. **Pick B** carries a stated, CV-backed hypothesis about the private test, such as a prior shift, a bigger model or a different lineage.

Then freeze every artifact both picks depend on. LMSYS's winner lost a pick because "I deleted another model that I had uploaded" ([LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)). In code competitions, write a valid `submission.csv` from the cheapest model first and wrap every optional stage in a time guard. Our own rule is the same: every add-on step runs in a subprocess with a deadline and falls back to the source submission ([[What Works]]).

## Chapter 4. Small hacks with measured wins, and the ones that bit us

The tables below collect the cheap tricks that winners report with numbers. Each entry carries its evidence tier: **A** means two or more independent top teams, or an explicit ablation in a top-3 write-up; **B** means one top team with a number; **C** means listed as working but without a number, or conflicting reports. One rule cuts across all six tables. **Post-processing that encodes a fact about how the data were generated transfers to private. Post-processing whose parameters were tuned to the public LB is a coin flip** unless a CV that mimics the private split confirms it.

### Training hacks: inputs and regularisation beat optimiser tricks

| Hack | Reported gain | Tier | Risk / when it fails |
|---|---|---|---|
| Pool only over the tokens that carry the label (head mask) | "Biggest impact… the 'magic'" in CommonLit Summaries 2nd ([CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)) | A | AES2 1st found attention pooling did not work ([AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)) |
| Keep `\n` and formatting through the tokenizer | Single-fold public 0.595 → 0.630 ([FP2021 4th](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)) | A | None reported; DeBERTa's tokenizer silently drops line breaks |
| Prefix a data-source or prompt tag | QWK 0.8370 → 0.8446 ([AES2 4th](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516639)); prompts +0.002–0.005 ([FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)) | A | Needs the test-like tag at inference |
| Dropout 0 in the backbone for regression | "[High Impact]" ([ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)), echoed by five other top teams | A | For classification, useful only on MLM-pretrained weights ([FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381)) |
| AWP / FGM adversarial training from epoch 2 | +0.01 CV / +0.003 LB ([FP2021 1st](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)); 0.005–0.01 ([FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)); FGM 0.002–0.005 ([USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)) | A | Neutral on small regression sets (CLES 2nd–4th, AES2 2nd); 1.5–2× step cost |
| Sweep train-vs-inference max length | CV 0.4984 → 0.4867 ([CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524)); train 2048 / infer 640 "[High Impact]" (ELL 3rd); train 512 beat 1024, 0.965 vs 0.957 private ([DAIGT 7th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643)) | A | The sign varies by competition; CLES 4th's insurance sub without extension scored nearly the same |
| Start from a task-adjacent checkpoint | "Up to 20 points" on CV and LB, read as about 0.020 log loss ([LMSYS 5th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669)) | B | Needs a relevant public pretraining set |
| LoRA r = 64 on all linear layers | De-facto default across LMSYS 1st/3rd/4th and DAIGT 1st ([LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)) | A | Searching alpha, DoRA or rsLoRA gave "no significant gain" |
| Find the LR cliff | Accuracy flat at about 0.70 for 7e-5 to 12e-5, collapse to 0.508 at 14e-5 ([WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)) | B | Larger models need a smaller LR |
| LLRD, re-init, freezing, EMA, SWA, multi-sample dropout | EMA 0.001–0.003 ([USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)); MSD 0.002–0.005 (FPE 3rd) | B/C | As many top teams call them useless; CLRP 3rd's chain moved public 0.462 → 0.453 while private stayed ~0.461 ([CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)) |
| Compare 3-seed averages; retrain on full data | Full-train weighted ×2 beat 1:1; single model 0.8710 (1 fold) → 0.8745 (full) → 0.8761 (5 folds) ([USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)) | A | Full fits behave slightly differently from OOFs (AES2 1st) |

### Data hacks: soft pseudo-labels and distillation are the largest repeatable wins

| Hack | Reported gain | Tier | Risk / when it fails |
|---|---|---|---|
| Soft pseudo-labels, one label set per fold, pre-train on them then fine-tune on gold | +0.004–0.007 (AES2 1st); CV 0.4581 → 0.4476 (CLES 2nd); CV 0.4514 → 0.4469 ([ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)) | A | Labelling with all folds leaks: QUEST 1st's leaked 0.445 vs an honest 0.422 ([QUEST 1st](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840)). Distribution mismatch: ELL 3rd's CV improved 0.4470 → 0.4370 with no LB gain. A second round is often neutral |
| Knowledge distillation and self-distillation on OOF soft labels | +0.005–0.01 ([PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)); 0.7051 → 0.7135 at T = 5, w = 0.9, and a 14B self-distil matched a 72B teacher ([WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)) | A | Needs per-fold labels; the gain is label cleaning, so expect less on clean labels |
| Pre-train on related or old labels, then fine-tune on the target | +0.015, the largest gain in AES2 (AES2 1st); private 0.967 → 0.983 ([DAIGT 2nd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395)) | A | Needs related labelled data |
| Task-specific augmentation | Full A/B swap in one optimiser step: stable +0.003 ([LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)); random typos ([DAIGT 4th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179)) | A | Generic back-translation fails (FP2021 1st, FPE 15th, AES2 1st) |
| Confident test pseudo-labels into a cheap branch | Private 0.893 → 0.927 with public unchanged ([DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)) | B | Invisible on public, so it must be judged on CV |
| Unsupervised fits over train plus *real* test | Santander LB .910 → .914 → .921 ([Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)) | A | Fake rows must be removed (Santander 3rd); MLM on a tiny test set failed outright at 0.689 (DAIGT 7th) |
| Shift predictions to the test mean | −0.02 shift gave +0.001 LB ([CLRP 15th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260800)) | B | Needs a reliable estimate of the shift |

### Tabular and ranking hacks: features about how a candidate was found

| Hack | Reported gain | Tier | Risk / when it fails |
|---|---|---|---|
| Candidate-source flags plus rank-within-source | 0.02855 → 0.03262; without them, more candidates made CV "very very poor" ([H&M 3rd](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)) | A | None; it is what makes recall expansion safe |
| More, and more diverse, retrieval sources; stage-1 rank as a feature | 3 → 20 co-visit matrices 0.575 → 0.590, reranker +0.011 ([Otto 3rd](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)); stage-1 rank "the important feature" ([Otto 5th](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382802)) | A | Compute |
| Groupby aggregates and target encoding inside nested folds, brute-forced on GPU | Best 500 of 10,000 features won 1st ([NVIDIA cuDF FE](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)) | A | Leaks without nested CV |
| OOF meta-features from a finer-grained model | 0.799 → 0.800 ([AMEX 11th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786)) | B | – |
| DART with a very large `min_data_in_leaf` | "The main thing that helped us align Train / CV with LB" ([AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)) | B | Slow; select features with gbdt |
| Full-data refit at the CV-best round count plus a margin | "CV matched LB very well" ([Otto 2nd](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839)); Deotte's +25% rounds trick ([PS S5E5 1st](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611)) | A | None reported |
| Ranking objective (CatBoost ranker, lambdarank) | +0.0007 / +0.002 / +0.0012 (Otto 2nd); graded gains +0.0005 (Otto 5th) | B | Second-order: H&M 1st won with binary ([H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)) |
| Adversarial-validation feature drop as a separate pick | That submission "was the best in privateLB" (AMEX 11th) | B | Costs CV |
| Hard-negative mining | Did not help recall@32 (Eedi 1st); "hard negatives were not effective" ([Eedi 5th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551391)) | C | Check recall@K before keeping it |
| Seed or hyperparameter bags | Single ranker 0.604 → 9-ranker bag 0.605 (Otto 1st) | B | "Many models with different seed" did not help AMEX 2nd |

### Post-processing: encode facts about the test, never public-LB guesses

| Hack | Reported gain | Tier | Risk / when it fails |
|---|---|---|---|
| Learned second stage over over-recalled candidates | +0.036 CV and LB, 0.706 → 0.742 ([FP2021 1st](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)) | A | Low; CV equals LB |
| Discretise predictions to the train distribution for a rank metric | +0.027–0.030 Spearman per model ([QUEST 1st](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840)) | A | Helps rank-correlation metrics only; it hurt our SHL RMSE+Pearson metric |
| Hard assignment under a *guaranteed* group constraint (LSA) | Single model 0.98997 private with the constraint ([Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)) | A | One violation cascades through a group; this failed for us in NTL |
| Metric-structure constraint (weighted-mean removal) | Ensemble 5.8142 → 5.8117 ([Optiver 2023 1st](https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/487446)) | A | Very low |
| Minimum-cardinality rule | Public 0.724 → 0.743 ([Shopee 1st](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238136)) | A | Thresholds tuned on the LB |
| Novel-candidate prior boost | Private 0.564 → 0.600 ([Eedi 3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498)) | A | The prior was estimated by LB probing |
| Prior / label-shift rescale | About 0.001 log loss over the global rescale ([Quora 1st](https://www.kaggle.com/competitions/quora-question-pairs/discussion/34355)); estimation via BBSE ([Lipton et al.](https://arxiv.org/abs/1802.03916)) | A/B | A wrong π_test; covariate shift that is not pure label shift |
| Partial Sinkhorn balancing toward uniform class mass | CV +0.002–0.007 in-domain, 0.12 → 0.19 under shift ([LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)) | B | α choice; NTL's test class mix is unequal and differs between public and private |
| Per-class scaling or thresholds | "Consistently 0.02-0.03" ([PII 2nd](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352)); Powell thresholds averaged over 15 starts (AES2 1st) | B | Thresholds for ordinal metrics are high-variance; DSB 2019 1st selected models on RMSE instead ([DSB 1st](https://www.kaggle.com/competitions/data-science-bowl-2019/discussion/127469)) |
| Match the train mean (calibration) | Built into every model by Team Hydrogen ([FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)) | B | Low |
| Thresholds or rare-class counts tuned on public; metric hacks | Home Credit hack about 0.03 against "0.00X" model gaps ("it became gambling"); ICR thresholds hurt private | A (as a warning) | Very high |
| Rules mined from train-only invariants | NTL punctuation veto: OOF +0.0007, LB −0.003 to −0.005 in public forks (LoRA x Adaptive Centering) | C | High; it also failed for us |

### Ensembling and TTA: hill climbing with replacement, selective TTA

Caruana's ensemble selection is the best-evidenced combiner. It means greedy forward selection over saved OOF predictions, **with replacement, a sorted initialisation and bagging**. It beat every other method on 10 of 10 metrics (0.969 normalised score against 0.948 for Bayesian model averaging). Logistic-regression stacking over 2,000 correlated models "overfits dramatically" ([Caruana et al. 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf)).

Deotte won Playground S5E5 with GPU hill climbing. Three XGBoost models that were weak on their own moved CV from 0.05890 to 0.05880 and private from 0.05847 to 0.05841: "Diversity is more important than single model CV score" ([PS S5E5 1st](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/582611)). Allowing negative weights gained ELL 3rd **+0.0010 CV and +0.0060 private** ([ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)). Psi bounded his weights to [1, 3] to avoid overfitting, even though an unselected negative-weight submission would have finished #2 ([ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)).

Selection can also overfit. LANL's SVR improved CV, yet "just blending LGB and NN would have produced the best private LB score" ([LANL 1st](https://www.kaggle.com/competitions/LANL-Earthquake-Prediction/discussion/94390)). The fix is to run hill climbing inside an outer CV ([PS S3E24 thread](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450827)).

Match the averaging space to the metric. Use log1p for RMSLE ([Deotte on RMSLE](https://www.kaggle.com/competitions/playground-series-s5e5/discussion/576111)), log-probabilities for log loss and argmax metrics, and ranks for AUC and MRR. Rank averaging ([ISIC 2024 1st](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533196)) and Lasso or linear blends ([Mercari 1st](https://www.kaggle.com/competitions/mercari-price-suggestion-challenge/discussion/50256); [QUEST 1st](https://www.kaggle.com/competitions/google-quest-challenge/discussion/129840)) were as good as anything fancier. Jigsaw 2018's LightGBM stacker added 0.9885 → 0.9890 ([Jigsaw Toxic 1st](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557)).

For TTA, the evidence splits by whether the augmentation encodes a real invariance:

- **Swapping A/B order** in pairwise preference was worth about **0.007 log loss** ([LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)).
- **Selective TTA**: WSDM 1st recovered most of the benefit by applying TTA only to "33% of samples with the most uncertain predictions" ([WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)).
- **Back-translation** helped most when used at both train and test time: 0.9862 → 0.9874 ([Jigsaw Toxic 1st](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/discussion/52557)).
- **Generic perturbation TTA** gave ISIC 2024's winner nothing, and it helped Recursion on public only ([ISIC 2024 1st](https://www.kaggle.com/competitions/isic-2024-challenge/discussion/533196); [Recursion 1st](https://www.kaggle.com/competitions/recursion-cellular-image-classification/discussion/110543)).

When an ensemble must fit a time limit, select by elimination. CLRP 4th "started with all of my models and iteratively removed the worst… until I could get down to the GPU time limit" ([CLRP 4th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258148)).

### Workflow and compute hacks multiply experiments per hour

| Hack | Reported effect | Source |
|---|---|---|
| GPU dataframes (cuDF-pandas, zero code change) | About 150× on a 5 GB join/groupby benchmark | [NVIDIA cuDF](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/) |
| GPU GBDT and RAPIDS | 100 XGB models for nested 10×10 CV in 2 h on one V100 | [AMEX 14th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641) |
| polars instead of pandas merges; CatBoost GPU inference | 40× faster joins; GPU inference 30× faster than CPU LightGBM | [Otto 2nd](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839) |
| Cache expensive LLM outputs offline | Retriever +2–4%, and it "costs nothing in submission" | [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651) |
| No-padding packing / varlen attention; micro-batch collation | Pad-free training and inference; DeBERTa rewrite 15%+ faster training, 30%+ faster inference | [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685); [AES2 4th](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516639) |
| Sort inputs by length at inference | Estimated 7 h; without sorting it would have exceeded the 9 h limit | [CLES 1st](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/447293) |
| 8-bit instead of 4-bit quantisation | "No decrease in score"; int8 315 vs 210 tok/s for BF16, while int4 lost about 10 AIME points | [LMSYS 9th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704); [NemoSkills AIMO-2](https://arxiv.org/html/2504.16891) |
| vLLM prefix caching | About 10% of inference time saved | [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651) |
| Headless runs with `kaggle kernels push`, pulled with `kaggle kernels output` | Several runs in flight without browser sessions | [Kaggle API kernels docs](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/kernels.md) |

### What already failed for us, and why the literature predicted it

These are our own leaderboard results. They are tried-and-failed for us, not recommendations. Each one matches a precedent above.

| What we tried | Our result | Why it failed | Lesson |
|---|---|---|---|
| NTL per-prompt Hungarian / linear-sum assignment | CV 0.342 → 0.433, but LB 0.345 → **0.335** | The hidden test does not keep train's one-passage-per-model-per-prompt structure. Recursion's LSA works only because its plate design *guarantees* the constraint | Use constraints only when the test design guarantees them, and validate in a test-assembly simulation |
| NTL punctuation veto | **−0.003 to −0.005** LB | Train-only lexical invariants; public forks lost the same amount | No hard rules mined from train |
| Merging two NTL public notebooks that share a base | **0.550**, below both parents | Correlated members (Hydrogen, Mercari) | Measure blend gain on shared OOF before merging |
| SHL rounding to the label grid | Worse | QUEST's discretisation helps rank metrics; our 0.6·RMSE term pays for every quantisation error, and Pearson is not rank-based | Match post-processing to the metric's algebra |
| SHL extra frozen encoders; kernel ridge heads | Worse | Blend-composition changes; selection overfits small OOF sets (Caruana; LANL) | Change procedure, not membership |
| SHL co-training on unlabelled clips | Worse | Pseudo-labels fail under distribution mismatch (ELL 3rd; PII 1st) | Self-distil on our own OOF instead, if anything |
| SHL TTA over shifted windows | Worse | Perturbation TTA gave nothing in ISIC 2024 and helped Recursion on public only | Skip it unless the augmentation encodes a real invariance |
| SHL LLM judges; bigger audio LLMs (Voxtral-Small, Phi-4); a 45 s batch offset | Worse | Bigger models that cannot be trained well are the LMSYS-27B rabbit hole | Timebox to two attempts |
| *Contrast:* SHL random crops, bias fix, attention pooling on random crops | **Transferred** | These change how the model learns; feature/procedure gains correlate with LB (Home Credit 1st) | Spend the remaining SHL effort here |

## Chapter 5. Compute: free hours first, rented hours for training, T4 tricks for inference

### Kaggle's free tier plus the Colab link is the cheapest GPU time available

Kaggle's documentation gives the free tier as follows ([Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks); [Kaggle Docs: TPU](https://www.kaggle.com/docs/tpu); [Kaggle Docs: Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)):

| Resource | Free-tier allowance |
|---|---|
| GPU quota | "30 hours or sometimes higher depending on demand", resetting weekly, on a P100 or 2×T4 |
| TPU quota | 20 h a week, at most 9 h per session |
| Session length | 12 h for CPU and GPU sessions |
| Machine | 4 CPU cores, 29–30 GB RAM |
| Disk | 20 GB auto-saved `/kaggle/working` |

Submission reruns for code competitions "are always run on Intel Skylake CPUs" ([Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks)). Kaggle's own tips are worth following: don't use commits for checkpointing, cancel duplicate batch sessions, stop sessions before closing the tab, and push versions through the API instead of holding interactive sessions open ([Kaggle Docs: Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)). We have hit one more limit ourselves: at most **two batch GPU sessions at a time** per account ([[What Does Not Work]]). The API's accelerator list now also names L4, A100, H100, RTX Pro 6000 and TPU v5e/v6e, but "some of these are only available to participants of specific competitions" ([Kaggle API kernel metadata](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/kernels_metadata.md)).

The most underused upgrade is linking Colab. "Colab Pro and Pro+ users will get 15 and 30 hours of extra GPU hours per week respectively on Kaggle… These extra GPU hours don't consume your Colab compute units", on the same T4/P100 hardware. Kaggle marks the scheme experimental ([Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks)). Colab Pro is about $9.99 a month for 100 compute units, and a T4 burns about 1.2–1.8 units an hour; these are secondary sources, since Google's pricing page is JavaScript-only ([McCormick ML](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/); [gpuperhour](https://gpuperhour.com/blog/google-colab-alternatives)).

Our arithmetic follows. One $9.99 subscription buys about 65 extra Kaggle 2×T4 hours a month, roughly **$0.15 an hour** before counting Colab's own units, and it arrives in the environment where our code-competition notebooks already run. Our working team budget is about 30 GPU-hours a week for research. Kaggle's quota is per account, and the CASMI log already sends the teammate's own quota to NTL copies ([[Competitions/CASMI26]]). Two Colab Pro links would add 30 hours a week for about $20 a month.

### Rented GPUs cost cents per hour, and buying one rarely pays

On-demand prices on 17 October 2026, in dollars per GPU-hour:

| GPU | RunPod Community / Secure | Vast.ai median (verified) | Lambda | GCP Spot (whole VM) |
|---|---|---|---|---|
| RTX 3090 | 0.22 / 0.50 | 0.25 | – | – |
| RTX 4090 | 0.34 / 0.89 | 0.64 | – | – |
| RTX 5090 | 0.69 / 1.19 | 0.94 | – | – |
| L4 | 0.44 / 0.59 | 0.36 | – | 0.42 (g2-standard-4) |
| A100 80GB | 1.19–1.39 / 1.79 | 0.74–0.79 | 1.99 (40GB); 2.79 per GPU in 8× nodes | 3.04 |
| H100 | 1.99–2.69 / 2.89–3.99 | 2.51–3.32 | 3.29–4.29 | 6.60 |

Sources: [RunPod pricing](https://www.runpod.io/pricing), [Vast.ai pricing](https://vast.ai/pricing), [Lambda pricing](https://lambda.ai/pricing), [GCP Spot pricing](https://cloud.google.com/spot-vms/pricing). GCP spot prices "can change up to once every day" and offer "discounts of up to 91%", which suits checkpointed training but not long uninterrupted generation jobs.

At these rates the heavy steps are cheap. Fifty hours on a RunPod Community RTX 4090 cost about **$17**. Twenty hours on an A100 80GB cost about $24–36, and twenty on an H100 PCIe about $40.

Owning hardware looks worse by comparison. A used RTX 3090 sells for about $1,000–1,350 and a used 4090 for about $2,250–3,000, with prices pushed up by local-AI demand ([InsiderLLM](https://insiderllm.com/guides/used-rtx-3090-buying-guide/); [runaihome](https://runaihome.com/blog/used-rtx-4090-vs-new-rtx-5090-2026/)). Against $0.22–0.34 rental, that buys roughly 3,000–6,000 rented hours. At 20–40 hours a week, a card would take two to five years to pay for itself, before electricity and before resale value. That is our arithmetic. Owning pays for convenience and uninterrupted experiments, as Dieter's 3×1080 Ti and James Day's 2×3090 + 4090 show, not for cost.

### What winners actually spent

ML Contests' 2025 survey covered winners across all platforms. Most winning solutions "would have cost under $100 to train using on-demand cloud compute." Nine winners used Kaggle Notebooks, one Colab Free and one Colab Pro; six used personal hardware and three rented cloud servers. T4 and P100 still trained "several winning solutions" ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)).

The tail is heavy:

| Who | Compute |
|---|---|
| At least four 2025 winners | More than $500 each |
| Jeroen Cottaar (GWI 2nd) | About $700 |
| LLM 20 Questions winner | About $500 for a 455M-cell lookup table built on rented 8×4090 ([LLM 20Q 1st](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106)) |
| Eedi 2nd | Days of H100 per retriever: "GPU and Credit Card is all you need… Never open the billing page" ([Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)) |
| AIMO-2 winners (an all-NVIDIA team) | 512 H100s for 48 hours, estimated at over $60k ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)) |

NVIDIA's grandmasters compete "as part of our work responsibilities" ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)). A non-sponsored but serious budget is therefore roughly **$100–700 per competition**. Gold is attainable on far less when inference fits the free hardware.

### Inference on 2×T4: 8-bit, sorted, cached and time-guarded

Both of our code competitions rerun within 9 hours ([CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra); [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)). The T4 is a constrained target. Flash Attention 2 and torch.compile "do not work on T4 GPUs" and bfloat16 is unsupported, which pushed the AIMO-1 winners to 8-bit GPTQ calibrated on their training data ([HF: NuminaMath](https://huggingface.co/blog/winning-aimo-progress-prize)). vLLM needs compute capability 7.5 or higher ([vLLM install](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)), so it runs on T4 but not P100, and FP8 is unavailable on Turing ([vLLM quantization](https://docs.vllm.ai/en/latest/features/quantization/)). Several recent competitions have offered L4 or H100 in the evaluation environment ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)), so always check what the rerun offers.

The winning T4 recipe has six habits:

- **Fit the model in two T4s at fp16 or 8-bit.** "T4x2 is sufficient to run 7b-9b models in fp16" when layers are split across both GPUs ([LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)).
- **Merge fold LoRAs, then quantise to 8-bit GPTQ.** LMSYS 1st averaged its fold LoRAs first ([LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)). LMSYS 3rd wished it had known that "8bit was faster" ([LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)).
- **Calibrate quantisation at the inference sequence length.** In WSDM 3rd, accuracy was 0.681 unquantised; a lite calibration gave 0.6747 and a full-length calibration about 0.6809 ([WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)).
- **Sort by length and batch dynamically** (CLES 1st; [LMSYS 9th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704)).
- **Read label logprobs from a single prefill instead of generating text.** Eedi 2nd turned vLLM into a classifier this way ([Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)).
- **Spend expensive models only where they pay.** LMSYS 9th ran its third model only on the length buckets where it helped. Eedi 1st's cascade gave the 72B model only the top 5 candidates ([Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)).

Guard the clock as AIMO-2's winners did: a 350 s base budget per item, unused time pooled into a shared buffer, and early stopping when samples agree ([NemoSkills AIMO-2](https://arxiv.org/html/2504.16891)). Ship models and adapters as pinned Kaggle Datasets or Models, and install wheels offline. Our own most expensive lesson was environmental: re-running a public notebook without its pinned Docker image cost 0.014 ([[What Does Not Work]]).

### Cost-saving tactics for a 30-hour week

The tactics compound:

- **Keep CPU work in CPU sessions.** These include CASMI's RDKit enumeration, fingerprints and analogue generation; our generator sweep ran there and used no GPU quota ([[Competitions/CASMI26]]).
- **Cache every expensive intermediate as a versioned Dataset**, so each experiment only loads features and predictions ([H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)).
- **Develop on fold 0 with a smaller backbone or a shorter max length**, and promote only the winners.
- **Rent for training steps that need no Kaggle hardware**: bf16 LoRA fine-tunes, teacher soft labels, synthetic data. Keep the free quota for inference timing runs and final fits.
- **Distil big teachers into the model that fits the box.** Unsloth fits a 14B model into 16 GB ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)).
- **Run lightweight scripts for debugging and heavy models only for final runs** ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)).

For us, $20 a month in Colab links plus roughly $50–150 of rental across the season covers every training step we have planned. That is our estimate from the price tables above.

## Chapter 6. Read the write-ups first, then two books

Grandmasters agree on the best curriculum, and it is free. CPMP: "Reading top solution posted by Kagglers after each competition. Every time I can learn some new tricks", and "The few days after a competition end is when you will learn the most" ([NVIDIA GTC Q&A](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)). Ruchi Bhatia: "I studied top solutions, asked 'why did this work,' and rebuilt models from scratch", and "read discussion forums like research papers" ([NVIDIA blog](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)). Kaggle now publishes long-form solution posts as Writeups ([ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)).

The verdicts below are our judgements, based on each resource's scope and date.

| Resource | What it is | Cost | Verdict |
|---|---|---|---|
| Top-5 write-ups of the 2–3 most similar past competitions | Ablation tables and "didn't work" lists by winners | Free | **Essential, every competition.** For NTL read DAIGT, LMSYS and WSDM; for CASMI read Eedi, LECR, Otto, H&M and BELKA; for SHL read ELL, AES2 and CommonLit, the closest precedents for scoring language proficiency |
| [NVIDIA "Kaggle Grandmasters Playbook"](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) (Sept 2025; Onodera, Viel, Titericz, Deotte) | Seven techniques: fast experiments, train/test checks, diverse baselines, mass feature generation, hill climbing and stacking, pseudo-labels, extra training | Free, about 20 min | **Must-read.** Its GPU tactics also apply on Kaggle's T4s |
| [The Kaggle Book, 2nd ed.](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/) (Massaron, Tunguz, Banachewicz; Packt, 19 Dec 2025; 708 pp.) | Validation, metrics, ensembling, plus new chapters on GenAI/LLM competitions, Kaggle Models and time series | Paid | **Read.** The only current competition-specific book, written by three GMs. Too new for any GM endorsement yet |
| The Kaggle Workbook ([TUM record](https://search.ub.tum.de/vufind/Record/ZDB-30-ORH-089971450)) | Exercise companion to the first edition | Paid | **Optional**; skip if short of time |
| Approaching (Almost) Any ML Problem, Abhishek Thakur ([KDnuggets](https://www.kdnuggets.com/2021/02/approaching-almost-any-machine-learning-problem.html)) | Code-along book on project structure and CV, released under CC BY 4.0 | Free PDF | **Skim.** Its 2020 NLP and CV chapters predate the LLM era |
| [Hands-On ML with Scikit-Learn and PyTorch](https://oreilly.com/library/view/hands-on-machine-learning/9798341607972/) (Géron, Oct 2025) | PyTorch successor to the TF edition, covering transformers, diffusion, LLM fine-tuning and RL | Paid | **Best single reference textbook** |
| [Deep Learning with Python, 3rd ed.](https://deeplearningwithpython.io/) (Chollet & Watson, Sept 2025, 648 pp.) | Keras 3, PyTorch and JAX; building a GPT-like LLM | Free online | **Read the transformer chapters**; Keras-centric |
| [NLP with Transformers, rev. ed.](https://oreilly.com/library/view/natural-language-processing/9781098136789/) (2022) | HF Trainer, distillation, quantisation | Paid | **Dated**; for NTL, recent write-ups are more current |
| [fast.ai Practical Deep Learning](https://course.fast.ai/) | Part 1 (8 lessons) and Part 2 (lessons 9–25, Stable Diffusion from scratch) | Free | **Part 1 only if deep-learning intuition is missing**; pair it with HF or Unsloth docs |
| Effective XGBoost (Matt Harrison, ~2023; [Real Python ep. 169](https://realpython.com/podcasts/rpp/169/)) | Tabular tuning and interpretability | Paid | **Optional**; the Playbook covers what matters for Kaggle |
| Short methodology reads: [Ng's MLY](https://raw.githubusercontent.com/ajaymache/machine-learning-yearning/master/full%20book/machine-learning-yearning.pdf), [Karpathy's recipe](https://karpathy.github.io/2019/04/25/recipe/), [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml), [Caruana 2004](https://www.cs.cornell.edu/~alexn/papers/shotgun.icml04.revised.rev2.pdf), [Bouthillier et al.](https://arxiv.org/pdf/2103.03098v1) | Error analysis, training sanity checks, ranking errors, ensemble selection, ablation statistics | Free | **Read once, keep open**; together they are Chapter 3 |
| [ML Contests annual report](https://mlcontests.com/state-of-machine-learning-competitions-2025/) | What won in each year, on what hardware, at what cost | Free | **Read each spring.** In 2025 Qwen models won all three grand-prize-style Kaggle competitions and only one winner used a BERT-style encoder |

Our ranking for the next six months is: the similar-competition write-ups, the Playbook, the Kaggle Book 2nd edition, then Géron as a reference. The books teach vocabulary, but the write-ups are where the medals come from.

## Chapter 7. How each chapter applies to SHL, Name That LLM and CASMI

The three competitions sit at different points on the medal curve. SHL is a gold to defend (if it awards medals), Name That LLM is a silver within 0.004 of gold, and CASMI is a silver to hold. The order below follows the marginal value of the next medal.

| Chapter | SHL Hiring Assessment 2026 | Name That LLM | CASMI 2026 |
|---|---|---|---|
| 1. Rules and cutoffs | Our likely first gold *if* SHL awards medals (unverified) | Gold is top 12 (~0.560); silver margin only 12–15 places | Gold is top 16, 84 places up; silver is top 174 |
| 2. Choice and shake-ups | Shake-up risk sits in public-fitted blend weights | Merge only with an independent pipeline before 6 Jan | Domain niche; a merge before 7 Dec is the only realistic route to gold |
| 3. Workflow | Keep an honest pick; procedure beats composition | Rebuild CV as a test-assembly simulation | Funnel per class; apply the 0.002 forecast haircut against the ±0.004 floor |
| 4. Hacks | Dropout 0, seeds, full refit, self-distil, OOF scale calibration | Self-distil, family gates, Sinkhorn α hedge, 8-bit headroom | Source/rank features, dedup, novelty prior, ranking objective |
| 5. Compute | Encoder training on free quota | Rent LoRA training; 8-bit vLLM inference | CPU sessions for generation; GPU for the reranker |
| 6. Reading | ELL / AES2 / CLRP write-ups | DAIGT / LMSYS / WSDM | Eedi / LECR / Otto / H&M / BELKA |

### SHL Hiring Assessment: defend the gold with an honest pick

We are #2 on public at 0.3116 against #1 at 0.3064. The metric is 0.6·RMSE + 0.4·(1 − Pearson), lower is better. Our 0.3116 is a mix whose weights were fitted to public scores, including probe files, so it is public-tuned. The honestly validated main model scored 0.3178. The 0.0062 gap between the two measures what public tuning bought *on public*. The HPA, ICR and Home Credit cases in Chapter 4 all say that much of it will not survive the private split. The team count, the private-split size and the deadline are not recorded in this handbook, so the exact gold cutoff is unknown. On public we are inside gold at any field size, though. *If SHL awards medals*, **this is the single most important medal of the season**, because it is the one that makes us Masters. If it does not, the same picks protect the hiring result.

The final-pick rule follows directly from Chapter 3:

- **Pick A**: the best honestly validated model, or a blend whose weights were fitted only on out-of-fold predictions. It must not depend on public-fitted weights.
- **Pick B**: the public-tuned mix.

Best-of-two scoring makes B a free option, but only if A is honest. We should not pick two public-tuned variants. That would put the whole medal on the assumption that public and private agree, which the "metric hack" and "LB-tuned threshold" rows of Chapter 4 call a gamble.

Our own record says where the remaining effort should go. Procedure changes (crops, a bias fix, attention pooling on random crops) transferred to the leaderboard. Composition changes (more frozen encoders, kernel ridge, co-training, window TTA, LLM judges, bigger audio LLMs, label-grid rounding) did not. The procedure-type hacks that remain untested for us all come from the ELL, AES2 and CommonLit writing-proficiency competitions, the closest precedents for scoring language proficiency:

- dropout 0 in the regression backbone, rated "[High Impact]" ([ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609));
- comparing only 3-seed averages, then submitting full-data refits blended over seeds ([ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578); [FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536));
- self-distillation on our own OOF soft labels, which let a single model "achieve performance close to ensemble models" ([USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234));
- AWP from epoch 2 if an encoder is fine-tuned, ablated, because it was neutral on the small CommonLit and AES2 regression sets.

One more move follows from the metric's algebra rather than from a source. Pearson is scale-invariant, but the RMSE term (weight 0.6) pays for predictions that regress toward the mean. A two-parameter calibration `a·p + b`, fitted on OOF predictions to minimise the actual metric, extends the bias fix that already transferred. It is a procedure, not a public fit, and it should be validated with the 12-of-15 paired rule. This is our inference.

### Name That LLM: a 0.004 gap that procedure can close and rules cannot

We sit at rank about 53–56 on 0.556. Gold (top 12) is at about 0.560 and silver (top 68) at about 0.555, with 26 teams at 0.555 on 18 October ([[Competitions/Name That LLM]]). Our 0.556 is an exact pinned copy of the best public line. Gold therefore needs work beyond the public frontier, and the silver can be lost simply by the field taking one step up. The public notebook reports that its "predicted gains have matched our public-LB gains to within about 0.002-0.004 per version" ([LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering)). On that evidence, one or two genuine procedure wins would close the gap.

Three changes are already ruled out by our own leaderboard: per-prompt Hungarian assignment (−0.010), the punctuation veto (−0.003 to −0.005) and merging notebooks that share a base (0.550). A soft version of the per-prompt constraint rests on the same false premise and should not be revived.

Chapter 3 comes first. Rebuild CV as a test-assembly simulation with two or three passages per prompt and unequal per-model counts, because the hidden test has both properties and its class mix differs between public and private. Then test the procedure-type candidates, roughly in order of evidence:

1. **Self-distillation on our own prompt-grouped OOF soft labels.** WSDM 3rd gained 0.7051 → 0.7135 at T = 5 and soft weight 0.9, and a 14B model distilled on itself matched a 72B teacher, suggesting label cleaning between near-identical sibling models ([WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)).
2. **Family gates trained on extra passages whose prompts carry every family label.** OOF +0.013 for the GPT gate plus a second view; the v11 fork's Claude gate moved public 0.541 → 0.544. Without the extra passages the gate did not help ([LoRA x Adaptive Centering](https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering); [NTL v11](https://www.kaggle.com/code/liurouhan/name-that-llm-ntl-v11-sinkhorn-0-25)). Our research log also notes that about half of all errors stay within the same lab, which is where gates act.
3. **Sinkhorn α chosen on the simulation, not the LB, and hedged across the two picks.** Balanced accuracy is mean per-class recall, so balancing toward uniform is a prior correction ([[06 Calibration & Label Shift]]). With unequal, shifting test class counts, α = 1 forces wrong marginals.
4. **LoRA inference headroom.** Use r = 64 on all linear layers, merge the fold LoRAs, and quantise to 8-bit GPTQ calibrated on training passages at the inference length. Read 18 label logprobs from one prefill. The runtime this frees inside the 9-hour limit can carry a second backbone or selective TTA on the uncertain third of rows ([WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)).
5. **A train/inference length sweep on OOF.** DAIGT favoured training short and inferring long.
6. **A Caruana-style bagged, with-replacement hill climb** as a robustness check on the L-BFGS blend weights.

The DAIGT data tricks round out the list: typo and normalisation augmentation, and adding the most-wrong examples per class each round ([DAIGT 4th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179); [DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)).

On compute, LoRA iteration is the one place where renting clearly pays. Fifty RTX 4090 hours cost about $17, while QLoRA for a 9B model on 2×T4 manages only one fold-0 run per idea. A merge before **6 January** makes sense only with a team whose OOF predictions are uncorrelated with ours. One licence note from our own research: TabPFN weights are non-commercial, a risk for any prize-eligible pick ([[Competitions/Name That LLM]]).

Final picks follow the Chapter 3 default. **Pick A** is the seed-averaged ensemble with α chosen on the simulation. **Pick B** changes one stated assumption, for example the other α or the gate variant.

### CASMI 2026: hold silver cheaply, chase gold only through recall or a merge

We are at about rank 100 with a best public of 0.451. Silver runs to rank 174 and gold to rank 16, and the public leaderboard is about 124 molecules, so a single extra rank-1 hit is worth about 0.008 and run-to-run noise is about ±0.004 ([[Competitions/CASMI26]]). Most catalogued reranking hacks are worth +0.0005–0.002 each (ranking objective, graded gains, seed bags). Under our 0.002 forecast haircut, **none of them can be confirmed by a single CASMI submission**. They belong in paired CV and should ship only as bundles.

Chapter 3's error analysis has already shown where the points are:

- Class 3 (novel molecules) scored MRR 0.151 on the natural-product bench against 0.967 for class 1.
- Most class-3 losses are recall failures traced to analogue *parent selection*. Raising the generator from 6 to 20 parents added 23 novel answers (46 → 69 of 130) on the bench.
- On the leak-free unseen bench it did no harm to class 2 (0.256 → 0.260).
- Database coverage is saturated: only 485 of NPAtlas's 36,454 structures are new to our pool by the metric's identity.

The literature agrees on the order of work. Recall first, as in Eedi 1st's recall@32 priority. Then candidate-source flags with rank-within-source and an `n_sources_hit` count, which turned recall expansion from harmful into helpful in H&M 3rd. Then InChIKey-14 deduplication before ranking. Then a novelty prior tuned on a bench whose novel share matches the test, which was worth +0.036 private in Eedi 3rd ([Eedi 3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498)). Count features over the whole candidate universe, real test queries included, follow the Santander pattern. An adversarial-validation-pruned reranker is a natural second-pick hedge, the AMEX 11th pattern ([AMEX 11th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786)).

The team has already set the timetable: experimental submissions backed by the bench until about **17 November**, then sure-shot only, with picks locked before **14 December**. The current plan pairs the V22 copy with a different-lineage hedge, consistent with Chapter 3's portfolio logic.

Gold here would need about 84 places. That happens only through a structural recall gain on novel molecules or a merge with an independent, gold-adjacent pipeline before **7 December**. Absent either, the honest base case is a silver. That is still valuable: it is one of the two silvers Master needs, and CASMI is our best training ground for the domain-niche solo attempt Dieter's path recommends.

### Allocating 15 research hours and 30 GPU-hours across three competitions

Submissions are plentiful; hours are not. Rank work by marginal medal value. The SHL gold defence is worth Master if SHL awards medals; otherwise it is worth the hiring result, and NTL becomes the main route to Master. The NTL gold push is worth a second gold and also protects a thin silver. The CASMI silver hold protects the second silver and is mostly CPU work.

A defensible split of the weekly 15 research hours is about **6 for SHL, 6 for NTL and 3 for CASMI**. Shift CASMI's hours to NTL after CASMI's 17 November freeze. Split the ~30 Kaggle GPU-hours roughly **12 SHL / 12 NTL / 6 reserve**, with CASMI's generation and replays in CPU sessions. Add two Colab Pro links (about +30 GPU-hours a week for about $20 a month) and about $50–150 of rented 4090/A100 time for NTL LoRA training and any teacher labels. This allocation is our judgement, built from the medal arithmetic in Chapter 1 and the compute prices in Chapter 5.

## Conclusion

The evidence reframes what stands between us and Grandmaster. It is not model capability. It is *variance management*: knowing when a leaderboard gain is real. Our three competitions taught the same lesson three ways. NTL's Hungarian assignment won on a CV that reproduced the wrong test structure. SHL's blend reshuffles won on public and nowhere else. CASMI's single-hit granularity of about 0.008 dwarfs most published hacks. The cheap hacks that survive private leaderboards share one property: they change how the model learns, or they encode a constraint the test design guarantees. Our failures share the opposite property: each encoded an assumption we could not verify. That gives a usable filter for every future idea. Before building anything, ask which fact about the hidden test it relies on, and whether our CV reproduces that fact.

The second implication is about sequencing. Master is plausibly one honest final pick away, and Grandmaster then needs four more golds including a solo. That puts the median timeline around 2028–2029, and the fastest recent Kagglers did it in 10–15 months from Master. Two habits shorten the path. The first is building one reusable, cached, headless pipeline per problem family, so a late entry costs days instead of weeks. The second is spending a deliberate solo season in a domain where our insight compounds, as Dieter did with Bengali script and CPMP with post-processing. For us, MS/MS-style chemistry is the strongest candidate. Money cannot buy that edge, but at roughly $20 a month plus a few hundred dollars per season it buys every compute hour we need.
