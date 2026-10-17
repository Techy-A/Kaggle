# Path to Kaggle Competitions Grandmaster: progression rules, GM journeys, books/courses, compute (as of 2026-10-17)

Method note: kaggle.com pages are JavaScript-rendered and WebFetch could not resolve them, so the Kaggle facts below come from Kaggle's own page bundle (the Progression page's JS chunk) and Kaggle's public internal JSON API (`/api/i/...`). The endpoints used were `users.ProgressionService/GetCurrentUserProgression`, `users.RankingService/GetUserRankingsCounts`, `users.RankingService/GetUserRankingsV2`, `users.RankingService/GetUserRankingHistory`, `users.ProfileService/GetProfile`, `cms.LegacyCmsService/GetPage` (Kaggle docs), `discussions.DiscussionsService/GetForumTopicById` and `competitions.CompetitionService/GetCompetition`. All were queried on 2026-10-17. Each claim is cited to the human-readable kaggle.com page that shows the same data. GPU prices are live snapshots taken the same day.

## 1. Kaggle progression system in 2026: tiers, medals, cutoffs, rule changes

### Takeaway
Competitions GM still needs **5 gold medals including at least 1 solo gold**, and Master still needs **1 gold + 2 silver**. Kaggle explicitly left the Competitions track unchanged in its July 2025 overhaul. That overhaul **retired the Discussion track entirely**, **retired the Novice/Contributor tiers** (everyone below Expert is now just a "Kaggler"), made **only Expert+ upvotes count** for Code/Dataset medals, and added **Grandmaster Levels**: +1 level per extra 5 competition golds. Gold cutoffs depend on team count. For 250+ teams, gold goes to the top 10 + 0.2% of teams.

### Cited Findings
**Current tier requirements (live tier-requirement data behind the Progression page, 2026-10-17)**
- Competitions: **Expert = 2 bronze**; **Master = 1 gold + 2 silver**; **Grandmaster = 5 gold + 1 solo gold** — [Kaggle Progression](https://www.kaggle.com/progression), [Progression dashboard](https://www.kaggle.com/progression/dashboard)
- Datasets: Expert = 3 bronze; Master = 1 gold + 4 silver; Grandmaster = 5 gold + 5 silver — [Kaggle Progression](https://www.kaggle.com/progression)
- Code (Notebooks): Expert = 5 bronze; Master = 10 silver; Grandmaster = 15 gold — [Kaggle Progression](https://www.kaggle.com/progression)
- The old "Contributor" tier (run 1 notebook, make 1 competition submission, give 1 upvote) still exists in the API but is now displayed as "Kaggler" — [Kaggle Progression](https://www.kaggle.com/progression); [Kaggle Progression Update, 2025-07-07](https://www.kaggle.com/discussions/product-announcements/588704)
- The Progression page now describes **three tracks: Competitions, Datasets, and Code**. "Your overall tier is the highest of each of the three tracks." "Some Kagglers have 'Legacy' tier titles. This means the program or calculation under which they earned the title is no longer active." — [Kaggle Progression](https://www.kaggle.com/progression)
- The page also says: "Competition medals are harder to achieve than Code or Datasets medals, since they represent months of work and since they are limited to only the top participants in each competition. Consequently, fewer of them are needed to advance." — [Kaggle Progression](https://www.kaggle.com/progression)

**Competition medal cutoffs (exact table on the Progression page)**

| | 0–99 teams | 100–249 teams | 250–999 teams | 1000+ teams |
|---|---|---|---|---|
| Bronze | Top 40% | Top 40% | Top 100 | Top 10% |
| Silver | Top 20% | Top 20% | Top 50 | Top 5% |
| Gold | Top 10% | Top 10 | Top 10 + 0.2% | Top 10 + 0.2% |

- "(Top 10 + 0.2%) means that an extra gold medal will be awarded for every 500 additional teams in the competition. For example, a competition with 500 teams will award gold medals to the top 11 teams and a competition with 5000 teams will award gold medals to the top 20 teams." "Percentage calculations are rounded down. For example, a competition with 9 teams will not award any gold medals." — [Kaggle Progression](https://www.kaggle.com/progression)
- Which competitions award medals: "Each competition clearly states whether it awards Points and Medals on its overview tab. Typically Getting Started, Playground, and Community competitions will not award Points and Medals." — [Kaggle Progression](https://www.kaggle.com/progression). The Competitions docs say Getting Started competitions "offer no prizes or points", and Playground prizes "are typically kudos and public recognition". — [Kaggle Docs: Competitions](https://www.kaggle.com/docs/competitions)
- Competition points: "Competition points are awarded based on how well a team did in a competition, the number of members on the team, and the number of teams in the competition. The algorithm for competition points has not changed since the 13th of May 2015." Points decay over time, but medals and tiers do not. — [Kaggle Progression](https://www.kaggle.com/progression)
- Teams: "Everyone competes as a team (even if it's a team of one)." Merging is allowed until the Team Merger Deadline unless the combined team exceeds the size limit or the historical submission limit. "Submission limits (usually 5 per day) apply to the whole team." — [Kaggle Docs: Competitions](https://www.kaggle.com/docs/competitions)
- Solo gold rationale: the solo requirement exists so that you cannot reach GM simply by joining strong teams (secondary source summarising Kaggle) — [TDS: How to become a Kaggle Competitions Grandmaster](https://towardsdatascience.com/how-to-become-a-kaggle-competitions-grandmaster-9d77431c5b7d/)
- Code medals need 5 / 20 / 50 counted votes for bronze / silver / gold. Dataset medals need 5 / 10 / 25. Since 2025, forking (copy & edit) counts as an upvote. — [Kaggle Progression](https://www.kaggle.com/progression)

**Grandmaster Levels**
- "Once you have reached the Grandmaster Tier in one track you are said to be a 1x Grandmaster. Earning the Grandmaster Tier in the other two tracks allows you to become a 2x or 3x Grandmaster. In addition you can increase your Grandmaster level further by repeating the requirement for the Grandmaster tier in the Competitions track (no additional levels are granted for repeating the Datasets or Code tracks)." — [Kaggle Progression](https://www.kaggle.com/progression)
- At the July 2025 launch, Giba was top at "13x Grandmaster thanks to his mindboggling 64 gold competition medals". His profile shows 66 golds on 2026-10-17. — [Kaggle Progression Update](https://www.kaggle.com/discussions/product-announcements/588704); [Giba profile](https://www.kaggle.com/titericz)

**Rule changes 2024–2026**
- **July 2024:** Kaggle removed all medals from the Accomplishments forum. One user went from 61 discussion golds to 23 and lost Discussions GM. — [Kaggle discussion 522145](https://www.kaggle.com/discussions/general/522145), which links to [Removing Medals from the Accomplishments Forum](https://www.kaggle.com/discussions/product-feedback/522141)
- **Sept 10, 2024:** Badges added to Kaggle. — [Kaggle Progression Update](https://www.kaggle.com/discussions/product-announcements/588704)
- **Oct 2024:** Community Competition hosts may offer prize pools up to US$10,000, which was not possible before. — [ML Contests 2025 report](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- **2025-07-07 "Kaggle Progression Update" (Myles O'Neill, Kaggle):**
  - "We are not making any changes to the Competitions progression track."
  - Discussion track retired effective immediately. Existing Discussion Experts, Masters and GMs got "Discussions Legacy" awards, and anyone at least 80% of the way to the next tier got the next tier's legacy award.
  - Novice and Contributor tiers retired.
  - Forking now auto-upvotes, applied retroactively.
  - Only votes from Expert+ users count for Code and Dataset medals and rankings. Point decay is now based on the item's creation date. Medals are re-computed, but lost tiers are kept as "Legacy".
  - New Grandmaster page and GM Levels, city/organization ranking filters, progression dashboard, personal ranking history, Awards and Badges tabs, and an opt-out of progression.
  - Source: [Kaggle Progression Update](https://www.kaggle.com/discussions/product-announcements/588704)
- ML Contests called this "a significant overhaul" of the progression system in 2025. Kaggle also added Benchmarks, Game Arena and Writeups (long-form solution posts) in 2025. — [ML Contests 2025 report](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

**Medal math for the reader's two competitions (live team counts, 2026-10-17)**
- Enveda CASMI 2026: **3,495 teams**, US$50,000 prize, code competition (`onlyAllowKernelSubmissions`), CPU and GPU runtime limit 540 min (9 h) each, max team size 5, team-merger deadline 2026-12-07, final deadline 2026-12-14 — [Kaggle: Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
- Name That LLM: **1,363 teams**, US$50,000, host Kaggle, code competition, 540 min CPU/GPU limit, max team size 5, merger deadline 2027-01-06, final deadline 2027-01-14 — [Kaggle: Name That LLM](https://www.kaggle.com/competitions/name-that-llm)

### Inferences
- Applying the cutoff table to current counts:
  - CASMI (3,495 teams): gold = top 10 + floor(6.99) = **top 16**; silver = top 5% = **top 174**; bronze = **top 349**.
  - Name That LLM (1,363 teams): gold = 10 + floor(2.73) = **top 12**; silver = **top 68**; bronze = **top 136**.
  - Both counts will keep growing until entry closes, so cutoffs will drift upward a little.
- For the reader (Expert, competing as a team of two): **Master = 1 gold + 2 silver**. If both current silvers hold, the only missing piece for Master is one gold. Team medals count toward each member's tier. The "solo gold" requirement only matters for the GM step.
- Because Discussion medals are gone and Code/Dataset votes now need Expert+ voters, Competitions is the only track the reader can directly "earn" by performance. The 2025 overhaul also made Competitions GM relatively more prestigious.
- Grandmaster Level numbers on profiles do not reproduce exactly from "1 + floor((golds − 5)/5) + other GM tracks". For example, Psi shows level 4 with 38 golds, and Dieter shows 8 with 54. Kaggle may be counting only certain golds, such as golds after reaching GM, or golds from points-awarding competitions. Treat the formula as approximate.

### Gaps
- Kaggle's current pages do not say explicitly whether a team gold counts as a full gold for every member, or how "solo gold" is defined if a teammate leaves. Long-standing community understanding is that each member of a medal-winning team receives the medal and "solo" means a team of one at the deadline, but I found no 2026 primary text stating this.
- No primary text on whether Research or Featured competitions *always* award medals. The rule is per-competition, shown on each competition's overview tab.
- The exact GM-Level computation is not documented beyond the announcement text.

## 2. How long it takes; GM counts; named GM journeys and what worked

### Takeaway
As of 2026-10-17 there are **407 Competitions Grandmasters**, 2,366 Masters and 12,448 Experts among ranked users. Among the GMs currently in the competitions top 100, the median time from **Expert to GM is about 2.3 years**, and from **Master to GM about 1.9 years**. Time from account creation is longer (median about 5.3 years), but join date is a poor proxy for when serious competing started. Strong recent GMs did it in about 2.5–3 years from joining. Recurring advice: build reusable pipelines, join many diverse competitions, team up with people near your LB position, read every top solution afterwards, and pick a **niche/research competition with room for unique ideas** for the solo gold.

### Cited Findings
**Counts**
- Ranked-user counts by tier (2026-10-17):
  - Competitions: **GM 407**, Master 2,366, Expert 12,448, Kaggler 207,024.
  - Code: GM 101, Master 569, Expert 6,128.
  - Datasets: GM 72, Master 203, Expert 1,583.
  - Source: [Kaggle Rankings](https://www.kaggle.com/rankings), [Grandmasters page](https://www.kaggle.com/rankings/grandmasters)
- The Competitions ranking listed 15,240 ranked users on that date, and Chris Deotte's profile shows "rank 17 of 222,348" in Competitions. — [Kaggle Rankings: Competitions](https://www.kaggle.com/rankings/competitions); [Chris Deotte profile](https://www.kaggle.com/cdeotte)
- Historical points: Dieter wrote that when he got his solo gold (Bengali, early 2020) the title was "currently only 240 people hold world wide" ([Dieter's blog, 2022-11-20](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)). That implies roughly 240 to 407 Competition GMs from 2020 to 2026. A Wikipedia snapshot (April 2025) gives 612 Grandmasters across all categories (secondary source) — [Wikipedia: Kaggle](https://en.wikipedia.org/wiki/Kaggle)
- Kaggle ran 68 competitions in 2025, of which 34 were Community competitions. Across all platforms, the median competition had 36 teams and the mean 351. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

**Time-to-GM sample (own analysis of the tier-milestone dates in Kaggle's personal ranking history for the 58 Competition GMs currently in the top 100; 2026-10-17)**
- Restricted to the 56 who reached GM after the progression system launched on 2016-07-15: median join→GM **5.28 years** (IQR 3.47–7.55); median Expert→GM **2.26 years** (IQR 1.65–3.82, n=49); median Master→GM **1.85 years** (IQR 1.17–3.05, n=51). — [Kaggle Rankings](https://www.kaggle.com/rankings/competitions) (ranking-history panel)
- GM-achievement year for those 58: 2016 (4), 2018 (4), 2019 (3), 2020 (4), 2021 (5), 2022 (2), 2023 (6), 2024 (7), 2025 (8), 2026 to date (15). — same source
- Golds for recent GMs are usually just at the threshold (5–9). Examples: Jeroen Cottaar 5, tomoon33 5, James Day 6, Yurnero 8. — same source

**Named journeys (joined → Competitions Expert → Master → GM; golds/silvers/bronzes as of 2026-10-17; source: each Kaggle profile and its ranking history)**
- **Chris Deotte** ([profile](https://www.kaggle.com/cdeotte), NVIDIA): joined 2018-03-14 → Expert 2019-06-25 → Master 2019-10-04 → **GM 2020-06-17** (about 27 months from joining, 12 months from Expert). Medals 26/31/10. Before that he was Notebooks GM on 2019-06-06. His account: "With my background as a teacher, I quickly became a Discussions and Notebooks Grandmaster through sharing. Then I became more serious and earned Competitions and Datasets Grandmaster." His four tips: "robust local validation… accelerate your experimental pipeline with GPU (cuML/cuDF)… analyzing and addressing differences between train and test… understanding the competition metric." — [NVIDIA blog, 2025-04-29](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- **Christof Henkel "Dieter"** ([profile](https://www.kaggle.com/christofhenkel), NVIDIA; #2 in Competitions on 2026-10-17): joined 2017-11-17 → Expert 2018-06-29 → Master 2019-01-11 → **GM 2020-03-17**. Medals 54/18/4. He earned his golds largely in teams (6 golds within a year, including 6th of 6,000+ in Santander with "The Zoo"), then the solo gold: **5th of 2,000+ in Bengali.AI** on 3×GTX 1080 Ti at home. In his words:
  - "I joined competitions at a later stage, but at the same time participated in more competitions within a year."
  - On reusable baselines: "more and more parts of the work became standardized."
  - On the solo: "it's important to choose a suited competition when going for grandmaster… challenging enough to leave room for unique ideas"
  - His key trick was domain research: he relabelled training data after learning Bengali script rules.
  - Source: [Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)
- **Theo Viel** ([profile](https://www.kaggle.com/theoviel), NVIDIA): joined 2018-07-12 → Expert 2019-02-14 → Master 2019-06-28 → **GM 2020-09-17** (about 26 months). Medals 24/16/4.
- **Psi / Philipp Singer** ([profile](https://www.kaggle.com/philippsinger), Prior Labs): Expert 2019-04-16 → Master 2019-06-04 → **GM 2020-06-17** (about 14 months from Expert). Medals 38 gold / 13 silver.
- **CPMP / Jean-François Puget** ([profile](https://www.kaggle.com/cpmpml), NVIDIA): Expert 2016-08-31 → Master 2017-06-13 → **GM 2018-08-14**. Medals 29/37/5. He calls himself the "Post-processing Grandmaster": his 5th gold was a solo gold won "specifically using post-processing". His advice: study the metric (ranking vs. probabilistic), and "The few days after a competition end is when you will learn the most." — [NVIDIA GTC Q&A blog, Sep 2021 (older)](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)
- **Abhishek Thakur** ([profile](https://www.kaggle.com/abhishek)): "World's first quadruple grandmaster". Competitions GM was backfilled at the 2016-07-15 system launch. Medals 21/40/23.
- **Bojan Tunguz** ([profile](https://www.kaggle.com/tunguz), TabulAI): Master 2018-02-09 → **GM 2021-04-05**, so about 3 years stuck at Master. Medals 9/17/28. He was on the team that won Home Credit (then the largest Kaggle competition). — [NVIDIA blog](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)
- **hengck23** ([profile](https://www.kaggle.com/hengck23)): Master 2017-12-15 → **GM 2023-06-16**, 5.5 years at Master. This illustrates how hard the solo-gold or 5th-gold step can be.
- **Giba** ([profile](https://www.kaggle.com/titericz), NVIDIA RAPIDS): 66 golds, rank-1 peak; KazAnova ([profile](https://www.kaggle.com/kazanova), H2O.ai): 41 golds. Both had GM backfilled in 2016.
- **Recent fast GMs (2023+ joiners)**:
  - **Jeroen Cottaar** ([profile](https://www.kaggle.com/jeroencottaar)): joined 2023-05-07 → Expert 2025-02-06 → Master 2025-04-09 → **GM 2026-02-03**, currently #4. He took 2nd in Yale/UNC-CH Geophysical Waveform Inversion and said "my solution cost about $700 in cloud compute costs to develop and run". — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
  - **Yurnero** ([profile](https://www.kaggle.com/samson8)): joined 2023-03-03 → GM 2025-09-11.
  - **tomoon33** ([profile](https://www.kaggle.com/tomoon33)): joined 2023-05-22 → Expert 2024-09 → Master 2024-10 → GM 2026-01-24.
  - **James Day** ([profile](https://www.kaggle.com/jsday96)): joined 2019-07 → Expert 2022-08 → Master 2024-01 → GM 2026-01-24. He was 5th in LLM-Detect AI Generated Text, an authorship-style competition directly relevant to Name That LLM, with a home rig of 2×RTX 3090 + 1×RTX 4090. — [Kaggle writeup](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)
  - **sayoulala** ([profile](https://www.kaggle.com/sayoulala)): got his "first solo gold medal" by winning LMSYS Chatbot Arena (Aug 2024) → GM 2024-12-19. — [LMSYS 1st place writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
- **Teaming advice (NVIDIA GMs, GTC 2021)**:
  - "A good way for beginners to get started is to team up… Do not ask people on top of the leaderboard to team up with you without doing much. Try to ask people who are close to your leaderboard position."
  - "In every single team-up, I learned stuff."
  - Bo Liu: "If I do not have any interesting ideas, then I do not join."
  - Source: [NVIDIA blog (2021)](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)
- **More advice**:
  - Ruchi Bhatia: "simulating the public/private LB split using multiple CV folds to estimate potential leaderboard shakeups", and "read discussion forums like research papers".
  - David Austin: spend the first days visualizing data.
  - Source: [NVIDIA blog, 2025-04-29](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- In 2025, more than half of winning teams were one person, for the third year running. Over half were first-time winners. NVIDIA's KGMoN members make up more than 10% of the current top-50 Kaggle users. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

### Inferences
- A realistic plan for an Expert with two silvers: about 6–18 months to Master, then a median of about 2 years Master→GM. Top-100 GMs are a survivor-biased sample, so the typical path is likely slower.
- The pattern in Dieter's, Deotte's and Theo's journeys is: earn team golds first, build a reusable pipeline across many competitions, then pick a solo competition where domain insight gives an edge that teams cannot easily buy. For the reader, CASMI (MS/MS chemistry) is the archetype of such a domain-heavy competition.
- Recent GMs show the threshold can be met with exactly 5 golds in under 3 years of serious play. Gold count after GM is what separates the top 10.

### Gaps
- I could not page past the top 100 of the rankings API, so the time-to-GM statistics cover only the 58 GMs in the current top 100, not all 407.
- I found no published median "number of competitions entered before GM".
- The Chai Time Data Science and Kaggle "Grandmaster Series" interviews were not fetched in this pass.

## 3. Books and courses: what exists in 2026, and short verdicts

### Takeaway
The best Kaggle-specific texts are **The Kaggle Book 2nd ed. (Dec 2025, three GMs)** and **Abhishek Thakur's free AAAMLP (2020, dated but practical)**. For modern deep learning, use **Géron's new Scikit-Learn + PyTorch edition (Oct 2025)** and **Chollet's DLwP 3rd ed. (Sept 2025, free online)**. For short GM-written technique lists, the **NVIDIA "Kaggle Grandmasters Playbook" (Sept 2025)** blog is excellent. Grandmasters themselves overwhelmingly say the best "book" is reading top solution writeups after each competition.

### Cited Findings
- **The Kaggle Book, 2nd ed.** (Packt): authors Luca Massaron, Bojan Tunguz, Konrad Banachewicz (some listings add Anthony Goldbloom, who wrote the foreword); 708 pages; ISBN 9781835083208; published 2025-12-19. Subtitle: "Master data science competitions with machine learning, GenAI, and LLMs". New chapters on Kaggle Models, time series, and Generative AI competitions. — [O'Reilly listing](https://www.oreilly.com/library/view/the-kaggle-book/9781835083208/); [i-programmer](https://www.i-programmer.info/book-watch-archive/18594-the-kaggle-book-2nd-ed-packt.html); [ebookpoint listing](https://ebookpoint.pl/ksiazki/the-kaggle-book-master-data-science-competitions-with-machine-learning-genai-and-llms-second-ed-luca-massaron-bojan-tunguz-konrad-banachewicz-a,e_4hzq.htm)
- **The Kaggle Workbook** (Banachewicz & Massaron, Packt, self-learning exercises companion) — [TUM library record](https://search.ub.tum.de/vufind/Record/ZDB-30-ORH-089971450)
- **Approaching (Almost) Any Machine Learning Problem** (Abhishek Thakur): free PDF, CC BY 4.0. It is a code-along book on "how and what should you use", and does not explain algorithms. A reviewer notes there is no plan to update it. — [KDnuggets](https://www.kdnuggets.com/2021/02/approaching-almost-any-machine-learning-problem.html); [Goodreads](https://www.goodreads.com/book/show/54948686-approaching-almost-any-machine-learning-problem)
- **NVIDIA "The Kaggle Grandmasters Playbook: 7 Battle-Tested Modeling Techniques for Tabular Data"** (2025-09-18; Kazuki Onodera, Théo Viel, Gilberto Titericz, Chris Deotte). The techniques:
  - fast experimentation and careful validation
  - EDA with train-test distribution checks
  - diverse baselines (linear, GBDT, NN)
  - generating thousands of features with cuDF
  - hill climbing and stacking
  - pseudo-labelling
  - extra training (multi-seed, full-data retrain)
  - "The biggest lever we have in any competition… is the number of high-quality experiments we can run."
  - Source: [NVIDIA Technical Blog](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- **Hands-On Machine Learning with Scikit-Learn and PyTorch** (Géron, O'Reilly, Oct 2025; about 830–878 pages). This is the PyTorch successor to the TF-based 3rd edition. It covers transformers, diffusion, fine-tuning LLMs, and RL. — [O'Reilly](https://oreilly.com/library/view/hands-on-machine-learning/9798341607972/); [Ingram](https://ingramacademic.com/products/hands-on-machine-learning-with-scikit-learn-and-pytorch-9798341607989)
- **Deep Learning with Python, 3rd ed.** (Chollet & Watson, Manning, Sept 2025, 648 pages). Covers Keras 3, PyTorch, JAX, and GenAI, with chapters on transformers, building a GPT-like LLM, and diffusion. The authors' site says it "is available for free online". — [deeplearningwithpython.io](https://deeplearningwithpython.io/); [Manning](https://manning.com/books/deep-learning-with-python-third-edition)
- **Natural Language Processing with Transformers, Revised ed.** (Tunstall, von Werra, Wolf; O'Reilly, 2022, about 408 pages) — [O'Reilly](https://oreilly.com/library/view/natural-language-processing/9781098136789/)
- **fast.ai Practical Deep Learning for Coders**: free. Part 1 has 8 lessons (incl. NLP, random forests, collaborative filtering, CNNs). Part 2 runs lessons 9–25 (Stable Diffusion from scratch, mixed precision, attention/transformers). — [course.fast.ai](https://course.fast.ai/)
- **Effective XGBoost** (Matt Harrison): released around 2023 (inferred from 2023 podcast promotion; no primary listing found) — [Real Python podcast ep. 169](https://realpython.com/podcasts/rpp/169/)
- What grandmasters actually recommend:
  - CPMP: "Reading top solution posted by Kagglers after each competition. Every time I can learn some new tricks."
  - Also from the GTC Q&A: "read what the host writes… and read top solutions in similar recent competitions. Surprisingly, some competitions are won by models that are publicly shared from previous competitions and adapted… People do not read enough."
  - Source: [NVIDIA blog (2021)](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)
  - Ruchi Bhatia: "I studied top solutions, asked 'why did this work,' and rebuilt models from scratch." — [NVIDIA blog 2025](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- 2025 winners' toolkit:
  - Python almost universal; PyTorch dominant.
  - Qwen models won all three Kaggle grand-prize-style competitions.
  - LoRA common; vLLM used by four winners; Unsloth quoted for fitting 14B in 16 GB.
  - Only one winning team used a BERT-style encoder.
  - Source: [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

**Short verdicts (inference, based on the above)**
- The Kaggle Book 2e: **Read.** The only up-to-date, competition-specific book (validation, metrics, ensembling, plus GenAI/LLM competitions), written by three GMs.
- The Kaggle Workbook: **Optional.** Exercise companion to the 1st ed.; skip if time-poor.
- AAAMLP (Thakur): **Skim, free.** Great project structure and CV patterns, but its 2020 NLP/CV sections predate the LLM era.
- NVIDIA GM Playbook: **Must-read, 20 minutes.** The tabular checklist from four GMs, with GPU tactics (cuDF/cuML) that also apply on Kaggle's T4s.
- fast.ai: **Do Part 1 if you lack DL intuition; Part 2 for from-scratch understanding.** It is pre-LLM-fine-tuning era, though, so pair it with HF/Unsloth docs.
- Designing ML Systems (Huyen): **Low priority for medals** (production focus). See Gaps.
- Hands-On ML (Géron) PyTorch ed.: **Best single reference textbook** if you want one book.
- DLwP 3e: **Free online.** Good transformer/LLM chapters, though Keras-centric.
- NLP with Transformers: **Dated (2022)** but still useful on HF Trainer, distillation and quantization. For Name That LLM-style tasks, recent Kaggle writeups are more current.
- Effective XGBoost: **Optional.** Tabular tuning and interpretability; the Playbook covers what matters for Kaggle.
- Kaggle Days talks / YouTube: no primary source fetched (see Gaps). In practice the highest-value "course" is competition writeups, now published as Kaggle Writeups.

### Gaps
- No primary listing fetched for Designing Machine Learning Systems (Chip Huyen, O'Reilly, 2022 per common knowledge, unverified here), the Kaggle Workbook's publication date, or Effective XGBoost's exact date and pages.
- Kaggle Days talks and YouTube channels (e.g., Abhishek Thakur's, Kaggle's official channel) were not researched in this pass. No GM quote recommending specific channels was found.
- No direct GM quote recommending The Kaggle Book 2e (it is too new for interviews found).

## 4. Compute: Kaggle quotas, Colab, cloud rental prices, local GPUs, real LLM-competition budgets

### Takeaway
Kaggle's free tier in 2026 gives **30 GPU h/week (sometimes more) on P100 or 2×T4**, **20 TPU h/week**, 12 h sessions (9 h on TPU), 29–30 GB RAM, 4 CPU cores, and a 20 GB persistent `/kaggle/working`. **Linking a Colab Pro ($9.99/mo) or Pro+ account adds 15 or 30 extra Kaggle GPU h/week.** That is the cheapest compute upgrade available.

Rented GPUs on 2026-10-17:
- RTX 4090: about $0.34–0.64/h
- RTX 5090: about $0.69–0.94/h
- A100 80GB: about $0.79–1.79/h
- H100: about $2.0–4.3/h

Most 2025 winning solutions would have cost under $100 to train on demand. LLM-heavy winners used far more: Eedi 2nd place used days of H100 time, and AIMO-2 used 512 H100s for 48 h.

### Cited Findings
**Kaggle free tier (Kaggle docs, fetched 2026-10-17)**
- "12 hours execution time for CPU and GPU notebook sessions and 9 hours for TPU notebook sessions; 20 Gigabytes of auto-saved disk space (/kaggle/working); additional scratchpad disk space… not saved." — [Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks)
- Hardware specs per session:
  - CPU: 4 cores, 30 GB RAM
  - P100: 1×P100, 4 CPU, 29 GB RAM
  - T4×2: 2×T4, 4 CPU, 29 GB RAM
  - TPU 1VM: 96 CPU cores, 330 GB RAM
  - Submission reruns for code competitions "are always run on Intel Skylake CPUs".
  - Source: [Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks)
- GPU quota: "up to a quota limit per week… The quota resets weekly and is 30 hours or sometimes higher depending on demand and resources." Interactive sessions persist until a "60 minute idle timeout" (the Notebooks page says 20 min idle for the editor). Kaggle tips: avoid commits for checkpointing, cancel duplicate batch sessions, stop sessions before closing the tab, and use the Kaggle API to push versions without interactive sessions. — [Kaggle Docs: Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage); [Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks)
- TPU: "You can use up to 20 hours per week of TPUs and up to 9h at a time in a single session." TPU v3-8 has 8 cores and 128 GB of HBM. — [Kaggle Docs: TPU](https://www.kaggle.com/docs/tpu). Kaggle's current front-end code labels the accelerator "TPU v3-8 (Deprecated)" and also lists "TPU VM v3-8", "TPU v5e-8" and "TPU v6e-8" — [Kaggle site bundle](https://www.kaggle.com/code). It is unverified which of these are offered to free users today.
- The floating GPU quota (≥30 h, varying with demand) was introduced in 2020 and applies only to GPUs, not TPUs (older source) — [Kaggle product-feedback 173129](https://www.kaggle.com/product-feedback/173129)
- **Colab link bonus:** "Colab Pro and Pro+ users will get 15 and 30 hours of extra GPU hours per week respectively on Kaggle… These extra GPU hours don't consume your Colab compute units… same Kaggle hardware (CPUs, T4s, P100s, TPUv3-8)". It is marked experimental and can change. — [Kaggle Docs: Notebooks](https://www.kaggle.com/docs/notebooks)
- Code-competition limits in the reader's competitions: 540 min (9 h) CPU and GPU runtime per submission rerun, notebook-only submissions — [CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra); [Name That LLM](https://www.kaggle.com/competitions/name-that-llm). Some recent Kaggle competitions have allowed newer L4 or H100 GPUs in the evaluation environment. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

**Google Colab tiers (secondary sources; Google's pricing page is JS-only)**
- Pro is about $9.99/mo for 100 compute units. Pro+ is about $49.99/mo, with sources disagreeing on 500 vs. 600 units. Pay-as-you-go is $9.99 per 100 units. A T4 burns about 1.2–1.8 CU/h and an A100 about 15 CU/h. Specific GPUs are not guaranteed. — [McCormick ML: Colab GPUs Features & Pricing (2024)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/); [gpuperhour: Colab alternatives (2026)](https://gpuperhour.com/blog/google-colab-alternatives)

**Cloud GPU rental, live snapshot 2026-10-17 ($/GPU-hour, on-demand)**
- **RunPod** (Community / Secure):
  - RTX 3090 $0.22 / $0.50
  - RTX 4090 $0.34 / $0.89
  - RTX 5090 $0.69 / $1.19
  - L4 $0.44 / $0.59
  - L40S $0.79 / $1.09
  - A100 PCIe 80GB $1.19 / $1.79
  - A100 SXM $1.39 / $1.79
  - H100 PCIe $1.99 / $2.89
  - H100 SXM $2.69 / $3.99
  - H200 $3.59 / $5.29
  - RTX Pro 6000 $1.69 / $2.49
  - B200 $5.98 / $7.99
  - Source: [RunPod pricing](https://www.runpod.io/pricing)
- **Vast.ai** (marketplace; verified, single-GPU, on-demand offers; price is total $/h including storage, n in brackets):
  - RTX 3090: min $0.17, median $0.25 (32)
  - RTX 4090: min $0.43, median $0.64 (36)
  - RTX 5090: min $0.66, median $0.94 (33)
  - L4: median $0.36 (5)
  - A100 SXM4: median $0.79 (8)
  - A100 PCIe: median $0.74 (7)
  - H100 SXM: min $2.14, median $3.32 (4)
  - H100 PCIe: median $2.51 (3)
  - Source: [Vast.ai pricing](https://vast.ai/pricing) (via the public offers API)
- **Lambda** (on-demand, 1-GPU instances):
  - H100 SXM $4.29, H100 PCIe $3.29
  - A100 40GB $1.99, A10 $1.29, A6000 $1.09
  - GH200 $2.29, B200 $6.99
  - 8-GPU instances: H100 SXM $3.99/GPU, A100 80GB SXM $2.79/GPU
  - No consumer RTX cards.
  - Source: [Lambda pricing](https://lambda.ai/pricing)
- **GCP Spot VMs** (whole-VM $/h, default region shown on page; spot prices "can change up to once every day", "discounts of up to 91%"):
  - g2-standard-4 (1×L4) $0.42
  - a2-highgpu-1g (1×A100 40GB) $2.20
  - a2-ultragpu-1g (1×A100 80GB) $3.04
  - a3-highgpu-1g (1×H100) $6.60
  - a3-highgpu-8g $52.79 (≈$6.60/GPU)
  - Source: [GCP Spot VM pricing](https://cloud.google.com/spot-vms/pricing)
- AWS p5 (H100) spot is reported at about $2.96–3.69/GPU-h (secondary, Mar 2026) — [Hostinger cloud GPU pricing 2026](https://www.hostinger.com/tutorials/?p=158956); [Spheron GCP A3 analysis (May 2026)](https://www.spheron.network/blog/google-cloud-a3-h100-pricing/)
- Lightning AI and Paperspace prices could not be extracted (JS pages). See Gaps.

**Owning a GPU (secondary sources, mid/late 2026)**
- Used RTX 3090 is about $1,000–1,350 (eBay sold listings reported at $1,300–1,500 in June 2026). Used RTX 4090 is about $2,250–3,000 (about $2,527 30-day eBay average, latest sales near $3,010). Prices have been pushed up by local-AI demand and a "memory supercycle". — [InsiderLLM used 3090 guide](https://insiderllm.com/guides/used-rtx-3090-buying-guide/); [runaihome: used 4090 vs new 5090 (2026)](https://runaihome.com/blog/used-rtx-4090-vs-new-rtx-5090-2026/); [openclawdc: is the 4090 worth it (2026)](https://openclawdc.com/blog/is-the-rtx-4090-worth-it-local-llm/)

**What winners actually used**
- 2025 across all platforms:
  - Most winning solutions "would have cost under $100 to train using on-demand cloud compute".
  - Nine winners used Kaggle Notebooks, one Colab Free, one Colab Pro. Six used personal hardware and three rented cloud servers.
  - H100 became the most-used GPU among winners. RTX 5090 and RTX Pro 6000 appeared. T4 and P100 still trained "several winning solutions".
  - No winners trained on AMD or TPU.
  - At least four 2025 winners exceeded $500 of training compute, against a 2024 maximum of about $500.
  - Source: [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- **AIMO-1 (2024, Numina, 1st):**
  - Kaggle runtime was "either a P100 GPU or 2xT4 GPUs and up to 9 hours to solve the 50 problems".
  - Training: "On one node of 8 x H100 GPUs, our models took 10 hours to train."
  - T4 has no bf16 and Flash Attention 2 doesn't work, so they used AWQ/GPTQ quantization.
  - Source: [HF blog: How NuminaMath Won the 1st AIMO Progress Prize](https://huggingface.co/blog/winning-aimo-progress-prize)
- **AIMO-2 (2025, NVIDIA NemoSkills, 1st):** "training lasted 48 hours on 512 H100 (yes, 512!)", estimated at over $60k on demand. The second-largest 2025 budget (about $2.5k) was 32 H100s × 27 h plus 8 H100s × 24 h. Both teams were entirely NVIDIA employees. WSDM Cup and ARC Prize 2025 winners used four 8×H100 nodes. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- **LMSYS Chatbot Arena (2024):**
  - 1st place (sayoulala, solo) fine-tuned Llama-3-70B and Qwen2-72B (QLoRA) plus Gemma-2-9B (LoRA r=64), distilled 70B logits into 9B, averaged 5-fold LoRAs, then quantized to 8-bit GPTQ for inference. Hardware not stated. — [Kaggle writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
  - 2nd place (tascj): "it's possible to train 7B models using single A100 80G, 9B models requires two A100s… During the last 10 days… I used A100 80G x4"; "T4x2 is sufficient to run 7b-9b models in fp16". — [Kaggle writeup](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
- **Eedi Mining Misconceptions (Dec 2024):**
  - 2nd place: "GPU and Credit Card is all you need… Never open the billing page". Qwen2.5-14B data generation took about 2–3 days on H100; Qwen2.5-32B and QwQ about 5 days on H100; Qwen2.5-72B fine-tuning about 8 h on H100. — [Kaggle writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
  - 4th place: Qwen2.5-32B GPTQ-Int4, "each fold took 10 hours on an A100" (another config ran 10 h per fold on 4×A100). — [Kaggle writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559)
  - 1st place (Raja Biswas): Qwen2.5-14B retriever and ranker, with 72B models used to pseudo-label about 10k synthetic examples (hardware not stated). — [Kaggle writeup](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)
  - A public notebook showed Qwen2.5-72B and Llama-3.3-70B inference on 2×T4. — [Kaggle discussion 550223](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/550223)
- **LLM-Detect AI Generated Text (Jan 2024, 5th, James Day):** vLLM on 2×RTX 3090 + 1×RTX 4090 to generate AI-written text, 4-bit AWQ for models of 13B+, about 1.7M training examples. — [Kaggle writeup](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)
- **How GMs budget:**
  - NVIDIA KGMoN members compete as part of their job: "As part of our work responsibilities, we can compete on Kaggle." — [NVIDIA blog 2025](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
  - Non-sponsored winners report hundreds of dollars (Cottaar about $700). — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
  - Dieter got his solo gold on a home 3×1080 Ti PC. — [Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)
- **Cost-saving tactics GMs cite:**
  - GPU dataframes and ML (cuDF/cuML/GPU XGBoost) to multiply experiments per hour — [NVIDIA Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
  - "Use lightweight scripts for debugging, heavy models only for final runs. Save intermediate features and predictions."
  - Pruning, faster inference, distillation — [NVIDIA blog 2025](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
  - Distil big-model logits into a small model that fits the inference box (LMSYS 1st)
  - Unsloth to fit 14B in 16 GB — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)

### Inferences (practical plan for a 2-person team on Kaggle free tier, in CASMI and Name That LLM)
- **Baseline budget:** 2 accounts × 30 GPU h = **~60 T4×2/P100 hours per week**, plus 2 × 20 TPU h if the pipeline can use TPUs. One account per person is a Kaggle rule; never share or multiply accounts.
  - Each person adding Colab Pro ($9.99/mo) adds 15 Kaggle GPU h/week, giving **~90 h/week for about $20/mo total**. That is the best $/h available, about $0.05/GPU-h, though only on T4/P100.
- **Rent only for the heavy steps that don't need to run on Kaggle:** LLM fine-tuning in bf16, synthetic data generation, large teachers for distillation.
  - 50 h on a RunPod Community RTX 4090 is about **$17**, and 50 h on an RTX 5090 about $35.
  - 20 h on an A100 80GB is about $24–36. 20 h on an H100 PCIe is about $40.
  - A typical "serious but non-sponsored" competition budget is roughly **$100–700**, matching the reported winners.
- **Inference must fit 9 h on 2×T4 (or P100)** in both competitions. So:
  - train big, distil or quantize small (AWQ/GPTQ, fp16 not bf16, vLLM where it works on sm75)
  - cache embeddings and features as Kaggle Datasets
  - keep only the final ensemble for submission runs.
- **Buying vs. renting:** a used 3090 (~$1,000–1,350) costs as much as about 3,000–6,000 h on RunPod Community or Vast.ai 3090/4090. It only pays off with sustained use of more than about 20 h/week over a year or more, plus electricity. A used 4090 at about $2,500 is harder to justify at today's $0.34–0.64/h rental. The case for owning is convenience and uninterrupted experimentation, as with Dieter and James Day.
- **Spot instances** (GCP up to 91% off, Vast.ai interruptible offers) suit checkpointed training, not long uninterrupted generation jobs.

### Gaps
- Kaggle does not publish a 2026 "exact" weekly GPU quota beyond "30 hours or sometimes higher". The real number is shown in each user's account settings.
- Free-user availability of TPU v5e-8 / v6e-8 on Kaggle is unconfirmed.
- Official Colab prices could not be fetched (JS-only page), and the Pro+ unit count conflicts across sources (500 vs. 600).
- Lightning AI, Paperspace, and AWS on-demand prices were not extracted from primary pages.
- Eedi 1st place and LMSYS 1st place training hardware is not stated in their writeups.
- No primary source on the KGMoN team's DGX allocation.

## 5. Competition selection strategy for medals

### Takeaway
Pick competitions where the medal arithmetic and your edge line up. Under 1,000 teams, silver is a fixed **top 50** and bronze a fixed **top 100** (250–999 teams). Gold is only about the top 10–16 almost everywhere. Prefer research or domain-heavy competitions (CV, science, chemistry) that draw fewer teams and reward domain insight. Use multi-fold CV to estimate shake-up risk. Team up with people near your LB position, and save the solo attempt for a competition with "room for unique ideas".

### Cited Findings
- The medal table (Section 1) gives a fixed top 50 for silver and top 100 for bronze at 250–999 teams, and percentage cutoffs at 1000+ — [Kaggle Progression](https://www.kaggle.com/progression)
- "Tabular competitions are typically more popular… computer vision competitions often attract significantly fewer competitors, mostly due to large data sizes" (GM advice summarised by secondary source) — [TDS: How to become a Kaggle Competitions Grandmaster](https://towardsdatascience.com/how-to-become-a-kaggle-competitions-grandmaster-9d77431c5b7d/)
- Dieter's pattern: join competitions later but enter more per year with a reusable pipeline. For the solo gold, choose a competition that is "interesting enough to provide fun during weeks of working alone, but also… challenging enough to leave room for unique ideas", where domain research is the edge. — [Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)
- Team-ups: ask people "close to your leaderboard position", after doing real work yourself. — [NVIDIA blog (2021)](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)
- Teams also produce "a significant blending effect" even when members work independently and only combine models at the end. The cost is that a solo competitor must beat teams. — [Dieter's blog](https://christofhenkel.github.io/dieters-blog-v2/posts/2022-11-20-chapter-3.html)
- Shake-up control: "simulating the public/private LB split using multiple CV folds to estimate potential leaderboard shakeups." Deotte stresses robust local validation and analysing train/test differences. — [NVIDIA blog 2025](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- Read host posts and top solutions from similar past competitions, since some competitions are won by adapting prior public solutions. If you have no novel idea, consider skipping. — [NVIDIA blog (2021)](https://developer.nvidia.com/blog/competition-and-community-insights-from-nvidias-kaggle-grandmasters/)
- In 2025 more than half of winning teams were solo, and over half were first-time winners, so newcomers do win. — [ML Contests 2025](https://mlcontests.com/state-of-machine-learning-competitions-2025/)
- Code competitions run on hidden test data. In two-stage competitions, a common pitfall is notebooks not robust to the unseen stage-2 test set. — [Kaggle Docs: Competitions](https://www.kaggle.com/docs/competitions)
- Practical deadlines: team merges and new entries close one week before the end (e.g., 2026-12-07 for CASMI, 2027-01-06 for Name That LLM). — [CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra); [Name That LLM](https://www.kaggle.com/competitions/name-that-llm)

### Inferences
- **Medal arithmetic:**
  - In CASMI (about 3,500 teams), silver is the top ~174 but gold only the top ~16, so moving silver to gold is a large jump.
  - In Name That LLM (about 1,360 teams, still growing for 3 months), gold is the top ~12 and silver the top ~68.
  - For the Master goal (1 gold), the better gold odds are wherever the team has a structural edge: domain chemistry in CASMI, or authorship/LLM-fingerprint expertise in NTL, where 2024's LLM-Detect writeups are directly reusable.
- **Merging:** merging with a nearby silver/gold-zone team before the merger deadline is the standard way to convert silver into gold. Gold-medal teams of up to 5 all receive the gold. The GM path still needs one solo gold later.
- **Shake-up:** pick final submissions by CV, not public LB, especially when the public LB is a small fraction of test data. Pick one "CV-best" and one "robust/diverse ensemble" for the two allowed final submissions (CASMI allows 2 scored submissions).
- **Mid-tier sweet spot:** for future medal hunting, competitions with 250–999 teams give the friendliest silver/bronze cutoffs (fixed top 50/100). Niche science or research competitions in that range, ideally ones where large data or heavy domain knowledge deters casual entrants, are the best value. Late entry works if you have a reusable pipeline (Dieter's pattern), but reserve 3–4 weeks minimum for a medal attempt.

### Gaps
- No quantitative study found of shake-up magnitude by competition type, or of medal odds by entry timing.
- No primary statistics found on how many golds come from team merges versus solo efforts.
