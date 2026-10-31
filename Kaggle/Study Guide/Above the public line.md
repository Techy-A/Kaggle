---
tags: [kaggle, ntl, casmi26, research, leaderboard]
date: 2026-10-31
---
# Above the public line: who scores NTL 0.664+ and CASMI 0.48+, and how

Research only (31 Oct 2026, written 23:xx IST): fresh leaderboard downloads, our saved snapshots, both forums (NTL 88
topics, CASMI 140, all fetched with comments), public notebooks/datasets/models and tiers of every member of the top
~30 teams. Nothing built, pushed or submitted. Back to [[Name That LLM]], [[CASMI26]], [[Predictions vs Results]].

**Bottom line.** NTL: the teams above the public line are mostly *gradual climbers that have kept a +0.01 to +0.05
lead over the copy plateau since 18-19 Oct*, and the top 3 made **sudden leaps** (+0.09 to +0.16 in one window) that
line up with private external data (Hod Carriers says so) and, for the #1, with running an open-weight *candidate*
model (Kimi K3) as a teacher. Paid data is the field's main lever and is out of reach for us; the free part of it
(bigger cleaned public pools, open-weight GLM/Kimi models) is reachable but expensive. CASMI: only **10 teams** are
clearly above the line (>= 0.485); the other 28 above 0.479 are within one molecule of it. The top 5 have had
**independent pipelines since day one** (0.49 on 16 Oct when the public line was 0.446) and climb slowly with 90-200
submissions on a ~125-molecule public board; nothing they did is public. Correction found: the cross-test-spectrum
"yes" in 752297 was posted as **HOST**, so that lever is allowed.

## Method and caveats
- Fresh downloads: 31 Oct 22:54 IST (NTL 4,082 teams, CASMI 4,460). History: our saved snapshots, NTL 17 (16 Oct
  22:22 to 31 Oct 22:54 IST), CASMI 12 (16 Oct 22:22 to 31 Oct 22:54 IST). **Gaps:** NTL 20-24 Oct and 24 Oct
  15:46 to 31 Oct 03:34 (one extra point: the 30 Oct 01:37 IST scan in [[Name That LLM]]); CASMI 19-23 Oct and 24-31
  Oct. So "leap" means "a large gain inside one window", not a timestamped single submission.
- "Plateau" = the most common score among the top 15% (the copy plateau). "Excess" = team score minus plateau at that
  snapshot. One CASMI public molecule is 0.004 (rank 2 to 1) to 0.008 (miss to rank 1); NTL identical code re-scores
  within about +-0.005.
- Tiers are Kaggle overall performance tiers. lianmatsuo's GitHub repo (github.com/lianmatsuo/name-that-llm,
  `experiments/teacher_runner`) could not be read: github.com is blocked here (403) and attaching it was refused.
- Working files (scratch, not in the repo): `/tmp/claude-0/-home-user/b6319532-f3ee-57bf-b6c7-f5215a48473a/scratchpad/above/`
  (`lbfresh_*`, `forum_*.json`, `users_*.json`, `scripts/`).

## 1. Name That LLM: who is above 0.664

**Counts (31 Oct 22:54 IST).** Above 0.664: **67**; at 0.664: 57; above our 0.662: 197; at 0.662: 237 (the largest
plateau). Bands: 0.665-0.674: 41 (within rerun noise of the public line); **>= 0.675: 26; >= 0.69: 12; >= 0.70: 10**.
Medal lines: gold (rank 18) **0.683**, silver (rank 204) 0.662, bronze (rank 408) 0.662. Us: 0.662, rank 228 (below
silver on tie order). Median submissions: 35 above the line, 46 in the top 25, 12 at the 0.662 plateau. Team size:
1.15 on average above the line; 4 multi-member teams in the top 25, all merged between 24 and 31 Oct.
Tiers of the 37 members of the top 30 teams: 3 Grandmasters (analokamus, yujiariyasu, both Team Aillis;
verracodeguacas), 3 Masters (horikitasaku, godhand, sorenravn), 14 Experts, 17 Contributors.

**Plateau over time** (copy plateau / top score): 16 Oct 22:22 0.541 / 0.584; 18 Oct 23:34 0.556 / 0.650;
19 Oct 23:58 0.569 (0.592-0.597 best public) / 0.650; 24 Oct 15:46 0.636 / 0.700; 31 Oct 22:54 0.662 / 0.736.
Teams above the plateau: 11, 146, 181, 75, 197.

**Reading the histories.**
- **Sudden leaps (private assets):** Lian Matsuo (0.634 at the plateau on 24 Oct, then 0.725 by 30 Oct 01:37 IST and
  0.736 by 31 Oct, 22 submissions in all), Hod Carriers (0.583 to 0.700 between 19 and 24 Oct; in 749447 he wrote
  "My best LB score before adding external data was 0.583"), Bootstrapping (0.526 to 0.688 in the same window, +0.036
  inside 3.5 h on 24 Oct), kulshekhar and Harveen (+0.08 each, 17-18 Oct, as the first external-data datasets appeared).
  Every big leap falls in a window where paid or pooled external data entered the field (17-24 Oct) or, for Lian,
  where a new open-weight teacher was uploaded.
- **Gradual climbers (iteration):** about 15 of the top 25 (Le Viet, Søren Ravn, Team Aillis, Kimiの名は, Vishak,
  Meni, Xerneas, Pirhosseinlou, Maniceet, ...) kept a steady +0.01 to +0.05 over the plateau since 18-19 Oct and
  move with it: they track the public chain and add their own data/members on top.
- **Leads erode:** Adisak Sukul was #1 at 0.650 on 19 Oct (+0.08 over the plateau) and is 0.666 (rank 51) now;
  Søren Ravn's lead fell from +0.053 to +0.029. Whatever an early private edge was, the public chain caught it.
- **Merges:** Team Aillis = Horikita Saku (solo 0.664 on 24 Oct) + Sho Okiyama's trio (0.642) + yujiariyasu, merged
  by 31 Oct (0.704). A merge shows up as max-of-members plus later gains, not as a method change.

**What the top teams' public footprint reveals** (all 37 members checked: competition notebooks, datasets, models,
forum posts):
- **lianmatsuo (#1, Contributor, UK, joined 2024):** four public datasets `kimik3-iq2xxs-slices-1..4`, uploaded
  30 Oct 01:44-02:38 IST: unsloth's Kimi K3 UD-IQ2_XXS GGUF (revision a0836360...) cut into layer-range slices (0-23,
  ..., 70-92 of 93 layers; about 130-180 GB each) "for llama.cpp layer-range runs", made by `slicer.py` from his repo's
  `experiments/teacher_runner`. Kimi K3 is **one of the 18 labels**, so this runs the true generator (2.78T MoE at
  2 bits) as a *teacher*: either per-token likelihoods under Kimi (a generator-likelihood feature) or distillation
  targets. The slices exist to fit one layer range per pass, i.e. push all passages through 23 layers at a time.
  Licence: the included Kimi K3 License is MIT-style ("without restriction"). No notebook is public. Timing: his
  0.725 predates the upload by minutes, so the teacher may be part of the +0.011 to 0.736 or of earlier private runs;
  not provable from the board.
- **tononnh (Hod Carriers, #2, Expert):** no public NTL assets; the only method statement is the external-data quote
  above. He also reported slow /kaggle/input copies (big model inputs).
- **wujianming (Bootstrapping, #3, Expert):** only old Kaggle Models (Qwen2.5-14B/32B-Instruct); no NTL assets. The
  team name suggests self-training/pseudo-labelling, unconfirmed.
- **Team Aillis (#8):** 2 Grandmasters + 1 Master + 2 Experts; no NTL assets.
- **verracodeguacas (Wilmer Henao, #11, Grandmaster):** two early public notebooks (Prism Signatures + EDA, 0.547)
  and a feature-atlas dataset, nothing recent for NTL.
- **nishantdahal (#22):** the public two-stage LoRA lineage (0.592, 0.620) and the pool datasets we already use.
- **jusill (muDILA, #19):** the Sinkhorn-on-uneven-test study (752365), already in our notes.
- **menidavidgottesman (#15):** asks (752198) about running ~30B MoE base models in 4 bit on 2xT4 within 9 h, and
  (752199) how to validate once external data and test-time grouping are in: bigger open models are being tried.
- **maniceet (#25):** asked on 15 Oct whether paid API generation is allowed; his OpenRouter GLM/GPT-5.6 passages
  are credited in 752810. **godhand (#10, Master)** on "where does the gap between 0.66 and 0.70+ come from?" (754463,
  31 Oct): "both important" (data and modelling).
- Nobody in the top 30 has a public generated-data dataset. Since the staff answer of **29 Oct (752084)**,
  self-generated data within a reasonable cost (token spend under about the $5k prize) **need not be published**,
  public community datasets (including Benchmarks-proxy ones) are fine to train on, and public LoRA weights are fine.
  So private paid-API data is legal for them and will stay invisible until write-ups.
- Field numbers on the data lever: external data worth about +0.11 to +0.12 (749447, James Warner); pool 36k to 124k
  passages took one recipe from LB 0.586 to 0.617 (752810, danielandresmiranda); our own SVC test: cleaned Open Pool
  v9 vs v5 +0.014 at argmax, +0.002 after Sinkhorn.
- **Open-weight labels:** kimi-k3, glm-5.3 and glm-5.3-flash have open weights on Kaggle (Kimi K3 MXFP4 model
  mirror; GLM-5.3-Flash UD-IQ3_XXS GGUF, rahim3, 2 parts, 120 GB; GLM-5.3 UD-IQ2_M GGUF shards). These three are
  among our weakest classes (18 Oct lab recall: glm-5.3-flash 0.18, kimi-k3 0.20, glm-5.3 0.23) and GLM<->Claude is
  the biggest cross-lab confusion. A generator-likelihood signal targets exactly that, at zero token cost.


## 2. CASMI26: who is above 0.479

**Counts (31 Oct 22:54 IST).** Above 0.479: **38**; at 0.479: 91 (us among them, rank 73); plateau 0.477: 458 teams.
Bands: **0.480-0.484: 28** (within one molecule of the line); **>= 0.485: 10; >= 0.49: 8**. Medal lines: gold
(rank 18) **0.483**, silver (rank 223) 0.477, bronze (rank 446) 0.477. Median submissions: **74.5** above the line vs
25 at the plateau; the top 4 have 89-197. Team size 1.11 on average; 3 multi-member teams in the top 32. Tiers of the
36 members of the top 32 teams: 0 Grandmasters, 6 Masters (CHU1PC's lowasitgoes, JJ's three, heliosli, linshokaku),
4 Experts, 26 Contributors; the top 4 teams are Contributors/Expert (domain skill, not Kaggle rank).


**Plateau over time** (copy plateau / top): 16 Oct 0.446 / 0.495; 19 Oct 0.451 / 0.495; 23 Oct 0.464 / 0.504;
31 Oct 22:54 0.477 / 0.521. Teams above the plateau: 172, 268, 64, 178.

**Reading the histories.**
- **Independent lineages from day one:** on 16 Oct, when the public line was 0.446-0.451, pikachu was 0.495,
  Parfionov 0.493, CHU1PC 0.491 and MarvinTMB 0.484 (reached by 9 Oct with 16 submissions; inactive since). Since
  then they climbed **slowly** (+0.000 to +0.026 in 15 days) while the plateau rose +0.031, so their lead shrank from
  about +0.045 to +0.02-0.04. That is a different, stronger base pipeline, not a late trick.
- **Steps:** Savko (at the plateau 0.452 on 19 Oct, 0.490 by 23 Oct, 0.517 and 0.521 by 31 Oct; 197 submissions),
  Shehab (0.470 to 0.501 between 24 and 31 Oct; 186 submissions). Each step is 3-5 molecules; with this many
  submissions on ~125 public molecules (SE about 0.04), part of it is public-board selection, and the private split
  (~275 molecules) can reorder them.
- **Line + private add-on:** heliosli (team "Day: chem. Night: Say my name.", Master) is 0.486 = public 0.479 plus
  his private rr3/rr4 reranker: +0.007, about one molecule. His datasets are still private on 31 Oct; he published the
  0.479 notebook (83 votes) without them.
- **Ranks 11-38 (0.480-0.484):** forks of the 0.477/0.479 line plus a tweak or a lucky molecule.

**Public footprint of the top teams.** None of the 36 members has a public CASMI notebook, dataset or model except
heliosli (his 0.479 notebook). Forum posts by top members carry only hints:
- MarvinTMB (#12, the early leader): "You do not need NIST or any additional data for a better score"; estimates a
  realistic maximum around 0.55 (746302, 9 Oct).
- Tra Crew (#20): a calibrated mock scorer from held-out train predicts the last ~15 submissions; one class-3 lever
  worth +0.003-0.004, a possible +0.007-0.010 one not shown (751300, 751356).
- Shehab (#4): nothing on method.
- Nobody in the top 32 declares external spectra; the one public declaration (Zenodo GNPS + MassSpecGym harmonised
  spectra, CC BY 4.0, 754574) comes from a team at 0.343.

**Rules and data facts that change the lever list.**
- **Cross-test-spectrum processing is allowed.** In 752297, Marie Killian's reply ("Yes, automated, unsupervised
  processing across test spectra at inference time is allowed", 26 Oct 20:53 IST) is posted with `authorType: HOST`
  (the forum JSON marks her, davidhealeyenveda and inversion as HOST/ADMIN). Our notes called it a participant answer;
  that was wrong.
- **The enveda-180 rows are the visible placeholder, not the hidden test.** Staff (inversion, 745715, 8 Oct): no ID
  overlap between train and test, or between the dummy test and the rerun test; the visible test.parquet is a
  placeholder built from training spectra. So "every visible test spectrum is verbatim in train" is expected and says
  nothing about the hidden 400 molecules. It is not a lever for anyone; keep it on the stop list.
- **Private reranker datasets** (heliosli rr3/rr4): +0.007 publicly; may be released (P about 0.3).


## 3. Conclusions: what separates them, what we can reach

### Name That LLM (public line 0.664, ours 0.662, gold 0.683)
**What most plausibly separates 0.664+:**
1. **Private in-style external data** (paid API generations from the 18 models, legal up to about $5k and unpublished
   since 29 Oct). Strongest evidence: Hod Carriers' own statement, the leap windows (18-24 Oct), +0.11-0.12 field
   estimate, 36k to 124k pool = +0.031 LB. Explains most of the 0.70+ group.
2. **Open-weight candidate models as teachers/features** (Kimi K3 for #1; GLM-5.3/-Flash are also open). Narrow
   (3 of 18 labels) but aimed at the weakest classes.
3. **Iteration and merges on top of the public chain** (top-25 median 46 submissions; 4 merged teams): the steady
   +0.01 to +0.03 of the 0.675-0.69 band.
4. Bigger backbones (27-30B MoE/dense, two T4s) and the 0.665-0.674 band's knob tuning, which is mostly rerun noise
   (+-0.005).

**Reachable for us (free compute, no paid API, commercial/host-allowed licences):**
| Lever | Reachable? | Cost | Honest expected public gain |
|---|---|---|---|
| Paid in-style generations | **No** (constraint) | - | (+0.03 to +0.07 for those who did it) |
| Retrain our two-stage LoRA (Qwen3-8B, 24 layers) with stage 1 on the large cleaned public pool (Open Pool v10 198k, whose 69k Benchmarks-proxy rows staff allow since 29 Oct, + gaggle's ~13k clean extra + enaimaan's 9.7k OpenAI passages, CC BY-SA/CC BY), added beside nishant's 8B (keep the old member; 753177 lost 0.004 by swapping) | **Yes** | about 13 GPU h (scaled from our OLMo-13B stage 1: 6.6 h for 66k rows on 2xT4) | +0.000 to +0.006; P(beats 0.664 honestly) about 0.2 |
| GLM-5.3-Flash / Kimi K3 likelihood features | Technically yes (weights public, MIT-style Kimi licence; GLM licence to check) | days of engineering; 120 GB to ~0.7 TB of weights; hidden-run time risk on 9 h | unknown; P(working in time) about 0.1 |
| Public-notebook copies | Yes | low | holds the line only (0.662-0.664) |

**Odds.** Gold (0.683, about +0.02): **about 5%**. Beating 0.664 with an honestly validated own change: **about 20%**.
**Single most promising next step:** the pool-retrained LoRA member as an *additional* member: train stage 1 on the
cleaned 200k+ pool and stage 2 on train with 5 prompt-grouped folds, then compare OOF of the blend with vs without it
(paired bootstrap over prompts, P(better) >= 0.9; remember pool-trained OOF reads 0.03-0.04 high, so judge only the
delta). It is a training change (eligible under our rules) and the only lever with a measured LB effect (+0.031 on
752810's recipe).

### CASMI26 (public line 0.479, ours 0.479, gold 0.483)
**What most plausibly separates 0.48+:**
1. **A different, stronger base pipeline** built early (top 5: +0.04 over the public line on 16 Oct). Not visible.
2. **Heavy public-board iteration** (89-197 submissions at the top) on 125 molecules: part of the excess is
   selection noise and may not survive the private split.
3. **Private add-ons** on the public line (heliosli's reranker, +1 molecule).
4. Possibly cross-test-spectrum methods (now host-allowed); nobody says who uses them.

**Reachable for us:**
| Lever | Reachable? | Expected public gain |
|---|---|---|
| Build a top-5-class base pipeline | Not in 6 weeks; our own stack is 0.465 | - |
| Copy heliosli if his datasets go public | Yes, 0.45 GPU h | +0.000 to +0.007 (P release about 0.3) |
| Cross-test-spectrum consistency (allowed): within the 400 test molecules, pair spectra by modified cosine and a known mass shift; when one partner has a confident rank 1 (library cosine >= 0.87), add a small rank-preserving term for candidates of the other partner that share its scaffold/differ by that transformation | Yes, CPU first | +0.000 to +0.008 (0-1 molecule); P(>= 1 molecule) about 0.1-0.15 |
| More submissions on the public board | Yes but not honest | noise |

**Odds.** Gold (0.483, +1 molecule net): **about 10%**; >= 0.486 honestly: **about 5%**.
**Single most promising next step:** a CPU pre-check of the cross-test-spectrum lever on the enveda-np-examples bench
(the closest proxy to the hidden natural products): count how many bench molecules have a spectral partner inside the
set (modified cosine >= 0.7 with a precursor delta in the analog rulebook), and on those pairs replay a rank-1-
protected boost with the partner's answer known (oracle) and with the partner's own rank 1 (realistic). Go to a
Kaggle run only if the realistic replay gains with zero class-1 rank-1 losses and P(better) >= 0.9. If partners are
rare (< 5% of molecules), drop it.

## 4. Top-25 tables
Tiers: GM Grandmaster, M Master, E Expert, C Contributor. Excess = score minus the copy plateau at that snapshot.

### NTL top 25 (31 Oct 22:54 IST)
| # | Team | Score | n | Tiers | Subs | Last sub (IST) | Score path (IST) | Excess over plateau 19/24/31 Oct | Pattern |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Lian Matsuo | 0.736 | 1 | C | 22 | 31/10 08:38 | 24/10 0.634 → 31/10 0.736 | - / -0.002 / +0.074 | **leap**: at the plateau on 24 Oct, 0.725 by 30 Oct 01:37 IST, 0.736 by 31 Oct; Kimi K3 slices uploaded 30 Oct 01:44 IST |
| 2 | Hod Carriers | 0.732 | 1 | E | 47 | 30/10 23:31 | 16/10 0.530 → 17/10 0.561 → 18/10 0.564 → 19/10 0.583 → 24/10 0.700 → 31/10 0.732 | +0.014 / +0.064 / +0.070 | small lead, then **leap** 19→24 Oct (+0.117); says "0.583 before adding external data" |
| 3 | Bootstrapping | 0.714 | 1 | E | 30 | 31/10 14:45 | 16/10 0.466 → 17/10 0.503 → … → 24/10 0.688 → 31/10 0.711 → 31/10 0.714 | -0.043 / +0.052 / +0.052 | below plateau until 19 Oct, **leap** 19→24 Oct (0.526→0.688, +0.036 within 3.5 h on 24 Oct) |
| 4 | kulshekhar | 0.714 | 1 | C | 38 | 30/10 05:36 | 17/10 0.526 → 18/10 0.605 → 18/10 0.607 → 19/10 0.630 → 24/10 0.662 → 31/10 0.714 | +0.061 / +0.026 / +0.052 | **early leap** 17→18 Oct (+0.079), then steps |
| 5 | Le Viet | 0.711 | 1 | E | 52 | 30/10 12:32 | 16/10 0.515 → 17/10 0.554 → … → 24/10 0.666 → 24/10 0.672 → 31/10 0.711 | +0.046 / +0.036 / +0.049 | gradual (10 steps), lead +0.04 to +0.05 all along |
| 6 | Harveen Singh Chadha | 0.706 | 1 | E | 33 | 31/10 02:52 | 16/10 0.513 → 18/10 0.592 → 18/10 0.595 → 19/10 0.596 → 19/10 0.644 → 31/10 0.706 | +0.075 / +0.008 / +0.044 | two leaps (18 Oct, 19 Oct), flat to 24 Oct, +0.062 since |
| 7 | t.shiga | 0.705 | 1 | E | 23 | 31/10 07:05 | 24/10 0.666 → 31/10 0.705 | - / +0.030 / +0.043 | first seen 24 Oct at +0.030, steps |
| 8 | Team Aillis | 0.704 | 5 | GM/M/E/E/GM | 70 | 31/10 12:41 | 16/10 0.508 → 17/10 0.518 → … → 19/10 0.608 → 24/10 0.664 → 31/10 0.704 | +0.039 / +0.028 / +0.042 | gradual + 5-way merge (Horikita solo 0.664 on 24 Oct) |
| 9 | Kimiの名は | 0.704 | 2 | C/E | 76 | 31/10 07:11 | 16/10 0.487 → 17/10 0.531 → … → 24/10 0.653 → 31/10 0.693 → 31/10 0.704 | +0.021 / +0.017 / +0.042 | gradual + 2-way merge |
| 10 | Civitasmass | 0.701 | 1 | M | 49 | 31/10 08:30 | 16/10 0.373 → 17/10 0.461 → 19/10 0.470 → 24/10 0.628 → 31/10 0.670 → 31/10 0.701 | -0.099 / -0.008 / +0.039 | catch-up to plateau by 24 Oct, then +0.031 today |
| 11 | Wilmer E. Henao | 0.698 | 1 | GM | 52 | 31/10 06:25 | 16/10 0.426 → 17/10 0.546 → 18/10 0.547 → 24/10 0.655 → 31/10 0.692 → 31/10 0.698 | -0.022 / +0.019 / +0.036 | catch-up, then gradual |
| 12 | Søren Ravn Andersen | 0.691 | 1 | M | 40 | 31/10 11:18 | 16/10 0.569 → 17/10 0.585 → … → 24/10 0.673 → 31/10 0.674 → 31/10 0.691 | +0.053 / +0.037 / +0.029 | lead from day 1 (0.569 on 16 Oct), gradual, lead shrinking |
| 13 | Santos x Cantone | 0.688 | 2 | C/C | 44 | 31/10 16:55 | 24/10 0.468 → 31/10 0.682 → 31/10 0.688 | - / -0.168 / +0.026 | merge of two sub-plateau teams, 0.468→0.682 |
| 14 | Vishak K Bhat | 0.688 | 1 | E | 42 | 31/10 20:56 | 16/10 0.519 → 17/10 0.528 → 18/10 0.552 → 19/10 0.586 → 24/10 0.673 → 31/10 0.688 | +0.017 / +0.037 / +0.026 | gradual |
| 15 | Meni David Gottesman | 0.687 | 1 | C | 53 | 31/10 09:06 | 17/10 0.549 → 18/10 0.568 → … → 24/10 0.659 → 31/10 0.679 → 31/10 0.687 | +0.040 / +0.023 / +0.025 | gradual |
| 16 | XiaoXiaoJ | 0.685 | 1 | C | 51 | 31/10 05:42 | 16/10 0.456 → 19/10 0.571 → 19/10 0.579 → 24/10 0.638 → 31/10 0.685 | +0.010 / +0.002 / +0.023 | catch-up 19 Oct, then gradual |
| 17 | BoW | 0.683 | 1 | C | 64 | 31/10 05:35 | 17/10 0.359 → 18/10 0.556 → … → 24/10 0.657 → 31/10 0.681 → 31/10 0.683 | +0.027 / +0.021 / +0.021 | catch-up, then gradual |
| 18 | Xerneas | 0.683 | 1 | C | 77 | 31/10 11:23 | 16/10 0.345 → 17/10 0.523 → … → 24/10 0.650 → 31/10 0.674 → 31/10 0.683 | +0.031 / +0.014 / +0.021 | gradual |
| 19 | muDILA | 0.682 | 2 | C/C | 38 | 31/10 05:22 | 24/10 0.636 → 24/10 0.646 → 31/10 0.682 | - / +0.010 / +0.020 | merge (jusill 0.646 on 24 Oct), gradual |
| 20 | Pirhosseinlou | 0.682 | 1 | E | 66 | 31/10 09:03 | 18/10 0.520 → 19/10 0.566 → 24/10 0.651 → 31/10 0.682 | -0.003 / +0.015 / +0.020 | gradual |
| 21 | VitalyR | 0.681 | 1 | C | 17 | 30/10 04:35 | 17/10 0.373 → 18/10 0.583 → 18/10 0.597 → 19/10 0.606 → 24/10 0.633 → 31/10 0.681 | +0.037 / -0.003 / +0.019 | early leap 18 Oct, gradual since |
| 22 | Nishant Dahal | 0.680 | 1 | C | 33 | 31/10 18:07 | 16/10 0.331 → 18/10 0.592 → 19/10 0.608 → 24/10 0.642 → 31/10 0.677 → 31/10 0.680 | +0.039 / +0.006 / +0.018 | early lead (two-stage LoRA, later published), gradual |
| 23 | .. | 0.680 | 1 | C | 55 | 31/10 08:45 | 17/10 0.553 → 18/10 0.569 → 19/10 0.576 → 24/10 0.636 → 31/10 0.678 → 31/10 0.680 | +0.007 / +0.000 / +0.018 | gradual, near plateau |
| 24 | Karsten | 0.677 | 1 | C | 13 | 30/10 19:10 | 24/10 0.603 → 31/10 0.677 | - / -0.033 / +0.015 | steps from below |
| 25 | Maniceet Sahay | 0.675 | 1 | E | 46 | 31/10 10:40 | 16/10 0.460 → 18/10 0.550 → 19/10 0.594 → 24/10 0.639 → 31/10 0.675 | +0.025 / +0.003 / +0.013 | gradual (OpenRouter generations) |

### CASMI top 25 (31 Oct 22:54 IST)
| # | Team | Score | n | Tiers | Subs | Last sub (IST) | Score path (IST) | Excess over plateau 19/24/31 Oct | Pattern |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Savko | 0.521 | 1 | C | 197 | 31/10 14:12 | 16/10 0.452 → 23/10 0.490 → 31/10 0.517 → 31/10 0.521 | +0.001 / +0.026 / +0.044 | at plateau 19 Oct, **steps** 0.490 (23 Oct), 0.517, 0.521 |
| 2 | Oleksandr Parfionov | 0.519 | 1 | E | 89 | 31/10 19:13 | 16/10 0.493 → 23/10 0.504 → 31/10 0.517 → 31/10 0.519 | +0.042 / +0.040 / +0.042 | **independent lead from day 1** (0.493 on 16 Oct), slow climb |
| 3 | pikachu | 0.504 | 1 | C | 155 | 31/10 10:11 | 16/10 0.495 → 23/10 0.499 → 31/10 0.504 | +0.044 / +0.035 / +0.027 | lead from day 1 (0.495), +0.009 since |
| 4 | Shehab Alshehabi | 0.501 | 1 | C | 186 | 31/10 00:56 | 16/10 0.454 → 17/10 0.460 → 18/10 0.463 → 23/10 0.470 → 31/10 0.501 | +0.012 / +0.006 / +0.024 | near plateau to 24 Oct, **step** to 0.501 |
| 5 | CHU1PC | 0.495 | 2 | M/C | 92 | 29/10 15:47 | 16/10 0.491 → 17/10 0.495 | +0.044 / +0.031 / +0.018 | 0.495 since 17 Oct, frozen |
| 6 | democatXamer | 0.495 | 1 | E | 71 | 31/10 12:08 | 16/10 0.443 → 18/10 0.451 → 23/10 0.469 → 31/10 0.484 → 31/10 0.495 | +0.000 / +0.005 / +0.018 | gradual, +0.011 today |
| 7 | team051 | 0.490 | 1 | C | 53 | 30/10 07:03 | 16/10 0.448 → 23/10 0.462 → 31/10 0.490 | -0.003 / -0.002 / +0.013 | step 24→31 Oct |
| 8 | Paweł Biernat | 0.490 | 1 | C | 60 | 31/10 02:41 | 16/10 0.446 → 23/10 0.474 → 31/10 0.490 | -0.005 / +0.010 / +0.013 | steps |
| 9 | JJ | 0.489 | 3 | M/M/M | 126 | 31/10 09:48 | 16/10 0.461 → 19/10 0.470 → 23/10 0.489 | +0.019 / +0.025 / +0.012 | gradual + merges (3 Masters) |
| 10 | Day: chem. Night: Say my name. | 0.486 | 1 | M | 34 | 31/10 07:43 | 23/10 0.464 → 31/10 0.486 | - / +0.000 / +0.009 | public 0.479 + private reranker (heliosli) |
| 11 | deoxy | 0.484 | 1 | M | 47 | 31/10 08:16 | 16/10 0.246 → 17/10 0.248 → 18/10 0.361 → 19/10 0.367 → 23/10 0.446 → 31/10 0.484 | -0.084 / -0.018 / +0.007 | catch-up, then +1 molecule |
| 12 | MarvinTMB | 0.484 | 1 | C | 16 | 09/10 02:18 | 16/10 0.484 | +0.033 / +0.020 / +0.007 | 0.484 by 9 Oct with 16 subs, inactive since |
| 13 | l1aF | 0.484 | 1 | C | 109 | 31/10 13:41 | 16/10 0.459 → 23/10 0.471 → 31/10 0.480 → 31/10 0.484 | +0.008 / +0.007 / +0.007 | within one molecule of 0.479 (fork/tweak/noise) |
| 14 | Igor (Yehuda) Itkin | 0.484 | 1 | C | 54 | 31/10 06:55 | 16/10 0.441 → 18/10 0.444 → 19/10 0.454 → 23/10 0.464 → 31/10 0.484 | +0.003 / +0.000 / +0.007 | within one molecule of 0.479 (fork/tweak/noise) |
| 15 | Oliver Ford | 0.483 | 1 | C | 149 | 28/10 19:35 | 16/10 0.468 → 17/10 0.472 → 18/10 0.474 → 23/10 0.483 | +0.023 / +0.019 / +0.006 | within one molecule of 0.479 (fork/tweak/noise) |
| 16 | 青葉 涼風 | 0.483 | 1 | C | 30 | 29/10 14:54 | 16/10 0.346 → 31/10 0.483 | -0.105 / -0.118 / +0.006 | within one molecule of 0.479 (fork/tweak/noise) |
| 17 | HMC_Stevelee | 0.483 | 1 | C | 164 | 31/10 11:18 | 16/10 0.445 → 17/10 0.451 → 23/10 0.461 → 31/10 0.483 | +0.000 / -0.003 / +0.006 | within one molecule of 0.479 (fork/tweak/noise) |
| 18 | Rento Yabuki | 0.482 | 1 | C | 18 | 31/10 18:58 | 17/10 0.086 → 23/10 0.469 → 31/10 0.472 → 31/10 0.482 | -0.365 / +0.005 / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 19 | 什么叫你的化学老师是贩冰冰 | 0.482 | 1 | C | 5 | 27/10 21:34 | 31/10 0.482 | - / - / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 20 | Tra Crew | 0.482 | 1 | C | 120 | 31/10 06:30 | 16/10 0.450 → 18/10 0.455 → 19/10 0.459 → 23/10 0.468 → 31/10 0.482 | +0.008 / +0.004 / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 21 | Dunyuan Zha | 0.482 | 1 | C | 95 | 28/10 06:05 | 16/10 0.451 → 17/10 0.455 → 18/10 0.460 → 23/10 0.482 | +0.009 / +0.018 / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 22 | Sirui Zeng | 0.482 | 1 | E | 93 | 31/10 16:09 | 16/10 0.442 → 17/10 0.453 → 18/10 0.459 → 31/10 0.482 | +0.008 / -0.005 / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 23 | Tristan Zhang | 0.482 | 1 | C | 78 | 31/10 06:23 | 16/10 0.450 → 17/10 0.454 → 23/10 0.455 → 31/10 0.482 | +0.003 / -0.009 / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 24 | AibePC | 0.482 | 1 | E | 51 | 31/10 06:10 | 23/10 0.464 → 24/10 0.465 → 31/10 0.481 → 31/10 0.482 | - / +0.001 / +0.005 | within one molecule of 0.479 (fork/tweak/noise) |
| 25 | 無敵 | 0.481 | 1 | C | 28 | 31/10 05:48 | 16/10 0.331 → 23/10 0.403 → 31/10 0.481 | -0.120 / -0.061 / +0.004 | within one molecule of 0.479 (fork/tweak/noise) |
