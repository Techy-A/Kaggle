---
tags: [kaggle, casmi26, research]
date: 2026-10-24
---
# CASMI 2026: clues for getting above 0.466

Research only. I made no submissions and no pushes. Our standing: rank 288 of 3,924 at 0.464, an exact copy of
dmitriigluzdov v24. On 24 Oct 06:08 UTC the public top was 0.504, gold (rank 17) was 0.471, 56 teams were at 0.466 or
above, 92 at 0.465 or above and 452 at 0.464 or above.

Sources read:
- The sanpier wrapper notebook (`/home/user/work/pull_sanpier_candidate-aware-pubchem-promotion-0-465_1024`), with
  its embedded code decoded.
- dariushafshar's chain dataset ([notebook](https://www.kaggle.com/code/dariushafshar/casmi26-the-public-chain-version-by-version),
  dataset `dariushafshar/casmi26-notebook-chain-changes-and-scores`: 481 scored versions, CC BY 4.0).
- About 30 forum threads, read through the API.
- Nine leaderboard snapshots, 16–24 Oct.

## 1. The 0.465 notebook (sanpier/candidate-aware-pubchem-promotion-0-465)

**The headline:** the 0.465 does not need the missing model bundle.
- Cell 4 holds the whole 0.465 pipeline as a base64 notebook, `combined_code/frozen_0465.ipynb`
  (sha256 `aa1a43cd…88743`).
- Its papermill end time, 2026-10-23 13:03 UTC, matches
  [sanpier/casmi-26-the-data-behind-spectra-to-structures v2](https://www.kaggle.com/code/sanpier/casmi-26-the-data-behind-spectra-to-structures),
  which is the version that scored 0.465 and now returns 404.
- The wrapper's own markdown says it "starts from the candidate-specific PubChem promotion pipeline that scored
  0.465" and adds a neural re-rank on top. "No leaderboard gain is assumed."
- So the 0.465 in the title belongs to the embedded baseline. No public score for the neural layer is visible.
- Our earlier note ("0.465 not reproducible") was wrong.
- The exact copy is decoded, its hash checked and its outputs cleared at `/home/user/work/push_casmi465_exact`. It uses
  the same five dmitriigluzdov datasets and image `…bf3fc647d461` as our 0.464. It is not pushed.

### What the 0.465 changes (code diff against dmitriigluzdov v16)
**Base:** dmitriigluzdov **v16**, the version that went from 0.456 to 0.464 with the N1 challenger:
- one fold-safe FPNet (fold 0);
- no sixth PubChem slot.

Our copy is v24, which is v20 plus documentation cells. v20 adds the five-fold FPNet ensemble and the PubChem slot at
rank 24, and both of those scored a flat 0.464.

**The only ranking change is the PubChem rank-1 promotion gate.**

**Mechanism (the bug):** in the post-simulator cell (our cell 56), the PubChem list is re-ordered by
`ice_fuse.rerank` + `gl_rerank` inside same-formula groups:
- `p['pc']` and `p['pc_keys']` are permuted, so `p['pc'][0]` can become a different structure;
- `fz0` is kept on purpose, so that "gate decisions are unchanged".

The merge cell (our cell 58) then:
- gates promotion on channel-level values: `p['S']` and `p['top_pop']`, which belong to the *pre-simulator*
  S-leader, and `rel = fz_top − best_pool_fz`, where `fz_top` is the best raw fingerprint score among non-pool
  structures;
- but `promote()` puts the *current* `p['pc'][0]` at rank 1.

So a candidate could be promoted on another candidate's evidence, or blocked by it. The N1 challenger re-orders more
lists, which makes this mismatch more frequent on v16 and later.

**The fix:**
- The PubChem worker stores `promotion_evidence[score_key] = {smiles, raw_fz, S, pop}` for every listed candidate.
- The gate looks up the promoted candidate's own S, pop and `raw_fz − best_pool_fz`.
- Thresholds are unchanged: S > 6 with pop ≥ 5, OR margin > 150.
- Interleaving slots still use the channel `rel`.
- It writes `pubchem_promotion_audit.csv`.
- Two side edits: `pc_lists.json` is unlinked before the runner, and `rr.check_returncode()` is added.

**Effect:** +0.001 public (0.464 → 0.465). That is less than one molecule moving one rank (one molecule from missed to
rank 1 is 0.0076).

### The neural spectrum model (the add-on, not the 0.465)
**Architecture.** `model.py` and `features.py` are only in the missing bundle `casmi_spectrum_model_v1.zip`, so the
architecture is inferred from `neural_scores.py`:
- Class `SpectrumCandidateModel(cfg)`.
- `encode_spectra(peaks[max_peaks], lengths, meta, adduct_idx)` returns two things:
  - Z, a spectrum embedding;
  - P, predicted fingerprint bits.
- Spectrum inputs: peaks via `feat.peaks(mzs, intensities, precursor_mz, max_peaks)`, with a `min_peaks` filter;
  metadata via `feat.metadata(precursor_mz, adduct, ionization_mode, collision_energy_ev)` plus an adduct embedding.
- `encode_molecules(F)` turns the candidate fingerprint F (`fp_bits` bits from `feat.structure(smiles)`) into an
  embedding M.
- Score: `cos(Z, M) / temperature + fp_score_weight · (P·F) / fp_bits`, averaged over **every** query spectrum, with
  its std also kept.
- This is a CLIP-style contrastive dual encoder (a set or transformer peak encoder against a fingerprint MLP) with an
  auxiliary fingerprint-prediction head.

**Training data:**
- the competition `train.parquet` only (`data_audit.json` pins its sha256);
- an identity split stored in `molecules.parquet` (`split == 'validation'`);
- `best_validation_model.pt`, the checkpoint chosen on `validation_predictions.csv`;
- `final_model.pt`, trained on everything.

**The training code is not in the notebook.**

**The held-out panel** ("fold-safe" in the wrapper's sense) is built by `panel.py`:
- 120 molecules sampled from the neural model's held-out validation identities, excluding those used to pick the
  checkpoint;
- every spectrum of those identities is removed from the runtime reference library, and the external reference
  extension is disabled;
- the frozen 0.465 pipeline is re-run on that panel as a "labelled replay";
- the panel is split 60 tune / 60 audit.

The wrapper itself admits that the public engines (FPNet full1, the rankers) may have trained on these molecules.

This is a different thing from dmitriigluzdov's **Fold-Safe FPNet** dataset, which the 0.465's PubChem channel uses
(`z_AB + 0.5·z_fold0`). Those are FPNet checkpoints trained without identity fold k and without the 250-molecule NP
panel.

**The re-rank** works only inside the final 25 (it can never add a candidate):
- `RRF = 1/(3 + baseline_rank) + w/(3 + neural_rank)`, with w ∈ {0, 0.25, 0.5, 1};
- w is chosen as the best tune-split MRR, ties go to the smaller weight, and w = 0 is kept unless the gain is
  positive;
- it is reported on audit with a 2,000-sample bootstrap;
- at w = 0.25, base rank 2 overtakes rank 1 only if the neural model ranks the old top-1 18th or worse and the rank-2
  candidate first, so it almost never touches class 1;
- at w = 1, base rank 2 overtakes once the neural model ranks the old top-1 3rd or worse (with the rank-2 candidate
  first).

`ranking.py` also holds an unused LightGBM-style stacking feature set (`BASE_FEATURES`, `residual_scores`), a sign of
an earlier, abandoned design.

**`selected_policy.json`** holds:
- `method: 'rrf'`, `weight` (0 / 0.25 / 0.5 / 1), `offset: 3.0`;
- `selected_policy` (a name), `selection_split: 'tune only'`, `tune_mrr25`, `tune_baseline_mrr25`;
- `audit` (that policy's audit-split summary: MRR, top-1, recall, delta with CI, wins, losses, changed lists);
- `frozen_source_sha256`, `model_pins` (sha256 of the 8 bundle files);
- the two checkpoint names and a `limitation` string.

It is just the chosen weight plus its evidence.

**Could we train an equivalent?** Yes, technically, from `train.parquet` alone. The licence is fine, since the host
allows models trained on train.parquet.
- A dual encoder on about 2.5 M spectra / 275 k structures takes about 3–6 h on a T4 (1–2 sessions).
- The replay is the expensive part: the full pipeline over 120 panel molecules plus the hidden run (sanpier: "several
  hours").
- The panel and fold splits already exist: dmitriigluzdov's panel dataset has `fold_identity` / `fold_scaffold`.

**Is it worth it?**
- No LB evidence that it helps.
- Our own independent FP+ranker engine fused by RRF gained only +0.001.
- Forum threads ([743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254))
  report that re-rank gains below about 0.03 offline are invisible on the public LB.
- The +0.001 of "0.465" is the gate fix, not the neural model.
- **Low priority** (idea 4 below).

## 2. Public notebooks at 0.464 or above, and what each changes
From the chain `versions.csv` (scores read 23 Oct 19:31 UTC) plus the kernel list sorted by score:

| Score | Notebook | Change against its parent |
|---|---|---|
| 0.465 | sanpier/casmi-26-the-data-behind… v2 (404 now; embedded in the wrapper) | candidate-level PubChem promotion gate on v16 |
| 0.464 | dmitriigluzdov/…data-behind… v16 | **N1 challenger**: proxy GLACIER kept where ICEBERG exists, joint ICE+GL re-order (+0.008) |
| 0.464 | dmitriigluzdov v19 / v20 / v24 | five fold-safe FPNets (0), sixth PubChem slot at rank 24 (0), tables (0) |
| 0.464 | evgendvorkin/enveda-casmi-2026 v27 | becomes dmitriigluzdov v20 |
| 0.464 | amrkhalid124 upstream-data-v16, hailonglv v16/v19 runs, lavinwins casmi-26-lb-top-1, bhuvneshrichhariya v20, d1m4official v28/v29, pjmathematician e255-dg464 | exact copies |
| 0.464 | guhongbin v492/v493 (copies), v495 (N1 arm N1A_D→N1A), v496 (LOTUS weight 0.5→1.0), v497 (S/POP_TAU 6/5→5/4) | all 0 |
| 0.464 | rajrajak99 v800/v810/v830/v840/v850/v870 | dual gate rel>125 & pop≥3, rel>120 & pop≥3.5: 0; v820 rel>165 → 0.459 |
| 0.463 | ahmedberatozer v22-n1-challenger, v25-unq | N1 challenger on his own v15 |
| 0.462 | flexonafft closing-the-evidence-gap v47/v48 | the N1 challenger's origin (+0.008); v49 CE ensemble −0.005 |
| 0.460 | yjsshisfsf v15e3 | v15 plus a third engine from non-public inputs (+0.005), not reproducible |

**Claims above 0.464:** "lavinwins CASMI 26: LB TOP 1 ⭐🏅??" and "Top 1% Gold" are forks (0.464 and 0.462; v2 of
top-1-gold scored 0.423). No public notebook other than sanpier's is above 0.464.

**Patterns.** No public notebook changes:
- the candidate pools (COCONUT / LOTUS / ChEBI / LMSD / PubChemLite are all already in);
- adduct handling beyond N1A;
- class-3 generation.

Every 0.464 variant is gate, prior or slot fiddling, and all of it is flat.

## 3. Which change types gain, mined from 481 versions
**Granularity:** the public set is about 132 of the about 400 hidden molecules, and one molecule from missed to rank 1
is 0.0076. Every large step in the chain is ±0.008, i.e. exactly one molecule.

**Reruns are deterministic:** identical code, data versions and image gave the same score in 32 of 33 rerun groups
(`rerun_groups.csv`); the exception differs by 0.002. So our "±0.004 run noise" came from unpinned copies (Kino's
0.447), not from GPU noise. A public A/B is a clean measurement on those 132 molecules. It is still a small sample
relative to the private 2/3.

| Change type | Replications | Steps | Verdict |
|---|---|---|---|
| Forward-model evidence coverage: N1 challenger (proxy GL where ICE exists, joint re-order) | 3 on (dg v16, flexon v47, ahmed v22) + 2 off | +0.008 ×3; switching it off −0.008 ×2 (+0.004 on a weaker base) | **the only big repeatable lever** (the same public molecule each time) |
| N1A proxies alone ([M+NH4]+ / [M−H2O+H]+ rebuilt as [M+H]+) | 3 | 0, +0.001, +0.008 (bundled) | small alone; needed for the challenger |
| PubChem promotion rel 250 → 150 | 5 | +0.003 to +0.006 | reliable, already in |
| engine-2 rank-2 insert | 4 | +0.002 each | reliable, already in |
| PubChemLite CC0 in engine 2 | 4 | +0.004, +0.003, +0.008 (with image); −0.003 when depth 7 | in; interacts with depth |
| engine-2 depth 6 ↔ 7 | 5 | 7 helped on the old pools (+0.003/+0.004); with PubChemLite, 7→6 gave +0.007 and our 6→7 on 0.464 gave −0.008 | stay at 6 |
| **wider generation** (20 parents, C3 16 / 400, generated rank-1 rule P1b) | 4 | −0.008 (dg v17), −0.010 (ours), −0.001, −0.012 | **consistently negative** |
| GLACIER multi-CE / meansim / CE ensemble | 3 | −0.005 ×3 | negative |
| [M+Na]+ shift → [M+H]+ GLACIER (E007) | 3 | −0.008, −0.008, −0.013 | negative |
| FPNet checkpoint swaps (hard-negative, S views, DreaMS fp, bioext pool) | 5 | −0.007 to −0.010 | negative |
| fusion weight ALPHA 0.6→0.8, TOPN 60→40, window 40→80 | 4 | −0.011, −0.008, −0.007, 0 | negative or 0 |
| PubChem prior or threshold tweaks (pop 0.15, NP bonus, LOTUS, synonyms, dual or loose gates, supplements, 10 k shortlist, sixth slot) | ~20 | all 0 (±0.001) | **flat; stop** |
| Docker image (old → default) | 2 | +0.008 (guhongbin v459), 0 (v14) | base-specific |
| candidate-level gate | 1 | +0.001 | a bug fix, tiny |

**Reading:**
- Gains come from more forward-model evidence on lists that already hold the answer (classes 1/2 at rank 2–5).
- Losses come from anything that changes list composition or swaps a model.
- Gold at 0.471 is about public + 1 molecule, and 0.473–0.475 (ranks 9–15) is about +1–1.5 molecules.
- The snapshots show ranks 9–19 jumping 0.02–0.04 between 19 and 23 Oct, e.g. Savko 0.452→0.490, CheMusyoku
  0.439→0.474, D: Discipline 0.436→0.473 with 19 submissions. This is consistent with own pipelines merged onto the
  public 0.464, though that is inference.

## 4. Forum (quotes, with links)
- **Host, NC licences now allowed (17 Oct)**, [750330](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750330),
  David Healey: "decided to remove the restriction on noncommercial license open source data (for example, CC-BY-NC)
  … from here on, solutions can use external data that is open sourced with noncommercial provisions … NIST is not
  permitted." **So the CC BY-NC-SA flag on the ahmedberatozer / casmi26-attributed-v4g assets is no longer an
  eligibility blocker.** NIST and NIST-trained models still disqualify (e.g. the ICEBERG NIST'23 atlas).
- **Test composition.** Marie Killian (host) in [745758](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745758):
  "We will not provide the breakdown of the test set, but our hope is that there are enough novel structures to make
  it worth working on them." David Healey in [742274](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/742274):
  "Class 3 is absent from both PubChem and COCONUT." Press release: about 400 molecules, about 2,500 unpublished
  spectra.
- **IDs:** inversion in [745715](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745715):
  "There is no overlap in IDs between train and test, or between dummy test and rerun test. It's a relatively
  low-observation test set …"
- **Enveda-180 extras:** Marie Killian in [745148](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745148):
  the 816 Zenodo structures absent from train "likely were filtered out in the train.parquet preparation. Data in the
  Zenodo release of Enveda-180 are fine to use." They are probably not test answers.
- **Allowed list:** [748270](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748270)
  lists train.parquet and models trained on it, MassSpecGym-trained models, ICEBERG / GLACIER, DreaMS, ChEBI / LMSD,
  CFM-ID 4, newer GNPS / MassBank / MoNA, GNPS-NIST14-MATCHES, Enveda-180 Zenodo, PubChem and COCONUT.
- **"What worked"**, Udam Liyanage in [743254](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/743254):
  - "Most database-route errors are isomer errors … the wrong winner has the same molecular formula about 98% of the
    time";
  - "adding generated candidates to the final list has not yet paid off on the leaderboard, even when only inserted
    for molecules predicted to be novel";
  - deleting-the-truth simulations "overstated reach about 3×";
  - "offline gains below about 0.03 were invisible on the public leaderboard";
  - "Evidence that depends on the structure itself (in-silico fragmentation and similar) is where we found gains."
- **Class 3**, [751300](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/751300):
  - Bull: local class-3 gains "don't translate to the LB at all";
  - Tra Crew (23 Oct): "I managed a class 3 lever with consistent tiny leaderboard gains but … no more than
    +.003–.004 … close to another lever worth +.007–.010."
- **Metric incentive**, Mahou Lettuce in [741659](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/741659):
  "one generated false positive that pushes a confident rank-1 database hit to rank 2 can wipe out the benefit of
  multiple successful but lower-ranked de novo discoveries." This is our gen20 lesson.
- **Public = ~132 molecules, about half novel:** Antriksh Arya in [748029](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/748029).
- **Kino's thread** [750241](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/750241)
  published our 100/130 parent-generation analysis. dmitriigluzdov tried it the next day (v17, gen 20 parents):
  0.464 → 0.456.
- **Ceiling:** MarvinTMB (rank 6, 0.484 with 16 submissions on 8 Oct) estimated "around 0.76, with a more realistic
  maximum probably around 0.55" in [746302](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/746302).
- hengck23 in [745029](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/discussion/745029)
  points to Enveda's MS2Mol / EnvedaDark (de novo NP structures) and PRISM, i.e. the hosts' own interest is class 3.

## 5. Leaderboard, top 20 (24 Oct 06:08 UTC; trajectory across the 16 Oct → 19 Oct → 23 Oct snapshots)
1. Oleksandr Parfionov (parfyon): 0.493 → 0.493 → 0.504, 65 submissions
2. pikachu (jzhpfn): 0.495 → 0.495 → 0.499, 123
3. CHU1PC (lowasitgoes, tadashikato): 0.491 → 0.495 → 0.495, 81
4. Savko (savchenko): 0.452 → 0.452 → 0.490, 158
5. JJ (jjinho, ttyn4519): 0.461 → 0.470 → 0.489, 97
6. MarvinTMB: 0.484 → 0.484 → 0.484, 16 (last submission 8 Oct)
7. Oliver Ford (oliness): 0.468 → 0.474 → 0.483, 144
8. Dunyuan Zha: 0.451 → 0.460 → 0.482, 83
9. Parth Talsania (3 members): 0.450 → 0.454 → 0.475, 103
10. Paweł Biernat (pawebiernatingenix): 0.446 → 0.446 → 0.474, 40
11. CheMusyoku (4 members): 0.439 → 0.439 → 0.474, 74
12. Alex #2: 0.459 → 0.464 → 0.473, 171
13. D: Discipline (5 members): 0.436 → 0.436 → 0.473, 19
14. Hortus Regis Alti: 0.445 → 0.454 → 0.473, 66
15. Shuta TAKANO: 0.446 → 0.466 → 0.473, 104
16–19. Kucing Garong, rekhasri, l1aF, Yuuki Aikawa: all 0.471
20. Shehab Alshehabi: 0.470, 151

**Gold line:** 0.456 (16 Oct) → 0.463 (19 Oct) → 0.471 (24 Oct), so it rises about 0.002 a day. Expect 0.475+ by
mid-November.

**Top-team public work:** none of the top-18 users I checked has a public CASMI kernel or dataset (CLI user listings
show nothing). No papers or GitHub repos found. The top 3 were already above 0.49 on 16 Oct, before the public chain
reached 0.456, so they run independent pipelines.

**CASMI history** (background knowledge, not re-fetched):
- Winners in CASMI 2016/2017/2022 were retrieval plus in-silico evidence: CSI:FingerID / SIRIUS with COSMIC,
  MetFrag with **reference-count (citation) priors**, and CFM-ID.
- The priors match this competition's popularity lever.
- Forward models (ICEBERG / MassFormer / FraGNNet) gained mainly on isomer separation.
- De novo methods (MSNovelist, MS2Mol, DiffMS) had low top-1 rates.
- No CASMI 2026 write-up exists yet; winning code will be open-sourced after the deadline.

## 6. Ranked top 6 to get above 0.466

| # | Idea | Evidence | Expected public gain | Effort / compute | Licence / data risk | How to validate |
|---|---|---|---|---|---|---|
| 1 | **Candidate-level PubChem gate**: (a) exact 0.465 copy, or (b) the gate on our v24 base | sanpier v2 +0.001 on v16; it is a logic fix, not tuning | (a) 0.465 by construction (reruns are deterministic); (b) 0.464–0.466, point 0.465 | Built: `/home/user/work/push_casmi465_exact`, `/home/user/work/push_casmi464_candgate`. One T4 run each (same as the 0.464 run), no CPU work | Same datasets as the 0.464 (CC BY-NC-SA, now allowed) | Bench is not trusted for PubChem-gate changes (calibration). Check `pubchem_promotion_audit.csv`: "decisions changed" should be a handful of ~400. Then one LB A/B |
| 2 | **genfix** (features-frozen gen20, δ 2.0), stacked on #1 only after #1 scores | Ungated gen20 lost −0.008 / −0.010 on two bases. genfix removes the re-scoring mechanism: 0 / 270 thin class-1 rank-1 losses, c3 +0.08 (leak-inflated), honest c3 about +0.008 | −0.002 to +0.004 (a one-molecule bet) | Built (`push_casmi464_genfix`). +45–60 min GPU commit; 2.5× engine-1 CPU | same | Done on the thin class-1 bench (genfix_bench.md). Reruns are deterministic, so one LB A/B is clean for the public 132 |
| 3 | **Forward-model coverage audit, then targeted extension** (the N1-challenger family): find molecules or lists where the answer can sit at ranks 2–5 without ICE or GL evidence (no [M+H]+, PubChem list rows that ICE never scored, engine-2 candidates missing from the ICE / GL feed) and give them evidence | The only lever with +0.008 three times. Udam: "evidence that depends on the structure itself is where we found gains" | 0 to +0.008 | CPU: read stage health and `n_mols_scored` from the 0.464 log and bench snapshot (1–2 h). Then one GPU run per extension | ICEBERG / GLACIER allowed; avoid the NIST atlas | Bench: count lists without evidence per class; gate on 0 thin class-1 rank-1 losses. Avoid the failed variants ([M+Na]+ shift, multi-CE meansim: −0.005 to −0.013) |
| 4 | **Own neural dual-encoder re-rank inside the top 25** (sanpier's add-on), RRF w = 0.25, offset 3 | No LB evidence. Our own RRF engine gave +0.001; w = 0.25 almost never changes rank 1 | 0 to +0.004 | 3–6 GPU h training on train.parquet (Kaggle T4 or Colab), plus a replay run | train.parquet models are allowed | Thin class-1 bench: 0 rank-1 losses; c2 / c3 MRR on the honest unseen slice; tune / audit split like sanpier's |
| 5 | **Retrain the engine-1 ranker with source flags** (is_generated, rank within source) over a rebuilt `sim1x`, so generated candidates are calibrated: the real class-3 lever | genfix analysis; the hosts' interest is novel structures; Tra Crew's class-3 lever | +0.004 to +0.010 if it works; high variance | 10–40 CPU-h per setting (Kaggle CPU sessions, free) plus 1 GPU run | Ahmed's engine (NC, allowed) | Leave-out queries; thin class-1 bench; a class-3 slice of real absent molecules (Udam's 3× warning) |
| 6 | **Third evidence source for molecules with no forward-model score**: negative-mode-only and adduct-only molecules get only MetFrag-lite today; try CFM-ID 4 (allowed, supports [M−H]−) only there | MetFrag-lite fallback +0.001 ×2; CFM-ID on LOTUS failed (0 to −0.018) as a pool, not as a fallback | 0 to +0.004 | CPU-heavy CFM-ID on about 25 candidates × few molecules; 2–4 CPU-h | CFM-ID 4 allowed (host) | Bench: only rows without evidence may change; class-1 rows untouched by construction |

**Order:**
1. Submit #1a (bank 0.465) and #1b (tests the gate on v20).
2. Then #2 on the better of the two.
3. #3–#6 are research.

**Stop doing:** PubChem prior, threshold or slot tweaks (about 20 flat results), generation that changes the list
without calibration, FPNet swaps.

## 7. Building the candidate gate on our base (`/home/user/work/push_casmi464` → `/home/user/work/push_casmi464_candgate`)
Script: `enveda-casmi26/variants/port_candgate.py <0.464 dir> <out dir> <slug> <title>`. Three edits, copied verbatim
from the frozen 0.465:
1. **Cell 49, the `V17_PATCH` string literal** (PubChem worker `probe_one`):
   - `promotion_evidence = {}` before `out, out_s, out_k, out_p, out_sc, seen = …`;
   - inside the listing loop after `out.append(...)`:
     `promotion_evidence[k] = dict(smiles=smis[idx[j]], raw_fz=float(fz[j]), S=float(sc[j]), pop=float(pop[idx[j]]))`;
   - after `diag['n_listed'] = len(out)`: `diag['promotion_evidence'] = promotion_evidence`.
2. **Cell 49, the `RUNNER` string literal:** the `res[mid] = dict(...)` line gains
   `promotion_evidence=d.get('promotion_evidence', {})`.
3. **Cell 58 (merge):**
   - before `rows, stats = …`: add `candidate_promotion_evidence(p)` (look up `p['pc_keys'][0]`, assert the SMILES
     matches `p['pc'][0]`, return S, pop, `raw_fz − best_pool_fz` or 1e9) and `promotion_audit = []`;
   - replace `S = p.get('S'); pop_top = …; if (S > S_TAU and pop_top >= POP_TAU) or rel > 150.0:` with
     `old_promote` / `new_promote` and `if new_promote:`;
   - after `print('merge stats', stats)`: write `pubchem_promotion_audit.csv`.

Kept from v20: five-fold FPNet, `PC_SIXTH_SLOT = 24`, the `REL_TH` slot choice on channel `rel`.

**Checks run:**
- the patched `V17_PATCH` is identical to the frozen 0.465's;
- `RUNNER` differs from it only by the v20 five-fold block;
- the diff against the 0.464 is these edits only (36 lines);
- all code cells parse.

**Not added:** sanpier's `pc_lists.json` unlink and `rr.check_returncode()`, which are robustness only and are not
needed on a fresh run.

**Risk:** the gate's `assert evidence is not None` fails a run if a PubChem list lacks evidence. It held on sanpier's
full hidden run.

## Sources
- [sanpier wrapper](https://www.kaggle.com/code/sanpier/candidate-aware-pubchem-promotion-0-465)
- [chain notebook](https://www.kaggle.com/code/dariushafshar/casmi26-the-public-chain-version-by-version)
- [dmitriigluzdov data-behind](https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures)
- forum threads as linked above
- [CASMI history (EPA tools review)](https://dx.doi.org/10.3390/metabo10060260)
- [CASMI: And the Winner is](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3901266/)
- [MassFormer](https://arxiv.org/pdf/2111.04824)
