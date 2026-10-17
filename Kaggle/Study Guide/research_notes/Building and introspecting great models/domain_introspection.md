# Domain-specific evaluation, error analysis and introspection: (A) MS/MS molecule identification and (B) LLM authorship attribution

Scope note: the primary sources were downloaded and read as full text (arXiv HTML, J. Cheminformatics, Fiehn Lab CASMI pages, GitHub READMEs). Where a number comes only from a search-engine summary and not from the primary text, the entry says so. Numbers in tables are copied as printed. Kaggle relevance is written up under "Inferences" in each section, for (1) CASMI 2026 (MRR@25 on 2D/stereo-free InChIKey14; test classes 1/2/3, with class 3 about 40-50% novel molecules; pipeline = library matching + analog candidate generation + GBDT rerankers + PubChem candidates) and (2) "Name That LLM" (18-way, balanced accuracy, same-lab models, about 2-3 passages per prompt in test).

---

## A1. How CASMI organisers and MS/MS papers evaluate and break down failures

### Takeaway
Experts report rank-based metrics per challenge (top-1/top-3/top-10, mean/median rank, medal counts; later hit@k and MCES@1). They split results by ionisation mode, by category (MS/MS-only vs. with metadata), and by the stage of the pipeline (adduct, then formula, then compound class, then 2D structure). They also cluster challenges by per-method rank to show where methods complement each other. The main pattern is a large gap between getting the formula right (often over 90%) and getting the 2D structure right (about 25-30% best in CASMI 2022). Metadata or prior terms, such as reference counts, can more than double top-1.

### Cited Findings
**CASMI 2016 (Schymanski et al. 2017, J. Cheminform. 9:22), Categories 2 and 3, automated methods**
- 208 challenges (188 substances; 127 positive [M+H]+, 81 negative [M-H]-), plus 312 training spectra. All were acquired on a Q Exactive Plus Orbitrap with stepped 20/35/50 HCD — [CASMI 2016 paper](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- Candidate lists came from ChemSpider via MetFrag2.3 (monoisotopic mass ±5 ppm). The correct answer was checked to be present in every list (a guaranteed-recall setting) — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- Train/challenge split by spectral similarity: a spectrum was a challenge only if no MassBank spectrum matched above 0.85 similarity (MetFusion). This split later made the MoNA library-match weight "too optimistic", and adding MoNA hurt Category 3 results — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Stereo handling:** "submissions were filtered post-submission to remove duplicate stereoisomers using the first block of the InChIKey. The highest scoring isomer was retained." For tied scores, the rank used is the average rank over the tied candidates — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- Metrics: absolute rank of the correct answer, gold/silver/bronze per challenge, mean/median rank, Top-X counts, relative ranking position (RRP), a "Formula 1 score" (top-10 ranks weighted 25,18,15,12,10,8,6,4,2,1) and a Medal score. Non-integer ranks are rounded up — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- Headline: in Category 2 (in silico fragmentation only), CSI:IOKR (Team Brouard) won 86/208 (41%), CSI:FingerID won 82 (39%) and MS-FINDER 70 (34%). The best methods reached >30% top-1, and about 50% of challenges had the answer in the top 10. In Category 3 (with metadata), the best reached about 70% top-1 and >80% top-10 — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Breakdown by ionisation mode:** in positive mode CSI:IOKR_AR and CSI:FID were similar (69 vs 70 wins), but all CSI methods were beaten by MS-FINDER and CFM in negative mode, which had less training data. CSI:FingerID did not submit negative mode at all for lack of training data. Combining MetFrag+CFM in negative mode raised top-1 from 9 (MetFrag) and 12 (CFM) to 20 — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Training-data effect:** retraining CSI:IOKR on more data (7352 GNPS+MassBank spectra) moved Challenge 178 from rank 3101 to rank 1 — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Memorisation check during the contest:** Team Dührkop noticed that the top-scoring candidate matched a training compound in 67 challenges. They submitted a "leave out" entry with all challenge compounds removed from training to measure true generalisation — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Metadata or prior dominates:** ChemSpider reference count alone "outperformed all individual metadata categories and even the combined fragmenters clearly". "Lowest ChemSpider ID" (older = better known) also outperformed the combined fragmenters. Retention time alone performed poorly. MetFrag+CFM+RT+Refs was the best Category 3 entry. An equal-weight combination of six fragmenters (CFM, CSI:FID, CSI:IOKR, MAGMa+, MetFrag, MS-FINDER) beat every individual Category 2 entry — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- Organisers' caution: metadata "is essential to guide users to the identification for known substances—but misleading when assessing the performance of computational methods", and "cannot assist in the identification of true unknowns" — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Challenge-by-method clustering:** heatmaps of log-rank were clustered across challenges and methods. Mean candidate counts per challenge cluster were 380 to 1603 (negative mode, mean 816) and 722 to 5297 (positive mode, mean 1281). The organisers' conclusion: "candidate numbers are not the only driver of performance". Number of peaks was also inconclusive — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Case-level failures:** all ML methods failed on Challenge 36, a 3-peak spectrum of a compound usually measured in positive mode. Combinatorial fragmenters failed on monobenzyl phthalate (Challenge 41), whose main peak is a rearrangement they do not model. ML methods did well there because phthalates are well covered in training data. Abietic acid was missed by everyone even though its spectrum was informative and its candidate list small — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)

**CASMI 2022 (Fiehn Lab results page): staged breakdown by pipeline step**
- Results were scored in four categories: (1) correct adduct, (2) correct elemental formula, (3) correct ClassyFire compound class, (4) correct 2D structure. The organisers framed the trial as gauging "how close we are with respect to compound ID of unknowns" — [CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)
- Priority set (about 249 challenges): best adduct accuracy was 93-94% (Dührkop 93%, Nikolic 94%). Best formula accuracy was 94% (Dührkop), 93% (Nothias) and 91% (Rutz). Best compound class was 69% (Dührkop). Best 2D structure was only 26% (Dührkop 57/222), 25% (Nothias, Rutz) and 38% for one small-attempt entry (Nikolic_KUCA, 21/55) — [CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)
- Bonus set: best 2D structure was 29% (Dührkop 64/219), with formula 95% and class 75-77% — [CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)
- The spread is wide: formula accuracy ranged from 0% to 94% and 2D structure from 0% to 26-38% across 16+ teams. Several teams attempted only a subset, so the denominators differ (e.g., 22 vs 249 attempts) — [CASMI 2022 results](https://fiehnlab.ucdavis.edu/casmi/casmi-2022-results)

**MassSpecGym (Bushuiev et al., NeurIPS 2024 D&B)**
- 231k spectra and 29k unique molecules. Only [M+H]+ and [M+Na]+ adducts are included. 98% of spectra are linked to Orbitrap or QTOF, and about 53% have collision energy — [MassSpecGym arXiv](https://arxiv.org/abs/2410.23326)
- Retrieval metrics: hit rate@1/5/20 plus **MCES@1**, the edit distance of the top-1 candidate to the truth. This gives partial credit for near misses. De novo metrics: top-1/top-10 exact accuracy, plus min MCES and max Tanimoto among the top-k — [MassSpecGym](https://arxiv.org/abs/2410.23326)
- Candidate sets have two tracks: same mass (mass-based) and same formula ("bonus"). The two tracks are scored separately. Candidates are drawn in priority order from three molecule pools, capped at a fixed maximum (256 per spectrum per audits) and randomly sampled if more exist — [MassSpecGym](https://arxiv.org/abs/2410.23326); cap of 256 per [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624) (the cap value comes from the audit's text on MIST-CF filtering; the original paper's number was lost in HTML math)
- Original baselines: no de novo baseline achieved accuracy above zero. The SMILES Transformer did not beat random chemically valid generation — [MassSpecGym](https://arxiv.org/abs/2410.23326)
- Corrected leaderboard (audit, formula-known bonus track, 99.9% BCa CIs): Random 3.06% HR@1; DeepSets 4.42%; FP FFN 5.09%; MIST (MSG version) 9.57%; JESTR 11.82%; MVP 13.96%; FLARE 22.66%; ICEBERG forward-model retrieval 36.18% HR@1 / 78.83% HR@20; MIST vFRIGID 53.76% — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- Mass-based track: Random 0.43% HR@1; random restricted to ground-truth formula 12.98%; MIST with ground-truth formula 51.45%; MIST with a data-safe predicted top-1 formula 26.64%. Formula knowledge alone almost doubles HR@1 — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)

**De novo numbers (typically low top-1)**
- DiffMS (formula-conditioned graph diffusion, 2025): on NPLIB1, top-1 8.34% / top-10 15.44%; on MassSpecGym, top-1 2.30% / top-10 4.25% (MCES 18.45, Tanimoto 0.28). On MassSpecGym, MIST+MSNovelist scored 0.00% top-1 and MADGEN 1.31% — [DiffMS](https://arxiv.org/abs/2502.09571)
- MSNovelist (Stravs et al., Nat. Methods 2022): 25% top-1 and 45% retrieved among the top-128 on 3,863 GNPS spectra, versus 39% top-1 for a CSI:FingerID database search. On CASMI 2016: 26% top-1 and 57% retrieved — [MSNovelist, Nat. Methods](https://link.springer.com/article/10.1038/s41592-022-01486-3) (numbers taken from a search summary of the paper; the PMC page was bot-blocked)
- MIST+MolForge claimed 28% top-1 / 36% top-10 on MassSpecGym — [One Small Step with Fingerprints](https://arxiv.org/abs/2508.04180). The audit shows this came from a faulty batched-MIST checkpoint. **Corrected: 10.73% top-1 / 14.48% top-10**, encoder Tanimoto 0.457 vs 0.812 inflated — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)

### Inferences
- For CASMI 2026, the classic expert breakdown is to report MRR@25 per test class (1/2/3). Within each class, also report by ionisation mode/adduct, formula correct vs. wrong, candidate-set size bin, and max Tanimoto/MCES of the true molecule to the nearest training/library molecule. The CASMI 2016 negative-mode and training-coverage results suggest the class-3 (novel) gap will be driven by coverage, not candidate count.
- Building a staged CASMI-2022-style funnel for our own pipeline (adduct, then formula, then candidate-set recall, then rank) is the most diagnostic single view. If the formula is right for about 90% but the true structure is not in our candidate set, rerankers cannot help.
- Expect novel-molecule (class 3) top-1 to be closer to MassSpecGym numbers (10-50% depending on formula knowledge) than to library-match numbers. De novo generation alone is unlikely to beat retrieval at top-1 (2-11% top-1 on MCES-split benchmarks once leaks are fixed).
- The CASMI 2016 finding that equal-weight combination of six fragmenters beats any single one supports rank- or score-fusion of heterogeneous scorers in our GBDT reranker. Examples: library cosine, CSI-like fingerprint score, forward-simulation score, PubChem prior.

### Gaps
- CASMI 2017 results paper was not retrieved. Its numbers and breakdowns are not in these notes.
- No CASMI 2022 peer-reviewed results paper was found. Only the Fiehn Lab tables (no per-class or per-mode breakdown).
- Per-compound-class (ClassyFire superclass) breakdowns of top-1 for MIST/ICEBERG/CSI:FingerID were not extracted from primary sources.

---

## A2. Known evaluation pitfalls in MS/MS identification

### Takeaway
The pitfalls that matter most are leakage of near-duplicate structures, leakage of formulas, and shortcuts in the candidate set. Near-duplicate leakage comes from InChIKey-only splits, external pretraining data and simulators. Candidate-set shortcuts include SMILES formatting differences, PubChem default order, and "popular compound" priors. A 2026 audit found evaluation issues in at least 17 of 26 papers reporting MassSpecGym results. Spectrum-blind baselines reached 49-99% HR@1 through these artifacts.

### Cited Findings
- **InChIKey-2D splits leak near-duplicates:** most prior work split on 2D InChIKey, which "can be compromised by minor structural modifications". MassSpecGym instead clusters molecules with MCES (agglomerative, min linkage, threshold 10), so no train-test pair has MCES < 10. Figure 3 shows that InChIKey splits leak test molecules with Tanimoto > 0.85 — [MassSpecGym](https://arxiv.org/abs/2410.23326)
- The MIST-CANOPUS (NPLIB1) benchmark is about 9k molecules / 11k spectra, uses a 2D-InChIKey split and "result[s] in data leakage" — [MassSpecGym](https://arxiv.org/abs/2410.23326)
- **Random splits are mostly lookups:** 95.3% of MassSpecGym-random and 62.9% of NPLIB1-random test structures appear in training by 2D InChIKey, versus 14.3% and 1.6% on scaffold splits. The nearest-neighbour upper bound is Jaccard 0.972 on MSG-random but 0.489 on the scaffold split — [Coley lab audit repo](https://github.com/coleygroup/Audit_ML_MS_analysis)
- **Subset-denominator pitfall:** a nearest-neighbour baseline skipped test items without a same-formula training candidate, discarding up to 62% of some splits (the hard ones). This made it look better than MIST — [Coley lab audit repo](https://github.com/coleygroup/Audit_ML_MS_analysis)
- **Checkpoint/metric mismatch:** MIST selected by validation BCE while reporting binarised Jaccard ended up as epoch-5/epoch-10 under-trained checkpoints, with a hard-coded 0.5 threshold — [Coley lab audit repo](https://github.com/coleygroup/Audit_ML_MS_analysis)
- **Audit scale:** evaluation issues in at least 17 of 26 papers reporting MassSpecGym results in its first year. Three classes: data leakage, shortcut learning, implementation/metric divergence — [MassSpecGym in the Wild (Liu, Bushuiev, … Pluskal, Coley, 2026)](https://arxiv.org/abs/2606.19624)
- **Pretraining leakage:** decoders pretrained on large corpora excluding only exact 2D-InChIKey test matches memorise near analogs. Per-compound top-1 accuracy rises with max Tanimoto to the pretraining set. In one example, a poor encoder fingerprint (Tanimoto 0.61) was "fixed" by copying a pretraining molecule at Tanimoto 0.98. Recommendation: exclude analogs above a Tanimoto threshold, and report performance stratified by max similarity to the pretraining set — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Simulator (synthetic data) leakage:** MIST trained on ICEBERG-augmented spectra gives retrieval R@1 of 67.75% (ICEBERG trained on train+test, augmenting test molecules), 63.85% (train-only ICEBERG but augmenting test molecules) and 53.76% (leak-free). The no-augmentation MIST gives 9.57% — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Formula leakage:** a formula annotator (MIST-CF) trained with test spectra raised MIST mass-track HR@1 from 26.64% to 35.15% (+9 points, 35% relative). Restricting random to the true formula raises random 30× (0.43% to 12.98%) — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Halogens and high mass:** formula prediction "remains challenging in some cases such as halogenated molecules or molecules with masses above 600 Da" — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **SMILES canonicalisation shortcut:** ground-truth SMILES kept repository formatting while decoys were RDKit-canonical. Effects: a ChemBERTa SMILES-only classifier reached **99.66% HR@1** (8.2% after fixing); DreaMS+ChemBERTa reached 82.41%, then 6.70%; ChemFormer two-stage reached 63.89%, then 7.60% — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **PubChem ranking-bias shortcut:** ranking candidates by PubChem PUG-REST default order, ignoring the spectrum, gives **49.98% HR@1**. Ranking by number of chiral atoms gives 4.88% HR@1, on par with learned FP baselines. True molecules are "common metabolites, widely available standards, or easily synthesizable compounds" with higher deposition frequency — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- Recommended sanity check: "masking all input MS/MS spectra with a constant value and confirming that performance degrades accordingly". Property-matched decoys reduce spectrum-blind baselines to near random (Gupta & Skinnider, concurrent work, as cited) — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Metric divergence:** InChIKey matching should use the 14-character 2D connectivity block. The full 27-character key treats stereoisomers as distinct and under-counts hits. Tanimoto depends on fingerprint type, radius and bits (recommended: 2048-bit ECFP4). MCES uses heuristic solvers (recommended threshold=15, always_stronger_bound=True). Recommended reporting: 99.9% CIs from 20,000 bootstrap resamples — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Implementation bug:** the MIST/DiffMS attention used `attn += attn_mask` with a boolean mask, so padding tokens were not excluded in batched mode. A reused checkpoint then inflated downstream de novo results about 3× — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Meta-score pitfall (deliberate demo):** "MAD HATTER" combines CSI:FingerID with database metadata, including melting point and the count of words starting with "u" in the compound description. It reaches 97.6% correct annotations searching PubChem, built to demonstrate evaluation flaws — [MAD HATTER, Metabolites 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10053663/) (from search summary of abstract)
- **Coverage bias:** widely used small-molecule datasets "lack uniform coverage of biomolecular structures". Kretschmer et al. propose myopic MCES distance plus UMAP, compound-class distributions and natural-product-likeness to assess coverage — [Coverage bias, Nat. Commun. 2025](https://www.nature.com/articles/s41467-024-55462-w)
- **Fingerprint-vs-retrieval trade-off:** losses that optimise fingerprint Tanimoto (IoU loss) are worst for retrieval, and contrastive losses (best retrieval) are worst for fingerprint accuracy. This is a Pareto front, and its severity depends on candidate-set similarity structure ("near duplicates in the candidate set"). Cosine scoring at inference beat other similarity functions regardless of the training loss — [De Waele et al. 2026, "what are we optimizing for?"](https://arxiv.org/abs/2602.16507)
- The same paper notes that MassSpecGym's reported MIST ran with ground-truth formulas even in the mass-based track ("a form of data leakage, or ... 'oracle' setting"; MassSpecGym GitHub issue #52) — [De Waele et al. 2026](https://arxiv.org/abs/2602.16507)
- **Recall ceiling by construction:** CASMI 2016 verified that the correct solution was present in every candidate list — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1). MassSpecGym's candidate sets are also constructed to contain the true molecule ("By definition, the ground truth fingerprint should be part of the candidate set") — [De Waele et al. 2026](https://arxiv.org/abs/2602.16507)

### Inferences
- **CASMI 2026 (InChIKey14 MRR@25):** dedupe candidates by InChIKey14 *before* ranking. Otherwise stereoisomers/tautomers of the same 2D skeleton waste slots in the top-25 and push the true key down (CASMI 2016 dedupes by first block keeping the highest-scoring isomer). Tautomer canonicalisation must be applied identically to candidates and to our generated analogs.
- **PubChem prior is a legitimate signal but a dangerous one:** popularity features (PubChem CID order, reference/patent counts, CID age) can take a reranker to about 50% HR@1 on known molecules with no spectrum at all. They likely help on classes 1/2 and hurt on class 3 (novel), where the truth is by definition unpopular or absent. Train or validate the reranker separately per class, or add an explicit "novelty" interaction. Ablate the prior features on a novel-like validation fold.
- **Validation split:** our local validation must mimic class 3. Use an MCES- or Tanimoto-disjoint holdout (e.g., max Tanimoto to train < 0.85, or MCES ≥ 10), not a random or InChIKey split. A random split overstates results massively (95% overlap on MSG-random).
- **Leakage through components:** any spectrum simulator (ICEBERG/CFM-ID), formula annotator or embedding model (DreaMS, MS2DeepScore) we use for features must not have been trained on our validation molecules. If they were, the GBDT will over-weight them.
- **Shortcut tests to run:** (1) constant-spectrum ablation of the reranker; (2) a SMILES-format/source classifier check: make sure candidates from library hits, analog generation and PubChem are canonicalised the same way, so the reranker cannot learn "which pipeline produced this candidate"; (3) a feature-only-prior baseline (PubChem rank only) as a floor.
- **Recall ceiling:** unlike CASMI 2016/MassSpecGym, Kaggle class-3 truths may not be in PubChem at all. Measure candidate-set recall@any per class, and treat analog-based generation as the way to raise that ceiling.

### Gaps
- Gupta & Skinnider's property-matched-decoy paper was not retrieved directly. It is known here only as cited by the audit.
- No primary source was found that quantifies tautomer-related mismatch rates for InChIKey14 specifically. Standard InChI handles some tautomerism (mobile H) but not all; this was not verified in these sources.
- Instrument/adduct shift was not quantified beyond CASMI 2016's positive/negative gap and MassSpecGym's restriction to [M+H]+/[M+Na]+.

---

## A3. Introspection tools in MS/MS identification

### Takeaway
Experts look into predictions with: fragmentation trees and subformula peak annotation (SIRIUS); stratifying accuracy by similarity to training or pretraining data; the per-challenge heatmaps and clustering above; spectrum-masking and spectrum-blind baselines; and selective prediction (risk-coverage curves using retrieval-level confidence). For de novo outputs they report "close match" quality (MCES, Tanimoto), not only exact hits.

### Cited Findings
- SIRIUS fragmentation trees were computed for all candidate formulas, and trees scoring below 80% of the optimal tree score were discarded before CSI:FingerID fingerprint prediction (with Platt-calibrated bit probabilities). Fragment intensities were down-weighted in learning — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- MIST assigns chemical subformulae to peaks and predicts fingerprints with a formula transformer, then ranks by cosine — [MassSpecGym](https://arxiv.org/abs/2410.23326). MassSpecGym also removed spectra where more than 50% of total intensity is not explained by plausible subformulae, which is a peak-explanation quality filter — [MassSpecGym](https://arxiv.org/abs/2410.23326)
- MS-FINDER scores candidates with mass accuracy, isotope ratio, product-ion and neutral-loss assignment, bond dissociation energy and hydrogen-rearrangement penalties. MetFrag scores explained peak mass and intensity minus bond-breaking energy — [CASMI 2016](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-017-0207-1)
- **Similarity-stratified accuracy plot:** per-compound top-1 accuracy with 95% bootstrap CIs, binned by maximum Tanimoto (2048-bit ECFP4) to the pretraining set. This is the recommended diagnostic for generalisation claims — [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624)
- **Selective prediction:** fingerprint-level uncertainty (per-bit) is "a poor proxy for retrieval success". Cheap first-order retrieval confidence and retrieval-level aleatoric uncertainty give strong risk-coverage trade-offs and beat Bayesian epistemic estimates. Distribution-free risk control gives annotation subsets with a guaranteed error rate — [Selective prediction for MS/MS retrieval (2026)](https://arxiv.org/abs/2603.10950)
- De novo evaluation reports top-k max-Tanimoto and min-MCES because exact match is rare. DiffMS argues that "structurally close matches can offer valuable insights", and shows leucine/isoleucine spectra are "essentially indistinguishable", an identifiability limit — [DiffMS](https://arxiv.org/abs/2502.09571)
- Coverage introspection: UMAP plus myopic MCES, compound-class distributions and natural-product-likeness to see where a dataset is out of domain — [Coverage bias](https://www.nature.com/articles/s41467-024-55462-w)

### Inferences
- **For the Kaggle reranker:** log, per query, (a) whether the formula of the top candidate equals that of the truth, (b) the truth's rank before and after reranking, (c) max Tanimoto of the truth to the nearest library spectrum's molecule, and (d) the score margin between rank-1 and rank-2. Plot MRR@25 against (c): it should mirror the audit's similarity-stratified plot, and class 3 should sit at the low-similarity end.
- Use the margin and top-1 score as confidence features. Per the selective-prediction paper they predict correctness better than fingerprint uncertainty. They can also gate how much weight goes to analog candidates vs. PubChem priors.
- GBDT feature importance (gain/SHAP) on the reranker should be checked for the prior-feature dominance warned about in A2. If PubChem-popularity features dominate class-3 predictions, that is a red flag.
- Fragment-explanation features (fraction of intensity explained by subformulae of the candidate, number of explained peaks, fragmentation-tree score) are what experts use. They are interpretable reranker features that transfer to novel molecules better than priors.
- GNPS-style analog search / molecular networking: no primary numbers were retrieved here (see Gaps). The documented concern for analog candidates is that their accuracy decays with structural distance, which is why MCES@1 and Tanimoto-to-truth are reported.

### Gaps
- GNPS analog search / molecular networking (modified cosine, MS2DeepScore, Spec2Vec) evaluation numbers were not retrieved within budget.
- ICEBERG/CFM-ID per-peak attribution and fingerprint-bit importance analyses (which bits drive CSI:FingerID retrieval) were not covered by fetched sources.

---

## B1. How LLM attribution / AI-text detection papers evaluate and introspect

### Takeaway
Model-attribution papers fine-tune embedding classifiers and show near-perfect cross-family accuracy, for example 97.1% for 5-way ChatGPT/Claude/Grok/Gemini/DeepSeek. Within-family accuracy is much lower (59.8% for 4 Qwen2.5 sizes). They introspect with transformation ablations (strip formatting, shuffle words or letters, paraphrase, translate, summarise), with TF-IDF+LR coefficient phrase lists, first-word distributions, markdown-only inputs, confusion matrices and leave-one-model-out "similarity" probes. Detection benchmarks (RAID, MAGE, M4GT) stress unseen generators, unseen domains, decoding settings and attacks, and report accuracy at a fixed false-positive rate.

### Cited Findings
**Sun et al. 2025, "Idiosyncrasies in Large Language Models" (ICML 2025)**
- Setup: fine-tune LLM2vec embeddings with LoRA and a K-way head. Inputs are truncated to 512 tokens. 11K responses per model (10K train / 1K val), with the same UltraChat prompts across models — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- 5-way chat APIs (GPT-4o, Claude-3.5-Sonnet, Grok-2, Gemini-1.5-Pro, DeepSeek-V3): **97.1%** (chance 20%). Pairwise 97.2-99.9%. 4-way instruct (Llama3.1-8b, Gemma2-9b, Qwen2.5-7b, Mistral-v3-7b): 96.3%. Base models: 87.3% — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Within family:** 4 Qwen2.5 instruct sizes (7B/14B/32B/72B) give **59.8%** for 4-way. Pairwise ranges from 63.1% (14B vs 32B) to 85.5% (14B vs 72B). Qwen2.5-7b base vs instruct: 96.8% — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **OOD prompts:** classifiers trained on UltraChat transfer to Cosmopedia, LmsysChat and WildChat at 88.3-99.8% — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Prompt-level controls:** adding a length constraint (≤100 words, one paragraph) gives 93.0%. A plain-text-only format constraint gives 91.4% (original 96.3%, instruct) — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Input length:** about 50% accuracy from a *single first token* for chat/instruct models. Accuracy saturates beyond about 256 tokens — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Decoding:** distinguishing the same model's greedy, softmax, top-k and top-p outputs is near chance (50.0-59.6% pairwise). 5 temperatures (0 to 1) give 37.9% vs 20% chance. Decoding settings barely change model identity — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Embedding backbone:** ELMo 90.8 / BERT 91.1 / T5 90.5 / GPT-2 92.1 / LLM2vec 97.1 (chat) — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Data size:** 10 samples per class already gives 40.3% (chat). Accuracy converges around 10K per class — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Ablations (chat / instruct / base):** original 97.1/96.3/87.3; special characters removed 95.1/93.8/75.4; **words shuffled 88.9/88.9/68.3**; letters shuffled 39.1/38.6/38.9; **markdown skeleton only** ("xxx" replacing text) 73.1/77.7/38.5 — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Rewriting by GPT-4o-mini (chat / instruct / base):** paraphrase 91.4/92.2/71.7; translate to Chinese 91.8/92.7/74.0; summarise 58.1/57.5/44.7. Semantic and content choices carry signal beyond surface wording — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Interpretable baseline:** TF-IDF (uni+bigram) + logistic regression reaches 85.5% (chat) / 83.7% (instruct). Top coefficients: ChatGPT "such as", "certainly", "overall", "below is"; Claude "here", "according to", "based on (the text)". Claude uses "the/and/to/of" much less often. Letter frequencies are nearly identical across models — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Markdown:** Claude has a high density at zero bold and zero headers. ChatGPT bolds each enumerated key point and uses headers — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Similarity probe (leave-one-model-out):** a 5-way classifier labels held-out Grok outputs as ChatGPT 82.8% of the time. ChatGPT and DeepSeek outputs are often labelled Phi-4 (55.9% and 76.0%) — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Synthetic-data inheritance:** SFT of Llama3.1-8b and Gemma2-9b on the same ChatGPT-generated UltraChat drops their separability from 96.5% to 59.8%. Fine-tuning Qwen2.5-7B on Llama outputs vs Gemma outputs yields 98.9% separability — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- 10-way (5 chat APIs + 5 instruct incl. Phi-4): 92.2% — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)

**Same-lab / same-release confusion (code domain)**
- On JavaScript, 5-class attribution within the GPT family gets 90.2% (BERT-base), about 8 points below the cross-family task. "Most errors arise from confusion between closely related variants of the same release (like GPT-4o vs. GPT-4o-mini, or among GPT-5 variants)". The best custom model (CodeT5-JSA) gets 95.8% (5-class), 94.6% (10-class) and 88.5% (20-class). It is robust to minification (91%) and mangling (93%) — [LLM-NodeJS attribution, 2025](https://arxiv.org/abs/2510.10493)

**M4GT-Bench (Wang et al., ACL 2024), Task 2: multi-way generator attribution**
- 7 labels (human, davinci-003, ChatGPT, GPT-4, Cohere, Dolly-v2, BLOOMz), with a leave-one-domain-out evaluation. Metrics: accuracy, macro-F1 and per-class F1. "davinci-003 is the most challenging generator to distinguish, followed by ChatGPT", which are the two OpenAI siblings. BLOOMz is easiest (>90% F1 even for NELA-SVM). Humans given 5 demonstrations could hardly tell generators apart — [M4GT-Bench](https://arxiv.org/abs/2402.11175)

**Detection benchmarks**
- RAID (Dugan et al., ACL 2024): 6M+ generations, 11 generators, 8 domains, 11 adversarial attacks, 4 decoding strategies (greedy/sampling × with/without repetition penalty); 12 detectors. Accuracy is reported at a fixed FPR of 5% — [RAID](https://arxiv.org/abs/2405.07940)
- MAGE: 27 LLMs, 7 writing tasks, 3 prompt types (continuation, topical, specified), and 8 testbeds of increasing "wildness" with leave-one-model-set-out and leave-one-domain-out folds. Metric is AvgRec (mean of human and machine recall) plus AUROC. Longformer AvgRec drops from 90.53% to 68.40% on unseen domains. The best detector misclassifies 61.95% of human texts from unseen domains — [MAGE](https://arxiv.org/abs/2305.13242)
- Ghostbuster: features from token probabilities under weaker LMs (unigram to davinci), chosen by structured forward feature search, with a linear classifier. 99.0 F1 in-domain, and +7.5 / +2.1 / +4.4 F1 generalisation across domains, prompts and models. ChatGPT was prompted to match human document length "to avoid spurious effects of document length" — [Ghostbuster](https://arxiv.org/abs/2305.15047)
- Binoculars: ratio of perplexity to cross-perplexity between observer and performer LMs (Falcon pair), zero-shot. Detects >90% of ChatGPT samples at **0.01% FPR** — [Binoculars](https://arxiv.org/abs/2401.12070)

### Inferences
- **For "Name That LLM" (18-way, balanced accuracy):** cross-lab pairs will be near-solved. The balanced-accuracy loss will concentrate in same-lab clusters (sizes or versions of one family). Sun et al.'s 59.8% for 4 Qwen sizes and the JS paper's GPT-4o vs 4o-mini confusions are the expected failure shape. Build the confusion matrix first and judge progress by block-diagonal (within-lab) accuracy.
- A fine-tuned modern embedding/encoder (LLM2vec-, DeBERTa- or Qwen-based) is the strong baseline. A TF-IDF uni/bigram LR is a cheap, interpretable baseline that should land within about 10 points, and its coefficients reveal the phrases that carry the signal.
- Signal is front-loaded (about 50% from the first token), so features of the opening (first word or phrase, greeting, "Certainly", "Here is") deserve explicit treatment. Do not truncate from the left.
- With about 2-3 passages per prompt in the test set, prompt-level grouping can be exploited. Passages from the same prompt are likely from *different* models (if the generation design assigns each prompt to several models). A joint per-prompt assignment (e.g., Hungarian matching or "no duplicate model within a prompt" constraints over the softmax) can lift balanced accuracy. Verify this on train, where the prompt-to-model structure is known.
- Decoding-setting differences are nearly invisible (Sun et al.), so temperature variation in the data should not be treated as a separable factor. Repetition penalty may, however, change surface statistics (RAID), which matters for perplexity features.

### Gaps
- Sun et al.'s per-class confusion matrices (Appendix B.1) were not extracted, so per-model error shares for chat APIs are not listed.
- M4GT-Bench Task 2 per-detector accuracy/F1 numbers were not extracted from tables (only the qualitative ranking of hard generators).
- I found no primary text-domain (non-code) study that isolates same-release pairs such as GPT-4o vs GPT-4o-mini. A TextPulse web preprint claims version-specific "modelolects" but is not peer-reviewed ([TextPulse](https://textpulse.ai/research/modelometry)) and was not verified.

---

## B2. Pitfalls in LLM attribution / AI-text detection evaluation

### Takeaway
Reported accuracy is fragile under shifts in generator, domain, decoding (especially repetition penalty), prompt type, and attacks (homoglyphs, synonyms, paraphrase). Confounds include prompt/topic leakage, length, and dataset construction artifacts. Training on synthetic data from one teacher can erase or transplant idiosyncrasies. Evaluation choices (AUC vs fixed-FPR, random vs grouped splits) change conclusions.

### Cited Findings
- Repetition penalty "decreases accuracy by up to 38 points regardless of decoding strategy". Greedy is much easier than sampling — [RAID](https://arxiv.org/abs/2405.07940). A secondary summary quotes "up to 32 points"; the v1 primary text says 38.
- Strong detectors "sometimes deteriorate from perfect accuracy to complete failure": changing the generator, the decoding strategy or adding a repetition penalty "was enough to introduce a 95+% error rate" — [RAID](https://arxiv.org/abs/2405.07940)
- **Training-distribution bias:** RoBERTa-GPT2 gets 95+% on GPT-2 text in five domains but "rarely achieves beyond 60%" on the same domains from other models. A heatmap of detectors by models and domains reveals what each was trained on — [RAID](https://arxiv.org/abs/2405.07940)
- **Attacks:** synonym swaps degrade Binoculars and other metric-based detectors by up to 36.1%. Homoglyphs drop five detectors by 40.6% on average (GPTZero only 0.3%). Some "attacks" *improve* detectors when the attacker model resembles the detector's training data (T5 paraphrase, BERT synonyms for RoBERTa-GPT2) — [RAID](https://arxiv.org/abs/2405.07940)
- **Metric choice:** few detectors can operate below 1% FPR. Some plateau at 16.9% (ZeroGPT), 0.88% (FastDetectGPT) and 0.62% (Originality). "Fixing and reporting FPR is crucial" — [RAID](https://arxiv.org/abs/2405.07940). AUC/F1-only reporting is "inadequate" — [Binoculars](https://arxiv.org/abs/2401.12070)
- **Benchmark overfitting:** RAID warns that practitioners will "specialize to the particular aspects of robustness we cover", and keeps a hidden test set and a split leaderboard (trained-on-RAID vs not) — [RAID](https://arxiv.org/abs/2405.07940)
- **Prompt confound ("capybara problem"):** prompt content changes perplexity regardless of author. Binoculars normalises with cross-perplexity. Memorised famous texts (e.g., the US Constitution) score as machine (3 of 11 famous texts on the machine side) — [Binoculars](https://arxiv.org/abs/2401.12070)
- **Non-native writers:** commercial detectors flagged TOEFL essays as AI 48-76% of the time (Liang et al. 2023, as cited). Binoculars' scores are unchanged by grammar-correcting ESL essays — [Binoculars](https://arxiv.org/abs/2401.12070)
- **Prompt-type difficulty:** text from "specified" prompts (with source details) is harder to detect than continuation or topical prompts — [MAGE](https://arxiv.org/abs/2305.13242)
- **Length and prompt artifacts in dataset construction:** Ghostbuster generated prompts from human texts when originals were missing, and length-matched outputs to avoid length shortcuts. Human reviewers were used to check for remaining artifacts. Highly expressive RoBERTa showed "catastrophic worst-case performance" under domain shift (up to 21.3 F1 drop), worse than perplexity-only — [Ghostbuster](https://arxiv.org/abs/2305.15047)
- **Synthetic-data contamination:** SFT on the same teacher data reduces separability (96.5% to 59.8%). Distilled or teacher-trained models get mislabelled as the teacher, e.g., Grok outputs labelled ChatGPT 82.8% of the time in leave-one-out — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Summarisation destroys most signal** (58.1% vs 97.8% chat), while paraphrase and translation keep >90%. Attacks that change content and structure, not wording, are the real threat to attribution — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Same-family hardest:** davinci-003 and ChatGPT are the hardest generators in M4GT-Bench — [M4GT-Bench](https://arxiv.org/abs/2402.11175). Same-release GPT variants dominate errors in code attribution — [LLM-NodeJS](https://arxiv.org/abs/2510.10493)

### Inferences
- **Validation design for Name That LLM:** use GroupKFold by prompt, so that no prompt appears in both train and validation folds. Otherwise topic and prompt features let the classifier "recognise the prompt" rather than the model, and CV accuracy will exceed the leaderboard. If the test prompts are unseen, group splitting is mandatory.
- **Length and format confounds:** check per-class length and markdown-count distributions. If one model's outputs are systematically longer or more formatted (as with ChatGPT vs Claude), a length/markdown-only GBDT baseline gives the confound's floor. Sun et al.'s markdown-only 73-78% shows how strong such a floor can be.
- **Truncation:** if test passages are excerpts (not whole responses), opening-token signals may be missing. Train with random-crop augmentation to avoid over-reliance on first-token cues.
- **Same-lab models:** expect the classifier to collapse sibling models toward the more frequent or "teacher" sibling. Monitor per-class recall (balanced accuracy penalises this directly) and consider per-family specialist heads or hierarchical classification (lab, then model).

### Gaps
- No source measured how attribution accuracy varies with passage length for 18+ classes. Sun et al.'s length curve covers 4-5 classes only.
- DetectGPT primary numbers were not fetched. It appears only via Binoculars/RAID comparisons (e.g., a DetectGPT misclassification of the capybara text).

---

## B3. Introspection techniques used in attribution research (and how to apply them)

### Takeaway
The standard toolkit is controlled text transformations that isolate one information level at a time, linear-model coefficient inspection on n-grams, first-token and markdown-element distributions, LLM-as-judge open-ended descriptions, leave-one-generator-out probing, and model-by-domain heatmaps.

### Cited Findings
- **Transformation ladder** (each step removes one information level): remove special characters (punctuation, markdown, whitespace), then shuffle words (keeps the bag of words), then shuffle letters (keeps character frequencies only), then markdown skeleton only, then paraphrase, translate, summarise with an external LLM. The resulting accuracies localise the signal mostly to word choice and content: 97.1, then 95.1, 88.9, 39.1 (letters), 73.1 (markdown only), 91.4 (paraphrase), 58.1 (summary) — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Lexical introspection:** TF-IDF uni/bigrams + LR coefficients give the top-10 phrases per model after removing the 20 most common shared words. Top-20 word-frequency plots and first-word distributions per model are also used — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Markdown counting:** per-response counts of bold, italic, headers, enumerations, bullets and code blocks, plotted as distributions per model — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **LLM-as-judge characterisation:** a judge compares paired responses to the same prompt repeatedly, then summarises traits. Example: ChatGPT is "descriptive and detailed", while Claude "prioritizes simplicity" — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Text-similarity diagnostics:** ROUGE-1 0.499 across LLMs vs 0.660 within an LLM. BERTScore 0.220 vs 0.482 — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Leave-one-model-out probing** to infer model similarity and teacher/distillation lineage — [Sun et al. 2025](https://arxiv.org/abs/2502.12150)
- **Heatmaps of detector accuracy by generator and domain** to infer each detector's training data — [RAID](https://arxiv.org/abs/2405.07940). Leave-one-domain-out and leave-one-model-set-out cross-validation — [MAGE](https://arxiv.org/abs/2305.13242); [M4GT-Bench](https://arxiv.org/abs/2402.11175)
- **Robustness to surface rewriting (code):** minified, mangled and deobfuscated variants test whether attribution rests on deep structure or on surface style — [LLM-NodeJS](https://arxiv.org/abs/2510.10493)
- **Feature-search interpretability:** Ghostbuster's selected features are explicit arithmetic combinations of per-token probabilities under named LMs, which makes them inspectable — [Ghostbuster](https://arxiv.org/abs/2305.15047)

### Inferences
- **Name That LLM checklist:** (1) 18×18 confusion matrix, sorted by lab; (2) retrain or evaluate on the transformation ladder to see whether our model leans on markdown or format (fragile if the test is stripped) or on word choice and content (robust); (3) TF-IDF-LR coefficients per *confused pair* (e.g., 4o vs 4o-mini) to find pair-specific discriminators, then hand-craft features or pair-specialist models; (4) leave-one-model-out to see which sibling absorbs each model; (5) prompt-grouped CV.
- Signals likely to separate same-lab siblings, consistent with sources: refusal/hedging phrases, opening tokens, markdown density, response length, and lexical preferences (e.g., "certainly", "overall"). Expect smaller models of a family to be confused with each other more than with the flagship (Qwen 14B vs 32B was the hardest pair at 63.1%).
- Ensembling a fine-tuned encoder with a sparse n-gram LR is well motivated. The LR captures discrete lexical markers that survive shuffling (88.9%), while the encoder captures semantics that survive paraphrase (91.4%).

### Gaps
- No source gave token-level attribution (integrated gradients/SHAP on transformer attribution classifiers) for model attribution specifically. Introspection in the sources is mainly transformation ablation plus linear coefficients.
