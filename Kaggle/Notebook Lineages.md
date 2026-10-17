---
updated: 2026-10-17
---
# Notebook lineages: how the public notebooks climbed, and why

Back to [[Kaggle Home]]. Sources: the LLM Council research (17 Oct), code diffs of the public notebooks, and the notebooks' own write-ups. Rule from [[What Works]]: **read the diff, not the title.**

## CASMI26: MS/MS spectrum → molecule (MRR@25)
All the strong public notebooks are one family tree. Each tier adds one layer that fixes a specific failure.

### The shared skeleton (all tiers)
1. **Engine 1** (Ahmed Berat Özer's v4n, on the v3.6 backbone). It predicts a molecular fingerprint from the spectra (FPNet), then pulls candidates of the right mass from a 711k pool (training structures plus COCONUT natural products). It ranks them with a GBM using fragment, analogue and derivation features, and it can also generate close relatives of known molecules.
2. **Engine 2** (prvsiyan's analog propagation plus a second ranker). It runs spectral similarity search against the training library, with candidates from COCONUT, the training structures and later ChEBI/LMSD. Two HistGradientBoosting rankers are blended 88/12.
3. **PubChem channel.** It looks up PubChem structures of the right mass that are not in the pool, scored by fingerprint fit plus a **popularity prior** (how often a molecule is cited or registered). A gate promotes them only when they beat the pool clearly (`rel` threshold).
4. **Forward simulators ICEBERG/GLACIER.** These predict the spectrum each top candidate *would* produce and re-rank by how well it matches the real spectrum.
5. **Fusion.** Reciprocal-rank fusion of the engines, then the simulator re-rank, then the final 25 guesses.

### ~0.41–0.43 tier: getting the layers right
Gluzdov's version table shows how each layer added up:

| Version | Change | Public LB |
|---|---|---|
| v13 | starting point | 0.341 |
| v17 | + PubChem + ICEBERG | 0.366 |
| v18 | + GLACIER | 0.373 |
| v19 | + deeper PubChem shortlist + popularity prior | 0.386 |
| v21 | + second engine, fused | 0.401 |
| v24 | GLACIER only on [M+H]+ | 0.406 |
| v31 | fragment score where ICEBERG is missing | 0.410 |
| v39 | tuned fusion weights | 0.415 |
| v40 | GLACIER alone in the final re-order | 0.422 |

- **Pool popularity re-rank:** +0.010. Common molecules are more likely answers.
- The 0.43s, such as Apex v180 at 0.428 and our 0.430/0.436 adaptation runs, sit here: the full skeleton, but none of the slot-saving tricks below.
- **Why they lose:** duplicate guesses waste slots, molecules the simulators can't handle get no re-rank, and the two engines are never hedged against each other.

### ~0.44 tier: stop wasting slots, cover the gaps
| Notebook | Change | Public LB | What it fixed |
|---|---|---|---|
| **E006** (canqiang) | **metric-identity fusion** + ICEBERG fallback | 0.440 | Candidates are merged by the *metric's own* identity (tautomer-canonical, stereo-free InChIKey14), so two drawings of the same molecule never take two slots. |
| **v7c** (Ahmed) | **MetFrag-lite "frag-gap"** | 0.441 | Molecules with no simulator coverage get a cheap fragmentation-based re-rank. |
| **E2R2** (antriksharya) | engine 2's top pick **inserted at rank 2** | 0.443 | A hedge: when the engines disagree, engine 2's answer still scores 1/2. |
| **rel > 150** (seyitkaangunes / Ahmed v11) | PubChem gate loosened from 250 to 150 | 0.446 | More PubChem-only answers reach the list. |

Variants that failed at this tier:

| Variant | Public LB |
|---|---|
| JESTR fusion | 0.418 |
| CFM-ID LOTUS reorder | 0.440 / 0.422 |
| stepped-CE GLACIER | 0.436 |
| MSEntropy | 0.440 |
| window 80 | 0.440 |
| Na GLACIER | 0.433 |
| per-polarity merged spectra | 0.424 |
| ungated FRIGID de novo at rank 2 | −0.009 on its base |

### ~0.45 tier: better coverage for engine 2
| Notebook | Change | Public LB |
|---|---|---|
| omerfarukmerey | v7c + **CC-BY ChEBI/LMSD catalogue** (98k extra structures for engine 2) + E2R2 keyed on metric identity | 0.445 |
| **V22** (easonyanyan = our pinned copy) | the above + rel>150, a one-line diff | **0.451** |
| ghazarosbarseghyan91 "v14seeds" / Gluzdov v51 | **engine-2 GBM max_depth 6→7**; the "seed bag" is dead code | +0.003 / +0.004 on their bases, 0.444 / 0.450 |
| saddamaditya / nursrijan | V22 + group-gap fallback | 0.451 (no gain) |

**Our own tests on V22 (17 Oct):**

| Variant | Public LB |
|---|---|
| rel100 | 0.445 |
| E2R3 | 0.450 |
| depth 7 | pending |

### Above 0.455 (leaders 0.49)
No public notebook gets there. The leaders built their own pipelines. MarvinTMB said no extra data is needed and put a realistic ceiling at about 0.55. The gap is almost certainly **class 3**: novel molecules that are in no database, an estimated 39–52% of the test. Every public notebook can only *retrieve*, never *invent*, those.

## Name That LLM: which of 18 models wrote this paragraph (balanced accuracy)
Key facts:
- Test prompts are **unseen**, and topic dominates the text, so every gain comes from isolating *style*.
- The class mix differs between public and private, so class balancing matters.

### The shared skeleton (Lucas Boesen's "v5" lineage, used by every top notebook)
A **log-probability blend** of many readers:
- style-aware n-grams + LinearSVC;
- stylometric features + LightGBM;
- **frozen LLM hidden states** (Qwen3-8B layers 7+22, Qwen2.5-7B layers 6+20, Qwen3-4B, GLM-4-9B) with logistic regression on top;
- frozen encoders (DeBERTa-v3-large, ELECTRA);
- the UltraBERT 5-fold ensemble.

Two transductive steps, which use the test set but no labels:
- standardization of test features;
- **Sinkhorn marginal balancing**, which nudges predicted class counts toward an even mix.

### ~0.53 tier: lightningv08 "Qwen LoRA ensemble" (0.531; our copy 0.532)
- 75% "Public D" (a cached frozen-feature ensemble plus a Gemma tail) and 25% own branch.
- The own branch adds a **fine-tuned Qwen3-8B LoRA**: 24 layers, trained 5-fold by prompt, with a plain head and a topic-centred head.
- Conditional decoding assumes distinct authors within recovered topic groups.

### ~0.54 tier: Lucas Boesen "LoRA × Adaptive Centering" v1 (0.541)
**Adaptive prompt-centering:**
1. Cluster the test passages by topic, using TF-IDF distance averaged with GLM-4-9B embedding distance, cut at 0.60.
2. Each cluster of 2–18 passages is probably one prompt answered by different models.
3. Subtract the cluster mean from the features, which removes the shared topic and leaves the style.
4. The centred "twin" heads are **added** as extra evidence; the plain heads are kept.

On their simulations this is worth +0.02 to +0.05.

Plus the LoRA plain readout (weight 0.282) and partial Sinkhorn balancing.

### ~0.55 tier
| Notebook | Change | Public LB |
|---|---|---|
| Lucas Boesen v3 | v1 + their own second LoRA run (seed 7) + learned token-attention heads (Qwen3.5, DeBERTa) | 0.549 |
| kirilltym | cohesion-aware centering | 0.550 |
| lightningv08 | v3 + **TabPFN** on PCA-compressed frozen features + "masked" style TF-IDF (letters → case pattern, content words → CONTENT) at 10% weight, plus two punctuation rules (curly quotes penalise Claude/GLM; ≥2 em dashes penalise Gemini) | 0.551 |
| **zbigniewkirchner ntl-v6 (our copy)** | v3 + a third LoRA with a **within-Claude 5-way head**, trained with extra public Claude-API passages, plus **Claude gates**: when the blend predicts *a* Claude, Claude-only specialists pick *which* one | **0.552** |

zbigniewkirchner's own version table:

| Version | Public LB |
|---|---|
| fork of 0.541 | 0.541 |
| + Claude gate | 0.544 |
| + own LoRA | 0.547 |
| + both gates on the 0.549 base | 0.552 |

**Why they differ:**
- 0.53 → 0.54: removing topic (centering). That's the biggest structural gain.
- 0.54 → 0.55: more and better fine-tuned readers, and fixing the most confused family (the Claude models).
- The newest gains are **targeted at specific confusions** (Claude vs Claude, Gemini dashes). That's where the next gains probably are too: find the class pairs the blend confuses most and add a specialist for each.
