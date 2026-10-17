---
tags: [study, chemistry, casmi26]
---
# 08 Cheminformatics & mass spectrometry

Back to [[00 Study Guide Home]].

> **Core idea:** a molecule has many valid "spellings": SMILES strings, tautomers, stereo forms. The metric decides which spellings count as the same answer, so every candidate must be **normalised the way the metric does**.

## 1. Identifiers
| Identifier | What it is | Note |
|---|---|---|
| SMILES | a text graph of the molecule | not unique; canonicalise with RDKit |
| InChI | a layered, canonical identifier | standard InChI already merges some tautomers |
| InChIKey | a 27-character hash of the InChI | the first 14 characters (**InChIKey14**) encode the skeleton, without stereo or charge |
| Tautomer-canonical form | one chosen tautomer per family | RDKit `TautomerEnumerator().Canonicalize` |

![[tautomers.png]]

**Keto vs enol** forms of the same compound have different raw keys but the **same metric key**.
- If two candidates share a metric key, they are one answer. **De-duplicate** them before ranking, or the list wastes slots.
- The CASMI metric key is the tautomer-canonical, stereo-free InChIKey14.

## 2. Fingerprints and similarity
- **Morgan (ECFP) fingerprints:** radius 2, 2048 bits; the default for analog search.
- **Tanimoto similarity:** |A∩B| / |A∪B|. Above 0.7 is usually a close analog.
- Watch out: Tanimoto against small molecules is noisy, and a single changed bit matters a lot.

## 3. MS/MS basics
- **Precursor m/z:** the mass of the ion. With the adduct (e.g. [M+H]⁺) it gives the **formula candidates**, at a tolerance of about 5 ppm or 0.004 Da.
- **Fragments:** the peaks after collision. They show substructures.
- **Spectral library match:** cosine or entropy similarity against known spectra. This works for class 1/2, i.e. molecules with known spectra.
- **Class 3 (novel):** no library spectrum.
  - Generate candidates: analog parent + edit rules (`derive`) reaching the precursor mass.
  - Score them with predicted spectra (ICEBERG-style fragment prediction) or learned spectrum↔structure embeddings.
- **Typical edits:** +O (hydroxylation), +CH₂ (methylation), +C₆H₁₀O₅ (hexose), −H₂O, +SO₃, +C₂H₂O (acetylation).

## In our comps
- **CASMI:** MRR@25 on the metric key.
  - Classes 1/2/3, with class 3 novel (about 39–52% of the test).
  - Class-3 losses are mostly **parent selection**: `near_miss.py` shows the rulebook often reaches the answer from the oracle parent, but engine 1 didn't pick that parent.
  - gen20 widened the parent search.
  - Next lever: a better parent choice, e.g. spectrum-aware parent ranking instead of a structure-only one.

## Exercises
1. In RDKit: take acetylacetone keto and enol SMILES; compute the raw InChIKey14 and the tautomer-canonical InChIKey14 of both.
2. For 10 class-3 misses in `near_miss.csv`, look at `oracle_delta`. Which edits recur? Are they in `derive`'s rulebook?
3. Compute the Tanimoto between a molecule and its +O analog. How often is it above 0.7?

## Resources
- RDKit "Getting Started in Python" and the MolStandardize docs.
- Dührkop et al., *SIRIUS 4* (Nature Methods 2019): formula + fragmentation trees.
- Goldman et al., *ICEBERG* (2023/24): predicting MS/MS fragments.
- The CASMI challenge write-ups (casmi-contest.org).
