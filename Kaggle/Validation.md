---
tags: [kaggle, validation]
---
# Validation

Back to [[Kaggle Home]].

- **Build splits that match the test's hard part.** In CASMI26 that means molecules with **no public spectra**, answers reachable only through PubChem (S2). Library molecules (S1) flatter popularity priors.
- **Accept a change only if it helps the hardest split.** Gains on easy splits often vanish on the LB.
- **Expect the LB to move ~4x less than CV for re-ranking changes**, because about half the public molecules are unreachable novel structures.
- **Hold out structures, not spectra.** Match identities the way the metric does (InChIKey first block).
- **Calibrate class shares from several LB submissions.** Bull's estimate: ~28% class 1, ~20% class 2, ~52% class 3.
- A generated-candidate test with relatives still in the library is **optimistic**. Remove similar structures too (Tanimoto ≥ 0.7) to simulate truly novel molecules.

## Leakage / contamination audit (18 Oct)
**CASMI26**
- **Visible test.parquet = 1,213 spectra copied from train enveda-180**, so every molecule in it is a library hit. Never validate on it.
- Class 1 (test molecules with public spectra in train) is contamination by design. The hidden test mixes classes 1/2/3.
- Hidden test m/z has a one-electron correction (−0.000548 Da) that train/Zenodo lack. That's under 2 ppm and negligible.
- The np bench is leaky: all 250 np structures have about 170 other-library spectra each in train, so the models know them. **Use it only for structural questions** (can the generator make the answer?).
- The unseen bench (Zenodo Enveda-180) was checked against train by raw ik14 **and** the metric's tautomer key: 0 twins.
- The Zenodo data is used only for validation, never in a submission.
- Rule risks: no NIST data. The inherited Ahmed weights fall under the hosts' 'must be reproducible' rule, a lineage-wide risk.

**Name That LLM**
- Train and test prompts are disjoint by design. **CV must group by prompt.** A random split mixes topics across folds and misleads.
- `seyitkaangunes/ntl-extra-api-passages` is clean: 1,000 *new* prompts written by an LLM agent, deduplicated and filtered against the train topics. 'No competition texts, labels or prompts' were used; only the train length distribution was. Official Claude API. Use prompt_id as the CV group.
- Transductive steps (test clustering, Sinkhorn balancing) use unlabeled test text, which is allowed.
- Risks:
  - TabPFN weights are non-commercial (keep them out of final picks);
  - BALANCE_ALPHA 0.25 was found by public-LB tuning, and the private class mix differs (hedge with α 0.5).
