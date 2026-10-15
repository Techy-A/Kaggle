---
tags: [kaggle, validation]
---
# Validation

Back to [[Home]].

- **Build splits that match the test's hard part.** In CASMI26 that means molecules with **no public spectra**, answers reachable only through PubChem (S2). Library molecules (S1) flatter popularity priors.
- **Accept a change only if it helps the hardest split.** Gains on easy splits often vanish on the LB.
- **Expect the LB to move ~4x less than CV for re-ranking changes**, because about half the public molecules are unreachable novel structures.
- **Hold out structures, not spectra.** Match identities the way the metric does (InChIKey first block).
- **Calibrate class shares from several LB submissions.** Bull's estimate: ~28% class 1, ~20% class 2, ~52% class 3.
- A generated-candidate test with relatives still in the library is **optimistic**. Remove similar structures too (Tanimoto ≥ 0.7) to simulate truly novel molecules.
