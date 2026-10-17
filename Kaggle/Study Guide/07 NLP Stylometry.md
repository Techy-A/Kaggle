---
tags: [study, nlp, stylometry]
---
# 07 NLP stylometry

Back to [[00 Study Guide Home]].

> **Core idea:** authors, LLMs included, leave fingerprints in **how** they write, not only in what they write: function words, punctuation, sentence length, favourite phrasings.

## 1. Classic stylometric features
| Feature | Example | Why it works |
|---|---|---|
| Char n-grams (2–5) | `", an"`, `"—"` | capture punctuation, morphology, spacing habits |
| Function-word frequencies | the, of, which, however | topic-independent |
| Punctuation profile | em dash, semicolons, Oxford comma, curly quotes | strong signal for LLM families |
| Sentence / paragraph length | mean, variance | verbosity style |
| Lexical richness | type-token ratio, hapax ratio | vocabulary habits |
| Formatting | bullet use, bold, headings | strong for chat models |

Model: TF-IDF features into logistic regression or a linear SVM. This is a strong, fast baseline and a good partner to an LLM in an ensemble.

## 2. Topic vs style: centering
All authors answer the same prompt, so the raw features mix **topic** (the prompt) with **style** (the author).
- **Centering:** subtract the prompt's mean feature vector from each passage. What's left is how this author differs from the others **on the same prompt**.
- This is the core of the 0.556 public notebook (the "centered stylegram").
- **Caveat:** centering needs several passages per prompt, and the hidden test has only about 2–3 per prompt.
  - Small groups make the mean noisy.
  - Idea to try: **size-weighted** centering, i.e. shrink the centering toward zero for small groups.

## 3. Hand rules
Hard-coded "punctuation rules" (e.g. em-dash density, typography that only one model family uses):
- can fix confident errors cheaply;
- overfit easily, so every rule needs a grouped-CV check.

## In our comps
- **NTL:**
  - 18 LLMs; metric balanced accuracy.
  - The 0.551 line uses TabPFN + punctuation rules; the 0.556 line uses centered stylegrams.
  - Future ideas: size-weighted centering, Gemini/GLM-specific gates, and the prvsiyan extended dataset (**licence check first**).

## Exercises
1. Build char 2–5-gram TF-IDF + logistic regression with grouped-by-prompt CV. Record the balanced accuracy with and without centering.
2. Plot the em-dash rate per LLM in train. Which models separate on this one feature?
3. Simulate the test's 2–3 passages per prompt in CV (subsample the groups). How much does centering's gain shrink?

## Resources
- Stamatatos, *A Survey of Modern Authorship Attribution Methods* (2009).
- Kaggle: *LLM – Detect AI Generated Text* (2024) winning write-ups.
