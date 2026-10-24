# Draft: Name That LLM discussion post

**Suggested title:** 0.569 → 0.639 in a week by copying carefully: what each public step bought, what failed, and what we think is left

*Draft for review. Replace the bracketed links before posting. Nothing here has been posted.*

---

## TL;DR

- We climbed from 0.559 to **0.639**, every step an **exact copy of a public notebook**. **We have no scored change of our own in this competition that beat its parent.** Our earlier attempts (adding a stylegram, a GLM gate) scored 0.550 and 0.555.
- We read the public lineage (55 notebooks, 134 scored versions) and think it shows which changes repeat and which don't.
- We are tied on the silver line (0.636) and see no clean path to gold from copying alone.

## Our progression

| Date | Copy of | Score |
|---|---|---|
| 18-19 Oct | public 0.559, 0.563, 0.569 | 0.559 / 0.561 / 0.569 |
| 19 Oct | lightningv08 two-stage LoRA + OLMo2 | 0.594 |
| 19 Oct | alexchilton 0.592 + 0.569 | 0.597 |
| 24 Oct | cosmorin reproduction of nishantdahal's part two | 0.622 |
| 24 Oct | fleongg *NTL Part Two Complement* | 0.636 |
| 24 Oct | kirilltym *teach more, keep the signature* | **0.639** |

Identical-code copies landed on the same displayed score in 11 of 14 cases and +0.001 in the other 3, so we treat ±0.001 as noise.

## What the public lineage shows

| Change type | What it bought | Repeats? |
|---|---|---|
| Bigger / cleaner training pool (35k → 84k) | +0.028, then +0.003, +0.002 | yes, the biggest lever |
| Two-stage recipe (stage 1 on public pool, stage 2 on train) | +0.023 once | one-off |
| Gaussian group heads | +0.006, +0.006, +0.006, +0.005, +0.004 | yes, about +0.005 each |
| Complementary stack (0.85 main + 0.15 stack) | up to +0.016; removing it −0.006 / −0.011 | yes |
| ATLAS at 10% on low-margin rows | +0.002 to +0.007 (−0.002 at 15%) | mostly |
| ModernBERT / desklib encoders | +0.002 to +0.004 on the 0.620 base; **0.634 on the 0.636** | no (already in the stack) |
| New large LLM members | early +0.005 / +0.006; **nothing above 0.61** | no |
| Reweighting, typography guards, Hungarian assignment | flat to negative | no |

One reading we found useful: the encoder members *hurt* the 0.636 (0.634) because desklib, UltraBERT, ELECTRA and DeBERTa are already inside its complementary stack, so they were counted twice.

## What was inside our notebooks

- All our submitted notebooks are **private exact copies** (credit cell on top, code unchanged). One technical note for anyone copying: if the original runs on the "latest" Kaggle environment, **do not pin the image digest**. We pinned `kaggle-images/python@sha256:17b2fc…`, which is Kaggle's CPU image, and got "Torch not compiled with CUDA" even with a GPU enabled.
- **The structure of the 0.636/0.639 line** (credited to nishantdahal, dhruvtibarewal, fleongg, kirilltym and others): the two-stage LoRA part-two ensemble (n-gram SVC, Qwen3-14B and Qwen2.5-14B LoRAs, DeBERTa-v3-large) at 0.85; a fixed 0.15 complementary stack; 10% ATLAS on rows whose top-2 margin is below 0.3; then Sinkhorn balancing. The 0.639 adds a 10% readout bank on the LoRA features.
- **Things we built but did not submit.** (1) Our own rebuild of the two encoder members (ModernBERT-large and desklib): out-of-fold 0.553 and 0.499 against nuridudak's 0.569 and 0.523, because we used a 75k pool instead of his 122k. We didn't submit it. (2) Our own re-implementation of ATLAS (shared LoRA with per-source up-projections on a frozen Qwen3-8B), verified against the public notebook's call but not trained at full scale.

## Open questions we would like help with

- **Licences.** The 0.639's readout bank dataset is "unknown" and is fit partly on 4,404 privately generated passages; the train-only version is 0.637. Rule 2.6 on privately generated data looks unsettled; has the host answered adisak's question?
- Has anyone put Gaussian group heads on the two 14B LoRAs of the 0.620 main? We can't find it in any public version, and it repeated about +0.005 four times.

## Would we reach gold?

Gold sits at roughly 0.653 and the top at 0.700. We think the gap from 0.639 to the top is **mostly data, not architecture**: the #1 poster writes that part of their pre-external-data 0.583 may have been probing, and #3 reports that removing an API passage set dropped validation 0.463 → 0.456. Our honest odds of gold by copying and small additions are **low**. The one concrete idea we plan to test is Gaussian group heads on the 14B LoRAs (public record: about +0.005 each time), which would still leave us short of 0.653.

## Credits

lucasboesen, kirilltym, prvsiyan, nishantdahal, dhruvtibarewal, fleongg, cosmorin, lightningv08, alexchilton, nuridudak, mrpicklerick, romantamrazov, josephadamski91 (ATLAS, CC BY-SA 4.0), pjmathematician.

*[Links: our notebooks, once made public]*
