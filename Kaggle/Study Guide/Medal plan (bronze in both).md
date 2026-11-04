# Medal plan: bronze in both (written 4 Nov 2026)

Goal: a bronze medal in NTL and in CASMI (Kaggle Competitions Expert). Medals come from the **private** leaderboard of the two selected submissions.

## Deadlines (Kaggle API, IST)
- CASMI: final submission Tue 15 Dec 2026 05:29 (14 Dec 23:59 UTC); team merger 7 Dec.
- NTL: final submission Fri 15 Jan 2027 05:29 (14 Jan 23:59 UTC).
- GPU quota: 30 h/week, resets Saturday 05:30 IST. About 6 quotas left for CASMI, 10 for NTL.

## Where we stand (4 Nov 00:43 IST)
| | score | rank | bronze line | silver | gold |
|---|---|---|---|---|---|
| NTL | 0.666 | 243 / 4,500 | 0.665 (r450) | 0.666 (r225) | 0.694 |
| CASMI | 0.479 | 478 / 4,664 | 0.479 (r466) | 0.480 (r233) | 0.485 |
CASMI is the one at risk. Lines rise about 0.001-0.002 a week as public notebooks improve.

## Standing rules
1. Daily scan of both competitions (notebooks + discussion). Copy and submit any public notebook whose author shows a real score above our best; forecast logged first. A score in the title is not a claim; read the text.
2. Never sit in a tie at the line for long: a tie loses on submission time.
3. Our own pieces before more copies: they are the only way above the crowd.
4. One change at a time on a scored base, forecast first, logged in the vault either way.
5. GPU: no single week spends more than 20 h on one competition; keep 5 h for copies of new public notebooks.
6. Final selections: two submissions that fail in different ways (best public copy + our strongest own variant), chosen in the last 3 days; shortlist kept in this file.

## CASMI (priority until 15 Dec)
| Week (Sat-Fri) | Work | GPU | Gate |
|---|---|---|---|
| 3-7 Nov | Train our own scorer (s2, 2.4 h); attach to the 0.480 copy so its two-model average actually runs (0.4 h). | 2.8 h | log shows "OURS BUNDLE READY"; submit; forecast 0.480 (0.476-0.484) |
| 8-14 Nov | If s2 helps: a third member (second seed, 2.4 h) in the same average. Track new public notebooks. | 3-6 h | only if s2 >= 0.480 |
| 15-28 Nov | Copy new public leaders; stack our scorers onto each new public base (the merge that worked). | 5-10 h/week | each new base scored verbatim first |
| 29 Nov-7 Dec | Decide on a team merger (deadline 7 Dec) if it lifts us. | - | - |
| 8-14 Dec | Shortlist; no new ideas after 10 Dec, only re-runs of the best. Select 2 by 13 Dec. | 3 h | two different failure modes |
Risk accepted: every CASMI submission above our own 0.465 uses third-party weights the host may remove; 0.465 cannot reach bronze, so we select the best public line and make as much of it reproducible as we can (our own scorer).

## NTL (to 15 Jan)
| Weeks | Work | GPU | Gate |
|---|---|---|---|
| 3-14 Nov | 0.667 copy result; diversity check of the q8pool member (CPU) against the 0.666 base. | 0.5 h | member adds >= 0.005 balanced accuracy on held-out folds (multi-cut recipe) |
| 15 Nov-6 Dec | If it passes: build the 0.666/0.667 base + our member as a variant; one submission per change. If not: second member on Open Pool v11 (different backbone or seed). | 8-15 h/week | paired bootstrap P >= 0.9 |
| 7 Dec-5 Jan | Keep up with public leaders; re-train our member on the newest pool version. | 5-10 h/week | - |
| 6-13 Jan | Shortlist; select 2 by 13 Jan. | 3 h | different families |

## Weekly check (every Saturday after the quota reset)
Fresh leaderboards, our ranks vs the three lines, GPU used, what changed, and which week of this plan we are in. If CASMI is below the bronze line for 7 days running, CASMI gets the next week's GPU first.

## Final-selection shortlist (update as scores land)
- CASMI: **0.480 (alexchilton 0.480 copy with his scorer, 57991848)**, 0.479 (baidalin midx copy, 57830973), 0.479 (bgunshui copy without second model, 57880522); tried: 0.474 (with our own scorer, 57961190) - dropped.
- NTL: **0.668 (omerfarukmerey 0.668 copy, pool v13, 57991450)**, 0.666 (sovereign copy, 57919709), 0.665 (chy663 copy, 57919589); tried: 0.663 (alexchilton 0.667 copy, 57957346).
