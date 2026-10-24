# ATLAS (josephadamski91/ntl-atlas-qwen3-8b-v1): rebuild spec

Reverse-engineered on 2026-10-24 from the dataset (downloaded into a scratch directory; code read as text, arrays and
configs read with `python -I` only; nothing of it executed) and from cell 24 of the 0.636 notebook
(`/home/user/work/push_ntl636`, fleongg/ntl-part-two-complement). Our rebuild: package `lab/atlas/` (`ntl_atlas`,
`ntl_rlcd`), trainer `lab/08_train_atlas.py`, CPU check `lab/atlas/check_atlas.py`, Colab notebook
`colab/ntl_atlas_train.ipynb`, notebook variant `/home/user/work/push_ntl636_ouratlas`.

**Licence update.** The dataset's Kaggle licence field still says "other", but since 2026-10-20 the package ships
`LICENSES.md`: the weights and docs are **CC BY-SA 4.0** and the five `.py` files are **Apache-2.0** (© Joseph Adamski).
Both allow commercial use. Their weights could therefore be eligible already, with attribution and share-alike. Rebuilding is
still the clean route, since ShareAlike and the competition's rules on external models are the user's call.

## Dataset layout

```
atlas_adapter/atlas_adapter/          # the dir holding atlas_bundle.json is the "bundle"
  atlas_bundle.json                   # labels, training_config, completed_steps=110, file sha256s
  adapters.safetensors                # 462.7 MB, 289 fp32 tensors, 115.68 M params
  checkpoint.json                     # step 110, elapsed 2108 s, reason "final", same config, 2026-10-17
  ntl_atlas/{__init__,model,inference}.py
  ntl_rlcd/{__init__,data}.py         # LABELS, plus an unused prompt/letters helper from an older "RLCD" project
  {training,inference}-source-manifest.json   # hashes; names ntl_atlas/train.py, which is NOT shipped
CREDITS.md LICENSES.md LICENSE-*.txt base-fingerprint.json   # base = Qwen/Qwen3-8B-Base @ 49e3418f
```

## Architecture

- **Base:** Qwen3-8B-Base, frozen, fp16, all 36 layers. No layer truncation, no pooling and no classifier head: the
  classifier is the LM's own next-token likelihood.
- **What the "18 source adapters" are:** they are not 18 separate LoRAs. In every layer, each of `q_proj, k_proj, v_proj, o_proj` (and
  no MLP projection) becomes `W x + (alpha/r) · B[s] · A · dropout(x)`, where:
  - `A` (16 × 4096) is **shared** by all sources;
  - `B[s]` is **per source**: (18, 4096, 16) for q and o, (18, 1024, 16) for k and v (8 KV heads × 128);
  - r = 16, alpha = 32 (scale 2), dropout 0.05 on the adapter input, adapter maths in fp32.

  The 18th parameter group is `control` (18, 1, 4096), one learned **prefix embedding per source**, prepended to the
  passage. Source s therefore means: route every row's adapters to `B[s]` and prepend `control[s]`. 36 × 4 × 2 + 1 = 289 tensors.
- **Init:** A uses Kaiming-uniform (a=√5). B[s] is N(0, 1e-4), not zero, so the sources differ slightly from step 0. control is
  the mean token embedding plus 0.01 · embedding-std noise.
- **Trained weights (read-only stats):** the per-source B std is about 1e-3 (10× its init). Measured as the source-mean share of
  B's energy, 20-70% of B is common to all sources, so B learns a shared adaptation as well as per-source ones. A moved slightly
  from init (std 0.0091 → 0.0097). control barely moved (pairwise spread 0.04 on a norm of 0.19).
- **Scored tokens:** the input is `[control[s], x_0 .. x_{L-1}]`. The outputs at x_0..x_{L-2} predict x_1..x_{L-1}, and the
  prefix's own output (which would predict x_0) is dropped. Padding goes on the right and is masked. No BOS (the Qwen3 tokenizer adds
  none) and no EOS. Truncation is at 512 tokens (train max is 381 tokens, mean 147, so it never triggers).
- **Readout ("argmax-prod"):** let `lp[s, t]` be the token log-prob under source s. Then `dis[k, t] = Σ_j max(lp[j,t] − lp[k,t], 0)` and
  `evidence[k] = mean over T ∈ {1,2,4,8} of mean over valid tokens of exp(−dis[k,t]/T)`. Probabilities are evidence clamped at
  1e-12 and normalised over the 18 sources, with no calibration. A token's best source has dis = 0, so each token votes
  softly for its argmax source, and the clipping limits how far any one token can push.

## Label order

`ntl_rlcd.data.LABELS = sorted(18 labels)`, which equals `atlas_bundle.json["labels"]` and the notebook's `LABELS` (also
`sorted(...)`). Order: claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5, gemini-3.1-pro-preview,
gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, glm-5.3, glm-5.3-flash, gpt-5.5, gpt-5.6-luna, gpt-5.6-sol,
gpt-5.6-terra, gpt-6-astra, kimi-k3, qwen3.8-max. The notebook still remaps by name (`out[:, [_alab.index(l) for l in LABELS]]`).

## Training recipe

What `training_config` and `checkpoint.json` state:

| | |
|---|---|
| data | the official train.csv only (2699 rows, sha256 40a22ce8…, which matches ours), all 18 labels, no external passages |
| epochs / batch | 5 epochs, effective batch 128 = micro 4 × 32 accumulation |
| steps | 110 = 5 × ceil(2699/128) = 5 × 22, so batching is per epoch with a partial last batch (no drop_last) |
| optimizer | lr 1e-4, weight decay 0.01, grad clip 1.0, warm-up ratio 0.05 |
| precision | fp16 base, fp32 adapters; gradient checkpointing (`use_reentrant=False`); the LM head runs in 64-token chunks, checkpointed |
| seed | 17 |
| hardware / time | one A100 (per inference manifest note), 2108 s for the 110 steps |
| loss | `model.loss`: next-token NLL under the row's **own** source, mean over valid tokens per row, then mean over rows |
| eval policy | "in-sample diagnostics establish training health only" (36 rows every 22 steps); **no validation numbers and no OOF** |

Our guesses, because `ntl_atlas/train.py` is not shipped:
1. **Loss = own-source NLL only.** The shipped `model.loss` does this, but the *training* `model.py` has a different hash
   (19d2a0… vs the inference 3fb939…; the manifest says the inference copy only adds cross-GPU transfers). A contrastive or
   other-source term from the paper can't be ruled out.
2. **Optimizer = AdamW** (betas default), weight decay on all trainable tensors, control included.
3. **Schedule:** linear warm-up for ceil(0.05 · 110) = 6 steps, then **cosine to 0**. Only the warm-up ratio is known.
4. **Loss weighting under accumulation:** each micro-batch is weighted by rows / batch rows, so a step is the mean over its
   rows. We also length-sort the rows inside each 128-row batch (same gradient, less padding).
5. **Shuffling:** a fresh permutation per epoch (numpy, seed + epoch). Theirs is unknown.
6. **fp16 loss scaling:** GradScaler with init scale 2^10. Unknown whether they used one.
7. **Control init std:** ours is the global within-dimension std of the embedding matrix, about what their formula gives.
8. **Training precision on T4:** a free T4 can't hold the fp16 8B (16.4 GB), so our Colab run uses an **NF4 base**
   (bitsandbytes, fp16 compute) for training and OOF scoring, while the notebook scores with the fp16 base on two T4s.
   It switches to fp16 automatically on a ≥ 30 GB GPU. The full unit stores 36 in-sample probe rows' NF4 probabilities, so an
   fp16 re-score of those rows checks the mismatch.

## Output contract

`model.score(ids, mask, source_chunk)` returns `(probs, raw)`, both (n, 18) in bundle-label order. `probs` rows sum to 1 and are
finite and non-negative. The notebook asserts `np.isfinite(out).all() and np.allclose(out.sum(1), 1, atol=1e-4)` and then
remaps to its LABELS. `inference.predict(texts, base_path, bundle_path, batch_size=1)` returns the same as float32 (n, 18).
Scoring costs 18 conditioned forward passes per row (`source_chunk` of them batched together).

## How the 0.636 notebook calls it (cell 24)

1. `P = 0.85 · softmax(main LOGIT) + 0.15 · softmax(complement)`, before Sinkhorn.
2. ATLAS rows are those with top-1 − top-2 margin of P below 0.3 (all rows if the test has under 50 rows).
3. `rglob('atlas_bundle.json')` under `/kaggle/input` must find **exactly one**. The base is the only `config.json` under a
   `/qwen-3/…/8b-base/` path (Kaggle model qwen-lm/qwen-3/Transformers/8b-base/1).
4. `atlas_scores(bundle, rows, budget)`:
   - `stage_checkpoint(bundle)` copies it to /kaggle/temp (≤ 3 GiB);
   - it puts the bundle on `sys.path`, purges `ntl_atlas*` / `ntl_rlcd*` modules, and imports `ntl_atlas.model.load_model` and
     `ntl_rlcd.data.LABELS`;
   - `cfg = bundle['training_config']`, then `load_model(base, LABELS, cfg, device='auto')`. That loads the fp16 base over both
     T4s with 3 GiB headroom each, and fails if any of it is off-GPU;
   - then `load_trainable(adapters.safetensors)` and `eval()`;
   - it tokenises with `max_length=cfg['max_length']` and scores rows shortest-first in batches of 4 with `source_chunk=3`.
5. **Time budget:** `_remaining = 32400 − (elapsed since OVERALL_START) − 600`, and `budget = (time.time() − T0) + _remaining`.
   Every 25 batches (100 rows) it projects the finish time token-proportionally × 1.1. It raises `atlas would overrun the time budget` if
   more than 50 rows are done and the projection exceeds the budget. A raise is fatal (no try/except): ATLAS is required.
6. **Probe stop:** `atlas_probe = 0` here, so it is off. When set, the same 100-row checkpoints raise `atlas probe stop` once
   `done ≥ probe` (a timing probe for test runs). Neither check runs between checkpoints.
7. `P[rows] = 0.9 · P[rows] + 0.1 · atlas`, then Sinkhorn (test ≥ 50), then argmax. `enhanced_probabilities.npz` keeps
   `atlas` and `atlas_rows`.

## Our rebuild

The package (`lab/atlas`, written from scratch; same module names and call signatures; the same tensor names
so either bundle loads in either loader) and the trainer (`08_train_atlas.py`):
- Units: `fold0..fold4` use GroupKFold(5) on prompt_id, the same folds as `07_train_encoders.py`, so OOFs line up for blending.
  `full` is all rows and writes `bundle/`. `merge` gives `oof_atlas.npz` and `summary_atlas.json`.
- Only `bundle/` contains an `atlas_bundle.json` (the notebook asserts exactly one), so fold adapters aren't kept.
- Resume every 10 steps. Diagnostics every 44 steps on a 36-row probe (held out for folds, in-sample for full).
- `atlas_bundle.json` keeps their keys (labels, training_config, completed_steps, files) and adds our provenance
  (train_base_quant, history, final probe probabilities).

## Comparison with theirs

Their bundle publishes no validation numbers or OOF, and it was trained on all 2699 rows, so scoring it "on our folds"
would be in-sample and would say nothing. We did not spend GPU on it. Honest comparisons:
1. Our OOF balanced accuracy (`summary_atlas.json`) stands alone. It is new information; theirs never had one.
2. The A/B is the 0.636 notebook with theirs (0.636) vs `push_ntl636_ouratlas` with ours. Everything else is identical, so
   the public difference is the ATLAS swap at 10% on uncertain rows. Forecast first: at 10% weight, the public delta is
   likely within ±0.003.
3. Agreement on test: both runs save `enhanced_probabilities.npz['atlas']`, so the two can be compared on the same
   `atlas_rows`.
4. Implementation parity (optional, GPU): load *their* adapters with *our* loader on Kaggle and compare with the 0.636 run's
   saved `atlas` column. Equal numbers would mean our model and readout code is exact.
