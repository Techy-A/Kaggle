# Encoder members ("OURS") of the 0.624-0.627 NTL notebooks: rebuild spec

Reverse-engineered on 2026-10-24 from these sources:
- `enc_infer.py` and the members cell in nuridudak's 0.624, 0.626 and typoguard notebooks, in mrpicklerick's 0.624 and 0.627
  notebooks, and in romantamrazov's "who am I" notebook. `enc_infer.py` and the members cell are identical in all six.
  The notebooks differ only in the `OURS` weights, `TYPO_GUARDS`, and mrpicklerick's shortcut for the public sample.
- The markdown of nuridudak/0-624-when-llm-decoders-meet-bert-encoders (its step 6).
- The **public** dataset nuridudak/ntl-two-stage-encoders (Apache-2.0, 1.46 GB). Its layout matches what the notebooks read
  from the now-private nuridudak/ntl-ours-v1. From it I read only `fusion.json`, the configs, the tokenizer configs and the
  keys of `light_members.npz`; no weights.

Training rebuild: `07_train_encoders.py`. Kaggle job: `jobs/ntl_encoder_rebuild/`.

## What the notebooks read

```
<any dir under /kaggle/input>/
  light_members.npz        # marker only: the notebooks use os.path.dirname(find("light_members.npz")) as OURS_ROOT
  modernbert/              # HF AutoModelForSequenceClassification dir
  desklib/                 # HF AutoModelForSequenceClassification dir
```

- `find()` does a recursive glob over `/kaggle/input` and takes the first sorted hit, so the dataset slug does not matter.
- No notebook reads the contents of `light_members.npz`. In nuridudak's dataset it holds his hashed n-gram member, which no
  notebook uses. Its keys:
  - `labels` (18, U22)
  - `svc4_{char,word,shape,func}_cols` (int32) and `svc4_{...}_idf` (float32)
  - `svc4_W` (1961549×18 f32) and `svc4_b` (18)
  - `probe_ids` (300) and `probe_svc4` (300×18)

  Our rebuild ships it with `labels` only. ponytail: the hashed n-gram member is not rebuilt; add it only if some notebook
  starts reading those keys.
- Each member dir is run in its own subprocess, one T4 each (`CUDA_VISIBLE_DEVICES = gpu index`), by `enc_infer.py`:
  - `AutoTokenizer.from_pretrained(dir)`, then `tok(texts, truncation=True, max_length=512)`. Special tokens are added
    (CLS/SEP). There is no prompt and no prefix.
  - `AutoModelForSequenceClassification.from_pretrained(dir, dtype=float32)`, under fp16 autocast, batch 64 with
    length-sorted dynamic padding. A batch with any non-finite logit is redone in fp32.
  - Labels come from `config.id2label[i]` for `i < num_labels`. The output is raw logits, then
    `softmax(logits[:, [lab.index(l) for l in LABELS]])`. The head's column order is free because it is remapped by name.
    The real heads use alphabetical order: claude-fable-5, claude-opus-4-6, claude-opus-4-8, claude-opus-5, claude-sonnet-5,
    gemini-3.1-pro-preview, gemini-3.6-flash, gemini-3.7-flash, gemini-3.8-flash, glm-5.3, glm-5.3-flash, gpt-5.5,
    gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra, gpt-6-astra, kimi-k3, qwen3.8-max.
  - The vote is `LOGIT += w * log(clip(P, 1e-9, 1))`. The weight scale is the same as the base's SVC 1.0, Qwen3-14B 0.4
    (+ twin 0.1), Qwen2.5-14B 0.2 (+ twin 0.1) and DeBERTa 0.25. Members are added **after** all base members and **before**
    Sinkhorn (300 iterations, equal class totals, test ≥ 50 rows). Optionally the typography guards run after that.
  - There is no centring or grouping for the encoder members. The centred twin applies to the LoRAs only, and Hungarian
    assignment is off.
  - Any failure (missing dir, non-zero exit, row-order mismatch) prints `SKIPPED` and leaves the base blend unchanged.
- Transformers: the configs were saved by 4.57.3 and use `dtype` (not `torch_dtype`), as does `from_pretrained(dtype=...)`.
  Train in the notebooks' pinned image (`gcr.io/kaggle-private-byod/python@sha256:9a6bcfda…`) so the checkpoints load there.

## The two members

| | `modernbert/` | `desklib/` |
|---|---|---|
| start | `answerdotai/ModernBERT-large` (Apache-2.0) | `desklib/ai-text-detector-v1.01` (MIT; DeBERTa-v3-large AI-text detector) |
| class | `ModernBertForSequenceClassification` | `DebertaV2ForSequenceClassification` |
| pooling / head | `classifier_pooling="mean"`, then the standard ModernBERT head (dense, GELU, LayerNorm) and `classifier` (1024→18, `classifier_bias=false`); all dropout 0 | standard DeBERTa `pooler` (CLS token, dense 1024, GELU, `pooler_dropout=0`) and `classifier` (1024→18); hidden/attention dropout 0.1 |
| conversion | none | desklib's encoder is stored under `model.*` and its head is 1-logit `classifier.*`. Rename `model.` to `deberta.`, drop `classifier.*`; the pooler and classifier start fresh |
| tokenizer | ModernBERT `PreTrainedTokenizerFast` (`tokenizer.json`), `[CLS] … [SEP]`, pad id 50283 | desklib's (`spm.model` + `tokenizer.json`, DebertaV2Tokenizer, `[MASK]` = 128000) |
| files | config.json, model.safetensors (bf16, 792 MB), tokenizer.json, tokenizer_config.json, special_tokens_map.json | config.json, model.safetensors (bf16, 870 MB), spm.model, tokenizer.json, tokenizer_config.json, special_tokens_map.json, added_tokens.json |
| config | `id2label` / `label2id` with the 18 labels alphabetical, `num_labels` 18 | same |

## Training recipe (as published by nuridudak; our deviations marked ⚠)

| | ModernBERT-large | desklib DeBERTa-v3-large |
|---|---|---|
| stage 1 data | 122,425 merged public passages (nuridudak/ntl-130k-clean-passages). ⚠ We use nishantdahal's `pool_v4.parquet` (75,783), minus pool passages with content TF-IDF cosine ≥ 0.3 to a train passage (Nishant's leakage filter), because the rules allow only pools the notebooks already attach | same |
| stage 1 | 2 epochs, lr 2e-5, batch 32, **class-weighted** CE | 2 epochs, lr 2.8e-5, batch 64, class-weighted CE, gradient checkpointing |
| stage 2 | 2 epochs on the official train, lr 1e-5, batch 32 | same |
| both | AdamW (wd 0.01), 10 % linear warm-up then cosine, grad clip 1.0, bf16 autocast (⚠ fp16 + GradScaler on T4), max 512 tokens, length-grouped batches | |
| reported | stage 1 only: 0.542 on train. Prompt-grouped 5-fold OOF after stage 2: **0.569** | 0.522 / **0.523** |

Assumed where the write-up is silent (⚠):
- Stage 2 uses plain (unweighted) CE, since train is balanced.
- There is one learning rate for the whole model.
- The CV is GroupKFold(5) on `prompt_id`: stage 1 runs once, then each fold runs stage 2 from the stage-1 weights. The
  shipped model is stage 2 on the full train.
- The seed is 42.

## OOF fusion in nuridudak's own dataset (not used by the notebooks)

`fusion.json`: modernbert 0.4419, desklib 0.2851, svc4 0.3345, `hung_min` 2, `typo_guards` true, OOF NLL 1.0492, blend
0.592. These are his encoder-only blend weights, not the weights the notebooks use.

## Weights the public notebooks use (`OURS`)

| notebook | modernbert | desklib | guards | public |
|---|---|---|---|---|
| nuridudak 0.624 | 0.25 | 0.10 | on | 0.624 |
| nuridudak 0.626 v2 | 0.40 | 0.10 | on | 0.626 |
| mrpicklerick 0.624 | 0.25 | 0.10 | off | 0.624 |
| mrpicklerick 0.627 ("ratio 6 to 1") | 0.30 | 0.05 | off | 0.627 |
| romantamrazov "who am I" | 0.30 | 0.05 | off | — |

The 0.627 notebook also writes a fallback submission when test has fewer than 50 rows, which skips **every** member. Its
commit run therefore never exercises the encoders. They load only in the hidden scoring run.
