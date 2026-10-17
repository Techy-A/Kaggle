# Grandmaster Workflow, Iteration-Speed and Compute/Runtime Hacks for Kaggle Code Competitions

Scope note: these notes are about workflow and compute tricks only. Validation design, ablation-table style, pseudo-labeling, distillation and the theory of final-pick selection are already in `../Building and introspecting great models/kaggle_grandmaster_practice.md` and are not repeated here, except where a compute number matters. I pulled the primary write-ups as raw markdown through Kaggle's discussion API: LMSYS 1st/2nd/3rd/9th, Eedi 1st/2nd/3rd/4th, LLM 20 Questions 1st, and DAIGT 1st/5th. Other sources are the HF NuminaMath (AIMO-1) blog, the NVIDIA NemoSkills AIMO-2 paper, NVIDIA KGMoN blogs, the vLLM docs, the XGBoost docs and the official Kaggle API docs. The target team is 2 people with about 30 Kaggle GPU-hours a week (T4 x2 / P100) and 5 submissions a day. Applicability to **(1) CASMI 2026** (CPU-heavy RDKit/PubChem candidate generation + GBDT reranking) and **(2) Name That LLM** (LoRA LLM, 18-way classification on T4s) is marked **[CASMI]** / **[NTL]**.

---

## Q1. Iteration speed: subset/single-fold experiments, caching, GPU dataframes and GBDTs, logging, config-driven runs, parallel runs, Kaggle API pushes, offline harnesses

### Takeaway
KGMoN put it plainly: "the biggest lever ... is the number of high-quality experiments we can run," so they optimize the whole pipeline for speed and not just the training step. The concrete levers with published numbers:
- GPU dataframes: cuDF-pandas is about 150x on a 5 GB join/groupby benchmark, and it let Deotte test more than 10k features in days.
- GPU GBDTs and cuML, so many baselines and stacks finish in hours rather than days.
- Developing on one fixed fold. Eedi 1st trained on folds 1-4 and validated on fold 0 throughout development.
- Caching every expensive intermediate: features, OOFs, LLM-generated candidate text.
- Avoiding padding: sequence packing or varlen attention.
- Pushing and running notebooks headlessly with `kaggle kernels push`, so several runs can be in flight at once.

### Cited Findings

**Principle**
- KGMoN Playbook: "The biggest lever we have in any competition or real-world project is the number of high-quality experiments we can run ... In practice, that means we optimize our entire pipeline for speed, not just our model training step." Their prescriptions are GPU drop-ins for pandas/Polars, and cuML or the GPU backends of XGBoost/LightGBM/CatBoost. With these, a diverse baseline sweep gives "better insight in minutes, not hours"; stacks run "an order of magnitude faster ... in hours instead of days"; and "several pseudo-labeling cycles in hours" replace "days on a CPU". — [NVIDIA KGMoN Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
- Deotte's 4 tips include: "accelerate your experimental pipeline with GPU using accelerators like cuML or NVIDIA cuDF in order to run experiments as fast as possible." His loop is baseline → local validation → new idea → "We repeat this process as fast as possible!" — [NVIDIA: GMs Unveil Winning Strategies](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- Ruchi Bhatia: "Use lightweight scripts for debugging, heavy models only for final runs. Save intermediate features and predictions. Treat it like a production ML pipeline." — [NVIDIA: GMs Unveil Winning Strategies](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)

**GPU dataframes and GBDT numbers**
- cuDF-pandas (`%load_ext cudf.pandas`, zero code change): on the DuckDB database-like ops benchmark with 5 GB of data, "minutes of processing into just 1 or 2 seconds", about 150x. It falls back to CPU automatically for unsupported ops. — [NVIDIA: cuDF accelerates pandas nearly 150x](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/)
- Deotte, Playground S5E2 (backpack prices, 1st place): cuDF-pandas let him "rapidly generate and test over 10,000 engineered features", and the best 500 went into XGBoost. "In a drastically reduced timeframe – days instead of potential months". The main pattern was a brute-force search over `groupby(COL1)[COL2].agg(STAT)` (mean/std/count/min/max/nunique/skew, quantiles, histogram bins), with nested CV whenever COL2 is the target. "The first place result ... wasn't just about finding the right features – it was about having the speed to find them." — [NVIDIA: GM Pro Tip, cuDF-pandas feature engineering](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)
- Overnight automated feature search: random groups of 2-6 of 23 columns, target-encoded with cuDF, keeping a combination only if validation improves. Deotte: "I recently won a competition doing just that". — [SED 1817 transcript](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)
- Hill climbing on GPU: "With CuPy on GPUs, we can vectorize metric calculations (like RMSE or AUC) and evaluate thousands of weight combinations in parallel." — [KGMoN Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
- XGBoost GPU recipe: `device="cuda"`, `tree_method="hist"` (select a GPU with `cuda:<ordinal>`). Data is stored on device in compressed ELLPACK format, and the docs recommend building `xgboost.QuantileDMatrix` first to save memory. Current docs list CUDA >= 12.9 for the prebuilt GPU wheels. — [XGBoost GPU docs](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- DAIGT 1st, Ghostbuster branch: "Rapids was used to train significantly faster than sklearn" (SVM + RF on 100 token-probability features). — [DAIGT 1st](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295)

**Single fold, fixed folds, full fit only at the end**
- Eedi 1st: "During development, I always trained my models on folds 1-4 and validated on fold 0 ... The final models were trained on the entire dataset (full fit) using the same hyper parameters from the development stage." — [Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)
- Eedi 4th spent 10 hours per fold on an A100 to train a 32B retriever. Because inference took about 50 min per fold, "only 2 folds were used for the final submission." — [Eedi 4th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559)

**Removing wasted compute in LLM training**
- LMSYS 2nd (tascj): `flash_attn_varlen_func` plus a custom collator that concatenates samples and builds `cu_seqlens`, so "attention_mask and padding are unnecessary", with "no padding from start to finish". They added `RMSNorm` and `FusedRoPEFunc` from `transformer_engine`. Full-parameter training of a 7B model fits on 1x A100-80G (9B needs 2) using BF16 plus an optimizer with Kahan summation. — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
- NuminaMath (AIMO-1) used TRL `SFTTrainer` "packing" into 2048-token chunks with gradient checkpointing. 10 h on 8x H100. — [HF: How NuminaMath won AIMO-1](https://huggingface.co/blog/winning-aimo-progress-prize)
- DAIGT 5th: mamba-790m was "roughly as fast as deberta-v3-large and consumes less GPU memory during training despite having over 2x as many parameters". A 1D-conv ResNet took about 10 min to score on the LB without a GPU but topped out at about 0.87. — [DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)

**Caching and precomputing expensive LLM outputs offline**
- Eedi 2nd: LLM-written explanations for every candidate misconception were generated offline. "Since we don't need to run inference for misconception in submission, the approach costs nothing in submission". It added about 2-4% to the retriever. — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- LLM 20 Questions 1st precomputed a 35,000 × 13,000 (≈455M-cell) p(keyword, question) table with vLLM, using Yes/No next-token probabilities from 3 LLMs. He started on a local 4090 and rented 8x RTX 4090 on Runpod for the last week, about $500 total. The submission was then mostly a table lookup. — [LLM 20Q 1st](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106)
- LMSYS 3rd generated 500k pseudo-label responses with vLLM. "continuous batching support from vLLM provided a lot more throughput" than ctranslate2. They grouped conversations by number of turns rather than token length, because "output tokens are much slower than prompt tokens". — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)

**Headless runs with the Kaggle API (parallel experiments, version-pinned submissions)**
- `kaggle kernels init -p DIR` creates `kernel-metadata.json`. `kaggle kernels push -p DIR [--accelerator NvidiaTeslaT4]` uploads and runs the notebook, and `--no-run` saves a version without running it. `kaggle kernels output owner/slug[/version] -p out --file-pattern ...` downloads outputs, and `kaggle kernels pull -m` gets code plus metadata. The metadata fields include `enable_gpu`, `enable_internet`, `machine_shape`, `dataset_sources`, `competition_sources`, `kernel_sources` and `model_sources`. Accelerators listed as of Sep 2026: NvidiaTeslaT4 (T4 ×2), NvidiaL4, NvidiaL4X1, NvidiaTeslaA100, NvidiaH100, NvidiaRtxPro6000, TpuV5E8 and TpuV6E8, "Some of these are only available to participants of specific competitions". — [Kaggle API kernels.md](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/kernels.md); [kernels_metadata.md](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/kernels_metadata.md)
- You submit a code-competition notebook version from the CLI with `kaggle competitions submit <comp> -k <user>/<slug> -f submission.csv -v <version> -m "<msg>"`. The message field is a free submission log. — [Kaggle API tutorials.md](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/tutorials.md)
- An independent "autoresearch for Kaggle" write-up built a local pre-screen of about 40 s before committing to a full Kaggle run (secondary, not a GM source). — [noqta.tn](https://noqta.tn/en/tutorials/autoresearch-kaggle-autonomous-ai-competition-2026)

**Platform budget facts (verify in your own account settings)**
- Session limits are reported as 12 h for CPU/GPU and 9 h for TPU. The weekly GPU quota is not fixed in the docs and is commonly about 30 h, with historical reports of 30-41 h. Linking Colab Pro/Pro+ reportedly adds 15/30 GPU-h a week. — search summary of Kaggle notebook docs via [gpuperhour.com](https://gpuperhour.com/blog/free-cloud-gpus-and-credits) and [aimultiple](https://aimultiple.com/free-cloud-gpu) (secondary. I could not render kaggle.com/docs/notebooks, which is client-side.)

### Inferences
- **Recipe: fast-loop ladder.** (a) Smoke test on 1-5% of the data in a CPU session (free of GPU quota). (b) One fixed fold (fold 0, stored as a `fold` column in a dataset so every experiment shares the same split). (c) Full k-fold only for changes that win on fold 0. (d) Full fit plus seeds only for submission candidates. This is Eedi 1st's habit plus Bhatia's "lightweight scripts for debugging".
- **Recipe: config-driven runs.** One `train.py` reads `cfg.yaml`. The Kaggle notebook is a thin wrapper: `!python train.py --cfg cfgs/exp042.yaml`, writing `oof_exp042.parquet`, `metrics.json` and the cfg to `/kaggle/working`. Push N copies with different cfgs using `kaggle kernels push`, which runs them concurrently and does not tie up a browser session. Pull results with `kaggle kernels output` and append them to a single CSV log (exp id, git hash, cfg diff, fold-0 score, full CV, runtime, GPU-h). Kaggle limits how many sessions run at once and the cap varies, so check your account. If you hit the cap, queue the runs.
- **[CASMI]** The bottleneck is CPU (RDKit candidate enumeration, fingerprints, PubChem lookups), not GPU. Highest-ROI hacks:
  - Run candidate generation once per formula/mass bucket and cache it to a Kaggle Dataset as parquet: candidate SMILES, InChIKey-14, precomputed fingerprints, descriptors. Every reranker experiment then just loads it. This is Eedi 2nd's "costs nothing in submission" principle applied to the candidate side.
  - Parallelize RDKit with `multiprocessing`/`joblib` over all vCPUs (core count is unverified, see Gaps), with chunked SMILES lists. Store fingerprints as packed uint8/bit arrays.
  - Use GPU XGBoost (`device="cuda"`, `QuantileDMatrix`), or LightGBM/XGBoost with ranking objectives, for the reranker, so fold sweeps take minutes. Use cuDF for groupby feature generation over the query×candidate table (Deotte's groupby-agg search maps directly onto per-query candidate statistics: rank of mass error within query, z-score of spectral similarity within query, etc.).
  - Use cuML or GPU kNN for spectral/embedding nearest-neighbour candidate channels.
- **[NTL]** Iteration cost is LoRA fine-tuning time:
  - Develop on fold 0 with a shorter max_len and a smaller backbone first, e.g. a 1-3B model, to rank ideas. Only promote winners to the 7-9B model.
  - Use packing (TRL `packing=True`, or varlen attention) to remove pad waste.
  - Cache tokenized datasets.
  - Run cheap ideas such as perplexity/GLTR features from small reference LMs (DAIGT) as GBDT experiments on cached per-token logprobs. These cost seconds per iteration.

### Gaps
- I found no Kaggle post by Chris Deotte specifically prescribing "subset first / single fold first". The single-fold practice is documented here via Eedi 1st and the "lightweight scripts" advice via Bhatia.
- I found no authoritative source for the current Kaggle concurrent-session cap, the CPU core count or RAM of T4x2 sessions, or the hidden-rerun environment flag. A search for `KAGGLE_IS_COMPETITION_RERUN` turned up nothing authoritative.
- I found no published speedup numbers for LightGBM/CatBoost GPU on Kaggle hardware, nor any GM-specific experiment trackers (W&B vs spreadsheet). The prior note has the same gap.

---

## Q2. Inference within Kaggle code-competition limits: vLLM on T4, quantization, length sorting and dynamic batching, both T4s, KV/prefix caching, distillation, precomputation, ONNX/TensorRT, time guards, offline wheels

### Takeaway
On 2x T4 (sm75, no bf16, no FlashAttention-2), the winning recipes are:
- Fit a 7-9B model with fp16 weights split across both GPUs, or quantize to **8-bit**. Several LMSYS top teams found 8-bit *faster* than 4-bit with no score loss.
- Sort by length and batch dynamically.
- Distill large teachers into a 9B model, merge LoRA before quantizing, and calibrate GPTQ/AWQ on task data.
- Use vLLM continuous batching plus prefix caching (about 10% saved in Eedi 2nd).
- Spend leftover time on TTA, and spend models selectively on the subset where they matter.
- Hard time guards with a shared time buffer (AIMO-2) and early stopping once answers agree.
- vLLM needs compute capability ≥7.5, so a **P100 cannot run vLLM**. Use T4x2 for LLM inference.

### Cited Findings

**Hardware constraints on T4**
- NuminaMath (AIMO-1; each submission got a P100 or 2xT4 and up to 9 h for 50 problems): "standard methods like Flash Attention 2 or torch.compile do not work on T4 GPUs. Similarly, modern data types like bfloat16 are not supported, which prompted us to explore post-training quantization methods like AWQ and GPTQ." Casting to fp16 degraded their model, and fp32 did not fit. "a 16-bit model consumes approximately 32GB VRAM just to load the weights" (7B with headroom), so they quantized to **8-bit with AutoGPTQ using the training data for calibration**: "a small drop in accuracy but ... the best compromise". A static KV cache plus torch.compile gave 2-3x on H100 but "cryptic errors on the Kaggle T4s". Larger models (20B, 33B, Mixtral) were "very slow on 2xT4" with "mysterious timeouts". — [HF NuminaMath blog](https://huggingface.co/blog/winning-aimo-progress-prize)
- vLLM GPU requirement: "compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, L4, H100 ...)". — [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). (The P100 is sm60, so current vLLM does not support it. That sm value is my own knowledge, not from this source.)
- vLLM quantization support on **Turing (T4)**: AWQ ✅, GPTQ ✅, Marlin kernels ✅* (asterisk, limited), llm-compressor INT8 W8A8 ✅, bitsandbytes ✅, GGUF ✅. **FP8 W8A8 ❌ on Turing** (it needs Ada/Hopper). — [vLLM quantization supported hardware](https://docs.vllm.ai/en/latest/features/quantization/)

**8-bit vs 4-bit, merging, and model splitting across 2 GPUs**
- LMSYS 1st: merge (average) the 5-fold LoRAs, then "Quantize to 8-bit using GPTQ and use TTA (length 2000) during submission". Distilled Gemma2-9B, private 0.96898 (log loss). — [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
- LMSYS 9th: "use 8bit quantization model instead of 4bit", "merge_and_unload()", "In the case of 8-bit quantization, no decrease in score was observed", and "Dynamically change batch size based on token length". — [LMSYS 9th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704)
- LMSYS 3rd: "I wish we had figured out that running the models in 8bit was faster, because there's a chance we could have put more effort into Gemma2 and submitted it using proper TTA." Their final submission was 2x Gemma2-9B, "sorted by length", with 4k and 3k max length. — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
- LMSYS 2nd: "T4x2 is sufficient to run 7b-9b models in fp16. Transformers can be almost evenly distributed across 2 GPUs, just needing slight code modifications to make executions on two GPUs pipelined." On sm75 they used Triton kernels: `context_attention_fwd` (lightllm, with logit softcapping added), `rms_norm`/`fused_rotary_emb` (lmdeploy), and fused GELU/SiLU-and-mul (lightllm). They used xformers `memory_efficient_attention` for Llama3. Inference also used sequence collate with no padding. They turned off Gemma-2 sliding-window attention because they were unsure sm75 kernels supported it (max input 4340 tokens). — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
- Selective inference to fit the budget (LMSYS 9th): "it was difficult to infer two gemma2 models and one llama3 model, so the llama3 model was only inferred for samples with token counts of 512 or less and 1500 or more", i.e. the third model ran only on the length buckets where it helped. — [LMSYS 9th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704)
- AIMO-2 1st (NemoSkills; 4x L4, 5 h, 50 problems answered one at a time), measured on 4x L4 with maj@12:

  | Format | tok/s | AIME24 | AIME25 |
  |---|---|---|---|
  | BF16 | 210 | 82.7 | 66.7 |
  | W8A16 int8 | 315 | 82.7 | 66.7 |
  | W4A16 int4 | 436 | 72.7 | 60.7 |
  | FP8 | 310 | 83.3 | 68.7 |
  | FP8 + ReDrafter speculative decoding | 554 | 81.3 | 71.3 |

  8-bit is a free 1.5x, while 4-bit cost about 10 points. — [NemoSkills AIMO-2 paper, arXiv 2504.16891](https://arxiv.org/html/2504.16891)

**Quantization quality: calibration and tooling**
- Eedi 1st quantized every fine-tuned ranker with AutoAWQ, using "task specific calibration datasets for each model to mitigate performance degradation". He included an ablation on a 32B reranker (the figure is an image). His upfront hypothesis: "Quantization method and the calibration datasets could be important. We may need to add recovery adapters to the quantized models". — [Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)
- Eedi 2nd used intel/auto-round: "easier to use and caused minimal accuracy loss (typically less than 2%)" vs AutoGPTQ/AutoAWQ, with output loadable in vLLM. Calibration used the training set. Qwen2.5-72B's intermediate_size 29568 broke multi-GPU quantized inference, so they padded it to 29696 following the Qwen GPTQ troubleshooting doc. — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- DAIGT 5th (offline data generation): "Used 4-bit AWQ quantization to conserve GPU memory for any model with >= 13 billion parameters" in vLLM, and randomized sampling params per request. — [DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093)

**vLLM tricks: prefix caching, logprob classification heads**
- Eedi 2nd: "By setting the `enabling_prefix_cache` to `True`, we were able to save approximately 10% of the inference time." (The vLLM arg is `enable_prefix_caching`.) — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- vLLM Automatic Prefix Caching "caches the KV cache of existing queries, so that a new query can directly reuse the KV cache if it shares the same prefix with one of the existing queries, allowing the new query to skip the computation of the shared part." — [vLLM APC docs](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- Eedi 2nd turned a generative LLM into a classifier inside vLLM: "We implemented the reranker using `logits_processors` and `logprobs` by assigning a weight of +100 to specific tokens ... We also tried using a classification head directly, but the results were not satisfactory, and it was not easy to perform inference with vLLM." — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- Eedi 4th modified vLLM to compute embeddings for retrieval at submission time. — [Eedi 4th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559)

**Spend the budget where it pays: time–accuracy tables**
- Eedi 2nd's retriever table, with inference time per model: Linq-Embed-Mistral 50 min (private 0.461); Qwen2.5-14B 45 min (0.485); Qwen2.5-32B 140 min (0.495); QwQ-32B 140 min (0.500). "Our best submission used an ensemble of Mistral and 2 x qwen2.5-14B to give enough time to 72b rerank". CoT was dropped from one pipeline "due to inference time", since for 14B/32B "adding CoT to the prompt will double the inference time". Sliding-window reranking repeated over the top 10 beat reranking more candidates. — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- Eedi 1st cascade: the cheap 14B model scores all 32-64 candidates, the 32B scores the top 8, and the 72B ranks only the top 5. Listwise shuffle-TTA was dropped: "I was unable to run TTA during inference due to time constraints." — [Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)
- Large models on T4: the Eedi 2nd write-up links a public Eedi thread, "Qwen2.5-72B & Llama 3.3-70B on 2xT4 (questionable performance)", as the starting point that made 72B rerankers feasible on Kaggle. — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651) (thread 550223; I could not retrieve its body)
- Last-day risk: Eedi 2nd's Athene-V2 submission "got timeout due to gpu and timeout issues" but showed 0.609 on the LB. — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)

**Time guards and early stopping**
- AIMO-2 1st: "allocating 350 seconds per question as the base time limit. If a question completed early, the unused time was added to a shared buffer. The next question could then draw up to 210 extra seconds from this buffer" (max 560 s). They used async batched generation with early stopping: "in a batch of 16, if the first 4-5 completions agreed on the final answer, we canceled the remaining generations". Stragglers were cut by "canceling the last n pending requests once the rest finished". The TensorRT-LLM in-flight batching "releas[es] each sample as soon as it completes". Their local evals used fixed token budgets, but the final models wrote longer answers, which interacted badly with the time-management logic. — [NemoSkills AIMO-2](https://arxiv.org/html/2504.16891)
- AIMO-1 (Numina): "The evaluation API provides problems in random order, so tactics like early stopping produce high variance". They chose N=48 candidates × M=4 depth because "Increasing either parameter did not improve performance, so we took a conservative approach to stay within the time limit." — [HF NuminaMath blog](https://huggingface.co/blog/winning-aimo-progress-prize)

**Artifacts: models and adapters as Kaggle datasets, frozen**
- LMSYS 1st: "I had one sub that also failed to run because I deleted another model that I had uploaded." — [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
- Code-competition submissions re-run the whole notebook against hidden data, and if allowed you attach the trained model as a data source instead of training on the fly. — [fast.ai forum: Kaggle Code Competitions](https://forums.fast.ai/t/kaggle-code-competitions/103660) (secondary)
- Offline install pattern: download wheels on a connected machine and `pip install --no-index --find-links <dir>`. vLLM recommends pinning by full wheel URL or using uv, because pip mixes indexes. — [zenn.dev](https://zenn.dev/yuto_mo/articles/08f104b4eff503); [vLLM install docs](https://docs.vllm.ai/en/v0.15.0/getting_started/installation/gpu/) (I found no Kaggle-official doc on this pattern)

### Inferences
- **Recipe: [NTL] T4x2 inference stack.**
  1. Train LoRA, then `merge_and_unload()`.
  2. Quantize the merged model to 8-bit (GPTQ W8 or llm-compressor INT8 W8A8, both Turing-supported in vLLM), calibrated on ~256-512 *training passages*, not generic text. Upload it as a private Kaggle Model or Dataset and pin the version.
  3. Use vLLM with `tensor_parallel_size=2` (or two independent replicas, one per GPU, `CUDA_VISIBLE_DEVICES=0/1`, each taking half the sorted test set; for a ≤9B model that fits one T4 in 8-bit, data-parallel replicas usually beat TP on PCIe T4s, but this is unverified, so benchmark both), `dtype="half"`, `enable_prefix_caching=True` (the shared instruction prefix across all passages), and `max_model_len` set to the passage-length P99.
  4. For 18-way classification, read the next-token `logprobs` for 18 single-token labels (Eedi 2nd's logits-processor trick) instead of generating text. That needs only one prefill per passage, so decoding cost is roughly zero.
  5. Alternative: HF `AutoModelForSequenceClassification` with LoRA, sorted by length, with dynamic batch size (LMSYS 9th) and the model split across both GPUs (LMSYS 2nd).
  - Budget: a few thousand passages ≤1-2k tokens with prefill-only 8-bit 9B fits easily in 9 h, which leaves room for a second model or TTA. Reserve TTA for inputs near the decision boundary.
- **Recipe: time guard.** At notebook start, `T0=time.time()`. Before each optional stage (TTA, second model), check `time.time()-T0 < BUDGET - margin_est`, where margin_est comes from a timing run on a public test-sized dummy set × (hidden/public size ratio). Always write a valid `submission.csv` from the cheapest model first, then overwrite it as better stages finish. A crash or timeout late in the run still leaves a scored file only if the notebook exits cleanly, so wrap stages in try/except.
- **[CASMI]** Precompute everything that does not depend on the hidden spectra: candidate structures per formula, fingerprints, predicted fragment/spectra features, PubChem metadata. Ship it as a Dataset so the hidden rerun only does spectrum-dependent scoring plus GBDT inference. Put a per-query time budget with a shared buffer (AIMO-2 style) on any expensive per-query step (e.g., in-silico fragmentation), with a fallback to the cheap score when the budget is exhausted. Use ONNX/Treelite-style GBDT inference only if the reranker itself is slow, which is rare. CPU candidate generation is the place to optimize.
- 4-bit is mainly for fitting a model that would not fit otherwise (72B on T4s). When the model fits in 8-bit, the LMSYS and AIMO-2 evidence says 8-bit is both faster and more accurate.

### Gaps
- I found no first-hand T4 benchmark of vLLM tensor parallel vs two data-parallel replicas for 7-9B models.
- I found no source quantifying ONNX/TensorRT gains for DeBERTa/LLM inference on Kaggle T4s. AIMO-2's TensorRT-LLM numbers are on L4 with FP8, which T4 lacks.
- I could not read the Eedi 1st AWQ calibration ablation (it is an image) or thread 550223 (72B on 2xT4).

---

## Q3. Submission management and teaming

### Takeaway
The winners' write-ups show three habits:
- They spend submissions to confirm CV↔LB correlation early, then mostly trust CV.
- They log a CV/public/private table per model.
- They use extra final-pick slots for the CV-favoured but LB-disfavoured submission.

Code competitions add operational risk: timeouts and deleted artifacts. So the submission plan must include "safe" re-runs and frozen datasets. For teaming, the evidence is split work along the pipeline (data generation vs model vs inference). The submission-count rule for team merges is secondhand.

### Cited Findings
- CPMP: submit a baseline to see "if my cross-validation correlates with the score. If it does, then I don't need to submit too much work with my local setting." — [SED 1817](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)
- LMSYS 3rd: CV/LB correlation was "extremely tight", so "0.001 improvements in CV would show up on the LB". Pseudo-labels shifted the absolute CV level while keeping the correlation. — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
- Eedi 2nd kept a per-model table with Private LB, Public LB *and inference time*, and used it to allocate runtime. Of the 3-pick rule: "we trusted the validation scores and included the 72B model in our final submissions (special thanks to the three-submission rule!)". — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- Eedi 3rd spent two late submissions on deliberate LB probes (seen-only vs unseen-only predictions) to measure a test property, then post-processed (+0.036 private). — [Eedi 3rd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551498)
- Team division of labour, as documented:
  - LMSYS 3rd: one member trained the models, while "thanks to Raja for busting out the whole 500k set" of pseudo-labels.
  - LMSYS 2nd: one member found the full-swap augmentation, another ran the experiments on 4x A100 "During the last 10 days".
  - LMSYS 3rd's final ensemble was one Gemma2 per teammate "for diversity".
  — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766); [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
- Team-merge submission rule (secondhand): on Kaggle a merge is reportedly allowed only if the combined submission count does not exceed max daily submissions × days elapsed. AIcrowd uses the same rule. — [AIcrowd discourse](https://discourse.aicrowd.com/t/is-there-a-limit-on-the-total-number-of-submissions-when-merging-teams/7825) (secondary; verify in the competition Rules)
- Compute disparity: Eedi 2nd trained retrievers for 2-5 days on an H100 each ("GPU and Credit Card is all you need ... Never open the billing page during the competition"). The LLM 20Q winner spent about $500 on rented 4090s. LMSYS 2nd used 4x A100-80G. — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651); [LLM 20Q 1st](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106); [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)

### Inferences
- **Recipe: 5/day budget for 2 people.** Roughly:
  - 1 slot: the current best pipeline, re-run as the "known-good" anchor or a seed/fold variant.
  - 2 slots: experiment A/B that already beat CV on fold 0 and the full CV.
  - 1 slot: a timing/robustness probe (largest model, full TTA) to learn the hidden-set runtime margin.
  - 1 slot: held in reserve.
  - Early in the competition, spend 3-5 submissions on a CV↔LB calibration set (deliberately diverse models) and plot it.
  - Log each submission to `submissions.csv` with: date, notebook slug+version, dataset versions, cfg id, fold-0 score, full CV, public LB, runtime (from the submission page), and the `-m` message.
- **Split work along the pipeline, not by model.** For [CASMI], person A owns candidate generation, caching and the recall@K harness, and person B owns reranker features, GBDT and post-processing. For [NTL], A owns data and augmentation (multi-generator, prompt-grouped folds), and B owns training and inference (quantization, vLLM, time guard). Each delivers versioned Kaggle Datasets to the other.
- Freeze each final-pick candidate's dependencies. Never delete or overwrite a dataset version that a selected notebook references (LMSYS 1st's lost submission).

### Gaps
- I found no first-hand GM post giving an explicit daily-submission allocation scheme. The scheme above is my synthesis.
- I could not reach the official Kaggle text of the team-merge rule.

---

## Q4. Mindset and process hacks: read public notebooks and discussions, reproduce the best public baseline, study past winners, trust CV, time allocation, avoiding rabbit holes

### Takeaway
Winners routinely start from strong public work. They borrow public notebooks, techniques shared in discussions, or a previous competition's data, and they say so in their write-ups. They set explicit hypotheses and kill rabbit holes when time-to-signal is poor. Many big gains came from reusing a shared trick (Agent Alpha, the 72B-on-T4 thread, public pseudo-label data), not from inventing one.

### Cited Findings
- AIMO-1 winners: "Initially, we used Abdur Rafae's public notebook for our submissions, but found the high variance to be problematic", which led them to SC-TIR. — [HF NuminaMath](https://huggingface.co/blog/winning-aimo-progress-prize)
- Eedi 2nd credits a public thread on running 72B/70B models on 2xT4: "In this competition, larger models tend to perform better, which has been very helpful for us." — [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651)
- LLM 20Q 1st exploited the community "Agent Alpha" protocol and took the Yes/No-probability-table idea from another team ("a strategy similar to the maejimakun team was used"). — [LLM 20Q 1st](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106)
- LMSYS 1st built on publicly shared extra data: the "ut" dataset (discussion 499756) and the 33k dataset (500973). — [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
- CPMP on prioritizing: "you estimate the time it takes to code it and run it ... after years of doing this every week, we get some feeling of what might work first." — [SED 1817](https://softwareengineeringdaily.com/wp-content/uploads/2025/02/SED1817-Chris-Deotte-and-Jean-Francois-Puget.txt)
- Rabbit holes, as reported in the write-ups:
  - LMSYS 2nd: "I spent a lot of time trying gemma-2-27b without getting good results" until batch size and grad clipping were fixed by comparing with a teammate's code.
  - LMSYS 3rd: "We tried Gemma2-27B ... but we couldn't get it to train properly."
  - AIMO-1: model merging (DARE/TIES/WARP) gave "significant regressions ... and we ran out of time".
  - BELKA 1st: a month on ZINC pretraining for no gain (see the prior note).
  — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685); [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766); [HF NuminaMath](https://huggingface.co/blog/winning-aimo-progress-prize)
- Diverse baselines first (KGMoN): "We spin up a diverse set of baselines across model types right away." In Rainfall, an un-engineered baseline ensemble took 2nd. — [KGMoN Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data)
- Bhatia: read forums "like research papers" and simulate public/private splits with CV folds. — [NVIDIA: GMs Unveil Winning Strategies](https://developer.nvidia.com/blog/kaggle-grandmasters-unveil-winning-strategies-for-data-science-superpowers/)
- Hypothesis list up front (Eedi 1st "Meta Thoughts") "helped me to stick to an idea even though the initial attempts were not successful". — [Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688)

### Inferences
- **Recipe: week-0 checklist.**
  1. Read every top-voted public notebook and the discussions with high vote counts.
  2. Fork and reproduce the best public baseline *inside your own harness*, so your CV for it is known.
  3. Read the 1st-5th write-ups of the 2-3 most similar past competitions:
     - [NTL]: DAIGT, LMSYS, LLM Prompt Recovery.
     - [CASMI]: Eedi/LECR for retrieve-and-rerank; BELKA for chemistry generalization.
  4. Write a "Meta Thoughts" hypothesis list with an estimated cost per hypothesis (CPMP).
- **Rabbit-hole rule.** Timebox any idea that has given no fold-0 signal after two attempts. A teammate cross-check of the code (LMSYS 2nd's 27B fix) is the cheapest debugging step before abandoning.

### Gaps
- I found no source quantifying time allocation between data, features and models for a GM. Podcasts and interviews give qualitative advice only.

---

## Q5. Concrete training- and inference-budget numbers (LMSYS 2024, Eedi 2024, LLM 20Q, AIMO, DAIGT, KGMoN)

### Takeaway
Top 2024 LLM competitions were trained on rented A100/H100s, from hours to days per model. On Kaggle, inference ran on 2xT4 under about 9 h, using 8-bit 9B models, AWQ/GPTQ-quantized 14-72B models in cascades, and vLLM. The KGMoN tabular wins rest on GPU speedups of roughly 10-150x for dataframes and GBDTs.

### Cited Findings

| Competition / source | Training compute | Inference setup and budget | Source |
|---|---|---|---|
| LMSYS 1st | 5-fold QLoRA Llama3-70B / Qwen2-72B teachers, distilled to Gemma2-9B; max_len 1024 | 5 LoRAs averaged → GPTQ 8-bit; TTA at length 2000 | [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629) |
| LMSYS 2nd | Full fine-tune; 7B on 1x A100-80G, 9B on 2x; 4x A100 in the last 10 days | fp16 9B split over T4x2 with Triton kernels, no padding; max input 4340 | [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685) |
| LMSYS 3rd | QLoRA r=64, seq 1800; 500k vLLM-generated pseudo-labels | 2x Gemma2-9B, length-sorted, 4k/3k max length; "8bit was faster" in hindsight | [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766) |
| LMSYS 9th | LoRA r=32/128, global batch 16; train max_len 1536 | inference max_len 1792; 8-bit not 4-bit; dynamic batch; 3rd model only on length buckets | [LMSYS 9th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527704) |
| Eedi 2nd | 14B retriever ~2-3 days H100; 32B/QwQ ~5 days H100; 14B reranker ~2 h, 72B reranker ~8 h on H100 | Retriever inference 45-140 min each; auto-round quantization; vLLM prefix caching −10% time | [Eedi 2nd](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551651) |
| Eedi 4th | 32B retriever: 10 h per fold on A100 | 50 min per fold of inference, so only 2 folds were used | [Eedi 4th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551559) |
| Eedi 1st | 7/14/32/72B Qwen2.5 cascade | AWQ with task calibration; vLLM; shuffle-TTA dropped for time | [Eedi 1st](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551688) |
| LLM 20Q 1st | No fine-tune; 455M-cell Yes/No table from 3 LLMs via vLLM, 8x4090, about $500 | Lookup-table agent | [LLM 20Q 1st](https://www.kaggle.com/competitions/llm-20-questions/discussion/531106) |
| AIMO-1 (Numina) | 10 h on 8x H100, full fine-tune with packing | P100 or 2xT4, 9 h, 50 problems; 8-bit GPTQ; N=48 × M=4 SC-TIR | [HF blog](https://huggingface.co/blog/winning-aimo-progress-prize) |
| AIMO-2 (NemoSkills) | SFT of Qwen2.5 1.5-32B | 4x L4, 5 h, 50 problems; TensorRT-LLM; FP8+ReDrafter 554 tok/s vs BF16 210; 350 s per question + 210 s buffer; early stop when 4-5 of 16 agree; 34/50 private | [arXiv 2504.16891](https://arxiv.org/html/2504.16891) |
| DAIGT 5th | vLLM on 2x3090+4090 to generate 1.7M docs; AWQ 4-bit for ≥13B | Teacher soft-labels test, then short-context students fine-tuned *at test time* inside the notebook | [DAIGT 5th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470093) |
| DAIGT 1st | 160k samples; Mistral-7B QLoRA + DeBERTa | RAPIDS for SVM/RF "significantly faster than sklearn" | [DAIGT 1st](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/473295) |
| KGMoN (tabular) | cuDF-pandas ~150x on 5 GB ops; 10k+ features tested in days; 100-seed XGB ensembles practical on GPU | — | [cuDF 150x](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/); [Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data); [cuDF FE tip](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/) |

### Inferences
- **[NTL] budget reality.** Top LLM teams trained off-Kaggle on A100/H100. With about 30 T4-hours a week:
  - Train a single 7-9B LoRA at a modest max_len, or a 1-3B model for iteration.
  - Consider distilling the soft labels of a larger rented or free-tier teacher into it (LMSYS 1st pattern).
  - Keep the Kaggle quota for inference timing runs and final fits.
  - QLoRA on 2xT4 for a 9B model at ≤1k tokens is feasible but slow. A single fold-0 run per idea is the realistic unit.
- **[CASMI]** Reranker training is cheap (GBDT, minutes on GPU), so most of the 30 GPU-hours can go to many reranker experiments on cached candidates. Candidate generation should run in CPU sessions, which do not count against the GPU quota, and be cached to datasets.

### Gaps
- Exact Kaggle runtime limits per competition (LMSYS/Eedi used 9 h, I believe, but did not re-verify here) and the hidden-test sizes are not stated in the write-ups I fetched.
- I found no published per-sample latency numbers for 9B 8-bit inference on T4 in these write-ups.
