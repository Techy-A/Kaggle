# Grandmaster Small Hacks, Big Wins: Cheap Training and Modeling Tricks with Measured Gains

Scope and method. I pulled the raw markdown of about 65 top-placed write-ups through Kaggle's internal discussion API (`discussions.DiscussionsService/GetForumTopicById`, with topic lists from `GetTopicListByForumId`). All numbers below are the authors' own. I also used two NVIDIA KGMoN blog posts. The companion file `../Building and introspecting great models/kaggle_grandmaster_practice.md` already covers validation design, ablation-doc format, and the DAIGT/LMSYS/Eedi/LECR/BELKA pipeline overviews. These notes do not repeat that material. They catalogue individual tricks, each with the reported delta, when it helped or hurt, and a minimal recipe.

Evidence tiers used throughout:
- **[A]** A number reported by two or more independent top-10 teams, or an explicit ablation in a top-3 write-up.
- **[B]** A single top-placed team reports a number.
- **[C]** Listed as "worked" without a number, or the reports conflict.

Deltas are in the competition's metric. Lower is better for RMSE, MCRMSE and log-loss; higher is better for everything else.

---

## Q1. NN / transformer fine-tuning tricks (AWP, FGM, EMA/SWA, LLRD, re-init, dropout, MSD, pooling, max_len, padding, LoRA/QLoRA, schedules, freezing, full-data retrain)

### Takeaway
The best-evidenced cheap NN tricks fall into four groups:
1. **Adversarial training (AWP/FGM).** Typically +0.003 to +0.01, repeatedly in the Feedback Prize series and USPPPM. It sometimes does nothing on small regression sets such as CommonLit 2 and AES2.
2. **Turning dropout off for regression heads.** Marked "high impact" by several teams.
3. **Getting the input formatting and pooling right.** Head-mask pooling, keeping `\n`, prompt tokens. This often beats any optimizer trick.
4. **Length tricks.** Train short and infer long, or the reverse, with dynamic padding.

LLRD, re-init, SWA, EMA and multi-sample dropout are cheap "maybe" tricks: about as many top teams report them as useless as useful. For LLMs, LoRA r=64 on all linear layers is the de-facto default. Hyperparameter search on alpha, DoRA or rsLoRA brought no clear gain in LMSYS.

### Cited Findings

**Adversarial training: AWP / FGM [A]**
- Feedback Prize 2021, 1st (2022): "adversarial learning (awp/fgm): cv increase 0.01, lb 5fold ensemble increase 0.003." — [FP2021 1st](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)
- Feedback Prize Effectiveness, 3rd (Raja Biswas et al., 2022): "AWP: 0.005 to 0.01" (log-loss). AWP is also listed as a training-stabiliser that let them train 6-7 epochs. — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)
- FPE 2nd (2022): "For Tom AWP was useful, and he reported around 0.003 cv improvement with AWP (eps=1e-3 and lr=1e-2 for large models, 1e-4 for xlarge models)." — [FPE 2nd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347359)
- USPPPM 1st (gezi, 2022): "5 epochs training, start AWP training from the 2nd epoch. AWP helps a lot in all my nlp contests recently." The same author used AWP only for the last 2 of 4 epochs in ELL 2nd. — [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243); [ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)
- USPPPM 2nd (2022): "FGM, eps: 0.1, single model cv 0.002-0.005". — [USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)
- ELL 1st lists AWP among its standard ingredients (no number). — [ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)
- **Hurts or neutral:** CommonLit Summaries 2nd lists "Did not work: AWP, SWA". 3rd lists AWP and FGM under "did not work". 4th: "Not working for us: mlm, awp, svr". AES2 2nd also lists AWP as not working. — [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573); [CLES 3rd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446686); [CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524); [AES2 2nd](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)
- Recipe, from the numbers above:
  - AWP: `adv_lr≈1e-2` (large) or `1e-4` (xlarge), `adv_eps≈1e-3`. Switch it on after epoch 1, or for the last half of training, so the model first fits normally.
  - FGM: perturb the embedding by eps=0.1 × normalised gradient, do a second forward/backward, then restore. Cost is about 1.5-2x per step.

**EMA / SWA of weights [C, mixed]**
- USPPPM 2nd: "EMA, decay: 0.999, single model cv 0.001-0.003". — [USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)
- CLES 3rd (Dmitrii Krasnov) ranked EMA #2 of his tricks: "using EMA (honestly, without EMA it was very unstable, so it's probably just necessary)". — [CLES 3rd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446686)
- CommonLit Readability 4th: "Epochs: 6 with SWA after epoch 3", combined with no dropout, weight decay 1.0 and checkpointing every 5 steps. — [CLRP 4th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258148)
- **Did not work:** SWA in FPE 3rd, FPE 15th and CLES 2nd. — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433); [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381); [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)

**Dropout = 0 for regression / fine-tuning [A]**
- ELL 3rd (Chris Deotte's team): "`hidden_dropout_prob = 0.0` and `attention_probs_dropout_prob = 0.0` [High Impact]". — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)
- CLRP 2nd (Takoi): "I trained the model with dropout set to 0 except for 1 and 2". CLRP 4th: "No dropout". CLRP 15th: two members used "dropout = 0". — [CLRP 2nd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328); [CLRP 4th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258148); [CLRP 15th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260800)
- AES2 1st: "Dropout: 0" in its best-params table. CLES 2nd: "Turn off dropout in transformer backbone" (did work). — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791); [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)
- FPE 15th (a classification task) adds a caveat: "Set dropout to ZERO, just usefull for mlm-pretrained weight". — [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381)
- For LLM classifiers, LMSYS 3rd used QLoRA "dropout=0.0". LMSYS 4th used lora_dropout 0.05. — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766); [LMSYS 4th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067)

**Multi-sample dropout (MSD) in the head [B]**
- FPE 3rd: "Mask Augmentation + MSD: 0.002 - 0.005". — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)
- CLRP 3rd: "weighted average of all layers and multisample dropout ... + separate lr for head" took public 0.462 → 0.459. Private did not move with these public gains (see the CLRP caveat below). — [CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)
- PII 1st (2024): "Multi-Sample Dropout Custom Model: Helped to train with a better-fitted train curve and less of spikes". CLES 2nd: "Multisample dropout in head" worked. — [PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374); [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)

**Layer-wise LR decay (LLRD), differential LR, re-initialising top layers, freezing [B/C, mixed]**
- FPE 15th recipe for stable DeBERTa: "Smaller LR, like 5e-6; LR decay when layer go deeper, we split layers to 6 groups and use 0.8 decay ratio; Warmup LR scheduler; AWP". — [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381)
- CLRP 3rd's "3 group decaying lr for encoder" and "separate lr for head" are in its "what worked" list. — [CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)
- USPPPM 1st:
  - Differential LR: backbone 2e-5 or 3e-5, head 1e-3, "especially useful when adding LSTM which need large lr".
  - LR choice mattered a lot: deberta-v3-large at "2e-5 much better then 3e-5", CV 0.861 → 0.8627.
  - "Freeze bert embedding layer (not much difference maybe but I used it for final models)". — [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)
- ELL 1st "What Worked": "freezing top n layers", "re_init top n layers", "Training with Differential learning rate". ELL 3rd: "Last layer reinitialisation [Medium Impact]". — [ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457); [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)
- CLES 2nd: "Layer wise learning rate decay" and "Freezing layers (bottom 8)" both worked. CLES 4th used freezing for some models and LLRD for others. — [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573); [CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524)
- **Counter-evidence:**
  - Psi (ELL 5th): "I use differential learning rate for backbone and head of the model. I do not use any other techniques suggested in forums like differential lr across layers of backbone or reinitialization." — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)
  - AES2 1st: "Differential LR - tiny difference". — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
  - PII 1st: "Freezing layers" did not work. — [PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)
- Head re-initialisation for LLM classifiers (LMSYS 2nd): "`XXXForSequenceClassification` initialization of head gives high loss on early iterations, so the head is re-initialized" with a 2-layer MLP head (Dropout 0.1, Linear h→h/2, GELU, Linear→3). — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
- Freezing or not training embeddings for added tokens (LMSYS 3rd): "I didn't add those separator tokens to the vocabulary because training my embeddings seemed to hurt my models." — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)

**Gradient clipping, batch size, precision [B]**
- ELL 3rd: "Clip grad norm with `max_norm = 10` [High Impact]" and "Train with `batch_size = 1` [High Impact]". AES2 1st also used "Grad clip 10". — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609); [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- LLM full fine-tunes:
  - LMSYS 2nd got gemma-2-27b to train only after it "adjusted the batch_size to 80 and turned off grad_clip". — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
  - WSDM Cup 2025 1st: "We increase gradient clipping threshold to 64.0". — [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
- FPE 3rd lists "**Full precision training**" among the things that stabilised DeBERTa training. — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)
- LMSYS 2nd trained 7-9B models with full parameters in BF16, using an optimizer "with kahan summation support". 7B fits on one A100-80G; 9B needs two. — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)

**Warmup and schedules [B]**
- FP2021 4th: "trained for 5k/7k steps with 500/700 warmup steps ... learning rate is basically 3e-5 ... decayed linearly ... gradient clipping with 1.0 and gradient checkpointing ... evaluate 20 times per epoch". — [FP2021 4th](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)
- AES2 1st: warmup 15% of pre-train, cosine decay, LR torso 1e-5 / 5e-6 and head 2e-5 / 1e-5 (pre-train / fine-tune), init `normal_(mean=0.0, std=0.02), bias=0`. — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- WSDM 1st (LLM full FT): "50 warm-up steps followed by linear decay into 10% of the peak lr (4e-6)". — [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
- CLRP 3rd: "a 3 stage lr schedule" (public 0.459 → 0.454). The pseudo-label fine-tune used "low lr 3 epoch and an lr scheduler with a decaying rate (1,0.5,0.25) for each epoch, no warmup". — [CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)
- AMEX 14th (Deotte, tabular transformer): "4 cycles of cosine learning schedule". Cycles 1 and 3 are knowledge distillation on soft labels; cycles 2 and 4 are a warm restart at lower LR on hard labels. — [AMEX 14th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641)
- WSDM 3rd ran an LR sweep for Qwen2.5-14B LoRA. Accuracy was flat around 0.70 from 7e-5 to 12e-5 and collapsed to 0.508 at 14e-5 ("Loss spike starts to occur around 14e-5"). For 72B, "a smaller learning rate is better". — [WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)
- Frequent evaluation with best-checkpoint saving on small data: CLRP 1st evaluated "every 10 steps" with LR 7e-6 to 1e-5; CLRP 4th checkpointed "every 5 steps". — [CLRP 1st](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/257844); [CLRP 4th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258148)
- Contrast: Psi ran "3 epochs ... all with cosine decay learning rate always picking the last epoch" (no early stopping), as did AES2 2nd ("Save the checkpoints for the last epoch"). — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578); [AES2 2nd](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)

**Pooling and "what the head looks at" (often the largest single win) [A]**
- CommonLit Summaries 2nd (Ivan Aerlic): "Head Mask". The normal attention mask goes to the model, but mean pooling uses a mask that is 1 only on the student-summary tokens. "This had the biggest impact out of all the tricks I used ... In my opinion this was the 'magic' for this competition." — [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)
- CLES 4th attention-pools only the first 12 tokens, an instruction prefix ("Evaluating the summarized text and calculating content and wording score :"). CLES 3rd concatenates the CLS token with a mean-pool of the student text only. — [CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524); [CLES 3rd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446686)
- FPE 3rd: "Mean pooling to extract span representations as opposed to [CLS] token based ones". FPE 15th found first-token pooling beat span mean or attention pooling ("not work"). — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433); [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381)
- ELL 1st built diversity from pooling: Mean, Concat, WeightedLayer, GeM and LSTM pooling. — [ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)
- ELL 3rd: "2 stage pooling. First pool either words, sentences, or paragraphs. Then pool that result. [Medium Impact]". AES2 2nd exp321 used the same idea (sentence → paragraph → document mean pool). — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609); [AES2 2nd](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)
- BiLSTM head on top of the transformer: USPPPM 1st CV 0.858 → 0.861, and "Add LSTM help me get good enough single model, which is the key for win on PB (8745 -> 8779)". — [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)
- **Counter:** AES2 1st lists "Attention pooling" under "Did not work". It used CLS and GeM. — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- For decoder-LLM embeddings, Eedi 2nd found last-token pooling beat mean pooling for Qwen. This is already in the companion notes.

**Tokenisation and formatting details that move scores [A]**
- FP2021 4th: `offset_mapping` labelling instead of `word_ids` (which drops `\n`) took single-fold bigbird public from 0.595 to 0.630. The deberta-v2/v3 tokenizer also "removes line-break `\n` characters"; fixing that brought deberta-v3-large to CV 0.7038. — [FP2021 4th](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)
- ELL 3rd: "Since Deberta tokenizer ignores '\n' we replaced '\n\n' with '|' [Medium Impact]". AES2 1st added tokens `"\n"` and `"  "` "as @cdeotte shared". — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609); [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- AES2 4th (tascj): prefixing each text with a data-source tag (`[A]` non-Persuade, `[B]` Persuade) took best-epoch QWK from 0.8370 to 0.8446. At test time every sample is tagged with the test source. — [AES2 4th](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516639)
- FPE 3rd: adding the prompt or topic to the essay gave "Prompts: 0.002 to 0.005". USPPPM 1st/2nd: putting the other targets of the same anchor into the input context was "key magic/trick to the gold". — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433); [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243); [USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)
- LMSYS 3rd: "if you disable the softcapping on Gemma2, CV and LB were better by roughly 0.002". — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)

**max_len: train-vs-infer mismatch, gradient checkpointing, dynamic padding, length bucketing [A]**
- ELL 3rd: "Train with `max_len=2048`, infer `max_len = 640` [High Impact]". — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)
- CLES 4th went the other way: training at 850 and inferring at 1500 moved CV from 0.4984 to 0.4867 (0.4927 at 1024). The caveat: "some models may degrade", and the insurance sub without length extension did nearly as well on private (0.45597 vs 0.45515). — [CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524)
- DAIGT 7th: DeBERTa-v3-large trained at 512 beat training at 1024 (private 0.965 vs 0.957), both inferred at 1024. DAIGT 3rd trained at "max length 256 (1512 for inference)". — [DAIGT 7th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643); [DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)
- LMSYS 4th: train 2048, infer 3072. CLES 2nd: pseudo-labels "allowed the models to learn at a higher maximum length" (training went from 896-1280 to 1280-2048; inference at 1792/2048). — [LMSYS 4th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067); [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)
- FP2021 2nd (Deotte): "Notice how we use FP16, gradient_accumulation_steps, and gradient_checkpointing. With these settings, you can basically train any size model on any GPU including Kaggle's P100 16GB GPU". All models were trained at 1536 tokens. — [FP2021 2nd](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389)
- Psi (ELL 5th): "All my models are trained and predicted with dynamic padding". — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)
- tascj (AES2 4th): "Dynamic Micro Batch Collation". Each batch is split into micro-batches under a total-token cap and the loss is scaled by micro-batch size: "faster and memory usage stable". His DeBERTa rewrite was 15%+ faster in training and 30%+ faster in inference. — [AES2 4th](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516639)
- LMSYS 2nd: unpadded sequence concatenation with `flash_attn_varlen_func` and `cu_seqlens`, so no pad tokens are used in either training or inference. — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)
- Sorting by length at inference:
  - CLES 1st: "Sort index according to the length of the input text ... estimated to be 7 hours, but if you do not do so, the inference time will exceed the limit of 9 hours." — [CLES 1st](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/447293)
  - LMSYS 5th (Psi) also sorted by length, ran half the data on each T4, and used INT8 bitsandbytes with fp16 compute. — [LMSYS 5th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669)

**LoRA / QLoRA settings that won [A for "r=64, all linear"]**
- LMSYS 1st: "lora(9b) qlora(llama3 and qwen2) all linear ... r=64,a=128 max_len=1024 epoch = 2 global batch_size = 64". — [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
- LMSYS 3rd: QLoRA "r = 64 and alpha = 16. Targeting all linear layers also worked the best". On the second backbone: "I experimented with different LoRA settings (e.g. rslora, lora+, unfreeze embed, ranks, grad norm). However, there wasn't any significant gain ... DoRA and full-tuning wasn't clearly better." — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
- LMSYS 4th: "lora_r: 64, lora_alpha: 16, lora_dropout: 0.05" on q,k,v,o,gate,up,down; `freeze_layers: 0`. — [LMSYS 4th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067)
- DAIGT 1st: Mistral-7B "(Q)LoRA ... on QKVO with r=64" gave 0.984 private / 0.966 public, the team's best single model. — [DAIGT 1st short](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121)
- Eedi 5th: listwise reranker Qwen2.5-32B with LoRA r=24 α=48, lr 5e-5, cosine, 10 warmup steps, 1 epoch, global batch 8, unsloth. Bi-encoder LoRA r=32 α=64. — [Eedi 5th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551391)
- Several teams skipped LoRA entirely in favour of full fine-tuning at 9-14B: LMSYS 2nd and WSDM 1st/2nd. — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685); [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
- Post-training tricks:
  - LMSYS 1st: "Directly average the LoRA layers of the 5 folds", then "Quantize to 8-bit using GPTQ". — [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
  - WSDM 1st: "The final model is a linear merge of two models". WSDM 3rd: "trained the same model with different seeds and linearly merged the weight". — [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902); [WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)
- Quantisation calibration matters (WSDM 3rd, auto-round on a 3B proxy): accuracy went from 0.681 unquantised to 0.6747 with the lite setting, versus about 0.6809 with seqlen=6144, nsamples=512, iters=3000. Calibration seqlen should match the inference length. — [WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)
- Starting from a task-adjacent checkpoint beats starting from a chat model:
  - LMSYS 5th (Psi): pre-training gemma-2-9b on UltraFeedback binary win/loss "boosted our scores by up to 20 points on CV and public LB". I read "points" as 0.001 log-loss units, i.e. about 0.020; that unit is my interpretation. — [LMSYS 5th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669)
  - LMSYS 3rd started from reward models (`sfairXC/FsfairX-Gemma2-RM-v0.1` instead of `gemma-2-9b-it`). — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
- Restricting the loss to the answer tokens (WSDM 3rd): an SFT trainer with "loss only for Token 'A' and 'B'", using AutoModelForCausalLM so that vLLM can serve it. — [WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)

**Domain MLM / continued pre-training [B, mixed]**
- FPE 3rd: "Span MLM: 0.02 to 0.03", the team's largest single gain. Recipe: mask rate 40-50% instead of 15%, contiguous spans of 3-15 tokens, chunk length 720 to match the fine-tune length, on about 15k old essays plus 25k T5-generated essays. — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)
- FPE 15th: "MLM pretrained on FB2021 & FB2022 can improve both stability and accuracy". AES2 2nd: "10 epochs of MLM on all the training set" for every model. — [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381); [AES2 2nd](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)
- **Did not help:** USPPPM 2nd ("Not worked: ... MLM"), CLES 4th, and PII 1st ("MLM pretraining"). — [USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234); [CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524); [PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)

**Auxiliary losses and loss-function swaps [B]**
- ELL 2nd (gezi):
  - Ranking loss at rate 0.1 took CV from 0.4514 to 0.4505. "Rank loss always help."
  - A Pearson loss helped "hard" targets but hurt easy ones.
  - Putting the six target names as tokens in the input, each with its own FC head ("Let bert deal with the target relation"), also helped. — [ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)
- ELL 3rd: "Different loss rates per target ... [High Impact]". — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)
- USPPPM 1st: "Pearson loss worked best for me". — [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)
- FPE 1st: an auxiliary loss predicting each discourse's type "helped with regularizing the model and allowed for a bit longer training". — [FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)
- AES2 2nd: ordinal regression via BCE on cumulative targets, and "scale the score from 1-6 to 0-1 and use bce loss". AES2 1st also used BCE on [0,1]-scaled targets. — [AES2 2nd](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790); [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- CLES 2nd: auxiliary heads predicting Feedback-3 pseudo targets, `(loss * .5) + (aux_loss * .5)` every second step, "[Moderate Impact]". — [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)
- LMSYS 3rd: label smoothing 0.03 on the LLaMA baseline. Predicting the model identity as an auxiliary loss "didn't help". — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)

**Full-data retrain after CV [A]**
- Team Hydrogen (FPE 1st): "always subbed a blend of 3 seeds of models trained on the full data". Psi (ELL 5th): "As always, I retrained my models on full data for final subs". — [FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536); [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)
- USPPPM 1st: "Using 5 folds + 1 full train for each backbone, with full train model weight * 2", which "tested indeed better then 1:1 for LB and PB". Single deberta-v3-large private: 1-fold 0.8710 → full-train 0.8745 → 5-fold 0.8761. — [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)
- CLES 4th: replacing 4-fold models with 1 full-train model per architecture let them ensemble 7 architectures inside the time limit. The best-CV sub of that kind finished 4th private, versus 11th for the best-LB sub. — [CLES 4th](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524)
- AES2 1st: "All my submissions are full-fits ... which meant they always behaved somewhat differently from the OOFs even when using the same seed. Averaging probably helped because of this too." — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- FP2021 2nd: a deberta-large "trained with 100% train data" was the team's best private model (0.721). — [FP2021 2nd](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389)

**Seed averaging for NNs on small data [A]**
- Psi (ELL 5th): "for each experiment ... I trained three unique seeds and always only compared the average of these three seeds". — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)
- AES2 1st: "3-seed-average everything (mostly reduced variance ...)". Redoing only the fine-tuning stage with a different seed "made hardly any difference in performance and saved a lot of time". — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- CLRP 4th: 5 folds × 5 seeds = 25 runs per experiment, carried over from Tweet Sentiment Extraction. — [CLRP 4th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258148)

**Tricks that won on public but not on private (cautionary) [B]**
- CLRP 3rd: the chain of attention head + LLRD + MSD + layer averaging + 3-stage LR + early stop took public from 0.462 to 0.453, but "the private did not agree ... Private LB stayed at ~0.461". The real private gains came from pseudo-labelling and adding DeBERTa (0.447). — [CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)

### Inferences
- Default recipe for an encoder or LLM fine-tune on small labelled data, ranked by evidence:
  1. Get the input right: keep `\n`, add source or prompt tags, and pool only over the span that carries the label.
  2. Use dropout 0 in the backbone for regression and soft-label tasks.
  3. Compare 3-seed averages, never single runs.
  4. Try AWP from epoch 2 (cheap, +0.003-0.01 when it works).
  5. Use a separate head LR.
  6. Retrain on full data with the same epoch count.

  Then treat LLRD, re-init, freezing, EMA, SWA and MSD as diversity knobs for the ensemble, not as guaranteed gains.
- For LLM classifiers, the cheapest large lever is the starting checkpoint (reward- or task-adjacent pre-training: +0.020 in LMSYS). Next come answer-token-only loss and LoRA r=64 on all linear layers. LoRA hyperparameter search rarely pays.
- Train and infer lengths are a free hyperparameter, but the sign of the effect varies by competition: train long/infer short helped ELL; train short/infer long helped CLES and DAIGT. Always sweep inference length on OOF before submitting.

### Gaps
- I found no top write-up reporting a clean on/off ablation of gradient checkpointing's effect on score. It is used purely to fit longer contexts.
- QLoRA-specific tricks beyond rank, alpha and target modules (for example double quantisation or paged optimisers) are not discussed with numbers in any write-up I retrieved.
- AWP hyperparameters other than FPE 2nd's eps/lr pair are rarely reported.

---

## Q2. Data tricks: pseudo-labelling, distillation, external/synthetic data, augmentation, dedup/noise removal, "train on test"

### Takeaway
Pseudo-labelling with soft labels on related unlabeled data, generated leak-free per fold, is the single most repeated large win in NLP write-ups, typically +0.004 to +0.01. Pre-training on pseudo-labels and then fine-tuning on gold labels is the safer variant. A second round helps sometimes and is neutral or harmful about as often. Distilling large teachers' soft labels gives +0.005-0.01 (PII, WSDM). In WSDM 3rd, self-distillation from an equally sized model reached the same CV as a 72B teacher, which points to label cleaning as the mechanism. Classical NLP augmentation (back-translation, random token aug) mostly fails; task-specific augmentation (A/B swap, LLM paraphrase of the prompt, random typos, source-tag swap) works. Counts or unsupervised fits that include test rows are the classic "magic" (Santander), but only after removing fake or irrelevant test rows.

### Cited Findings

**Pseudo-labelling: soft vs hard, rounds, thresholds, leak-free per fold [A]**
- CLRP 1st (Mathis Lucka, 2021):
  - Retrieve the 5 most similar external snippets per training excerpt with Sentence-BERT (`paraphrase-MiniLM-L6-v2`) and label them with the best models.
  - Filter out pseudo-labels that deviate from the source excerpt's target by more than its standard error, so the label distribution stays matched.
  - Pre-train on pseudo-labels for 1-4 epochs, then fine-tune on train.
  - "while multiple rounds of pseudo-labeling continuously improved my CV, my LB score got worse." — [CLRP 1st](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/257844)
- CLRP 3rd: "Using all folds to label lead to unrealistic low cv (due to leak) and no improvement to LB, so I used 5 fold pseudo label". Two cycles took public 0.453 → 0.450 → 0.448. Mixing relabelled train data into the PL set at weight 0.5 (versus 0.8 or 1) gave the best private, 0.446. — [CLRP 3rd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258095)
- FPE 2nd: "Retrain model with crossentropy on pseudo label probabilities (not discretized) ... 3 epochs on pl labels only first and then 3 more epochs on gt labels only". Rounds: "I saw improvement until 5 rounds ... For Tom, it was only helpful for one round". — [FPE 2nd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347359)
- ELL 1st:
  - "Mean of Train Pl / Actual Labels [This gives nice boost in both CV/LB]".
  - Warning: "generating Soft/Pseudo labels on train data leads to huge overfitting, CV around 0.437". — [ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)
- ELL 2nd: FB1 pseudo-label training alone took CV from 0.4514 to 0.4469. Combined with rank loss and FB3 training it reached 0.4456 (private 0.434726). — [ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)
- Psi (ELL 5th): pre-train on pseudo-labels, then fine-tune on train only. "By doing pre-training and finetuning, I did not need to adjust the distribution of the pseudo labels". — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)
- **Negative case:** ELL 3rd (Deotte): pseudo-labels from FP1 "boosted our single model best CV score from 0.4470 to 0.4370 and did not improve our LB score". The FP1 target distribution was shifted higher, and the final sub used no PL. — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)
- AES2 1st:
  - Round 1 used `mean(score, ensemble_pred)` as the label on old data; round 2 used the prediction alone. "Pseudo-label old data (+0.004-0.007)".
  - Round 2 broke CV/LB correlation, but its best-CV model scored 0.839 private as a single model. — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- CLES 2nd: PL took grouped-kfold CV from 0.4581 to 0.4476, using a concatenation of original and pseudo labels. — [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)
- CLES 1st: "Meta pseudo label (3 rounds), the most critical and time-consuming part" (Pham et al. 2020), on LLM-generated topics and summaries. Two-stage: PL-only for 2 epochs, then train data for 2-3 epochs. — [CLES 1st](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/447293)
- LMSYS 3rd: "PL was huge for us". Soft-labelled 500k+ generated pairs; "We ran it for 2 rounds, but the second round didn't make much of a difference". PL also "caused a break in our CV/LB correlation". — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)
- WSDM 2nd: confidence filter on pseudo-labels, "Removed data with `abs(winner_model_a - winner_model_b) <= 0.03`". A pseudo-label source was dropped when its model pairs had "minimal capability gaps ... which could introduce noise". Pre-training was then re-run on the 95k pseudo-labels, followed by fine-tuning on gold, "to avoid pseudo-label noise". — [WSDM 2nd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948)
- Hard pseudo-labels with confidence thresholds on test:
  - Santander 1st: LGBM used "2700 highest predicted test points as 1 and 2000 lowest as 0". The NN used 5000/3000 and went from 0.92497 to 0.92546 private. — [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
  - Santander 5th: PL worth "almost another +0.0005". — [Santander 5th](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88897)
  - DAIGT 3rd: adding 1k test samples where the transformer ensemble was most confident (p<0.01 or >0.99) to the TF-IDF pipeline left public unchanged and took private from 0.893 to 0.927. — [DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)
- Did not work: PII 1st ("Pseudo-labelling the test data"); FPE 3rd ("on hindsight, this was a big miss for us"); FPE 15th ("Meta Pseudo"). — [PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374); [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433); [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381)

**Knowledge distillation (soft teacher labels) [A]**
- USPPPM 2nd: "soft label from ensemble oof. In this way, single model can achieve performance close to ensemble models ... Make sure to use only the corresponding label for each fold to avoid leakage. The actual performance of second or more rounds is almost the same as first round". — [USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)
- PII 1st: teacher→student KD with T=3, α=0.5 (CE + T²·KL) "improved CV and LB by about 0.005-0.01". — [PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)
- WSDM 3rd ablation (Qwen2.5-14B, accuracy):
  - Baseline 0.7051.
  - KD with T=1, w=0.5: 0.7062. T=10: 0.7082. T=5: 0.7103.
  - T=5 with soft-loss weight 0.7: 0.7121; with 0.9: 0.7135.
  - More teachers: 0.7169.
  - KL alone was enough; KL+cosine "not obvious".
  - Key finding: "the 72B model is not necessary. Using the logit of 14B model and perform 'self-distillation' can reach the same CV score. Maybe it is the the effect of label cleaning brought by soft logit that really matters." — [WSDM 3rd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)
- WSDM 1st: 5× Qwen2.5-72B fold teachers soft-labelled out-of-fold. They "manually review samples which had the highest disagreement with human annotations in order to correct mislabelled pairs". Student loss was CE + KL + cosine. — [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
- AMEX 14th (Deotte): distil LGBM soft predictions on train and test rows into a transformer, then fine-tune on hard labels. NN 0.798/0.799, LGBM 0.799/0.799, 50/50 blend 0.800/0.801 (CV/LB). "both train and test data are used for knowledge distillation which helps the Transformer learn the test data distribution." — [AMEX 14th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641)
- LMSYS 1st distillation (70B/72B → 9B) is covered in the companion notes. — [LMSYS 1st](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)

**Task-specific augmentation (works) versus generic NLP augmentation (usually doesn't) [A]**
- **Works:**
  - LMSYS 2nd "Full swap": include both the A/B order and its swap in the same optimizer step, for "a stable 0.003 improvement" over random swap. WSDM 1st copied it. — [LMSYS 2nd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685); [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
  - CLES 2nd: an LLM wrote 10 paraphrases per prompt question, used as training augmentation, "[Moderate Impact]". — [CLES 2nd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)
  - DAIGT 4th: random character typos inserted into 30% of generated text, because the hosts' obfuscations "were completely random". — [DAIGT 4th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470179)
  - FPE 3rd: T5 label-preserving synthetic discourse, "-0.002 to 0.005", plus "Mask Augmentations (10-25%)". Team Hydrogen: "Only mask augmentations worked ... bringing some diversity for the ensemble". — [FPE 3rd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433); [FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)
  - USPPPM 1st: "Random shuffle targets during each training step ... improved a lot on LB". — [USPPPM 1st](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)
  - Santander 1st/5th: within-class column-shuffle augmentation (positives ×16 and negatives ×4 in 1st; ×20 and ×2 in 5th) took LGBM to 0.920x. This is only valid because the features were independent. — [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003); [Santander 5th](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88897)
  - AMEX 2nd: duplicate each client with the last statement removed, a second training row validated only on last statements, for robustness to temporal drift. — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)
- **Back-translation:**
  - Fails as direct augmentation: "back translation" is among FP2021 1st's useless attempts; FPE 15th lists "back translation augmentation" as not working; AES2 1st lists "Backtranslation" as not working. — [FP2021 1st](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177); [FPE 15th](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347381); [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
  - Works as pre-training data: ELL 2nd used 14-language back-translation only for a 2-epoch pre-train stage, taking CV from 0.4505 to 0.4498, with a large vocabulary-target gain. — [ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)
- Generic augmentation fails on regression: Psi: "Augmentations (as always, specifically tricky with regression)". ELL 1st: "Augmentations" did not work. — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578); [ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)
- Mixup/cutmix: no NLP or tabular write-up I retrieved reports it (see Gaps).

**External, related and synthetic data, and two-stage training [A]**
- AES2 1st: "Pretrain -> finetune on new data (+0.015)", the largest gain in the competition, because old and new essays used different scoring criteria. AES2 2nd made the same two-stage move its #1 key. — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791); [AES2 2nd](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516790)
- ELL 2nd: pre-training on Feedback-2 effectiveness labels took CV from 0.4505 to 0.4488. — [ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)
- DAIGT 2nd (Guanshuo Xu):
  - About 500k SlimPajama human/AI pairs as "pretraining", giving 0.916 public / 0.967 private.
  - Then fine-tuning on essays from generators LM-fine-tuned to mimic students ("including citations ... and even typos"), giving 0.94/0.98.
  - Final 0.967/0.983. — [DAIGT 2nd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470395)
- DAIGT 3rd: models trained on 0.5M, 1M and 1.2M generated samples, from about 35 open models with randomised sampling parameters. Best single (about 1M) scored 0.956 public / 0.967 private. Fine-tuning these on a curated 11k set raised public slightly and lowered private by about 0.005. — [DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)
- DAIGT 7th: "use only non-instruction-tuned models to generate data". DeBERTa-v3-large trained on Persuade + SlimPajama scored 0.965 private, versus 0.875 for base trained on Persuade alone. — [DAIGT 7th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643)
- PII 2nd mixed external generated data "with a lower weight of 0.5". — [PII 2nd](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352)
- WSDM 1st's synthetic set was filtered to prompts with Arena-Hard complexity ≥3, preserving the language and domain distribution of the competition (50% non-English, 8 MT-Bench categories). WSDM 2nd increased the non-English share across its pseudo-label sets and trained "1st epoch on English subset, 2nd epoch on multilingual subset", which raised private from 0.708 to 0.716. — [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902); [WSDM 2nd](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948)
- Otto 3rd (Theo Viel): recovering sessions the hosts had removed and using them for training gave a "nice (0.0005 to 0.001) boost" on carts and orders. — [Otto 3rd Theo](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382975)

**Hard-example data curation, dedup and noise removal [B]**
- DAIGT 3rd: iterative selection. Each round adds the 500 human and 500 generated samples "that the previous model failed to predict correctly ... with the highest distance from the true label". LB was checked every 4th iteration and the process stopped when LB stopped improving. — [DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)
- LMSYS 3rd removed "~300 examples with high train loss". WSDM 1st filtered ties, multi-turn responses and duplicate prompts before teacher training, and relabelled teacher-human disagreements by hand. — [LMSYS 3rd](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766); [WSDM 1st](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)
- CLES 3rd: found near-duplicate essays with Levenshtein similarity and merged them, then used "reverse autocorrect" augmentation. This gave 0.453 private on one fold. — [CLES 3rd](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446686)
- AMEX 2nd: "isle" de-noising. Sort the unique values, group them by gaps, then floor-divide by the group's minimum step to undo injected uniform noise. Built on raddar's public de-noised dataset. — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)
- FP2021 4th: fixed cp1252/utf-8 mojibake with `difflib` offset correction ("doesn't seem to improve the final scores significantly"). — [FP2021 4th](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)

**"Train on test": unsupervised fits and counts that include test [A where there is a shift]**
- DAIGT 1st: "train a randomly initialized deberta-v3-small with custom vocabulary from scratch during inference run. We derived a custom tokenizer using train + test essays, performed MLM pre-training & subsequent fine-tuning on training datamix enriched with test set pseudo labels". It scored 0.943 private / 0.942 public and served as an ensemble member, not the best model. — [DAIGT 1st short](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121)
- Counter: DAIGT 7th "MLM pretraining on the test set (10 epochs with deberta-v3-xsmall), then finetune on my dataset" scored 0.689, a failure. — [DAIGT 7th](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470643)
- Santander 2019 "magic":
  - 5th: "Count of unique values of all features, of train + real test data together. If you also include the fake data in the calculations, the uplift will NOT SHOW on LB", taking LightGBM to about 0.914x. — [Santander 5th](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88897)
  - 1st built 200 categorical "has one" features (value unique in train, unique in train+real test, or co-occurring only with target 1 or only with 0) plus 200 numeric features with unique values replaced by the column mean: .910 → .914 → .921 LB with LGBM. — [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)
  - 3rd: "we figured out counts improved CV however LB did not changed and after rereading forums we realised that we had to remove fake test". — [Santander 3rd](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88902)
- USPPPM 2nd: at inference, groupings of targets by anchor were built "after concatenating train-set and test-set ... (Why concat? Because overlap anchors in train and test.)" — [USPPPM 2nd](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)
- DAIGT 3rd post-processing: per prompt with more than 1000 samples, fit UMAP on test TF-IDF and rescale predictions by the ratio of distances to the 7 nearest human versus generated samples, clipped to (0.9, 1.1). It "slightly improved public and private LB". — [DAIGT 3rd](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470333)
- CLRP 15th (Deotte): "the test data has a mean target that is 0.02 lower than the train target. Therefore if we subtract 0.02 from our final predictions this boosted LB +0.001". — [CLRP 15th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260800)

### Inferences
- PL recipe with the best evidence:
  - Labels come from an ensemble.
  - Generate k fold-specific label sets.
  - Use soft labels, or the mean of soft label and gold where gold exists.
  - Pre-train on PL, then fine-tune on gold. This sidesteps PL distribution mismatch (Psi, ELL 5th; CLES 1st; WSDM 2nd).
  - Filter low-margin PL (WSDM 2nd: |pA−pB| ≤ 0.03 removed).
  - Stop at 1-2 rounds unless CV keeps tracking LB.

  Watch for a sudden large CV gain without an LB gain. This was a leak or distribution mismatch in ELL 3rd, ELL 1st and CLRP 1st.
- Self-distillation (student and teacher the same size, soft OOF labels) is nearly free and gave the same gain as a 72B teacher in WSDM 3rd. Try it before paying for a big teacher.
- "Train on test" pays when there is an exploitable train→test shift and the test rows are real (Santander fake rows, DAIGT unseen generators). It can fail badly when the test set is too small for MLM (DAIGT 7th, 0.689).

### Gaps
- I found no write-up in the retrieved set reporting mixup or cutmix for NLP or tabular. Those are CV-competition tricks and outside these sources.
- I found no quantified ablation of hard versus soft pseudo-labels on the same pipeline. The evidence that soft labels win is indirect: almost every top NLP team chose soft labels.

---

## Q3. GBDT tricks: seed bagging, target encoding inside folds, aggregates/"magic", count encoding, interaction search, GPU iteration, DART, early stopping, full-data retrain

### Takeaway
On tabular data the biggest gains came from features, not hyperparameters:
- Groupby aggregations including target encoding computed inside folds (Deotte's 1st place with the best 500 of more than 10,000 generated features).
- Count/uniqueness encodings over train+real-test (Santander 1st: LGBM LB .910 → .914 → .921 as uniqueness features were added and fake test rows removed).
- Lag/diff/ratio aggregates and OOF "meta features" (AMEX 11th: +0.001).

Cheap model-side tricks with measured effect: DART plus a very large `min_data_in_leaf` (AMEX 2nd: the key to aligning CV and LB), training on full data with a few hundred to 1000 rounds beyond CV-best, null-importance and adversarial-validation feature pruning, and GPU (cuDF/cuML/XGBoost-GPU/FIL) to make thousands of experiments affordable. Seed averaging is cheap insurance, but AMEX 2nd found "many models with different seed" did not help.

### Cited Findings

**Groupby aggregations and target encoding inside folds; mass feature search on GPU [A]**
- Chris Deotte, Kaggle Playground Feb 2025 (backpack price), 1st:
  - "rapidly generate and test over 10,000 engineered features ... the best 500 discovered features significantly boosted the accuracy of my XGBoost model, securing 1st place".
  - "The most powerful feature engineering technique is groupby aggregations ... groupby(COL1)[COL2].agg(STAT)". "When COL2 is the target column, then we use nested cross-validation to avoid leakage ... this operation is called Target Encoding."
  - Further tricks: per-group histogram bins and quantiles (5…95) as features; "All NANs as Single Base-2 Column"; rounding a key numeric column to several precisions; extracting decimal digits of a float; all pairwise combinations of 8 categoricals (28 new columns); merging the original dataset's group-mean price as an "MSRP" feature; division features such as count/nunique and std/count. — [NVIDIA blog, Deotte Apr 2025](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)
- KGMoN Playbook (Onodera, Viel, Titericz, Deotte, Sep 2025): "In one Kaggle competition, the dataset had eight categorical columns. By combining pairs of them, we created 28 new categorical features". Fast experimentation is "the biggest lever". — [NVIDIA Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- AMEX 14th (Deotte):
  - "RAPIDS cuDF ... groupby aggregation on GPU 10-100x faster". "With 1xV100 GPU, XGB can train 100 models for Nested 10 in 10 K-Fold ... in only 2 hours."
  - Permutation importance with RAPIDS FIL: "shuffle each feature column 10 times for each of 10 folds (and average 100 results)". CPU LightGBM DART models can be converted to FIL. — [AMEX 14th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641)
- AMEX 2nd: brute-forced about 7k features. Aggregations were last/mean/std/median/min/max/nunique; diffs and ratios of the last value against those stats ("std diff worked better than any other"); lags 1, 2, 3, 6, 11; plus EWM, monotonic-increase flags and per-month standardisation. — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)
- AMEX 11th "meta features (most important features!)": assign the target to every pre-aggregation row, train, take OOF predictions, then aggregate those by time window. Single model public went from 0.799 to 0.800. (Idea from DSB2019 2nd.) — [AMEX 11th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786)

**Count / frequency / uniqueness encoding (Santander 2019 "magic") [A]**
- See Q2 for the 1st/3rd/5th details. 2nd place (Onodera): remove fake test → concat train+test → count encoding → count encoding of rounded values → unpivot the 200 vars into 4M rows (one model over var-id and value) → combine the 200 per-feature predictions as odds `9*p/(1-p)`. — [Santander 2nd](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88939)
- 5th: for LGBM, `minmaxscale(-4,4)` the feature and then use `X**count` and `count**X` with the count clipped to [1,3]. A NN with input shape (200, 2) of (value, count) beat LGBM at 0.924 public. — [Santander 5th](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88897)
- 1st: final LGBM 0.92522 public / 0.92332 private. A weight-shared per-feature NN with attention-weighted pooling scored 0.92546 private. Blend 2.1 NN : 1 LGBM gave 0.927 public. — [Santander 1st](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/89003)

**DART, regularisation and early stopping [B]**
- AMEX 2nd:
  - "DART never worked better and here we have proof that in DS 'all depends.'"
  - Best LGBM: `boosting_type='dart'`, lr 0.01, `num_leaves=64`, `min_data_in_leaf=2048`, `feature_fraction=0.2`, `feature_fraction_bynode=0.3`, 17,001 rounds (5,000 for gbdt).
  - "The main thing that helped us align Train / CV with LB was very high 'min_data_in_leaf'."
  - They ran feature-selection experiments with gbdt because it "was very aligned with dart mode but was significantly faster". — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)
- AMEX 11th also used DART, with a callback to keep the best-AMEX-metric iteration. — [AMEX 11th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786)
- Full-data retrain with a fixed round count:
  - AMEX 2nd: "Train on all data without folds splitting and stop just 1000 rounds further then CV showed."
  - Otto 2nd (senkin13): "retrain all train data with fixed round number for test data prediction, the CV matched LB very well". — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637); [Otto 2nd senkin](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839)

**Seed averaging / bagging [B, mixed]**
- KGMoN Playbook: XGBoost across 100 seeds plus retraining on 100% of the data gave additional gains in Predicting Optimal Fertilizers (detail in the companion notes). — [NVIDIA Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- Otto 1st: "single LGBMRanker : LB 0.604; ensemble of 9 LGBMRankers with different hyperparameters : LB 0.605". H&M 3rd's final model is "the average score of 16 models which were trained with different hyperparameters (learning rate, max_depth and even random seed)". — [Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022); [H&M 3rd](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)
- AMEX 14th: "we average 5 seeds per model". — [AMEX 14th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641)
- **Counter:** AMEX 2nd "What didn't work: Stacking by any mean; Many models with different seed and fe order". — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)

**Feature selection that transferred to private LB [B]**
- AMEX 2nd:
  - Drop exactly-zero-importance features (1.5k removed).
  - "Stepped hierarchical permutation importance": add a feature subset, permute, drop features with negative mean over 5 seeds, repeat.
  - Forward selection keeping a subset only if AUC improves by more than 0.0003.
  - Each method alone gave about 0.800 LB; the union and intersection gave 0.801. — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)
- AMEX 11th:
  - Null importance (keep features whose real importance exceeds the mean shuffled-target importance) cut 4300 features to 1300.
  - Adversarial validation (train/private AUC 0.99) identified R_1, D59, S_11 and B_29 to drop.
  - The submission "with risk of time-series changes", excluding features important in adversarial validation, "was the best in privateLB" and took 11th. — [AMEX 11th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347786)

**GPU / fast-iteration engineering [A]**
- Otto 3rd (Deotte): "To find the above 20 covisit matrices, i computed hundreds of covisit matrices and then computed local CV score ... RAPIDS cuDF to make each covisit matrix on GPU in under 1 minute." — [Otto 3rd Deotte](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)
- Otto 2nd: "pandas.merge -> polars.join = 40X faster". "TreeLite to accelerate lightgbm inference speed (2X faster), catboost-gpu is 30X faster than lightgbm-cpu inference" (also stated in H&M 1st). Onodera (Otto 2nd): "If I hadn't used cuDF and cuML, I couldn't manage a lot of experiments." — [Otto 2nd senkin](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839); [H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070); [Otto 2nd Onodera](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382790)
- Otto 1st: "For quick iteration of improvements, I conducted experiments by training with 5% of the data and evaluating with other 10% of the data." — [Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022)
- H&M 1st: a "feature store" saves intermediate feature files so that "exsiting features will not be create again". — [H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)

**What tabular CV deltas mean [B]**
- Home Credit 2024 1st: "differences in CV of around 0.001 to 0.005 had a low correlation with LB, while differences above 0.01 had a high correlation with LB. In addition, CV improvement through model parameter adjustment had a relatively low correlation with LB, and CV improvement through FE had a relatively high correlation with LB." CatBoost was the best model on about 117 categoricals. Tabular transformers (TabNet, FT-Transformer) did not beat a simple DenseLight MLP. — [Home Credit 1st](https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/discussion/508337)

**NN-on-GBDT stacking or second stage [B, mixed]**
- Santander 3rd: switching the blender from linear to a NN (4-layer Conv1D over OOF, count, density, deviation and raw input) "made us jump to 5th on the first try". Santander 5th: "Stacking added around +0.001. NN was significantly better at stacking than any other method." — [Santander 3rd](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88902); [Santander 5th](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88897)
- Counter: AMEX 2nd found no stacking gain, and kept only LR/Lasso for blending. — [AMEX 2nd](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347637)

### Inferences
- GBDT priority order by evidence:
  1. Groupby aggregates and target encoding in nested folds, generated by the thousand on GPU and selected by forward selection or permutation importance.
  2. Count/frequency over the full candidate universe, including real test rows.
  3. OOF meta-features from a finer-grained model.
  4. Strong leaf regularisation (`min_data_in_leaf` in the thousands at 5M+ rows).
  5. Full-data retrain at about 1.0-1.1× the CV-best rounds.
  6. 5-16 seed or hyperparameter bags.

  Use DART only once the feature set is frozen (it is slow).
- Per Home Credit 1st, treat a GBDT CV delta under about 0.005 from hyperparameter tuning as noise. A feature-engineering delta of the same size is more trustworthy.

### Gaps
- I did not retrieve a GBDT write-up that quantifies seed averaging in isolation (N seeds versus 1). The KGMoN 100-seed claim has no number in the excerpt I saw.
- AMEX 1st's write-up content is mostly images. I could not extract its feature-level ablations.

---

## Q4. Ranking / reranking tricks: hard negatives, listwise/lambdarank losses, query-level features, candidate-source flags

### Takeaway
In candidate-rerank pipelines the cheapest large wins are features describing how each candidate was retrieved:
- **Source flags plus rank within each source.** H&M 3rd: 0.02855 → 0.03262. These flags are what made expanding the candidate list help instead of hurt.
- **Stage-1 rank as a feature** (Otto 5th).
- **More, and more diverse, retrieval sources** feeding count features (Otto 3rd: 3 → 20 co-visitation matrices, 0.575 → 0.590 rules-only, +0.011 more with an XGB reranker).

Ranking objectives (CatBoost YetiRank-style ranker, lambdarank variants, graded label gains) give small but consistent +0.0005-0.002 gains over binary GBDT. Negative downsampling to a manageable ratio costs nothing measurable. Hard-negative mining is not a free win: two Eedi top-5 teams found it did not help, or hurt recall.

### Cited Findings

**Candidate-source flags and rank-within-source [A]**
- H&M 3rd (2022): "whether this article is recalled by `strategy_name` and the rank of this article under the `strategy_name` ... These features, plus an expansion of candidates num for each user from dozens to hundred, boost my LB score from 0.02855 to 0.03262 ... if I only increase the recall num and don't adding the recall features, the CV score is very very poor." — [H&M 3rd](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)
- Otto 5th: "Using the ranking of the stage 1 candidate generation model, as a feature of the stage 2 model ... the stage 1 model can score 0.585 on the lb, so using this ranking was the important feature". — [Otto 5th](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382802)
- Otto 1st features: "rank by covisitation matrix at candidate generation", "cosine similarity by NN at candidate generation". Popularity "worked well when ranked". — [Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022)
- Otto 3rd (Deotte): turning each co-visitation matrix's count into a separate candidate feature, then an XGB reranker, gave "Reranker Boost CV and LB +0.011". Adding 17 more co-visit matrices took rules-only from 0.575 to 0.590 and the single XGB to 0.601. — [Otto 3rd Deotte](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)
- FP2021 1st (a span-candidate reranker, the same structure):
  - Over-generate candidates at a low threshold, 3M samples for about 95% mean recall.
  - Add "65% length with the highest probability" sub-span candidates (+0.008).
  - A LightGBM on about 170 features gave +0.036 CV and LB. — [FP2021 1st](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)

**Query-level / relative features [B]**
- Otto 1st session features: "length", "aid dupplication rate", "ts between the last aid and the second last aid". Ablation: removing aid features cost 0.0035 weighted recall@20; removing session features cost 0.0002. — [Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022)
- FPE 2nd stacker: per-essay mean and std of each class probability, plus previous and next discourse predictions (neighbour features). Stacking gave "around 0.004". — [FPE 2nd](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347359)
- H&M 1st "Difference/Ratio" features, e.g. "difference between age and mean age of who purchased item". — [H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)
- LECR 1st's per-topic max/min of retrieval scores and gap-to-top post-processing are in the companion notes.

**Ranking objectives vs binary [B]**
- Otto 2nd (senkin13): "at last days I tried catboost ranker, the improvement surprised me, order model: 0.0007 up, cart model: 0.002 up, clicks model: 0.0012 up". — [Otto 2nd senkin](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382839)
- H&M 2nd: "I managed to write a lambdarankmap objective for LGBMRanker ... and it was slightly better than lambdarank (which uses NDCG objective)". — [H&M 2nd](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324197)
- Otto 5th:
  - Graded relevance "with ranking label gain as Orders (6) -> Carts(3) -> Clicks(1)" gave about +0.0005.
  - "Training all clicks, carts, and orders with a single model ... 0.001 to 0.002 better".
  - Going from 80 to 120 candidates gave about +0.001; 200 "did not help". — [Otto 5th](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382802)
- Otto 1st used LGBMRanker; H&M 1st used a binary LightGBM. Both won, so the objective is second-order. — [Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022); [H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070)
- DAIGT 1st: a deberta-v3-large with pairwise ranking loss scored 0.963 private, used for diversity. ELL 2nd: "Rank loss always help" as an auxiliary loss. — [DAIGT 1st short](https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/470121); [ELL 2nd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369369)

**Negative downsampling [B]**
- H&M 1st: "negative downsampling is must, we found 1 million ~ 2 million negative samples for each week has better performace". H&M 3rd keeps "30*len(pos_samples)" negatives (pos:neg was about 1:300). — [H&M 1st](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324070); [H&M 3rd](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)
- Otto 1st: negative sampling rate "clicks : 5%, carts : 25%, orders : 40%", chosen to fit memory. Otto 5th: "5% negative sampling". — [Otto 1st](https://www.kaggle.com/competitions/otto-recommender-system/discussion/384022); [Otto 5th](https://www.kaggle.com/competitions/otto-recommender-system/discussion/382802)
- Positives only include ground-truth items that the candidate generator retrieved (H&M 3rd: "The pos samples are only the ones which are recalled in the candidates"). — [H&M 3rd](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)

**Hard-negative mining: mixed evidence [B]**
- Eedi 1st summary: "I skipped hard mining during training as it was not helpful for `recall@32`. One important detail was to avoid having multiple demonstrations of a given misconception in a single batch." Reranker side: "For each positive, I sampled 24 negatives". — [Eedi 1st summary](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551402)
- Eedi 5th: "While we perform negative mining, hard negatives were not effective, so we include slightly relaxed samples". Settings: `range_min: 512`, `num_negatives: 2`, CachedMultipleNegativesRankingLoss. — [Eedi 5th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551391)
- Eedi 7th (public 2nd) did use "Hard Negative Sampling(N=100) removed if unseen in training data" for an SFR-Embedding-2 retriever. — [Eedi 7th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551388)

**Listwise LLM reranking with single-token choices [B]**
- Eedi 5th: present 52 candidates labelled A-Z/a-z and read the logit of each letter token. 2 windows (top 104) beat 1 window: fold-4 private 0.581 vs 0.572. A third window gained on local CV but not on LB. Two-letter, alphanumeric and hiragana labels "all performed worse than using just the 52 alphabets". — [Eedi 5th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551391)
- Eedi 7th: voting ensemble of pipelines using reciprocal-rank weights (1/rank summed), vLLM AWQ with `enable_prefix_caching=True`. — [Eedi 7th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551388)
- Eedi 5th: distilled reasoning text ("why the wrong answer"), generated by Qwen2.5-32B and fed to the bi-encoder, gave "cv+0.04". For synthetic data, few-shot examples chosen by embedding-similar misconception names beat random examples by about +0.01 CV. — [Eedi 5th](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/discussion/551391)

### Inferences
- For any candidate-list reranker, the minimum feature set (H&M 3rd and Otto 3rd/5th/1st) is:
  - For every retriever: `in_source_k` (bool), `rank_in_source_k`, `score_in_source_k`.
  - `n_sources_hit`.
  - Within-query relatives: `score − max(score in query)`, `rank / list_len`, z-score within query.
  - Query-level descriptors: list length, top-1 vs top-2 gap, entropy of the score distribution.

  Without the source/rank features, adding more candidates hurts (H&M 3rd's "very very poor" CV).
- Switching the objective (binary → lambdarank/YetiRank/CatBoost ranker) is a 5-minute experiment worth +0.0005-0.002, and it gives a second model for blending.
- Hard negatives: start with random or "relaxed" negatives (Eedi 5th: skip the top 512, sample 2) and many negatives per positive (Eedi 1st: 24). Measure recall@K, not only MRR, before keeping hard-negative mining.

### Gaps
- No retrieved write-up gives an A/B test of LambdaMART versus pointwise within an otherwise identical pipeline other than Otto 2nd's CatBoost-ranker deltas.
- I found no top write-up on molecule or spectrum ranking. These are transfers by analogy.

---

## Q5. Grandmasters' stated favourite tricks (Deotte, Psi, CPMP, Tunguz, KGMoN)

### Takeaway
Across their own write-ups:
- **Chris Deotte:** hill climbing with negative weights, a RAPIDS SVR head on frozen transformer embeddings, dropout-zero / grad-clip / batch-size-1 recipes, KD from GBDT into NNs, and GPU-accelerated brute-force feature or co-visitation search.
- **Psi (Philipp Singer, Team Hydrogen / Danube):** compare only 3-seed averages, pseudo-label pre-train → gold fine-tune, train-mean calibration, full-data retrain, and simple models with diversity judged in the blend.
- **CPMP:** contributed a faster QWK implementation (AES2). His methodological stance is in the companion notes.
- **Bojan Tunguz:** I found no first-hand trick write-up in the retrieved set.

### Cited Findings
- **Deotte, hill climbing with negative weights:** "We allowed hill climbing to use negative weights which boosted CV +0.0010 and private LB +0.0060 versus using positive weights only." Hill climbing picked diverse models, not the best-CV ones, first. — [ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)
- **Deotte, RAPIDS SVR head:** "if we extracted the final hidden layer activations ... and trained a RAPIDS support vector regression model and then ensemble this model with the original model head we could boost all our models' CV and LB. The TensorFlow models boosted LB +0.002 and most PyTorch models boosted LB +0.001." — [CLRP 15th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260800)
  - Reused in ELL: SVR on embeddings "without train on comp data [Medium Impact]" ([ELL 3rd](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)).
  - ELL 1st added three embedding+SVR models (bart-large, all_datasets_v3_roberta-large, bart-large-mnli) to its blend ([ELL 1st](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369457)).
- **Deotte, ensembling spans with Weighted Box Fusion:** individual models averaged about 0.700 CV and the WBF ensemble reached 0.741. — [FP2021 2nd](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389)
- **Deotte, KD from LGBM to transformer** plus nested 10×10 CV for leak-free tuning of the stack. — [AMEX 14th](https://www.kaggle.com/competitions/amex-default-prediction/discussion/347641)
- **Deotte, GPU brute force:** hundreds of co-visit matrices (Otto), 10,000 groupby features (Playground). — [Otto 3rd](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013); [NVIDIA blog](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)
- **Psi, experiment protocol:** compare LR=1e-5 vs 2e-5 with 3 seeds on 1 fold, and "only if the average of the three seeds improves, I would run on all my 5-folds". Nelder-Mead per-target weights with bounds [1, 3] to avoid overfitting. "I have an unselected sub ... with best local CV that would score #2 on private, which has unrestricted weights ... also with negative weights". — [ELL 5th](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)
- **Psi / Team Hydrogen, train-mean calibration:** "log loss metric is always only optimal if the mean prediction per column matches the mean of the target columns ... we added an additional optimization after each model to adjust to the train mean". — [FPE 1st](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347536)
- **Psi (LMSYS 5th):** pre-train on public reward data. Final = the same config trained twice on full data for one epoch, "Each run has swapped training order for response A and B for some tiny diversity", with swapped-order TTA. — [LMSYS 5th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527669)
- **CPMP:** AES2 1st thanks "@cpmpml for sharing a faster QWK calculation". — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- **Ensemble selection by elimination instead of addition (CLRP 4th):** "I started with all of my models and iteratively removed the worst models until I could get down to the GPU time limit. (I found that this worked better than greedily adding models)". The final ensemble had about 38 models. — [CLRP 4th](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258148)
- **Threshold fitting for an ordinal metric (AES2 1st):** "minimize from scikit.optimize with 'Powell', evaluating on 1 - qwk ... I average 15 different starting points". Thresholds are fitted on 3-seed OOFs. "Typical thresholds for 1/2 were 1.7 and for 5/6 even 4.9". — [AES2 1st](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)
- **Per-class threshold / prior scaling as post-processing:**
  - PII 1st: "Using different thresholds for each PII label".
  - PII 2nd: "argmax after scaling down class O probabilities ... consistently 0.02-0.03".
  - CLRP 2nd: piecewise multiplicative scaling by prediction range gave +0.001-0.002. — [PII 1st](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374); [PII 2nd](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497352); [CLRP 2nd](https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/258328)
- **Rule post-processing for degenerate inputs (LMSYS 4th):** an empty response gets fixed probabilities [0.04, 0.88, 0.08] for empty / non-empty / tie; identical responses get a tie with [0.06, 0.06, 0.88]. — [LMSYS 4th](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/529067)

### Inferences
- Deotte's and Psi's habits are mostly about variance control (seeds, full-data refits, bounded weights) and diversity (SVR heads, pooling variants, negative weights). Their measured single-trick gains are about 0.001-0.006, but these tricks hold up on private LB.

### Gaps
- I found no Kaggle discussion write-up by Bojan Tunguz or by CPMP within these competitions that lists favourite training tricks with numbers. Their contributions in this sample are tooling (CPMP's QWK) or interviews (covered in the companion notes).
- I did not search X/Twitter threads. The Kaggle API was the primary source.

---

## Q6. Evidence-ranked shortlist and transfer to (1) CASMI 2026 and (2) Name That LLM

### Takeaway
Ranked by evidence strength times cheapness, the tricks most worth trying first:
1. Pseudo-label or self-distil with soft labels, fold-wise, pre-train → fine-tune.
2. Candidate-source flags plus rank-within-source in rerankers.
3. Correct input formatting and pooling.
4. AWP or FGM.
5. Dropout 0 for regression.
6. Full-data refit plus 3-seed averaging.
7. Count/frequency and groupby/target-encoding features computed in nested folds over the full (train+test) universe.
8. Order-swap or invariance TTA plus full-swap training.
9. Train/infer length sweep.
10. Hill climbing with negative weights, and per-class threshold or prior scaling on OOF.

CASMI benefits mostly from items 2, 7, 10 and the ranking objective. Name That LLM benefits mostly from items 1, 3, 4, 5, 6, 9, 10, the LoRA defaults, and the DAIGT-style data and test-adaptation tricks.

### Cited Findings (with the measured anchor for each ranked item)
1. Soft PL / KD:
   - +0.004-0.007 (AES2 1st, [link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516791)).
   - CV 0.4581 → 0.4476 (CLES 2nd, [link](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)).
   - KD +0.005-0.01 (PII 1st, [link](https://www.kaggle.com/competitions/pii-detection-removal-from-educational-data/discussion/497374)).
   - KD accuracy 0.7051 → 0.7135 (WSDM 3rd, [link](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567584)). The author says a 14B self-distillation matched the 72B teacher.
2. Source flags + rank: 0.02855 → 0.03262 (H&M 3rd, [link](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/discussion/324129)). Reranker +0.011 over rules (Otto 3rd, [link](https://www.kaggle.com/competitions/otto-recommender-system/discussion/383013)).
3. Input/pooling:
   - Head-mask pooling "biggest impact" (CLES 2nd, [link](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446573)).
   - `\n` handling 0.595 → 0.630 (FP2021 4th, [link](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313330)).
   - Source tag QWK 0.8370 → 0.8446 (AES2 4th, [link](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/discussion/516639)).
4. AWP/FGM: +0.01 CV / +0.003 LB (FP2021 1st, [link](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313177)); 0.005-0.01 (FPE 3rd, [link](https://www.kaggle.com/competitions/feedback-prize-effectiveness/discussion/347433)); FGM 0.002-0.005 (USPPPM 2nd, [link](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332234)).
5. Dropout 0: "[High Impact]" (ELL 3rd, [link](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369609)), plus 5 other top teams (Q1).
6. Full-data refit and seeds: USPPPM 1st full-train weight ×2 beat 1:1 ([link](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332243)); 3-seed protocol (ELL 5th, [link](https://www.kaggle.com/competitions/feedback-prize-english-language-learning/discussion/369578)).
7. Count/TE features: Santander count encoding to 0.914x / 0.92x ([5th](https://www.kaggle.com/competitions/santander-customer-transaction-prediction/discussion/88897)); 10k-feature groupby search for a 1st place ([NVIDIA](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/)).
8. Swap TTA and full swap:
   - TTA about 0.007 (LMSYS 3rd, [link](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527766)).
   - Full swap 0.003 (LMSYS 2nd, [link](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527685)).
   - Uncertainty-gated TTA on only the 33% most uncertain samples (WSDM 1st, [link](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/569902)).
   - WSDM 2nd ran the swapped model only where |pA−pB| < 0.8, which raised private from 0.708 to 0.716 together with its other changes ([link](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567948)).
9. Length sweep: 0.4984 → 0.4867 (CLES 4th, [link](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/446524)); train-2048/infer-640 "[High Impact]" (ELL 3rd).
10. Negative-weight hill climbing +0.006 private (ELL 3rd). Per-class thresholds (PII 1st/2nd). Powell thresholds averaged over 15 starts (AES2 1st).

### Inferences: transfer to CASMI 2026 (MS/MS molecule ID, MRR@25, GBDT reranker over candidate lists, analog generation, about 40-50% novel test molecules)
- **Candidate-source features (H&M 3rd / Otto 5th pattern).**
  - Retrievers to treat as sources: formula match, mass-window DB lookup, spectral-library similarity, learned spectrum→fingerprint embedding, analog generator, MS-Finder/SIRIUS/CSI:FingerID-style scorers.
  - Per source: `hit_k`, `rank_k`, `score_k`, `score_k − max_query(score_k)`, plus `n_sources_hit`.
  - Do this before expanding candidate lists. H&M 3rd shows that more candidates without these features hurts.
- **Analog-generated candidates.** Flag them as their own source with their own rank and score. Eedi 7th's retriever note, "Hard Negative Sampling(N=100) removed if unseen in training data", shows that top teams handled never-seen candidates separately in training. Pair this with the prior-boost idea in the companion notes (Eedi 3rd).
- **Count / frequency encodings across the whole query universe (Santander pattern).**
  - How many queries in train+test contain this candidate InChIKey-14 or scaffold in their list.
  - How often its formula appears among candidates.
  - Candidate-popularity ranks within the database.
  - Compute these over real test queries too, never over padded or fake rows (Santander 3rd/5th).
- **Groupby / target-encoding features in nested folds (Deotte).** Examples: group by candidate scaffold, adduct, or instrument, aggregating `is_true`, scores and ranks. This needs nested CV grouped by query molecule or scaffold, exactly as Deotte warns, because target encoding of candidate-level labels leaks quickly when the same molecule appears in many queries.
- **Ranking objective.**
  - Try LightGBM `lambdarank` and CatBoost `YetiRank`/`QueryRMSE` against binary log-loss, which gave +0.0007-0.002 in Otto 2nd. MRR is closest to lambdarank with a truncation level of about 25.
  - Use graded gains only if partial-credit labels exist (e.g. same InChIKey-14 but different stereo = graded 1 vs exact = 2, as in Otto 5th's 6/3/1).
  - Downsample negatives per query (Otto 1st: 5-40%) and keep positives only if they were retrieved (H&M 3rd).
- **Variance control.** Average 5-16 seeds or hyperparameter variants of the reranker (Otto 1st 0.604 → 0.605; H&M 3rd 16 models). Refit on all queries with rounds ≈ CV-best × 1.0-1.1 (AMEX 2nd, Otto 2nd). Treat MRR deltas under about 1 SE across folds as noise (Home Credit 1st).
- **Feature pruning for shift.** Run adversarial validation on query-level features (train vs test spectra) and drop the top shift features in one candidate submission (AMEX 11th: this was the best private).
- **GBDT→NN distillation (AMEX 14th).** Deotte's pattern: soft-label a neural reranker or spectrum encoder with GBDT OOF scores on train and test candidates, then fine-tune on hard labels. This is a cheap diversity member.
- **Do not lean on hard-negative mining for the spectrum encoder without checking recall@K** (Eedi 1st and 5th). Prefer many in-batch negatives and dedupe the same molecule within a batch (Eedi 1st: "avoid having multiple demonstrations of a given misconception in a single batch").

### Inferences: transfer to Name That LLM (18-way LLM authorship, balanced accuracy, LoRA LLM features + stylometry, about 2-3 passages per prompt in test)
- **LoRA defaults.** r=64 on all linear modules; alpha 16-128 makes no measurable difference (LMSYS 1st/3rd/4th). Answer-token-only loss or a reinitialised MLP head (LMSYS 2nd, WSDM 3rd). LR sweep: flat until it diverges, so find the cliff (WSDM 3rd: collapse at 14e-5 for 14B). Average or merge fold and seed LoRAs instead of running k models at inference (LMSYS 1st, WSDM 1st/3rd). Calibrate quantisation at the inference sequence length (WSDM 3rd).
- **Self-distillation / soft labels.** Retrain the LoRA classifier on its own 5-fold OOF soft labels at T≈5 with soft weight 0.9 (WSDM 3rd: +0.008 accuracy from KD at T=5, w=0.9; the author reports 14B self-distillation matched a 72B teacher). It also cleans label noise, e.g. near-identical outputs from sibling models.
- **Source/meta tags in the input (AES2 4th).** If training passages come from different generation pipelines (prompt sets, temperature regimes), prefix a tag and train a source head, then set the tag to the test-like value at inference. Expect a gain if sources score differently.
- **DAIGT data tricks.**
  - Random-typo and normalisation augmentation, so the model can't key on surface artefacts (DAIGT 4th).
  - Non-instruction-tuned and varied decoding for any synthetic data you add (DAIGT 7th).
  - Hard-example iterative selection: each round add the most-wrong samples per class (DAIGT 3rd).
  - Large-corpus pre-train on "which generator" pairs, then fine-tune on the competition domain (DAIGT 2nd: 0.967 → 0.983 private).
- **Test-time adaptation.** A tokenizer and TF-IDF vocabulary fitted on train+test text (DAIGT 1st) is cheap for the stylometry branch. Confident test pseudo-labels (p>0.99) into the TF-IDF/GBDT branch were +0.034 private in DAIGT 3rd. MLM on a tiny test set can fail outright (DAIGT 7th, 0.689), so ablate it.
- **Group-level joint decoding.** With 2-3 passages per prompt, average the passage logits within a prompt group if the passages share an author. If they come from different models, run assignment-style decoding: Hungarian matching over the group's 18-class probabilities when passages in a group are known to come from different LLMs. This is the analogue of LECR's assignment post-processing and FPE 2nd's per-essay mean/std stacker features. Verify the group structure on train first.
- **Balanced-accuracy post-processing.** Fit per-class multiplicative prior weights or thresholds on OOF with Powell from multiple starts, using 3-seed OOFs and bounded weights (AES2 1st; PII 1st per-label thresholds; PII 2nd class-scaling 0.02-0.03; Psi's [1, 3] weight bounds). Balanced accuracy rewards moving predictions toward minority or confusable classes, just as QWK did in AES2.
- **Regularisation knobs.** Dropout 0 in the backbone and LoRA dropout 0-0.05. AWP from epoch 2 for the DeBERTa stylometry branch (FP/USPPPM evidence), but it is neutral on some small datasets (CLES, AES2), so ablate. Compare 3-seed averages only.
- **Length.** Sweep train vs inference max_len on OOF. DAIGT favoured training short (256-512) and inferring long (1024-1512), which fits authorship cues that are local and stylistic.
- **Ensembling.** Hill climbing with negative weights over LoRA-LLM, DeBERTa, TF-IDF/GBDT and perplexity-feature members (ELL 3rd: +0.006 private). Use RAPIDS SVR or logistic heads on frozen LLM embeddings as extra cheap members (CLRP 15th / ELL 1st).

### Gaps
- None of the retrieved write-ups is an authorship-attribution or MS/MS competition. All transfers are by structural analogy (candidate reranking; multi-class text-source classification), so each needs an ablation on the target data.
- I found no Kaggle evidence on Hungarian or assignment decoding for multi-class groups beyond the LECR precedent cited in the companion notes. Whether test prompt groups contain distinct authors in Name That LLM must be checked against the competition data.
