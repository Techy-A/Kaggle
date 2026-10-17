---
tags: [study, nlp, transformers]
---
# 03 Transformers & LoRA

Back to [[00 Study Guide Home]].

> **Core idea:** a pre-trained transformer already "knows" language. Fine-tuning only nudges it. LoRA does the nudge with a tiny number of trainable weights.

## 1. What to know about transformers
- **Self-attention:** each token builds its representation from all the others: softmax(QKᵀ/√d)·V.
- **Two families:**
  - Encoders (DeBERTa, RoBERTa) produce pooled embeddings and suit classification.
  - Decoders (Llama, Qwen, Mistral) generate text. They can classify through the last token's hidden state or the next-token log-probs.
- **Tokenisation matters for authorship.** Different LLMs favour different tokens, punctuation and phrasings. That signal is what NTL is about.

## 2. LoRA (low-rank adaptation)
![[lora.png]]

- The weight W stays frozen. We learn ΔW = A·B with rank r ≪ d.
- Typical settings:
  - r = 8–64;
  - α (the scaling) about 2r;
  - targets: the attention projections `q,k,v,o`, sometimes also the MLP.
- **QLoRA:** load the base model in 4-bit and train LoRA on top. A 7–8B model then fits on one T4 or P100 (16 GB).
- **Risks:**
  - overfitting small data: watch grouped CV, not train loss;
  - learning-rate sensitivity: try 1e-4 to 2e-4;
  - a mismatch between the training and inference prompt formats.

## 3. Kaggle-specific tips
- Kaggle T4 ×2: use `device_map='auto'` or DDP; fp16, not bf16, on T4.
- Inference time is the limit: batch by length, truncate long passages and cache tokenisation.
- Save the LoRA adapters (a few MB) as a Kaggle dataset and load them offline in the submission notebook.

## In our comps
- **NTL:**
  - The public line uses "lora3": several LoRA heads on top of an LLM. Its probabilities feed the Claude gates and the Sinkhorn balancing ([[06 Calibration & Label Shift]]).
  - Week 5 job: train our own small LoRA with **grouped-by-prompt** CV. That shows whether the public adapters overfit.

## Exercises
1. Count the trainable parameters of LoRA r = 16 on `q,v` of an 8B model with 32 layers and d = 4096. Compare with the full model.
2. Fine-tune DeBERTa-v3-small on NTL with grouped CV. What balanced accuracy does an encoder baseline reach?

## Resources
- Vaswani et al., *Attention Is All You Need* (2017).
- Hu et al., *LoRA* (2021); Dettmers et al., *QLoRA* (2023).
- Tunstall, von Werra & Wolf, *NLP with Transformers* (O'Reilly).
- Hugging Face PEFT docs.
