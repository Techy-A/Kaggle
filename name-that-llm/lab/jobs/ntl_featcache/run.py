# Kaggle job: cache frozen LLM features of every NTL train passage, once, for CPU experiments in name-that-llm/lab.
# Same models, layers and pooling as the 0.556 blend's LM heads (code adapted from Lucas Boesen's notebook, Apache 2.0):
#   q3_8b: Qwen3-8B-Base, mean-pooled hidden states of layers 7 and 22 (model truncated to 23 layers);
#   glm9b: GLM-4-9B-0414 (lucasboesen's 25-layer truncation), layers 8, 3 and the clustering layer 24.
# -> /kaggle/working/ntl_feats.npz (float16: q3_8b (n, 8192), glm9b (n, 12288), ids)
import glob, json, os

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

train = pd.read_csv(glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True)[0])
TR = train['text'].fillna('').astype(str).tolist()


def find_model_dir(model_type, hidden):
    for cfg in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if c.get('model_type') == model_type and c.get('hidden_size') == hidden:
            return os.path.dirname(cfg)


def load_truncated(path, keep):
    cfg = AutoConfig.from_pretrained(path)
    cfg.num_hidden_layers = keep
    if getattr(cfg, 'layer_types', None):
        cfg.layer_types = cfg.layer_types[:keep]
    return AutoModel.from_pretrained(path, config=cfg, dtype=torch.float16, device_map='auto').eval()


@torch.no_grad()
def embed_layers(model, tok, texts, layers, bs=8, max_len=512):
    """Mean-pooled hidden_states[L] for each L in layers, concatenated (the notebook's embed_layers, pool='mean')."""
    H = model.config.hidden_size
    out = np.zeros((len(texts), len(layers) * H), dtype=np.float32)
    order = np.argsort([len(t) for t in texts])   # length-sorted batches: less padding
    dev = next(model.parameters()).device
    for b0 in range(0, len(texts), bs):
        idx = order[b0:b0 + bs]
        batch = [(tok.bos_token + texts[i]) if tok.bos_token else texts[i] for i in idx]
        enc = tok(batch, return_tensors='pt', padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states
        if not all(torch.isfinite(hs[L]).all() for L in layers):   # fp16 overflow guard, as in the notebook
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True).hidden_states
        m = enc['attention_mask'].float().unsqueeze(-1)
        for li, L in enumerate(layers):
            h = hs[L].float()
            out[idx, li * H:(li + 1) * H] = ((h * m.to(h.device)).sum(1) / m.sum(1).clamp(min=1).to(h.device)).cpu().numpy()
    return np.nan_to_num(out)


feats = {}
for key, mt, hid, layers in [('q3_8b', 'qwen3', 4096, [7, 22]), ('glm9b', 'glm4', 4096, [8, 3, 24])]:
    path = find_model_dir(mt, hid)
    assert path, f'{key}: model not attached'
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    tok.pad_token = tok.pad_token or tok.eos_token
    model = load_truncated(path, max(layers) + 1)
    feats[key] = embed_layers(model, tok, TR, layers).astype(np.float16)
    print(key, path, feats[key].shape, flush=True)
    del model
    torch.cuda.empty_cache()
np.savez_compressed('/kaggle/working/ntl_feats.npz', ids=train['id'].values, **feats)
print('saved', {k: v.shape for k, v in feats.items()})
