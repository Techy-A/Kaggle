"""OLMo-2-13B LoRA member: trunk loading, the LoRA wrapper and pooled features, shared by training and inference.

What: the first `keep` decoder layers of allenai/OLMo-2-1124-13B (Apache-2.0), base weights 4-bit NF4 (fp16 compute) with
embeddings, norms and the residual stream in fp32, final norm dropped; LoRA kept unmerged (y = W x + B A x alpha / r);
features = mean hidden state (first token left out) at the pooled layers, one LayerNorm each, as in nishantdahal's 14B
members. `olmo_features` is the inference entry point; its contract is theirs (meta.json, adapter/ in peft key layout,
heads.safetensors), plus cfg["quant"].
Why: 09_train_olmo13b.py trains with this module and the 0.636 notebook variant embeds it verbatim, so test features
are computed by the same code and the same NF4 base as in training. Merging into an fp16 base, as nishant's code
does, would read a base the heads never saw (the NF4 rounding is not in the adapter).
Output: none (a module). Usage: from olmo13b_member import olmo_features; Z, heads = olmo_features(ckpt, base, texts,
8, LABELS); check: lab/tests/check_olmo13b.py.
"""
import gc
import json
import math
import os

import numpy as np
import torch
from torch import nn

OLMO_SPEC = dict(model_type='olmo2', hidden_size=5120, num_hidden_layers=40)   # find_base() spec in the notebook


class LoRALinear(nn.Module):
    """y = W x + (B A dropout(x)) * alpha / r, with W frozen (nn.Linear or bnb Linear4bit); A and B in fp32."""

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop, self.scale = nn.Dropout(dropout), alpha / r

    def forward(self, x):
        return self.base(x) + (self.drop(x).to(self.A.dtype) @ self.A.T @ self.B.T).to(x.dtype) * self.scale


def load_trunk(base_dir, keep, quant, device, compute_dtype='float16'):
    """The first `keep` layers; quant='nf4' quantizes every linear layer, everything else stays fp32."""
    from transformers import AutoConfig, AutoModel
    mc = AutoConfig.from_pretrained(base_dir)
    mc.num_hidden_layers, mc.use_cache = keep, False
    if getattr(mc, 'layer_types', None) is not None:
        mc.layer_types = mc.layer_types[:keep]
    kw = dict(config=mc, dtype=torch.float32)
    if quant == 'nf4':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
                                                       bnb_4bit_use_double_quant=False)
        kw['device_map'] = {'': device}
    else:
        assert quant is None, quant
    model = AutoModel.from_pretrained(base_dir, **kw)
    if quant is None:
        model.to(device)
    assert len(model.layers) == keep
    model.norm = nn.Identity()   # features read the raw residual stream, as in the 14B members
    model.requires_grad_(False)
    return model


def attach_lora(model, r, alpha, dropout):
    """Wrap every linear layer of the decoder layers; returns {module name: LoRALinear} (names as peft keys use)."""
    lora = {}
    for name, m in list(model.named_modules()):
        for child, sub in list(m.named_children()):
            if isinstance(sub, nn.Linear) and name.startswith('layers.'):
                setattr(m, child, LoRALinear(sub, r, alpha, dropout))
                lora[f'{name}.{child}'] = getattr(m, child)
    return lora


def adapter_state(lora):
    """peft key layout, as nishant's merge_lora reads it."""
    sd = {}
    for k, m in lora.items():
        sd[f'base_model.model.{k}.lora_A.weight'] = m.A.detach().float().cpu().contiguous()
        sd[f'base_model.model.{k}.lora_B.weight'] = m.B.detach().float().cpu().contiguous()
    return sd


@torch.no_grad()
def load_adapter(lora, sd):
    used = 0
    for k, m in lora.items():
        m.A.copy_(sd[f'base_model.model.{k}.lora_A.weight'])
        m.B.copy_(sd[f'base_model.model.{k}.lora_B.weight'])
        used += 2
    assert used == len(sd), f'adapter has {len(sd)} tensors, the trunk takes {used}'


def make_norms(H, n, device):
    return nn.ModuleList([nn.LayerNorm(H) for _ in range(n)]).to(device).float()


def encode(tok, texts, max_len, eos_id):
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=max_len)['input_ids']
    return [x + [eos_id] for x in ids]


def pool(model, tok, seqs, layers, norms, device):
    """Pooled features of a batch of token lists: mean over tokens (first one left out), LayerNorm per layer."""
    enc = tok.pad({'input_ids': seqs}, return_tensors='pt').to(device)
    hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
    m = enc['attention_mask'].unsqueeze(-1).float()
    m[:, 0] = 0   # the first token soaks up attention whatever it is
    den = m.sum(1).clamp(min=1)
    return torch.cat([nm((hs[l].float() * m).sum(1) / den) for l, nm in zip(layers, norms)], -1)


def tokenizer(base_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = 'right'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def features(model, tok, ids, layers, norms, bs, device):
    """Z for a list of token lists, length-sorted batches; a non-finite batch is redone one passage at a time."""
    order = np.argsort([len(x) for x in ids], kind='stable')
    Z = np.zeros((len(ids), norms[0].normalized_shape[0] * len(layers)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        z = pool(model, tok, [ids[i] for i in idx], layers, norms, device)
        if not torch.isfinite(z).all():
            z = torch.cat([pool(model, tok, [ids[i]], layers, norms, device) for i in idx])
            assert torch.isfinite(z).all(), 'non-finite features even one passage at a time'
        Z[idx] = z.cpu().numpy()
    return Z


@torch.no_grad()
def olmo_features(ckpt_dir, base_dir, texts, bs, labels, device='cuda:0'):
    """Pooled features Z (one row per passage) and the checkpoint's heads, like the notebook's lora_features."""
    from safetensors.torch import load_file
    meta = json.load(open(os.path.join(ckpt_dir, 'meta.json')))
    assert meta['labels'] == list(labels), "label order differs from the checkpoint's heads"
    cfg = meta['cfg']
    tok = tokenizer(base_dir)
    model = load_trunk(base_dir, cfg['keep'], cfg['quant'], device, cfg['compute_dtype'])
    acfg = json.load(open(os.path.join(ckpt_dir, 'adapter', 'adapter_config.json')))
    lora = attach_lora(model, acfg['r'], acfg['lora_alpha'], 0.0)
    load_adapter(lora, load_file(os.path.join(ckpt_dir, 'adapter', 'adapter_model.safetensors')))
    model.eval()
    heads = load_file(os.path.join(ckpt_dir, 'heads.safetensors'))
    norms = make_norms(model.config.hidden_size, len(cfg['layers']), device)
    norms.load_state_dict({k[len('norms.'):]: v for k, v in heads.items() if k.startswith('norms.')})
    eos_id = cfg['eos_id']
    print(f"OLMo trunk of {cfg['keep']} layers ({cfg['quant']}), end token {eos_id}, tokenizer's {tok.eos_token_id}",
          flush=True)
    Z = features(model, tok, encode(tok, texts, cfg['max_len'], eos_id), cfg['layers'], norms, bs, device)
    del model, lora
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Z, {k: v.float().numpy() for k, v in heads.items() if not k.startswith('norms.')}
