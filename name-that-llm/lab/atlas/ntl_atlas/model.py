"""Our ATLAS scorer for Name That LLM: one frozen causal LM turned into 18 source-conditioned LMs (arXiv:2610.19583).

Written from scratch for this repo (see lab/atlas_rebuild_spec.md). Per source s (one per label):
- every attention projection (q, k, v, o of every layer) gets a LoRA delta B[s] @ A, where the down-projection A is shared
  by all sources and the up-projection B[s] belongs to source s;
- one learned prefix embedding control[s] is put in front of the passage.
Training is next-token NLL of a passage under its own source. Scoring computes the per-token log-likelihood under all 18
sources, then the clipped pairwise ("prod") readout averaged over temperatures, normalised over sources.

The 0.636 notebook calls exactly: load_model(base, labels, cfg, device='auto') -> (model, tokenizer);
model.load_trainable(state); model.eval(); model.device; model.score(ids, mask, source_chunk=c) -> (probs, raw).
Trainable tensor names: base.model.layers.<i>.self_attn.<q|k|v|o>_proj.<a|b> and control.
"""
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj')


class Router:
    """The source index of every row in the current forward pass, shared by all routed layers."""
    src = None


class RoutedLoRA(nn.Module):
    """y = W x + scale * B[src] (A dropout(x)), with W frozen, A shared and B per source (fp32 adapter maths)."""

    def __init__(self, base, n_src, rank, alpha, dropout, init_std, router):
        super().__init__()
        self.base, self.router, self.scale = base, router, alpha / rank
        dev = base.weight.device
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.b = nn.Parameter(init_std * torch.randn(n_src, base.out_features, rank, device=dev, dtype=torch.float32))
        self.drop = nn.Dropout(dropout)
        base.requires_grad_(False)

    def forward(self, x):
        src = self.router.src
        if src is None or src.numel() != x.shape[0]:
            raise ValueError('every row needs a source index')
        y = self.base(x)
        low = self.drop(x.float()) @ self.a.t()                          # (n, t, r)
        up = self.b[src.to(self.b.device)]                                # (n, out, r)
        return y + (self.scale * torch.bmm(low, up.transpose(1, 2))).to(y.dtype)


def prod_readout(lp, valid, temperatures):
    """lp (n, S, T) token log-probs per source, valid (n, T) mask -> (n, S) non-negative evidence.
    For source k and token t the disadvantage is sum_j max(lp_j - lp_k, 0); evidence is the mean over valid tokens of
    exp(-disadvantage / temp), averaged over the temperatures."""
    lp = lp.float()
    dis = torch.relu(lp[:, None, :, :] - lp[:, :, None, :]).sum(2)     # [n, k, t]: sum over j of (lp_j - lp_k)+
    w = valid.float()[:, None, :]
    den = w.sum(-1).clamp_min(1)
    return torch.stack([(torch.exp(-dis / float(t)) * w).sum(-1) / den for t in temperatures]).mean(0)


def to_probs(raw):
    if not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('ATLAS evidence must be finite and non-negative')
    raw = raw.clamp_min(1e-12)
    return raw / raw.sum(-1, keepdim=True)


class AtlasModel(nn.Module):
    def __init__(self, base, labels, config):
        super().__init__()
        self.base, self.labels, self.config, self.router = base, list(labels), dict(config), Router()
        base.requires_grad_(False)
        base.config.use_cache = False
        n_src, wrapped = len(self.labels), 0
        for mod in list(base.modules()):
            for leaf in TARGETS:
                child = getattr(mod, leaf, None)
                if isinstance(child, nn.Linear):
                    setattr(mod, leaf, RoutedLoRA(child, n_src, config['rank'], config['alpha'], config['dropout'],
                                                  config['source_b_init_std'], self.router))
                    wrapped += 1
        if not wrapped:
            raise ValueError('no attention projections found')
        emb = base.get_input_embeddings().weight
        with torch.no_grad():   # prefix init: the mean token embedding plus 1% of the embedding std as noise, per source
            total = torch.zeros(emb.shape[1], device=emb.device)
            sq = torch.zeros((), device=emb.device)
            for rows in emb.split(4096):
                rows = rows.float()
                total += rows.sum(0)
                sq += rows.square().sum()
            mean = total / emb.shape[0]
            std = (sq / emb.numel() - mean.square().mean()).clamp_min(0).sqrt()
            start = mean.expand(n_src, 1, -1) + 0.01 * std * torch.randn(n_src, 1, emb.shape[1], device=emb.device)
        self.control = nn.Parameter(start.contiguous())

    @property
    def device(self):
        return self.control.device

    def trainable_state(self):
        return {k: p.detach().cpu().contiguous() for k, p in self.named_parameters() if p.requires_grad}

    def load_trainable(self, state):
        own = dict((k, p) for k, p in self.named_parameters() if p.requires_grad)
        if set(own) != set(state):
            raise ValueError(f'adapter keys differ: {sorted(set(own) ^ set(state))[:5]}')
        with torch.no_grad():
            for k, p in own.items():
                v = state[k]
                if v.shape != p.shape or not torch.isfinite(v).all():
                    raise ValueError(f'bad adapter tensor {k}')
                p.copy_(v.to(p.device, p.dtype))

    def token_logp(self, ids, mask, src):
        """(n, L-1) log p(x_t | control[src], x_<t) for t = 1..L-1 (the first passage token is not scored)."""
        self.router.src = src
        tok = self.base.get_input_embeddings()(ids)
        pre = self.control[src.to(self.control.device)].to(tok.device, tok.dtype)
        x = torch.cat([pre, tok], 1)
        att = torch.cat([torch.ones_like(mask[:, :1]), mask], 1)
        h = self.base.model(inputs_embeds=x, attention_mask=att, use_cache=False).last_hidden_state
        h, y = h[:, 1:-1], ids[:, 1:]   # outputs at x_0..x_{L-2} predict x_1..x_{L-1}
        head = self.base.get_output_embeddings()

        def chunk_lp(hc, yc):
            logits = head(hc.to(head.weight.device)).float()
            return -F.cross_entropy(logits.flatten(0, 1), yc.to(logits.device).flatten(), reduction='none').view(yc.shape)

        step, out = self.config['head_chunk_tokens'], []
        for s in range(0, h.shape[1], step):
            hc, yc = h[:, s:s + step], y[:, s:s + step]
            if self.training and torch.is_grad_enabled():
                out.append(checkpoint(chunk_lp, hc, yc, use_reentrant=False))
            else:
                out.append(chunk_lp(hc, yc))
        return torch.cat(out, 1).to(ids.device)

    def loss(self, ids, mask, src):
        """Mean over rows of the per-row mean token NLL under the row's own source."""
        lp = self.token_logp(ids, mask, src)
        v = mask[:, 1:].float()
        return -((lp * v).sum(1) / v.sum(1).clamp_min(1)).mean()

    @torch.no_grad()
    def score(self, ids, mask, source_chunk=None):
        """(probs, raw), both (n, 18) in self.labels order; `source_chunk` sources share one forward pass."""
        was = self.training
        self.eval()
        try:
            n, S = len(ids), len(self.labels)
            c = source_chunk or self.config['source_chunk_size']
            parts = []
            for s0 in range(0, S, c):
                srcs = torch.arange(s0, min(S, s0 + c), device=ids.device)
                lp = self.token_logp(ids.repeat(len(srcs), 1), mask.repeat(len(srcs), 1), srcs.repeat_interleave(n))
                parts.append(lp.view(len(srcs), n, -1))
            raw = prod_readout(torch.cat(parts).transpose(0, 1), mask[:, 1:], self.config['prod_temperatures'])
            return to_probs(raw), raw
        finally:
            self.train(was)


def load_model(path, labels, config, training=False, device='cuda', quant=None):
    """Frozen base + fresh adapters, and the base tokenizer (right padding). device='auto' spreads the fp16 base over all
    GPUs (two T4s at inference) and fails if any of it lands off-GPU. quant='nf4' loads the base 4-bit (training on one T4)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = 'right'
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, config.get('base_dtype', 'float16'))
    kw = dict(dtype=dtype, attn_implementation='sdpa')
    gpu = torch.cuda.is_available()
    if device == 'auto' and not gpu and os.environ.get('NTL_ATLAS_ALLOW_CPU') != '1':
        raise RuntimeError('ATLAS needs a GPU')
    if device == 'auto' and gpu:
        kw['max_memory'] = {i: int(torch.cuda.get_device_properties(i).total_memory - 3 * 2**30)
                            for i in range(torch.cuda.device_count())}
    if device != 'cpu' and (device != 'auto' or gpu):
        kw['device_map'] = device
    if quant == 'nf4':
        from transformers import BitsAndBytesConfig
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_compute_dtype=dtype)
    base = AutoModelForCausalLM.from_pretrained(path, **kw)
    if device == 'auto' and gpu and any(p.device.type != 'cuda' for p in base.parameters()):
        raise RuntimeError('the fp16 base does not fit on the GPUs (two T4s or one >= 24 GB GPU needed)')
    model = AtlasModel(base, labels, config)
    if training:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.train()
    else:
        model.eval()
    return model, tok
