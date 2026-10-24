"""CPU check of our ATLAS package on a tiny random Qwen3 (with the real Qwen3 tokenizer): readout maths, per-row routing,
padding and chunk invariance, the (n, 18) probability contract and the adapter round trip.

Usage: python lab/atlas/check_atlas.py [--make-tiny DIR]   (DIR: also write the tiny model there, for 08's smoke test)
"""
import argparse
import itertools
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('NTL_ATLAS_ALLOW_CPU', '1')
from ntl_atlas.model import load_model, prod_readout  # noqa: E402
from ntl_rlcd.data import LABELS  # noqa: E402

CFG = dict(rank=4, alpha=8, dropout=0.05, source_b_init_std=1e-4, max_length=512, source_chunk_size=2,
           head_chunk_tokens=5, prod_temperatures=[1.0, 2.0, 4.0, 8.0], base_dtype='float32')


def make_tiny(path):
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-8B-Base')
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, tie_word_embeddings=False)
    Qwen3ForCausalLM(cfg).save_pretrained(path)
    tok.save_pretrained(path)
    return path


def naive_readout(lp, valid, temps):
    n, S, T = lp.shape
    out = torch.zeros(n, S)
    for i, k in itertools.product(range(n), range(S)):
        acc = 0.0
        for t_ in temps:
            num = sum(math.exp(-sum(max(float(lp[i, j, t] - lp[i, k, t]), 0) for j in range(S)) / t_)
                      for t in range(T) if valid[i, t])
            acc += num / max(int(valid[i].sum()), 1)
        out[i, k] = acc / len(temps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--make-tiny')
    a = ap.parse_args()
    torch.manual_seed(1)
    lp, valid = torch.randn(2, 5, 7), torch.tensor([[1] * 7, [1] * 4 + [0] * 3])
    assert torch.allclose(prod_readout(lp, valid, CFG['prod_temperatures']), naive_readout(lp, valid, CFG['prod_temperatures']),
                          atol=1e-5), 'prod readout'
    eq = prod_readout(torch.zeros(1, 18, 3), torch.ones(1, 3), [1.0])
    assert torch.allclose(eq, torch.ones(1, 18)), 'equal sources -> evidence 1 each'

    tiny = a.make_tiny or tempfile.mkdtemp()
    if not os.path.exists(os.path.join(tiny, 'config.json')):
        make_tiny(tiny)
    torch.manual_seed(2)
    model, tok = load_model(tiny, LABELS, CFG, device='auto')
    names = dict(model.trainable_state())
    assert len(names) == 2 * 4 * 2 + 1 and names['control'].shape == (18, 1, 32), 'trainable layout'
    assert names['base.model.layers.0.self_attn.k_proj.b'].shape == (18, 16, 4), 'per-source B (kv dim 16)'
    with torch.no_grad():   # make the sources clearly different
        for k, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.3 * torch.randn_like(p))
    texts = ['The quick brown fox jumps over the lazy dog.', 'Short one.', 'A passage of moderate length, with commas; and more.']
    enc = tok(texts, truncation=True, max_length=512)['input_ids']
    pk = tok.pad({'input_ids': enc}, padding=True, return_tensors='pt')
    P, R = model.score(pk['input_ids'], pk['attention_mask'], source_chunk=3)
    assert P.shape == (3, 18) and torch.isfinite(P).all() and torch.allclose(P.sum(1), torch.ones(3)), 'contract'
    for c in (1, 7, 18):
        assert torch.allclose(model.score(pk['input_ids'], pk['attention_mask'], source_chunk=c)[0], P, atol=1e-5), 'chunks'
    for i, e in enumerate(enc):   # alone (no padding) == inside the padded batch
        x = torch.tensor([e])
        assert torch.allclose(model.score(x, torch.ones_like(x))[0][0], P[i], atol=1e-4), 'padding'
    x, m = pk['input_ids'], pk['attention_mask']
    mixed = model.token_logp(x, m, torch.tensor([3, 11, 17]))
    for i, s in enumerate([3, 11, 17]):   # routing is per row
        alone = model.token_logp(x[i:i + 1], m[i:i + 1], torch.tensor([s]))
        assert torch.allclose(mixed[i, m[i, 1:].bool()], alone[0, m[i, 1:].bool()], atol=1e-4), 'routing'
    assert not torch.allclose(model.token_logp(x, m, torch.zeros(3, dtype=torch.long)),
                              model.token_logp(x, m, torch.ones(3, dtype=torch.long))), 'sources differ'
    st = model.trainable_state()
    other, _ = load_model(tiny, LABELS, CFG, device='auto')
    other.load_trainable(st)
    assert torch.allclose(other.score(x, m)[0], P, atol=1e-5), 'adapter round trip'
    try:
        other.load_trainable({k: v for k, v in st.items() if k != 'control'})
        raise AssertionError('missing key accepted')
    except ValueError:
        pass
    model.train()
    loss = model.loss(x, m, torch.tensor([0, 1, 2]))
    loss.backward()
    assert model.control.grad is not None and model.base.model.layers[1].self_attn.o_proj.b.grad.abs().sum() > 0, 'grads'
    assert model.base.model.layers[1].self_attn.o_proj.base.weight.grad is None, 'base frozen'
    print('check_atlas: all checks passed')


if __name__ == '__main__':
    main()
