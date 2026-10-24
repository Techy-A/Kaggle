"""Check 09_train_olmo13b.py end to end on CPU with a tiny random OLMo-2 (8 layers, keep 6), in about 5 minutes.

Runs, on 10 train prompts and 150 pool passages: stage1 with 1 rank and (separately) with 2 ranks under torchrun
(gloo), then fold0..fold4, full (whose reload check scores the export with olmo13b_member.olmo_features, the notebook's
code path) and merge; in fp32 and, when bitsandbytes is importable, in NF4. It asserts the outputs exist and that 1 and
2 ranks give the same stage-1 weights (the gradient sum over ranks is the single-process gradient; no dropout).

Usage: python lab/tests/check_olmo13b.py --tokenizer DIR_WITH_OLMo2_TOKENIZER --train train.csv --pool pool_v4.parquet
       [--work DIR]   (DIR is kept; a tokenizer dir is any copy of allenai/OLMo-2-1124-13B's tokenizer files)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, env=None):
    print('$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if 'warn' not in l.lower() and l.strip()]
    print('\n'.join('    ' + l for l in lines[-12:]), flush=True)
    assert r.returncode == 0, f'exit {r.returncode}'
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--work', default='/tmp/check_olmo13b')
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, Olmo2Config
    if os.path.exists(a.work):
        shutil.rmtree(a.work)
    base = os.path.join(a.work, 'tiny_olmo2')
    torch.manual_seed(0)
    cfg = Olmo2Config(vocab_size=100352, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=4, eos_token_id=100257, pad_token_id=100277,
                      bos_token_id=None)
    AutoModel.from_config(cfg).save_pretrained(base)   # Olmo2Model keys, as AutoModel loads them
    for f in os.listdir(a.tokenizer):
        if f.startswith(('tokenizer', 'special_tokens', 'vocab', 'merges')):
            shutil.copy(os.path.join(a.tokenizer, f), base)
    common = ['--train', a.train, '--pool', a.pool, '--base', base, '--keep', '6', '--layers', '1,2,4,6',
              '--limit-pool', '150', '--limit-prompts', '10', '--allow-cpu', '--save-every', '3', '--eval-bs', '4']
    script = os.path.join(LAB, '09_train_olmo13b.py')
    try:
        import bitsandbytes  # noqa: F401
        quants = ['none', 'nf4']
    except ImportError:
        quants = ['none']
        print('bitsandbytes not importable: NF4 path not checked on CPU', flush=True)
    for q in quants:
        qa = ['--quant', q, '--compute-dtype', 'float32']
        out1, out2 = os.path.join(a.work, f'out_{q}'), os.path.join(a.work, f'out_{q}_2ranks')
        # no dropout, so 1 rank and 2 ranks must agree exactly in stage 1
        nodrop = ['--lora-dropout', '0', '--head-drop', '0']
        run([sys.executable, script, '--unit', 'stage1', '--out', out1, *common, *qa, *nodrop])
        run([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', '2', '--master_port', '29533', script,
             '--unit', 'stage1', '--out', out2, *common, *qa, *nodrop], env={'OMP_NUM_THREADS': '1'})
        from safetensors.torch import load_file
        s1, s2 = (load_file(os.path.join(o, 'stage1', 'lora_heads.safetensors')) for o in (out1, out2))
        diff = max((s1[k] - s2[k]).abs().max().item() for k in s1)
        moved = max(s1[k].abs().max().item() for k in s1 if 'lora_B' in k)
        print(f'[{q}] stage1 1 rank vs 2 ranks: max |diff| {diff:.2e}; max |B| {moved:.2e}', flush=True)
        assert set(s1) == set(s2) and diff < 1e-4 and moved > 0
        run([sys.executable, script, '--unit', 'fold0,fold1,fold2,fold3,fold4,full,merge', '--out', out1, *common, *qa])
        summ = json.load(open(os.path.join(out1, 'summary_olmo13b.json')))
        print(f'[{q}] merge scores:', json.dumps(summ['scores'])[:300], flush=True)
        z = np.load(os.path.join(out1, 'oof_olmo13b.npz'))
        assert np.isfinite(z['plain']).all() and z['plain'].shape == (len(z['y']), 18) and len(z['y']) >= 170
        member = os.path.join(out1, 'lora_olmo2_13b')
        meta = json.load(open(os.path.join(member, 'meta.json')))
        assert sorted(os.listdir(member)) == ['adapter', 'heads.safetensors', 'meta.json']
        assert sorted(os.listdir(os.path.join(member, 'adapter'))) == ['adapter_config.json', 'adapter_model.safetensors']
        assert meta['cfg']['quant'] == (None if q == 'none' else q) and meta['cfg']['eos_id'] == 100257
        assert json.load(open(os.path.join(out1, 'summary_full.json')))['reload_agreement'] >= 0.98
        assert not os.listdir(os.path.join(out1, 'state')), 'resume points must be deleted when units finish'
    # the notebook's merge_lora reads our adapter keys: each one names a module of the trunk
    from safetensors.torch import load_file
    sd = load_file(os.path.join(a.work, 'out_none', 'lora_olmo2_13b', 'adapter', 'adapter_model.safetensors'))
    names = {n for n, _ in AutoModel.from_pretrained(base).named_modules()}
    keys = {k.split('.lora_A.')[0] for k in sd if '.lora_A.' in k}
    assert all(k[k.index('layers.'):] in names for k in keys) and len(keys) == 6 * 7, len(keys)
    print('OK: 09_train_olmo13b.py runs end to end on CPU', quants, flush=True)


if __name__ == '__main__':
    main()
