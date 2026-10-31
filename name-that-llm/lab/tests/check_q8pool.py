"""Check the q8pool member end to end on CPU with a tiny random Qwen3 (8 layers, keep 6), in a few minutes.

1. 14_build_stage1_pool.py on a fake Open Pool made from pool_v4 (flags set on some rows, a few train passages and a
   two-label text planted): asserts the flagged, train-equal and two-label texts are gone and v4 copies of flagged texts too.
2. 09_train_olmo13b.py --quant fp16 (the q8pool settings): stage1, fold0, full.
3. The 0.664 notebook's own l24_predict (taken from its infer.py cell, run unchanged) on the exported lora_q8pool with
   the spec the variant notebook uses (no_final_norm, eos_id): its plain log-probs must match the training code's
   (unmerged LoRA) logits: top-1 agreement >= 0.95 and |diff| small. This is the check that the notebook reads the export.

Usage: python lab/tests/check_q8pool.py --tokenizer DIR_WITH_QWEN3_TOKENIZER --train train.csv --pool pool_v4.parquet
       --notebook 0.664.ipynb [--work DIR]
"""
import argparse
import ast
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print('$', ' '.join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in (r.stdout + r.stderr).splitlines() if 'warn' not in l.lower() and l.strip()]
    print('\n'.join('    ' + l for l in lines[-14:]), flush=True)
    assert r.returncode == 0, f'exit {r.returncode}'


def notebook_l24(nb_path):
    """l24_predict and the names it needs, from the notebook's infer.py cell (the function source, unchanged)."""
    nb = json.load(open(nb_path))
    src = next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('%%writefile infer.py'))
    src = src.split('\n', 1)[1]
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'l24_predict' or
            isinstance(n, ast.Assign) and any(getattr(t, 'id', '') in ('LABELS',) for t in n.targets)]
    import gc
    import time
    import torch
    from transformers import AutoModel, AutoTokenizer
    g = dict(np=np, torch=torch, json=json, os=os, gc=gc, AutoModel=AutoModel, AutoTokenizer=AutoTokenizer,
             log=lambda *x: print('   [nb]', *x, flush=True), time=time)
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'infer.py', 'exec'), g)
    return g['l24_predict'], g['LABELS']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--notebook', required=True)
    ap.add_argument('--work', default='/tmp/check_q8pool')
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, Qwen3Config
    if os.path.exists(a.work):
        shutil.rmtree(a.work)
    os.makedirs(a.work)
    base = os.path.join(a.work, 'tiny_qwen3')
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    AutoModel.from_config(cfg).save_pretrained(base)
    for f in os.listdir(a.tokenizer):
        if f.startswith(('tokenizer', 'special_tokens', 'vocab', 'merges', 'chat_template')):
            shutil.copy(os.path.join(a.tokenizer, f), base)

    # 1. pool
    tr = pd.read_csv(a.train)
    v4 = pd.read_parquet(a.pool)
    keep_p = v4.prompt_id.drop_duplicates().sample(40, random_state=0)
    v4 = v4[v4.prompt_id.isin(keep_p)].reset_index(drop=True)
    op = v4.sample(frac=0.7, random_state=1).rename(columns={'prompt_key': 'prompt'})[['text', 'model', 'prompt']]
    op = op.reset_index(drop=True)
    rng = np.random.default_rng(0)
    op['flag_kb_proxy'] = rng.random(len(op)) < 0.2
    op['flag_train_prompt_regen'] = rng.random(len(op)) < 0.05
    op['source_dataset'] = 'fake'
    plant = pd.DataFrame(dict(text=list(tr.text[:3]) + [op.text[0]], model=list(tr.model[:3]) + ['kimi-k3'],
                              prompt='planted', flag_kb_proxy=False, flag_train_prompt_regen=False, source_dataset='plant'))
    if op.model[0] == 'kimi-k3':
        plant.loc[3, 'model'] = 'glm-5.3'
    op = pd.concat([op, plant], ignore_index=True)
    op_path, v4_path = os.path.join(a.work, 'ntl_open_pool.parquet'), os.path.join(a.work, 'pool_v4.parquet')
    op.to_parquet(op_path), v4.to_parquet(v4_path)
    pdir = os.path.join(a.work, 'out')
    run([sys.executable, '-I', os.path.join(LAB, '14_build_stage1_pool.py'), '--open-pool', op_path, '--v4', v4_path,
         '--train', a.train, '--out', pdir])
    P = pd.read_parquet(os.path.join(pdir, 'pool_stage1.parquet'))
    flagged = set(op.text[op.flag_kb_proxy | op.flag_train_prompt_regen])
    assert not set(P.text) & flagged, 'flagged texts left (Open Pool or v4 copies)'
    assert not set(P.text) & set(tr.text), 'train texts left'
    assert op.text[0] not in set(P.text), 'two-label text left'
    assert P.text.is_unique and P.prompt_id.notna().all() and P.prompt_id.str.match(r'^(op|v4):').all()
    assert (P.prompt_id.str.startswith('v4:')).any() and (P.prompt_id.str.startswith('op:')).any()
    print(f'pool ok: {len(P)} rows from {len(op)} fake Open Pool + {len(v4)} v4', flush=True)

    # 2. training, q8pool settings at tiny scale
    script = os.path.join(LAB, '09_train_olmo13b.py')
    common = ['--train', a.train, '--pool', os.path.join(pdir, 'pool_stage1.parquet'), '--base', base, '--out', pdir,
              '--quant', 'fp16', '--keep', '6', '--layers', '1,2,4,6', '--max-len', '384', '--micro-triplets', '2',
              '--limit-pool', '150', '--limit-prompts', '10', '--allow-cpu', '--save-every', '3', '--eval-bs', '4',
              '--member', 'lora_q8pool', '--base-name', 'Qwen/Qwen3-8B-Base', '--base-class', 'Qwen3Model',
              '--pool-note', 'test']
    run([sys.executable, script, '--unit', 'stage1', *common])
    run([sys.executable, script, '--unit', 'fold0,full', *common])
    member = os.path.join(pdir, 'lora_q8pool')
    meta = json.load(open(os.path.join(member, 'meta.json')))
    assert meta['cfg']['quant'] == 'fp16' and meta['cfg']['model_dir'] == 'Qwen/Qwen3-8B-Base'
    eos = meta['cfg']['eos_id']
    assert eos == 151643, eos
    z = np.load(os.path.join(pdir, 'oof_fold0.npz'))
    assert np.isfinite(z['plain']).all() and len(z['rows']) > 0
    summ = json.load(open(os.path.join(pdir, 'summary_full.json')))
    assert summ['reload_agreement'] >= 0.98, summ

    # 3. the notebook's l24_predict on the export vs the training code's own reader
    sys.path.insert(0, LAB)
    import olmo13b_member as om
    trs = tr[tr.prompt_id.isin(sorted(tr.prompt_id.unique())[:10])]
    texts = trs.text.tolist()[:60]
    l24_predict, NB_LABELS = notebook_l24(a.notebook)
    spec = dict(name='q8pool', type='l24', path=base, adapters=[member], keep=6, layers=[1, 2, 4, 6], w=0.3, bs=4,
                max_len=meta['cfg']['max_len'] + 1, twin=0.35, no_final_norm=True, eos_id=eos)
    nb_plain, _ = l24_predict(spec, texts)
    Z, heads = om.olmo_features(member, base, texts, 4, meta['labels'], device='cpu')
    ours = Z @ heads['plain.weight'].T + heads['plain.bias']
    ours = ours[:, [meta['labels'].index(l) for l in NB_LABELS]]
    ours = ours - np.log(np.exp(ours - ours.max(1, keepdims=True)).sum(1, keepdims=True)) - ours.max(1, keepdims=True)
    agree = float((nb_plain.argmax(1) == ours.argmax(1)).mean())
    diff = float(np.abs(nb_plain - ours).max())
    print(f'notebook l24_predict vs training reader: top-1 agreement {agree:.3f}, max |log-prob diff| {diff:.2e}', flush=True)
    assert agree >= 0.95 and diff < 0.05, (agree, diff)
    print('OK: q8pool pool, fp16 training and the notebook reader agree', flush=True)


if __name__ == '__main__':
    main()
