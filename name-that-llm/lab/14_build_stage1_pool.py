"""Step 14: the stage-1 pool for a new two-stage LoRA member: Open Pool (latest) cleaned for compliance + nishant's pool_v4.

What: reads crtproduction/name-that-llm-open-pool (ntl_open_pool.parquet, CC BY-SA 4.0) and drops
- flag_kb_proxy rows (written with Kaggle Benchmarks credits; staff 28 Oct: not to be used for training data),
- flag_train_prompt_regen rows (answers to the train prompts),
- labels outside the 18;
adds nishantdahal/ntl-handwriting-blend pool_v4 (CC BY-SA 4.0, the pool our OLMo member used) minus any text that the
Open Pool flags; then, on the NFKC + whitespace key (as steps 12 and 13): drops texts equal to a train.csv passage,
texts carrying two labels, and duplicates (Open Pool first). Rows without a prompt are dropped: stage 1 trains on
triplets of one prompt. Prompt ids are prefixed by source (op:/v4:), so the two pools' ids never merge.
09_train_olmo13b.py then applies its own content-cosine >= 0.3 filter against train (07's leak_filter).

Why: the research note (Study Guide/Above the public line.md) ranks "retrain the two-stage LoRA with stage 1 on the
large cleaned public pool" as the one reachable lever with a measured LB effect; this is that pool, strictly compliant.

Output: <out>/pool_stage1.parquet (text, model, prompt_id, source) and <out>/pool_stage1_summary.json (counts per
step and per label, the Open Pool's columns). Cached: an existing output is reused.
Usage: python -I 14_build_stage1_pool.py --open-pool ntl_open_pool.parquet --train train.csv --out DIR [--v4 pool_v4.parquet]
"""
import argparse
import json
import os
import re
import unicodedata

import pandas as pd

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview',
          'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8',
          'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
PROMPT_COLS = ['prompt_id', 'prompt', 'prompt_text', 'question', 'instruction']   # first one present is used


def key(t):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(t or ''))).strip()


def prompt_ids(df, prefix):
    col = next((c for c in PROMPT_COLS if c in df.columns), None)
    assert col, f'no prompt column among {PROMPT_COLS}; columns are {list(df.columns)}'
    p = df[col].astype('string').str.strip()
    return (prefix + ':' + p.map(lambda s: key(s)[:400], na_action='ignore')).where(p.notna() & (p != '')), col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--open-pool', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--v4')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = os.path.join(a.out, 'pool_stage1.parquet')
    if os.path.exists(out):
        print('cached', out)
        return
    os.makedirs(a.out, exist_ok=True)
    S = {}
    op = pd.read_parquet(a.open_pool)
    S['open_pool_columns'], S['open_pool_rows'] = list(op.columns), len(op)
    flag = op.flag_kb_proxy.fillna(False).astype(bool) | op.flag_train_prompt_regen.fillna(False).astype(bool)
    S['flag_kb_proxy'], S['flag_train_prompt_regen'] = int(op.flag_kb_proxy.fillna(False).astype(bool).sum()), \
        int(op.flag_train_prompt_regen.fillna(False).astype(bool).sum())
    op['key'] = op.text.map(key)
    flagged_keys = set(op.key[flag])
    op = op[~flag & op.model.isin(LABELS)].copy()
    S['open_pool_after_flags_and_labels'] = len(op)
    op['prompt_id'], S['open_pool_prompt_col'] = prompt_ids(op, 'op')
    op['source'] = op['source_dataset'] if 'source_dataset' in op.columns else 'open_pool'
    parts = [op[['text', 'model', 'prompt_id', 'source', 'key']]]
    if a.v4:
        v4 = pd.read_parquet(a.v4)
        S['v4_rows'] = len(v4)
        v4['key'] = v4.text.map(key)
        v4 = v4[v4.model.isin(LABELS) & ~v4.key.isin(flagged_keys)].copy()
        S['v4_after_flags_and_labels'] = len(v4)
        v4['prompt_id'], S['v4_prompt_col'] = prompt_ids(v4, 'v4')
        v4['source'] = 'nishantdahal/ntl-handwriting-blend:pool_v4'
        parts.append(v4[['text', 'model', 'prompt_id', 'source', 'key']])
    df = pd.concat(parts, ignore_index=True)
    S['union_rows'] = len(df)
    tr = pd.read_csv(a.train)
    df = df[~df.key.isin(set(tr.text.map(key)))]
    S['after_train_exact_dedup'] = len(df)
    two = df.groupby('key').model.nunique()
    df = df[~df.key.isin(set(two.index[two > 1]))]
    S['after_two_label_drop'] = len(df)
    df = df.drop_duplicates('key')
    S['after_dedup'] = len(df)
    S['no_prompt_dropped'] = int(df.prompt_id.isna().sum())
    df = df[df.prompt_id.notna()]
    S['final_rows'], S['final_prompts'] = len(df), int(df.prompt_id.nunique())
    S['final_by_source_top'] = df.source.value_counts().head(25).to_dict()
    S['final_per_label'] = df.model.value_counts().to_dict()
    S['final_from_v4_only'] = int(df.prompt_id.str.startswith('v4:').sum())
    df.drop(columns='key').reset_index(drop=True).to_parquet(out)
    json.dump(S, open(os.path.join(a.out, 'pool_stage1_summary.json'), 'w'), indent=1, default=int)
    print(json.dumps(S, indent=1, default=int))


if __name__ == '__main__':
    main()
