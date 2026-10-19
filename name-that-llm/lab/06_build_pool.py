"""Step 6: build the public passage pool of the 0.592 two-stage LoRA notebook (CPU).

nishantdahal's "Two-stage LoRA on the public pool" (LB 0.592) trains its LoRAs first on ~35k public passages from the 18
models, then on train. Its pool dataset is private, but its recipe is published: 14 public datasets (CC BY 4.0, one CC
BY-SA 4.0), single-paragraph passages of 30-300 words with no markdown, exact duplicates and train passages dropped,
and a mark filter: when a model never shows a mark in train (em dash, curly quotes, ...), its pool passages with that
mark are dropped. Their pool has 34,940 passages (1,100-2,700 per model); this rebuild should land close to that.

Usage: python 06_build_pool.py   (expects the 14 datasets unzipped under data/pool_src/<owner>_<slug>/)
Output: data/pool_v1.parquet (columns text, model, prompt_id, source) + per-model counts.
"""
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'data/pool_src')
FILES = {   # dataset -> (file, label column); the prompt column is 'prompt_id' unless listed in PROMPT_COL
    'abdullah1afify_ntl-generations': ('ntl_generations.csv', 'label'),
    'alperen5252525_ntl-vertex-gemini-proxy-gpt-generations': ('ntl_alp_generations.csv', 'label'),
    'dophuchung_ntl-proxy-generations': ('ntl_proxy_generations.csv', 'label'),
    'haileycheng_name-that-llm-same-prompt-passages': ('passages.csv', 'label'),
    'kaplinskiy_ntl-gpt-proxy-passages': ('ntl_gpt_proxy_passages.csv', 'label'),
    'keshavatearth_ntl-gpt-new-prompts-v1': ('accepted.csv', 'model'),
    'lucasboesen_ntl-free-proxy-generations': ('gen_passages.csv', 'model'),
    'lxdlxdlxd_ntl-synth-generations-v1': ('generations.parquet', 'label'),
    'michaelsemenoff_ntl-own-extra-v1': ('ntl_own_extra_v1.csv', 'model'),
    'prvsiyan_ntl-own-prompt-generations-v1': ('generated_v1.parquet', 'label'),
    'prvsiyan_prvsiyans-extended-name-that-llm-dataset': ('passages_clean.csv', 'model'),
    'seyitkaangunes_ntl-extra-api-passages': ('ntl_extra_passages_claude.csv', 'model'),
    'willrice_name-that-llm-openrouter-harvest': ('passages.csv', 'label'),
    'ziyuzhou1207_ntl-zz-proxy-generations': ('passages.csv', 'model'),
}
PROMPT_COL = {'lxdlxdlxd_ntl-synth-generations-v1': 'pid', 'michaelsemenoff_ntl-own-extra-v1': 'group'}
MARKS = {'em dash': '—', 'en dash': '–', 'curly apostrophe': '’', 'curly quotes': '[“”]',
         'ellipsis': '…'}
MARKDOWN = re.compile(r'(?m)\*\*|^\s*(#|[-*•] |\d+\. )')


def main():
    train = pd.read_csv(os.path.join(HERE, 'data/train.csv'))
    labels = set(train.model)
    parts = []
    for d, (f, col) in FILES.items():
        p = os.path.join(SRC, d, f)
        df = pd.read_parquet(p) if f.endswith('.parquet') else pd.read_csv(p)
        df = df.rename(columns={col: 'model', PROMPT_COL.get(d, 'prompt_id'): 'prompt_id'})
        df = df[['text', 'model', 'prompt_id']].assign(source=d)
        df['prompt_id'] = d + ':' + df.prompt_id.astype(str)   # triplet batches need passages of the same prompt
        parts.append(df)
        print(f'{d:55s} {len(df):6d} rows, {df.model.isin(labels).sum():6d} with a competition label')
    pool = pd.concat(parts, ignore_index=True).dropna(subset=['text', 'model'])
    pool['text'] = pool.text.astype(str).str.strip()
    pool = pool[pool.model.isin(labels)]
    n = len(pool)
    words = pool.text.str.split().str.len()
    pool = pool[(words >= 30) & (words <= 300) & ~pool.text.str.contains('\n') & ~pool.text.str.contains(MARKDOWN)]
    print(f'\n{n} labelled -> {len(pool)} after 30-300 words, single paragraph, no markdown')
    pool = pool.drop_duplicates('text')
    pool = pool[~pool.text.isin(set(train.text.str.strip()))]
    print(f'-> {len(pool)} after exact dedupe and dropping train passages')
    for name, rx in MARKS.items():   # a mark the model never uses in train -> drop its pool passages with that mark
        never = [m for m, g in train.groupby('model') if not g.text.str.contains(rx).any()]
        bad = pool.model.isin(never) & pool.text.str.contains(rx)
        pool = pool[~bad]
        print(f'   {name:16s} never used in train by {len(never):2d} models -> dropped {int(bad.sum())}')
    pool = pool.reset_index(drop=True)
    print(f'-> {len(pool)} passages (0.592 notebook: 34,940)\n')
    print(pool.model.value_counts().to_string())
    per = pool.groupby('prompt_id').model.nunique()
    print(f'\n{len(per)} prompts; {int((per >= 3).sum())} with >= 3 models ({int(pool.prompt_id.map(per).ge(3).sum())} passages)')
    pool.to_parquet(os.path.join(HERE, 'data/pool_v1.parquet'), index=False)


if __name__ == '__main__':
    main()
