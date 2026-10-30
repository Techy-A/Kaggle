"""Step 13: audit three public NTL datasets for genuinely new, clean, in-genre passages. (CPU, no training)

What: for elliotthankinson/ntl-exact-gens-v3 (all its parquets, unioned), riccapucho/gaggle-ntl-superset-v1 and
crtproduction/name-that-llm-open-pool (v10; v9 = ids below ntlop_263484), reports rows and labels, word-length
quantiles against train.csv, the share of junk under discussion 753779's rules (approximated by regex: API dumps,
refusals, chat preambles, CJK, length outside 36-275 words, no final punctuation = cut-off, markdown/newlines =
off-genre format, same text with two labels), exact-text overlap with train.csv (leak check) and with the pools the
0.663 notebook already uses (nlm_pool_v5, nlm_matched_clean, abdullah1afify, dophuchung, lucasboesen, seyitkaangunes,
prvsiyan extended), and how many clean rows are new (not in those pools; and not in Open Pool v10 either).
Text identity = NFKC + whitespace-collapsed key, as in infer.py and step 12.

Why: decide whether any of these datasets is worth a member retrain or an SVC-pool test (step 12).

Deviation (ponytail): off-genre *sources* (chat logs, benchmark answers) are approximated by format checks only;
the ceiling is that in-format off-genre prose passes. Upgrade path: a per-source style model as in 753779.

Output: <out>/audit13.json (cached; delete to recompute). Usage:
  python -I 13_new_datasets_audit.py --eh DIR --gg DIR --op DIR --dt DIR [--src data/pool_src] [--train data/train.csv]
"""
import argparse
import glob
import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
V10_FIRST = 'ntlop_263484'
SRC_663 = {'abdullah1afify_ntl-generations/ntl_generations.csv', 'dophuchung_ntl-proxy-generations/ntl_proxy_generations.csv',
           'lucasboesen_ntl-free-proxy-generations/gen_passages.csv',
           'seyitkaangunes_ntl-extra-api-passages/ntl_extra_passages_claude.csv',
           'prvsiyan_prvsiyans-extended-name-that-llm-dataset/passages_clean.csv'}
RULES = {
    'api_dump': re.compile(r'^\s*[\{\[]|"(choices|role|content|finish_reason)"\s*:|HTTP/\d|Content-Type:'),
    'refusal': re.compile(r"I can(?:no|')t (?:help|assist|provide|write|comply)|I'm sorry, but|I am sorry, but|As an AI\b", re.I),
    'preamble': re.compile(r"^\s*(?:Sure|Certainly|Of course|Absolutely|Here is|Here's|Okay|OK)\b[,!.:]?"),
    'cjk': re.compile(r'[぀-ヿ㐀-鿿가-힯]'),
    'format': re.compile(r'\n|^\s*#|\*\*|^\s*[-*] |^\s*\d+\. |<think>|```', re.M),
}
END = re.compile(r'[.!?]["\'”’)\]*_]*\s*$')


def key(t):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(t or ''))).strip()


def junk(df):
    t = df.text.fillna('').astype(str)
    w = t.str.split().str.len()
    J = pd.DataFrame({k: t.str.contains(r) for k, r in RULES.items()})
    J['length'] = (w < 36) | (w > 275)
    J['cutoff'] = ~t.map(lambda s: bool(END.search(s)))
    lab = df.assign(k=df.key).groupby('k').model.nunique()
    J['two_labels'] = df.key.map(lab).gt(1).values
    return J, w


def summary(df, train_keys, pool663, extra_keys=None, train_words=None):
    J, w = junk(df)
    bad = J.any(axis=1)
    in_train = df.key.isin(train_keys)
    in663 = df.key.isin(pool663)
    new = ~bad & ~in_train & ~in663
    out = {'rows': len(df), 'unique_texts': int(df.key.nunique()), 'labels': int(df.model.nunique()),
           'words_q05_50_95': [int(x) for x in np.percentile(w, [5, 50, 95])],
           'ends_punct': round(float(1 - J.cutoff.mean()), 4),
           'junk_share': {k: round(float(v), 4) for k, v in J.mean().items()}, 'junk_any': round(float(bad.mean()), 4),
           'train_exact_overlap': int(in_train.sum()), 'in_663_pools': int(in663.sum()),
           'new_clean_vs_663': int(df.key[new].nunique()),
           'new_clean_vs_663_per_label': df[new].drop_duplicates('key').model.value_counts().to_dict()}
    if extra_keys is not None:
        n2 = new & ~df.key.isin(extra_keys)
        out['new_clean_vs_663_and_openpool_v10'] = int(df.key[n2].nunique())
        out['new_clean_vs_663_and_openpool_v10_per_label'] = df[n2].drop_duplicates('key').model.value_counts().to_dict()
    return out, new


def load_src(path):
    d = pd.read_csv(path)
    d = d.rename(columns={'label': 'model'}) if 'label' in d else d
    return d[['text', 'model']]


def main():
    ap = argparse.ArgumentParser()
    for a in ('--eh', '--gg', '--op', '--dt'):
        ap.add_argument(a, required=True)
    ap.add_argument('--src', default=os.path.join(HERE, 'data/pool_src'))
    ap.add_argument('--train', default=os.path.join(HERE, 'data/train.csv'))
    ap.add_argument('--out', default=os.path.join(HERE, 'artifacts'))
    a = ap.parse_args()
    out = os.path.join(a.out, 'audit13.json')
    if os.path.exists(out):
        print(open(out).read())
        return
    tr = pd.read_csv(a.train)
    labels = set(tr.model)
    train_keys = set(tr.text.map(key))
    tw = tr.text.str.split().str.len()
    pools = [pd.read_parquet(os.path.join(a.dt, f))[['text', 'model']] for f in ('nlm_pool_v5.parquet', 'nlm_matched_clean.parquet')]
    pools += [load_src(os.path.join(a.src, f)) for f in sorted(SRC_663)]
    pool663 = set(pd.concat(pools).text.map(key))
    R = {'train': {'rows': len(tr), 'words_q05_50_95': [int(x) for x in np.percentile(tw, [5, 50, 95])],
                   'words_min_max': [int(tw.min()), int(tw.max())]}, 'pool663_unique_texts': len(pool663)}

    op = pd.read_parquet(a.op)
    op['key'] = op.text.map(key)
    op_keys = set(op.key)
    v9 = op.id < V10_FIRST
    R['open_pool_v10'], new = summary(op, train_keys, pool663)
    R['open_pool_v10']['v9_rows'] = int(v9.sum())
    R['open_pool_v10']['flag_kb_proxy'] = int(op.flag_kb_proxy.sum())
    R['open_pool_v10']['v10_added_rows'] = int((~v9).sum())
    R['open_pool_v10']['v10_added_new_clean_vs_663'] = int(op.key[new & ~v9].nunique())
    R['open_pool_v10']['v10_added_new_clean_vs_663_and_v9'] = int(op.key[new & ~v9 & ~op.key.isin(set(op.key[v9]))].nunique())
    R['open_pool_v10']['v10_added_by_source'] = op[~v9].source_dataset.value_counts().to_dict()

    eh = []
    for f in sorted(glob.glob(os.path.join(a.eh, '*.parquet'))):
        d = pd.read_parquet(f)
        d['file'] = os.path.basename(f)
        eh.append(d)
    eh = pd.concat(eh, ignore_index=True)
    eh['route'] = np.where(eh.id.str.startswith('g3or_'), 'openrouter_paid', 'kaggle_benchmarks')
    eh['key'] = eh.text.map(key)
    eh = eh[eh.model.isin(labels)].drop_duplicates(['key', 'model'])
    R['eh_union'], new = summary(eh, train_keys, pool663, op_keys)
    R['eh_union']['rows_by_route'] = eh.route.value_counts().to_dict()
    for route in ('openrouter_paid', 'kaggle_benchmarks'):
        m = eh.route == route
        s, _ = summary(eh[m], train_keys, pool663, op_keys)
        R[f'eh_{route}'] = s
    dp = eh.discriminator_pass.fillna(True).astype(bool)   # pool_v8j_add has no flag: it is the author's own pass set
    s, _ = summary(eh[dp & (eh.route == 'openrouter_paid')], train_keys, pool663, op_keys)
    R['eh_openrouter_discriminator_pass'] = s

    gg = pd.read_parquet(a.gg)
    gg['key'] = gg.text.map(key)
    R['gaggle'], new = summary(gg, train_keys, pool663, op_keys)
    R['gaggle']['by_release'] = gg.release.value_counts().to_dict()
    R['gaggle']['quality_flag'] = gg.quality_flag.value_counts().to_dict()
    R['gaggle']['top_via_dataset'] = gg.via_dataset.value_counts().head(8).to_dict()
    n2 = new & ~gg.key.isin(op_keys)
    R['gaggle']['new_vs_663_and_op_by_original_dataset'] = gg[n2].drop_duplicates('key').original_dataset.value_counts().head(15).to_dict()
    R['gaggle']['benchmarks_word_in_original'] = int(gg.original_dataset.str.contains('proxy|bench', case=False).sum())
    os.makedirs(a.out, exist_ok=True)
    json.dump(R, open(out, 'w'), indent=1, default=int)
    print(json.dumps(R, indent=1, default=int))


if __name__ == '__main__':
    main()
