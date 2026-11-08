"""21: drop generation topics that are too close to a train prompt group (keeps 20's prompts out of 'near train prompt' filters).

What: TF-IDF (word 1-2 grams) of each topic string against the concatenated 18 passages of each of the 150 train prompt
groups; a topic whose best cosine >= THR is commented out in gen/topics.txt (prefixed '# near-train <pid> <cos>').
Output: gen/topics.txt rewritten in place, gen/topic_filter.json (per-topic best cosine and prompt group).
Usage: python 21_topic_filter.py [thr=0.12]
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
HERE = Path(__file__).parent; THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.12
t = pd.read_csv(HERE / 'data' / 'train.csv'); g = t.groupby('prompt_id').text.apply(' '.join)
lines = open(HERE / 'gen' / 'topics.txt').read().splitlines()
tops = [l for l in lines if l.strip() and not l.startswith('#')]
v = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', sublinear_tf=True).fit(list(g.values) + tops)
S = (v.transform(tops) @ v.transform(g.values).T).toarray()
best = S.argmax(1); bc = S.max(1); rep = {}
out = []
for l in lines:
    if l.strip() and not l.startswith('#'):
        i = tops.index(l); rep[l] = dict(cos=round(float(bc[i]), 3), pid=g.index[best[i]])
        out.append(f'# near-train {g.index[best[i]]} {bc[i]:.2f}: {l}' if bc[i] >= THR else l)
    else:
        out.append(l)
open(HERE / 'gen' / 'topics.txt', 'w').write('\n'.join(out) + '\n'); json.dump(rep, open(HERE / 'gen' / 'topic_filter.json', 'w'), indent=1)
drop = sorted([(v['cos'], k, v['pid']) for k, v in rep.items() if v['cos'] >= THR], reverse=True)
print('topics', len(tops), '| dropped', len(drop), '| kept', len(tops) - len(drop)); print('\n'.join(f'  {c:.2f} {p} {k}' for c, k, p in drop[:40]))
