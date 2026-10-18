"""Step 5: mine orthographic invariants (a pattern some lab never writes) like the 0.559 notebook's shield (CPU).

The shield (curly quotes -> not Claude/GLM; >= 2 em-dashes -> not Gemini) was the only change the calibrated blend sim
found robust at both test shapes. This looks for more rules of that kind: for every candidate pattern (each non-ASCII
character, plus a few typographic habits), the rate per lab in train and in prvsiyan's extended data (independent API
calls, CC BY 4.0). A rule (pattern -> penalise lab) is kept when the lab's rate is <= --max-rate in BOTH sets while the
pattern appears in >= --min-other of the other labs' train passages (so it fires often enough to matter).

Usage: python 05_invariants.py [--max-rate 0.005] [--min-other 0.05] [--level lab|model]
Output: candidate rules, sorted by how many train passages of other labs they fire on.
"""
import argparse
import os
import re

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HABITS = {   # typographic habits beyond single characters
    'spaced em-dash': r' — ', 'unspaced em-dash': r'\w—\w', 'en-dash range': r'\d–\d',
    'spaced hyphen dash': r'\w - \w', 'double hyphen': r'--', 'ellipsis char': '…', 'three dots': r'\.\.\.',
    'straight apostrophe': r"\w'\w", 'markdown bold': r'\*\*', 'markdown header': r'(?m)^#', 'bullet line': r'(?m)^[-*] ',
    'numbered line': r'(?m)^\d+\. ', 'semicolon': ';', 'colon': ':', 'exclamation': '!', 'parenthesis': r'\(',
    'double space': '  ', 'newline': '\n', 'oxford comma': r', (?:and|or) ', 'british -our': r'\b\w+our\b',
    'em-dash >= 2': None, 'em-dash >= 3': None,
}


def table(df, level):
    lab = df.model if level == 'model' else df.model.str.split('-').str[0]
    pats = {f'char {c!r} U+{ord(c):04X}': re.escape(c) for c in sorted({c for t in df.text for c in t if ord(c) > 127})}
    pats.update(HABITS)
    out = {}
    for name, rx in pats.items():
        if rx is None:
            hit = df.text.str.count('—') >= int(name[-1])
        else:
            hit = df.text.str.contains(rx, regex=True)
        out[name] = hit.groupby(lab).mean()
    return pd.DataFrame(out).T, lab.value_counts()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-rate', type=float, default=0.005)
    ap.add_argument('--min-other', type=float, default=0.05)
    ap.add_argument('--level', choices=['lab', 'model'], default='lab')
    args = ap.parse_args()
    tr = pd.read_csv(os.path.join(HERE, 'data/train.csv'))
    ex = pd.read_csv(os.path.join(HERE, 'data/prvsiyan/passages_clean.csv')).dropna(subset=['text'])
    ex = ex[ex.model.isin(set(tr.model))]
    T, nt = table(tr, args.level)
    E, _ = table(ex, args.level)
    rows = []
    for pat in T.index:
        for lab in T.columns:
            e = E.loc[pat, lab] if pat in E.index else float('nan')
            other = (T.loc[pat].drop(lab) * nt.drop(lab)).sum() / nt.drop(lab).sum()
            if T.loc[pat, lab] <= args.max_rate and e <= args.max_rate and other >= args.min_other:
                rows.append(dict(pattern=pat, lab=lab, train_rate=T.loc[pat, lab], ext_rate=e, other_train=other))
    print(pd.DataFrame(rows).sort_values('other_train', ascending=False).to_string(index=False))


if __name__ == '__main__':
    main()
