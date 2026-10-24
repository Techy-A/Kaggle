"""Score the genfix bench (build_genfix_bench.py): base 0.464 engine-1 list vs ungated gen20 vs features-frozen gen20.

Why: the gate for any list-composition change is how often the correct class-1 answer loses rank 1 on a thin
class-1 slice (results/calibration.md). This script reads the bench log, rebuilds the genfix list for a sweep of
penalties with the exact functions of variants/port_genfix.py, checks the bench's identity runs, and reports per class
MRR@25 / top-1 / rank-1 losses with a paired stratified bootstrap. The penalty is chosen by 2-fold cross-fitting
(molecule hash halves), so the reported genfix numbers are not tuned on the molecules they are scored on.

Output: a markdown report on stdout (and --json for the numbers).
usage: python genfix_analyze.py <bench log text or .jsonl> [--json out.json]
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'variants'))
import port_genfix as pg  # noqa: E402

NS = {'np': np}
exec(pg.GENFIX_SRC.replace('__ON__', 'True').replace('__DELTA__', '0.0').replace('__MAXNEW__', '60'), NS)
TOPN = 60
DELTAS = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
MIX = {1: 0.28, 2: 0.20, 3: 0.52}


def load(path):
    recs, ident = [], None
    for line in open(path):
        line = line.rstrip('\n')
        if line.startswith('GFB_JSON '):
            recs.append(json.loads(line[9:]))
        elif line.startswith('GFB_IDENTITY '):
            ident = json.loads(line[13:])
        elif line.startswith('{'):
            recs.append(json.loads(line))
    return recs, ident


def genfix_keys(r, delta, cap=60):
    new = [(k, k, '', m) for k, m in r['new'][:cap]]
    placed = NS['_genfix_place'](new, r['scs1'], delta, TOPN)
    if not placed:
        return list(r['fin1'])
    S, K, _, _ = NS['_genfix_insert'](list(r['fin1']), list(r['fin1']), [0.0] * len(r['fin1']), [''] * len(r['fin1']), placed)
    return K


def rank(keys, k):
    try:
        return keys.index(k) + 1
    except ValueError:
        return 0


def rr(r):
    return 1.0 / r if 0 < r <= 25 else 0.0


def boot(d_by_cls, n_boot=4000, seed=0, mix=None):
    """Paired stratified bootstrap of the mean delta (mix-weighted over classes if mix): (mean, lo, hi, P>0)."""
    rng = np.random.default_rng(seed)
    cls = sorted(d_by_cls)
    w = {c: (mix[c] if mix else len(d_by_cls[c])) for c in cls}
    tot = sum(w.values())
    point = sum(w[c] * np.mean(d_by_cls[c]) for c in cls) / tot
    sims = np.zeros(n_boot)
    for c in cls:
        a = np.asarray(d_by_cls[c]); idx = rng.integers(0, len(a), (n_boot, len(a)))
        sims += w[c] * a[idx].mean(1) / tot
    return point, np.quantile(sims, 0.025), np.quantile(sims, 0.975), float((sims > 0).mean()), float((sims < 0).mean())


def main():
    recs, ident = load(sys.argv[1])
    ok = [r for r in recs if 'error' not in r]
    out = dict(n=len(recs), errors=[(r['mid'], r['error']) for r in recs if 'error' in r])
    # answer keys: the bench stores the answer rank in its own lists; the key comes from bench_answers.csv when given,
    # else from the logged rank of the base list (ar1 is over the full engine-1 order)
    ans = {}
    apath = os.path.join(os.path.dirname(sys.argv[1]), 'bench_answers.csv')
    if os.path.exists(apath):
        import pandas as pd
        a = pd.read_csv(apath, dtype=str, keep_default_na=False)
        ans = dict(zip(a.molecule_id, a.metric_key)); sl = dict(zip(a.molecule_id, a.slice))
    else:
        sys.exit('bench_answers.csv must sit next to the log')
    for r in ok:
        r['k'] = ans[r['mid']]; r['slice'] = sl[r['mid']]
        r['r_base'] = rank(r['fin1'], r['k']); r['r_g20'] = rank(r['fin2'], r['k'])
        r['h'] = int(hashlib.md5(r['mid'].encode()).hexdigest(), 16) % 2
    print(f"# genfix bench\n\nmolecules {len(recs)}, errors {len(out['errors'])}, "
          f"with list {sum(1 for r in ok if r['fin1'])}")
    if ident:
        rec = {r['mid']: r for r in ok}
        rebuilt = {m: genfix_keys(rec[m], 0.0) == k for m, k in ident['on_keys'].items() if m in rec}
        print(f"\nidentity: GENFIX=False vs original base-list cell on {ident['molecules']} molecules: byte-identical "
              f"{ident['byte_identical']} (differing {ident['differing']}, original re-run differing "
              f"{ident['orig_rerun_differing']}); main-loop base list == identity-run list: "
              f"{sum(1 for r in ok if r.get('id_ok'))}/{sum(1 for r in ok if 'id_ok' in r)}; offline genfix rebuild == "
              f"notebook GENFIX=True list: {sum(rebuilt.values())}/{len(rebuilt)}; notebook genfix stats {ident['genfix_stats']}")
        out['identity'] = dict(ident={k: v for k, v in ident.items() if not k.endswith('_keys')}, rebuilt=rebuilt)
    t1 = np.array([r['t1'] for r in ok]); t2 = np.array([r['t2'] for r in ok])
    print(f"\nengine-1 CPU time per molecule: run 1 {t1.mean():.1f}s, gen20 run {t2.mean():.1f}s "
          f"(genfix = {1 + t2.sum() / t1.sum():.2f}x engine 1); generated candidates {np.mean([r['ng1'] for r in ok]):.0f} -> "
          f"{np.mean([r['ng2'] for r in ok]):.0f}; candidates {np.mean([r['n1'] for r in ok]):.0f} -> {np.mean([r['n2'] for r in ok]):.0f}")
    scale = np.concatenate([np.asarray(r['scs1'][:25]) for r in ok if len(r['scs1']) > 1])
    gaps = [r['scs1'][0] - r['scs1'][1] for r in ok if len(r['scs1']) > 1]
    print(f"score scale: top-25 scores sd {scale.std():.2f}, median gap rank1-rank2 {np.median(gaps):.2f}")

    def evaluate(keyfn, rows):
        res = {}
        for r in rows:
            K = keyfn(r); rk = rank(K, r['k'])
            orig = [k for k in K if k in set(r['fin1'])]
            res[r['mid']] = dict(r=rk, rr=rr(rk), lost1=int(r['r_base'] == 1 and rk != 1), gain1=int(r['r_base'] != 1 and rk == 1),
                                 top5_changed=int(orig[:5] != [k for k in r['fin1'] if k in set(K)][:5]),
                                 new_top1=int(bool(K) and K[0] not in set(r['fin1'])),
                                 new_top25=sum(1 for k in K[:25] if k not in set(r['fin1'])))
        return res

    groups = [('thin c1 (np)', lambda r: r['cls'] == 1 and r['slice'] == 'np'), ('thin c1 (t)', lambda r: r['slice'] == 't'),
              ('c1 all', lambda r: r['cls'] == 1), ('c2', lambda r: r['cls'] == 2), ('c3', lambda r: r['cls'] == 3)]
    base = evaluate(lambda r: r['fin1'], ok)

    def table(name, res, rows):
        print(f"\n## {name}\n\n| slice | n | base MRR | var MRR | d | top-1 base->var | correct loses rank 1 | gains rank 1 |"
              f" top-5 order of originals changed | new cand at rank 1 | mean new in top 25 |\n|---|---|---|---|---|---|---|---|---|---|---|")
        for g, f in groups:
            ms = [r['mid'] for r in rows if f(r)]
            if not ms:
                continue
            b = np.mean([base[m]['rr'] for m in ms]); v = np.mean([res[m]['rr'] for m in ms])
            print(f"| {g} | {len(ms)} | {b:.4f} | {v:.4f} | {v - b:+.4f} | {sum(base[m]['r'] == 1 for m in ms)}->"
                  f"{sum(res[m]['r'] == 1 for m in ms)} | {sum(res[m]['lost1'] for m in ms)} | {sum(res[m]['gain1'] for m in ms)} | "
                  f"{np.mean([res[m]['top5_changed'] for m in ms]):.3f} | {np.mean([res[m]['new_top1'] for m in ms]):.3f} | "
                  f"{np.mean([res[m]['new_top25'] for m in ms]):.2f} |")
        d = {c: [res[r['mid']]['rr'] - base[r['mid']]['rr'] for r in rows if r['cls'] == c] for c in (1, 2, 3)}
        d = {c: v for c, v in d.items() if v}
        bn = boot(d); bm = boot(d, mix=MIX)
        print(f"\npaired bootstrap: natural mix d {bn[0]:+.4f} [{bn[1]:+.4f}, {bn[2]:+.4f}] P(better) {bn[3]:.2f} P(worse) {bn[4]:.2f}; "
              f"28/20/52 mix d {bm[0]:+.4f} [{bm[1]:+.4f}, {bm[2]:+.4f}] P(better) {bm[3]:.2f} P(worse) {bm[4]:.2f}")
        for c, v in d.items():
            bc = boot({c: v})
            print(f"  class {c}: d {bc[0]:+.4f} [{bc[1]:+.4f}, {bc[2]:+.4f}] P(better) {bc[3]:.2f}")
        return dict(nat=bn, mix=bm, per_class={c: boot({c: v}) for c, v in d.items()},
                    lost1_c1=sum(res[r['mid']]['lost1'] for r in rows if r['cls'] == 1),
                    n_c1=sum(1 for r in rows if r['cls'] == 1),
                    base_top1_c1=sum(base[r['mid']]['r'] == 1 for r in rows if r['cls'] == 1))

    out['gen20'] = table('ungated gen20 (one engine-1 run, list-relative features re-scored)', evaluate(lambda r: r['fin2'], ok), ok)
    sweep = {}
    print('\n## genfix penalty sweep (all molecules; selection is in-sample here)\n\n| delta | c1 MRR d | c1 rank-1 losses | c2 d | c3 d | c3 gains rank 1 | mix d |\n|---|---|---|---|---|---|---|')
    for dl in DELTAS:
        res = evaluate(lambda r, dl=dl: genfix_keys(r, dl), ok)
        row = {}
        for c in (1, 2, 3):
            ms = [r['mid'] for r in ok if r['cls'] == c]
            row[c] = (np.mean([res[m]['rr'] - base[m]['rr'] for m in ms]), sum(res[m]['lost1'] for m in ms), sum(res[m]['gain1'] for m in ms))
        mixd = sum(MIX[c] * row[c][0] for c in row)
        sweep[dl] = dict(row=row, mix=mixd)
        print(f"| {dl:g} | {row[1][0]:+.4f} | {row[1][1]} | {row[2][0]:+.4f} | {row[3][0]:+.4f} | {row[3][2]} | {mixd:+.4f} |")
    out['sweep'] = {str(k): v for k, v in sweep.items()}

    # 2-fold cross-fit: on each half choose the smallest delta with zero class-1 rank-1 losses and the best mix delta
    def choose(rows):
        best = None
        for dl in DELTAS:
            res = evaluate(lambda r, dl=dl: genfix_keys(r, dl), rows)
            lost = sum(res[r['mid']]['lost1'] for r in rows if r['cls'] == 1)
            mixd = sum(MIX[c] * np.mean([res[r['mid']]['rr'] - base[r['mid']]['rr'] for r in rows if r['cls'] == c]) for c in (1, 2, 3))
            if lost == 0 and (best is None or mixd > best[1] + 1e-12):
                best = (dl, mixd)
        return best[0] if best else max(DELTAS)
    picks = {h: choose([r for r in ok if r['h'] == h]) for h in (0, 1)}
    cf = {}
    for h in (0, 1):
        dl = picks[1 - h]
        for r in ok:
            if r['h'] == h:
                cf[r['mid']] = genfix_keys(r, dl)
    print(f"\ncross-fit penalties: half 0 scored with delta {picks[1]} (chosen on half 1), half 1 with delta {picks[0]}")
    out['crossfit_picks'] = picks
    out['genfix_crossfit'] = table('genfix, cross-fitted penalty', evaluate(lambda r: cf[r['mid']], ok), ok)
    full = choose(ok)
    out['delta_full'] = full
    out['genfix_full'] = table(f'genfix at delta {full} (chosen on all molecules: in-sample)', evaluate(lambda r: genfix_keys(r, full), ok), ok)
    # lib_max view: where do the genfix changes happen
    res = evaluate(lambda r: genfix_keys(r, full), ok)
    for lo, hi in ((0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 2)):
        ms = [r for r in ok if lo <= r['lib_max'] < hi]
        if ms:
            print(f"lib_max [{lo}, {hi}): n {len(ms)}, mean new in top 25 {np.mean([res[r['mid']]['new_top25'] for r in ms]):.2f}, "
                  f"d MRR {np.mean([res[r['mid']]['rr'] - base[r['mid']]['rr'] for r in ms]):+.4f}")
    if len(sys.argv) > 3 and sys.argv[2] == '--json':
        json.dump(out, open(sys.argv[3], 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
