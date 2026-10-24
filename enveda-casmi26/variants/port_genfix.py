"""Features-frozen generation ("genfix") for engine 1 of the public 0.464 notebook (dmitriigluzdov).

Why: engine 1's ranker features are list-relative (ranks, z-scores, gaps to the best, softmax, formula share in
casmi/engine.py, and the rank/gap/vs-lead family features in fe_v4/fams). Wider generation (gen20) adds candidates to
the list, which re-scores the candidates the 0.464 list already had and reshuffles their top 5 in 48-60% of molecules.
The ranker (ranker_v4b.pkl, 4 lambdarank boosters on 160 features) is pre-trained offline, so the features cannot be
redefined without retraining it.

What: with GENFIX = True, cell "all molecules -> base lists" keeps the 0.464 engine-1 run and list exactly as they are
(same candidates, features, scores, pool-popularity order). It then runs engine 1 a second time with the wider
generator (GENFIX_CFG), keeps only the generated structures that the first run does not have, maps their second-run
scores onto the first run's scale (quantile matching over the candidates both runs share), subtracts GENFIX_DELTA and
inserts each one behind the original candidates that score higher. The originals keep their relative order, so a
generated addition can move an original down by its own slot but can never re-score it.

GENFIX = False (the default written by `off`) leaves every output byte-identical to the 0.464: the added code only
defines functions and is skipped.

usage: python port_genfix.py <off|on> <0.464 push dir> <out dir> <kernel slug> <title> [delta] [max_new]
"""
import glob
import json
import os
import sys

LOOP = 'for gi, (mid, sub) in enumerate(mols):\n    nms = sub.nm.values[np.isfinite(sub.nm.values)]\n'

GENFIX_SRC = r'''GENFIX = __ON__   # features-frozen generation (False = the 0.464 behaviour, byte-identical)
GENFIX_CFG = dict(gen_n_analog=20, gen_max_per_parent=100, gen_max_total=500)   # gen20, second engine-1 run only
GENFIX_DELTA, GENFIX_MAX_NEW = __DELTA__, __MAXNEW__
GENFIX_STATS = dict(runs=0, failed=0, new=0, inserted=0, inserted_top25=0, inserted_top1=0)


def _genfix_new(spectra, target, C1, s1, max_new):
    """Generated structures of a gen20 engine-1 run that the 0.464 run lacks, best first, as
    (smiles, key, formula, mapped score); mapped = second-run score on the first run's scale."""
    old = {k: getattr(E.cfg, k) for k in GENFIX_CFG}
    try:
        for k, v in GENFIX_CFG.items():
            setattr(E.cfg, k, v)
        C2, X2, F2, n2, _ = V.run(spectra, target)
    finally:
        for k, v in old.items():
            setattr(E.cfg, k, v)
    if not len(C2.pid):
        return []
    s2 = rank_score(np.concatenate([X2, F2], 1), list(FEATURES) + list(n2))
    b1, b2 = {}, {}
    for i, k in enumerate(C1.key):
        b1[k] = max(b1.get(k, -np.inf), float(s1[i]))
    for i, k in enumerate(C2.key):
        if k in b1:
            b2[k] = max(b2.get(k, -np.inf), float(s2[i]))
    if len(b2) < 2:
        return []
    xs = np.sort(np.fromiter(b2.values(), float)); ys = np.sort(np.array([b1[k] for k in b2], float))

    def mapped(v):   # monotone quantile map, slope-1 extrapolation outside the shared range
        if v > xs[-1]:
            return float(ys[-1] + v - xs[-1])
        if v < xs[0]:
            return float(ys[0] + v - xs[0])
        return float(np.interp(v, xs, ys))
    out, seen = [], set(b1)
    for i in np.argsort(-s2, kind='stable'):
        k = C2.key[i]
        if k in seen or C2.pid[i] >= 0:
            continue
        seen.add(k)
        out.append((C2.smiles[i], k, C2.formula[i], mapped(float(s2[i]))))
        if len(out) >= max_new:
            break
    return out


def _genfix_place(new, top_scores, delta, topn):
    """(p, smiles, key, formula) for the additions that land inside the top-n: p = original candidates scoring higher."""
    top = np.asarray(top_scores, float)
    out = []
    for s, k, f, m in new:
        p = int((top > m - delta).sum())
        if p < topn:
            out.append((p, s, k, f))
    return out


def _genfix_insert(smis, keys, scs, forms, placed):
    """Insert the additions into the final (post-popularity) list; the originals keep their order."""
    placed = sorted(placed, key=lambda t: t[0])
    S, K, Sc, Fo, j = [], [], [], [], 0
    for i in range(len(smis) + 1):
        while j < len(placed) and placed[j][0] <= i:
            _, s, k, f = placed[j]; j += 1
            S.append(s); K.append(k); Fo.append(f + '|gen')
            Sc.append(Sc[-1] if Sc else (scs[0] if scs else 0.0))
        if i < len(smis):
            S.append(smis[i]); K.append(keys[i]); Sc.append(scs[i]); Fo.append(forms[i])
    return S, K, Sc, Fo


'''

RUN_ANCHOR = '                    if len(smis) >= TOPN: break\n'
RUN_ADD = '''                if GENFIX:   # features-frozen gen20: the list above stays as it is; additions are placed later
                    try:
                        _gx = _genfix_place(_genfix_new(spectra, target, C, score, GENFIX_MAX_NEW), scs, GENFIX_DELTA, TOPN)
                        GENFIX_STATS['runs'] += 1; GENFIX_STATS['new'] += len(_gx)
                    except Exception as e:
                        print('genfix failed, keeping the 0.464 list', mid, repr(e)); GENFIX_STATS['failed'] += 1; _gx = []
'''
INIT_ANCHOR = '    pids_ = []\n'
INIT_ADD = '    _gx = []\n'
STORE_ANCHOR = '    BASE[mid] = (smis, keys, lib_max, scs, forms)\n'
STORE_ADD = '''    if _gx:
        smis, keys, scs, forms = _genfix_insert(smis, keys, scs, forms, _gx)
        _gk = {t[2] for t in _gx}
        GENFIX_STATS['inserted'] += len(_gx); GENFIX_STATS['inserted_top25'] += sum(k in _gk for k in keys[:25])
        GENFIX_STATS['inserted_top1'] += int(keys[0] in _gk)
'''
PRINT_OLD = "print('errors', n_err, '| popularity re-ranked lists', n_pp)"
PRINT_NEW = PRINT_OLD + "\nif GENFIX:\n    print('GENFIX', GENFIX_CFG, 'delta', GENFIX_DELTA, GENFIX_STATS, flush=True)"


def patch_cell54(src, on=False, delta=0.0, max_new=60):
    """The base-list cell with the genfix option; on=False is byte-identical in behaviour."""
    def rep(s, old, new):
        assert s.count(old) == 1, old
        return s.replace(old, new)
    head = (GENFIX_SRC.replace('__ON__', repr(bool(on))).replace('__DELTA__', repr(float(delta)))
            .replace('__MAXNEW__', repr(int(max_new))))
    src = rep(src, LOOP, head + LOOP)
    src = rep(src, INIT_ANCHOR, INIT_ANCHOR + INIT_ADD)
    src = rep(src, RUN_ANCHOR, RUN_ANCHOR + RUN_ADD)
    src = rep(src, STORE_ANCHOR, STORE_ADD + STORE_ANCHOR)
    src = rep(src, PRINT_OLD, PRINT_NEW)
    return src


def base_cell_index(cells):
    hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and LOOP in ''.join(c['source'])
            and "'''all molecules -> base lists" in ''.join(c['source'])]
    assert len(hits) == 1, hits
    return hits[0]


CREDIT = ('**Variant of a pinned copy** of [dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures]'
          '(https://www.kaggle.com/code/dmitriigluzdov/casmi-26-the-data-behind-spectra-to-structures) (public 0.464). '
          'All credit to its author and the authors credited below.\n\n'
          'One change, in the cell "all molecules -> base lists": **features-frozen generation**. Engine 1 (Ahmed Berat '
          'Özer\'s v4n) runs exactly as in the original; a second engine-1 run with wider derivative generation '
          '(20 parent analogs, 100 per parent, 500 in total) contributes only structures the first run lacks. They are '
          'placed into the original list by a score mapped onto the first run\'s scale, minus a penalty of __DELTA__, '
          'so the original candidates are never re-scored (the ranker features are relative to the whole list).\n')


def main():
    mode, base, out, slug, title = sys.argv[1:6]
    delta = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
    max_new = int(sys.argv[7]) if len(sys.argv) > 7 else 60
    nb_path = glob.glob(os.path.join(base, '*.ipynb'))[0]
    nb = json.load(open(nb_path))
    cells = nb['cells']
    i = base_cell_index(cells)
    cells[i]['source'] = patch_cell54(''.join(cells[i]['source']), mode == 'on', delta, max_new).splitlines(True)
    if mode == 'on':
        assert ''.join(cells[0]['source']).startswith('**Pinned copy**')
        cells[0]['source'] = CREDIT.replace('__DELTA__', f'{delta:g}').splitlines(True)
    for c in cells:   # compile what compiles (cells without IPython magics) to catch broken edits
        src = ''.join(c['source'])
        if c['cell_type'] == 'code' and not any(l.lstrip().startswith(('!', '%')) for l in src.splitlines()):
            compile(src, 'cell', 'exec')
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, os.path.basename(nb_path)), 'w'), indent=1)
    m = json.load(open(os.path.join(base, 'kernel-metadata.json')))
    m.update(id=f'aryanawasthiteykik/{slug}', title=title, is_private=True)
    json.dump(m, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', out, '| base-list cell', i)


if __name__ == '__main__':
    main()
