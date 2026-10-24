"""Build the CPU-only genfix bench from the 0.464 pinned copy: engine 1 on a pseudo-test with a THIN class-1 slice.

Why: the gen20 post-mortem (results/calibration.md) found that adding generated candidates re-scores the candidates
engine 1 already had, and the old benches could not see the damage because their class-1 answers keep ~180 library
spectra each (test-like structures have a median of 6). This bench gives class 1 only 1-5 library spectra, so the
rate at which the correct answer loses rank 1 under a list change becomes measurable, and it compares
  base   : the 0.464 engine-1 list (gen6),
  gen20  : the old ungated gen20 list (one engine-1 run with the wider generator, list-relative features re-scored),
  genfix : base + gen20-only additions placed by a mapped score minus a penalty (variants/port_genfix.py),
all from the same two engine-1 runs per molecule, so the penalty can be swept offline (genfix_analyze.py).

Pseudo-test (answers hidden per class):
  np slice: the 250 enveda-np-examples structures with the class split of build_bench.py (130 / 50 / 70);
            class 1 keeps only k = 1..5 of its other library spectra (k from a hash of the inchikey).
  t slice : 200 extra class-1 structures that have enveda-180 (timsTOF) spectra and spectra in other libraries;
            the enveda-180 spectra are the queries, the other spectra are thinned to k = 1..5.
Class-1 tautomer twins lose all spectra; external reference spectra (0.464 knowledge table) are dropped for every
answer structure. Leak control as build_bench.py: engine 1 uses FPNet A instead of fpnet_full1.

Before the main loop the bench runs the original base-list cell, the patched cell with GENFIX = False and with
GENFIX = True on 16 molecules: the first two must give identical BASE lists (identity proof), and the third is
compared offline with the reconstruction from the logged runs.

Output: GFB_JSON lines in the run log (one per molecule) and /kaggle/working/genfix_bench.jsonl. Never submitted.

usage: python build_genfix_bench.py <0.464 push dir> <out dir>
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'variants'))
import port_genfix as pg  # noqa: E402

BENCH_CELL = r'''"""GENFIX BENCH: pseudo-test with a thin class-1 slice (answers hidden per class). Never submitted."""
import hashlib, json, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
sys.path.insert(0, V4CODE)
from casmi import chem as _bchem
_BT0 = time.time()
BENCH_DIR = '/kaggle/working/bench_comp'; os.makedirs(BENCH_DIR, exist_ok=True)
_TRAIN = os.path.realpath(os.path.join(COMP, 'train.parquet'))
_pf = pq.ParquetFile(_TRAIN)
_TEST_COLS = ['ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct', 'ionization_mode',
              'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev',
              'collision_energy_orig_units']
_md5 = lambda s: hashlib.md5(s.encode()).hexdigest()
A = pq.read_table(_TRAIN, columns=['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula']).to_pandas()
A['row'] = np.arange(len(A))
_isnp = (A.ingest_lib == 'enveda-np-examples').values
# np slice: same answers and class split as build_bench.py
ANS = A[_isnp].drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']].reset_index(drop=True)
ANS['h'] = [_md5(k) for k in ANS.inchikey14]
ANS = ANS.sort_values('h', ignore_index=True)
_n = len(ANS); _n3 = round(0.52 * _n); _n2 = round(0.20 * _n)
ANS['cls'] = [3] * _n3 + [2] * _n2 + [1] * (_n - _n3 - _n2)
ANS['slice'] = 'np'; ANS['molecule_id'] = 'b_' + ANS.inchikey14
# t slice: structures with enveda-180 spectra (the queries) and spectra in other libraries (thinned)
_npik = set(ANS.inchikey14)
_o = A[~_isnp & ~A.inchikey14.isin(_npik)]
_g = _o.assign(e=(_o.ingest_lib == 'enveda-180').values).groupby('inchikey14').e.agg(['sum', 'size'])
_ok = list(_g.index[(_g['sum'] >= 1) & (_g['size'] - _g['sum'] >= 1)])
_T = A[A.inchikey14.isin(set(_ok))].drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']]
_T = _T.assign(h=[_md5('t' + k) for k in _T.inchikey14]).sort_values('h', ignore_index=True).head(200)
_T['cls'] = 1; _T['slice'] = 't'; _T['molecule_id'] = 't_' + _T.inchikey14
ANS = pd.concat([ANS, _T], ignore_index=True)
ANS['metric_key'] = [_bchem.score_key(s) for s in ANS.normalized_smiles]
_tik = set(_T.inchikey14)
# tautomer twins (same formula, same metric key, other inchikey14)
_fa = set(ANS.molecular_formula)
_ff = A[A.molecular_formula.isin(_fa)].drop_duplicates('inchikey14')
_k2ik = {}
for _ik, _s in zip(_ff.inchikey14, _ff.normalized_smiles):
    _k = _bchem.score_key(_s)
    if _k is not None:
        _k2ik.setdefault(_k, set()).add(_ik)
_twins = lambda d: {t for _, r in d.iterrows() for t in _k2ik.get(r.metric_key, set()) if t != r.inchikey14}
_c1 = ANS[ANS.cls == 1]; _c23 = ANS[ANS.cls >= 2]; _c3 = ANS[ANS.cls == 3]
TW1 = _twins(_c1) - set(ANS.inchikey14)
DROP_SPEC = set(_c23.inchikey14) | (_twins(_c23) - set(_c1.inchikey14))
DROP3 = set(_c3.metric_key.dropna()) | set(_c3.inchikey14) | (_twins(_c3) - set(_c1.inchikey14))
# thin class 1: keep k = 1..5 of the non-query spectra (hash order)
_K = {ik: 1 + int(_md5('k' + ik), 16) % 5 for ik in _c1.inchikey14}
_q1 = _isnp | (A.inchikey14.isin(_tik) & (A.ingest_lib == 'enveda-180')).values
_cand1 = A[A.inchikey14.isin(set(_K)).values & ~_q1]
_cand1 = _cand1.assign(h=[_md5(f'{ik}:{r}') for ik, r in zip(_cand1.inchikey14, _cand1.row)]).sort_values(['inchikey14', 'h'])
_keep1 = set(_cand1.row[(_cand1.groupby('inchikey14').cumcount() < _cand1.inchikey14.map(_K)).values])
_n_before = _cand1.groupby('inchikey14').size()
ANS['k_keep'] = [(_K[ik] if ik in _K else 0) for ik in ANS.inchikey14]
ANS['n_lib_before'] = [int(_n_before.get(ik, 0)) for ik in ANS.inchikey14]
ANS['n_lib_after'] = [min(int(_n_before.get(ik, 0)), _K.get(ik, 0)) for ik in ANS.inchikey14]
_drop = _q1 | A.inchikey14.isin(DROP_SPEC | TW1).values | (A.inchikey14.isin(set(_K)).values & ~A.row.isin(_keep1).values)
KEEP = ~_drop
print(f'BENCH answers {len(ANS)}:', ANS.groupby(['slice', 'cls']).size().to_dict(),
      f'| class-1 twins dropped {len(TW1)} | class-2/3 ik14 without spectra {len(DROP_SPEC)} | class-3 keys {len(DROP3)}',
      f'| class-1 library spectra before median {float(np.median(ANS[ANS.cls == 1].n_lib_before)):.0f}, '
      f'after {float(np.median(ANS[ANS.cls == 1].n_lib_after)):.0f}', flush=True)
assert (ANS[ANS.cls == 1].n_lib_after >= 1).all()
# filtered train
_w = None; _kept = 0; _r0 = 0; _QP = []
for _gi in range(_pf.num_row_groups):
    _t = _pf.read_row_group(_gi); _nr = _t.num_rows
    _qm = _q1[_r0:_r0 + _nr] & A.inchikey14.iloc[_r0:_r0 + _nr].isin(set(ANS.inchikey14)).values
    if _qm.any():
        _QP.append(_t.filter(pa.array(_qm)).select(['inchikey14'] + _TEST_COLS).to_pandas())
    _t = _t.filter(pa.array(KEEP[_r0:_r0 + _nr])); _kept += _t.num_rows; _r0 += _nr
    if _w is None:
        _w = pq.ParquetWriter(os.path.join(BENCH_DIR, 'train.parquet'), _t.schema)
    _w.write_table(_t)
_w.close(); del _t
assert _r0 == len(A)
NPQ = pd.concat(_QP, ignore_index=True).merge(ANS[['inchikey14', 'molecule_id']], on='inchikey14')
NPQ['spectrum_id'] = [f'bs_{i:05d}' for i in range(len(NPQ))]
_te0 = pd.read_parquet(os.path.join(os.path.dirname(_TRAIN), 'test.parquet'))
NPQ[list(_te0.columns)].to_parquet(os.path.join(BENCH_DIR, 'test.parquet'), index=False); del _te0
pd.DataFrame({'molecule_id': sorted(ANS.molecule_id), 'smiles': ';'.join(['CCO'] * 25)}).to_csv(
    os.path.join(BENCH_DIR, 'sample_submission.csv'), index=False)
ANS.drop(columns='h').to_csv('/kaggle/working/bench_answers.csv', index=False)
assert set(NPQ.molecule_id) == set(ANS.molecule_id), (NPQ.molecule_id.nunique(), len(ANS))
print(f'BENCH train rows kept {_kept:,} of {len(A):,}; pseudo-test {len(NPQ)} spectra / {NPQ.molecule_id.nunique()} molecules', flush=True)
# external reference spectra: none for any answer structure (or a twin)
_ts = pd.read_parquet(os.path.join(POOL_DIR, 'train_structs.parquet'), columns=['inchikey14', 'sid'])
_BENCH_EXT_DROP = set(_ts[_ts.inchikey14.isin(set(ANS.inchikey14) | DROP_SPEC | TW1)].sid.astype(int))
json.dump(sorted(DROP3), open('/kaggle/working/bench_drop3.json', 'w'))
os.environ['BENCH_DROP3'] = '/kaggle/working/bench_drop3.json'
del A, _o, _ff, _cand1
COMP = BENCH_DIR
print(f'BENCH ready {time.time() - _BT0:.0f}s | COMP {COMP} | external-reference drop {len(_BENCH_EXT_DROP)} sids', flush=True)
'''

LOOP_CELL = r'''"""GENFIX BENCH loop: identity checks, then per molecule the 0.464 engine-1 run (gen6) and a gen20 run."""
import copy, pickle
_SRC54 = __SRC54__
_OFF = __SRC54_OFF__
_ON = __SRC54_ON__
_MOLS = "mols = list(te.groupby('molecule_id', sort=False))"
assert _SRC54.count(_MOLS) == _OFF.count(_MOLS) == _ON.count(_MOLS) == 1
_ACLS = dict(zip(ANS.molecule_id, ANS.cls.astype(int))); _AKEY = dict(zip(ANS.molecule_id, ANS.metric_key))
_ids = sorted(ANS.molecule_id, key=lambda m: hashlib.md5(m.encode()).hexdigest())
_IDSUB = set([m for m in _ids if _ACLS[m] == 1][:6] + [m for m in _ids if _ACLS[m] == 2][:4] + [m for m in _ids if _ACLS[m] == 3][:6])

def _exec54(src, tag):
    t0 = time.time()
    exec(compile(src.replace(_MOLS, "mols = [g for g in te.groupby('molecule_id', sort=False) if g[0] in _IDSUB]"),
                 f'<cell54 {tag}>', 'exec'), globals())
    print(f'identity run {tag}: {len(BASE)} molecules {time.time() - t0:.0f}s', flush=True)
    return copy.deepcopy(BASE)
_B_ORIG = _exec54(_SRC54, 'orig')
_B_OFF = _exec54(_OFF, 'genfix-off')
_same = pickle.dumps(_B_ORIG) == pickle.dumps(_B_OFF)
_diff = [m for m in _B_ORIG if _B_ORIG[m] != _B_OFF.get(m)]
_rep = None
if not _same:
    _B_ORIG2 = _exec54(_SRC54, 'orig-again')
    _rep = [m for m in _B_ORIG if _B_ORIG[m] != _B_ORIG2.get(m)]
_B_ON = _exec54(_ON, 'genfix-on')
print('GFB_IDENTITY ' + json.dumps(dict(molecules=len(_B_ORIG), byte_identical=_same, differing=_diff,
                                         orig_rerun_differing=_rep, genfix_stats=GENFIX_STATS,
                                         on_keys={m: v[1] for m, v in _B_ON.items()},
                                         orig_keys={m: v[1] for m, v in _B_ORIG.items()})), flush=True)

# main loop: exactly the base-list cell's spectra construction, then two engine-1 runs
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
_grp = dict(list(te.groupby('molecule_id', sort=False)))
_cap = {}
_run0 = V.run
def _run_cap(*a, **k):
    r = _run0(*a, **k); _cap['last'] = r; return r
V.run = _run_cap

def _base_entry(C, score):
    """== the base-list cell: top-TOPN unique keys by score, then the pool-popularity re-rank."""
    order = np.argsort(-score, kind='stable'); seen = set(); keys, scs, pids_ = [], [], []
    for i in order:
        k = C.key[i]
        if k in seen: continue
        seen.add(k); keys.append(k); scs.append(float(score[i])); pids_.append(int(C.pid[i]))
        if len(keys) >= TOPN: break
    pre = list(keys); scs0 = list(scs)
    if POOLPOP_MU > 0 and len(scs) > 1:
        s_ = np.asarray(scs, np.float64); z_ = (s_ - s_.mean()) / (s_.std() + 1e-9)
        pid_ = np.asarray(pids_); isg_ = pid_ < 0
        f_ = z_ + POOLPOP_MU * np.where(isg_, 0.0, POOL_POP[np.maximum(pid_, 0)])
        pool_i = np.where(~isg_)[0]; o_ = np.empty(len(pid_), np.int64)
        o_[isg_] = np.where(isg_)[0]; o_[~isg_] = pool_i[np.argsort(-f_[pool_i], kind='stable')]
        keys = [keys[i] for i in o_]
    return pre, scs0, keys, [int(p < 0) for p in pids_]

def _arank(C, score, k):
    order = np.argsort(-score, kind='stable'); seen = []
    for i in order:
        if C.key[i] in seen: continue
        seen.append(C.key[i])
        if C.key[i] == k: return len(seen), int(C.pid[i] < 0)
    return 0, 0

_out = open('/kaggle/working/genfix_bench.jsonl', 'w')
_T0L = time.time()
for _gi, _mid in enumerate(_ids):
    sub = _grp[_mid]
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    rec = dict(mid=_mid, cls=_ACLS[_mid])
    try:
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        _t = time.time()
        C1, X1, F1, n1, info = V.run(spectra, target)
        rec['t1'] = round(time.time() - _t, 2); rec['lib_max'] = float(info.get('lib_max', 0.0))
        s1 = rank_score(np.concatenate([X1, F1], 1), list(FEATURES) + list(n1))
        pre1, scs1, fin1, g1 = _base_entry(C1, s1)
        _t = time.time()
        new = _genfix_new(spectra, target, C1, s1, 60)
        rec['t2'] = round(time.time() - _t, 2)
        C2, X2, F2, n2, _ = _cap['last']
        s2 = rank_score(np.concatenate([X2, F2], 1), list(FEATURES) + list(n2))
        pre2, scs2, fin2, g2 = _base_entry(C2, s2)
        k = _AKEY.get(_mid)
        rec.update(n1=len(C1.pid), n2=len(C2.pid), ng1=int((C1.pid < 0).sum()), ng2=int((C2.pid < 0).sum()),
                   pre1=pre1, scs1=scs1, fin1=fin1, g1=g1, fin2=fin2, g2=g2, new=[[x[1], x[3]] for x in new],
                   ar1=_arank(C1, s1, k), ar2=_arank(C2, s2, k))
        if _mid in _B_ORIG:
            rec['id_ok'] = fin1 == _B_ORIG[_mid][1]
    except Exception as e:
        rec['error'] = repr(e)
    line = json.dumps(rec)
    _out.write(line + '\n'); _out.flush()
    print('GFB_JSON ' + line, flush=True)
    if (_gi + 1) % 25 == 0:
        print(f'  {_gi + 1}/{len(_ids)} molecules {time.time() - _T0L:.0f}s', flush=True)
_out.close()
V.run = _run0
print('GFB_DONE', len(_ids), f'{time.time() - _T0L:.0f}s', flush=True)
'''


def src(cell):
    return ''.join(cell['source'])


def code(s):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.splitlines(True)}


def rep(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


def main():
    base, out = sys.argv[1], sys.argv[2]
    nb_path = glob.glob(os.path.join(base, '*.ipynb'))[0]
    nb = json.load(open(nb_path))
    md = json.load(open(os.path.join(base, 'kernel-metadata.json')))
    cells = nb['cells']
    want = {'rdkit': "assert rdBase.rdkitVersion == '2026.03.3'", 'data': 'def locate_data_dir():',
            'mounts': "TABLE = dataset_dir('casmi26-natural-product-knowledge-table'",
            'setup': "V4CODE = os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/engine.py')))",
            'determinism': 'def _v43_patch_engine(', 'cache': 'def _build_external_reference_cache(',
            'engine': "RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))"}
    ix = {}
    for k, marker in want.items():
        hits = [i for i, c in enumerate(cells) if c['cell_type'] == 'code' and marker in src(c)]
        assert len(hits) == 1, (k, hits)
        ix[k] = hits[0]
    i54 = pg.base_cell_index(cells)
    s51 = src(cells[ix['cache']])
    a = ("                if link['sid'] is None or not r['in_train_structure'] or q['mass_check_status'] != "
         "'within_diagnostic_threshold':\n                    continue\n")
    s51 = rep(s51, a, a + "                if int(link['sid']) in _BENCH_EXT_DROP:   # BENCH: no external spectra of answers\n"
                          "                    continue\n")
    s51 = rep(s51, "        check(len(extra_meta) == pins['added_rows'] and len(used_keys) == pins['added_keys'], "
                   "'eligible extension count differs')\n",
              "        print('BENCH external reference rows', len(extra_meta), 'of', pins['added_rows'], flush=True)\n")
    s52 = src(cells[ix['engine']])
    s52 = rep(s52, "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m",
              "_FULL = []   # BENCH leak control: FPNet A (fpnet_full1 was trained on the np panel)")
    w = "L = _external_reference_library(_EXTERNAL_REFERENCE_STATE); P = Pool('work')\n"
    s52 = rep(s52, w, w + """if os.environ.get('BENCH_DROP3'):   # BENCH: class-3 answers never enter engine 1's candidate window
    _bd3 = set(json.load(open(os.environ['BENCH_DROP3'])))
    _bdp = np.flatnonzero(pd.Series(P.key).isin(_bd3).values)
    _bw0 = P.window
    def _bench_window(target, ppm, _w0=_bw0, _d=_bdp):
        c = _w0(target, ppm)
        return c[~np.isin(c, _d)]
    P.window = _bench_window
    print('BENCH engine-1 window hides', len(_bdp), 'pool rows', flush=True)
""")
    s54 = src(cells[i54])
    # the bench needs the base-list cell's globals (POOLPOP_MU, POOL_POP, GENFIX functions): define them once
    head = s54[:s54.index(pg.LOOP)]
    defs = pg.GENFIX_SRC.replace('__ON__', 'False').replace('__DELTA__', '0.0').replace('__MAXNEW__', '60')
    loop = (LOOP_CELL.replace('__SRC54_OFF__', repr(pg.patch_cell54(s54, False)))
            .replace('__SRC54_ON__', repr(pg.patch_cell54(s54, True, 0.0, 60))).replace('__SRC54__', repr(s54)))
    title = {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# CASMI26 genfix bench (CPU, never submitted)\n', '\n',
        'Engine 1 of the public 0.464 notebook (dmitriigluzdov; engine by Ahmed Berat Özer) on a pseudo-test with a thin '
        'class-1 slice. Compares the 0.464 list, ungated gen20 and features-frozen gen20. Built by '
        'enveda-casmi26/bench/build_genfix_bench.py; credits as in the original cells below.\n']}
    nb['cells'] = ([title] + [cells[ix[k]] for k in ('rdkit', 'data', 'mounts', 'setup', 'determinism')]
                   + [code(BENCH_CELL), code(s51), code(s52), code(head + '\n' + defs), code(loop)])
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            compile(src(c), 'cell', 'exec')
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, 'notebook.ipynb'), 'w'), indent=1)
    md.update(id='aryanawasthiteykik/casmi26-genfix-bench', title='CASMI26 genfix bench', is_private=True,
              enable_gpu=False, code_file='notebook.ipynb', keywords=[])
    md.pop('machine_shape', None)
    json.dump(md, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', out, len(nb['cells']), 'cells', ix, 'base-list', i54)


if __name__ == '__main__':
    main()
