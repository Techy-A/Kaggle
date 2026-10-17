"""Build the CASMI26 local test bench notebook from the V22 pinned copy (never submitted).

The bench runs V22 on a pseudo-test made from the 250 enveda-np-examples natural products (timsTOF, the closest
thing to the hidden test), with the answers hidden per simulated class:
  class 1 (28%): only the query spectra are removed; other public spectra of the structure stay in the libraries.
  class 2 (20%): every spectrum of the structure (and of its tautomer twins) is removed; the structure stays a candidate.
  class 3 (52%): additionally the structure is removed from every candidate source (v4 pool, COCONUT + ChEBI/LMSD,
                 PubChem channel). Generated candidates are left alone: generating the answer is a real skill.
Class shares follow the community fit to LB scores (28/20/52, thread 748524).

Leakage controls (BENCH_LEAKFREE=1): engine 1 uses FPNet A instead of fpnet_full1 (trained on the panel), engine 2
uses megayak's leak-free FPNets (all simulation structures held out) and fits its 'ours' GBMs without the np query set.
Remaining known exposure: PubChem-channel FPNets A+B and the pv GBM rows (rosters unknown).

After the normal pass the notebook scores every stage snapshot, then replays cells 25-37 (no GPU stage) for each
variant in VARIANTS from a snapshot of the state before cell 25, and scores those too: bench_results.json.

usage: python build_bench.py <v22_dir> <out_dir> [--depth 7] [--tag name]
"""
import argparse
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
E2R3_FROM = '/home/user/work/push_v22_e2r3/notebook.ipynb'

BENCH_CELL = r'''"""BENCH: pseudo-test from enveda-np-examples with the answers hidden by simulated class. Never submitted."""
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
_np_parts, _formulas = [], {}
for _g in range(_pf.num_row_groups):
    _t = _pf.read_row_group(_g, columns=['ingest_lib', 'inchikey14', 'normalized_smiles', 'molecular_formula'] + _TEST_COLS).to_pandas()
    _np_parts.append(_t[_t.ingest_lib == 'enveda-np-examples'])
NPQ = pd.concat(_np_parts, ignore_index=True); del _np_parts
ANS = NPQ.drop_duplicates('inchikey14')[['inchikey14', 'normalized_smiles', 'molecular_formula']].reset_index(drop=True)
ANS['metric_key'] = [_bchem.score_key(s) for s in ANS.normalized_smiles]
ANS['h'] = [hashlib.md5(k.encode()).hexdigest() for k in ANS.inchikey14]
ANS = ANS.sort_values('h', ignore_index=True)
_n = len(ANS); _n3 = round(0.52 * _n); _n2 = round(0.20 * _n)
ANS['cls'] = [3] * _n3 + [2] * _n2 + [1] * (_n - _n3 - _n2)
ANS['molecule_id'] = 'b_' + ANS.inchikey14
_c23 = ANS[ANS.cls >= 2]; _c3 = ANS[ANS.cls == 3]
_k23 = set(_c23.metric_key.dropna()) | set(_c23.inchikey14)
_f23 = set(_c23.molecular_formula)
# tautomer twins: training structures with the same formula whose metric key equals an answer's
_cand = {}
for _g in range(_pf.num_row_groups):
    _t = _pf.read_row_group(_g, columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).to_pandas()
    _t = _t[_t.molecular_formula.isin(_f23)].drop_duplicates('inchikey14')
    for _ik, _s in zip(_t.inchikey14, _t.normalized_smiles):
        _cand.setdefault(_ik, _s)
DROP_SPEC = set(_c23.inchikey14) | {ik for ik, s in _cand.items() if _bchem.score_key(s) in _k23}
DROP3 = set(_c3.metric_key.dropna()) | set(_c3.inchikey14)
DROP3 |= {ik for ik, s in _cand.items() if ik not in DROP3 and _bchem.score_key(s) in DROP3}
print(f'BENCH answers {_n}: class3 {_n3}, class2 {_n2}, class1 {_n - _n3 - _n2}; '
      f'spectra dropped for {len(DROP_SPEC)} ik14 (twins incl.); class-3 keys {len(DROP3)}', flush=True)
# filtered train: no np-examples rows, no spectra of class-2/3 answers or their twins
_w = None; _kept = 0
for _g in range(_pf.num_row_groups):
    _t = _pf.read_row_group(_g)
    _lib = _t.column('ingest_lib').to_pandas(); _ik = _t.column('inchikey14').to_pandas()
    _m = ~((_lib == 'enveda-np-examples') | _ik.isin(DROP_SPEC)).values
    _t = _t.filter(pa.array(_m)); _kept += _t.num_rows
    if _w is None:
        _w = pq.ParquetWriter(os.path.join(BENCH_DIR, 'train.parquet'), _t.schema)
    _w.write_table(_t)
_w.close(); del _t
# pseudo-test in the test schema
NPQ = NPQ.merge(ANS[['inchikey14', 'molecule_id']], on='inchikey14')
NPQ['spectrum_id'] = [f'bs_{i:05d}' for i in range(len(NPQ))]
_te0 = pd.read_parquet(os.path.join(os.path.dirname(_TRAIN), 'test.parquet'))
NPQ[list(_te0.columns)].to_parquet(os.path.join(BENCH_DIR, 'test.parquet'), index=False); del _te0
pd.DataFrame({'molecule_id': sorted(ANS.molecule_id), 'smiles': ';'.join(['CCO'] * 25)}).to_csv(
    os.path.join(BENCH_DIR, 'sample_submission.csv'), index=False)
ANS.drop(columns='h').to_csv('/kaggle/working/bench_answers.csv', index=False)
print(f'BENCH train rows kept {_kept:,}; pseudo-test {len(NPQ)} spectra / {NPQ.molecule_id.nunique()} molecules', flush=True)
# class-3 answers stay in the pool files (engine-1 feature caches are row-aligned to them); they are hidden
# from engine 1's candidate window (cell 19 patch, the author's own drop_pid mechanism) and from the PubChem gate.
json.dump(sorted(DROP3), open('/kaggle/working/bench_drop3.json', 'w'))
json.dump([[ik, s] for ik, s in zip(ANS[ANS.cls == 2].inchikey14, ANS[ANS.cls == 2].normalized_smiles)],
          open('/kaggle/working/bench_add2.json', 'w'))
os.environ['BENCH_DROP3'] = '/kaggle/working/bench_drop3.json'
os.environ['BENCH_ADD2'] = '/kaggle/working/bench_add2.json'
os.environ['BENCH_EXCL_NP'] = '1'
os.environ['BENCH_LEAKFREE'] = '__LEAKFREE__'
os.environ['BENCH_DEPTH'] = '__DEPTH__'
COMP = BENCH_DIR; IS_RERUN = True; ICE_BUDGET = 5400
E004_PREFLIGHT['input_ids'] = sorted(ANS.molecule_id); E004_PREFLIGHT['runtime_input_path'] = BENCH_DIR
print(f'BENCH ready {time.time() - _BT0:.0f}s | COMP {COMP} | POOL_DIR {POOL_DIR}', flush=True)
'''

# engine-2 patches, applied to the ENG_FILES strings and the runner source inside cell 14
ENG_PATCH = r'''
# ---- BENCH patches for engine 2 (class-3 drop, class-2 re-add, ranker rows without the np panel) ----
_ce = ENG_FILES['casmi_engine.py']
_a = '    log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")\n'
assert _ce.count(_a) == 1
_ce = _ce.replace(_a, _a + """    import json as _bj
    _d3 = set(_bj.load(open(os.environ['BENCH_DROP3']))) if os.environ.get('BENCH_DROP3') else set()
    if _d3:
        _bm = ~pd.Series(co_keys).isin(_d3).values
        co_fp, co_mass, co_keys, co_smi = co_fp[_bm], co_mass[_bm], co_keys[_bm], co_smi[_bm]
        log(f"BENCH dropped {int((~_bm).sum())} class-3 structures from COCONUT/bio")
""")
_a = '    tr = tr[~tr.ik.isin(set(co_keys))]\n'
assert _ce.count(_a) == 1
_ce = _ce.replace(_a, _a + """    if _d3:
        tr = tr[~tr.ik.isin(_d3)]
    if os.environ.get('BENCH_ADD2'):
        _ad = pd.DataFrame(_bj.load(open(os.environ['BENCH_ADD2'])), columns=['ik', 'smi'])
        _ad = _ad[~_ad.ik.isin(set(co_keys)) & ~_ad.ik.isin(set(tr.ik))]
        tr = pd.concat([tr, _ad], ignore_index=True)
        log(f"BENCH re-added {len(_ad)} class-2 structures to the training-structure pool")
""")
_a = '    X, Y, S = z["X"], z["Y"], z["S"]\n'
assert _ce.count(_a) == 1
_ce = _ce.replace(_a, _a + """    if os.environ.get('BENCH_EXCL_NP') and 'Q' in z.files and 'sets' in z.files:
        _sets = [str(s) for s in z['sets']]
        if 'np' in _sets:
            _qm = z['Q'] != _sets.index('np')
            X, Y, S = X[_qm], Y[_qm], S[_qm]
            log(f"BENCH ours-ranker rows without the np panel: {int(_qm.sum()):,}")
""")
ENG_FILES['casmi_engine.py'] = _ce
'''

RUNNER_OLD = "    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200\\n"
RUNNER_NEW = (RUNNER_OLD +
              "    if os.environ.get(\\'BENCH_DEPTH\\'):\\n"
              "        import pv as _pv; _bd = int(os.environ[\\'BENCH_DEPTH\\'])\\n"
              "        E.RANK.GBM[\\'max_depth\\'] = _bd; _pv.CFG.GBM[\\'max_depth\\'] = _bd; print(\\'BENCH depth\\', _bd, flush=True)\\n")
FP_OLD = "    fp_models = sorted(p for p in glob.glob(\\'/kaggle/input/**/fp_*.pt\\', recursive=True) if \\'casmi26-fp-models-v2\\' in p)\\n"
FP_NEW = (FP_OLD +
          "    if os.environ.get(\\'BENCH_LEAKFREE\\') == \\'1\\':\\n"
          "        fp_models = sorted(glob.glob(\\'/kaggle/input/**/ours_fpnet_*.pt\\', recursive=True))\\n")

V17_OLD = "    _init_worker0(pc_dir, bits_path, pool_meta_path, code_dir)\\n"
V17_NEW = (V17_OLD +
           "    if os.environ.get('BENCH_DROP3'):\\n"
           "        import json as _bj; _W['pool_keys'] = set(_W['pool_keys']) | set(_bj.load(open(os.environ['BENCH_DROP3'])))\\n")

FULL_OLD = "_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m"
FULL_NEW = ("_FULL = [] if os.environ.get('BENCH_LEAKFREE') == '1' else "
            "glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)   # v4m (BENCH)")

WIN_OLD = "L = Library('work'); P = Pool('work')\n"
WIN_NEW = WIN_OLD + """if os.environ.get('BENCH_DROP3'):   # BENCH: class-3 answers never enter engine 1's candidate window
    _bd3 = set(json.load(open(os.environ['BENCH_DROP3'])))
    _bdp = np.flatnonzero(pd.Series(P.key).isin(_bd3).values)
    _bw0 = P.window
    def _bench_window(target, ppm, _w0=_bw0, _d=_bdp):
        c = _w0(target, ppm)
        return c[~np.isin(c, _d)]
    P.window = _bench_window
    print('BENCH engine-1 window hides', len(_bdp), 'pool rows', flush=True)
"""

# PubChem runner: the gate's best pool f.z must not see class-3 answers (single-quoted literal: escape quotes)
GATE_OLD = "    pfp = np.load(os.path.join(POOL_DIR, \\'pool_fp.npy\\'), mmap_mode=\\'r\\')\\n"
GATE_NEW = (GATE_OLD +
            "    _bk = pd.read_parquet(os.path.join(POOL_DIR, \\'pool_meta.parquet\\'), columns=[\\'key\\']).key.values\\n"
            "    _bd3 = set(json.load(open(os.environ[\\'BENCH_DROP3\\']))) if os.environ.get(\\'BENCH_DROP3\\') else set()\\n"
            "    _bdm = np.isin(_bk, list(_bd3)); print(\\'BENCH gate hides\\', int(_bdm.sum()), \\'pool rows\\', flush=True)\\n")
BEST_OLD = "            best = float((F @ z.astype(np.float64)).max())\\n"
BEST_NEW = ("            _sc = F @ z.astype(np.float64); _sc[_bdm[a:b]] = -np.inf\\n"
            "            best = float(_sc.max()) if np.isfinite(_sc).any() else float(\\'nan\\')\\n")

SNAP_CELL = r'''"""BENCH: snapshot of the state before cell 25 (everything after it is CPU-only and replayable)."""
import copy as _copy, types as _types
_BSKIP = {'L', 'P', 'E', 'V', 'bank', 'In', 'Out', 'NPQ', 'BENCH_SNAP'}
BENCH_SNAP = {}
for _k, _v in list(globals().items()):
    if _k.startswith('_') or _k in _BSKIP or isinstance(_v, (_types.ModuleType, _types.FunctionType, type)):
        continue
    try:
        if isinstance(_v, (dict, list, set, tuple, str, int, float, bool, type(None))):
            BENCH_SNAP[_k] = _copy.deepcopy(_v)
        elif isinstance(_v, (np.ndarray, pd.DataFrame, pd.Series)):
            BENCH_SNAP[_k] = _v.copy()
    except Exception as _e:
        print('BENCH snapshot skip', _k, repr(_e)[:80])
print('BENCH snapshot', len(BENCH_SNAP), 'globals')
import pickle as _pk
try:
    _pk.dump(BENCH_SNAP, open('/kaggle/working/bench_snap.pkl', 'wb'), protocol=4)
    print('BENCH snapshot saved', os.path.getsize('/kaggle/working/bench_snap.pkl') // 2**20, 'MB')
except Exception as _e:
    print('BENCH snapshot not saved', repr(_e)[:120])
'''

SCORE_CELL = r'''"""BENCH: score every stage snapshot (main pass) against the hidden answers."""
import glob as _glob
_BANS = pd.read_csv('/kaggle/working/bench_answers.csv', dtype=str, keep_default_na=False)
_BANS['akey'] = [canonicalize_smiles_to_ik14(s) for s in _BANS.normalized_smiles]
_BKEY = dict(zip(_BANS.molecule_id, _BANS.akey)); _BCLS = dict(zip(_BANS.molecule_id, _BANS.cls.astype(int)))

def bench_score(path):
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    rr = {}
    for mid, smi in zip(f.molecule_id, f.smiles):
        r = 0.0
        for i, s in enumerate([x for x in smi.split(';') if x][:25]):
            if canonicalize_smiles_to_ik14(s) == _BKEY.get(mid):
                r = 1.0 / (i + 1); break
        rr[mid] = r
    out = {'all': float(np.mean([rr.get(m, 0.0) for m in _BKEY]))}
    for c in (1, 2, 3):
        ms = [m for m in _BKEY if _BCLS[m] == c]
        out[f'c{c}'] = float(np.mean([rr.get(m, 0.0) for m in ms]))
        out[f'c{c}_top1'] = int(sum(rr.get(m, 0.0) == 1.0 for m in ms))
        out[f'c{c}_hit25'] = int(sum(rr.get(m, 0.0) > 0 for m in ms))
    out['rr'] = rr
    return out

BENCH_RESULTS = {'main': {}}
for _p in ['baseline_submission.csv', 'submission_reference.csv', 'submission_e004_control.csv',
           'submission_base_e2r2.csv', 'submission.csv']:
    if os.path.exists(_p):
        BENCH_RESULTS['main'][_p] = bench_score(_p)
        print('BENCH main', _p, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in BENCH_RESULTS['main'][_p].items() if k != 'rr'}, flush=True)
json.dump(BENCH_RESULTS, open('/kaggle/working/bench_results.json', 'w'))
print('BENCH_JSON ' + json.dumps({'main': BENCH_RESULTS['main']}), flush=True)
'''

REPLAY_CELL = r'''"""BENCH: replay cells 25-37 (CPU only) per variant from the pre-25 snapshot, score each final submission."""
import copy as _copy, shutil as _sh, traceback as _tb
BENCH_CELLS = __CELLS__
BENCH_VARIANTS = __VARIANTS__
_BMAIN = {k: _copy.deepcopy(v) for k, v in BENCH_SNAP.items()}
for _name, _spec in BENCH_VARIANTS.items():
    _t0 = time.time()
    try:
        for _k, _v in _BMAIN.items():
            globals()[_k] = _copy.deepcopy(_v)
        for _idx in [i for i in sorted(BENCH_CELLS, key=int) if i not in _spec.get('skip', [])] + (['e2r3'] if _spec.get('e2r3') else []):
            _src = BENCH_CELLS[_idx] if _idx != 'e2r3' else BENCH_E2R3
            for _old, _new in _spec.get('edits', {}).get(str(_idx), []):
                assert _src.count(_old) == 1, (_name, _idx, _old)
                _src = _src.replace(_old, _new)
            exec(compile(_src, f'<bench {_name} cell {_idx}>', 'exec'), globals())
        BENCH_RESULTS[_name] = bench_score('submission.csv')
        _sh.copy('submission.csv', f'/kaggle/working/bench_sub_{_name}.csv')
        print('BENCH variant', _name, f'{time.time() - _t0:.0f}s',
              {k: (round(v, 4) if isinstance(v, float) else v) for k, v in BENCH_RESULTS[_name].items() if k != 'rr'}, flush=True)
    except Exception:
        print('BENCH variant FAILED', _name); _tb.print_exc()
    json.dump(BENCH_RESULTS, open('/kaggle/working/bench_results.json', 'w'))
    if _name in BENCH_RESULTS:
        print('BENCH_JSON ' + json.dumps({_name: BENCH_RESULTS[_name]}), flush=True)
'''

VARIANTS = {
    'replay_v22': {},
    'rel100': {'edits': {'25': [['(rel > 150.0)', '(rel > 100.0)']]}},
    'rel250': {'edits': {'25': [['(rel > 150.0)', '(rel > 250.0)']]}},
    'e2r3': {'e2r3': True},
    'no_e2r2': {'skip': ['37']},
}


def code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': src.splitlines(True)}


def src_of(cell):
    return ''.join(cell['source'])


def patch(cells, i, old, new):
    s = src_of(cells[i])
    assert s.count(old) == 1, (i, old[:80], s.count(old))
    cells[i]['source'] = s.replace(old, new).splitlines(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('v22_dir'); ap.add_argument('out_dir')
    ap.add_argument('--depth', default='')
    ap.add_argument('--leakfree', default='1')
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    nb = json.load(open(os.path.join(a.v22_dir, 'notebook.ipynb')))
    md = json.load(open(os.path.join(a.v22_dir, 'kernel-metadata.json')))
    cells = nb['cells']
    assert len(cells) == 38 and 'SMOKE switch' in src_of(cells[12])
    # cell 14: engine-2 patches (runner + module strings)
    patch(cells, 14, RUNNER_OLD, RUNNER_NEW)
    patch(cells, 14, FP_OLD, FP_NEW)
    s = src_of(cells[14]).splitlines(True)
    assert s[3].startswith('for _n, _src in ENG_FILES.items():')
    cells[14]['source'] = s[:3] + ENG_PATCH.splitlines(True) + s[3:]
    # cell 16: PubChem channel never lists class-3 answers
    s = src_of(cells[16])
    assert s.count(V17_OLD) == 1
    cells[16]['source'] = s.replace(V17_OLD, V17_NEW).splitlines(True)
    patch(cells, 19, FULL_OLD, FULL_NEW)
    patch(cells, 19, WIN_OLD, WIN_NEW)
    patch(cells, 16, GATE_OLD, GATE_NEW)
    patch(cells, 16, BEST_OLD, BEST_NEW)
    replay = {str(i): src_of(cells[i]) for i in range(25, 38) if cells[i]['cell_type'] == 'code'}
    e2r3 = src_of(json.load(open(E2R3_FROM))['cells'][-1])
    assert 'E2R3' in e2r3
    bench = BENCH_CELL.replace('__LEAKFREE__', a.leakfree).replace('__DEPTH__', a.depth)
    title = {'cell_type': 'markdown', 'metadata': {}, 'source': [
        f'# CASMI26 local test bench{(" (" + a.tag + ")") if a.tag else ""}: V22 on hidden-answer natural products\n', '\n',
        'Never submitted. Runs V22 on the 250 enveda-np-examples natural products as a pseudo-test with the answers '
        'hidden per simulated class (1/2/3 = 28/20/52%), scores every stage and replays the CPU-only tail for variants. '
        'Built by enveda-casmi26/bench/build_bench.py from the V22 pinned copy; all credits as in the cells below.\n']}
    out = [title] + cells[:13] + [code(bench)] + cells[13:25] + [code(SNAP_CELL)] + cells[25:38] + [code(SCORE_CELL)]
    out.append(code('BENCH_E2R3 = ' + repr(e2r3) + '\n' + REPLAY_CELL.replace('__CELLS__', repr(replay))
                    .replace('__VARIANTS__', repr(VARIANTS))))
    nb['cells'] = out
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(nb, open(os.path.join(a.out_dir, 'notebook.ipynb'), 'w'), indent=1)
    slug = 'casmi26-bench' + (f'-{a.tag}' if a.tag else '')
    md.update(id=f'aryanawasthiteykik/{slug}', title=f'CASMI26 bench{(" " + a.tag) if a.tag else ""}', is_private=True)
    md.pop('id_no', None)
    json.dump(md, open(os.path.join(a.out_dir, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', a.out_dir, len(out), 'cells, kernel', md['id'])


if __name__ == '__main__':
    main()
