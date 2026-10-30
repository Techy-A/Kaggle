"""Build the CASMI26 bench notebook `casmi26-bench477` from our exact copy of busyaprime's 0.477 (never submitted).

What: the 0.477 notebook runs the frozen public 0.465 pipeline (FROZEN_B64, a base64 notebook pinned by
FROZEN_SHA256) in a subprocess, then the neural RRF variants (V0-V4, CUSTOM 0.25/0.75), Roman Tamrazov's fragment
check on the V2 order and busyaprime's merge. This script decodes FROZEN_B64, appends build_bench.py's BENCH block
(the 250 enveda-np-examples molecules, answers hidden by simulated class 1/2/3 = 28/20/52%) to the frozen SMOKE-switch
cell 30, applies the V22 bench's leak controls at their anchors in this pipeline, re-encodes it byte-compatibly
(compact JSON, as the original) and updates FROZEN_SHA256, the only pin on it (runner.py takes the hash as argv).

Changes against build_bench.py (V22): BENCH sits at the end of frozen cell 30 instead of in a new cell, so the
outer stage labels (keyed on frozen cell numbers) stay right; the engine-2 class-3 drop runs after PubChemLite (this
engine adds it after ChEBI/LMSD, so the V22 anchor would let PubChemLite re-add the answers); the external reference
spectra (MassBank/MoNA/GNPS/MSnLib linked to train SIDs, frozen cell 36) of class-2/3 answers and their twins are
hidden too; the engine-1 window patch follows the new library line. No depth or generation options.

Outer notebook: after the pipeline, COMP and SAMPLE_IDS point at the bench files (the fragment check reads
COMP/test.parquet); the fragment check's own list and the merged 0.477 list are copied into variants/; a final cell
scores every list against the answers (bench_results.json: MRR overall and per class, top-1, hit@25, per-molecule
reciprocal ranks), writes bench_candidate_truth.parquet (is_answer for every listed and outside candidate) and packs
the intermediates (variant lists, neural features with neural_score/neural_cos/neural_ok, frozen_candidate_evidence
with source ranks and library_max, reports) into bench477_intermediates.tar.gz, also printed as BENCH_BLOB lines
because kernel outputs can be unreachable from the sandbox. COMMIT_MODE stays 'smoke' and is asserted not 'fast';
the frozen smoke switch is overridden by BENCH (IS_RERUN = True), so every bench molecule runs the full pipeline.

Output: <out_dir>/casmi26-bench477.ipynb and kernel-metadata.json (private, same image pin and data sources as the
copy: the BENCH block needs only the competition train.parquet; the leak-free FPNets ship inside the v4g assets).

usage: python build_bench477.py [--src /home/user/work/push_c477] [--out /home/user/work/push_bench477]
"""
import argparse
import base64
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_bench as B  # noqa: E402  (BENCH block and leak-control patches of the V22 bench)

SLUG = 'casmi26-bench477'

# ---- frozen pipeline patches ----
ENG_PATCH = B.ENG_PATCH.replace(
    r"""_a = '    log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")\n'""",
    r"""_a = '        log(f"PubChemLite CC0 skipped: {_pe!r}")\n'""")
assert ENG_PATCH != B.ENG_PATCH

BENCH_TAIL = r"""
json.dump(sorted(DROP_SPEC), open('/kaggle/working/bench_dropspec.json', 'w'))
os.environ['BENCH_DROPSPEC'] = '/kaggle/working/bench_dropspec.json'
"""

EXT_OLD = "        check(len(extra_meta) > 0, 'no eligible external references')\n"
EXT_NEW = EXT_OLD + """        if os.environ.get('BENCH_DROPSPEC'):   # BENCH: no external spectra of class-2/3 answers or their twins
            import json as _bj
            _bd = set(_bj.load(open(os.environ['BENCH_DROPSPEC'])))
            _ts = pq.read_table(base / 'train_structs.parquet', columns=['sid', 'inchikey14']).to_pandas()
            _bad = set(_ts.sid[_ts.inchikey14.isin(_bd)].tolist())
            _bk = [d['sid'] not in _bad for d in extra_meta]
            print('BENCH external references hidden', len(_bk) - sum(_bk), 'of', len(_bk), flush=True)
            extra_meta = [x for x, k in zip(extra_meta, _bk) if k]
            extra_mz = [x for x, k in zip(extra_mz, _bk) if k]
            extra_it = [x for x, k in zip(extra_it, _bk) if k]
            check(len(extra_meta) > 0, 'no eligible external references')
"""

# V17_OLD sits in a single-quoted literal here (V17_PATCH), so the added lines need escaped quotes
V17_NEW = B.V17_OLD + B.V17_NEW[len(B.V17_OLD):].replace("'", "\\'")

WIN_OLD = "L = _external_reference_library(_EXTERNAL_REFERENCE_STATE); P = Pool('work')\n"
WIN_NEW = WIN_OLD + B.WIN_NEW[len(B.WIN_OLD):]

# ---- outer notebook patches ----
RUN_OLD = 'PIPELINE_DONE = run_public_pipeline()\n'
RUN_NEW = RUN_OLD + """assert COMMIT_MODE != 'fast' and PIPELINE_DONE, 'BENCH: the full pipeline must run'
assert (ROOT / 'bench_comp/sample_submission.csv').is_file(), 'BENCH block did not run'
COMP = ROOT / 'bench_comp'   # BENCH: the fragment check and the answer checks read the bench files
SAMPLE_IDS = pd.read_csv(COMP / 'sample_submission.csv', dtype=str).molecule_id.tolist()
print('BENCH outer COMP', COMP, '| molecules', len(SAMPLE_IDS), flush=True)
"""

SAVE_PHYSICAL = """# BENCH: the fragment check's own list (V2 order), before the merge cell overwrites submission.csv
shutil.copy2(ROOT / 'submission.csv', VARIANTS_DIR / 'submission_physical_V2.csv')
print('BENCH saved', VARIANTS_DIR / 'submission_physical_V2.csv', json.dumps(PARALLAX_DIAGNOSTICS, default=str)[:400], flush=True)
"""
SAVE_MERGED = """# BENCH: the merged 0.477 list
shutil.copy2(ROOT / 'submission.csv', VARIANTS_DIR / 'submission_merged477.csv')
print('BENCH saved', VARIANTS_DIR / 'submission_merged477.csv', flush=True)
"""

SCORE_CELL = r'''"""BENCH: score every list against the hidden answers, save per-candidate truth, pack the intermediates."""
import functools, tarfile
_mi = {'__name__': 'bench_metric'}
exec(compile(next(''.join(c['source']) for c in json.loads(FROZEN.read_text())['cells']
                  if c['cell_type'] == 'code' and 'def metric_identity' in ''.join(c['source'])), 'bench_metric', 'exec'), _mi)
_ik = functools.lru_cache(maxsize=None)(_mi['canonicalize_smiles_to_ik14'])
_BANS = pd.read_csv(ROOT / 'bench_answers.csv', dtype=str, keep_default_na=False)
_BKEY = {m: _ik(s) for m, s in zip(_BANS.molecule_id, _BANS.normalized_smiles)}
_BCLS = dict(zip(_BANS.molecule_id, _BANS.cls.astype(int)))
assert sorted(_BKEY) == sorted(SAMPLE_IDS) and None not in _BKEY.values()


def bench_score(path):
    f = pd.read_csv(path, dtype=str, keep_default_na=False)
    rr = {}
    for mid, smi in zip(f.molecule_id, f.smiles):
        keys = [_ik(s) for s in smi.split(';') if s][:25]
        rr[mid] = next((1.0 / (i + 1) for i, k in enumerate(keys) if k == _BKEY.get(mid)), 0.0)
    out = {'all': float(np.mean([rr.get(m, 0.0) for m in _BKEY]))}
    for c in (1, 2, 3):
        ms = [m for m in _BKEY if _BCLS[m] == c]
        out[f'c{c}'] = float(np.mean([rr.get(m, 0.0) for m in ms]))
        out[f'c{c}_n'] = len(ms)
        out[f'c{c}_top1'] = int(sum(rr.get(m, 0.0) == 1.0 for m in ms))
        out[f'c{c}_hit25'] = int(sum(rr.get(m, 0.0) > 0 for m in ms))
    out['rr'] = rr
    return out


BENCH_LISTS = {'baseline0465': BASELINE, 'public0466': ROOT / 'submission_public0466.csv',
               **{v: VARIANTS_DIR / f'submission_{v}.csv' for v in ('V0', 'V1', 'V2', 'V3', 'V4', 'CUSTOM')},
               'physical_V2': VARIANTS_DIR / 'submission_physical_V2.csv',
               'merged477': VARIANTS_DIR / 'submission_merged477.csv'}
BENCH_RESULTS = {'classes': _BCLS}
for _n, _p in BENCH_LISTS.items():
    if _p.is_file():
        BENCH_RESULTS[_n] = bench_score(_p)
        print('BENCH', _n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in BENCH_RESULTS[_n].items() if k != 'rr'}, flush=True)
    else:
        print('BENCH missing', _n, _p, flush=True)
(ROOT / 'bench_results.json').write_text(json.dumps(BENCH_RESULTS))
print('BENCH_JSON ' + json.dumps({k: v for k, v in BENCH_RESULTS.items()}), flush=True)

# is_answer for every candidate the variants could use (listed + outside), so step 3 needs no RDKit
_rows = []
for _p in [ROOT / 'neural_candidate_features.parquet', ROOT / 'outsider_neural_features.parquet']:
    if _p.is_file():
        _f = pd.read_parquet(_p, columns=['molecule_id', 'smiles'])
        _rows += list(zip(_f.molecule_id.astype(str), _f.smiles))
for _p in BENCH_LISTS.values():
    if _p.is_file():
        _f = pd.read_csv(_p, dtype=str, keep_default_na=False)
        _rows += [(m, s) for m, smi in zip(_f.molecule_id, _f.smiles) for s in smi.split(';') if s]
_truth = pd.DataFrame(sorted(set(_rows)), columns=['molecule_id', 'smiles'])
_truth['ik14'] = [_ik(s) for s in _truth.smiles]
_truth['is_answer'] = [k is not None and k == _BKEY.get(m) for m, k in zip(_truth.molecule_id, _truth.ik14)]
_truth['cls'] = _truth.molecule_id.map(_BCLS)
_truth.to_parquet(ROOT / 'bench_candidate_truth.parquet', index=False)
print('BENCH truth rows', len(_truth), '| answers listed', int(_truth.is_answer.sum()), flush=True)

_pack = [p for p in sorted(VARIANTS_DIR.glob('submission_*.csv'))] + [ROOT / n for n in (
    'submission_baseline0465.csv', 'submission_public0466.csv', 'submission.csv', 'bench_results.json',
    'bench_answers.csv', 'bench_candidate_truth.parquet', 'neural_candidate_features.parquet',
    'outsider_evidence.parquet', 'outsider_neural_features.parquet', 'frozen_candidate_evidence.parquet',
    'frozen_run_info.json', 'execution_receipt.json', 'spectral_parallax_report.json', 'spectral_parallax_audit.csv',
    'merge_report.json', 'bench_drop3.json', 'bench_dropspec.json', 'e004_candidate_inputs.json')]
with tarfile.open(ROOT / 'bench477_intermediates.tar.gz', 'w:gz') as _tf:
    for _p in _pack:
        if _p.is_file():
            _tf.add(_p, arcname=str(_p.relative_to(ROOT)))
_blob = base64.b64encode((ROOT / 'bench477_intermediates.tar.gz').read_bytes()).decode()
print('BENCH intermediates', len(_blob) // 1024, 'KB base64,', sum(p.is_file() for p in _pack), 'files', flush=True)
if len(_blob) <= 12 << 20:   # ponytail: the log carries up to 12 MB; above that, fetch the file from the kernel output
    for _i in range(0, len(_blob), 60000):
        print('BENCH_BLOB ' + _blob[_i:_i + 60000], flush=True)
print('BENCH_BLOB_END', hashlib.sha256((ROOT / 'bench477_intermediates.tar.gz').read_bytes()).hexdigest(), flush=True)
'''


def code(src, cid):
    return {'cell_type': 'code', 'execution_count': None, 'id': cid, 'metadata': {}, 'outputs': [],
            'source': src.splitlines(True)}


def src_of(cell):
    return ''.join(cell['source'])


def patch(cells, i, old, new):
    s = src_of(cells[i])
    assert s.count(old) == 1, (i, old[:80], s.count(old))
    cells[i]['source'] = s.replace(old, new).splitlines(True)


def build_frozen(raw):
    """Decoded frozen notebook bytes -> bench frozen notebook bytes (same compact JSON format)."""
    nb = json.loads(raw)
    assert json.dumps(nb, separators=(',', ':')).encode() == raw, 'frozen source is not compact JSON'
    cells = nb['cells']
    assert len(cells) == 59 and 'SMOKE switch' in src_of(cells[30])
    bench = B.BENCH_CELL.replace('__LEAKFREE__', '1').replace('__DEPTH__', '')
    cells[30]['source'] = (src_of(cells[30]) + '\n' + bench + BENCH_TAIL).splitlines(True)
    # cell 32, engine 2: class-3 drop / class-2 re-add / ranker rows without np (module strings), leak-free FPNets
    s = src_of(cells[32]).splitlines(True)
    assert s[3].startswith('for _n, _src in ENG_FILES.items():')
    cells[32]['source'] = s[:3] + ENG_PATCH.splitlines(True) + s[3:]
    patch(cells, 32, B.FP_OLD, B.FP_NEW)
    # cell 34, PubChem channel: class-3 answers count as pool structures; the gate never sees them
    patch(cells, 34, B.V17_OLD, V17_NEW)
    patch(cells, 34, B.GATE_OLD, B.GATE_NEW)
    patch(cells, 34, B.BEST_OLD, B.BEST_NEW)
    # cell 36: external reference spectra of class-2/3 answers hidden; cell 37: FPNet A, class-3 window hide
    patch(cells, 36, EXT_OLD, EXT_NEW)
    patch(cells, 37, B.FULL_OLD, B.FULL_NEW)
    patch(cells, 37, WIN_OLD, WIN_NEW)
    return json.dumps(nb, separators=(',', ':')).encode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/user/work/push_c477')
    ap.add_argument('--out', default='/home/user/work/push_bench477')
    a = ap.parse_args()
    md = json.load(open(os.path.join(a.src, 'kernel-metadata.json')))
    nb = json.load(open(os.path.join(a.src, md['code_file'])))
    cells = nb['cells']
    assert len(cells) == 23 and src_of(cells[9]).startswith('# ── Pinned sources')
    s9 = src_of(cells[9])
    old_sha = re.search(r"^FROZEN_SHA256 = '([0-9a-f]{64})'$", s9, re.M).group(1)
    old_b64 = re.search(r"^FROZEN_B64 = '([A-Za-z0-9+/=]+)'$", s9, re.M).group(1)
    raw = base64.b64decode(old_b64)
    assert hashlib.sha256(raw).hexdigest() == old_sha
    new_raw = build_frozen(raw)
    new_sha = hashlib.sha256(new_raw).hexdigest()
    new_b64 = base64.b64encode(new_raw).decode()
    patch(cells, 9, f"FROZEN_SHA256 = '{old_sha}'", f"FROZEN_SHA256 = '{new_sha}'")
    patch(cells, 9, f"FROZEN_B64 = '{old_b64}'", f"FROZEN_B64 = '{new_b64}'")
    patch(cells, 12, RUN_OLD, RUN_NEW)
    assert 'PARALLAX_DIAGNOSTICS = apply(' in src_of(cells[13]) and 'pxm_merge' in src_of(cells[16])
    title = {'cell_type': 'markdown', 'id': 'bench477-title', 'metadata': {}, 'source': [
        '# CASMI26 bench477: the 0.477 pipeline on hidden-answer natural products\n', '\n',
        'Never submitted. Runs our exact copy of busyaprime\'s 0.477 on the 250 enveda-np-examples natural products '
        'with the answers hidden per simulated class (1/2/3 = 28/20/52%), and scores every variant list. Built by '
        'enveda-casmi26/bench/build_bench477.py; all credits as in the cells below.\n']}
    nb['cells'] = ([title] + cells[:14] + [code(SAVE_PHYSICAL, 'bench477-physical')] + cells[14:17]
                   + [code(SAVE_MERGED, 'bench477-merged')] + cells[17:] + [code(SCORE_CELL, 'bench477-score')])
    os.makedirs(a.out, exist_ok=True)
    for old in glob.glob(os.path.join(a.out, '*.ipynb')):
        os.remove(old)
    code_file = f'{SLUG}.ipynb'
    with open(os.path.join(a.out, code_file), 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')
    md.update(id=f'aryanawasthiteykik/{SLUG}', title='CASMI26 Bench477', code_file=code_file, is_private=True)
    md.pop('id_no', None)
    json.dump(md, open(os.path.join(a.out, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', a.out, len(nb['cells']), 'cells | kernel', md['id'], '| frozen sha', old_sha[:12], '->', new_sha[:12])


if __name__ == '__main__':
    main()
