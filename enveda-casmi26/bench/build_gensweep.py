"""Build a CPU-only notebook that sweeps engine 1's class-3 generator settings on the bench pseudo-test.

Runs V22's setup cells, the bench cell (np pseudo-test, answers hidden per class), engine 1 (cells 18-19, patched like
the bench) and then re-runs cell 21 (base lists) for the class-3 molecules plus 40 class-1/2 controls under each
generator setting in GRID. Per setting it prints GENSWEEP_JSON: per-class MRR / top-1 / hit@25 / hit@60 of engine 1's
own list and the mean number of generated candidates. No GPU (the GPU quota is untouched); never submitted.

usage: python build_gensweep.py <v22_dir> <out_dir>
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_bench as bb  # noqa: E402

GRID = {
    'base': dict(gen_n_analog=6, gen_min_sim=0.3, gen_max_per_parent=60, gen_max_total=150),
    'A12': dict(gen_n_analog=12, gen_min_sim=0.3, gen_max_per_parent=80, gen_max_total=300),
    'B20': dict(gen_n_analog=20, gen_min_sim=0.3, gen_max_per_parent=100, gen_max_total=500),
    'C30': dict(gen_n_analog=30, gen_min_sim=0.2, gen_max_per_parent=120, gen_max_total=800),
}

SWEEP = r'''"""GEN SWEEP: engine-1 generator settings on the bench pseudo-test (CPU)."""
import json
_AKEY = dict(zip(ANS.molecule_id, ANS.metric_key)); _ACLS = dict(zip(ANS.molecule_id, ANS.cls.astype(int)))
_sel = set(ANS[ANS.cls == 3].molecule_id) | set(sorted(ANS[ANS.cls < 3].molecule_id)[:40])
_SRC21 = __SRC21__
_old = "mols = list(te.groupby('molecule_id', sort=False))"
assert _SRC21.count(_old) == 1
_SRC21 = _SRC21.replace(_old, "mols = [g for g in te.groupby('molecule_id', sort=False) if g[0] in _sel]")
GRID = __GRID__

def _gscore(B):
    per = {}
    for mid, (smis, keys, lib_max, scs, forms) in B.items():
        k = _AKEY.get(mid)
        r = next((i + 1 for i, x in enumerate(keys) if x == k), 0)
        per[mid] = (r, sum(1 for f in forms if str(f).endswith('|gen')), float(lib_max))
    out = {}
    for c in (1, 2, 3):
        ms = [m for m in per if _ACLS[m] == c]
        if not ms:
            continue
        out[f'c{c}'] = dict(n=len(ms), mrr=float(np.mean([1 / per[m][0] if per[m][0] else 0 for m in ms])),
                            top1=sum(per[m][0] == 1 for m in ms), hit25=sum(0 < per[m][0] <= 25 for m in ms),
                            hit60=sum(per[m][0] > 0 for m in ms), mean_gen=float(np.mean([per[m][1] for m in ms])))
    return out, per

for _name, _cv in GRID.items():
    for _k, _v in _cv.items():
        setattr(E.cfg, _k, _v)
    _t0 = time.time()
    exec(compile(_SRC21, f'<cell21 {_name}>', 'exec'), globals())
    _res, _per = _gscore(BASE)
    print('GENSWEEP_JSON ' + json.dumps({_name: dict(cfg=_cv, secs=round(time.time() - _t0), **_res)}), flush=True)
    json.dump({m: list(v) for m, v in _per.items()}, open(f'/kaggle/working/gensweep_{_name}.json', 'w'))
'''


def main():
    v22, out = sys.argv[1], sys.argv[2]
    nb = json.load(open(os.path.join(v22, 'notebook.ipynb')))
    md = json.load(open(os.path.join(v22, 'kernel-metadata.json')))
    cells = nb['cells']
    # CPU run: drop the T4 assertion in the preflight cell
    s = bb.src_of(cells[10])
    t4 = "    gpu = subprocess.check_output(\n        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()\n    assert 'T4' in gpu, gpu\n"
    assert s.count(t4) == 1
    cells[10]['source'] = s.replace(t4, "    gpu = 'cpu (gen sweep)'\n").splitlines(True)
    bb.patch(cells, 19, bb.FULL_OLD, bb.FULL_NEW)
    bb.patch(cells, 19, bb.WIN_OLD, bb.WIN_NEW)
    src21 = bb.src_of(cells[21])
    bench = bb.BENCH_CELL.replace('__LEAKFREE__', '1').replace('__DEPTH__', '')
    title = {'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# CASMI26 generator sweep (CPU, never submitted)\n', '\n',
        'Engine 1 of V22 on the bench pseudo-test (enveda-np-examples, answers hidden per class). Re-runs the base-list '
        'cell under several class-3 generator settings and reports how often the hidden answer is generated and where it '
        'ranks. Built by enveda-casmi26/bench/build_gensweep.py; credits as in the V22 cells below.\n']}
    sweep = SWEEP.replace('__SRC21__', repr(src21)).replace('__GRID__', repr(GRID))
    nb['cells'] = [title] + cells[:13] + [bb.code(bench)] + cells[17:20] + [bb.code(sweep)]
    os.makedirs(out, exist_ok=True)
    json.dump(nb, open(os.path.join(out, 'notebook.ipynb'), 'w'), indent=1)
    md.update(id='aryanawasthiteykik/casmi26-gen-sweep', title='CASMI26 gen sweep', is_private=True, enable_gpu=False)
    md.pop('id_no', None); md.pop('machine_shape', None)
    json.dump(md, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', out, len(nb['cells']), 'cells')


if __name__ == '__main__':
    main()
