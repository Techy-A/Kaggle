"""Port one of our two v459 changes onto the public 0.456 notebook (v465 = v15 + pool popularity 0.15).

depth7: engine-1 ranker GBM max_depth 6 -> 7 (both embedded copies of CFG.GBM), as push_v459_depth7.
gen20:  when the best library match is weak (lib_max < 0.7), rerun engine 1 with 20-analog generation, as
        push_v459_gated_gen20 (bench: class-3 molecules gain candidates; strong matches are untouched).

usage: python port_v465.py <depth7|gen20> <v465 push dir> <out dir> <kernel slug> <title>
"""
import glob
import json
import os
import sys

change, base, out, slug, title = sys.argv[1:6]
nb_path = glob.glob(os.path.join(base, '*.ipynb'))[0]
nb = json.load(open(nb_path))
cells = [c for c in nb['cells'] if c['cell_type'] == 'code']


def replace(old, new, count=1):
    hits = [c for c in cells if old in ''.join(c['source'])]
    assert sum(''.join(c['source']).count(old) for c in hits) == count, (old, count)
    for c in hits:
        c['source'] = ''.join(c['source']).replace(old, new).splitlines(True)


if change == 'depth7':
    replace('GBM = dict(max_depth=6, max_iter=500', 'GBM = dict(max_depth=7, max_iter=500', count=2)
elif change == 'gen20':
    loop = 'for gi, (mid, sub) in enumerate(mols):\n    nms = sub.nm.values[np.isfinite(sub.nm.values)]\n'
    replace(loop, "GEN20_GATE, GEN20 = 0.7, dict(gen_n_analog=20, gen_max_per_parent=100, gen_max_total=500)   # gated gen20\n"
                  "GEN6 = {k: getattr(E.cfg, k) for k in GEN20}; n_g20 = 0\n" + loop)
    run = "            lib_max = float(info.get('lib_max', 0.0))\n"
    replace(run, run + """            if lib_max < GEN20_GATE:   # gated gen20: weak library match -> rerun engine 1 with 20-analog generation
                try:
                    for _k, _v in GEN20.items(): setattr(E.cfg, _k, _v)
                    C, X, F, names, info = V.run(spectra, target); n_g20 += 1
                except Exception as e:
                    print('gen20 failed, keeping the 6-analog list', mid, repr(e))
                finally:
                    for _k, _v in GEN6.items(): setattr(E.cfg, _k, _v)
""")
    replace("print('errors', n_err, '| popularity re-ranked lists', n_pp)",
            "print('errors', n_err, '| popularity re-ranked lists', n_pp, '| gated gen20 reruns', n_g20)")
else:
    sys.exit(f'unknown change {change}')
for c in cells:   # cells are plain python except for IPython magics; compile what compiles to catch broken edits
    src = ''.join(c['source'])
    if not any(l.lstrip().startswith(('!', '%')) for l in src.splitlines()):
        compile(src, 'cell', 'exec')
os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, os.path.basename(nb_path)), 'w'), indent=1)
m = json.load(open(os.path.join(base, 'kernel-metadata.json')))
m.update(id=f'aryanawasthiteykik/{slug}', title=title)
json.dump(m, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', out)
