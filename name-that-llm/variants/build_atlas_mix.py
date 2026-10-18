"""Mix 10% ATLAS into an NTL notebook's posterior before Sinkhorn balancing (one change), as the public 0.558 does.

josephadamski91's "Apex v4 + 10% ATLAS" adds P = 0.9 P + 0.1 P_atlas (frozen Qwen3-8B-Base with 18 source adapters,
trained on train.csv only) before balancing, and it moved two different bases by +0.002 each (0.549 -> 0.551,
0.556 -> 0.558). The block is copied verbatim from that notebook; the adapter dataset josephadamski91/ntl-atlas-qwen3-8b-v1
is attached. Its licence is 'other' with no terms given, so use the result for rank only, like TabPFN.

usage: python build_atlas_mix.py <base push dir> <out dir> <kernel slug> <title>
"""
import glob
import json
import os
import sys

base, out, slug, title = sys.argv[1:5]
nb_path = glob.glob(os.path.join(base, '*.ipynb'))[0]
nb = json.load(open(nb_path))
code = [c for c in nb['cells'] if c['cell_type'] == 'code']
assert len(code) == 1
src = ''.join(code[0]['source'])

anchor = """P = np.exp(LOGIT - LOGIT.max(1, keepdims=True))
P /= P.sum(1, keepdims=True)
"""
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + """# Independent ATLAS candidate: 90% blend / 10% ATLAS (josephadamski91, Apex v4 + 10% ATLAS; +0.002 on two bases).
import sys as _sys
_atlas_roots = list(pathlib.Path('/kaggle/input').rglob('atlas_bundle.json'))
assert len(_atlas_roots) == 1, 'Attach the versioned ATLAS adapter bundle'
_atlas_root = _atlas_roots[0].parent
_sys.path.insert(0, str(_atlas_root))
from ntl_atlas.inference import predict as _atlas_predict
from ntl_rlcd.data import LABELS as _atlas_labels
assert LABELS == _atlas_labels
_ap = _atlas_predict(TE, find_model_dir('qwen3', 4096), _atlas_root, batch_size=1)
assert _ap.shape == P.shape == (len(test), K), 'ATLAS row/class count mismatch'
assert np.isfinite(_ap).all() and (_ap >= 0).all() and (_ap <= 1).all()
assert np.allclose(_ap.sum(1), 1.0, atol=1e-6), 'ATLAS probabilities must sum to one'
P = 0.90 * P + 0.10 * _ap
P /= P.sum(1, keepdims=True)
""")
compile(src, 'nb', 'exec')
code[0]['source'] = src.splitlines(True)
nb['cells'].insert(0, {'cell_type': 'markdown', 'metadata': {}, 'source': [
    "**+ 10% ATLAS before balancing.** One change to this notebook: the ATLAS block of "
    "[josephadamski91's Apex v4 + 10% ATLAS](https://www.kaggle.com/code/josephadamski91/apex-v4-10-atlas) "
    "(adapter dataset josephadamski91/ntl-atlas-qwen3-8b-v1), copied verbatim. Credit to its author.\n"]})
os.makedirs(out, exist_ok=True)
json.dump(nb, open(os.path.join(out, os.path.basename(nb_path)), 'w'), indent=1)
m = json.load(open(os.path.join(base, 'kernel-metadata.json')))
m.update(id=f'aryanawasthiteykik/{slug}', title=title)
m['dataset_sources'] = m['dataset_sources'] + ['josephadamski91/ntl-atlas-qwen3-8b-v1']
json.dump(m, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', out)
