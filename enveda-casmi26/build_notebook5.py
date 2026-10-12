"""Assemble notebooks/notebook_fpnet_v2.ipynb: Morgan+MACCS fingerprint model v2 over PubChem candidates."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# CASMI26: a fingerprint model that doubles PubChem retrieval

Part 5 of my series ([1: validation](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test) · [2: fingerprint model](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-finding-molecules-not-in-any-library) · [4: the popularity-prior trap](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-the-popularity-prior-trap)).

Part 4 showed that popularity priors can't be validated on spectral-library molecules. So this part improves the **spectrum → fingerprint model itself** and measures it on a split popularity can't inflate: **490 timsTOF `enveda-180` structures that have no public spectra**, held out of training. Their answers are reachable only through PubChem (~10,600 candidates each, de-duplicated by InChIKey14).

**v2 changes:** predict MACCS keys (167 functional-group bits) as well as Morgan 2048; hidden 1,536 → 2,048; up to 16 spectra per structure instead of 8; 6 epochs instead of 4. Still CPU-trained, and still no external spectra.

Fingerprint likelihood alone (no cosine, no prior), answer reachable only through PubChem:

| Split | v1 MRR@25 | **v2 MRR@25** | v1 → v2 top-1 | v1 → v2 top-25 |
|---|---|---|---|---|
| 250 natural products (`enveda-np-examples`) | 0.132 | **0.262** | 8% → 17% | 35% → 53% |
| **490 structures with no public spectra** | 0.157 | **0.318** | 9% → 21% | 43% → 64% |

(Measured offline with `compare_models.py` in the model dataset; both models were trained without these 740 structures.)

This notebook ranks PubChem + COCONUT + training candidates with v2, puts confident cosine library hits (≥ 0.7) first, and adds a modest prior: **z(fingerprint) + 0.25 × popularity + 1 × natural-product flag**.

**Public LB: 0.221** (my previous best 0.176). The prior matters here: the same v2 run *without* it scored 0.146. With ~10,000 PubChem candidates per molecule, the fingerprint model alone lets too many obscure isomers through. Validation still overstates the prior hugely (see part 4), so keep its weight modest."""),
code("""import glob, subprocess, sys, time
from pathlib import Path

def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    return Path(hits[0]) if hits else None

try:
    import rdkit
except ImportError:
    tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
    wheel = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w][0]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', wheel], check=True)

CODE = find('pipeline.py').parent
sys.path.insert(0, str(CODE))
DATA = find('train.parquet').parent
COCO = find('coco_meta.pkl').parent
PUBCHEM = find('pc_smiles.npy').parent
POP = find('pc_lsid.npy').parent
print(CODE, DATA, COCO, PUBCHEM, POP, sep='\\n')"""),
md("## 1. Candidate pool fingerprints (training structures + COCONUT), Morgan 2048 + MACCS"),
code("""import pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
from casmi import formula_mass
from fingerprints import fp_packed_v2

class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy._core.multiarray', '_reconstruct'),
                              ('numpy.core.multiarray', '_reconstruct')}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')

with open(COCO / 'coco_meta.pkl', 'rb') as f:
    cm = NumpyOnly(f).load()
coco = pd.DataFrame({'ik14': cm['keys'].astype(str), 'smiles': cm['smiles'].astype(str),
                     'mass': np.load(COCO / 'coco_mass.npy'), 'source': 'coconut'})
tr = pd.read_parquet(DATA / 'train.parquet', columns=['inchikey14', 'normalized_smiles', 'molecular_formula'])
tr = tr.drop_duplicates('inchikey14')
tr = pd.DataFrame({'ik14': tr.inchikey14, 'smiles': tr.normalized_smiles,
                   'mass': tr.molecular_formula.map(formula_mass), 'source': 'train'})
structs = (pd.concat([tr, coco], ignore_index=True).dropna(subset=['mass'])
           .drop_duplicates('ik14').sort_values('mass', ignore_index=True))
with Pool(4) as pool:
    fps = np.stack(pool.map(fp_packed_v2, structs.smiles, chunksize=2000))
structs.to_parquet('/kaggle/working/structures.parquet'); np.save('/kaggle/working/fps.npy', fps)
print(len(structs), fps.shape)
del structs, fps, tr, coco, cm"""),
md("## 2. Rank and submit\n`pipeline.run` builds the cosine library, fingerprints every PubChem candidate in each molecule's 10 ppm window, scores and de-duplicates by InChIKey14 (the metric's identity)."),
code("""from pipeline import run
sub = run(DATA / 'train.parquet', DATA / 'test.parquet', '/kaggle/working/structures.parquet',
          '/kaggle/working/fps.npy', CODE / 'fpnet_v2_full.pt', PUBCHEM, POP, w_pop=0.25, w_np=1.0, gate=0.7,
          hidden=2048)
sub.to_csv('submission.csv', index=False)
print(sub.shape)
sub.head()"""),
md("""## Notes
- Runtime: about 25 minutes on CPU, mostly RDKit fingerprints (Morgan + MACCS) for the PubChem candidates.
- The weights (`fpnet_v2_full.pt`, plus `fpnet_v2h.pt` with the 740 validation structures held out) and all code are in [casmi26-spectrum-to-fingerprint-mlp](https://www.kaggle.com/datasets/aryanawasthiteykik/casmi26-spectrum-to-fingerprint-mlp).

*If this helped, an upvote helps others find it. Questions and ideas are welcome in the comments.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
out = HERE / 'notebooks' / 'notebook_fpnet_v2.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
