"""Assemble notebooks/notebook_pubchem.ipynb: PubChem candidates + fingerprint model + popularity prior."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# CASMI26: PubChem candidates + fingerprint model + popularity prior

Part 3 of my series ([part 1: validation](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test), [part 2: fingerprint model](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-finding-molecules-not-in-any-library)).

Part 2 scored 0.176 because most hidden answers aren't in COCONUT ∪ train. This version adds **PubChem** candidates (~106M structures, about 7,000 per molecule within 10 ppm), which contain about 97% of the simulated class-2 answers. With that many candidates the fingerprint model alone isn't enough, so a prior is added:

**score = z(fingerprint log-likelihood) + 0.25 × popularity + 1 × natural-product flag**, with confident cosine library hits (≥ 0.7) first.

- *popularity* = log(1 + PubChem substances) + log(1 + PubMed articles), from [casmi26-pubchem-popularity-prior](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior)
- *natural product* = listed in COCONUT, LOTUS or NPAtlas

**A warning about the validation numbers.** On my validation split the prior lifts class 2/3 from 0.11 to 0.61. That is surely too optimistic: the validation molecules come from public spectral libraries, so they are famous, and popularity favours famous molecules. The weights here are deliberately modest, and the leaderboard is the real test."""),
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
md("## 1. Candidate pool fingerprints (training structures + COCONUT)"),
code("""import pickle
from multiprocessing import Pool
import numpy as np, pandas as pd
from casmi import formula_mass
from fingerprints import fp_packed

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
    fps = np.stack(pool.map(fp_packed, structs.smiles, chunksize=2000))
structs.to_parquet('/kaggle/working/structures.parquet'); np.save('/kaggle/working/fps.npy', fps)
print(len(structs), fps.shape)
del structs, fps, tr, coco, cm"""),
md("## 2. Rank and submit\n`pipeline.run` builds the cosine library, fingerprints every PubChem candidate in each molecule's 10 ppm window, scores and de-duplicates by InChIKey14 (the metric's identity)."),
code("""from pipeline import run
sub = run(DATA / 'train.parquet', DATA / 'test.parquet', '/kaggle/working/structures.parquet',
          '/kaggle/working/fps.npy', CODE / 'fpnet_full.pt', PUBCHEM, POP, w_pop=0.25, w_np=1.0, gate=0.7)
sub.to_csv('submission.csv', index=False)
print(sub.shape)
sub.head()"""),
md("""## Notes
- Runtime: about 15–20 minutes on CPU, almost all of it RDKit fingerprints for ~3M PubChem candidates.
- Popularity priors can look far better on validation than on the leaderboard. Compare the LB of this notebook with part 2 (0.176) before trusting larger weights.

*Feedback is welcome in the comments.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
out = HERE / 'notebooks' / 'notebook_pubchem.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
