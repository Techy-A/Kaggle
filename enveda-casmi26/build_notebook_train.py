"""Assemble notebooks/notebook_gpu_training.ipynb: the GPU training run behind the submission notebook.

One run (T4, a few hours, not a submission) trains fingerprint model v3 twice (held-out and full), measures it
against v2 on the PubChem-only validation splits, rebuilds the fragmentation and analog features with it and
retrains the LambdaRank ranker. Everything the submission needs lands in /kaggle/working/out.
Set CASMI_SMOKE=1 to run a tiny version end to end (local check before spending GPU hours).
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# CASMI26: GPU training run (fingerprint model v3 + ranker)

Training happens here, once, so the submission notebook only runs inference inside its 9-hour limit.

| Stage | What | Roughly |
|---|---|---|
| 1 | Candidate pool (training structures + COCONUT) and Morgan + MACCS fingerprints | 15 min |
| 2 | Spectrum cache from `train.parquet` | 15 min |
| 3 | **v3 fingerprint model on GPU**: 3 x 4,096 hidden, up to 64 spectra per structure, 25 epochs, mixed precision. Trained twice: held-out (740 validation structures excluded) and full | 1-2 h |
| 4 | v2 vs v3 on the PubChem-only splits (fingerprint likelihood alone) | 45 min |
| 5 | Fragmentation + analog features with v3, LambdaRank ranker, 5-fold CV | 1 h |

Validation splits (answers reachable only through PubChem, ~10,000 candidates each): **S1** = 250 timsTOF natural products, **S2** = 490 `enveda-180` structures with no public spectra."""),
code("""import glob, os, shutil, subprocess, sys, time
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

CODE = find('train_gpu.py').parent
sys.path.insert(0, str(CODE))
DATA = find('train.parquet').parent
COCO = find('coco_meta.pkl').parent
PUBCHEM = find('pc_smiles.npy').parent
POP = find('pc_lsid.npy').parent
WORK, OUT = Path('/kaggle/working/work'), Path('/kaggle/working/out')
WORK.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get('CASMI_SMOKE') == '1'
if SMOKE:
    os.environ['CASMI_LIMIT'] = '40'                      # validation molecules per stage in the smoke run
for f in ['holdout_e180.txt', 'fpnet_v2h.pt']:
    shutil.copy(CODE / f, WORK / f)
T0 = time.time()
import torch
print(CODE, DATA, COCO, PUBCHEM, POP, 'cuda' if torch.cuda.is_available() else 'cpu', 'SMOKE' if SMOKE else '', sep='\\n')"""),
md("## 1. Candidate pool and fingerprints"),
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
with Pool() as pool:
    fps = np.stack(pool.map(fp_packed_v2, structs.smiles, chunksize=2000))
structs.to_parquet(WORK / 'structures.parquet'); np.save(WORK / 'fps_v2.npy', fps)
print(len(structs), fps.shape, f'{time.time() - T0:.0f}s')
del structs, fps, tr, coco, cm"""),
md("## 2. Spectrum cache"),
code("""from fpmodel import prep
prep(DATA / 'train.parquet', WORK)
print(f'{time.time() - T0:.0f}s')"""),
md("""## 3. Fingerprint model v3 on GPU
Featurisation runs on the GPU too; `selftest` checks it matches the CPU featurisation the submission uses, bin for bin."""),
code("""import train_gpu
train_gpu.selftest(WORK)
CFG = dict(hidden=4096, depth=3, epochs=25, max_per_struct=64)
if SMOKE:
    CFG = dict(hidden=64, depth=3, epochs=1, max_per_struct=1)
train_gpu.train(WORK, 'fpnet_v3h', exclude_validation=True, holdout_file=WORK / 'holdout_e180.txt',
                time_budget=2.5 * 3600, **CFG)
train_gpu.train(WORK, 'fpnet_v3_full', time_budget=2.5 * 3600, **CFG)
torch.cuda.empty_cache()
print(f'{time.time() - T0:.0f}s')"""),
md("""## 4. v2 vs v3 on the PubChem-only splits
Fingerprint likelihood alone: no cosine, no prior. Both models were trained without these 740 structures."""),
code("""import compare_models
V3 = f"fpnet_v3h:277:{CFG['hidden']}:{CFG['depth']}"
compare_models.main(str(DATA / 'train.parquet'), str(WORK), str(PUBCHEM), str(POP), 'fpnet_v2h:277:2048', V3)
print(f'{time.time() - T0:.0f}s')"""),
md("""## 5. Fragmentation + analog features and the ranker
Features for the strongest 50 candidates per molecule, chosen by v3's fit and popularity (never by label); analogs come from a library with the 740 answers' own spectra removed."""),
code("""import analogs, frag_features, learned_ranker
frag_features.main(str(DATA / 'train.parquet'), str(WORK), str(PUBCHEM), str(POP), model='fpnet_v3h')
print(f'fragmentation {time.time() - T0:.0f}s', flush=True)
analogs.main(str(DATA / 'train.parquet'), str(WORK), str(PUBCHEM), str(POP), fps_file='fps_v2.npy')
print(f'analogs {time.time() - T0:.0f}s', flush=True)
learned_ranker.main(str(WORK), 'fpnet_v3h')
print(f'{time.time() - T0:.0f}s')"""),
md("## 6. Outputs"),
code("""import json
for f in ['fpnet_v3h.pt', 'fpnet_v3_full.pt', 'learned_ranker_frag_an.txt', 'compare_models.csv',
          'frag_features.parquet', 'analog_features.parquet']:
    shutil.copy(WORK / f, OUT / f.replace('learned_ranker_frag_an', 'learned_ranker_v3'))
json.dump(dict(CFG, n_out=2216, fps='fp_packed_v2'), open(OUT / 'fpnet_v3_config.json', 'w'))
shutil.rmtree(WORK)                                      # keep the output small: only /kaggle/working/out stays
print(sorted(os.listdir(OUT)), f'total {time.time() - T0:.0f}s')"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
out = HERE / 'notebooks' / 'notebook_gpu_training.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
