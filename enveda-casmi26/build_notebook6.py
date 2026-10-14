"""Assemble notebooks/notebook_v3_ranker.ipynb: GPU fingerprint model v3 + fragmentation & analog ranker (public)."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SERIES = ('https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test',
          'https://www.kaggle.com/code/aryanawasthiteykik/casmi26-finding-molecules-not-in-any-library',
          'https://www.kaggle.com/code/aryanawasthiteykik/casmi26-the-popularity-prior-trap',
          'https://www.kaggle.com/code/aryanawasthiteykik/casmi26-fingerprint-model-v2-morgan-maccs')
TRAIN_NB = 'https://www.kaggle.com/code/aryanawasthiteykik/casmi26-gpu-training-fingerprint-v3-ranker'
DATASET = 'https://www.kaggle.com/datasets/aryanawasthiteykik/casmi26-spectrum-to-fingerprint-mlp'

cells = [
md(f"""# CASMI26: a GPU-trained fingerprint model (top-1 21% → 39%) + a fragmentation & analog ranker

Part 6 of my series ([1: validation]({SERIES[0]}) · [2: fingerprint model]({SERIES[1]}) · [4: the popularity-prior trap]({SERIES[2]}) · [5: fingerprint model v2]({SERIES[3]})). Everything here is my own code; candidate data comes from public datasets credited at the end.

**Public LB: 0.272** (from 0.222 three days ago), with one spectrum → fingerprint MLP, one LightGBM ranker and no spectrum simulators.

What changed:

1. **Fingerprint model v3, trained on a free T4** in a [separate training notebook]({TRAIN_NB}) (2.8 h end to end), so this notebook only runs inference. It is bigger (3 × 4,096 hidden), sees more spectra (up to 64 per structure, 1.8M) and trains longer (25 epochs, mixed precision, featurisation on the GPU).
2. **In-silico fragmentation.** Every 1- and 2-bond cut of a candidate gives fragment masses. How much of the measured MS/MS intensity they explain becomes a ranking feature.
3. **Analog propagation.** The query is searched against ~980k library spectra at any precursor mass, so close analogs with a different side chain also match. Candidates that are structurally similar (Tanimoto) to the best hits score higher.
4. **A LambdaRank ranker** over fit, popularity, fragmentation and analog features, one row per InChIKey14 (the metric's identity)."""),
md("""## Results

**Validation.** The splits come from [part 1]({0}): every answer is removed from the candidate pool except through PubChem, so each molecule has ~5,000-10,000 candidates.
- **S1:** 250 timsTOF natural products.
- **S2:** 490 `enveda-180` structures **with no public spectra**, so popularity can't help.

Both models were trained without these 740 structures.

| Fingerprint likelihood alone | v2 (CPU) | **v3 (GPU)** |
|---|---|---|
| S1 MRR@25 | 0.262 | **0.439** |
| S2 MRR@25 | 0.318 | **0.514** |
| S2 answer at rank 1 | 21% | **39%** |
| S2 answer in top 25 | 64% | **80%** |

| Ranker (5-fold CV by molecule), v3 features | S1 | S2 |
|---|---|---|
| fit + popularity | 0.818 | 0.502 |
| + fragmentation | 0.847 | 0.627 |
| **+ analogs** | **0.854** | **0.660** |

**Public leaderboard**, same candidates throughout:

| Run | LB |
|---|---|
| v2 + hand-tuned prior | 0.222 |
| + fragmentation | 0.245 |
| ranker + fragmentation + analogs (v2) | 0.256 |
| **ranker + fragmentation + analogs (v3)** | **0.272** |

Two lessons that cost me submissions:
- **Score a ranker with the same model it was trained on.** I trained the ranker on the held-out model's features but ran it with the full-data model. The likelihood scale shifts, the gap features shift with it, and the LB dropped (0.216 → 0.199).
- **The LB moves far less than validation**: S2 went up +0.14, the LB only +0.016. The rest of the gap is outside ranking: whether the answer is in the candidate pool at all, and how molecules with library spectra are handled.""".format(SERIES[0])),
code("""import glob, json, subprocess, sys, time
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

CODE = find('pipeline.py').parent            # my code + v2 weights
V3 = find('fpnet_v3_config.json').parent       # output of the GPU training notebook
sys.path.insert(0, str(CODE))
DATA = find('train.parquet').parent
COCO = find('coco_meta.pkl').parent
PUBCHEM = find('pc_smiles.npy').parent
POP = find('pc_lsid.npy').parent
CFG = json.load(open(V3 / 'fpnet_v3_config.json'))
print(CFG)"""),
md("""### v2 vs v3, measured in the training run
`compare_models.csv` is written by the training notebook: one row per validation molecule and model, holding the reciprocal rank of the answer."""),
code("""import numpy as np, pandas as pd
import matplotlib.pyplot as plt

res = pd.read_csv(V3 / 'compare_models.csv')
mrr = res.groupby(['split', 'model']).rr.mean().unstack()
splits = ['S1 np-examples', 'S2 enveda-180 holdout']
names = {'S1 np-examples': 'S1: library natural products', 'S2 enveda-180 holdout': 'S2: no public spectra'}

INK, INK2, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
COLORS = {'fpnet_v2h': '#2a78d6', 'fpnet_v3h': '#eb6834'}
LABELS = {'fpnet_v2h': 'v2 (CPU, 2 x 2,048)', 'fpnet_v3h': 'v3 (GPU, 3 x 4,096)'}
fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
x, w = np.arange(len(splits)), 0.34
for k, (model, off) in enumerate([('fpnet_v2h', -w / 2), ('fpnet_v3h', w / 2)]):
    vals = [mrr.loc[s, model] for s in splits]
    ax.bar(x + off, vals, w - 0.02, color=COLORS[model], label=LABELS[model], zorder=3)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10, color=INK2)
ax.set_xticks(x, [names[s] for s in splits], color=INK2)
ax.set_ylabel('MRR@25, fingerprint likelihood alone', color=INK2)
ax.set_ylim(0, max(0.6, mrr.values.max() * 1.15))
ax.set_title('Answers reachable only through PubChem (~5k-10k candidates each)', loc='left', color=INK, fontsize=12)
ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
for s in ('top', 'right'):
    ax.spines[s].set_visible(False)
for s in ('left', 'bottom'):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=INK2)
ax.legend(frameon=False, labelcolor=INK2)
plt.tight_layout(); plt.show()

table = res.groupby(['split', 'model']).agg(molecules=('rr', 'size'), MRR=('rr', 'mean'), top1=('top1', 'mean'),
                                            top25=('top25', 'mean')).round(3)
table"""),
md("""## 1. Candidate pool
Training structures ∪ COCONUT natural products, with Morgan 2048 + MACCS fingerprints (the v3 output layout). PubChem candidates (~106M structures, 10 ppm windows) are added per molecule inside `pipeline.run`."""),
code("""import pickle
from multiprocessing import Pool
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
md("""## 2. Rank and submit
For each test molecule, `pipeline.run`:
1. **Cosine gate.** Builds a cosine library from `train.parquet`. Library hits with cosine ≥ 0.7 go first.
2. **Fingerprint fit.** Predicts fingerprint probabilities with v3 and scores every candidate by log-likelihood.
3. **Features.** Fragments the 40 best-fitting and 10 most popular candidates against the measured peaks, and runs the analog search.
4. **Ranking.** Orders candidates with the LambdaRank booster.

The held-out v3 model (`fpnet_v3h`) is used deliberately: the ranker was trained on its features, so the feature scales match."""),
code("""from pipeline import run
t = time.time()
sub = run(DATA / 'train.parquet', DATA / 'test.parquet', '/kaggle/working/structures.parquet',
          '/kaggle/working/fps.npy', V3 / 'fpnet_v3h.pt', PUBCHEM, POP, gate=0.7,
          hidden=CFG['hidden'], depth=CFG['depth'], ranker_path=V3 / 'learned_ranker_v3.txt')
sub.to_csv('submission.csv', index=False)
print(sub.shape, f'{time.time() - t:.0f}s')
sub.head()"""),
md(f"""## Notes
- **Runtime:** about 1.5 h on CPU. Most of it is RDKit fingerprints and fragmentation for the PubChem candidates. No GPU is needed for inference.
- **Training:** the [GPU training notebook]({TRAIN_NB}) trains v3 twice (held-out and full), compares it against v2, rebuilds the fragmentation and analog features with v3 and retrains the ranker. Its featurisation runs on the GPU and is checked bin for bin against the CPU version used here.
- **Code:** all in [casmi26-spectrum-to-fingerprint-mlp]({DATASET}). See `fpmodel.py`, `train_gpu.py`, `fragments.py`, `analogs.py`, `learned_ranker.py` and `pipeline.py`.
- **Data credits:** COCONUT candidates by [prvsiyan](https://www.kaggle.com/datasets/prvsiyan/coconut-casmi26-candidates), the PubChem tier by [Ahmed Berat Özer](https://www.kaggle.com/datasets/ahmedberatozer/casmi26-pubchem-tier), the PubChem popularity prior by [Dmitrii Gluzdov](https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior), and the competition's RDKit wheel.
- **Combining with the top public notebooks:** merging this engine's lists into the public 0.437 pipeline by reciprocal-rank fusion gave 0.438. A second, independent engine helps a little even when it is much weaker on its own.

*If this was useful, an upvote helps others find it. Questions and ideas are welcome in the comments.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
out = HERE / 'notebooks' / 'notebook_v3_ranker.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
