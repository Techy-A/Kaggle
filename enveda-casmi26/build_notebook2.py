"""Assemble notebooks/notebook_fingerprint.ipynb (spectrum -> fingerprint ranking for classes 2/3)."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
MODEL_DATASET = 'casmi26-spectrum-to-fingerprint-mlp'

cells = [
md("""# CASMI26: finding molecules that aren't in any library

In [part 1](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test) I built a validation set that copies the hidden test's novelty classes. Plain cosine library search scored **0.94 on class 1 but 0.00 on classes 2/3**, and its public LB of 0.150 suggested only about 16% of the test is class 1. **This notebook goes after the other ~84%.**

**Approach**
1. **Candidates:** every structure in COCONUT 2.0 or the training set within 10 ppm of the molecule's mass. That's a median of 58 per molecule, and the right answer is in the pool for all 250 validation molecules.
2. **Spectrum → fingerprint MLP:** reads a binned spectrum (fragments, neutral losses, adduct, collision energy) and predicts the probability of each of 2,048 Morgan fingerprint bits.
3. **Ranking:** candidates are ordered by how likely their fingerprint is under that prediction, summed over all of a molecule's spectra.
4. **Gate:** if cosine search finds a confident library match (≥ 0.7), that match goes first.

**Results on the validation set** (the model never saw these 250 structures in training):

| Method | Class 1 | Class 2/3 | Expected at 16% class 1 |
|---|---|---|---|
| Cosine only (part 1) | 0.939 | 0.051 | 0.19 |
| Fingerprint only | 0.338 | 0.371 | 0.365 |
| **Gate: cosine ≥ 0.7 first, then fingerprint** | 0.785 | 0.337 | **0.409** |

Everything runs offline in about 15 minutes on CPU. The model weights are in the [`casmi26-spectrum-to-fingerprint-mlp`](https://www.kaggle.com/datasets/aryanawasthiteykik/casmi26-spectrum-to-fingerprint-mlp) dataset, with the code that trained them."""),
md("## Setup\nRDKit comes from the official offline wheel; the model code and weights come from the dataset."),
code(f"""import glob, subprocess, sys
from pathlib import Path

def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{{pattern}}', recursive=True), key=len)
    return Path(hits[0]) if hits else None

try:
    import rdkit
except ImportError:
    tag = f'cp{{sys.version_info.major}}{{sys.version_info.minor}}'
    wheel = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w][0]
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', wheel], check=True)

CODE = find('fpmodel.py').parent           # dataset: {MODEL_DATASET}
sys.path.insert(0, str(CODE))
DATA = find('train.parquet').parent
COCO = find('coco_meta.pkl').parent        # dataset: prvsiyan/coconut-casmi26-candidates
print(CODE, DATA, COCO, sep='\\n')"""),
code("""import pickle, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from casmi import (TEST_ADDUCTS, Library, make_validation, molecule_mass, mrr_at_25, rank_molecule, read_rows)
from fpmodel import SpectrumStore, fp_loglik, gated_order, make_model, predict_logits
from fingerprints import fp_packed
torch.set_num_threads(4)
TRAIN, TEST, PPM = DATA / 'train.parquet', DATA / 'test.parquet', 10"""),
md("""## 1. The candidate pool
The training structures plus COCONUT 2.0 natural products. The COCONUT bank ships its SMILES in a pickle, so it's loaded with an unpickler that only accepts numpy arrays. Fingerprints are 2,048-bit Morgan (radius 2), computed here with RDKit in about 2 minutes."""),
code("""class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy._core.multiarray', '_reconstruct'),
                              ('numpy.core.multiarray', '_reconstruct')}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {module}.{name}')

with open(COCO / 'coco_meta.pkl', 'rb') as f:
    cm = NumpyOnly(f).load()
coco = pd.DataFrame({'ik14': cm['keys'].astype(str), 'smiles': cm['smiles'].astype(str),
                     'mass': np.load(COCO / 'coco_mass.npy'), 'source': 'coconut'})

from casmi import formula_mass
tr = pd.read_parquet(TRAIN, columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
tr = pd.DataFrame({'ik14': tr.inchikey14, 'smiles': tr.normalized_smiles,
                   'mass': tr.molecular_formula.map(formula_mass), 'source': 'train'})
structs = (pd.concat([tr, coco], ignore_index=True).dropna(subset=['mass'])
           .drop_duplicates('ik14').sort_values('mass', ignore_index=True))

from multiprocessing import Pool
t = time.time()
with Pool(4) as pool:
    fps = np.stack(pool.map(fp_packed, structs.smiles, chunksize=2000))
masses, keys, smiles = structs.mass.to_numpy(), structs.ik14.to_numpy(), structs.smiles.to_numpy()
print(f'{len(structs):,} candidate structures', structs.source.value_counts().to_dict(), f'{time.time() - t:.0f}s')"""),
md("""## 2. The model
A three-layer MLP (1,536 hidden units, LayerNorm, GELU, dropout 0.2) maps a 2,514-dimensional spectrum vector to 2,048 fingerprint logits:

- **fragments:** 0.5 Da bins up to m/z 1000, sqrt intensity, top 60 peaks
- **neutral losses:** precursor m/z − fragment m/z, 1 Da bins up to 500 Da. Losses carry the functional-group information that separates isomers.
- **context:** adduct (one-hot), ion mode, collision energy, precursor m/z, peak count

It was trained with binary cross-entropy for 4 epochs on 1.3M training spectra (up to 8 per structure, 268k structures), about 27 minutes on 4 CPU cores. `fpnet_val.pt` excludes every validation structure; `fpnet_full.pt` sees everything and is used for the submission. The training code is `fpmodel.py` in the dataset.

**Ranking.** With independent bits, the log-likelihood of a candidate fingerprint *b* given predicted probabilities *p* is Σ [b·log p + (1−b)·log(1−p)], summed over the molecule's spectra. That's a single matrix product per molecule."""),
md("## 3. Validation: the same novelty-class split as part 1"),
code("""meta = pd.read_parquet(TRAIN, columns=['inchikey14', 'ingest_lib', 'adduct'])
_, novelty, _ = make_validation(meta)
banned = set(novelty.index[novelty == 2])
del meta

model = make_model(2048)
model.load_state_dict(torch.load(CODE / 'fpnet_val.pt', map_location='cpu'))
queries = read_rows(TRAIN, lambda df: (df.ingest_lib == 'enveda-np-examples') & df.adduct.isin(TEST_ADDUCTS))
logits = predict_logits(model, SpectrumStore.from_frame(queries), np.arange(len(queries)))

t = time.time()
val_lib = Library.build(TRAIN, keep=lambda df: (df.ingest_lib != 'enveda-np-examples')
                        & df.adduct.isin(TEST_ADDUCTS) & ~df.inchikey14.isin(banned))
print(f'cosine library: {time.time() - t:.0f}s')

rows = []
for key, spectra in queries.groupby('inchikey14'):
    m = molecule_mass(spectra)
    lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
    fp = fp_loglik(logits[spectra.index.to_numpy()], fps[lo:hi]) / len(spectra)
    _, cos = rank_molecule(val_lib, spectra, ppm=PPM)
    rows.append(dict(key=key, novelty=novelty[key], cands=keys[lo:hi].copy(), fp=fp,
                     cos=np.array([cos.get(k, 0.0) for k in keys[lo:hi]])))
del val_lib
print(f'{len(rows)} molecules, truth in candidate pool: {np.mean([r["key"] in set(r["cands"]) for r in rows]):.0%}, '
      f'median candidates: {np.median([len(r["cands"]) for r in rows]):.0f}')"""),
code("""def score(order_fn):
    rr = pd.DataFrame([dict(novelty=r['novelty'], rr=mrr_at_25(list(r['cands'][order_fn(r)]), r['key'])) for r in rows])
    by = rr.groupby('novelty').rr.mean()
    return by[1], by[2], 0.16 * by[1] + 0.84 * by[2]

table = {'cosine only': score(lambda r: np.argsort(-r['cos'], kind='stable')),
         'fingerprint only': score(lambda r: np.argsort(-r['fp'], kind='stable'))}
gates = [0.5, 0.6, 0.7, 0.8, 0.9]
for g in gates:
    table[f'gate {g}'] = score(lambda r, g=g: gated_order(r['fp'], r['cos'], g))
table = pd.DataFrame(table, index=['class 1', 'class 2/3', 'expected @16% class 1']).T.round(3)
table"""),
code("""fig, ax = plt.subplots(figsize=(7, 3.4))
g = table.loc[[f'gate {x}' for x in gates]]
ax.plot(gates, g['class 1'], 'o-', label='class 1')
ax.plot(gates, g['class 2/3'], 'o-', label='class 2/3')
ax.plot(gates, g['expected @16% class 1'], 'o-', lw=2.5, label='expected LB mix')
ax.set_xlabel('cosine gate'); ax.set_ylabel('MRR@25'); ax.legend(frameon=False); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()"""),
md("""The expected score is flat between gates of 0.6 and 0.8. I use **0.7**, the middle of that plateau, rather than the single best value on 250 molecules.

Two caveats:
- **Coverage.** In this simulation the answer is always in the pool. Hidden class-2 molecules are only guaranteed to be in *PubChem or COCONUT*, so some won't be in this pool, and the leaderboard will come out below the 0.41 estimate.
- **Isomers.** Most misses here lose to a candidate with the *same formula*, as the forum has found too. Better isomer separation (more fingerprint types, fragmentation scoring) is the next lever."""),
md("## 4. Submission\nThe full model, the full cosine library, and gate 0.7."),
code("""model.load_state_dict(torch.load(CODE / 'fpnet_full.pt', map_location='cpu'))
test = pd.read_parquet(TEST).reset_index(drop=True)
ok = test.adduct.isin(TEST_ADDUCTS).to_numpy()
test_logits = np.zeros((len(test), 2048), np.float32)
test_logits[ok] = predict_logits(model, SpectrumStore.from_frame(test[ok]), np.arange(ok.sum()))

t = time.time()
full_lib = Library.build(TRAIN, keep=lambda df: df.adduct.isin(TEST_ADDUCTS))
print(f'full library: {time.time() - t:.0f}s')

sub = []
for mol, spectra in test.groupby('molecule_id', sort=False):
    spectra = spectra[spectra.adduct.isin(TEST_ADDUCTS)]
    guesses = ['C']
    if len(spectra):
        m = molecule_mass(spectra)
        lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
        if hi > lo:
            fp = fp_loglik(test_logits[spectra.index.to_numpy()], fps[lo:hi]) / len(spectra)
            _, cos = rank_molecule(full_lib, spectra, ppm=PPM)
            cos = np.array([cos.get(k, 0.0) for k in keys[lo:hi]])
            guesses = list(smiles[lo:hi][gated_order(fp, cos, 0.7)[:25]])
    sub.append((mol, ';'.join(guesses)))
sub = pd.DataFrame(sub, columns=['molecule_id', 'smiles'])
sub.to_csv('submission.csv', index=False)
print(sub.shape, 'empty:', (sub.smiles == 'C').sum())
sub.head()"""),
md("""## Takeaways
1. **A small spectrum → fingerprint model lifts class 2/3 from 0.05 to 0.37** on the simulated split, with no external spectra and no GPU.
2. **Gating beats blending.** The two scores live on different scales, so let confident library hits go first and let the fingerprint model decide the rest.
3. **What's next:** more candidates (PubChem coverage), more fingerprint types, and a learned ranker for same-formula isomers.

*If this was useful, an upvote helps others find it. Feedback is welcome in the comments.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
out = HERE / 'notebooks' / 'notebook_fingerprint.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
