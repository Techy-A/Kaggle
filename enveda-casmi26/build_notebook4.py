"""Assemble notebooks/notebook_popularity.ipynb: why popularity priors look perfect on validation but not on the LB."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# CASMI26: the popularity-prior trap

Adding a PubChem popularity prior (substance + PubMed counts) to my pipeline lifted my validation MRR on class 2/3 from **0.11 to 0.61**. On the public leaderboard it changed **nothing: 0.176 before, 0.176 after.**

This notebook shows why, with no model at all. It just asks: *among all PubChem structures within 10 ppm of the answer's mass, how popular is the answer?*

| Molecule set | Popularity alone ranks the answer #1 | …in the top 25 |
|---|---|---|
| `enveda-np-examples` (the usual validation set) | **~67%** | **~99%** |
| public-library molecules (GNPS, MassBank, MoNA, RIKEN) | ~18% | ~53% |
| `enveda-180` structures with **no public spectra** | **~0%** | **~0.7%** |

**Molecules with public reference spectra are famous, so popularity finds them. Class-2 test molecules by definition have no public spectra**, and they look much more like the last row. Any validation built from spectral-library molecules will overrate a popularity prior.

Runs in a few minutes on CPU, with no RDKit and no model. Earlier parts: [1: validation](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-validation-that-behaves-like-the-test), [2: fingerprint model](https://www.kaggle.com/code/aryanawasthiteykik/casmi26-finding-molecules-not-in-any-library)."""),
code("""import glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def find(pattern):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    return Path(hits[0]) if hits else None

DATA, PUBCHEM, POP = find('train.parquet').parent, find('pc_mass.npy').parent, find('pc_lsid.npy').parent
mass = np.load(PUBCHEM / 'pc_mass.npy', mmap_mode='r')
ik14 = np.load(POP / 'pc_ik14.npy', mmap_mode='r')
lsid = np.load(POP / 'pc_lsid.npy', mmap_mode='r')
lpmid = np.load(POP / 'pc_lpmid.npy', mmap_mode='r')
print(f'{len(mass):,} PubChem structures')"""),
md("""## Molecule sets
- **validation:** the 250 `enveda-np-examples` natural products (timsTOF, the test instrument). All of them also have spectra in public libraries.
- **public libraries:** 1,000 random structures from GNPS, MassBank, MoNA and RIKEN.
- **no public spectra:** 1,000 random `enveda-180` structures that appear in no public library. They're synthetic, not natural products, but they share the defining property of class 2: nobody has published their spectra."""),
code("""MONO = {'C': 12.0, 'H': 1.0078250319, 'N': 14.0030740052, 'O': 15.9949146221, 'P': 30.97376151, 'S': 31.97207069,
        'F': 18.99840320, 'Cl': 34.96885271, 'Br': 78.9183376, 'I': 126.904468, 'Si': 27.9769265327, 'B': 11.0093055,
        'Se': 79.9165218}
import re
def formula_mass(f):
    if not isinstance(f, str) or not re.fullmatch(r'([A-Z][a-z]?\\d*)+', f):
        return np.nan
    parts = re.findall(r'([A-Z][a-z]?)(\\d*)', f)
    if any(el not in MONO for el, _ in parts):
        return np.nan
    return sum(MONO[el] * (int(n) if n else 1) for el, n in parts)

meta = pd.read_parquet(DATA / 'train.parquet', columns=['inchikey14', 'ingest_lib', 'molecular_formula'])
public = set(meta.loc[~meta.ingest_lib.str.startswith('enveda'), 'inchikey14'])
sets = {
    'validation (np-examples)': meta[meta.ingest_lib == 'enveda-np-examples'].drop_duplicates('inchikey14'),
    'public libraries': meta[meta.ingest_lib.isin(['gnps', 'massbank', 'mona', 'riken'])]
                        .drop_duplicates('inchikey14').sample(1000, random_state=0),
    'no public spectra (enveda-180)': meta[(meta.ingest_lib == 'enveda-180') & ~meta.inchikey14.isin(public)]
                                      .drop_duplicates('inchikey14').sample(1000, random_state=0),
}
del meta
{k: len(v) for k, v in sets.items()}"""),
md("""## Where does the answer rank by popularity alone?
For each molecule: take every PubChem structure within 10 ppm of its mass, merge rows by InChIKey14 (the metric's identity), and rank by popularity = log(1 + substances) + log(1 + PubMed articles)."""),
code("""rows = []
for name, df in sets.items():
    for key, formula in zip(df.inchikey14, df.molecular_formula):
        m = formula_mass(formula)
        if not np.isfinite(m):
            continue
        lo, hi = np.searchsorted(mass, [m * (1 - 1e-5), m * (1 + 1e-5)])
        ids = np.asarray(ik14[lo:hi]).astype(str)
        if key not in set(ids):
            rows.append(dict(set=name, in_pubchem=False))
            continue
        pop = pd.Series(np.asarray(lsid[lo:hi], np.float32) + np.asarray(lpmid[lo:hi], np.float32)).groupby(ids).max()
        rank = int((pop > pop[key]).sum()) + 1
        rows.append(dict(set=name, in_pubchem=True, candidates=len(pop), rank=rank, rr=1 / rank if rank <= 25 else 0,
                         percentile=(pop < pop[key]).mean(), popularity=pop[key]))
res = pd.DataFrame(rows)
hit = res[res.in_pubchem]
summary = pd.DataFrame({
    'in PubChem': res.groupby('set').in_pubchem.mean(),
    'median candidates': hit.groupby('set').candidates.median(),
    'answer #1 by popularity': hit.groupby('set')['rank'].apply(lambda r: (r == 1).mean()),
    'answer in top 25': hit.groupby('set')['rank'].apply(lambda r: (r <= 25).mean()),
    'popularity-only MRR@25': hit.groupby('set').rr.mean(),
}).round(3)
summary"""),
code("""fig, ax = plt.subplots(figsize=(8, 3.6))
for name, g in hit.groupby('set'):
    ax.hist(g.percentile, bins=np.linspace(0, 1, 41), histtype='step', lw=2, density=True, label=name)
ax.set_xlabel("answer's popularity percentile among its PubChem candidates"); ax.set_yticks([])
ax.legend(frameon=False, loc='upper left'); plt.tight_layout(); plt.show()"""),
md("""## What this means
1. **Spectral-library molecules sit at the very top of their windows.** Popularity alone almost solves them, which is why any validation built on them (holding out library molecules, `enveda-np-examples`, MassSpecGym-style splits) rewards a strong prior.
2. **Molecules without public spectra are ordinary PubChem entries.** For them popularity is close to noise, and a large prior weight pushes the answer *down* behind famous isomers.
**For your own CV:** score any prior on a split whose answers have *no* public spectra. One option is to report the gain on both splits and treat the true effect as lying somewhere in between, much closer to the second if the test is mostly class 2/3. My leaderboard (0.176 with and without the prior, at weight 0.25 on a z-scored spectral score) agrees with the pessimistic side.

*If this saved you some submissions, an upvote helps others find it.*"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                                'language': 'python'}})
out = HERE / 'notebooks' / 'notebook_popularity.ipynb'
nbf.write(nb, out)
print('wrote', out, len(cells), 'cells')
