"""Analog-seeded candidate generation for molecules that are in no database (class 3).

Novel natural products are usually close relatives of known ones: same scaffold, one or two different decorations.
The analog search (analogs.py) finds library structures whose spectra resemble the query. Here those structures are
mutated with a small set of biochemically common transformations (methylation, hydroxylation, glycosylation,
acylation, reduction, ...), one or two steps deep, and only products whose neutral mass matches the query's within
`ppm` are kept. The products are new candidates for the usual fingerprint / fragmentation ranking.

Usage (feasibility on the validation splits): python generate.py <train.parquet> <work_dir>
"""
import sys
import time

import numpy as np
import pandas as pd

# (name, reaction SMARTS); each applies at every matching site, one site per product
TRANSFORMS = [
    ('O-methylation', '[OX2H1:1][#6:2]>>[O:1](C)[#6:2]'),
    ('O-demethylation', '[CH3][OX2:1][#6:2]>>[OH1:1][#6:2]'),
    ('N-methylation', '[NX3;H1,H2;!$(NC=O):1]>>[N:1]C'),
    ('N-demethylation', '[CH3][NX3:1]>>[N:1]'),
    ('aromatic hydroxylation', '[cH1:1]>>[c:1]O'),
    ('aliphatic hydroxylation', '[CX4;H1,H2;!$(C[O,N]):1]>>[C:1]O'),
    ('dehydroxylation', '[#6:1][OX2H1]>>[#6:1]'),
    ('hexose', '[OX2H1:1][#6:2]>>[#6:2][O:1]C1OC(CO)C(O)C(O)C1O'),
    ('deglycosylation', '[#6:2][OX2:1][CH1]1O[CH1](CO)[CH1](O)[CH1](O)[CH1]1O>>[#6:2][OH1:1]'),
    ('deoxyhexose', '[OX2H1:1][#6:2]>>[#6:2][O:1]C1OC(C)C(O)C(O)C1O'),
    ('acetylation', '[OX2H1:1][#6:2]>>[#6:2][O:1]C(C)=O'),
    ('deacetylation', '[CH3]C(=O)[OX2:1][#6:2]>>[#6:2][OH1:1]'),
    ('reduction C=C', '[CX3:1]=[CX3:2]>>[C:1][C:2]'),
    ('desaturation C-C', '[CX4;H1,H2:1][CX4;H1,H2:2]>>[C:1]=[C:2]'),
    ('ketone reduction', '[CX3:1](=O)([#6])[#6]>>[C:1](O)'),
    ('alcohol oxidation', '[CX4;H1:1]([#6])([#6])[OX2H1]>>[C:1]=O'),
    ('ester hydrolysis', '[#6:1][CX3:2](=O)[OX2][#6]>>[#6:1][C:2](=O)O'),
    ('methyl ester', '[#6:1][CX3:2](=O)[OX2H1]>>[#6:1][C:2](=O)OC'),
    ('chain +CH2', '[CH3:1][CX4:2]>>[CH3:1]C[C:2]'),
    ('chain -CH2', '[CX4:1][CH2][CX4:2]>>[C:1][C:2]'),
    ('sulfation', '[OX2H1:1][#6:2]>>[#6:2][O:1]S(=O)(=O)O'),
    ('prenylation', '[cH1:1]>>[c:1]CC=C(C)C'),
]
_RXNS = None


def _reactions():
    global _RXNS
    if _RXNS is None:
        from rdkit.Chem import AllChem
        _RXNS = [(n, AllChem.ReactionFromSmarts(s)) for n, s in TRANSFORMS]
    return _RXNS


def mutate(mol, max_products=400):
    """All single-step products of `mol` (sanitised Mol objects, de-duplicated by canonical SMILES)."""
    from rdkit import Chem
    out = {}
    for _, rxn in _reactions():
        try:
            prods = rxn.RunReactants((mol,), maxProducts=50)
        except Exception:
            continue
        for p in prods:
            m = p[0]
            try:
                Chem.SanitizeMol(m)
                smi = Chem.MolToSmiles(m)
            except Exception:
                continue
            if smi not in out:
                out[smi] = m
            if len(out) >= max_products:
                return list(out.values())
    return list(out.values())


def generate(seed_smiles, target_mass, ppm=10.0, depth=2, max_seeds=20, beam=300):
    """Products of up to `depth` transformations of the seeds whose monoisotopic mass is within ppm of target.

    Returns {inchikey14: smiles}. Intermediate products are pruned to the `beam` closest to the target mass, so
    two-step paths that move toward the target survive.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    RDLogger.DisableLog('rdApp.*')
    tol = target_mass * ppm * 1e-6
    frontier, seen, hits = [], set(), {}
    for s in seed_smiles[:max_seeds]:
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            frontier.append(m)
            seen.add(Chem.MolToSmiles(m))
    for _ in range(depth):
        nxt = []
        for m in frontier:
            for p in mutate(m):
                smi = Chem.MolToSmiles(p)
                if smi in seen:
                    continue
                seen.add(smi)
                mass = ExactMolWt(p)
                if abs(mass - target_mass) <= tol:
                    flat = Chem.Mol(p)
                    Chem.RemoveStereochemistry(flat)
                    k = Chem.MolToInchiKey(flat)[:14]
                    if k:
                        hits.setdefault(k, smi)
                nxt.append((abs(mass - target_mass), smi))
            if len(nxt) > 4 * beam:                  # keep memory flat: only the closest-to-target survive
                nxt = sorted(nxt)[:beam]
        frontier = [Chem.MolFromSmiles(smi) for _, smi in sorted(nxt)[:beam]]
        frontier = [m for m in frontier if m is not None]
    return hits


def _feasibility_one(task):
    key, seeds, mass = task
    t = time.time()
    hits = generate(seeds, mass)
    return key, len(hits), key in hits, time.time() - t


def main(path, work):
    """How often does generation from the top analogs (answer's own spectra excluded) produce the answer?"""
    from multiprocessing import Pool

    from analogs import AnalogLibrary
    from casmi import TEST_ADDUCTS, molecule_mass, read_rows

    feats = pd.read_pickle(f'{work}/compare_features.pkl')
    answers = {f['key']: f['split'] for f in feats}
    holdout = set(open(f'{work}/holdout_e180.txt').read().split())
    t0 = time.time()
    lib = AnalogLibrary.build(path, exclude=set(answers))
    print(f'library {time.time() - t0:.0f}s', flush=True)
    queries = read_rows(path, lambda df: (((df.ingest_lib == 'enveda-np-examples')
                                           | ((df.ingest_lib == 'enveda-180') & df.inchikey14.isin(holdout)))
                                          & df.adduct.isin(TEST_ADDUCTS)))
    smiles_of = pd.read_parquet(f'{work}/structures.parquet', columns=['ik14', 'smiles']).drop_duplicates('ik14')
    smiles_of = smiles_of.set_index('ik14').smiles
    tasks = []
    for key, spectra in queries.groupby('inchikey14'):
        if key not in answers:
            continue
        hk, _ = lib.hits(spectra)
        seeds = [smiles_of[k] for k in hk if k in smiles_of.index and k != key]
        tasks.append((key, seeds, molecule_mass(spectra)))
    print(f'{len(tasks)} molecules, seeds ready {time.time() - t0:.0f}s', flush=True)
    import gc
    del lib, queries, smiles_of, feats
    gc.collect()                                     # workers must not inherit (and copy) the spectral library
    with Pool(4, maxtasksperchild=25) as p:
        rows = p.map(_feasibility_one, tasks, chunksize=1)
    res = pd.DataFrame(rows, columns=['key', 'n_generated', 'answer_generated', 'seconds'])
    res['split'] = res.key.map(answers)
    res.to_csv(f'{work}/generate_feasibility.csv', index=False)
    print(res.groupby('split').agg(molecules=('key', 'size'), answer_generated=('answer_generated', 'mean'),
                                   median_generated=('n_generated', 'median'),
                                   median_seconds=('seconds', 'median')).round(3).to_string())


if __name__ == '__main__':
    main(*sys.argv[1:3])
