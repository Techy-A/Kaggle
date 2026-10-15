"""End-to-end ranking for the hidden test: PubChem + COCONUT + training candidates, fingerprint likelihood,
popularity / natural-product prior, cosine gate.

    python pipeline.py <train.parquet> <test.parquet> <structures.parquet> <fps.npy> <model.pt>
                       <pubchem_dir> <popularity_dir> [w_pop] [w_np] [gate] [hidden]
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from casmi import TEST_ADDUCTS, Library, molecule_mass, rank_molecule
from fpmodel import SpectrumStore, fp_loglik, make_model, predict_logits
from pubchem import PubChemTier, repair

PPM = 10


def fp_and_smiles(smiles, maccs=False):
    """Packed fingerprint (Morgan 2048, plus MACCS when maccs=True) and a clean canonical SMILES for one candidate."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import MACCSkeys, rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    mol = repair(smiles)
    if mol is None:
        return np.zeros(277 if maccs else 256, np.uint8), ''
    bits = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol).astype(np.uint8)
    if maccs:
        extra = np.zeros(168, np.uint8)
        extra[:167] = np.array(list(MACCSkeys.GenMACCSKeys(mol)), np.uint8)
        bits = np.concatenate([bits, extra])
    return np.packbits(bits), Chem.MolToSmiles(mol, isomericSmiles=False)


def fp_and_smiles_maccs(smiles):
    return fp_and_smiles(smiles, maccs=True)


def prior_order(fp, cos, pop, natural, w_pop, w_np, gate, extra=0.0):
    """Confident library matches first (cosine >= gate); the rest by z-scored fingerprint likelihood + prior."""
    z = (fp - fp.mean()) / (fp.std() + 1e-9)
    score = np.where(cos >= gate, 1e6 + cos, z + w_pop * pop + w_np * natural + extra)
    return np.argsort(-score, kind='stable')


def pool_priors(structs, pop_dir):
    pp = pd.read_csv(f'{pop_dir}/pool_popularity.csv',
                     usecols=['inchikey14', 'pop', 'np_coconut', 'np_lotus', 'np_npatlas']).drop_duplicates('inchikey14')
    pp = pp.set_index('inchikey14').reindex(structs.ik14.to_numpy())
    pop = pp['pop'].fillna(0).to_numpy(np.float32)
    natural = (pp[['np_coconut', 'np_lotus', 'np_npatlas']].fillna(0).sum(1) > 0).to_numpy()
    return pop, natural | (structs.source == 'coconut').to_numpy()


def frag_z(ids, fp, pop, smi, spectra, n_fit=40, n_pop=10, missing=-3.0):
    """Per-row z-scored fragmentation explain score; candidates outside the fragmented set get `missing`.

    The fragmented set is the top n_fit identities by fingerprint fit plus the top n_pop by popularity,
    matching frag_features.py, where the weight was tuned.
    """
    from fragments import explain_scores, fragment_masses, molecule_peaks
    best = pd.DataFrame({'id': ids, 'fp': fp, 'pop': pop}).groupby('id', sort=False).max()
    chosen = list(dict.fromkeys(list(best.nlargest(n_fit, 'fp').index) + list(best.nlargest(n_pop, 'pop').index)))
    pos = {k: i for i, k in reversed(list(enumerate(ids)))}
    mz, it = molecule_peaks(spectra)
    positive = (spectra.ionization_mode == 'positive').mean() >= 0.5
    scores = {}
    for cid in chosen:
        if smi[pos[cid]]:
            f1, f2 = fragment_masses(smi[pos[cid]])
            scores[cid] = explain_scores(mz, it, f1, f2, positive)[0]
    if not scores:
        return np.zeros(len(ids))
    v = np.array(list(scores.values()))
    mu, sd = v.mean(), v.std() + 1e-9
    return np.array([(scores[k] - mu) / sd if k in scores else missing for k in ids])


def ranker_order(booster, ids, fp, cos, pop, natural, gate, smi=None, spectra=None, cfp=None, an_ctx=None):
    """Indices of distinct candidates: confident library matches (cosine >= gate) first, then the learned ranker.

    If the booster was trained with fragmentation features, the strongest candidates (top 40 by fit plus
    top 10 by popularity, chosen exactly as in frag_features.py) are fragmented against this molecule's peaks.
    With analog features, the same candidates are compared (Morgan Tanimoto, `cfp` row-aligned packed fingerprints)
    with the structures behind the most similar library spectra (`an_ctx` = (AnalogLibrary, hit_fp)).
    """
    from learned_ranker import FEATURES, FRAG_FEATURES, add_frag_features, candidate_frame
    first = [i for i in np.argsort(-cos, kind='stable') if cos[i] >= gate]
    df = candidate_frame(ids, fp, pop, natural)
    pos = {k: i for i, k in enumerate(ids)}                    # first row index of each identity
    features = list(FEATURES)
    if 'frag_int' in booster.feature_name():
        from fragments import explain_scores, fragment_masses, molecule_peaks
        chosen = list(dict.fromkeys(list(df.nlargest(40, 'fp').id) + list(df.nlargest(10, 'pop').id)))
        mz, it = molecule_peaks(spectra)
        positive = (spectra.ionization_mode == 'positive').mean() >= 0.5
        rows = []
        for cid in chosen:
            if smi[pos[cid]]:
                f1, f2 = fragment_masses(smi[pos[cid]])
                rows.append((0, cid, *explain_scores(mz, it, f1, f2, positive)))
        df['qid'] = 0
        df = add_frag_features(df, pd.DataFrame(rows, columns=['qid', 'id', 'frag_int', 'frag_peaks']))
        features += FRAG_FEATURES
        if 'an_max' in booster.feature_name():
            from analogs import AN_FEATURES, add_analog_features, molecule_analogs
            lib, hit_fp = an_ctx
            cands = [c for c in chosen if smi[pos[c]]]
            an = molecule_analogs(lib, spectra, cands, cfp[[pos[c] for c in cands], :256], hit_fp)
            df = add_analog_features(df, an.assign(qid=0))
            features += AN_FEATURES
    df['score'] = booster.predict(df[features])
    return first + [pos[k] for k in df.sort_values('score', ascending=False).id]


def run(train_path, test_path, structs_path, fps_path, model_path, pc_dir, pop_dir,
        w_pop=0.25, w_np=1.0, gate=0.7, n_out=25, hidden=1536, ranker_path=None, w_frag=0.0, depth=2,
        deadline=None, gen_path=None, n_gen=3):
    """deadline: time.time() after which remaining molecules are left without guesses (placeholder 'C').

    gen_path: also generate candidates from the top analogs (generate.py) and write, per molecule, the n_gen
    best-fitting generated structures that are not already candidates (plus the molecule's best cosine) to this CSV.
    """
    import torch
    t0 = time.time()
    test = pd.read_parquet(test_path).reset_index(drop=True)
    test = test[test.adduct.isin(TEST_ADDUCTS)].reset_index(drop=True)
    all_ids = pd.read_parquet(test_path, columns=['molecule_id']).molecule_id.unique()

    lib = Library.build(train_path, keep=lambda df: df.adduct.isin(TEST_ADDUCTS))
    print(f'cosine library {time.time() - t0:.0f}s', flush=True)

    structs = pd.read_parquet(structs_path)
    fps = np.load(fps_path)
    masses, keys, smiles = structs.mass.to_numpy(), structs.ik14.to_numpy(), structs.smiles.to_numpy()
    s_pop, s_np = pool_priors(structs, pop_dir)

    pc = PubChemTier(pc_dir)
    pc_ik14 = np.load(f'{pop_dir}/pc_ik14.npy', mmap_mode='r')
    pc_lsid = np.load(f'{pop_dir}/pc_lsid.npy', mmap_mode='r')
    pc_lpmid = np.load(f'{pop_dir}/pc_lpmid.npy', mmap_mode='r')
    pc_npf = np.load(f'{pop_dir}/pc_np.npy', mmap_mode='r')

    maccs = fps.shape[1] == 277              # the fingerprint layout follows the pool file
    model = make_model(fps.shape[1] * 8, hidden=hidden, depth=depth)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    logits = predict_logits(model, SpectrumStore.from_frame(test), np.arange(len(test)))

    groups = list(test.groupby('molecule_id', sort=False))
    windows, flat = [], []
    for mol, spectra in groups:
        lo, hi = pc.window(molecule_mass(spectra), PPM)
        windows.append((len(flat), lo, hi))
        flat.extend(pc.smiles(lo, hi))
    with Pool() as p:
        res = p.map(fp_and_smiles_maccs if maccs else fp_and_smiles, flat, chunksize=2000)
    del flat
    pc_fps = np.stack([r[0] for r in res])
    pc_smiles = np.array([r[1] for r in res], dtype=object)
    del res
    print(f'PubChem candidates {len(pc_fps):,} fingerprinted {time.time() - t0:.0f}s', flush=True)

    booster = None
    if ranker_path:
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(ranker_path))
    an_ctx = None
    if booster is not None and 'an_max' in booster.feature_name():
        from analogs import AnalogLibrary
        an_lib = AnalogLibrary.build(train_path)
        row_of = pd.Series(np.arange(len(keys)), index=keys)
        row_of = row_of[~row_of.index.duplicated()].to_dict()
        an_ctx = (an_lib, lambda k: fps[row_of[k], :256] if k in row_of else None)
        print(f'analog library {an_lib.matrix.shape[0]:,} spectra {time.time() - t0:.0f}s', flush=True)
    generated = {}
    if gen_path:
        from generate import generate_many
        if an_ctx is None:
            from analogs import AnalogLibrary
            an_lib = AnalogLibrary.build(train_path)
        else:
            an_lib = an_ctx[0]
        smiles_of = dict(zip(keys, smiles))
        tasks = []
        for mol, spectra in groups:
            hk, _ = an_lib.hits(spectra)
            tasks.append((mol, [smiles_of[k] for k in hk if smiles_of.get(k)], molecule_mass(spectra)))
        generated = generate_many(tasks)
        print(f'generated {sum(map(len, generated.values())):,} structures {time.time() - t0:.0f}s', flush=True)
    gen_rows = []
    out = {}
    for (mol, spectra), (start, plo, phi) in zip(groups, windows):
        if deadline and time.time() > deadline:
            print(f'deadline reached after {len(out)} of {len(groups)} molecules', flush=True)
            break
        n = phi - plo
        m = molecule_mass(spectra)
        lo, hi = np.searchsorted(masses, [m * (1 - PPM * 1e-6), m * (1 + PPM * 1e-6)])
        lg = logits[spectra.index.to_numpy()]
        _, cos = rank_molecule(lib, spectra, ppm=PPM)
        ids = np.concatenate([keys[lo:hi], np.asarray(pc_ik14[plo:phi]).astype(str)])
        smi = np.concatenate([smiles[lo:hi], pc_smiles[start:start + n]])
        fp = np.concatenate([fp_loglik(lg, fps[lo:hi]), fp_loglik(lg, pc_fps[start:start + n])]) / len(spectra)
        cosv = np.concatenate([[cos.get(k, 0.0) for k in keys[lo:hi]], np.zeros(n)])
        pop = np.concatenate([s_pop[lo:hi], np.asarray(pc_lsid[plo:phi], np.float32)
                              + np.asarray(pc_lpmid[plo:phi], np.float32)])
        natural = np.concatenate([s_np[lo:hi], np.asarray(pc_npf[plo:phi]) > 0])
        if generated.get(mol):
            known = set(ids)
            new = [(k, g) for k, g in generated[mol].items() if k not in known]
            if new:
                from fingerprints import fp_packed, fp_packed_v2
                gfp = np.stack([(fp_packed_v2 if maccs else fp_packed)(g) for _, g in new])
                gscore = fp_loglik(lg, gfp) / len(spectra)
                best_cos = float(cosv.max()) if len(cosv) else 0.0
                for r, j in enumerate(np.argsort(-gscore)[:n_gen]):
                    gen_rows.append((mol, r + 1, new[j][1], float(gscore[j]),
                                     float(gscore[j] - fp.max()) if len(fp) else 0.0, best_cos))
        if len(ids) == 0:
            continue
        guesses, seen = [], set()
        if booster is not None:
            cfp = np.concatenate([fps[lo:hi], pc_fps[start:start + n]]) if an_ctx else None
            order = ranker_order(booster, ids, fp, cosv, pop, natural, gate, smi, spectra, cfp, an_ctx)
        elif w_frag:
            order = prior_order(fp, cosv, pop, natural, w_pop, w_np, gate,
                                extra=w_frag * frag_z(ids, fp, pop, smi, spectra))
        else:
            order = prior_order(fp, cosv, pop, natural, w_pop, w_np, gate)
        for i in order:
            if not smi[i] or ids[i] in seen:        # skip unparsable rows and duplicate InChIKey14s
                continue
            seen.add(ids[i])
            guesses.append(smi[i])
            if len(guesses) == n_out:
                break
        out[mol] = ';'.join(guesses)
    if gen_path:
        pd.DataFrame(gen_rows, columns=['molecule_id', 'gen_rank', 'smiles', 'fp', 'fp_gap_to_pool', 'best_cos']
                     ).to_csv(gen_path, index=False)
    sub = pd.DataFrame({'molecule_id': all_ids})
    sub['smiles'] = sub.molecule_id.map(out).fillna('C').replace('', 'C')
    print(f'done {time.time() - t0:.0f}s, molecules without candidates: {(sub.smiles == "C").sum()}')
    return sub


if __name__ == '__main__':
    a = sys.argv[1:]
    sub = run(*a[:7], *(float(x) for x in a[7:10]), hidden=int(a[10]) if len(a) > 10 else 1536)
    sub.to_csv('submission.csv', index=False)
