"""Can generated candidates be ranked? Class-3 simulation on the natural-product split (S1).

For every S1 molecule the answer is removed from every pool. Candidates are then the PubChem/pool window (as in
compare_features.pkl, answer dropped) plus structures generated from the top analogs (generate.py). Each generated
structure is scored like any candidate: fingerprint log-likelihood (held-out model) and fragmentation explained
intensity. Reported:
- where the answer ranks among the generated structures,
- how its fit compares with the best surviving pool candidate (could it be inserted high in the final list?),
- the same comparison for the best generated structure when the answer IS in the pool (the cost of inserting).
Writes <work>/gen_eval.parquet.
"""
import sys
import time

import numpy as np
import pandas as pd


def _score_task(task):
    key, gen, mz, it, positive = task
    from fragments import explain_scores, fragment_masses
    out = []
    for k, smi in gen.items():
        f1, f2 = fragment_masses(smi)
        out.append((key, k, smi, explain_scores(mz, it, f1, f2, positive)[0]))
    return out


def _gen_task(task):
    from generate import generate
    key, seeds, mass = task
    return key, generate(seeds, mass)


def main(path, work, model='fpnet_v2h', nbytes=277, hidden=2048):
    from multiprocessing import Pool

    import torch
    from analogs import AnalogLibrary
    from casmi import TEST_ADDUCTS, molecule_mass, read_rows
    from fingerprints import fp_packed_v2
    from fpmodel import SpectrumStore, fp_loglik, make_model, predict_logits
    from fragments import molecule_peaks

    feats = [f for f in pd.read_pickle(f'{work}/compare_features.pkl') if f['split'].startswith('S1')]
    keys = {f['key'] for f in feats}
    all_answers = {f['key'] for f in pd.read_pickle(f'{work}/compare_features.pkl')}
    t0 = time.time()
    lib = AnalogLibrary.build(path, exclude=all_answers)
    queries = read_rows(path, lambda df: (df.ingest_lib == 'enveda-np-examples') & df.adduct.isin(TEST_ADDUCTS)
                        & df.inchikey14.isin(keys)).reset_index(drop=True)
    smiles_of = pd.read_parquet(f'{work}/structures.parquet', columns=['ik14', 'smiles']).drop_duplicates('ik14')
    smiles_of = smiles_of.set_index('ik14').smiles
    groups = dict(list(queries.groupby('inchikey14')))
    gen_tasks = []
    for key, spectra in groups.items():
        hk, _ = lib.hits(spectra)
        gen_tasks.append((key, [smiles_of[k] for k in hk if k in smiles_of.index and k != key],
                          molecule_mass(spectra)))
    import gc
    del lib
    gc.collect()
    print(f'seeds {time.time() - t0:.0f}s', flush=True)
    with Pool(4, maxtasksperchild=25) as p:
        generated = dict(p.map(_gen_task, gen_tasks, chunksize=1))
    print(f'generated {sum(map(len, generated.values()))} structures {time.time() - t0:.0f}s', flush=True)

    net = make_model(nbytes * 8, hidden=hidden)
    net.load_state_dict(torch.load(f'{work}/{model}.pt', map_location='cpu'))
    logits = predict_logits(net, SpectrumStore.from_frame(queries), np.arange(len(queries)))

    frag_tasks = []
    for key, spectra in groups.items():
        mz, it = molecule_peaks(spectra)
        frag_tasks.append((key, generated[key], mz, it, (spectra.ionization_mode == 'positive').mean() >= 0.5))
    with Pool(4, maxtasksperchild=25) as p:
        rows = [r for chunk in p.map(_score_task, frag_tasks, chunksize=1) for r in chunk]
    gen = pd.DataFrame(rows, columns=['key', 'id', 'smiles', 'frag_int'])
    fps = np.stack([fp_packed_v2(s) for s in gen.smiles]) if len(gen) else np.zeros((0, nbytes), np.uint8)
    gen['fp'] = np.nan
    for key, spectra in groups.items():
        sel = np.flatnonzero(gen.key.to_numpy() == key)
        if len(sel):
            gen.loc[gen.index[sel], 'fp'] = fp_loglik(logits[spectra.index.to_numpy()], fps[sel]) / len(spectra)
    gen['is_answer'] = gen.id == gen.key

    out = []
    for f in feats:
        key = f['key']
        pool_fp = pd.Series(f['fp'][model]).groupby(f['ids']).max()
        best_pool_other = pool_fp.drop(key, errors='ignore').max()        # class 3: the answer is not in the pool
        g = gen[gen.key == key]
        ans = g[g.is_answer]
        out.append(dict(key=key, n_gen=len(g), generated=len(ans) > 0,
                        gen_rank_fp=int((g.fp > ans.fp.iloc[0]).sum()) + 1 if len(ans) else None,
                        gen_rank_frag=int((g.frag_int > ans.frag_int.iloc[0]).sum()) + 1 if len(ans) else None,
                        answer_fp=ans.fp.iloc[0] if len(ans) else np.nan,
                        best_gen_fp=g.fp.max() if len(g) else np.nan,
                        best_pool_other_fp=best_pool_other,
                        answer_pool_fp=pool_fp.get(key, np.nan)))
    res = pd.DataFrame(out)
    res.to_parquet(f'{work}/gen_eval.parquet')
    gen.to_parquet(f'{work}/gen_candidates.parquet')
    hit = res[res.generated]
    print(f'{len(res)} S1 molecules, answer generated for {len(hit)} ({len(hit) / len(res):.0%})')
    print('rank of the answer among generated (by fingerprint fit):',
          hit.gen_rank_fp.value_counts().sort_index().head(5).to_dict())
    print(f'answer beats every surviving pool candidate on fit (class-3 sim): '
          f'{(hit.answer_fp > hit.best_pool_other_fp).mean():.0%} of generated')
    print(f'best generated beats the TRUE answer in the pool (class-2 cost): '
          f'{(res.best_gen_fp > res.answer_pool_fp).mean():.0%} of all molecules')
    print(f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(*sys.argv[1:3])
