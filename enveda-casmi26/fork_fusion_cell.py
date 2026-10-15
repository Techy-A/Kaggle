"""Added by aryanawasthiteykik: fuse the final lists with a second, independent engine (our own pipeline).

Our engine: spectrum -> fingerprint MLP over training + COCONUT + PubChem candidates, then a LambdaRank ranker
with in-silico fragmentation and analog-propagation features (code and weights in the
casmi26-spectrum-to-fingerprint-mlp dataset; v3 weights from the GPU training notebook when attached).
It runs in a subprocess with a deadline, so it can't exhaust memory or the 9-hour limit of this notebook.

Fusion per molecule, over metric identities (stereo-free InChIKey first block):
    score = 1/(K + rank_source) + BETA/(K + rank_ours),  K = 3
Source SMILES are kept for identities both lists share. With PROTECT_TOP1 the source's first guess stays first.
On any error, or when our engine has no list for a molecule, the source list is kept unchanged.

Class-3 candidates: our engine also mutates the structures behind the most similar library spectra (generate.py,
22 natural-product transformations, mass-filtered). When no library spectrum matches the molecule confidently
(best cosine < GEN_MAX_COS), its best-fitting generated structure is inserted at position GEN_POS. On validation
(natural products, answer removed from every database) the answer is generated for 42% of molecules and is the
best-fitting generated structure for 63% of those.
"""
import glob as _g, json as _json, os as _os, subprocess as _sp, sys as _s, time as _t, traceback as _tb
import pandas as _pd
OUR_BETA, OUR_K, PROTECT_TOP1 = 0.5, 3.0, True
GEN_POS, GEN_MAX_COS = 2, 0.7               # GEN_POS = 0 disables generated candidates
CONS_TOP, CONS_DEPTH = 3, 10               # agreement-only promotion (submission_consensus.csv)
LIMIT_S = 8.5 * 3600                      # stop our engine this long after the notebook started
_src = _pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
_src.to_csv('submission_before_fusion.csv', index=False)
try:
    import psutil as _ps
    _started = _ps.Process(_os.getpid()).create_time()
    _left = _started + LIMIT_S - _t.time()
    print(f'fusion: {(_t.time() - _started) / 3600:.2f} h elapsed, {_left / 3600:.2f} h left for our engine', flush=True)

    def _first(pattern):
        hits = sorted(_g.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
        return hits[0] if hits else None
    _code = _os.path.dirname(_first('casmi26-spectrum-to-fingerprint-mlp/**/pipeline.py'))
    _v3 = _first('fpnet_v3_config.json')                       # GPU training notebook output, if attached
    if _v3:
        _cfg = _json.load(open(_v3)); _d = _os.path.dirname(_v3)
        _model, _ranker = _os.path.join(_d, 'fpnet_v3h.pt'), _os.path.join(_d, 'learned_ranker_v3.txt')
        _hidden, _depth = _cfg['hidden'], _cfg['depth']
    else:
        _model, _ranker = _os.path.join(_code, 'fpnet_v2h.pt'), _os.path.join(_code, 'learned_ranker_frag_an.txt')
        _hidden, _depth = 2048, 2
    _runner = f'''
import pickle, sys, time
sys.path.insert(0, {_code!r})
from multiprocessing import Pool
import numpy as np, pandas as pd
from casmi import formula_mass
from fingerprints import fp_packed_v2
from pipeline import run
deadline = time.time() + {_left - 900:.0f}
class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in {{('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy._core.multiarray', '_reconstruct'),
                              ('numpy.core.multiarray', '_reconstruct')}}:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked {{module}}.{{name}}')
coco_dir = {_os.path.dirname(_first('coco_meta.pkl'))!r}
cm = NumpyOnly(open(coco_dir + '/coco_meta.pkl', 'rb')).load()
coco = pd.DataFrame({{'ik14': cm['keys'].astype(str), 'smiles': cm['smiles'].astype(str),
                     'mass': np.load(coco_dir + '/coco_mass.npy'), 'source': 'coconut'}})
tr = pd.read_parquet({_os.path.join(COMP, 'train.parquet')!r},
                     columns=['inchikey14', 'normalized_smiles', 'molecular_formula']).drop_duplicates('inchikey14')
tr = pd.DataFrame({{'ik14': tr.inchikey14, 'smiles': tr.normalized_smiles,
                   'mass': tr.molecular_formula.map(formula_mass), 'source': 'train'}})
structs = (pd.concat([tr, coco], ignore_index=True).dropna(subset=['mass'])
           .drop_duplicates('ik14').sort_values('mass', ignore_index=True))
with Pool(4) as pool:
    fps = np.stack(pool.map(fp_packed_v2, structs.smiles, chunksize=2000))
structs.to_parquet('/kaggle/working/our_structs.parquet'); np.save('/kaggle/working/our_fps.npy', fps)
del structs, fps, tr, coco, cm
sub = run({_os.path.join(COMP, 'train.parquet')!r}, {_os.path.join(COMP, 'test.parquet')!r},
          '/kaggle/working/our_structs.parquet', '/kaggle/working/our_fps.npy', {_model!r},
          {_os.path.dirname(_first('pc_smiles.npy'))!r}, {_os.path.dirname(_first('pc_lsid.npy'))!r},
          gate=0.7, hidden={_hidden}, depth={_depth}, ranker_path={_ranker!r}, deadline=deadline,
          gen_path='/kaggle/working/our_generated.csv' if {GEN_POS} else None)
sub.to_csv('/kaggle/working/our_submission.csv', index=False)
'''
    open('/kaggle/working/our_engine.py', 'w').write(_runner)
    if _left < 1800:
        raise RuntimeError('not enough time left for our engine')
    _r = _sp.run([_s.executable, '/kaggle/working/our_engine.py'], capture_output=True, text=True,
                 timeout=max(_left - 300, 60))
    print(_r.stdout[-3000:], _r.stderr[-3000:], flush=True)
    _ours = _pd.read_csv('/kaggle/working/our_submission.csv', dtype=str, keep_default_na=False)
    _ours = dict(zip(_ours.molecule_id, _ours.smiles))

    from functools import lru_cache as _lru
    from rdkit import Chem as _Chem
    from rdkit.Chem import rdmolops as _rdmolops

    @_lru(maxsize=None)
    def _ident(smi):
        m = _Chem.MolFromSmiles(smi)
        if m is None:
            return None
        _rdmolops.RemoveStereochemistry(m)
        k = _Chem.MolToInchiKey(m)
        return k[:14] if k else None

    _gen = {}
    if GEN_POS and _os.path.exists('/kaggle/working/our_generated.csv'):
        _g = _pd.read_csv('/kaggle/working/our_generated.csv')
        _g = _g[(_g.gen_rank == 1) & (_g.best_cos < GEN_MAX_COS)]
        _gen = dict(zip(_g.molecule_id, _g.smiles))
    _out, _out_nogen, _out_cons, _changed, _added, _inserted, _promoted = [], [], [], 0, 0, 0, 0
    for _mid, _smi in zip(_src.molecule_id, _src.smiles):
        _a = [x for x in _smi.split(';') if x]
        _b = [x for x in _ours.get(_mid, '').split(';') if x and x != 'C']
        if not _a or not _b:
            _out.append(_smi); _out_nogen.append(_smi); _out_cons.append(_smi); continue
        # agreement-only promotion: nothing new enters, the source's first guess stays first; a source candidate
        # at ranks 3-CONS_DEPTH that is among our engine's top CONS_TOP moves to rank 2
        _ours_top = {_ident(x) for x in _b[:CONS_TOP]} - {None}
        _ak = [_ident(x) for x in _a]
        _c = list(_a)
        for _i in range(2, min(CONS_DEPTH, len(_c))):
            if _ak[_i] is not None and _ak[_i] in _ours_top and _ak[_i] != _ak[0]:
                _c.insert(1, _c.pop(_i)); _promoted += 1
                break
        _out_cons.append(';'.join(_c[:25]))
        _score, _rep, _order = {}, {}, []
        for _i, _x in enumerate(_a):
            _k = _ident(_x) or f'src:{_x}'
            if _k not in _score:
                _score[_k], _rep[_k] = 1.0 / (OUR_K + _i + 1), _x; _order.append(_k)
        for _j, _x in enumerate(_b):
            _k = _ident(_x)
            if _k is None:
                continue
            if _k not in _score:
                _score[_k], _rep[_k] = 0.0, _x; _order.append(_k)
            _score[_k] += OUR_BETA / (OUR_K + _j + 1)
        _top = _order[0]
        _ranked = sorted(_order, key=lambda k: (-(1e9 if PROTECT_TOP1 and k == _top else _score[k])))[:25]
        _new = [_rep[k] for k in _ranked]
        _out_nogen.append(';'.join(_new))
        _gs = _gen.get(_mid)
        if _gs and _ident(_gs) and _ident(_gs) not in set(_ranked):
            _new = (_new[:GEN_POS - 1] + [_gs] + _new[GEN_POS - 1:])[:25]
            _inserted += 1
        _changed += _new != _a[:25]
        _added += len(set(_new) - set(_a))
        _out.append(';'.join(_new))
    _res = _pd.DataFrame({'molecule_id': _src.molecule_id, 'smiles': _out})
    assert len(_res) == len(_src) and _res.smiles.ne('').all()
    _res.to_csv('submission.csv', index=False)
    _pd.DataFrame({'molecule_id': _src.molecule_id, 'smiles': _out_nogen}).to_csv('submission_fusion_only.csv',
                                                                                   index=False)
    _pd.DataFrame({'molecule_id': _src.molecule_id, 'smiles': _out_cons}).to_csv('submission_consensus.csv',
                                                                                  index=False)
    print(f'consensus: {_promoted} source candidates promoted to rank 2 (submission_consensus.csv)')
    print(f'fusion: {_changed} of {len(_res)} lists changed, {_added} new candidates from our engine, '
          f'{len(_ours)} molecules ranked by our engine, {_inserted} generated candidates inserted at {GEN_POS}')
except Exception:
    _tb.print_exc()
    _src.to_csv('submission.csv', index=False)
    print('fusion FAILED; source submission kept')
