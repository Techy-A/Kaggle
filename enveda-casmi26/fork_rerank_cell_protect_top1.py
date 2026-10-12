"""Added by aryanawasthiteykik: re-rank each final list with the v2 fingerprint model.

Reciprocal-rank fusion: score = 1/(K + rank_source) + BETA/(K + rank_fingerprint), K = 3, BETA = 0.3.
Top guess protected: rank 1 never changes. Only the order inside each 25-candidate list can change (no candidate is added or removed), and the source
order wins unless the fingerprint model disagrees strongly. On any error the source submission is kept.
"""
import glob as _g, sys as _s, traceback as _tb
import numpy as _np, pandas as _pd, torch as _torch
OUR_BETA, OUR_K = 0.3, 3.0
_src = _pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
_src.to_csv('submission_before_fp_rerank.csv', index=False)
try:
    _code = os.path.dirname(sorted(_g.glob('/kaggle/input/**/casmi26-spectrum-to-fingerprint-mlp/**/fpmodel.py',
                                           recursive=True), key=len)[0])
    _s.path.insert(0, _code)
    from fpmodel import SpectrumStore as _Store, make_model as _make, predict_logits as _pred, fp_loglik as _ll
    from fingerprints import fp_packed_v2 as _fpv2
    from casmi import TEST_ADDUCTS as _ADD
    _model = _make(277 * 8, hidden=2048)
    _model.load_state_dict(_torch.load(os.path.join(_code, 'fpnet_v2_full.pt'), map_location='cpu'))
    _te = _pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _te = _te[_te.adduct.isin(_ADD)].reset_index(drop=True)
    _logits = _pred(_model, _Store.from_frame(_te), _np.arange(len(_te)))
    _rows = _te.groupby('molecule_id').indices
    _out, _moved = [], 0
    for _mid, _smi in zip(_src.molecule_id, _src.smiles):
        _c = [x for x in _smi.split(';') if x]
        if _mid not in _rows or len(_c) < 2:
            _out.append(_smi); continue
        _f = _np.stack([_fpv2(x) for x in _c])
        _sc = _ll(_logits[_rows[_mid]], _f)
        _r_fp = _np.empty(len(_c)); _r_fp[_np.argsort(-_sc, kind='stable')] = _np.arange(1, len(_c) + 1)
        _fused = 1.0 / (OUR_K + _np.arange(1, len(_c) + 1)) + OUR_BETA / (OUR_K + _r_fp)
        _order = [i for i in _np.argsort(-_fused, kind='stable') if i != 0]
        _new = [_c[0]] + [_c[i] for i in _order]   # top guess protected: only ranks 2-25 move
        _moved += _new[0] != _c[0]
        _out.append(';'.join(_new))
    _res = _pd.DataFrame({'molecule_id': _src.molecule_id, 'smiles': _out})
    assert len(_res) == len(_src) and _res.smiles.ne('').all()
    _res.to_csv('submission.csv', index=False)
    print(f'fingerprint re-rank: top-1 changed for {_moved} of {len(_res)} molecules')
except Exception:
    _tb.print_exc()
    _src.to_csv('submission.csv', index=False)
    print('fingerprint re-rank FAILED; source submission kept')