"""CASMI yardstick (step 1 of the hard-work plan, 7 Nov).
Question: does a timsTOF, scaffold-disjoint holdout rank scorer variants the way the leaderboard did, where the
kit's random holdout did not? Known LB facts: the long-trained scorer (40 x 2000 steps, same code) looked much better
on the random holdout but lost 0.019 on the LB as the sole scorer; heavier/longer neural models keep losing.
What it does (one GPU run, public kit sanpier/casmi-spectrum-model-v1, unchanged training code):
  1. kit prepare (patched only to remember each spectrum's ingest_lib);
  2. two holdouts excluded from training: TS = enveda-180 (timsTOF) molecules held out by Murcko scaffold (all
     molecules sharing those scaffolds leave training); R = random 3% of the remaining molecules (the kit's style);
  3. trains three configs: short 16x500 (sanpier v1), v80 80x500 patience 4, long 40x2000 (no early stop);
  4. scores each final and best checkpoint on TS (queries = timsTOF spectra only) and R (all competition spectra),
     256 candidates = true + same-formula decoys + nearby mass (the kit's diagnostic), MRR@25 and top-1.
Gate: R must prefer long; the yardstick is useful only if TS does not (matching the LB). Output: yard_results.json.
"""
import os, sys, json, time, shutil, subprocess, hashlib
from pathlib import Path
T0 = time.time()
def log(*a): print(f'[{time.time()-T0:7.0f}s]', *a, flush=True)
try:
    from rdkit import rdBase
except ImportError:
    _wh = sorted(Path('/kaggle/input').rglob('rdkit-*.whl')); assert _wh
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--find-links', str(_wh[0].parent), 'rdkit==2026.03.3'], check=True)
    from rdkit import rdBase
log('rdkit', rdBase.rdkitVersion)
import numpy as np, pandas as pd, torch
SMOKE = os.environ.get('YARD_SMOKE')  # local CPU code-path check on a small parquet slice
KW = Path(os.environ.get('YARD_WORK', '/kaggle/working'))
kit = Path(SMOKE) if SMOKE else sorted(Path('/kaggle/input').rglob('casmi_spectrum_model_v1/train.py'))[0].parent
stage = KW / 'kit'; shutil.copytree(kit, stage, dirs_exist_ok=True)
src = (stage / 'prepare.py').read_text()
for old, new in [("'smiles':['smiles','canonical_smiles']", "'smiles':['smiles','canonical_smiles','normalized_smiles']"),
                 ("str(row.get('molecule_id',''))", "str(row.get('ingest_lib',''))")]:
    assert old in src, old; src = src.replace(old, new)
(stage / 'prepare.py').write_text(src); sys.path.insert(0, str(stage))
import prepare, train
cfg0 = json.loads((kit / 'config.json').read_text())
cfg0['train_path'] = os.environ['YARD_TRAIN'] if SMOKE else str(sorted(Path('/kaggle/input').rglob('train.parquet'))[0])
cfg0['work'] = str(KW / 'cache'); cfg0['external_paths'] = []; cfg0['refit_full'] = False
if SMOKE: cfg0['require_gpu'] = False
audit = prepare.prepare(cfg0); log('PREP', audit['spectra'], audit['molecules'])

# ---- holdouts
W = Path(cfg0['work']); mol = pd.read_parquet(W / 'molecules.parquet'); spec = pd.read_parquet(W / 'spectra.parquet')
spec['lib'] = spec.original_molecule_id
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
def scaf(s):
    m = Chem.MolFromSmiles(s)
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m is not None else ''
    except Exception: return ''
enveda = set(spec.loc[spec.lib.eq('enveda-180'), 'key'])
mol['scaf'] = [scaf(s) or ('acyclic:' + k) for s, k in zip(mol.smiles, mol.key)]
h = lambda x: hashlib.sha256(('yard7nov' + x).encode()).hexdigest()
ev_scaf = mol[mol.key.isin(enveda)].groupby('scaf').key.size()
chosen, n = set(), 0
for s in sorted(ev_scaf.index, key=h):
    chosen.add(s); n += ev_scaf[s]
    if n >= (40 if SMOKE else 4000): break
ts = set(mol.key[mol.scaf.isin(chosen)])
rest = sorted(set(mol.key) - ts, key=h); r = set(rest[:int(0.03 * len(rest))])
mol['split'] = ['validation' if (k in ts or k in r) else 'train' for k in mol.key]
spec['split'] = spec.key.map(dict(zip(mol.key, mol.split)))
mol.drop(columns=['scaf']).to_parquet(W / 'molecules.parquet', index=False)
spec.drop(columns=['lib']).to_parquet(W / 'spectra.parquet', index=False)
log('HOLDOUT ts molecules', len(ts), 'ts with timsTOF', len(ts & enveda), 'scaffolds', len(chosen), 'r', len(r),
    'train', int(mol.split.eq('train').sum()))
data = train.Data(cfg0)
libs = spec.lib.values
key2i = {k: i for i, k in enumerate(data.mol.key)}
ts_ids = sorted([key2i[k] for k in ts & enveda], key=lambda i: h(data.mol.key.iloc[i]))[:1500]
r_ids = sorted([key2i[k] for k in r], key=lambda i: h(data.mol.key.iloc[i]))[:1500]
pools = train.hard_pools(data.mol, np.arange(len(data.mol)), cfg0['val_candidates'] - 1)
CANDS = {'TS': {i: np.r_[i, pools[i]] for i in ts_ids}, 'R': {i: np.r_[i, pools[i]] for i in r_ids}}
# kit early stopping looks at data.val[:1000]: make it a fixed half-half mix so no config is tuned on one side
data.val = np.array(ts_ids[:500] + r_ids[:500])

@torch.inference_mode()
def evaluate(model, name, device, cfg):
    model.eval(); out = {}
    for setname, cands in CANDS.items():
        rows = []
        for i, ids in cands.items():
            sr = data.by_mol[i]; sr = sr[data.spec.source.values[sr] == 0]
            if setname == 'TS': sr = sr[libs[sr] == 'enveda-180']
            if not len(sr): continue
            zs = 0; ps = 0; n = 0
            for st in range(0, len(sr), cfg['batch_size']):
                b = sr[st:st + cfg['batch_size']]; z, p = model.encode_spectra(*data.batch(b, device)); zs = zs + z.sum(0); ps = ps + p.sum(0); n += len(b)
            fp = torch.tensor(np.asarray(data.fp[ids]), device=device, dtype=torch.float32)
            sc = ((model.encode_molecules(fp) @ (zs / n)) / cfg['temperature'] + cfg['fp_score_weight'] * (fp @ (ps / n)) / cfg['fp_bits']).cpu().numpy()
            rank = 1 + int(np.sum(sc[1:] >= sc[0])); rows.append((data.mol.key.iloc[i], rank))
        f = pd.DataFrame(rows, columns=['key', 'rank']); f.to_csv(f'{KW}/ranks_{name}_{setname}.csv', index=False)
        out[setname] = dict(n=len(f), mrr25=float(np.where(f['rank'] <= 25, 1 / f['rank'], 0).mean()), top1=float((f['rank'] == 1).mean()))
    log('EVAL', name, json.dumps(out)); return out

from model import SpectrumCandidateModel
device = torch.device('cpu' if SMOKE else 'cuda'); assert SMOKE or torch.cuda.is_available()
CONFIGS = [('short', dict(epochs=16, steps_per_epoch=500, patience=4)),
           ('v80', dict(epochs=80, steps_per_epoch=500, patience=4)),
           ('long', dict(epochs=40, steps_per_epoch=2000, patience=999))]
if SMOKE: CONFIGS = [(n, dict(o, epochs=2, steps_per_epoch=3)) for n, o in CONFIGS]
results = dict(audit=dict(ts=len(ts), ts_timstof=len(ts & enveda), scaffolds=len(chosen), r=len(r),
                          n_eval_ts=len(ts_ids), n_eval_r=len(r_ids)))
for name, over in CONFIGS:
    cfg = dict(cfg0, **over); out = KW / f'run_{name}'; out.mkdir(parents=True, exist_ok=True)
    t = time.time(); res = train.train_stage(data, cfg, out, device, deadline=time.monotonic() + 2.4 * 3600)
    log('TRAINED', name, json.dumps(res), round(time.time() - t), 's')
    results[name] = dict(train=res, seconds=round(time.time() - t))
    for ck in ('best', 'last'):
        m = SpectrumCandidateModel(cfg).to(device)
        m.load_state_dict(torch.load(out / f'{ck}_validation_model.pt', map_location=device, weights_only=False)['state_dict'])
        results[name][ck] = evaluate(m, f'{name}_{ck}', device, cfg); del m
    for f in out.glob('*.pt'): f.unlink()
    (KW / 'yard_results.json').write_text(json.dumps(results, indent=2))
shutil.rmtree(W, ignore_errors=True)
log('YARD_DONE', json.dumps(results))
