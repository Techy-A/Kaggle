# DreaMS embeddings (step 2 of the DreaMS retriever): install DreaMS (MIT, Zenodo 10997887 weights) offline-safe,
# embed training molecules and all test spectra with the official checkpoint. Output: train_emb.npy, test_emb.npy.
import subprocess, time, os, glob, sys
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'  # official DreaMS checkpoint (Zenodo 10997887, host-approved); torch>=2.6 defaults to weights_only=True
def sh(c):
    print('$', c, flush=True); r = subprocess.run(c, shell=True, capture_output=True, text=True); print(r.stdout[-4000:], r.stderr[-3000:], flush=True)
W = '/kaggle/working'
sh('pip download --no-deps -d /kaggle/working/wheels git+https://github.com/pluskal-lab/DreaMS.git 2>&1 | tail -3')
sh('cd /kaggle/working && rm -rf dreams_src && mkdir dreams_src && cd dreams_src && unzip -q ../wheels/dreams-1.0.0.zip && ls && cat */pyproject.toml 2>/dev/null | sed -n 1,60p; cat */setup.py 2>/dev/null | head -60')
sh('cd /kaggle/working/dreams_src/* && pip install -q --no-deps . 2>&1 | tail -3')
import importlib
sh('pip install -q --no-deps git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main 2>&1 | tail -3; pip download -q --no-deps -d /kaggle/working/wheels git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main 2>&1 | tail -1; python -c "import msml,os;print(\'msml at\',os.path.dirname(msml.__file__))"')
sh('pip install -q --no-deps --ignore-requires-python matchms==0.27.0 pytorch-lightning==2.0.8 2>&1 | tail -2; pip download -q --no-deps --ignore-requires-python -d /kaggle/working/wheels matchms==0.27.0 pytorch-lightning==2.0.8 2>&1|tail -1')
for mod,pkg in [('pytorch_lightning','pytorch-lightning'),('lightning','lightning'),('torchmetrics','torchmetrics'),('pyteomics','pyteomics'),('h5py','h5py'),('wandb','wandb'),('igraph','python-igraph'),('rdkit','rdkit'),('pandarallel','pandarallel'),('huggingface_hub','huggingface_hub')]:
    try: importlib.import_module(mod)
    except Exception: sh(f'pip install -q --no-deps {pkg} 2>&1 | tail -2')
sh('curl -s https://zenodo.org/api/records/10997887 | python -c "import json,sys;d=json.load(sys.stdin);[print(f[\'key\'],f[\'size\'],f[\'links\'][\'self\']) for f in d[\'files\']]"')
sh('mkdir -p /kaggle/working/ckpt && cd /kaggle/working/ckpt && curl -s https://zenodo.org/api/records/10997887 | python -c "import json,sys;d=json.load(sys.stdin);print(\'\\n\'.join(f[\'links\'][\'self\']+\' \'+f[\'key\'] for f in d[\'files\']))" | while read u k; do curl -sL -o "$k" "$u"; done; ls -la')
sh('pip show dreams 2>/dev/null | head -12; python -c "import dreams,os;print(os.path.dirname(dreams.__file__))"')
sh('curl -sL https://raw.githubusercontent.com/pluskal-lab/DreaMS/main/LICENSE | head -5')
sh('curl -sL https://raw.githubusercontent.com/pluskal-lab/DreaMS/main/README.md | grep -iE "zenodo|huggingface|checkpoint|ssl_model|license|embedding" | head -40')
STUB="import sys,types; _m=types.ModuleType('pyopenms'); _m.__file__='/tmp/pyopenms_stub.py'; _m.__getattr__=lambda n: (_ for _ in ()).throw(AttributeError(n)) if n.startswith('__') else type(n,(),{}); sys.modules['pyopenms']=_m; "
MAP={'pickydict':'pickydict','sklearn':'scikit-learn','yaml':'pyyaml','umap':'umap-learn','fire':'fire','ase':'ase','molplotly':'molplotly','msml':'git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main','pyopenms':'pyopenms','igraph':'python-igraph','numba':'numba','sparsestack':'sparsestack','deprecated':'Deprecated','lxml':'lxml','pubchempy':'PubChemPy','pynndescent':'pynndescent','lightning_utilities':'lightning-utilities','torchmetrics':'torchmetrics','pytorch_lightning':'pytorch-lightning','wandb':'wandb','plotly':'plotly','SpectralEntropy':'SpectralEntropy','ms_entropy':'ms_entropy','dash':'dash','pyteomics':'pyteomics','tqdm':'tqdm'}
import re as _re
for _ in range(30):
    r=subprocess.run('python -c "%simport dreams.api"' % STUB,shell=True,capture_output=True,text=True)
    m=_re.search(r"No module named '([A-Za-z0-9_\.]+)'",r.stderr)
    if r.returncode==0: print('IMPORT OK',flush=True); break
    if not m: print('IMPORT FAIL',r.stderr[-2500:],flush=True); break
    mod=m.group(1).split('.')[0]; pkg=MAP.get(mod,mod)
    if mod=='matchms': print('matchms missing after pin; stop'); break
    print('installing',mod,'->',pkg,flush=True)
    sh(f'pip install -q --no-deps {pkg} 2>&1 | tail -2; pip download -q --no-deps -d /kaggle/working/wheels {pkg} 2>&1 | tail -1')

# ---------------------------------------------------------------------------------------------------------------
# Step 2: DreaMS embeddings for training molecules (up to 2 spectra each, most peaks first, mean-pooled) and for
# every test spectrum. Output float16 arrays for the retriever head (step 3).
exec(STUB)
import numpy as np, pandas as pd, pyarrow.parquet as pq, time
from dreams.api import dreams_embeddings
T0 = time.time(); BUDGET = 7.5 * 3600
CK = f'{W}/ckpt/embedding_model.ckpt'
def embed(df, tag):
    p = f'{W}/{tag}.mgf'
    with open(p, 'w') as f:
        for i, r in enumerate(df.itertuples()):
            f.write(f'BEGIN IONS\nTITLE={i}\nPEPMASS={r.precursor_mz}\nCHARGE=1+\n')
            for m, v in zip(r.ms2_mzs, r.ms2_normalized_intensities): f.write(f'{m} {v}\n')
            f.write('END IONS\n')
    E = dreams_embeddings(p, model_pth=CK, batch_size=256, progress_bar=False)
    os.remove(p); return np.asarray(E, dtype=np.float32)
comp = glob.glob('/kaggle/input/**/test.parquet', recursive=True)
te = pd.read_parquet([c for c in comp if 'competitions' in c or 'enveda' in c][0], columns=['molecule_id','spectrum_id','precursor_mz','ms2_mzs','ms2_normalized_intensities'])
t = time.time(); Et = embed(te, 'test'); print('TEST', Et.shape, round(time.time()-t,1), 's', flush=True)
np.save(f'{W}/test_emb.npy', Et.astype(np.float16)); te[['molecule_id','spectrum_id']].to_csv(f'{W}/test_ids.csv', index=False)
tr = glob.glob('/kaggle/input/**/train.parquet', recursive=True)[0]
pf = pq.ParquetFile(tr)
keys, embs, done = [], [], 0
for g in range(pf.num_row_groups):
    if time.time() - T0 > BUDGET: print('BUDGET STOP at row group', g, flush=True); break
    d = pf.read_row_group(g, columns=['inchikey14','precursor_mz','ms2_mzs','ms2_normalized_intensities','num_peaks']).to_pandas()
    d = d.sort_values('num_peaks', ascending=False).groupby('inchikey14').head(2)
    E = embed(d, f'g{g}')
    m = pd.DataFrame(E).groupby(d.inchikey14.values).mean()
    keys.append(m.index.values); embs.append(m.values.astype(np.float16)); done += len(d)
    print('GROUP', g, len(d), 'spectra', len(m), 'molecules', round((time.time()-T0)/60,1), 'min', flush=True)
K = np.concatenate(keys); X = np.concatenate(embs)
# a molecule can appear in more than one row group: average again
m = pd.DataFrame(X.astype(np.float32)).groupby(K).mean()
np.save(f'{W}/train_emb.npy', m.values.astype(np.float16)); pd.Series(m.index.values, name='inchikey14').to_csv(f'{W}/train_keys.csv', index=False)
print('TRAIN', m.shape, 'spectra embedded', done, round((time.time()-T0)/3600,2), 'h', flush=True)
sh('rm -rf /kaggle/working/dreams_src; du -sh /kaggle/working/*')
