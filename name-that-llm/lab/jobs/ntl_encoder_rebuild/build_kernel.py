"""Build the Kaggle job aryanawasthiteykik/ntl-encoder-rebuild: a notebook that embeds lab/07_train_encoders.py verbatim.

What it does on Kaggle (T4 x2, internet on for the HF backbones):
1. Fail fast with no GPU.
2. Train modernbert on GPU 0 and desklib on GPU 1 in parallel.
3. Write /kaggle/working/ntl-ours-rebuild/ in the notebooks' layout.
4. Run the notebooks' own enc_infer.py, verbatim, on the saved dirs.
It uses the 0.627 notebook's pinned docker image so the checkpoints load in that image.

Usage: python build_kernel.py   (then push with: kaggle kernels push -p .)
Output: ntl-encoder-rebuild.ipynb, kernel-metadata.json (this directory).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = open(os.path.join(HERE, '..', '..', '07_train_encoders.py')).read()
ENC_INFER = '''"""One encoder on one T4: fp32 weights + fp16 autocast (T4 tensor cores), length-sorted dynamic padding,
fp32 retry for any batch that overflows in fp16. Writes raw logits + the head's label order."""
import sys, json, os, time, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
path, data, out = sys.argv[1:4]; t0 = time.time()
test = pd.read_csv(data); texts = test.text.fillna('').astype(str).tolist()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).cuda().eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
ids = tok(texts, truncation=True, max_length=512)['input_ids']; order = np.argsort([len(x) for x in ids])
logits = np.zeros((len(texts), len(labels)), np.float32); retried = 0
with torch.inference_mode():
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]; b = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to('cuda')
        with torch.autocast('cuda', dtype=torch.float16):
            lg = model(**b).logits.float()
        if not torch.isfinite(lg).all():
            lg = model(**b).logits.float(); retried += 1
        logits[idx] = lg.cpu().numpy()
assert np.isfinite(logits).all()
np.savez(out, logits=logits, labels=np.array(labels, dtype=str), ids=test['id'].astype(str).to_numpy(dtype=str))   # unicode, not object
print(f'{os.path.basename(path)}: {len(texts)} rows in {time.time() - t0:.0f}s, fp32 retries {retried}', flush=True)
'''

GPU_CHECK = '''# Fail fast: pushes made while other sessions were busy have landed on CPU-only machines.
import subprocess, torch, transformers
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'nvidia-smi: none', flush=True)
print('gpus', torch.cuda.device_count(), 'torch', torch.__version__, 'transformers', transformers.__version__, flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise RuntimeError('NO GPU in this session: stopping before any work')'''

RUN = '''import glob, os, subprocess, sys, time
import numpy as np, torch
def find(p):
    return sorted(glob.glob(f'/kaggle/input/**/{p}', recursive=True))[0]
TRAIN, TEST, POOL = find('train.csv'), find('test.csv'), find('pool_v4.parquet')
OUT = '/kaggle/working/ntl-ours-rebuild'
os.makedirs(OUT, exist_ok=True)
HOURS = dict(modernbert=2.5, desklib=3.0)   # stage-1 wall-time caps: the whole job stays under ~4.5 h
NAMES = ['modernbert', 'desklib']
T0 = time.time()

def launch(name, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM='false')
    return subprocess.Popen([sys.executable, '07_train_encoders.py', '--member', name, '--train', TRAIN, '--pool', POOL,
                             '--out', OUT, '--stage1-hours', str(HOURS[name])],
                            env=env, stdout=open(f'/kaggle/working/log_{name}.txt', 'w'), stderr=subprocess.STDOUT)

def tail(name, n=3):
    lines = [l for l in open(f'/kaggle/working/log_{name}.txt', errors='replace').read().splitlines() if l.strip()]
    return '\\n  '.join(lines[-n:])

groups = [NAMES] if torch.cuda.device_count() >= 2 else [[n] for n in NAMES]   # one GPU: one after the other
rc = {}
for group in groups:
    procs = {n: launch(n, i) for i, n in enumerate(group)}
    while any(p.poll() is None for p in procs.values()):
        time.sleep(600)
        for n in group:
            print(f'[{(time.time() - T0) / 3600:.2f} h] {n}:\\n  {tail(n)}', flush=True)
    rc.update({n: p.returncode for n, p in procs.items()})
for n in NAMES:
    print(f'===== {n} (exit {rc[n]})\\n  {tail(n, 25)}', flush=True)
assert all(v == 0 for v in rc.values()), rc
print(f'training done in {(time.time() - T0) / 3600:.2f} h')'''

PACKAGE = '''# The notebooks only look for light_members.npz to find this folder; no notebook reads its contents.
import json, os, subprocess, sys
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
LABELS = sorted(json.load(open(f'{OUT}/modernbert/config.json'))['label2id'])
np.savez(f'{OUT}/light_members.npz', labels=np.array(LABELS, dtype=str))
train = pd.read_csv(TRAIN)
for i, n in enumerate(NAMES):   # the notebooks' enc_infer.py, verbatim, on the saved dirs
    for split, path in [('train', TRAIN), ('test', TEST)]:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(i % max(1, torch.cuda.device_count())))
        r = subprocess.run([sys.executable, 'enc_infer.py', f'{OUT}/{n}', path, f'/kaggle/working/{n}_{split}_logits.npz'],
                           env=env, capture_output=True, text=True)
        print(r.stdout[-500:], r.stderr[-1500:] if r.returncode else '', flush=True)
        assert r.returncode == 0, (n, split)
    z = np.load(f'/kaggle/working/{n}_train_logits.npz')
    lab = [str(l) for l in z['labels']]
    pred = np.array(lab)[z['logits'].argmax(1)]
    s = json.load(open(f'{OUT}/summary_{n}.json'))
    print(f"{n}: enc_infer on train (in-sample) bacc {balanced_accuracy_score(train.model, pred):.4f}; "
          f"OOF bacc {s['oof_bacc']:.4f}; stage-1-only {s['stage1_only_bacc']:.4f}; total {s['total_secs'] / 3600:.2f} h", flush=True)
print(subprocess.run(['du', '-sh', OUT], capture_output=True, text=True).stdout)'''


def cell(kind, src):
    c = dict(cell_type=kind, metadata={}, source=src.splitlines(True))
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


nb = dict(nbformat=4, nbformat_minor=4, metadata=dict(kernelspec=dict(name='python3', display_name='Python 3',
                                                                       language='python')),
          cells=[cell('markdown', '# NTL encoder rebuild\n\nTrains our own ModernBERT-large and desklib DeBERTa-v3-large '
                                  'members (two-stage: nishantdahal pool_v4, then train; prompt-grouped 5-fold OOF). '
                                  'Code: lab/07_train_encoders.py, embedded verbatim.'),
                 cell('code', GPU_CHECK),
                 cell('code', '%%writefile 07_train_encoders.py\n' + SCRIPT),
                 cell('code', '%%writefile enc_infer.py\n' + ENC_INFER),
                 cell('code', RUN),
                 cell('code', PACKAGE)])
json.dump(nb, open(os.path.join(HERE, 'ntl-encoder-rebuild.ipynb'), 'w'), indent=1)
meta = {
    'id': 'aryanawasthiteykik/ntl-encoder-rebuild', 'title': 'ntl-encoder-rebuild', 'code_file': 'ntl-encoder-rebuild.ipynb',
    'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
    'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4',
    'docker_image': 'gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435',
    'docker_image_pinning_type': 'original',
    'dataset_sources': ['nishantdahal/ntl-handwriting-blend'], 'competition_sources': ['name-that-llm'],
    'kernel_sources': [], 'model_sources': []}
json.dump(meta, open(os.path.join(HERE, 'kernel-metadata.json'), 'w'), indent=1)
print('wrote', HERE)
