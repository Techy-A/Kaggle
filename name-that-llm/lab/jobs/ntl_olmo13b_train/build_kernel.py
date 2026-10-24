"""Build the Kaggle job aryanawasthiteykik/ntl-olmo13b-train for one unit of lab/09_train_olmo13b.py (embedded verbatim).

Units (one kernel version each, each under 12 h, 2x T4):
- stage1: both T4s under torchrun (data parallel, NF4 base on each), stage 1 on the pool, capped at STAGE1_HOURS of
  training (the cosine schedule is shortened to fit), then train scored by the stage-1 model.
- stage2: needs stage1/ in the attached dataset aryanawasthiteykik/ntl-olmo13b-member; GPU 0 runs fold0-2, GPU 1 fold3,
  fold4 and full (the shipped member, reload-checked with the notebook's olmo_features), then merge.
The outputs land in /kaggle/working/ntl-olmo13b-member/; relay.py turns them into a version of the private dataset.
It fails fast without a GPU, and uses the latest GPU environment (no pinned image: a pinned gcr.io/kaggle-images digest
is the CPU image). bitsandbytes comes from the wheel the 0.636 notebook installs (dhruvtibarewal/nlm-offline-wheels).

Usage: python build_kernel.py stage1|stage2   (then: kaggle kernels push -p .)
Output: ntl-olmo13b-train.ipynb, kernel-metadata.json (this directory).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, '..', '..')
FILES = ['09_train_olmo13b.py', 'olmo13b_member.py', '07_train_encoders.py', '08_blend_oof.py']
STAGE1_HOURS = 8.0   # + ~0.6 h of loading, filtering and scoring: the stage1 kernel stays near 9 h
MEMBER_DS = 'aryanawasthiteykik/ntl-olmo13b-member'

GPU_CHECK = '''# Fail fast: pushes have landed on CPU-only machines ("Torch not compiled with CUDA").
import subprocess, torch, transformers
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout or 'nvidia-smi: none', flush=True)
print('gpus', torch.cuda.device_count(), 'torch', torch.__version__, 'transformers', transformers.__version__, flush=True)
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise RuntimeError('NO GPU in this session: stopping before any work')'''

SETUP = '''# bitsandbytes: the same wheel the 0.636 notebook installs, so training and inference quantize alike.
import glob, subprocess, sys
whl = sorted(glob.glob('/kaggle/input/**/bitsandbytes-*.whl', recursive=True))
r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-q', *(whl[:1] or ['bitsandbytes==0.50.2'])],
                   capture_output=True, text=True)
print(r.returncode, r.stderr[-500:])
import importlib.metadata as md
print({p: md.version(p) for p in ['torch', 'transformers', 'bitsandbytes', 'accelerate', 'safetensors']}, flush=True)'''

RUN = '''import glob, json, os, shutil, subprocess, sys, time
import torch
UNIT = __UNIT__
def find(p):
    hits = sorted(glob.glob(f'/kaggle/input/**/{p}', recursive=True))
    assert hits, p
    return hits[0]
def olmo_dir():
    for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
        j = json.load(open(c))
        if j.get('model_type') == 'olmo2' and j.get('hidden_size') == 5120 and j.get('num_hidden_layers') == 40:
            return os.path.dirname(c)
    raise FileNotFoundError('attach allenai/olmo-2/Transformers/1124-13b/1')
TRAIN = [f'{r}/train.csv' for r in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm']
         if os.path.exists(f'{r}/train.csv')][0]   # the competition's, not a dataset's
POOL, BASE = find('pool_v4.parquet'), olmo_dir()
OUT = '/kaggle/working/ntl-olmo13b-member'
os.makedirs(OUT, exist_ok=True)
NG = torch.cuda.device_count()
ARGS = ['--train', TRAIN, '--pool', POOL, '--base', BASE, '--out', OUT]
ENV = dict(os.environ, TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
           HF_HUB_DISABLE_PROGRESS_BARS='1')
T0 = time.time()
print('base', BASE, '| gpus', NG, flush=True)

def launch(cmd, gpu, name):
    env = dict(ENV) if gpu is None else dict(ENV, CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen(cmd, env=env, stdout=open(f'/kaggle/working/log_{name}.txt', 'w'), stderr=subprocess.STDOUT)

def tail(name, n=4):
    keep = [l for l in open(f'/kaggle/working/log_{name}.txt', errors='replace').read().splitlines()
            if l.strip() and 'UNEXPECTED' not in l and 'warn' not in l.lower()]
    return '\\n  '.join(keep[-n:])

def wait(procs):
    while any(p.poll() is None for p in procs.values()):
        time.sleep(600)
        for n in procs:
            print(f'[{(time.time() - T0) / 3600:.2f} h] {n}:\\n  {tail(n)}', flush=True)
    for n, p in procs.items():
        print(f'===== {n} (exit {p.returncode})\\n  {tail(n, 30)}', flush=True)
    assert all(p.returncode == 0 for p in procs.values()), {n: p.returncode for n, p in procs.items()}

if UNIT == 'stage1':
    cmd = ([sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(NG)] if NG > 1 else [sys.executable])
    wait({'stage1': launch(cmd + ['09_train_olmo13b.py', '--unit', 'stage1', '--stage1-hours', '__HOURS__', *ARGS],
                           None, 'stage1')})
else:   # stage 1 comes from the member dataset
    src = os.path.dirname(os.path.dirname(find('stage1/lora_heads.safetensors')))
    for f in ['stage1', 'stage1_train.npz', 'summary_stage1.json']:
        (shutil.copytree if os.path.isdir(f'{src}/{f}') else shutil.copy)(f'{src}/{f}', f'{OUT}/{f}')
    plan = [['fold0', 'fold1', 'fold2'], ['fold3', 'fold4', 'full']] if NG > 1 else [
        ['fold0', 'fold1', 'fold2', 'fold3', 'fold4', 'full']]
    wait({f'gpu{g}': launch([sys.executable, '09_train_olmo13b.py', '--unit', ','.join(u), *ARGS], g, f'gpu{g}')
          for g, u in enumerate(plan)})
    subprocess.run([sys.executable, '09_train_olmo13b.py', '--unit', 'merge', *ARGS], check=True)
shutil.rmtree(f'{OUT}/state', ignore_errors=True)   # resume points are not outputs
for f in sorted(glob.glob(f'{OUT}/summary_*.json')):
    print(f, open(f).read()[:1500], flush=True)
print(subprocess.run(['du', '-sh', OUT], capture_output=True, text=True).stdout, f'{(time.time() - T0) / 3600:.2f} h')'''


def cell(kind, src):
    c = dict(cell_type=kind, metadata={}, source=src.splitlines(True))
    if kind == 'code':
        c.update(outputs=[], execution_count=None)
    return c


def main():
    unit = sys.argv[1] if len(sys.argv) > 1 else 'stage1'
    assert unit in ('stage1', 'stage2'), unit
    cells = [cell('markdown', f'# NTL OLMo-2-13B two-stage LoRA, unit `{unit}`\n\nnishantdahal\'s 14B two-stage LoRA '
                              'recipe on allenai/OLMo-2-1124-13B (Apache-2.0), first 27 of 40 layers, NF4 QLoRA; stage 1 '
                              'on nishantdahal/ntl-handwriting-blend pool_v4 (CC BY-SA 4.0), stage 2 on train with '
                              'prompt-grouped 5-fold OOF. Code: lab/09_train_olmo13b.py, embedded verbatim.'),
             cell('code', GPU_CHECK), cell('code', SETUP)]
    for f in FILES:
        cells.append(cell('code', f'%%writefile {f}\n' + open(os.path.join(LAB, f)).read()))
    cells.append(cell('code', RUN.replace('__UNIT__', repr(unit)).replace('__HOURS__', str(STAGE1_HOURS))))
    nb = dict(nbformat=4, nbformat_minor=4, cells=cells,
              metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')))
    json.dump(nb, open(os.path.join(HERE, 'ntl-olmo13b-train.ipynb'), 'w'), indent=1)
    meta = {
        'id': 'aryanawasthiteykik/ntl-olmo13b-train', 'title': 'ntl-olmo13b-train', 'code_file': 'ntl-olmo13b-train.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': True, 'machine_shape': 'NvidiaTeslaT4', 'docker_image_pinning_type': 'latest',
        'dataset_sources': ['nishantdahal/ntl-handwriting-blend', 'dhruvtibarewal/nlm-offline-wheels'] +
                           ([MEMBER_DS] if unit == 'stage2' else []),
        'competition_sources': ['name-that-llm'], 'kernel_sources': [],
        'model_sources': ['allenai/olmo-2/Transformers/1124-13b/1']}
    json.dump(meta, open(os.path.join(HERE, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', HERE, 'unit', unit)


if __name__ == '__main__':
    main()
