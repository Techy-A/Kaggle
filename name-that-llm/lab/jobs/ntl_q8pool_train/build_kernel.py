"""Build the Kaggle jobs for the q8pool member: Qwen3-8B-Base two-stage LoRA24, stage 1 on the cleaned Open Pool + pool_v4.

What: one private kernel per unit, code embedded verbatim (lab/09_train_olmo13b.py with --quant fp16, olmo13b_member.py,
07/08 helpers, 14_build_stage1_pool.py); reuses the OLMo job's GPU check, bitsandbytes setup and cell helper.
- prep   (CPU, no GPU quota): step 14 only; prints the Open Pool's columns and the pool counts.
- stage1 (2x T4, torchrun): step 14, then stage 1 capped at STAGE1_HOURS of training, then train scored by stage 1.
- stage2 (2x T4, one unit per GPU): stage 1 from the member dataset; GPU 0 trains fold0 (stage 2 on 4/5 of the
  prompts, GroupKFold(5) on prompt_id = the lab's folds, scores the held-out fifth), GPU 1 trains full (the shipped
  member, exported as lora_q8pool/ and reload-checked). No merge (only fold0 is trained: the GPU budget).
Recipe = the l24 members of the 0.663/0.664 notebooks: first 24 of 36 layers, pooled at 2,4,6,8,12,16,20,24, plain +
triplet-centred heads, LoRA r16/a32 on every linear layer, fp16 base (what l24_predict merges into), max 384 tokens + the
<|endoftext|> end token, no final norm (the notebook spec sets no_final_norm and eos_id, as for nishant's n14b).
Outputs land in /kaggle/working/ntl-q8pool-member/; ../ntl_olmo13b_train/relay.py --kernel ... --dataset
aryanawasthiteykik/ntl-q8pool-member --folder ntl-q8pool-member publishes them.

Usage: python build_kernel.py prep|stage1|stage2   (then: kaggle kernels push -p ../ntl_q8pool_<unit>)
Output: ../ntl_q8pool_<unit>/ (notebook + kernel-metadata.json) for kernel aryanawasthiteykik/ntl-q8pool-<unit>.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..', 'ntl_olmo13b_train'))
import build_kernel as ob  # noqa: E402  (the OLMo job: GPU_CHECK, SETUP, cell)

FILES = ['09_train_olmo13b.py', 'olmo13b_member.py', '07_train_encoders.py', '08_blend_oof.py', '14_build_stage1_pool.py']
STAGE1_HOURS = 7.2   # training only; + ~0.5 h load/tokenise/leak filter/scoring: the kernel stays under ~8 h
MEMBER_DS = 'aryanawasthiteykik/ntl-q8pool-member'
MODEL_SRC = 'qwen-lm/qwen-3/Transformers/8b-base/1'

RUN = '''import glob, json, os, shutil, subprocess, sys, time
UNIT = __UNIT__
def find(p):
    hits = sorted(glob.glob(f'/kaggle/input/**/{p}', recursive=True))
    assert hits, p
    return hits[0]
TRAIN = [f'{r}/train.csv' for r in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm']
         if os.path.exists(f'{r}/train.csv')][0]
OUT = '/kaggle/working/ntl-q8pool-member'
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
if UNIT in ('prep', 'stage1'):
    subprocess.run([sys.executable, '14_build_stage1_pool.py', '--open-pool', find('ntl_open_pool.parquet'),
                    '--v4', find('pool_v4.parquet'), '--train', TRAIN, '--out', OUT], check=True)
if UNIT != 'prep':
    import torch
    def qwen_dir():
        for c in sorted(glob.glob('/kaggle/input/**/config.json', recursive=True)):
            j = json.load(open(c))
            if j.get('model_type') == 'qwen3' and j.get('hidden_size') == 4096 and j.get('num_hidden_layers') == 36 \\
                    and '8b-base' in c.lower():
                return os.path.dirname(c)
        raise FileNotFoundError('attach __MODEL__')
    BASE = qwen_dir()
    NG = torch.cuda.device_count()
    ARGS = ['--train', TRAIN, '--pool', f'{OUT}/pool_stage1.parquet', '--base', BASE, '--out', OUT, '--quant', 'fp16',
            '--keep', '24', '--layers', '2,4,6,8,12,16,20,24', '--max-len', '384', '--micro-triplets', '2',
            '--eval-bs', '16', '--member', 'lora_q8pool', '--base-name', 'Qwen/Qwen3-8B-Base', '--base-class', 'Qwen3Model',
            '--pool-note', 'crtproduction/name-that-llm-open-pool (latest) minus flag_kb_proxy and flag_train_prompt_regen, '
            '+ nishantdahal pool_v4 minus Open Pool-flagged texts, exact-text dedup vs train, minus content cosine >= 0.3 to train']
    ENV = dict(os.environ, TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               HF_HUB_DISABLE_PROGRESS_BARS='1')
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
            time.sleep(300)
            for n in procs:
                print(f'[{(time.time() - T0) / 3600:.2f} h] {n}:\\n  {tail(n)}', flush=True)
        for n, p in procs.items():
            print(f'===== {n} (exit {p.returncode})\\n  {tail(n, 30)}', flush=True)
        assert all(p.returncode == 0 for p in procs.values()), {n: p.returncode for n, p in procs.items()}
    if UNIT == 'stage1':
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--nproc_per_node', str(NG)] if NG > 1 else [sys.executable]
        wait({'stage1': launch(cmd + ['09_train_olmo13b.py', '--unit', 'stage1', '--stage1-hours', '__HOURS__', *ARGS],
                               None, 'stage1')})
    else:   # stage2: stage 1 from the member dataset, fold0 and full side by side
        src = os.path.dirname(os.path.dirname(find('stage1/lora_heads.safetensors')))
        for f in ['stage1', 'stage1_train.npz', 'summary_stage1.json', 'pool_stage1_summary.json']:
            (shutil.copytree if os.path.isdir(f'{src}/{f}') else shutil.copy)(f'{src}/{f}', f'{OUT}/{f}')
        plan = [['fold0'], ['full']] if NG > 1 else [['fold0', 'full']]
        wait({f'gpu{g}': launch([sys.executable, '09_train_olmo13b.py', '--unit', ','.join(u), *ARGS], g, f'gpu{g}')
              for g, u in enumerate(plan)})
    shutil.rmtree(f'{OUT}/state', ignore_errors=True)   # resume points are not outputs
if os.path.exists(f'{OUT}/pool_stage1.parquet') and UNIT != 'prep':
    os.remove(f'{OUT}/pool_stage1.parquet')   # rebuilt from the public datasets; not republished
for f in sorted(glob.glob(f'{OUT}/*summary*.json')):
    print(f, open(f).read()[:3000], flush=True)
print(subprocess.run(['du', '-sh', OUT], capture_output=True, text=True).stdout, f'{(time.time() - T0) / 3600:.2f} h')'''


def main():
    unit = sys.argv[1] if len(sys.argv) > 1 else 'stage1'
    assert unit in ('prep', 'stage1', 'stage2'), unit
    out = os.path.join(HERE, '..', f'ntl_q8pool_{unit}')
    name = f'ntl-q8pool-{unit}'
    os.makedirs(out, exist_ok=True)
    cells = [ob.cell('markdown', f'# NTL q8pool member, unit `{unit}`\n\nTwo-stage LoRA24 on Qwen3-8B-Base (Apache-2.0), '
                                 'fp16 trunk of 24 of 36 layers; stage 1 on crtproduction/name-that-llm-open-pool minus '
                                 'Kaggle-Benchmarks-proxy rows and answers to train prompts, plus nishantdahal pool_v4 '
                                 '(both CC BY-SA 4.0); stage 2 on train (fold0 held out for an honest check, and full). '
                                 'Code: lab/09_train_olmo13b.py, lab/14_build_stage1_pool.py, embedded verbatim.')]
    if unit != 'prep':
        cells += [ob.cell('code', ob.GPU_CHECK)]
    for f in FILES:
        cells.append(ob.cell('code', f'%%writefile {f}\n' + open(os.path.join(LAB, f)).read()))
    run = RUN.replace('__UNIT__', repr(unit)).replace('__HOURS__', str(STAGE1_HOURS)).replace('__MODEL__', MODEL_SRC)
    cells.append(ob.cell('code', run))
    nb = dict(nbformat=4, nbformat_minor=4, cells=cells,
              metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python')))
    json.dump(nb, open(os.path.join(out, name + '.ipynb'), 'w'), indent=1)
    meta = {
        'id': f'aryanawasthiteykik/{name}', 'title': name, 'code_file': name + '.ipynb',
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': unit != 'prep',
        'enable_tpu': False, 'enable_internet': True, 'docker_image_pinning_type': 'latest',
        'dataset_sources': ['crtproduction/name-that-llm-open-pool', 'nishantdahal/ntl-handwriting-blend'] +
                           ([MEMBER_DS] if unit == 'stage2' else []),
        'competition_sources': ['name-that-llm'], 'kernel_sources': [],
        'model_sources': [] if unit == 'prep' else [MODEL_SRC]}
    if unit != 'prep':
        meta['machine_shape'] = 'NvidiaTeslaT4'
    json.dump(meta, open(os.path.join(out, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', out, 'unit', unit)


if __name__ == '__main__':
    main()
