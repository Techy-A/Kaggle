"""Build /home/user/work/push_ntl664_q8pool: our exact 0.664 copy (push_ntl664b) + the q8pool member at a small weight.

What: three edits to an otherwise byte-identical copy of /home/user/work/push_ntl664b (zbigniewkirchner's 0.664 "One
Swap" on its pinned image):
1. inputs cell: AQ8POOL = the lora_q8pool folder of aryanawasthiteykik/ntl-q8pool-member (fails closed if missing);
2. members cell: one more l24 spec after ll8v5: Qwen3-8B-Base (B8, already attached), adapter [AQ8POOL], 24 layers,
   max_len 385 (384 tokens + the end token, as trained), no_final_norm, eos_id 151643, dp; weight --w (default 0.5;
   q8v5 is 2.0, ll8v5 1.0), centred twin --twin (default 0.35, as every l24 member), no Gaussian head;
3. infer.py: the q8pool entries are left out of wsum (the member count that divides the blend before Sinkhorn), so
   adding the member does not re-temper the other members' blend against the comp stack and ATLAS.
kernel-metadata.json: new private id, + the member dataset. A markdown cell at the top says what changed.
Why: the new member enters BESIDE the existing ones (753177: replacing a member lost 0.004), everything else unchanged.
Output: the push dir (not pushed, not submitted). Usage: python build_q8pool_664.py [--w 0.5] [--twin 0.35]
"""
import argparse
import glob
import json
import os
import shutil

SRC, DST = '/home/user/work/push_ntl664b', '/home/user/work/push_ntl664_q8pool'
MEMBER_DS = 'aryanawasthiteykik/ntl-q8pool-member'
EDITS = [
    ("ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]\n",
     "ALL8V5 = [adir(f'll8v5_f{k}') for k in range(5)] + [adir('ll8v5')]\n"
     "AQ8POOL = os.path.dirname(glob.glob('/kaggle/input/**/lora_q8pool/meta.json', recursive=True)[0])   # ours: q8pool\n"
     "print('q8pool', AQ8POOL, json.load(open(AQ8POOL + '/meta.json'))['cfg']['stage1'])\n"),
    ('"twin": 0.35, "dp": True},\n  {"name": "n14b"',
     '"twin": 0.35, "dp": True},\n'
     '  {"name": "q8pool", "type": "l24", "path": B8, "adapters": [AQ8POOL], "keep": 24, "layers": L, "w": __W__, "bs": 16, '
     '"max_len": 385, "twin": __TWIN__, "no_final_norm": True, "eos_id": 151643, "dp": True},   # ours (aryanawasthiteykik)\n'
     '  {"name": "n14b"'),
    ("    wsum = sum(abs(v).max() > 0 for v in LP.values())\n",
     "    wsum = sum(abs(v).max() > 0 for k_, v in LP.items() if not k_.startswith('q8pool'))   # ours: the added member does not re-temper the blend\n"),
]
NOTE = ('**Variant of our exact copy** of [zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public]'
        '(https://www.kaggle.com/code/zbigniewkirchner/0-664-one-swap-198k-clean-pool-fully-public) (public 0.664). '
        'One member added, nothing else changed: **q8pool**, our two-stage LoRA24 on Qwen3-8B-Base (stage 1 on '
        'crtproduction/name-that-llm-open-pool minus Kaggle-Benchmarks-proxy rows and answers to train prompts, plus '
        'nishantdahal pool_v4; stage 2 on train), weight __W__, centred twin __TWIN__, left out of the member count '
        'that divides the blend. All credit for the base to its author and the authors credited below.\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=float, default=0.5)
    ap.add_argument('--twin', type=float, default=0.35)
    a = ap.parse_args()
    nb_name = os.path.basename(glob.glob(f'{SRC}/*.ipynb')[0])
    nb = json.load(open(os.path.join(SRC, nb_name)))
    done = [0] * len(EDITS)
    for c in nb['cells']:
        s = ''.join(c['source'])
        for i, (old, new) in enumerate(EDITS):
            if old in s:
                assert s.count(old) == 1, old
                s = s.replace(old, new.replace('__W__', repr(a.w)).replace('__TWIN__', repr(a.twin)))
                done[i] += 1
        c['source'] = s.splitlines(True)
    assert done == [1] * len(EDITS), done
    nb['cells'].insert(0, dict(cell_type='markdown', metadata={}, source=NOTE.replace('__W__', repr(a.w)).replace(
        '__TWIN__', repr(a.twin)).splitlines(True)))
    if os.path.exists(DST):
        shutil.rmtree(DST)
    os.makedirs(DST)
    json.dump(nb, open(os.path.join(DST, nb_name), 'w'), indent=1)
    meta = json.load(open(os.path.join(SRC, 'kernel-metadata.json')))
    meta.update(id='aryanawasthiteykik/ntl-664-q8pool', title='NTL 664 q8pool')
    meta['dataset_sources'] = meta['dataset_sources'] + [MEMBER_DS]
    json.dump(meta, open(os.path.join(DST, 'kernel-metadata.json'), 'w'), indent=1)
    print('wrote', DST, 'w', a.w, 'twin', a.twin)


if __name__ == '__main__':
    main()
