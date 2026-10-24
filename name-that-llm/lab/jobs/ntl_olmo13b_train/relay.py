"""Relay a finished ntl-olmo13b-train unit into the private dataset aryanawasthiteykik/ntl-olmo13b-member (local, CPU).

What: downloads the kernel's output folder ntl-olmo13b-member/ into a fresh directory, merges it into a local mirror of
the dataset (files already there are kept, so stage1/ survives the stage2 version) and uploads the mirror as a new
dataset version (CC-BY-SA-4.0, as the pool it was trained on).
Why: a Kaggle kernel holds no API token here, so the kernel writes outputs and this script publishes them; the next
unit (or the Colab notebook) reads them from the dataset.
Output: the mirror dir (default /home/user/work/ds_ntl_olmo13b_member) and a new dataset version.
Usage: python relay.py --cli /path/to/kaggle_www.py [--mirror DIR]   (KAGGLE_API_TOKEN in the environment)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

KERNEL, DATASET = 'aryanawasthiteykik/ntl-olmo13b-train', 'aryanawasthiteykik/ntl-olmo13b-member'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cli', required=True)
    ap.add_argument('--mirror', default='/home/user/work/ds_ntl_olmo13b_member')
    a = ap.parse_args()
    kg = lambda *x, **kw: subprocess.run([sys.executable, a.cli, *x], **kw)
    dl = f'{a.mirror}_dl_{time.strftime("%m%d%H%M")}'
    os.makedirs(dl)
    kg('kernels', 'output', KERNEL, '-p', dl, check=True)
    src = os.path.join(dl, 'ntl-olmo13b-member')
    assert os.path.isdir(src), os.listdir(dl)
    os.makedirs(a.mirror, exist_ok=True)
    for f in os.listdir(src):
        if f == 'state':
            continue
        if os.path.isdir(os.path.join(src, f)):
            shutil.copytree(os.path.join(src, f), os.path.join(a.mirror, f), dirs_exist_ok=True)
        else:
            shutil.copy(os.path.join(src, f), a.mirror)
    for f in os.listdir(dl):   # keep the logs next to the download, drop the big copy
        if f != 'ntl-olmo13b-member':
            print('log:', os.path.join(dl, f))
    shutil.rmtree(src)
    json.dump({'title': DATASET.split('/')[1], 'id': DATASET, 'licenses': [{'name': 'CC-BY-SA-4.0'}]},
              open(os.path.join(a.mirror, 'dataset-metadata.json'), 'w'))
    exists = kg('datasets', 'status', DATASET, capture_output=True).returncode == 0
    files = sorted(f for f in os.listdir(a.mirror) if f != 'dataset-metadata.json')
    msg = 'relay: ' + ', '.join(files)
    cmd = ['datasets', 'version', '-p', a.mirror, '-m', msg[:200], '-r', 'zip'] if exists else \
        ['datasets', 'create', '-p', a.mirror, '-r', 'zip']
    kg(*cmd, check=True)
    print('uploaded', DATASET, files)


if __name__ == '__main__':
    main()
