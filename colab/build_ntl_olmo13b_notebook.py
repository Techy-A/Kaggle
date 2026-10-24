"""Build colab/ntl_olmo13b_train.ipynb: the free-Colab (T4) run of name-that-llm/lab/09_train_olmo13b.py.

The notebook embeds 09_train_olmo13b.py, olmo13b_member.py, 07_train_encoders.py and 08_blend_oof.py verbatim (no repo
access needed on Colab), so re-run this after changing any of them. Usage: python colab/build_ntl_olmo13b_notebook.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, '..', 'name-that-llm', 'lab')
FILES = ['09_train_olmo13b.py', 'olmo13b_member.py', '07_train_encoders.py', '08_blend_oof.py']

MD = """# NTL: OLMo-2-13B two-stage LoRA member, trained on free Colab (T4)

Runs `name-that-llm/lab/09_train_olmo13b.py` (embedded below): nishantdahal's 14B two-stage LoRA recipe on
allenai/OLMo-2-1124-13B (Apache-2.0), first 27 of 40 layers, base in 4-bit NF4 (QLoRA). Units, in this order:

1. `stage1`: 1 epoch of nishantdahal's public pool (pool_v4, CC BY-SA 4.0), about 20-27 h on one T4, so several
   sessions. It normally runs on Kaggle (2x T4, capped at 6.5 h); when it has, it is picked up from the dataset below and skipped.
2. `fold0` .. `fold4`: stage 2 on 4/5 of the prompts, scoring the held-out fifth (the OOF), about 1.5 h each.
3. `full`: stage 2 on all rows, writes `lora_olmo2_13b/` (the member the 0.636 notebook variant loads), about 2 h.
4. `merge`: the OOF summary (`oof_olmo13b.npz`, `summary_olmo13b.json`), seconds.

**What to do:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. Approve the Google Drive
mount, and allow notebook access to the `KAGGLE_API_TOKEN` secret (key icon in the left bar) if Colab asks.
Keep the tab open. The first run downloads about 37 GB of base weights (the shards of layers 0-26) to the runtime.

**Resuming.** Training saves a resume point to Google Drive (`MyDrive/ntl_olmo13b/`, about 0.5 GB) every 100 steps,
and every finished unit is kept there. If Colab disconnects, reconnect and **Run all** again: finished units are skipped
and the running one continues from its last save.

**Results** go to the private Kaggle dataset `aryanawasthiteykik/ntl-olmo13b-member` after every unit, together with
what is already there (the Kaggle units' outputs).
"""

CONFIG = """OUTPUTS = "aryanawasthiteykik/ntl-olmo13b-member"   # private; also where Kaggle-finished units come from
UNITS = ["stage1", "fold0", "fold1", "fold2", "fold3", "fold4", "full", "merge"]
WORK = "/content/drive/MyDrive/ntl_olmo13b"   # outputs and resume points, survive disconnects
CODE = "/content/ntl_olmo13b_code"            # the embedded scripts (re-written on every Run all)
DATA = "/content/data"                        # train.csv (competition) and pool_v4.parquet (nishantdahal)
BASE = "/content/olmo2_13b"                   # the base's config, tokenizer and the shards of layers < KEEP
HF_ID, KEEP = "allenai/OLMo-2-1124-13B", 27   # same weights as Kaggle's allenai/olmo-2/Transformers/1124-13b/1
EST_H = {"stage1": 24.0, "fold0": 1.6, "fold1": 1.6, "fold2": 1.6, "fold3": 1.6, "fold4": 1.6, "full": 2.0,
         "merge": 0.0}   # T4 estimates
"""

GPU = """# A GPU first, so a CPU runtime fails at once with the fix.
import shutil, subprocess

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True) if shutil.which("nvidia-smi") else None
if gpu is None or gpu.returncode != 0 or not gpu.stdout.strip():
    raise RuntimeError("No GPU in this runtime. Runtime → Change runtime type → T4 GPU, then Runtime → Run all again.")
GPU = gpu.stdout.strip().splitlines()[0]
print("GPU:", GPU)
"""

SETUP = """import json, os, sys, time
from importlib.metadata import PackageNotFoundError, version

from google.colab import drive, userdata

drive.mount("/content/drive")
try:
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
except Exception as e:
    raise RuntimeError("Add your Kaggle API token as the Colab secret KAGGLE_API_TOKEN (key icon in the left bar), "
                       "turn on its notebook access, then Run all again.") from e


def ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


need = [] if ver("transformers") == "5.18.0" else ["transformers==5.18.0"]   # the version 09 was checked with
need += [] if ver("bitsandbytes") == "0.50.2" else ["bitsandbytes==0.50.2"]   # the 0.636 notebook's wheel
need += [] if int((ver("kaggle") or "0").split(".")[0]) >= 2 else ["kaggle==2.2.4"]
need += [] if ver("accelerate") else ["accelerate"]
if need:
    print("pip install", *need)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
sh = lambda cmd: subprocess.run(cmd, shell=True, check=True)
os.makedirs(WORK, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
if not os.path.exists(f"{DATA}/train.csv"):
    sh(f"kaggle competitions download name-that-llm -f train.csv -p {DATA}")
    sh(f"cd {DATA} && (unzip -q -o train.csv.zip && rm -f train.csv.zip || true)")
if not os.path.exists(f"{DATA}/pool_v4.parquet"):
    sh(f"kaggle datasets download nishantdahal/ntl-handwriting-blend -f pool_v4.parquet -p {DATA}")
    sh(f"cd {DATA} && (unzip -q -o pool_v4.parquet.zip && rm -f pool_v4.parquet.zip || true)")
# Units already finished (on Kaggle or in an earlier session) come from the dataset; Drive copies win.
if subprocess.run(["kaggle", "datasets", "status", OUTPUTS], capture_output=True).returncode == 0:
    sh(f"kaggle datasets download {OUTPUTS} -p /content/member_ds --unzip -o")
    import zipfile
    for z in [f for f in os.listdir("/content/member_ds") if f.endswith(".zip")]:   # folders uploaded as zips
        zipfile.ZipFile(f"/content/member_ds/{z}").extractall(f"/content/member_ds/{z[:-4]}")
        os.remove(f"/content/member_ds/{z}")
    for f in os.listdir("/content/member_ds"):
        if not os.path.exists(f"{WORK}/{f}"):
            (shutil.copytree if os.path.isdir(f"/content/member_ds/{f}") else shutil.copy)(f"/content/member_ds/{f}",
                                                                                          f"{WORK}/{f}")
# The base: only the shards that hold the embeddings and layers 0..KEEP-1.
from huggingface_hub import hf_hub_download, snapshot_download
idx = json.load(open(hf_hub_download(HF_ID, "model.safetensors.index.json", local_dir=BASE)))["weight_map"]
shards = sorted({v for k, v in idx.items() if k.startswith("model.embed_tokens") or
                 (k.startswith("model.layers.") and int(k.split(".")[2]) < KEEP)})
snapshot_download(HF_ID, local_dir=BASE, allow_patterns=["*.json", "tokenizer*", "vocab*", "merges*", *shards])
print("train.csv, pool_v4.parquet ready | base", BASE, len(shards), "shards | outputs on Drive in", WORK,
      "|", sorted(os.listdir(WORK)))
"""

UPLOAD = """UP = "/content/upload"


def upload(msg, **progress):
    \"\"\"Upload every finished output (no resume points) and progress.json as a new version of OUTPUTS; 3 tries.\"\"\"
    if os.path.exists(UP):
        shutil.rmtree(UP)
    os.makedirs(UP)
    for f in sorted(os.listdir(WORK)):
        src = f"{WORK}/{f}"
        if os.path.isdir(src) and f != "state":
            shutil.copytree(src, f"{UP}/{f}")
        elif f.endswith((".npz", ".json")) and f != "progress.json":
            shutil.copy(src, UP)
    progress.update(event=msg, gpu=GPU, code_sha256=CODE_SHA, time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(progress, open(f"{UP}/progress.json", "w"), indent=1)
    json.dump({"title": OUTPUTS.split("/")[1], "id": OUTPUTS, "licenses": [{"name": "CC-BY-SA-4.0"}]},
              open(f"{UP}/dataset-metadata.json", "w"))
    for attempt in range(3):
        exists = subprocess.run(["kaggle", "datasets", "status", OUTPUTS], capture_output=True).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", UP, "-m", msg, "-r", "zip"] if exists
               else ["kaggle", "datasets", "create", "-p", UP, "-r", "zip"])
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"uploaded ({msg}) to https://www.kaggle.com/datasets/{OUTPUTS}", flush=True)
            return
        print(f"upload attempt {attempt + 1} failed: {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
        time.sleep(60)   # a just-created dataset takes a moment before it accepts versions
    print("WARNING: upload failed; the files are safe on Drive in", WORK, flush=True)


def done(unit):
    f = dict(stage1="summary_stage1.json", full="summary_full.json", merge="summary_olmo13b.json").get(unit, f"oof_{unit}.npz")
    return os.path.exists(f"{WORK}/{f}")
"""

RUN = """hours = {}
for unit in UNITS:
    if done(unit):
        print(unit, "already done")
        continue
    left = sum(EST_H[u] for u in UNITS if not done(u))
    print(f"\\n=== {unit} (about {EST_H[unit]} h on a T4; {left:.1f} h of units left)", flush=True)
    t = time.time()
    cmd = [sys.executable, "-u", f"{CODE}/09_train_olmo13b.py", "--unit", unit, "--train", f"{DATA}/train.csv",
           "--pool", f"{DATA}/pool_v4.parquet", "--base", BASE, "--out", WORK, "--micro-triplets", "3"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
                              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    for line in p.stdout:
        if "warn" not in line.lower() and "UNEXPECTED" not in line:
            print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"{unit} failed (exit {p.returncode}); Run all again to resume from the last save")
    hours[unit] = round((time.time() - t) / 3600, 2)
    summary = json.load(open(f"{WORK}/summary_olmo13b.json")) if os.path.exists(f"{WORK}/summary_olmo13b.json") else None
    upload(f"{unit} done", units_done=[u for u in UNITS if done(u)], hours_this_session=hours, summary=summary)
print("all units done:", [u for u in UNITS if done(u)])
if os.path.exists(f"{WORK}/summary_olmo13b.json"):
    print(open(f"{WORK}/summary_olmo13b.json").read()[:3000])
"""


def cell(kind, src):
    c = {'cell_type': kind, 'metadata': {}, 'source': src.splitlines(keepends=True)}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def main():
    write = ['# The scripts, embedded verbatim from the repo (name-that-llm/lab); CODE_SHA identifies this version.',
             'import hashlib, os', 'FILES = {}']
    digest = hashlib.sha256()
    for f in FILES:
        text = open(os.path.join(LAB, f)).read()
        digest.update(f.encode() + text.encode())
        write.append(f'FILES[{f!r}] = {text!r}')
    write += ['for name, text in FILES.items():',
              '    path = os.path.join(CODE, name)',
              '    os.makedirs(os.path.dirname(path), exist_ok=True)',
              '    open(path, "w").write(text)',
              f'CODE_SHA = {digest.hexdigest()[:16]!r}',
              'print("wrote", len(FILES), "files to", CODE, "| code", CODE_SHA)']
    nb = {'nbformat': 4, 'nbformat_minor': 0,
          'metadata': {'accelerator': 'GPU', 'colab': {'provenance': [], 'gpuType': 'T4'},
                       'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
          'cells': [cell('markdown', MD), cell('code', CONFIG), cell('code', GPU), cell('code', SETUP),
                    cell('code', '\n'.join(write) + '\n'), cell('code', UPLOAD), cell('code', RUN)]}
    out = os.path.join(HERE, 'ntl_olmo13b_train.ipynb')
    json.dump(nb, open(out, 'w'), indent=1)
    print('wrote', out, 'code', digest.hexdigest()[:16])


if __name__ == '__main__':
    main()
