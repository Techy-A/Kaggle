"""Build colab/ntl_atlas_train.ipynb: the free-Colab (T4) run of name-that-llm/lab/08_train_atlas.py.

The notebook embeds 08_train_atlas.py and the lab/atlas package verbatim (no repo access needed on Colab), so re-run
this after changing either. Usage: python colab/build_ntl_atlas_notebook.py
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, '..', 'name-that-llm', 'lab')
FILES = ['08_train_atlas.py'] + [f'atlas/{f}' for f in ['ntl_atlas/__init__.py', 'ntl_atlas/model.py',
                                                         'ntl_atlas/inference.py', 'ntl_rlcd/__init__.py',
                                                         'ntl_rlcd/data.py']]

MD = """# NTL: our own ATLAS bundle (Qwen3-8B-Base, 18 source adapters), trained on free Colab (T4)

Runs `name-that-llm/lab/08_train_atlas.py` (embedded below, with our `ntl_atlas` / `ntl_rlcd` package). It trains the
ATLAS method on train.csv only, with the public bundle's recipe (see `lab/atlas_rebuild_spec.md`). Units, in this order:

1. `fold0`: train on 4/5 of the prompts and score the held-out fifth. This is the first honest read, about 3-4 h.
2. `full`: train on all rows and write the drop-in `bundle/` that the 0.636 notebook variant loads, about 3.5-4.5 h.
3. `fold1` .. `fold4`, then `merge`: the full prompt-grouped OOF (`oof_atlas.npz`, `summary_atlas.json`), about 3-4 h each.

**What to do:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. Approve the Google Drive
mount, and allow notebook access to the `KAGGLE_API_TOKEN` secret (key icon in the left bar) if Colab asks.
Keep the tab open.

**Resuming.** Training saves a resume point to Google Drive (`MyDrive/ntl_atlas/`, about 1.4 GB) every 10 steps, and
every finished unit is kept there. If Colab disconnects, reconnect and **Run all** again: finished units are skipped and
the running one continues from its last save. To stop after the bundle, set `UNITS = ["fold0", "full"]`.

**Precision.** A T4 cannot hold the 16 GB fp16 base, so the base is loaded 4-bit (NF4) for training and OOF scoring. On a
GPU with at least 30 GB (A100) it uses the exact fp16 recipe. The bundle is scored on Kaggle with the fp16 base on two T4s.

**Results** go to the private Kaggle dataset `aryanawasthiteykik/ntl-atlas-rebuild` after every unit: `bundle/` (when
done), `oof_fold*.npz`, `summary_atlas.json` and `progress.json`.
"""

CONFIG = """OUTPUTS = "aryanawasthiteykik/ntl-atlas-rebuild"   # private, created on the first upload
UNITS = ["fold0", "full", "fold1", "fold2", "fold3", "fold4", "merge"]
MICRO = 8                                  # micro-batch: memory only, the effective batch stays 128
WORK = "/content/drive/MyDrive/ntl_atlas"  # outputs and resume points, survive disconnects
CODE = "/content/ntl_atlas_code"           # the embedded scripts (re-written on every Run all)
DATA = "/content/data"                     # train.csv from the competition
EST_H = {"fold0": 3.5, "full": 4.0, "fold1": 3.5, "fold2": 3.5, "fold3": 3.5, "fold4": 3.5, "merge": 0.0}   # T4 prior
"""

GPU = """# A GPU first, so a CPU runtime fails at once with the fix.
import shutil, subprocess

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                     capture_output=True, text=True) if shutil.which("nvidia-smi") else None
if gpu is None or gpu.returncode != 0 or not gpu.stdout.strip():
    raise RuntimeError("No GPU in this runtime. Runtime → Change runtime type → T4 GPU, then Runtime → Run all again.")
GPU = gpu.stdout.strip().splitlines()[0]
GPU_MB = int(GPU.split(",")[1])
QUANT = [] if GPU_MB >= 30000 else ["--quant", "nf4"]   # fp16 base needs >= 30 GB to train
print("GPU:", GPU, "| base:", "nf4" if QUANT else "fp16")
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


need = [] if ver("transformers") == "5.18.0" else ["transformers==5.18.0"]   # the version 08 was checked with
need += [] if int((ver("kaggle") or "0").split(".")[0]) >= 2 else ["kaggle==2.2.4"]
need += [p for p in ["accelerate", "bitsandbytes"] if ver(p) is None]
if need:
    print("pip install", *need)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
sh = lambda cmd: subprocess.run(cmd, shell=True, check=True)
os.makedirs(WORK, exist_ok=True)
if not os.path.exists(f"{DATA}/train.csv"):
    sh(f"kaggle competitions download name-that-llm -f train.csv -p {DATA}")
    sh(f"cd {DATA} && (unzip -q -o train.csv.zip && rm -f train.csv.zip || true)")
print("train.csv", os.path.getsize(f"{DATA}/train.csv"), "bytes | outputs on Drive in", WORK)
"""

UPLOAD = """import shutil

UP = "/content/upload"
log_lines = []


def upload(msg, **progress):
    \"\"\"Upload the finished outputs (no resume points) and progress.json as a new version of OUTPUTS; 3 tries.\"\"\"
    if os.path.exists(UP):
        shutil.rmtree(UP)
    os.makedirs(UP)
    for f in sorted(os.listdir(WORK)):
        src = f"{WORK}/{f}"
        if f == "bundle":
            shutil.copytree(src, f"{UP}/bundle")
        elif f.endswith((".npz", ".json")) and not f.endswith(".tmp.npz"):
            shutil.copy(src, UP)
    progress.update(event=msg, gpu=GPU, quant=QUANT, code_sha256=CODE_SHA,
                    time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
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
    if unit == "merge":
        return os.path.exists(f"{WORK}/summary_atlas.json")
    return os.path.exists(f"{WORK}/bundle/atlas_bundle.json" if unit == "full" else f"{WORK}/oof_{unit}.npz")
"""

RUN = """hours = {}
for unit in UNITS:
    if done(unit):
        print(unit, "already done")
        continue
    left = sum(EST_H[u] for u in UNITS if not done(u))
    print(f"\\n=== {unit} (about {EST_H[unit]} h on a T4; {left:.1f} h of units left)", flush=True)
    t = time.time()
    cmd = [sys.executable, "-u", f"{CODE}/08_train_atlas.py", "--unit", unit, "--train", f"{DATA}/train.csv",
           "--out", WORK, "--micro", str(MICRO), *QUANT]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env={**os.environ, "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    for line in p.stdout:
        if "warn" not in line.lower():
            print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"{unit} failed (exit {p.returncode}); Run all again to resume from the last save")
    hours[unit] = round((time.time() - t) / 3600, 2)
    summary = json.load(open(f"{WORK}/summary_atlas.json")) if os.path.exists(f"{WORK}/summary_atlas.json") else None
    upload(f"{unit} done", units_done=[u for u in UNITS if done(u)], hours_this_session=hours, summary=summary)
print("all units done:", [u for u in UNITS if done(u)])
if os.path.exists(f"{WORK}/summary_atlas.json"):
    print(open(f"{WORK}/summary_atlas.json").read())
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
    out = os.path.join(HERE, 'ntl_atlas_train.ipynb')
    json.dump(nb, open(out, 'w'), indent=1)
    print('wrote', out, 'code', digest.hexdigest()[:16])


if __name__ == '__main__':
    main()
