# %% [markdown]
# # SHL: DeBERTa-v3-large fine-tune "the winners' way", on free Colab (T4)
#
# Trains one arm of `src/06_finetune_text.py` from the SHL repo (branch `claude/tender-cray-a5xjla`): the main
# model's fine-tune recipe (`--crops --drop-zero --epochs 2`, the member of submission 33) with two changes,
#
# 1. `--dropout0`: every dropout in the DeBERTa config set to 0 (ELL 3rd place: "[High Impact]" for regression);
# 2. `--head bins`: a softmax over the 11 label bins 0, 0.5, ..., 5; the prediction is their expected value,
#    trained with MSE on it (Ma 2025), never rounded.
#
# 5 speaker-grouped folds for seeds 0, 1, 2, in that order, so seed 0 alone is already a usable run.
#
# **What to do:** Runtime → Change runtime type → **T4 GPU**, then Runtime → **Run all**. Approve the Google Drive
# mount, and allow notebook access to the `KAGGLE_API_TOKEN` secret (key icon in the left bar) if Colab asks.
# Keep the tab open.
#
# **Resuming.** Every finished fold is saved to Google Drive (`MyDrive/shl_ft/`). If Colab disconnects, reconnect
# and **Run all** again: finished folds are skipped. About 12 min per fold on a T4, about 1 h per seed.
#
# **Results** go to the private Kaggle dataset `aryanawasthiteykik/shl-ft-outputs`: the fold files and a
# `progress.json` after every fold, and the merged out-of-fold + test predictions after every seed.

# %%
REPO, BRANCH = "https://github.com/Techy-A/shl-grammar-scoring", "claude/tender-cray-a5xjla"
INPUTS = "aryanawasthiteykik/shl-ft-inputs"     # private: folds.csv, transcripts_prompt[_crops].parquet, train.csv
OUTPUTS = "aryanawasthiteykik/shl-ft-outputs"   # private, created on the first upload
FLAGS = "--crops --drop-zero --epochs 2 --dropout0 --head bins"   # main-model recipe + the two changes
RUN = "s4cz_dropout0_bins"                                # per-fold files 06 writes for these flags: <RUN>[_seed<n>]_fold<k>
PRED = "pred_s4_deberta_ft_crops_nozero_dropout0_bins"    # merged predictions: <PRED>[_seed<n>].parquet
SEEDS, FOLDS = [0, 1, 2], [0, 1, 2, 3, 4]
EPOCHS = 2
EST_MIN_PER_FOLD = 12   # prior estimate for a T4, replaced by the measured time once a fold has run
WORK = "/content/drive/MyDrive/shl_ft"   # survives disconnects
SHL = "/content/shl"                     # the repo clone (lost on disconnect, re-cloned on Run all)
INP = "/content/inputs"                  # the downloaded inputs dataset

# %%
# A GPU first, so a CPU runtime fails at once with the fix.
import shutil, subprocess

gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True) if shutil.which("nvidia-smi") else None
if gpu is None or gpu.returncode != 0 or not gpu.stdout.strip():
    raise RuntimeError("No GPU in this runtime. Runtime → Change runtime type → T4 GPU, then Runtime → Run all again.")
GPU = gpu.stdout.strip()
print("GPU:", GPU)

# %%
import json, os, re, sys, time
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

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


# Only what is missing: transformers at the repo's pinned version, a Kaggle CLI new enough for KAGGLE_API_TOKEN.
need = [] if ver("transformers") == "5.18.0" else ["transformers==5.18.0"]
need += [] if int((ver("kaggle") or "0").split(".")[0]) >= 2 else ["kaggle==2.2.4"]
need += [p for p in ["sentencepiece", "soundfile", "pyarrow"] if find_spec(p) is None]
if need:
    print("pip install", *need)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
sh = lambda cmd, **kw: subprocess.run(cmd, shell=True, check=True, **kw)

if os.path.exists(SHL):
    sh(f"git -C {SHL} pull -q --ff-only")
else:
    sh(f"git clone -q --depth 1 -b {BRANCH} {REPO} {SHL}")
COMMIT = subprocess.run(["git", "-C", SHL, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

# artifacts/ of the clone is Drive, so every fold 06 writes is kept; train.csv goes where common.py looks for it.
ART = f"{WORK}/artifacts"
os.makedirs(ART, exist_ok=True)
if not os.path.islink(f"{SHL}/artifacts"):
    os.symlink(ART, f"{SHL}/artifacts")
if not os.path.exists(f"{INP}/train.csv"):
    sh(f"kaggle datasets download {INPUTS} -p {INP} --unzip")
for f in ["folds.csv", "transcripts_prompt.parquet", "transcripts_prompt_crops.parquet"]:
    shutil.copy(f"{INP}/{f}", ART)
os.makedirs(f"{SHL}/data/Dataset_Final", exist_ok=True)
shutil.copy(f"{INP}/train.csv", f"{SHL}/data/Dataset_Final/")
print("repo", BRANCH, COMMIT, "| inputs ready | outputs on Drive in", ART)

# %%
import pandas as pd

UP = "/content/upload"
_upload = None


def fold_file(seed, k):
    return f"{ART}/{RUN}{f'_seed{seed}' if seed else ''}_fold{k}.parquet"


def finished(path):
    """A fold file counts only if it reads back (a disconnect can leave a half-written one)."""
    if not os.path.exists(path):
        return False
    try:
        pd.read_parquet(path)
        return True
    except Exception:
        os.remove(path)
        return False


def upload(msg, **progress):
    """Copy this arm's fold/pred files and progress.json to UP and upload them as a new version of OUTPUTS.
    progress uploads run in the background and are skipped while one is still running; the end-of-seed one
    (msg starting 'seed') waits for it and retries, so the merged predictions always get out."""
    global _upload
    final = msg.startswith("seed")
    if _upload is not None and _upload.poll() is None:
        if not final:
            return
        _upload.wait()
    os.makedirs(UP, exist_ok=True)
    for f in os.listdir(UP):
        os.remove(f"{UP}/{f}")
    for f in sorted(os.listdir(ART)):
        if f.startswith(RUN) or f.startswith(PRED) or f == "results.csv":
            shutil.copy(f"{ART}/{f}", UP)
    progress.update(event=msg, commit=COMMIT, gpu=GPU, flags=FLAGS,
                    time_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    json.dump(progress, open(f"{UP}/progress.json", "w"), indent=1)
    json.dump({"title": OUTPUTS.split("/")[1], "id": OUTPUTS, "licenses": [{"name": "other"}]},
              open(f"{UP}/dataset-metadata.json", "w"))
    for attempt in range(3 if final else 1):
        exists = subprocess.run(["kaggle", "datasets", "status", OUTPUTS], capture_output=True).returncode == 0
        cmd = ["kaggle", "datasets", "version", "-p", UP, "-m", msg] if exists else ["kaggle", "datasets", "create", "-p", UP]
        if not final:
            _upload = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"uploaded ({msg}) to https://www.kaggle.com/datasets/{OUTPUTS}", flush=True)
            return
        print(f"upload attempt {attempt + 1} failed: {(r.stdout + r.stderr).strip()[-300:]}", flush=True)
        time.sleep(60)   # a just-created dataset takes a moment before it accepts versions
    print("WARNING: upload failed; the files are safe on Drive in", ART, flush=True)


def left():
    return sum(not finished(fold_file(s, k)) for s in SEEDS for k in FOLDS)


minutes = []   # measured minutes per fold in this session


def per_fold():
    return sum(minutes) / len(minutes) if minutes else EST_MIN_PER_FOLD


def run_fold(seed, k):
    cmd = f"{sys.executable} -u src/06_finetune_text.py {FLAGS} --seed {seed} --folds {k}"
    print(f"\n=== seed {seed} fold {k}: {cmd}", flush=True)
    env = {**os.environ, "SHL_LOCAL": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
    t0, tail = time.time(), []
    p = subprocess.Popen(cmd, shell=True, cwd=SHL, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-30:]
        if re.match(rf"fold {k} epoch 0:", line) and not minutes:   # first epoch of this session: early estimate
            est = (time.time() - t0) / 60 * EPOCHS
            print(f">>> first epoch took {(time.time() - t0) / 60:.1f} min (incl. model load): about {est:.0f} min "
                  f"per fold on this {GPU.split(',')[0]}, about {est * left() / 60:.1f} h for the {left()} folds left",
                  flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"seed {seed} fold {k} failed:\n" + "".join(tail))
    if not finished(fold_file(seed, k)):
        raise RuntimeError(f"06 finished but {fold_file(seed, k)} is missing: RUN does not match FLAGS")
    minutes.append((time.time() - t0) / 60)


print(f"{left()} of {len(SEEDS) * len(FOLDS)} folds to run, about {per_fold():.0f} min each on a T4: "
      f"about {left() * per_fold() / 60:.1f} h (free sessions get cut after a few hours; Run all resumes).")
for seed in SEEDS:
    for k in FOLDS:
        if finished(fold_file(seed, k)):
            print(f"seed {seed} fold {k}: done (on Drive), skipped")
            continue
        run_fold(seed, k)
        n = left()
        print(f">>> seed {seed} fold {k} took {minutes[-1]:.1f} min | {n} folds left, ETA {n * per_fold() / 60:.1f} h",
              flush=True)
        upload(f"progress: seed {seed} fold {k}", seed=seed, fold=k, folds_left=n,
               min_per_fold=round(per_fold(), 1), eta_h=round(n * per_fold() / 60, 2))
    r = subprocess.run(f"{sys.executable} src/06_finetune_text.py {FLAGS} --seed {seed} --merge", shell=True, cwd=SHL,
                       env={**os.environ, "SHL_LOCAL": "1"}, capture_output=True, text=True)
    print(r.stdout + r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"merge of seed {seed} failed")
    upload(f"seed {seed} merged", seed=seed, folds_left=left(), merged=(r.stdout.strip().splitlines() or [""])[-1],
           min_per_fold=round(per_fold(), 1))
print("all seeds done:", [f"{PRED}{f'_seed{s}' if s else ''}.parquet" for s in SEEDS])
