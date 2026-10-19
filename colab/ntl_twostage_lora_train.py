# %% [markdown]
# # NTL two-stage LoRA, trained on free Colab (T4)
#
# Rebuilds the checkpoint of nishantdahal's public "Two-stage LoRA on the public pool" (LB 0.592), whose pool and
# checkpoints are private. Its recipe is published:
#
# 1. **Model:** Qwen3-8B-Base, first 24 layers, LoRA rank 16 / alpha 32 / dropout 0.05 on every linear layer.
#    The mean hidden state at layers 2, 4, 6, 8, 12, 16, 20 and 24 goes through one LayerNorm each, giving 8 x 4096
#    features, read by two linear heads (plain, and centred within a triplet).
# 2. **Batches:** 6 triplets per step. Each triplet is 3 passages answering the same prompt.
#    Loss = CE(plain head) + CE(centred head on triplet-centred features).
# 3. **Stage 1:** 1 epoch over the public pool, class-weighted. Stage 2: 2 epochs on train.csv.
#    AdamW with 10% warm-up then cosine; lr (LoRA / heads) is 1e-4 / 1e-3 in stage 1 and 5e-5 / 5e-4 in stage 2.
#
# The output folder `lora8b_v1/` matches what the 0.592 inference code loads: `meta.json`, `adapter/` in peft format,
# and `heads.safetensors`. It is uploaded as the private Kaggle dataset below.
#
# **One-time setup**
# 1. Runtime → Change runtime type → **T4 GPU**.
# 2. Secrets (key icon) → add `KAGGLE_API_TOKEN` (your Kaggle API token) with notebook access on.
# 3. Run all.
#
# **Resuming.** Training saves to Google Drive every `SAVE_EVERY` steps. If Colab disconnects, reconnect and
# **Run all** again: it picks up from the last save. On a T4 it takes about 3 h in total (stage 1 ~2.5 h).

# %%
OUT_DATASET = "aryanawasthiteykik/ntl-lora8b-twostage"   # private Kaggle dataset the checkpoint is uploaded to
POOL_DATASET = "aryanawasthiteykik/ntl-public-pool-v1"
BASE = "Qwen/Qwen3-8B-Base"      # same weights as Kaggle's qwen-lm/qwen-3/Transformers/8b-base
KEEP, LAYERS = 24, [2, 4, 6, 8, 12, 16, 20, 24]
R, ALPHA, DROPOUT = 16, 32, 0.05
MAX_LEN = 512
TRIPLETS_PER_STEP, TRIPLETS_PER_MICRO = 6, 2   # 2 triplets per forward pass x 3 accumulation steps, to fit a T4
STAGES = [dict(name="pool", epochs=1, lr_lora=1e-4, lr_head=1e-3, class_weighted=True),
          dict(name="train", epochs=2, lr_lora=5e-5, lr_head=5e-4, class_weighted=False)]
SAVE_EVERY = 100
SEED = 42
DRY_RUN = False   # True: a tiny random Qwen3 on CPU, for checking the code path (no GPU, no downloads)
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']   # the 0.592 inference code's head order

# %%
import json, math, os, subprocess, sys, time
import numpy as np, pandas as pd, torch
from torch import nn

if not DRY_RUN:
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "transformers>=4.51", "safetensors"],
                   check=True)
    WORK = "/content/drive/MyDrive/ntl_twostage"
    kaggle = lambda *a: subprocess.run(["kaggle", *a], check=True)
    os.makedirs("/content/data", exist_ok=True)
    if not os.path.exists("/content/data/train.csv"):
        kaggle("competitions", "download", "name-that-llm", "-p", "/content/data")
        subprocess.run("cd /content/data && unzip -q -o '*.zip' && rm -f *.zip", shell=True, check=True)
    if not os.path.exists("/content/data/pool_v1.parquet"):
        kaggle("datasets", "download", POOL_DATASET, "-p", "/content/data", "--unzip")
    DATA = "/content/data"
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"],
                         capture_output=True, text=True).stdout)
else:
    WORK, DATA = "/tmp/ntl_twostage_dry", os.environ.get("NTL_DATA", "data")
os.makedirs(WORK, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEV == "cuda" and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
if DEV == "cpu":
    DTYPE = torch.float32

# %%
train = pd.read_csv(f"{DATA}/train.csv")
pool = pd.read_parquet(f"{DATA}/pool_v1.parquet")
if DRY_RUN:
    pool = pool.groupby("model").head(12).reset_index(drop=True)
    train = train[train.prompt_id.isin(train.prompt_id.unique()[:4])].reset_index(drop=True)
lab = {l: i for i, l in enumerate(LABELS)}
assert set(train.model) <= set(LABELS) and set(pool.model) <= set(LABELS)


def triplets(df, rng):
    """One epoch: every passage once, in triplets of 3 passages (different models when possible) of one prompt."""
    out = []
    for _, g in df.groupby("prompt_id"):
        idx = rng.permutation(g.index.values)
        # interleave by model so consecutive passages tend to come from different models
        by_model = {}
        for i in idx:
            by_model.setdefault(df.at[i, "model"], []).append(i)
        order = [i for k in range(max(map(len, by_model.values()))) for v in by_model.values() if k < len(v) for i in [v[k]]]
        out += [order[s:s + 3] for s in range(0, len(order) - 2, 3)]
    out = [out[i] for i in rng.permutation(len(out))]
    return [out[s:s + TRIPLETS_PER_STEP] for s in range(0, len(out) - TRIPLETS_PER_STEP + 1, TRIPLETS_PER_STEP)]

# %%
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


class LoRALinear(nn.Module):
    """y = W x + (B A dropout(x)) * alpha / r, with W frozen; saved in peft's key layout."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.empty(R, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, R, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop, self.scale = nn.Dropout(DROPOUT), ALPHA / R

    def forward(self, x):
        return self.base(x) + (self.drop(x).to(self.A.dtype) @ self.A.T @ self.B.T).to(x.dtype) * self.scale


if DRY_RUN:
    from transformers import Qwen3Config
    mc = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=KEEP,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    model = AutoModel.from_config(mc)
    tok = AutoTokenizer.from_pretrained(os.environ.get("NTL_TOKENIZER", "Qwen/Qwen3-8B-Base"))
else:
    mc = AutoConfig.from_pretrained(BASE)
    mc.num_hidden_layers, mc.use_cache = KEEP, False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:KEEP]
    model = AutoModel.from_pretrained(BASE, config=mc, dtype=DTYPE).to(DEV)
    tok = AutoTokenizer.from_pretrained(BASE)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model.norm = nn.Identity()   # the 0.592 inference code reads the raw residual stream
for p in model.parameters():
    p.requires_grad_(False)
lora = {}
for name, m in list(model.named_modules()):
    for child, sub in list(m.named_children()):
        if isinstance(sub, nn.Linear) and name.startswith("layers."):
            setattr(m, child, LoRALinear(sub))
            lora[f"{name}.{child}"] = getattr(m, child)
H = model.config.hidden_size
norms = nn.ModuleList([nn.LayerNorm(H) for _ in LAYERS]).to(DEV)
plain, centered = nn.Linear(H * len(LAYERS), len(LABELS)).to(DEV), nn.Linear(H * len(LAYERS), len(LABELS)).to(DEV)
heads = nn.ModuleDict(dict(norms=norms, plain=plain, centered=centered))
if DEV == "cuda":
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
print(f"LoRA on {len(lora)} linear layers, {sum(p.numel() for m in lora.values() for p in (m.A, m.B)) / 1e6:.1f}M params")


def features(texts):
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=MAX_LEN)["input_ids"]
    enc = tok.pad({"input_ids": [x + [tok.eos_token_id] for x in ids]}, return_tensors="pt").to(DEV)
    with torch.autocast(DEV, dtype=DTYPE, enabled=DEV == "cuda"):
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
    m = enc["attention_mask"].unsqueeze(-1).float()
    m[:, 0] = 0   # first token = attention sink, left out of the mean (as in the inference code)
    den = m.sum(1).clamp(min=1)
    return torch.cat([nm((hs[l].float() * m).sum(1) / den) for l, nm in zip(LAYERS, norms)], -1)

# %%
def lr_lambda(total):
    warm = max(1, int(0.1 * total))
    return lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))


def save_state(path, stage, step, opt, sched, scaler):
    torch.save(dict(stage=stage, step=step, lora={k: (m.A.detach().cpu(), m.B.detach().cpu()) for k, m in lora.items()},
                    heads=heads.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                    scaler=scaler.state_dict()), path + ".tmp")
    os.replace(path + ".tmp", path)   # never leave a half-written resume file


state_path = f"{WORK}/state.pt"
resume = torch.load(state_path, map_location="cpu", weights_only=False) if os.path.exists(state_path) else None
if resume:
    for k, (A, B) in resume["lora"].items():
        lora[k].A.data.copy_(A), lora[k].B.data.copy_(B)
    heads.load_state_dict(resume["heads"])
    print(f"resuming at stage {resume['stage']} step {resume['step']}")

for si, st in enumerate(STAGES):
    if resume and si < resume["stage"]:
        continue
    df = (pool if st["name"] == "pool" else train).reset_index(drop=True)
    y_all = df.model.map(lab).values
    w = None
    if st["class_weighted"]:
        cnt = np.bincount(y_all, minlength=len(LABELS)).astype(float)
        w = torch.tensor(cnt.sum() / (len(LABELS) * np.maximum(cnt, 1)), dtype=torch.float32, device=DEV)
    rng = np.random.default_rng(SEED + si)
    steps = [b for _ in range(st["epochs"]) for b in triplets(df, rng)]
    opt = torch.optim.AdamW([{"params": [p for m in lora.values() for p in (m.A, m.B)], "lr": st["lr_lora"]},
                             {"params": heads.parameters(), "lr": st["lr_head"]}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(len(steps)))
    scaler = torch.amp.GradScaler(DEV, enabled=DTYPE == torch.float16)
    start = 0
    if resume and si == resume["stage"]:
        opt.load_state_dict(resume["opt"]), sched.load_state_dict(resume["sched"]), scaler.load_state_dict(resume["scaler"])
        start = resume["step"]
    model.train(), heads.train()
    t0, run, skipped = time.time(), [], 0
    for step in range(start, len(steps)):
        opt.zero_grad(set_to_none=True)
        for s in range(0, len(steps[step]), TRIPLETS_PER_MICRO):
            trip = steps[step][s:s + TRIPLETS_PER_MICRO]
            rows = [i for t in trip for i in t]
            Z = features(df.text.values[rows])
            yb = torch.tensor(y_all[rows], device=DEV)
            Zc = (Z.view(len(trip), 3, -1) - Z.view(len(trip), 3, -1).mean(1, keepdim=True)).view(len(rows), -1)
            ce = nn.functional.cross_entropy
            loss = ce(plain(Z), yb, weight=w) + ce(centered(Zc), yb, weight=w)
            if not torch.isfinite(loss):   # fp16 overflow on a rare passage: skip that micro-batch
                skipped += 1
                continue
            scaler.scale(loss * len(trip) / len(steps[step])).backward()
            run.append(loss.item())
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt), scaler.update(), sched.step()
        if (step + 1) % 20 == 0 or step + 1 == len(steps):
            el = time.time() - t0
            eta = el / (step + 1 - start) * (len(steps) - step - 1) / 60
            print(f"[{st['name']}] step {step + 1}/{len(steps)} loss {np.mean(run[-60:]):.3f} skipped {skipped} "
                  f"| {el / 60:.1f} min, ETA {eta:.0f} min", flush=True)
        if (step + 1) % SAVE_EVERY == 0:
            save_state(state_path, si, step + 1, opt, sched, scaler)
    save_state(state_path, si + 1, 0, opt, sched, scaler)
    resume = None
    print(f"stage {st['name']} done", flush=True)

# %%
# Export in the 0.592 inference layout, then upload as a private Kaggle dataset.
out = f"{WORK}/export/lora8b_v1"
os.makedirs(f"{out}/adapter", exist_ok=True)
sd = {}
for k, m in lora.items():
    sd[f"base_model.model.{k}.lora_A.weight"] = m.A.detach().float().cpu().contiguous()
    sd[f"base_model.model.{k}.lora_B.weight"] = m.B.detach().float().cpu().contiguous()
save_file(sd, f"{out}/adapter/adapter_model.safetensors")
json.dump({"r": R, "lora_alpha": ALPHA, "lora_dropout": DROPOUT, "peft_type": "LORA", "base_model_name_or_path": BASE,
           "target_modules": sorted({k.split(".")[-1] for k in lora})}, open(f"{out}/adapter/adapter_config.json", "w"))
save_file({k: v.detach().float().cpu().contiguous() for k, v in heads.state_dict().items()}, f"{out}/heads.safetensors")
json.dump({"labels": LABELS, "cfg": {"keep": KEEP, "layers": LAYERS, "max_len": MAX_LEN, "r": R, "alpha": ALPHA},
           "recipe": "two-stage: 1 epoch public pool (class-weighted) then 2 epochs train; triplet plain+centred CE"},
          open(f"{out}/meta.json", "w"), indent=1)
print("exported", sorted(os.listdir(out)), sorted(os.listdir(f"{out}/adapter")))

if not DRY_RUN:
    root = f"{WORK}/export"
    json.dump({"title": OUT_DATASET.split("/")[1], "id": OUT_DATASET, "licenses": [{"name": "CC-BY-SA-4.0"}]},
              open(f"{root}/dataset-metadata.json", "w"))
    exists = subprocess.run(["kaggle", "datasets", "status", OUT_DATASET], capture_output=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", root, "-m", "two-stage LoRA 8B", "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", root, "-r", "zip")
    print("uploaded to", OUT_DATASET)
