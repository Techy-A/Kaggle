"""Step 9: a two-stage LoRA member on OLMo-2-13B, with prompt-grouped OOF (GPU; QLoRA on T4s).

What: nishantdahal's 14B two-stage LoRA recipe ("Two-stage LoRA, part two") on allenai/OLMo-2-1124-13B (Apache-2.0):
- trunk: first 27 of 40 layers (the depth nishant keeps for the same-shape Qwen3-14B, 40 layers x 5120), pooled at
  layers 2, 4, 6, 9, 13, 17, 22, 27 (his Qwen3-14B layers), mean without the first token, LayerNorm per layer, head
  dropout 0.1, a plain head and a triplet-centred head; every passage ends with <|endoftext|> (100257), max 447 tokens;
- LoRA rank 16 / alpha 32 / dropout 0.05 on every linear layer of the trunk; the base is NF4 4-bit (fp16 compute), with
  embeddings, norms and the residual stream in fp32 (13B in fp16 does not fit a T4 and OLMo overflows fp16);
- batches of 6 triplets, 3 passages of one prompt each; loss = CE(plain) + CE(centred on triplet-centred features);
  AdamW wd 0.01, 10% warm-up then cosine, clip 1.0;
- stage 1: 1 epoch of nishant's pool_v4 minus passages with content cosine >= 0.3 to a train passage (07's filter),
  class-weighted, lr 1e-4 (LoRA) / 1e-3 (heads); stage 2: 2 epochs of train, 5e-5 / 5e-4.
Units: stage1 (the pool; it never sees train); fold<k> (stage 2 from stage 1 on 4/5 of the prompts, GroupKFold(5) on
prompt_id = 07's folds, scores the held-out fifth); full (stage 2 on all of train: the shipped member); merge (OOF).
Several GPUs: run under torchrun; each step's triplets are split over the ranks and the LoRA/head gradients summed.

Why: a new ensemble member from another model family. The #2 team (0.673) uses OLMo-2-13B; our 0.636 base has only
Qwen decoders. The OOF measures it alone and its value next to the base's n-gram SVC, before any submission.

Output (in --out); a unit whose output exists is skipped (the cache):
- stage1/lora_heads.safetensors, stage1_train.npz (train scored by the stage-1 model), summary_stage1.json;
- oof_fold<k>.npz: rows, ids, y, plain and centred logits (centred within each held-out prompt);
- lora_olmo2_13b/: meta.json, adapter/ (peft layout), heads.safetensors, the layout of nishant's 14B members plus
  cfg["quant"] = "nf4" (score it with olmo13b_member.olmo_features); summary_full.json;
- oof_olmo13b.npz + summary_olmo13b.json after merge (with --svc-oof: the blend with the SVC, paired bootstrap);
- state/<unit>.pt: the resume point (every --save-every steps), deleted when the unit is done.

Usage:
  torchrun --nproc_per_node 2 09_train_olmo13b.py --unit stage1 --train train.csv --pool pool_v4.parquet \
      --base OLMO_DIR --out OUT --stage1-hours 8.5
  python 09_train_olmo13b.py --unit fold0,fold1,fold2 ...   (one GPU each; also fold3,fold4,full; then --unit merge)
  CPU check: python lab/tests/check_olmo13b.py (a tiny random OLMo-2, end to end, NF4 and fp32, 1 and 2 ranks)
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import olmo13b_member as om  # noqa: E402

LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview',
          'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8',
          'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash',
          'kimi-k3']   # the part-two notebook's head order (meta.json is checked against it there)
K = len(LABELS)
MEMBER = 'lora_olmo2_13b'


def load_step(n):
    spec = importlib.util.spec_from_file_location(f's{n}', os.path.join(HERE, n))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- distributed helpers (one process = one GPU)
RANK, WORLD = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))


def log(*a):
    if RANK == 0:
        print(time.strftime('%H:%M:%S'), *a, flush=True)


def dist():
    import torch.distributed as d
    return d


def triplets(prompts, models, rng):
    """One epoch: every passage once (up to 2 per prompt left over), in triplets of one prompt, models interleaved
    so a triplet tends to hold 3 different models (as in colab/ntl_twostage_lora_train.py)."""
    out = []
    df = pd.DataFrame(dict(p=prompts, m=models))
    for _, g in df.groupby('p', sort=True):
        by_model = {}
        for i in rng.permutation(g.index.values):
            by_model.setdefault(models[i], []).append(int(i))
        order = [v[k] for k in range(max(map(len, by_model.values()))) for v in by_model.values() if k < len(v)]
        out += [order[s:s + 3] for s in range(0, len(order) - 2, 3)]
    return [out[i] for i in rng.permutation(len(out))]


class Trainer:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        self.tok = om.tokenizer(a.base)
        self.eos = self.tok.convert_tokens_to_ids('<|endoftext|>')
        t0 = time.time()
        self.model = om.load_trunk(a.base, a.keep, a.quant, str(dev), a.compute_dtype)
        torch.manual_seed(a.seed)   # the same LoRA init on every rank
        self.lora = om.attach_lora(self.model, a.r, a.alpha, a.lora_dropout)
        H = self.model.config.hidden_size
        self.heads = nn.ModuleDict(dict(norms=om.make_norms(H, len(a.layers), dev),
                                        plain=nn.Linear(H * len(a.layers), K), centered=nn.Linear(H * len(a.layers), K)))
        self.heads.to(dev)
        self.hdrop = nn.Dropout(a.head_drop)
        self.params = [p for m in self.lora.values() for p in (m.A, m.B)]
        if WORLD > 1:
            for p in self.params + list(self.heads.parameters()):
                dist().broadcast(p.data, 0)
        if dev.type == 'cuda':
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        log(f'trunk {a.keep} layers ({a.quant or "fp32"}) loaded in {time.time() - t0:.0f}s; LoRA on {len(self.lora)} '
            f'linear layers, {sum(p.numel() for p in self.params) / 1e6:.1f}M params')
        if dev.type == 'cuda':
            log(f'memory after load: {torch.cuda.memory_allocated(dev) / 2**30:.1f} GiB')

    def encode(self, texts):
        return om.encode(self.tok, texts, self.a.max_len, self.eos)

    # --- weights in and out
    def state(self):
        sd = {f'lora.{k}': v for k, v in om.adapter_state(self.lora).items()}
        sd.update({f'heads.{k}': v.detach().float().cpu().contiguous() for k, v in self.heads.state_dict().items()})
        return sd

    @torch.no_grad()
    def load_state(self, sd):
        om.load_adapter(self.lora, {k[5:]: v for k, v in sd.items() if k.startswith('lora.')})
        self.heads.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('heads.')})

    # --- training
    def loss(self, ids, rows, y, w):
        Z = om.pool(self.model, self.tok, [ids[i] for i in rows], self.a.layers, self.heads['norms'], self.dev)
        Z = self.hdrop(Z)
        n = len(rows) // 3
        Zc = (Z.view(n, 3, -1) - Z.view(n, 3, -1).mean(1, keepdim=True)).view(len(rows), -1)
        yb = torch.as_tensor(y[rows], device=self.dev)
        ce = nn.functional.cross_entropy
        return ce(self.heads['plain'](Z), yb, weight=w) + ce(self.heads['centered'](Zc), yb, weight=w)

    def probe(self, ids, y):
        """Forward + backward on the longest triplet of micro-batch size, so out-of-memory shows in minute one."""
        n = 3 * self.a.micro_triplets
        rows = list(np.argsort([len(x) for x in ids])[-n:])
        self.model.train()
        self.loss(ids, rows, y, None).backward()
        for p in self.params + list(self.heads.parameters()):
            p.grad = None
        if self.dev.type == 'cuda':
            log(f'memory probe ok ({n} passages of {len(ids[rows[-1]])} tokens): peak '
                f'{torch.cuda.max_memory_allocated(self.dev) / 2**30:.1f} GiB')

    def fit(self, ids, y, prompts, models, epochs, lr_lora, lr_head, class_weighted, unit, max_secs=None):
        a = self.a
        rng = np.random.default_rng(a.seed + (0 if unit == 'stage1' else 1))
        trips = [t for _ in range(epochs) for t in triplets(prompts, models, rng)]
        S = a.triplets_per_step
        plan = [trips[s:s + S] for s in range(0, len(trips) - S + 1, S)]
        w = None
        if class_weighted:
            cnt = np.bincount(y, minlength=K).astype(float)
            w = torch.tensor(np.where(cnt > 0, cnt.sum() / (K * np.maximum(cnt, 1)), 0.0), dtype=torch.float32,
                             device=self.dev)
        opt = torch.optim.AdamW([{'params': self.params, 'lr': lr_lora},
                                 {'params': list(self.heads.parameters()), 'lr': lr_head}], weight_decay=0.01)
        sch = dict(total=len(plan))
        warm = max(1, int(0.1 * len(plan)))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (
            1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, sch['total'] - warm)))))
        fp16 = self.dev.type == 'cuda' and a.quant == 'nf4' and a.compute_dtype == 'float16'
        scaler = torch.amp.GradScaler('cuda', init_scale=2.0 ** 12, enabled=fp16)
        state_path = os.path.join(a.out, 'state', f'{unit}.pt')
        start = 0
        if os.path.exists(state_path):
            st = torch.load(state_path, map_location='cpu', weights_only=False)
            self.load_state(st['weights'])
            opt.load_state_dict(st['opt']), sched.load_state_dict(st['sched']), scaler.load_state_dict(st['scaler'])
            start, sch['total'] = st['step'], st['total']
            log(f'{unit}: resumed at step {start}/{sch["total"]}')
        trainable = self.params + list(self.heads.parameters())
        self.model.train(), self.heads.train()
        t0, run, skipped, info = time.time(), [], 0, dict(planned=len(plan), passages=3 * S * len(plan))
        log(f'{unit}: {len(plan)} steps of {S} triplets, {WORLD} rank(s)')
        step = start
        while step < sch['total']:
            mine = plan[step][RANK::WORLD]
            for s in range(0, len(mine), a.micro_triplets):
                trip = mine[s:s + a.micro_triplets]
                lo = self.loss(ids, [i for t in trip for i in t], y, w)
                if not torch.isfinite(lo):   # fp16 overflow on a rare passage: skip that micro-batch
                    skipped += 1
                    continue
                scaler.scale(lo * len(trip) / S).backward()
                run.append(lo.item())
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            if WORLD > 1:   # sum the ranks' gradients (each already holds its share of the step's mean)
                flat = torch.cat([p.grad.reshape(-1) for p in trainable])
                dist().all_reduce(flat)
                o = 0
                for p in trainable:
                    p.grad.copy_(flat[o:o + p.numel()].view_as(p))
                    o += p.numel()
                del flat
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt), scaler.update(), sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step == start + 20 and max_secs:   # shorten the schedule to the time budget (cosine still reaches 0)
                per = (time.time() - t0) / 20
                new = torch.tensor([min(sch['total'], max(warm + 1, int(max_secs / per)))])
                if WORLD > 1:
                    dist().broadcast(new, 0)
                if int(new) < sch['total']:
                    log(f'{unit}: time budget {max_secs / 3600:.1f} h: schedule cut to {int(new)} of {sch["total"]} '
                        f'steps ({per:.1f} s/step)')
                sch['total'] = int(new)
            if step % 20 == 0 or step == sch['total']:
                el = time.time() - t0
                log(f'{unit} step {step}/{sch["total"]} loss {np.mean(run[-60:]) if run else float("nan"):.3f} '
                    f'skipped {skipped} scale {scaler.get_scale():.0f} | {el / 60:.1f} min, '
                    f'ETA {el / (step - start) * (sch["total"] - step) / 60:.0f} min')
            if step % a.save_every == 0 and step < sch['total'] and RANK == 0:
                os.makedirs(os.path.dirname(state_path), exist_ok=True)
                torch.save(dict(weights=self.state(), opt=opt.state_dict(), sched=sched.state_dict(),
                                scaler=scaler.state_dict(), step=step, total=sch['total']), state_path + '.tmp')
                os.replace(state_path + '.tmp', state_path)   # never a half-written resume point
        self.model.eval(), self.heads.eval()
        info.update(steps=sch['total'], secs=time.time() - t0, skipped=skipped,
                    final_loss=float(np.mean(run[-60:])) if run else None)
        return info

    # --- scoring
    def logits(self, ids, rows, groups):
        """Plain logits and logits of the centred head (features centred within each group) for `rows`.
        Several ranks: whole groups per rank, gathered on every rank."""
        my_groups = set(np.unique(groups[rows])[RANK::WORLD])
        mine = np.array([r for r in rows if groups[r] in my_groups], dtype=int)
        Z = om.features(self.model, self.tok, [ids[i] for i in mine], self.a.layers, self.heads['norms'],
                        self.a.eval_bs, self.dev)
        Zc = Z.copy()
        for g in np.unique(groups[mine]):
            sel = groups[mine] == g
            Zc[sel] -= Zc[sel].mean(0)
        hw = {k: v.detach().float().cpu().numpy() for k, v in self.heads.state_dict().items()}
        part = (mine, Z @ hw['plain.weight'].T + hw['plain.bias'], Zc @ hw['centered.weight'].T + hw['centered.bias'])
        parts = [part]
        if WORLD > 1:
            parts = [None] * WORLD
            dist().all_gather_object(parts, part)
        pos = {r: i for i, r in enumerate(rows)}
        P, C = np.zeros((len(rows), K), np.float32), np.zeros((len(rows), K), np.float32)
        for m, p, c in parts:
            idx = [pos[r] for r in m]
            P[idx], C[idx] = p, c
        return P, C


def bacc(y, pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y, pred))


def export(tr, a, out_dir, info):
    """nishant's checkpoint layout: meta.json, adapter/ (peft keys), heads.safetensors."""
    from safetensors.torch import save_file
    os.makedirs(os.path.join(out_dir, 'adapter'), exist_ok=True)
    save_file(om.adapter_state(tr.lora), os.path.join(out_dir, 'adapter', 'adapter_model.safetensors'))
    json.dump({'r': a.r, 'lora_alpha': a.alpha, 'lora_dropout': a.lora_dropout, 'peft_type': 'LORA', 'bias': 'none',
               'base_model_name_or_path': 'allenai/OLMo-2-1124-13B', 'auto_mapping': {'base_model_class': 'Olmo2Model'},
               'target_modules': sorted({k.split('.')[-1] for k in tr.lora})},
              open(os.path.join(out_dir, 'adapter', 'adapter_config.json'), 'w'), indent=1)
    save_file({k: v.detach().float().cpu().contiguous() for k, v in tr.heads.state_dict().items()},
              os.path.join(out_dir, 'heads.safetensors'))
    cfg = dict(model_dir='allenai/OLMo-2-1124-13B', keep=a.keep, layers=a.layers, max_len=a.max_len, r=a.r,
               alpha=a.alpha, ldrop=a.lora_dropout, head_drop=a.head_drop, lr_adapter=a.lr2[0], lr_head=a.lr2[1],
               wd=0.01, warmup=0.1, epochs=float(a.stage2_epochs), triplets_per_step=a.triplets_per_step, group=3,
               eval_bs=a.eval_bs, seed=a.seed, grad_ckpt=True, eos_id=int(tr.eos),
               eos_note='eos_id is the end token appended in training (<|endoftext|>)',
               quant=a.quant, compute_dtype=a.compute_dtype, nonquant_dtype='float32', final_norm=False,
               stage1=dict(epochs=a.stage1_epochs, lr_adapter=a.lr1[0], lr_head=a.lr1[1], class_weighted=True,
                           pool='nishantdahal/ntl-handwriting-blend pool_v4 minus content cosine >= 0.3 to train'))
    json.dump(dict(cfg=cfg, labels=LABELS, train_info=info), open(os.path.join(out_dir, 'meta.json'), 'w'), indent=1)


def merge(a, train, y, folds):
    rows, P, C = np.arange(len(train)), np.zeros((len(train), K)), np.zeros((len(train), K))
    for f in range(a.folds):
        z = np.load(os.path.join(a.out, f'oof_fold{f}.npz'))
        P[z['rows']], C[z['rows']] = z['plain'], z['centred']
    s08 = load_step('08_blend_oof.py')
    Pp, Pc = s08.softmax(P), s08.softmax(C)
    st1 = np.load(os.path.join(a.out, 'stage1_train.npz'))
    res = dict(oof_plain=bacc(y, P.argmax(1)), oof_centred_by_prompt=bacc(y, C.argmax(1)),
               oof_plain_sinkhorn=bacc(y, s08.blend([(1.0, Pp)])),
               oof_plain_plus_twin=bacc(y, s08.blend([(1.0, Pp), (0.25, Pc)])),
               stage1_only=bacc(y, st1['plain'].argmax(1)),
               fold_plain=[bacc(y[folds == f], P[folds == f].argmax(1)) for f in range(a.folds)])
    np.savez(os.path.join(a.out, 'oof_olmo13b.npz'), ids=train.id.astype(str).to_numpy(dtype=str), fold=folds, y=y,
             labels=np.array(LABELS, dtype=str), plain=P.astype(np.float32), centred=C.astype(np.float32))
    if a.svc_oof:   # value next to the base's SVC (weight 1.0 there); the LoRAs get 0.2-0.4 in the base
        z = np.load(a.svc_oof)
        assert list(z['ids']) == train.id.astype(str).tolist() and (z['fold'] == folds).all()
        Ps = s08.softmax(z['scores'])   # 08's oof_svc.npz, alphabetical label order -> ours
        Ps = Ps[:, [s08.LABELS.index(l) for l in LABELS]]
        base = s08.blend([(1.0, Ps)])
        res['svc_alone_sinkhorn'] = bacc(y, base)
        groups, ug = train.prompt_id.values, np.unique(train.prompt_id.values)
        gidx = {g: np.where(groups == g)[0] for g in ug}
        rng = np.random.default_rng(0)
        boots = [np.concatenate([gidx[g] for g in rng.choice(ug, len(ug))]) for _ in range(a.boot)]
        for wgt in (0.1, 0.2, 0.4):
            pr = s08.blend([(1.0, Ps), (wgt, Pp)])
            d = np.array([bacc(y[i], pr[i]) - bacc(y[i], base[i]) for i in boots])
            res[f'svc_plus_olmo_{wgt}_sinkhorn'] = bacc(y, pr)
            res[f'svc_plus_olmo_{wgt}_p_better'] = float((d > 0).mean())
    side = {f[8:-5]: json.load(open(os.path.join(a.out, f))) for f in ('summary_stage1.json', 'summary_full.json')
            if os.path.exists(os.path.join(a.out, f))}
    json.dump(dict(scores=res, **side), open(os.path.join(a.out, 'summary_olmo13b.json'), 'w'), indent=1)
    for k, v in res.items():
        log(f'{k:36s} {v if isinstance(v, list) else round(v, 4)}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', required=True, help='stage1, fold<k>, full, merge, or a comma list')
    ap.add_argument('--train', required=True)
    ap.add_argument('--pool')
    ap.add_argument('--base', help='OLMo-2-1124-13B dir (config, tokenizer, safetensors)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--quant', default='nf4', type=lambda s: None if s == 'none' else s, help='nf4 or none')
    ap.add_argument('--compute-dtype', default='float16')
    ap.add_argument('--keep', type=int, default=27)
    ap.add_argument('--layers', default='2,4,6,9,13,17,22,27', type=lambda s: [int(x) for x in s.split(',')])
    ap.add_argument('--max-len', type=int, default=447)
    ap.add_argument('--r', type=int, default=16)
    ap.add_argument('--alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--head-drop', type=float, default=0.1)
    ap.add_argument('--triplets-per-step', type=int, default=6)
    ap.add_argument('--micro-triplets', type=int, default=1, help='triplets per forward pass (memory only)')
    ap.add_argument('--stage1-epochs', type=int, default=1)
    ap.add_argument('--stage2-epochs', type=int, default=2)
    ap.add_argument('--lr1', default='1e-4,1e-3', type=lambda s: [float(x) for x in s.split(',')])
    ap.add_argument('--lr2', default='5e-5,5e-4', type=lambda s: [float(x) for x in s.split(',')])
    ap.add_argument('--stage1-hours', type=float, default=None, help='cap on stage-1 wall time (shortens the schedule)')
    ap.add_argument('--eval-bs', type=int, default=8)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--svc-oof', help="08's oof_svc.npz, for the blend check in merge")
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--limit-pool', type=int, default=None, help='random whole prompts, up to n passages (smoke tests)')
    ap.add_argument('--limit-prompts', type=int, default=None, help='first n train prompts only (smoke tests)')
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    assert a.layers[-1] <= a.keep and a.triplets_per_step % WORLD == 0
    units = a.unit.split(',')
    if not torch.cuda.is_available() and not a.allow_cpu and units != ['merge']:
        raise SystemExit('no CUDA GPU in this session (pass --allow-cpu only for smoke tests)')
    dev = torch.device(f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}' if torch.cuda.is_available() else 'cpu')
    if WORLD > 1:
        if dev.type == 'cuda':
            torch.cuda.set_device(dev)
        dist().init_process_group('nccl' if dev.type == 'cuda' else 'gloo')
        assert units == ['stage1'], 'several ranks only for stage1; run stage-2 units one GPU each'

    train = pd.read_csv(a.train)
    train['text'] = train['text'].fillna('').astype(str)
    if a.limit_prompts:
        train = train[train.prompt_id.isin(sorted(train.prompt_id.unique())[:a.limit_prompts])].reset_index(drop=True)
    y = train.model.map({l: i for i, l in enumerate(LABELS)}).values
    from sklearn.model_selection import GroupKFold
    folds = np.zeros(len(train), int)
    for f, (_, va) in enumerate(GroupKFold(n_splits=a.folds).split(train, groups=train.prompt_id)):
        folds[va] = f
    groups = train.prompt_id.astype(str).values
    tr_obj, s1_sd = None, None

    for unit in units:
        done = dict(stage1='summary_stage1.json', full='summary_full.json', merge='summary_olmo13b.json').get(
            unit, f'oof_{unit}.npz')
        if os.path.exists(os.path.join(a.out, done)):
            log(f'{unit}: cached ({done})')
            continue
        if unit == 'merge':
            if RANK == 0:
                merge(a, train, y, folds)
            continue
        T0 = time.time()
        if tr_obj is None:
            tr_obj = Trainer(a, dev)
            ids_tr = tr_obj.encode(train.text)
        tr = tr_obj
        if unit == 'stage1':
            pool = pd.read_parquet(a.pool)
            pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
            keep = load_step('07_train_encoders.py').leak_filter(pool.text.tolist(), train.text.tolist()) & \
                ~pool.text.isin(set(train.text))
            log(f'pool {len(pool)}, kept {int(keep.sum())} after the train-similarity filter')
            pool = pool[keep].reset_index(drop=True)
            if a.limit_pool:
                ps = pool.prompt_id.drop_duplicates().sample(frac=1, random_state=0)   # whole prompts, for triplets
                n = pool.groupby('prompt_id').size()[ps].cumsum()
                pool = pool[pool.prompt_id.isin(n.index[n <= a.limit_pool])].reset_index(drop=True)
            ids_pool = tr.encode(pool.text)
            yp = pool.model.map({l: i for i, l in enumerate(LABELS)}).values
            tr.probe(ids_pool, yp)
            info = tr.fit(ids_pool, yp, pool.prompt_id.astype(str).values, pool.model.values, a.stage1_epochs,
                          *a.lr1, True, unit, max_secs=a.stage1_hours * 3600 if a.stage1_hours else None)
            P, C = tr.logits(ids_tr, np.arange(len(train)), groups)
            if RANK == 0:
                from safetensors.torch import save_file
                os.makedirs(os.path.join(a.out, 'stage1'), exist_ok=True)
                save_file(tr.state(), os.path.join(a.out, 'stage1', 'lora_heads.safetensors'))
                np.savez(os.path.join(a.out, 'stage1_train.npz'), ids=train.id.astype(str).to_numpy(dtype=str), y=y,
                         plain=P, centred=C)
                summ = dict(unit=unit, pool_used=len(pool), info=info, stage1_only_bacc_plain=bacc(y, P.argmax(1)),
                            stage1_only_bacc_centred_by_prompt=bacc(y, C.argmax(1)), secs=time.time() - T0,
                            ranks=WORLD, device=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu',
                            torch=torch.__version__, transformers=__import__('transformers').__version__)
                json.dump(summ, open(os.path.join(a.out, 'summary_stage1.json'), 'w'), indent=1, default=float)
                log(f'stage 1 only, on train: plain {summ["stage1_only_bacc_plain"]:.4f}, centred by prompt '
                    f'{summ["stage1_only_bacc_centred_by_prompt"]:.4f}')
        else:
            from safetensors.torch import load_file
            if s1_sd is None:
                s1_sd = load_file(os.path.join(a.out, 'stage1', 'lora_heads.safetensors'))
            tr.load_state(s1_sd)
            fit_rows = np.arange(len(train)) if unit == 'full' else np.where(folds != int(unit[4:]))[0]
            sub = lambda v: v[fit_rows]
            info = tr.fit([ids_tr[i] for i in fit_rows], sub(y), sub(groups), sub(train.model.values),
                          a.stage2_epochs, *a.lr2, False, unit)
            if unit == 'full':
                out_dir = os.path.join(a.out, MEMBER)
                export(tr, a, out_dir, info)
                chk = np.arange(min(128, len(train)))
                P_mem, _ = tr.logits(ids_tr, chk, groups)
                tr_obj = tr = None   # free the training trunk: the reload check loads a second one
                import gc
                gc.collect()
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
                Z, heads = om.olmo_features(out_dir, a.base, train.text.values[chk].tolist(), a.eval_bs, LABELS,
                                            device=str(dev))
                P_re = Z @ heads['plain.weight'].T + heads['plain.bias']
                agree, diff = float((P_re.argmax(1) == P_mem.argmax(1)).mean()), float(np.abs(P_re - P_mem).max())
                log(f'reloaded {out_dir} with olmo_features: top-1 agreement {agree:.3f}, max |logit diff| {diff:.2e}')
                assert agree >= 0.98, agree
                json.dump(dict(unit=unit, info=info, reload_agreement=agree, reload_max_logit_diff=diff,
                               in_sample_bacc_128=bacc(y[chk], P_re.argmax(1)), secs=time.time() - T0,
                               device=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu',
                               torch=torch.__version__, transformers=__import__('transformers').__version__),
                          open(os.path.join(a.out, 'summary_full.json'), 'w'), indent=1, default=float)
            else:
                va = np.where(folds == int(unit[4:]))[0]
                P, C = tr.logits(ids_tr, va, groups)
                np.savez(os.path.join(a.out, f'oof_{unit}.npz'), rows=va, ids=train.id.values[va].astype(str), y=y[va],
                         fold=int(unit[4:]), plain=P, centred=C, labels=np.array(LABELS, dtype=str), info=json.dumps(info))
                log(f'{unit}: held-out balanced accuracy plain {bacc(y[va], P.argmax(1)):.4f}, centred by prompt '
                    f'{bacc(y[va], C.argmax(1)):.4f} on {len(va)} rows ({(time.time() - T0) / 60:.0f} min)')
        st = os.path.join(a.out, 'state', f'{unit}.pt')
        if RANK == 0 and os.path.exists(st):
            os.remove(st)
    if WORLD > 1:
        dist().barrier()
        dist().destroy_process_group()


if __name__ == '__main__':
    main()
