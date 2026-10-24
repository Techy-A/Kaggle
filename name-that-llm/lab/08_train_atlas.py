"""Step 8b: train our own ATLAS bundle (source-conditioned likelihood, arXiv:2610.19583) with prompt-grouped OOF (GPU).

What: Qwen3-8B-Base frozen; 18 sources, each with a LoRA up-projection B[s] on q/k/v/o of all 36 layers (A shared, rank 16,
alpha 32) and a one-token prefix; trained on train.csv only with next-token NLL under the passage's own source. The
recipe is josephadamski91/ntl-atlas-qwen3-8b-v1's training_config (5 epochs, batch 128, AdamW 1e-4, wd 0.01, 5% warm-up,
clip 1.0, seed 17); its training code is not published, so the gaps are our guesses, listed in atlas_rebuild_spec.md.
The model code is our package lab/atlas (ntl_atlas, ntl_rlcd), shipped inside the bundle.
Units: fold<k> trains on 4/5 of the prompts (GroupKFold(5) on prompt_id, the same folds as 07) and scores the held-out
fifth; full trains on all rows and writes the bundle; merge joins the folds.

Why: the 0.636 notebook takes 10% from ATLAS on uncertain rows. With a bundle we trained ourselves, a variant of that
notebook can be a final pick, and the OOF gives ATLAS's standalone balanced accuracy, which the public bundle never had.

Output (in --out):
- bundle/: the drop-in bundle (atlas_bundle.json, adapters.safetensors, ntl_atlas/, ntl_rlcd/), the only
  atlas_bundle.json in --out, because the notebook asserts exactly one under /kaggle/input.
- oof_fold<k>.npz: rows, ids, y, fold, probs and raw evidence (labels order) of the held-out prompts, plus diagnostics.
- oof_atlas.npz + summary_atlas.json after merge: OOF probabilities of every train row, balanced accuracy overall and
  per fold.
- state/<unit>.pt: the resume point (every --save-every steps), deleted when the unit is done.
A unit whose output exists is skipped (the cache).

Usage:
  python 08_train_atlas.py --unit fold0 --train train.csv --base Qwen/Qwen3-8B-Base --out OUT [--quant nf4]
  units: fold0 .. fold4, full, merge, or all (= fold0, full, fold1 .. fold4, merge)
  CPU smoke test: --allow-cpu --base <tiny dir> --base-dtype float32 --limit 72 --epochs 2 --batch 16
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'atlas'))
from ntl_atlas.model import load_model  # noqa: E402
from ntl_rlcd.data import LABELS  # noqa: E402

K = len(LABELS)
CFG = dict(   # the public bundle's training_config, field by field; only these keys reach our model
    base_model='Qwen/Qwen3-8B-Base', base_revision='49e3418fbbbca6ecbdf9608b4d22e5a407081db4', seed=17,
    rank=16, alpha=32, dropout=0.05, source_b_init_std=1e-4, learning_rate=1e-4, weight_decay=0.01, epochs=5,
    effective_batch_size=128, micro_batch_size=4, max_length=512, grad_clip=1.0, warmup_ratio=0.05,
    source_chunk_size=2, head_chunk_tokens=64, prod_temperatures=[1.0, 2.0, 4.0, 8.0], base_dtype='float16',
    adapter_dtype='float32')
PKG = ['ntl_atlas/__init__.py', 'ntl_atlas/model.py', 'ntl_atlas/inference.py', 'ntl_rlcd/__init__.py', 'ntl_rlcd/data.py']


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def batches(n, cfg, seed):
    """Optimizer steps as lists of row positions: a fresh permutation per epoch cut into effective batches; the last batch
    of an epoch is partial (2699 rows -> 22 steps per epoch, 110 in all, as in the public bundle's completed_steps)."""
    out, bs = [], cfg['effective_batch_size']
    for e in range(cfg['epochs']):
        p = np.random.default_rng(seed + e).permutation(n)
        out += [p[s:s + bs] for s in range(0, n, bs)]
    return out


def pad(tok, seqs, dev):
    pk = tok.pad({'input_ids': seqs}, padding=True, return_tensors='pt')
    return pk['input_ids'].to(dev), pk['attention_mask'].to(dev)


def score_rows(model, tok, ids, rows, bs, chunk):
    """probs and raw evidence (len(rows), K) for token lists ids[rows], shortest first."""
    rows = np.asarray(rows)
    order = rows[np.argsort([len(ids[i]) for i in rows], kind='stable')]
    pos = {r: i for i, r in enumerate(rows)}
    P, R = np.zeros((len(rows), K), np.float32), np.zeros((len(rows), K), np.float32)
    for s in range(0, len(order), bs):
        b = order[s:s + bs]
        x, m = pad(tok, [ids[i] for i in b], model.device)
        p, r = model.score(x, m, source_chunk=chunk)
        at = [pos[i] for i in b]
        P[at], R[at] = p.float().cpu().numpy(), r.float().cpu().numpy()
    return P, R


def diag_rows(y, rows, per_class=2):
    """A fixed small probe set: the first `per_class` rows of each class among `rows`."""
    return np.concatenate([rows[y[rows] == k][:per_class] for k in range(K)])


def train_unit(unit, tr, ev, ids, y, cfg, a, dev):
    """Train on rows tr; return the model and tokenizer. Resumes from state/<unit>.pt; `ev` rows are the diagnostic probe."""
    torch.manual_seed(cfg['seed'])
    model, tok = load_model(a.base, LABELS, cfg, training=True, device=dev, quant=a.quant)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f'{unit}: {sum(p.numel() for p in params) / 1e6:.1f}M trainable params, {len(tr)} rows, quant={a.quant}')
    steps = batches(len(tr), cfg, cfg['seed'])
    total, warm = len(steps), max(1, math.ceil(cfg['warmup_ratio'] * len(steps)))
    opt = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warm-up, then cosine to 0 (schedule shape is our guess)
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    fp16 = cfg['base_dtype'] == 'float16' and dev != 'cpu'
    scaler = torch.amp.GradScaler('cuda', init_scale=2.0**10, enabled=fp16)
    state = os.path.join(a.out, 'state', f'{unit}.pt')
    start, hist = 0, []
    if os.path.exists(state):
        st = torch.load(state, map_location='cpu', weights_only=False)
        model.load_trainable(st['adapters'])
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        scaler.load_state_dict(st['scaler'])
        start, hist = st['step'], st['hist']
        log(f'{unit}: resumed at step {start}/{total}')
    t0 = time.time()
    for step in range(start, total):
        rows = tr[steps[step]]
        rows = rows[np.argsort([len(ids[i]) for i in rows], kind='stable')]   # less padding; same gradient
        tot = 0.0
        for s in range(0, len(rows), cfg['micro_batch_size']):
            mb = rows[s:s + cfg['micro_batch_size']]
            x, m = pad(tok, [ids[i] for i in mb], model.device)
            loss = model.loss(x, m, torch.as_tensor(y[mb], device=model.device)) * (len(mb) / len(rows))
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, cfg['grad_clip']).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        hist.append(dict(step=step + 1, loss=tot, grad_norm=gn, lr=sched.get_last_lr()[0], scale=scaler.get_scale()))
        el = time.time() - t0
        log(f'{unit} step {step + 1}/{total} loss {tot:.4f} gnorm {gn:.3f} '
            f'({el / (step + 1 - start):.1f}s/step, eta {el / (step + 1 - start) * (total - step - 1) / 60:.0f} min)')
        if a.diag_every and (step + 1) % a.diag_every == 0 and step + 1 < total:
            P, _ = score_rows(model, tok, ids, ev, a.score_batch, a.score_chunk)
            log(f'{unit} diag step {step + 1}: acc {np.mean(P.argmax(1) == y[ev]):.3f} on {len(ev)} probe rows')
            hist[-1]['diag_acc'] = float(np.mean(P.argmax(1) == y[ev]))
        if (step + 1) % a.save_every == 0 and step + 1 < total:
            os.makedirs(os.path.dirname(state), exist_ok=True)
            torch.save(dict(step=step + 1, hist=hist, adapters=model.trainable_state(), opt=opt.state_dict(),
                            sched=sched.state_dict(), scaler=scaler.state_dict()), state + '.tmp')
            os.replace(state + '.tmp', state)   # never a half-written resume point
    return model, tok, hist, time.time() - t0


def write_bundle(model, cfg, hist, a, seconds, train_sha):
    from safetensors.torch import save_file
    out = os.path.join(a.out, 'bundle')
    tmp = out + '.tmp'
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    for f in PKG:
        os.makedirs(os.path.dirname(os.path.join(tmp, f)), exist_ok=True)
        shutil.copy(os.path.join(HERE, 'atlas', f), os.path.join(tmp, f))
    save_file(model.trainable_state(), os.path.join(tmp, 'adapters.safetensors'))
    meta = dict(labels=LABELS, training_config=cfg, completed_steps=len(hist), train_seconds=round(seconds),
                train_sha256=train_sha, train_base_quant=a.quant or 'none', trainer='lab/08_train_atlas.py (ours)',
                method='ATLAS (arXiv:2610.19583): shared LoRA A, per-source B and prefix; prod readout, T=1,2,4,8',
                files={f: sha(os.path.join(tmp, f)) for f in PKG + ['adapters.safetensors']}, history=hist)
    json.dump(meta, open(os.path.join(tmp, 'atlas_bundle.json'), 'w'), indent=1)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out


def merge(a, train, y, folds):
    P, R = np.full((len(train), K), np.nan, np.float32), np.full((len(train), K), np.nan, np.float32)
    for f in range(a.folds):
        z = np.load(os.path.join(a.out, f'oof_fold{f}.npz'))
        P[z['rows']], R[z['rows']] = z['probs'], z['raw']
    assert np.isfinite(P).all(), 'a fold is missing'
    bacc = balanced_accuracy_score(y, P.argmax(1))
    fb = [balanced_accuracy_score(y[folds == f], P[folds == f].argmax(1)) for f in range(a.folds)]
    np.savez_compressed(os.path.join(a.out, 'oof_atlas.npz'), ids=train.id.astype(str).to_numpy(dtype=str),
                        prompt_id=train.prompt_id.astype(str).to_numpy(dtype=str), fold=folds, y=y,
                        labels=np.array(LABELS), probs=P, raw=R)
    s = dict(oof_balanced_accuracy=bacc, fold_balanced_accuracy=fb, n=len(train),
             oof_log_loss=float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)))))
    json.dump(s, open(os.path.join(a.out, 'summary_atlas.json'), 'w'), indent=1)
    log(f'merge: OOF balanced accuracy {bacc:.4f}, per fold {np.round(fb, 4).tolist()}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--base', default=CFG['base_model'], help='HF id (pinned to base_revision) or local dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--quant', choices=['nf4'], default=None, help='4-bit base for training on one 16 GB GPU')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--micro', type=int, help='micro-batch (memory only; same effective batch)')
    ap.add_argument('--score-batch', type=int, default=4)
    ap.add_argument('--score-chunk', type=int, default=6, help='sources per scoring forward pass')
    ap.add_argument('--save-every', type=int, default=10)
    ap.add_argument('--diag-every', type=int, default=44, help='probe accuracy every N steps (0 = off)')
    ap.add_argument('--allow-cpu', action='store_true')
    ap.add_argument('--base-dtype', choices=['float16', 'float32'], help='smoke tests only')
    ap.add_argument('--limit', type=int, help='smoke tests: keep the first N rows')
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch', type=int, help='effective batch (smoke tests)')
    a = ap.parse_args()
    cfg = dict(CFG)
    for k, v in dict(micro_batch_size=a.micro, base_dtype=a.base_dtype, epochs=a.epochs, effective_batch_size=a.batch).items():
        if v is not None:
            cfg[k] = v
    if not os.path.isdir(a.base):
        from huggingface_hub import snapshot_download
        a.base = snapshot_download(a.base, revision=cfg['base_revision'],
                                   allow_patterns=['*.json', '*.safetensors', '*.txt', 'merges.txt'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu' and not a.allow_cpu:
        raise SystemExit('no GPU (pass --allow-cpu for a smoke test)')
    os.makedirs(a.out, exist_ok=True)

    train = pd.read_csv(a.train)
    train_sha = sha(a.train)
    if a.limit:
        train = train.iloc[:a.limit].reset_index(drop=True)
    y = train.model.map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    assert not np.isnan(y.astype(float)).any(), 'unknown label'
    folds = np.zeros(len(train), int)
    for f, (_, va) in enumerate(GroupKFold(n_splits=a.folds).split(train, groups=train.prompt_id)):
        folds[va] = f
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(a.base)(train.text.tolist(), truncation=True, max_length=cfg['max_length'])['input_ids']

    units = (['fold0', 'full'] + [f'fold{k}' for k in range(1, a.folds)] + ['merge']) if a.unit == 'all' else [a.unit]
    for unit in units:
        if unit == 'merge':
            merge(a, train, y, folds)
            continue
        done = os.path.join(a.out, 'bundle', 'atlas_bundle.json') if unit == 'full' else os.path.join(a.out, f'oof_{unit}.npz')
        if os.path.exists(done):
            log(f'{unit}: cached ({done})')
            continue
        if unit == 'full':
            tr, va = np.arange(len(train)), None
            ev = diag_rows(y, tr)   # in-sample probe: training health, and the nf4-vs-fp16 check rows
        else:
            f = int(unit[4:])
            tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
            ev = diag_rows(y, va)   # held-out probe: an early OOF read
        model, tok, hist, secs = train_unit(unit, tr, ev, ids, y, cfg, a, dev)
        t1 = time.time()
        if unit == 'full':
            Pd, Rd = score_rows(model, tok, ids, ev, a.score_batch, a.score_chunk)
            hist.append(dict(final_probe_rows=ev.tolist(), final_probe_acc=float(np.mean(Pd.argmax(1) == y[ev])),
                             final_probe_probs=np.round(Pd, 6).tolist()))
            log(f'full: bundle at {write_bundle(model, cfg, hist, a, secs, train_sha)}, '
                f'in-sample probe acc {np.mean(Pd.argmax(1) == y[ev]):.3f}')
        else:
            P, R = score_rows(model, tok, ids, va, a.score_batch, a.score_chunk)
            bacc = balanced_accuracy_score(y[va], P.argmax(1))
            np.savez_compressed(done + '.tmp.npz', rows=va, ids=train.id.astype(str).to_numpy(dtype=str)[va], y=y[va],
                                fold=folds[va], labels=np.array(LABELS), probs=P, raw=R,
                                hist=json.dumps(hist), train_seconds=secs, score_seconds=time.time() - t1,
                                quant=a.quant or 'none')
            os.replace(done + '.tmp.npz', done)
            log(f'{unit}: held-out balanced accuracy {bacc:.4f} on {len(va)} rows '
                f'(train {secs / 60:.0f} min, score {(time.time() - t1) / 60:.0f} min)')
        st = os.path.join(a.out, 'state', f'{unit}.pt')
        if os.path.exists(st):
            os.remove(st)
        del model
        if dev == 'cuda':
            torch.cuda.empty_cache()
    if a.unit == 'all' or a.unit.startswith('fold'):
        if all(os.path.exists(os.path.join(a.out, f'oof_fold{f}.npz')) for f in range(a.folds)) and \
                not os.path.exists(os.path.join(a.out, 'summary_atlas.json')):
            merge(a, train, y, folds)


if __name__ == '__main__':
    main()
