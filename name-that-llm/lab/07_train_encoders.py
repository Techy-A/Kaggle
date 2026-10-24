"""Step 7: train the two encoder members ("OURS") of the 0.624-0.627 NTL notebooks, with prompt-grouped OOF (GPU).

What: one member per run, either ModernBERT-large or the desklib DeBERTa-v3-large AI-detector, fine-tuned in two stages
as in nuridudak's write-up (see encoder_rebuild_spec.md).
- Stage 1: the public pool (nishantdahal's pool_v4, minus passages close to a train passage), class-weighted CE.
- Stage 2: the official train.
CV is GroupKFold(5) on prompt_id. Stage 1 runs once (it never sees train), each fold runs stage 2 from the stage-1 weights
and predicts its held-out prompts, and the shipped model is stage 2 on the whole train.

Why: the members' dataset nuridudak/ntl-ours-v1 went private, so those notebooks drop to the 0.620 base. This rebuilds
the members as our own work and measures them out of fold before anything is submitted.

Output (in --out):
- <member>/ is an AutoModelForSequenceClassification dir (fp16 safetensors, tokenizer, id2label alphabetical), the layout
  that enc_infer.py in the notebooks loads.
- oof_<member>.npz holds ids, prompt_id, fold, labels, y, oof logits and stage1 logits (train predicted by the
  stage-1-only model).
- summary_<member>.json holds balanced accuracies and timings.
An existing summary_<member>.json means the member is done and the run is skipped (the cache).

Usage:
  python 07_train_encoders.py --member modernbert --train train.csv --pool pool_v4.parquet --out OUT
  python 07_train_encoders.py --member desklib    --train train.csv --pool pool_v4.parquet --out OUT
  CPU smoke test: add --allow-cpu --backbone <tiny HF id or dir> --limit-pool 200 --stage1-epochs 1 --stage2-epochs 1
"""
import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

LABELS = sorted(['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview',
                 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8',
                 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash',
                 'kimi-k3'])   # alphabetical, as in nuridudak's heads; the notebooks remap by name anyway
K = len(LABELS)
MAX_LEN = 512   # enc_infer.py hard-codes 512
MEMBERS = {   # published recipe; micro = micro-batch for gradient accumulation (memory only, same maths)
    'modernbert': dict(backbone='answerdotai/ModernBERT-large', lr1=2e-5, bs1=32, micro=16, ckpt=False),
    'desklib': dict(backbone='desklib/ai-text-detector-v1.01', lr1=2.8e-5, bs1=64, micro=16, ckpt=True),
}
LEAK_COS = 0.3   # Nishant's filter: pool passages with content TF-IDF cosine >= 0.3 to a train passage are out


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def leak_filter(pool_texts, train_texts):
    """Boolean mask of pool passages kept: max content cosine to any train passage below LEAK_COS."""
    vec = TfidfVectorizer(stop_words='english', sublinear_tf=True).fit(list(train_texts) + list(pool_texts))
    P, T = vec.transform(pool_texts), vec.transform(train_texts)
    best = np.zeros(P.shape[0], np.float32)
    for s in range(0, P.shape[0], 5000):
        best[s:s + 5000] = (P[s:s + 5000] @ T.T).max(1).toarray().ravel()
    return best < LEAK_COS


def build_model(backbone, member):
    """The classifier at its starting point, and the backbone's tokenizer."""
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    path = backbone
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(backbone, allow_patterns=['*.json', '*.safetensors', '*.model', '*.txt'],
                                 ignore_patterns=['onnx/*'])
    kw = dict(num_labels=K, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    cfg = AutoConfig.from_pretrained(path, **kw)
    cfg.architectures = None
    if cfg.model_type == 'modernbert':
        cfg.classifier_pooling = 'mean'   # as in nuridudak's config.json
    st = os.path.join(path, 'model.safetensors')
    sd = load_file(st) if os.path.exists(st) else None
    if cfg.model_type == 'deberta-v2' and sd is not None and any(k.startswith('model.') for k in sd):
        # desklib stores its encoder under model.* with a 1-logit classifier: rename to deberta.*, start a fresh head
        model = AutoModelForSequenceClassification.from_config(cfg)
        sd = {'deberta.' + k[len('model.'):]: v for k, v in sd.items() if k.startswith('model.')}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        assert not unexpected and all(k.startswith(('pooler.', 'classifier.')) for k in missing), (missing, unexpected)
        log(f'desklib encoder loaded, fresh head: {missing}')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path, config=cfg, ignore_mismatched_sizes=True)
    tok = AutoTokenizer.from_pretrained(path)
    if MEMBERS[member]['ckpt']:
        model.gradient_checkpointing_enable()
    return model, tok


def encode(tok, texts):
    return tok(list(texts), truncation=True, max_length=MAX_LEN)['input_ids']   # exactly enc_infer.py's call


def amp_dtype(dev):
    if dev.type != 'cuda':
        return None
    return torch.bfloat16 if torch.cuda.get_device_capability(dev)[0] >= 8 else torch.float16


def batches(ids, bs, rng):
    """Length-grouped batches: shuffle, sort within chunks of 50 batches, shuffle the batches."""
    order = rng.permutation(len(ids))
    out = []
    for s in range(0, len(order), bs * 50):
        chunk = sorted(order[s:s + bs * 50], key=lambda i: len(ids[i]))
        out += [chunk[j:j + bs] for j in range(0, len(chunk), bs)]
    return [out[i] for i in rng.permutation(len(out))]


def fit(model, tok, ids, y, epochs, lr, bs, micro, dev, class_weight=None, max_secs=None, seed=42, tag=''):
    """AdamW (wd 0.01), 10% linear warm-up then cosine to 0, grad clip 1.0. If max_secs is set and the run would
    overrun it, the schedule is shortened after 30 steps (the cosine still reaches 0)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    plan = [b for _ in range(epochs) for b in batches(ids, bs, rng)]
    state = dict(total=len(plan))
    warm = max(1, int(0.1 * state['total']))

    def factor(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, state['total'] - warm))))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, factor)
    dt = amp_dtype(dev)
    scaler = torch.amp.GradScaler('cuda', enabled=dt == torch.float16)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w, reduction='sum')
    yt = torch.tensor(y, device=dev)
    model.train()
    t0, run, nrun = time.time(), 0.0, 0
    for step, b in enumerate(plan):
        if step >= state['total']:
            break
        denom = float(w[yt[b]].sum()) if w is not None else float(len(b))
        for m in range(0, len(b), micro):
            mb = b[m:m + micro]
            enc = tok.pad({'input_ids': [ids[i] for i in mb]}, return_tensors='pt').to(dev)
            with torch.autocast(dev.type, dtype=dt, enabled=dt is not None):
                logits = model(**enc).logits.float()
            loss = lossf(logits, yt[mb]) / denom
            scaler.scale(loss).backward()
            run += loss.item() * denom / len(b)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        if step == 30 and max_secs:
            per = (time.time() - t0) / 31
            if per * state['total'] > max_secs:
                state['total'] = max(warm + 1, int(max_secs / per))
                log(f'{tag} time budget: schedule cut to {state["total"]} of {len(plan)} steps ({per:.2f}s/step)')
        nrun += 1
        if (step + 1) % 200 == 0 or step + 1 == state['total']:
            log(f'{tag} step {step + 1}/{state["total"]} loss {run / nrun:.4f} {time.time() - t0:.0f}s')
            run, nrun = 0.0, 0
    model.eval()
    return dict(steps=min(state['total'], len(plan)), planned=len(plan), secs=time.time() - t0)


@torch.inference_mode()
def predict(model, tok, ids, dev):
    """Raw logits, the way enc_infer.py computes them: length-sorted batches of 64, fp16 autocast, fp32 retry."""
    model.eval()
    order = np.argsort([len(x) for x in ids])
    out = np.zeros((len(ids), K), np.float32)
    dt = amp_dtype(dev)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        b = tok.pad({'input_ids': [ids[i] for i in idx]}, return_tensors='pt').to(dev)
        with torch.autocast(dev.type, dtype=torch.float16 if dt else None, enabled=dt is not None):
            lg = model(**b).logits.float()
        if not torch.isfinite(lg).all():
            lg = model(**b).logits.float()
        out[idx] = lg.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', choices=list(MEMBERS), required=True)
    ap.add_argument('--train', required=True)
    ap.add_argument('--pool', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--backbone', help='override the HF id / local dir (tiny models for the CPU smoke test)')
    ap.add_argument('--stage1-epochs', type=int, default=2)
    ap.add_argument('--stage2-epochs', type=int, default=2)
    ap.add_argument('--stage2-lr', type=float, default=1e-5)
    ap.add_argument('--stage2-bs', type=int, default=32)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--stage1-hours', type=float, default=None, help='cap on stage 1 wall time (shortens the schedule)')
    ap.add_argument('--limit-pool', type=int, default=None, help='random subset of the pool (smoke tests)')
    ap.add_argument('--allow-cpu', action='store_true')
    a = ap.parse_args()
    cfgm = dict(MEMBERS[a.member])
    os.makedirs(a.out, exist_ok=True)
    summ_path = os.path.join(a.out, f'summary_{a.member}.json')
    if os.path.exists(summ_path):
        log(f'{summ_path} exists, nothing to do')
        return
    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit('no CUDA GPU in this session (pass --allow-cpu only for smoke tests)')
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    T0 = time.time()

    train = pd.read_csv(a.train)
    train['text'] = train['text'].fillna('').astype(str)
    pool = pd.read_parquet(a.pool)
    pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
    keep = leak_filter(pool.text.tolist(), train.text.tolist()) & ~pool.text.isin(set(train.text))
    log(f'pool {len(pool)}, kept {int(keep.sum())} after the train-similarity filter')
    pool = pool[keep].reset_index(drop=True)
    if a.limit_pool:
        pool = pool.sample(n=min(a.limit_pool, len(pool)), random_state=0).reset_index(drop=True)
    lab = {l: i for i, l in enumerate(LABELS)}
    y_tr, y_pool = train.model.map(lab).values, pool.model.map(lab).values

    model, tok = build_model(a.backbone or cfgm['backbone'], a.member)
    model.to(dev)
    ids_tr, ids_pool = encode(tok, train.text), encode(tok, pool.text)
    log(f'tokens: pool mean {np.mean([len(x) for x in ids_pool]):.0f}, max {max(len(x) for x in ids_pool)}; '
        f'train mean {np.mean([len(x) for x in ids_tr]):.0f}')

    # The longest micro-batch first, so an out-of-memory error shows in minute one, not hour three.
    longest = list(np.argsort([len(x) for x in ids_pool])[-cfgm['micro']:])
    model.train()
    enc = tok.pad({'input_ids': [ids_pool[i] for i in longest]}, return_tensors='pt').to(dev)
    with torch.autocast(dev.type, dtype=amp_dtype(dev), enabled=amp_dtype(dev) is not None):
        model(**enc).logits.float().sum().backward()
    model.zero_grad(set_to_none=True)
    if dev.type == 'cuda':
        log(f'memory probe ok: peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB')

    # Stage 1: pool only, class-weighted (inverse frequency, mean 1).
    cnt = np.bincount(y_pool, minlength=K).astype(float)
    cw = np.where(cnt > 0, cnt.sum() / np.maximum(cnt, 1) / K, 0.0)
    info1 = fit(model, tok, ids_pool, y_pool, a.stage1_epochs, cfgm['lr1'], cfgm['bs1'], cfgm['micro'], dev, class_weight=cw,
                max_secs=a.stage1_hours * 3600 if a.stage1_hours else None, tag='stage1')
    sd1 = {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}
    lg1 = predict(model, tok, ids_tr, dev)
    ba1 = balanced_accuracy_score(y_tr, lg1.argmax(1))
    log(f'stage 1 only, on train: balanced accuracy {ba1:.4f}')

    # Stage 2 per fold, from the stage-1 weights.
    folds = np.zeros(len(train), int)
    for f, (_, va) in enumerate(GroupKFold(n_splits=a.folds).split(train, groups=train.prompt_id)):
        folds[va] = f
    oof = np.zeros((len(train), K), np.float32)
    info2 = []
    for f in range(a.folds):
        model.load_state_dict(sd1)
        tr, va = np.where(folds != f)[0], np.where(folds == f)[0]
        info2.append(fit(model, tok, [ids_tr[i] for i in tr], y_tr[tr], a.stage2_epochs, a.stage2_lr, a.stage2_bs,
                         cfgm['micro'], dev, seed=42 + f, tag=f'fold{f}'))
        oof[va] = predict(model, tok, [ids_tr[i] for i in va], dev)
        log(f'fold {f}: balanced accuracy {balanced_accuracy_score(y_tr[va], oof[va].argmax(1)):.4f}')
    ba = balanced_accuracy_score(y_tr, oof.argmax(1))
    log(f'OOF after stage 2: balanced accuracy {ba:.4f}')
    np.savez(os.path.join(a.out, f'oof_{a.member}.npz'), ids=train.id.astype(str).to_numpy(dtype=str),
             prompt_id=train.prompt_id.astype(str).to_numpy(dtype=str), fold=folds, y=y_tr,
             labels=np.array(LABELS, dtype=str), oof=oof, stage1=lg1)

    # Shipped model: stage 2 on the whole train, saved for enc_infer.py, then reloaded the way enc_infer.py loads it.
    model.load_state_dict(sd1)
    del sd1
    info_f = fit(model, tok, ids_tr, y_tr, a.stage2_epochs, a.stage2_lr, a.stage2_bs, cfgm['micro'], dev, seed=42,
                 tag='final')
    lg_mem = predict(model, tok, ids_tr[:256], dev)
    out_dir = os.path.join(a.out, a.member)
    if MEMBERS[a.member]['ckpt']:
        model.gradient_checkpointing_disable()
    model.half().save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)
    del model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    re_tok = AutoTokenizer.from_pretrained(out_dir)
    re_model = AutoModelForSequenceClassification.from_pretrained(out_dir, dtype=torch.float32).to(dev).eval()
    assert [re_model.config.id2label[i] for i in range(re_model.config.num_labels)] == LABELS
    lg_re = predict(re_model, re_tok, encode(re_tok, train.text[:256]), dev)
    agree = float((lg_re.argmax(1) == lg_mem.argmax(1)).mean())
    log(f'reloaded {out_dir}: top-1 agreement with the in-memory model {agree:.3f} on 256 train rows')
    assert agree >= 0.97, agree

    summ = dict(member=a.member, backbone=a.backbone or cfgm['backbone'], recipe=cfgm, pool_used=len(pool),
                stage1=info1, stage1_only_bacc=ba1, oof_bacc=ba,
                fold_bacc=[balanced_accuracy_score(y_tr[folds == f], oof[folds == f].argmax(1)) for f in range(a.folds)],
                stage2=info2, final=info_f, reload_agreement=agree, total_secs=time.time() - T0,
                device=torch.cuda.get_device_name(dev) if dev.type == 'cuda' else 'cpu',
                torch=torch.__version__, transformers=__import__('transformers').__version__)
    json.dump(summ, open(summ_path, 'w'), indent=1, default=float)
    log(f'done in {time.time() - T0:.0f}s')


if __name__ == '__main__':
    main()
