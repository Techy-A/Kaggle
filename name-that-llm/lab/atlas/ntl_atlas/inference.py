"""predict(texts, base_path, bundle_path, batch_size=1) -> (n, 18) float32 probabilities in ntl_rlcd.data.LABELS order."""
import gc
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from ntl_rlcd.data import LABELS
from .model import load_model


@torch.inference_mode()
def predict(texts, base_path, bundle_path, batch_size=1, source_chunk=1):
    root = Path(bundle_path)
    meta = json.loads((root / 'atlas_bundle.json').read_text())
    if meta['labels'] != LABELS:
        raise ValueError('bundle label order differs from ntl_rlcd.data.LABELS')
    cfg = meta['training_config']
    model, tok = load_model(base_path, LABELS, cfg, device='auto')
    try:
        model.load_trainable(load_file(str(root / 'adapters.safetensors')))
        ids = tok(list(texts), truncation=True, max_length=cfg['max_length'])['input_ids']
        order = np.argsort([len(t) for t in ids], kind='stable')   # shortest first: least padding
        out = np.zeros((len(ids), len(LABELS)), np.float32)
        for s in range(0, len(order), batch_size):
            rows = order[s:s + batch_size]
            pack = tok.pad({'input_ids': [ids[i] for i in rows]}, padding=True, return_tensors='pt').to(model.device)
            out[rows] = model.score(pack['input_ids'], pack['attention_mask'], source_chunk=source_chunk)[0].float().cpu().numpy()
        if not np.isfinite(out).all() or not np.allclose(out.sum(1), 1, atol=1e-4):
            raise RuntimeError('invalid ATLAS probabilities')
        return out
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
