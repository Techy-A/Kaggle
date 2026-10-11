"""Spectrum -> Morgan fingerprint model, and candidate ranking by fingerprint likelihood.

    python fpmodel.py prep  <train.parquet> <work_dir>              # cache cleaned peaks of the training spectra
    python fpmodel.py train <work_dir> [--exclude-validation] [--epochs N]
"""
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi import TEST_ADDUCTS, clean_peaks

MZ_MAX, MZ_BIN = 1000.0, 0.5          # fragment bins
LOSS_MAX, LOSS_BIN = 500.0, 1.0       # neutral-loss bins (precursor m/z - fragment m/z)
N_FRAG, N_LOSS = int(MZ_MAX / MZ_BIN), int(LOSS_MAX / LOSS_BIN)
N_META = len(TEST_ADDUCTS) + 4        # adduct one-hot, ion mode, collision energy, precursor m/z, n peaks
N_IN = N_FRAG + N_LOSS + N_META
ADDUCT_IDX = {a: i for i, a in enumerate(TEST_ADDUCTS)}


# ---------------------------------------------------------------- features

class SpectrumStore:
    """Cleaned peaks of many spectra, stored flat; featurises batches without Python loops over peaks."""

    def __init__(self, mz, it, offsets, info):
        self.mz, self.it, self.offsets, self.info = mz, it, offsets, info.reset_index(drop=True)
        self.adduct = info.adduct.map(ADDUCT_IDX).to_numpy()
        self.neg = (info.ionization_mode == 'negative').to_numpy(np.float32)
        self.ce = info.ce.to_numpy(np.float32)
        self.prec = info.precursor_mz.to_numpy(np.float32)

    @classmethod
    def from_frame(cls, df, top_k=60):
        mzs, its, lens = [], [], []
        for p, a, b in zip(df.precursor_mz.to_numpy(), df.ms2_mzs, df.ms2_normalized_intensities):
            m, i = clean_peaks(a, b, p, top_k)
            mzs.append(m), its.append(i), lens.append(len(m))
        offsets = np.zeros(len(lens) + 1, np.int64)
        offsets[1:] = np.cumsum(lens)
        info = df[['adduct', 'ionization_mode', 'precursor_mz']].copy()
        info['ce'] = [float(np.mean(c)) if c is not None and len(c) else 30.0 for c in df.collision_energy_ev]
        if 'inchikey14' in df:
            info['inchikey14'] = df.inchikey14.to_numpy()
        return cls(np.concatenate(mzs) if mzs else np.zeros(0, np.float32),
                   np.concatenate(its) if its else np.zeros(0, np.float32), offsets, info)

    def features(self, rows):
        rows = np.asarray(rows)
        X = np.zeros((len(rows), N_IN), np.float32)
        starts, ends = self.offsets[rows], self.offsets[rows + 1]
        lens = ends - starts
        r = np.repeat(np.arange(len(rows)), lens)
        idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)]) if lens.sum() else np.zeros(0, int)
        mz, it = self.mz[idx], self.it[idx]
        fb = np.minimum((mz / MZ_BIN).astype(np.int64), N_FRAG - 1)
        np.maximum.at(X, (r, fb), it)
        loss = self.prec[rows][r] - mz
        ok = (loss > 0.5) & (loss < LOSS_MAX)
        lb = N_FRAG + (loss[ok] / LOSS_BIN).astype(np.int64)
        np.maximum.at(X, (r[ok], lb), it[ok])
        m0 = N_FRAG + N_LOSS
        X[np.arange(len(rows)), m0 + self.adduct[rows]] = 1
        X[:, m0 + len(TEST_ADDUCTS)] = self.neg[rows]
        X[:, m0 + len(TEST_ADDUCTS) + 1] = self.ce[rows] / 100
        X[:, m0 + len(TEST_ADDUCTS) + 2] = self.prec[rows] / 1000
        X[:, m0 + len(TEST_ADDUCTS) + 3] = lens / 60
        return X

    def save(self, path):
        np.savez(path, mz=self.mz, it=self.it, offsets=self.offsets)
        self.info.to_parquet(path + '.info.parquet')

    @classmethod
    def load(cls, path):
        z = np.load(path + '.npz')
        return cls(z['mz'], z['it'], z['offsets'], pd.read_parquet(path + '.info.parquet'))


def prep(train_path, work, max_ppm=20, batch_size=100_000):
    cols = ['inchikey14', 'ingest_lib', 'adduct', 'ionization_mode', 'precursor_mz', 'precursor_error_ppm',
            'collision_energy_ev', 'ms2_mzs', 'ms2_normalized_intensities']
    stores = []
    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=batch_size, columns=cols):
        df = batch.to_pandas()
        df = df[df.adduct.isin(TEST_ADDUCTS) & (df.precursor_error_ppm.abs() < max_ppm)]
        stores.append(SpectrumStore.from_frame(df))
        stores[-1].info['ingest_lib'] = df.ingest_lib.to_numpy()
    mz = np.concatenate([s.mz for s in stores])
    it = np.concatenate([s.it for s in stores])
    lens = np.concatenate([np.diff(s.offsets) for s in stores])
    offsets = np.zeros(len(lens) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    store = SpectrumStore(mz, it, offsets, pd.concat([s.info for s in stores], ignore_index=True))
    store.save(f'{work}/train_spectra')
    print(f'cached {len(store.info):,} spectra, {len(mz):,} peaks')


# ---------------------------------------------------------------- model

def make_model(n_out, hidden=1536, dropout=0.2):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(N_IN, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hidden, n_out))


def train(work, exclude_validation=False, epochs=4, batch=512, lr=1e-3, max_per_struct=8, seed=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = SpectrumStore.load(f'{work}/train_spectra')
    structs = pd.read_parquet(f'{work}/structures.parquet')
    fps = np.load(f'{work}/fps.npy')
    row_of = pd.Series(np.arange(len(structs)), index=structs.ik14)
    info = store.info
    keep = info.inchikey14.isin(row_of.index).to_numpy().copy()
    if exclude_validation:   # never train on any validation structure
        val_keys = set(info.loc[info.ingest_lib == 'enveda-np-examples', 'inchikey14'])
        keep &= ~info.inchikey14.isin(val_keys).to_numpy()
    idx = np.flatnonzero(keep)
    # cap spectra per structure so the 180k-structure synthetic library doesn't drown everything else
    order = rng.permutation(idx)
    grp = info.inchikey14.to_numpy()[order]
    rank = pd.Series(grp).groupby(grp).cumcount().to_numpy()
    idx = np.sort(order[rank < max_per_struct])
    y_row = row_of.reindex(info.inchikey14.to_numpy()[idx]).to_numpy()
    print(f'training on {len(idx):,} spectra of {len(set(y_row)):,} structures')

    model = make_model(fps.shape[1] * 8)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(idx) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    lossf = torch.nn.BCEWithLogitsLoss()
    t, step = time.time(), 0
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(len(idx))
        total = 0.0
        for b in range(len(idx) // batch):
            sel = perm[b * batch:(b + 1) * batch]
            X = torch.from_numpy(store.features(idx[sel]))
            Y = torch.from_numpy(np.unpackbits(fps[y_row[sel]], axis=1).astype(np.float32))
            loss = lossf(model(X), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
            step += 1
            if step % 500 == 0:
                print(f'epoch {ep} step {step}/{steps} loss {total / (b + 1):.4f} {time.time() - t:.0f}s', flush=True)
        name = 'fpnet_val' if exclude_validation else 'fpnet_full'
        torch.save(model.state_dict(), f'{work}/{name}.pt')
    print('done', time.time() - t)


# ---------------------------------------------------------------- ranking

def predict_logits(model, store, rows, batch=2048):
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(rows), batch):
            out.append(model(torch.from_numpy(store.features(rows[b:b + batch]))).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def fp_loglik(logits, cand_fps):
    """Sum over spectra of log P(candidate fingerprint | spectrum), bits independent.

    logits: (n_spectra, n_bits); cand_fps: (n_cands, n_bits/8) packed -> (n_cands,)
    """
    bits = np.unpackbits(cand_fps, axis=1).astype(np.float32)
    log_p = -np.logaddexp(0, -logits)        # log sigmoid
    log_q = -np.logaddexp(0, logits)         # log (1 - sigmoid)
    return (bits @ (log_p - log_q).sum(0)) + log_q.sum()


def gated_order(fp, cos, gate=0.7):
    """Candidate order: confident library matches (cosine >= gate) first by cosine, the rest by fingerprint likelihood."""
    score = np.where(cos >= gate, 1e6 + cos, fp - fp.max())
    return np.argsort(-score, kind='stable')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(sys.argv[2], sys.argv[3])
    elif cmd == 'train':
        epochs = int(sys.argv[sys.argv.index('--epochs') + 1]) if '--epochs' in sys.argv else 4
        train(sys.argv[2], exclude_validation='--exclude-validation' in sys.argv, epochs=epochs)
