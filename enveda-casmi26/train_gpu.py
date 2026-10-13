"""GPU training of the spectrum -> fingerprint model (v3: bigger, deeper, more spectra per structure, more epochs).

    python train_gpu.py <work_dir> --name fpnet_v3h --holdout <holdout.txt> --exclude-validation
    python train_gpu.py <work_dir> --name fpnet_v3_full

Same inputs and outputs as fpmodel.train (train_spectra cache, structures.parquet, packed fingerprint file), so the
weights load with fpmodel.make_model(n_out, hidden, depth=...) and run on CPU in pipeline.py. Featurisation runs
on the GPU (scatter-max into the same bins as SpectrumStore.features, checked by `selftest`), so the GPU is never
waiting for NumPy.
"""
import sys
import time

import numpy as np
import pandas as pd

from casmi import TEST_ADDUCTS
from fpmodel import LOSS_BIN, LOSS_MAX, MZ_BIN, N_FRAG, N_IN, N_LOSS, SpectrumStore, make_model


class GpuStore:
    """SpectrumStore peaks and metadata as device tensors; features() mirrors SpectrumStore.features."""

    def __init__(self, store, device):
        import torch
        self.device = device
        t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        self.mz, self.it = t(store.mz, torch.float32), t(store.it, torch.float32)
        self.offsets = t(store.offsets, torch.int64)
        self.adduct = t(store.adduct, torch.int64)
        self.neg, self.ce, self.prec = (t(store.neg, torch.float32), t(store.ce, torch.float32),
                                        t(store.prec, torch.float32))

    def features(self, rows):
        import torch
        rows = torch.as_tensor(rows, device=self.device)
        n = len(rows)
        starts = self.offsets[rows]
        lens = self.offsets[rows + 1] - starts
        r = torch.repeat_interleave(torch.arange(n, device=self.device), lens)
        first = torch.cumsum(lens, 0) - lens
        idx = torch.repeat_interleave(starts - first, lens) + torch.arange(int(lens.sum()), device=self.device)
        mz, it = self.mz[idx], self.it[idx]
        X = torch.zeros(n * N_IN, device=self.device)
        fb = torch.clamp((mz / MZ_BIN).long(), max=N_FRAG - 1)
        X.scatter_reduce_(0, r * N_IN + fb, it, 'amax')
        loss = self.prec[rows][r] - mz
        ok = (loss > 0.5) & (loss < LOSS_MAX)
        lb = N_FRAG + (loss[ok] / LOSS_BIN).long()
        X.scatter_reduce_(0, r[ok] * N_IN + lb, it[ok], 'amax')
        X = X.view(n, N_IN)
        m0 = N_FRAG + N_LOSS
        ar = torch.arange(n, device=self.device)
        X[ar, m0 + self.adduct[rows]] = 1
        X[:, m0 + len(TEST_ADDUCTS)] = self.neg[rows]
        X[:, m0 + len(TEST_ADDUCTS) + 1] = self.ce[rows] / 100
        X[:, m0 + len(TEST_ADDUCTS) + 2] = self.prec[rows] / 1000
        X[:, m0 + len(TEST_ADDUCTS) + 3] = lens.float() / 60
        return X


def select_rows(info, row_of, max_per_struct, rng, exclude=()):
    keep = info.inchikey14.isin(row_of.index).to_numpy() & ~info.inchikey14.isin(set(exclude)).to_numpy()
    order = rng.permutation(np.flatnonzero(keep))
    grp = info.inchikey14.to_numpy()[order]
    rank = pd.Series(grp).groupby(grp).cumcount().to_numpy()
    return np.sort(order[rank < max_per_struct])


def train(work, name, fps_file='fps_v2.npy', hidden=4096, depth=3, dropout=0.2, epochs=25, batch=1024, lr=1e-3,
          weight_decay=1e-4, max_per_struct=64, exclude_validation=False, holdout_file=None, seed=0,
          time_budget=None):
    """Train and save <work>/<name>.pt. time_budget (seconds) stops early, keeping the last finished epoch."""
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    store = SpectrumStore.load(f'{work}/train_spectra')
    structs = pd.read_parquet(f'{work}/structures.parquet', columns=['ik14'])
    fps = np.load(f'{work}/{fps_file}')
    row_of = pd.Series(np.arange(len(structs)), index=structs.ik14)
    row_of = row_of[~row_of.index.duplicated()]
    info = store.info
    exclude = set()
    if exclude_validation:
        exclude |= set(info.loc[info.ingest_lib == 'enveda-np-examples', 'inchikey14'])
    if holdout_file:
        exclude |= set(open(holdout_file).read().split())
    idx = select_rows(info, row_of, max_per_struct, rng, exclude)
    y_row = row_of.reindex(info.inchikey14.to_numpy()[idx]).to_numpy()
    print(f'{name}: {len(idx):,} spectra of {len(set(y_row)):,} structures on {device}', flush=True)

    gs = GpuStore(store, device)
    Y_all = torch.as_tensor(fps, device=device)                     # packed bits, unpacked per batch
    shifts = torch.arange(7, -1, -1, device=device, dtype=torch.uint8)
    idx_t = torch.as_tensor(idx, device=device)
    y_t = torch.as_tensor(y_row, device=device)
    model = make_model(fps.shape[1] * 8, hidden=hidden, dropout=dropout, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = len(idx) // batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=device == 'cuda')
    lossf = torch.nn.BCEWithLogitsLoss()
    t = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(idx), device=device)
        total = torch.zeros((), device=device)
        for b in range(steps_per_epoch):
            sel = perm[b * batch:(b + 1) * batch]
            X = gs.features(idx_t[sel])
            Y = ((Y_all[y_t[sel]].unsqueeze(-1) >> shifts) & 1).flatten(1).float()
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == 'cuda'):
                loss = lossf(model(X).float(), Y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.detach()
        torch.save(model.state_dict(), f'{work}/{name}.pt')
        print(f'{name} epoch {ep + 1}/{epochs} loss {total.item() / steps_per_epoch:.4f} {time.time() - t:.0f}s',
              flush=True)
        if time_budget and (time.time() - t) / (ep + 1) * (ep + 2) > time_budget:
            print(f'{name}: stopping after epoch {ep + 1} to stay inside the time budget', flush=True)
            break
    return model


def selftest(work, n=257):
    """GPU featurisation must equal the CPU featurisation the submission pipeline uses."""
    import torch
    store = SpectrumStore.load(f'{work}/train_spectra')
    rows = np.random.default_rng(1).choice(len(store.info), n, replace=False)
    cpu = store.features(rows)
    gpu = GpuStore(store, 'cuda' if torch.cuda.is_available() else 'cpu').features(rows).cpu().numpy()
    err = np.abs(cpu - gpu).max()
    print(f'featurisation max abs difference CPU vs device: {err:.2e}')
    assert err < 1e-5, err


if __name__ == '__main__':
    args = sys.argv[1:]

    def opt(flag, default, cast=str):
        return cast(args[args.index(flag) + 1]) if flag in args else default
    if args[0] == 'selftest':
        selftest(args[1])
    else:
        train(args[0], opt('--name', 'fpnet_v3'), fps_file=opt('--fps', 'fps_v2.npy'),
              hidden=opt('--hidden', 4096, int), depth=opt('--depth', 3, int), epochs=opt('--epochs', 25, int),
              max_per_struct=opt('--max-per-struct', 64, int), exclude_validation='--exclude-validation' in args,
              holdout_file=opt('--holdout', None), time_budget=opt('--time-budget', None, float))
