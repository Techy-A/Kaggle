"""62: DreaMS retriever head, gate test (step 3 of the DreaMS plan).

What: frozen DreaMS molecule embeddings (jobs/dreams/dreams_embed.py: mean of up to 2 spectra per training molecule,
1024-d) -> MLP -> 2048-bit Morgan fingerprint (radius 2) probabilities. Candidates are ranked by cosine between the
predicted probabilities and each candidate's fingerprint. Scored on exactly the 1,000 held-out molecules and 256
candidates per molecule that our own scorer s2 (casmi-scorer-train-s2, seed 20261103) was scored on, with the kit's
pessimistic-tie MRR@25; those 1,000 molecules are excluded from training. Paired bootstrap against s2 per molecule.
Why: the plan's gate: only a model at least as good as the public scorer (0.660 on its own holdout; s2 0.641 on this
one) goes into a submission. Output: artifacts/62_dreams_head.json, 62_per_molecule.csv. CPU, about 10-20 min.
Usage: python enveda-casmi26/62_dreams_head.py [--epochs 8] [--hidden 1536]
"""
import argparse, json, os, time
import numpy as np, pandas as pd, torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
KIT, EMB, S2 = '/home/user/work/kit', '/home/user/work/out_dreams_embed', '/home/user/work/out_s2/casmi_spectrum_model_aryan_s2'


def fingerprints(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = np.zeros((len(smiles), 2048), np.uint8)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None: out[i] = gen.GetFingerprintAsNumPy(m)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=8); ap.add_argument('--hidden', type=int, default=1536)
    a = ap.parse_args(); torch.manual_seed(0); np.random.seed(0); t0 = time.time()
    mol = pd.read_parquet(f'{KIT}/molecules.parquet')
    fpp = f'{ART}/62_fps.npy'
    if os.path.exists(fpp): FP = np.load(fpp)
    else: FP = fingerprints(mol.smiles.tolist()); np.save(fpp, FP)
    print('fps', FP.shape, round(time.time() - t0), 's', flush=True)
    X = np.load(f'{EMB}/train_emb.npy', mmap_mode='r'); keys = pd.read_csv(f'{EMB}/train_keys.csv').inchikey14.values
    row = {k: i for i, k in enumerate(keys)}
    cand = np.load(f'{S2}/validation_candidates.npz'); s2 = pd.read_csv(f'{S2}/validation_predictions.csv')
    ev = [int(k) for k in cand.files]
    has = np.array([k in row for k in mol.key.values])
    tr = np.setdiff1d(np.where(has)[0], ev)
    Xtr = torch.tensor(np.asarray(X[[row[k] for k in mol.key.values[tr]]], np.float32)); Ytr = torch.tensor(FP[tr], dtype=torch.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd
    net = torch.nn.Sequential(torch.nn.Linear(1024, a.hidden), torch.nn.GELU(), torch.nn.Dropout(0.2),
                              torch.nn.Linear(a.hidden, a.hidden), torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(a.hidden, 2048))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * (len(tr) // 1024 + 1))
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        net.train(); perm = torch.randperm(len(tr)); tot = 0
        for b in range(0, len(tr), 1024):
            i = perm[b:b + 1024]; opt.zero_grad(); l = lossf(net(Xtr[i]), Ytr[i]); l.backward(); opt.step(); sched.step(); tot += l.item() * len(i)
        print('epoch', ep, round(tot / len(tr), 4), round(time.time() - t0), 's', flush=True)
    net.eval(); rows = []
    with torch.no_grad():
        for i in ev:
            k = mol.key.values[i]
            if k not in row: rows.append((k, np.nan)); continue
            q = torch.sigmoid(net((torch.tensor(np.asarray(X[row[k]], np.float32)) - mu) / sd)).numpy()
            ids = cand[str(i)]; F = FP[ids].astype(np.float32)
            s = F @ q / (np.linalg.norm(F, axis=1) * np.linalg.norm(q) + 1e-9)
            rank = 1 + int((s[1:] >= s[0]).sum())          # gold is at index 0; pessimistic ties
            rows.append((k, rank))
    df = pd.DataFrame(rows, columns=['key', 'rank']).merge(s2[['key', 'rank']].rename(columns={'rank': 's2_rank'}), on='key')
    rr = lambda r: np.where((r <= 25) & np.isfinite(r), 1 / r, 0.0)
    df['dreams_rr'] = rr(df['rank'].values.astype(float)); df['s2_rr'] = rr(df.s2_rank.values.astype(float))
    d = (df.dreams_rr - df.s2_rr).values; rng = np.random.default_rng(0)
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
    out = dict(n=len(df), missing_emb=int(df['rank'].isna().sum()), dreams_mrr25=float(df.dreams_rr.mean()), s2_mrr25=float(df.s2_rr.mean()),
               dreams_top1=float((df['rank'] == 1).mean()), s2_top1=float((df.s2_rank == 1).mean()), diff=float(d.mean()),
               p_dreams_better=float((boot > 0).mean()), ci90=[float(np.quantile(boot, .05)), float(np.quantile(boot, .95))],
               corr_rr=float(np.corrcoef(df.dreams_rr, df.s2_rr)[0, 1]), gate_public=0.660, epochs=a.epochs, hidden=a.hidden, seconds=round(time.time() - t0))
    os.makedirs(ART, exist_ok=True); json.dump(out, open(f'{ART}/62_dreams_head.json', 'w'), indent=1); df.to_csv(f'{ART}/62_per_molecule.csv', index=False)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
