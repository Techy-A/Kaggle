"""Leak-free pseudo-test for the CASMI26 bench from Enveda-180 Zenodo spectra (CC BY 4.0, record 21346580) of
structures absent from train.parquet (and from the v4 pool by metric identity). Test schema, max-normalised
intensities, m/z rounded to 4 decimals, positive-mode m/z minus one electron mass (0.000548 Da) as the host applied to
the hidden test. Class 2 = left reachable (they are PubChem structures); class 3 = hidden from every candidate source."""
import hashlib, json, sys
import numpy as np, pandas as pd
SRC = '/home/user/benchdata/e180extra/e180_not_in_train.jsonl'
CLEAN = '/home/user/benchdata/e180extra/leakfree_structures.csv'
OUT = sys.argv[1]; N_MOL = int(sys.argv[2]) if len(sys.argv) > 2 else 320
STD = {'[M+H]+', '[M+NH4]+', '[M-H2O+H]+', '[M-2H2O+H]+', '[M+Na]+', '[M+K]+', '[M-H]-', '[M-H2O-H]-', '[M+CH2O2-H]-', '[M+Cl]-'}
clean = pd.read_csv(CLEAN)
keep = set(clean.ik14)
rows = []
for line in open(SRC):
    r = json.loads(line); ik = r['inchikey'][:14]
    if ik not in keep or r['adduct'] not in STD:
        continue
    try:
        ppm = abs(float(r['precursor_ppm'])); ex = float(r['explained_intensity'])
    except (TypeError, ValueError):
        continue
    if ppm > 20 or ex < 0.3 or int(r['num_peaks']) < 3:
        continue
    pos = 'Positive' in r['ionmode']
    mz = np.array([float(x) for x in r['mzs']]); it = np.array([float(x) for x in r['intensities']])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    base = float(it.max()); it = it / base; k = it >= 2e-4
    shift = 0.000548 if pos else 0.0
    ces = [float(x) for x in str(r['collision_energies']).split(',') if x.strip()]
    rows.append(dict(ik14=ik, smiles=r['smiles'], ms2_mzs=np.round(mz[k] - shift, 4), ms2_normalized_intensities=it[k],
                     base_peak_intensity=base, adduct=r['adduct'], ionization_mode='positive' if pos else 'negative',
                     instrument_type='timsTOF', precursor_mz=round(float(r['pepmass']) - shift, 4),
                     collision_energy_orig=','.join(str(int(c)) if c == int(c) else str(c) for c in ces),
                     collision_energy_ev=np.array(ces, dtype=float), collision_energy_orig_units='eV'))
df = pd.DataFrame(rows)
mols = sorted(df.ik14.unique(), key=lambda k: hashlib.md5(('u' + k).encode()).hexdigest())[:N_MOL]
df = df[df.ik14.isin(mols)].reset_index(drop=True)
n3 = round(0.6 * len(mols))
cls = {k: (3 if i < n3 else 2) for i, k in enumerate(mols)}
df['molecule_id'] = 'u_' + df.ik14
df['spectrum_id'] = [f'us_{i:05d}' for i in range(len(df))]
cols = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
        'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev',
        'collision_energy_orig_units']
for c in ['molecule_id', 'spectrum_id', 'adduct', 'ionization_mode', 'instrument_type', 'collision_energy_orig', 'collision_energy_orig_units']:
    df[c] = df[c].astype('string')
df[cols].to_parquet(f'{OUT}/unseen_test.parquet', index=False)
ans = df.drop_duplicates('ik14')[['ik14', 'smiles', 'molecule_id']].rename(columns={'ik14': 'inchikey14', 'smiles': 'normalized_smiles'})
ans = ans.merge(clean[['ik14', 'mkey']].rename(columns={'ik14': 'inchikey14', 'mkey': 'metric_key'}), on='inchikey14')
ans['cls'] = ans.inchikey14.map(cls); ans['molecular_formula'] = ''
ans.to_csv(f'{OUT}/unseen_answers.csv', index=False)
print(len(df), 'spectra |', len(mols), 'molecules |', ans.cls.value_counts().to_dict(), '| adducts', df.adduct.value_counts().to_dict())
