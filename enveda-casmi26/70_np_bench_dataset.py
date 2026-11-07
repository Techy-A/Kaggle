"""Leak-free natural-product bench, step 2: write the 67 selection in the unseen-bench dataset format.

What: converts artifacts/67_leakfree_test.parquet + 67_leakfree_answers.csv (128 COCONUT natural products from
MassBank 2026.03 / GNPS 2026-10 whose structures are absent from train.parquet) into unseen_test.parquet and
unseen_answers.csv exactly as bench/make_unseen.py writes them: m/z rounded to 4 decimals, positive-mode m/z and
precursor minus one electron mass (0.000548 Da, as the host applied to the hidden test), intensities max-normalised
with peaks below 2e-4 dropped, MassBank normalised collision energies ("40 % (nominal)") converted to eV as NCE * m/z / 500, instrument labelled timsTOF (the test's label), molecule ids n_<ik14>, spectrum ids
ns_00000..., classes 3/2 = 60/40% by md5 order (the unseen split), molecular_formula empty as in make_unseen.
Why: the unseen bench (Enveda-180) is largely synthetic and had no power for NP-specific ideas (69); this set is NP.
Output: <out>/unseen_test.parquet, <out>/unseen_answers.csv, <out>/dataset-metadata.json (private Kaggle dataset).
Usage: python 70_np_bench_dataset.py <out_dir>
"""
import hashlib, json, os, sys
import numpy as np, pandas as pd
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
t = pd.read_parquet(f'{ART}/67_leakfree_test.parquet'); a = pd.read_csv(f'{ART}/67_leakfree_answers.csv', dtype=str, keep_default_na=False)
rows = []
for r in t.itertuples():
    pos = r.ionization_mode == 'positive'; shift = 0.000548 if pos else 0.0
    mz = np.asarray(r.ms2_mzs, float); it = np.asarray(r.ms2_normalized_intensities, float)
    it = it / it.max(); k = it >= 2e-4
    ce = np.asarray(r.collision_energy_ev, float)
    if not len(ce) and '%' in str(r.collision_energy_orig):   # Thermo HCD normalised CE -> eV for z = 1: NCE * m/z / 500
        nce = float(str(r.collision_energy_orig).split('%')[0].strip()); ce = np.array([round(nce * float(r.precursor_mz) / 500, 1)])
    rows.append(dict(molecule_id='n_' + r.molecule_id[2:], ms2_mzs=np.round(mz[k] - shift, 4), ms2_normalized_intensities=it[k],
                     base_peak_intensity=float(r.base_peak_intensity), adduct=r.adduct, ionization_mode=r.ionization_mode,
                     instrument_type='timsTOF', precursor_mz=round(float(r.precursor_mz) - shift, 4),
                     collision_energy_orig=','.join(str(int(c)) if c == int(c) else str(c) for c in ce),
                     collision_energy_ev=ce, collision_energy_orig_units='eV' if len(ce) else ''))
df = pd.DataFrame(rows)
df = df[df.ms2_mzs.map(len) >= 3].reset_index(drop=True)
df['spectrum_id'] = [f'ns_{i:05d}' for i in range(len(df))]
cols = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities', 'base_peak_intensity', 'adduct',
        'ionization_mode', 'instrument_type', 'precursor_mz', 'collision_energy_orig', 'collision_energy_ev',
        'collision_energy_orig_units']
for c in ['molecule_id', 'spectrum_id', 'adduct', 'ionization_mode', 'instrument_type', 'collision_energy_orig', 'collision_energy_orig_units']:
    df[c] = df[c].astype('string')
mols = sorted(df.molecule_id.unique(), key=lambda m: hashlib.md5(('n' + m).encode()).hexdigest())
n3 = round(0.6 * len(mols)); cls = {m: (3 if i < n3 else 2) for i, m in enumerate(mols)}
df[cols].to_parquet(f'{OUT}/unseen_test.parquet', index=False)
a['molecule_id'] = 'n_' + a.molecule_id.str[2:]
a = a[a.molecule_id.isin(mols)]
ans = a[['inchikey14', 'normalized_smiles', 'molecule_id', 'metric_key']].copy(); ans['cls'] = ans.molecule_id.map(cls); ans['molecular_formula'] = ''
ans.to_csv(f'{OUT}/unseen_answers.csv', index=False)
json.dump({'title': 'casmi26-bench-np-leakfree', 'id': 'aryanawasthiteykik/casmi26-bench-np-leakfree',
           'licenses': [{'name': 'CC-BY-4.0'}]}, open(f'{OUT}/dataset-metadata.json', 'w'), indent=1)
print(len(df), 'spectra |', len(mols), 'molecules |', ans.cls.value_counts().to_dict(), '| adducts', df.adduct.value_counts().to_dict())
