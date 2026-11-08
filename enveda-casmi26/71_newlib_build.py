"""71: library of newer public MS/MS spectra whose structures are absent from train.parquet (for the 72 rank-1 overlay).

What: MassBank open records 2026.03 (MS2) and GNPS community libraries 2026-10-22 (public S3 mirrors, the same files 61
read), keyed by InChIKey-14; keeps spectra with a parseable SMILES, >= 3 peaks, precursor m/z > 50, ion mode pos/neg,
and a structure NOT in train.parquet; SMILES normalised like the competition answers (largest fragment, neutralised,
stereo removed). Keeps the per-record licence (MassBank) / library name (GNPS).
Why: 61's "0 test hits" was measured on the visible placeholder test (copies of train spectra), so the overlay is
untested on the hidden test; host allows newer releases of train's own sources (750684). 61's train-side false-hit
rate (cos >= 0.85, >= 5 shared peaks): 0.23% of molecules.
Output: artifacts/71_newlib.parquet (src, licence, k, smiles, mode, precursor_mz, mz, intensity) + 71_newlib.json.
Usage: python 71_newlib_build.py [--lib /home/user/work/extlib]
"""
import argparse, json, os
import numpy as np, pandas as pd, pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
LF, UN = rdMolStandardize.LargestFragmentChooser(), rdMolStandardize.Uncharger()


def norm(smi):
    m = Chem.MolFromSmiles(str(smi or ''))
    if m is None: return None, None
    try: m = UN.uncharge(LF.choose(m))
    except Exception: pass
    Chem.RemoveStereochemistry(m)
    try: return Chem.MolToSmiles(m), Chem.MolToInchiKey(m)[:14]
    except Exception: return None, None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lib', default='/home/user/work/extlib'); a = ap.parse_args()
    train_k = set(pd.read_parquet('/home/user/enveda_data/train_meta.parquet', columns=['inchikey14']).inchikey14)
    g = pq.read_table(f'{a.lib}/gnps.parquet', columns=['library', 'ion_mode', 'smiles', 'inchikey', 'precursor_mz', 'mz', 'intensity', 'ms_level']).to_pandas()
    g = g[g.ms_level == 2].assign(src='gnps', licence=lambda d: 'GNPS:' + d.library.astype(str))
    m = pq.read_table(f'{a.lib}/massbank.parquet', columns=['license', 'ion_mode', 'smiles', 'inchikey', 'precursor_mz', 'mz', 'intensity', 'spectrum_type']).to_pandas()
    m = m[m.spectrum_type == 'MS2'].assign(src='massbank', licence=lambda d: d.license.astype(str))
    L = pd.concat([d[['src', 'licence', 'ion_mode', 'smiles', 'inchikey', 'precursor_mz', 'mz', 'intensity']] for d in (g, m)], ignore_index=True)
    L['k0'] = L.inchikey.fillna('').str.upper().str[:14]
    L = L[(L.k0.str.len() == 14) & ~L.k0.isin(train_k)]
    L['mode'] = L.ion_mode.fillna('').str.lower().str[:3]
    L = L[L['mode'].isin(['pos', 'neg']) & (pd.to_numeric(L.precursor_mz, errors='coerce') > 50) & L.mz.map(lambda x: x is not None and len(x) >= 3)]
    ns = {s: norm(s) for s in L.smiles.dropna().unique()}
    L['smiles_n'] = L.smiles.map(lambda s: ns.get(s, (None, None))[0]); L['k'] = L.smiles.map(lambda s: ns.get(s, (None, None))[1])
    L = L[L.k.notna() & ~L.k.isin(train_k)]
    out = L[['src', 'licence', 'k', 'smiles_n', 'mode', 'precursor_mz', 'mz', 'intensity']].rename(columns={'smiles_n': 'smiles'})
    out['precursor_mz'] = out.precursor_mz.astype(float)
    out['mz'] = out.mz.map(lambda x: np.asarray(x, float)); out['intensity'] = out.intensity.map(lambda x: np.asarray(x, float))
    out = out.sort_values(['mode', 'precursor_mz']).reset_index(drop=True)
    os.makedirs(ART, exist_ok=True); out.to_parquet(f'{ART}/71_newlib.parquet', index=False)
    S = dict(spectra=len(out), structures=int(out.k.nunique()), by_src=out.src.value_counts().to_dict(),
             licences=out.licence.value_counts().head(12).to_dict(), modes=out['mode'].value_counts().to_dict())
    json.dump(S, open(f'{ART}/71_newlib.json', 'w'), indent=1); print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
