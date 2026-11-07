"""Leak-free CASMI bench, step 1: select molecules no public model has seen.

What: MS2 spectra from newer public library releases (MassBank open 2026.03, GNPS community 2026-10-22; public S3
mirrors) whose structure is NOT in train.parquet (InChIKey first block, plus a tautomer-canonical check against train
structures of the same formula), that are natural products (in COCONUT, as the test is), with a test-like adduct
([M+H]+, [M-H]-, [M+Na]+, [M+NH4]+) and >= 3 peaks. Writes them in the competition test format.
Why: the bench477 molecules are inside train.parquet, so every public component trained on it has seen them; the
leading explanation of the bench reversal (tail reranker bench +0.039, LB -0.021) is that leakage. Structures absent
from train cannot be memorised by any model trained on train.parquet.
Output: artifacts/67_leakfree_test.parquet (test columns, molecule_id l_<ik14>), artifacts/67_leakfree_answers.csv
(inchikey14, normalized_smiles, molecular_formula, n_spectra, sources), artifacts/67_leakfree_summary.json.
Usage: python 67_leakfree_bench_select.py [--lib /home/user/work/extlib] [--max 300]
"""
import argparse, json, os, re
import numpy as np, pandas as pd, pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog('rdApp.*')
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
D = '/home/user/enveda_data'
ADDUCTS = {'[M+H]+': 'positive', '[M-H]-': 'negative', '[M+Na]+': 'positive', '[M+NH4]+': 'positive'}
TE = rdMolStandardize.TautomerEnumerator()


def taut_key(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    Chem.RemoveStereochemistry(m)
    try: m = TE.Canonicalize(m)
    except Exception: pass
    return Chem.MolToInchiKey(m)[:14]


def ce_value(s):
    v = re.findall(r'(\d+(?:\.\d+)?)', str(s or ''))
    return [float(v[0])] if v and ('ev' in str(s).lower() or str(s).replace('.', '').isdigit()) else []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--lib', default='/home/user/work/extlib'); ap.add_argument('--max', type=int, default=300)
    a = ap.parse_args()
    meta = pd.read_parquet(f'{D}/train_meta.parquet', columns=['inchikey14', 'molecular_formula'])
    train_k = set(meta.inchikey14)
    coco = set(pd.read_parquet('/home/user/work/coconut.parquet', columns=['ik14']).ik14)
    g = pq.read_table(f'{a.lib}/gnps.parquet', columns=['spectrum_id', 'adduct', 'precursor_mz', 'ion_mode', 'instrument', 'ms_level', 'smiles', 'inchikey', 'mz', 'intensity']).to_pandas()
    g = g[g.ms_level == 2].rename(columns={'spectrum_id': 'sid'}); g['src'] = 'gnps'; g['ce'] = None
    m = pq.read_table(f'{a.lib}/massbank.parquet', columns=['accession', 'precursor_type', 'precursor_mz', 'ion_mode', 'instrument_type', 'spectrum_type', 'smiles', 'inchikey', 'collision_energy', 'mz', 'intensity']).to_pandas()
    m = m[m.spectrum_type == 'MS2'].rename(columns={'accession': 'sid', 'precursor_type': 'adduct', 'instrument_type': 'instrument', 'collision_energy': 'ce'}); m['src'] = 'massbank'
    L = pd.concat([g[['src', 'sid', 'adduct', 'precursor_mz', 'instrument', 'smiles', 'inchikey', 'ce', 'mz', 'intensity']],
                   m[['src', 'sid', 'adduct', 'precursor_mz', 'instrument', 'smiles', 'inchikey', 'ce', 'mz', 'intensity']]], ignore_index=True)
    L = L.dropna(subset=['inchikey', 'smiles', 'precursor_mz']); L['ik14'] = L.inchikey.str[:14]
    n0 = L.ik14.nunique()
    L = L[~L.ik14.isin(train_k) & L.ik14.isin(coco) & L.adduct.isin(ADDUCTS)]
    L = L[L.mz.map(lambda x: x is not None and len(x) >= 3)]
    print('structures in libs', n0, '| not in train, in COCONUT, test adduct, >=3 peaks:', L.ik14.nunique(), 'spectra', len(L), flush=True)
    # tautomer-canonical check against train structures of the same formula
    fsmi = L.drop_duplicates('ik14').set_index('ik14').smiles
    form = {k: CalcMolFormula(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None for k, s in fsmi.items()}
    forms = {f for f in form.values() if f}
    tr = pd.read_parquet(f'{D}/train_meta.parquet', columns=['inchikey14', 'molecular_formula'])
    tr = tr[tr.molecular_formula.isin(forms)].drop_duplicates('inchikey14')
    import pyarrow.dataset as pds
    smi_tr = pds.dataset(f'{D}/train.parquet').to_table(columns=['inchikey14', 'normalized_smiles'], filter=pds.field('inchikey14').isin(list(tr.inchikey14))).to_pandas().drop_duplicates('inchikey14')
    train_taut = {taut_key(s) for s in smi_tr.normalized_smiles}
    keep = [k for k, s in fsmi.items() if form.get(k) and taut_key(s) not in train_taut]
    print('after tautomer check:', len(keep), 'of', len(fsmi), flush=True)
    L = L[L.ik14.isin(keep)]
    # one molecule per structure; cap spectra per molecule at 16 (the test's maximum); sample up to --max molecules
    ks = sorted(set(L.ik14), key=lambda k: hash(('leakfree', k)) & 0xffffffff)[:a.max]
    L = L[L.ik14.isin(ks)].groupby('ik14', group_keys=False).head(16)
    rows = []
    for r in L.itertuples():
        mz = np.asarray(r.mz, float); it = np.asarray(r.intensity, float); o = np.argsort(mz); mz, it = mz[o], it[o]
        it = it / it.max() if it.max() > 0 else it
        ce = ce_value(r.ce)
        rows.append(dict(molecule_id='l_' + r.ik14, spectrum_id=f'{r.src}_{r.sid}', ms2_mzs=mz.tolist(), ms2_normalized_intensities=it.tolist(),
                         base_peak_intensity=1.0, adduct=r.adduct, ionization_mode=ADDUCTS[r.adduct], instrument_type=str(r.instrument or 'unknown'),
                         precursor_mz=float(r.precursor_mz), collision_energy_orig=str(r.ce or ''), collision_energy_ev=ce,
                         collision_energy_orig_units='eV' if ce else ''))
    test = pd.DataFrame(rows)
    ans = (L.drop_duplicates('ik14').assign(molecule_id=lambda d: 'l_' + d.ik14)[['molecule_id', 'ik14', 'smiles']]
           .rename(columns={'ik14': 'inchikey14', 'smiles': 'normalized_smiles'}))
    ans['molecular_formula'] = ans.normalized_smiles.map(lambda s: CalcMolFormula(Chem.MolFromSmiles(s)))
    ans['metric_key'] = ans.normalized_smiles.map(taut_key)
    os.makedirs(ART, exist_ok=True)
    test.to_parquet(f'{ART}/67_leakfree_test.parquet', index=False); ans.to_csv(f'{ART}/67_leakfree_answers.csv', index=False)
    s = dict(molecules=int(test.molecule_id.nunique()), spectra=len(test), adducts=test.adduct.value_counts().to_dict(),
             instruments=test.instrument_type.value_counts().head(8).to_dict(), sources=L.src.value_counts().to_dict(),
             mass_q=[float(x) for x in np.percentile(test.groupby('molecule_id').precursor_mz.first(), [5, 50, 95])])
    json.dump(s, open(f'{ART}/67_leakfree_summary.json', 'w'), indent=1); print(json.dumps(s))


if __name__ == '__main__':
    main()
