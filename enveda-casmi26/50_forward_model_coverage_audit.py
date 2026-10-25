"""Forward-model evidence coverage audit of the real test set (CPU only, offline, seconds).

What: for each of the 400 test molecules, which forward-model evidence the current 0.465/0.466 pipeline can give it,
decided from the spectra it has (adduct / ionization mode) and the rules coded in the pipeline:
  * ICEBERG  : positive [M+H]+ or [M+Na]+ spectra only (fuse.build_ice_input covered_only=True,
               benchdata/dl_fuse_1024/fuse.py).
  * GLACIER  : real [M+H]+ spectra only (v29); a molecule WITHOUT [M+H]+ gets N1A proxies from [M+NH4]+ /
               [M-H2O+H]+ spectra (relabelled [M+H]+); [M+Na]+, [M+K]+, [M-2H2O+H]+ and every negative adduct are never proxied.
  * N1 challenger needs ICE and GL together on one molecule.
Classes: the real test classes are hidden. The local train.parquet is NOT usable as a class proxy: every one of the 1,213
test spectra occurs verbatim (binned cosine 1.000) in it, yet the matched structure is in the pipeline's top 25 for only
12 of 400 molecules, so those rows carry no usable structure signal. Class rows are therefore the host-fitted prior shares
(28 / 20 / 52 %, Bull 748068) applied to the molecule-level result, valid only if adduct is independent of class.
Why: CASMI26.md Methods digest 25 Oct, next experiment 3: is any adduct group blind, so that extending the N1 family
could still buy a +0.008 step (one public molecule = 0.0076)?
What it cannot do: candidate-level coverage (which final-list candidates lack an ICE/GL score) and the real run's
n_mols_scored need the 0.466 run logs, which are not in any local cache; only molecule-level coverage is computed.
Output: artifacts/coverage_audit_molecules.csv and coverage_audit_summary.json (cached; delete to recompute).
usage: python 50_forward_model_coverage_audit.py
"""
import json
import os
import pandas as pd

D = '/home/user/enveda_data'
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'artifacts')
OUT_CSV, OUT_JSON = f'{ART}/coverage_audit_molecules.csv', f'{ART}/coverage_audit_summary.json'
PRIOR = {1: 0.28, 2: 0.20, 3: 0.52}
PUBLIC_MOLS, ONE_MOL = 132, 0.0076


def group(ads):
    ice = bool(ads & {'[M+H]+', '[M+Na]+'})
    gl = '[M+H]+' in ads or bool(ads & {'[M+NH4]+', '[M-H2O+H]+'})
    return 'ICE+GL' if ice and gl else 'GL only (N1A proxy)' if gl else 'ICE only (Na fallback)' if ice else 'none'


def main():
    os.makedirs(ART, exist_ok=True)
    te = pd.read_parquet(f'{D}/test.parquet', columns=['molecule_id', 'adduct'])
    m = te.groupby('molecule_id').adduct.agg(lambda x: set(x)).rename('ads').to_frame()
    m['group'] = m.ads.map(group)
    m['adducts'] = m.ads.map(lambda a: '|'.join(sorted(a)))
    m.drop(columns='ads').to_csv(OUT_CSV)
    n = len(m)
    S = dict(n=n, by_group=m.group.value_counts().to_dict(),
             by_adduct_set={a: g.group.value_counts().to_dict() for a, g in m.groupby('adducts')})
    unc = int((m.group != 'ICE+GL').sum())
    blind = int((m.group == 'none').sum())
    S['not_full_N1'] = unc
    S['share_none'] = blind / n
    S['expected_molecules_by_class_none'] = {c: round(blind * p, 2) for c, p in PRIOR.items()}
    # upper bound: every blind molecule goes from rank r to rank 1 with prob p_fix; one molecule = 0.0076 on the public set
    S['public_blind_expected'] = round(blind / n * PUBLIC_MOLS, 1)
    S['lb_gain_if_all_blind_fixed_perfectly'] = round(blind / n * PUBLIC_MOLS * ONE_MOL, 4)
    json.dump(S, open(OUT_JSON, 'w'), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == '__main__':
    main()
