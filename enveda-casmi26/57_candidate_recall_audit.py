"""57: candidate database audit. Which structures can reach the 25-lists, and would a newer or extra database add any?

What: (1) builds the key sets of every candidate source the 0.477/0.479 line uses (train.parquet structures, the v4
pool = train + prvsiyan COCONUT, engine 2's COCONUT + ChEBI/LMSD + PubChemLite CC0, the 105.9M-row PubChem tier of
the PubChem channel) and of candidate additions (COCONUT October 2026 full release, its ChEBI+Wikidata licensed
subset, LOTUS 2026-04-13, NPAtlas 2024_09); (2) recall test on structures absent from train.parquet: P1 = Enveda-180
Zenodo structures not in train (real Enveda timsTOF molecules, synthetic), P2 = NPAtlas compounds first reported
2021-2024 not in train (the newest natural products), P3 = 5,000 random LOTUS/NPAtlas structures not in train; a
structure counts as present when its raw InChIKey-14, its largest-fragment/uncharged parent key or its metric key
(scorer rule: stereo removed, RDKit tautomer canonical) matches; (3) per addition: structures new to engine 2's pool
and new to every current source (neutral parents, 150-1250 Da), recall gain on the probes, and the extra decoys per
+-10 ppm window (dilution); (4) where the final 25 of the 0.477 pipeline come from (bench477 run, 250 lists).
Why: the bench hides class-3 answers from every source, so it cannot measure database additions; the host says class 3
is absent from PubChem and COCONUT; wider PubChem and gen20 lost on the LB. This measures the database lever directly.
Output: artifacts/57_summary.json, artifacts/57_probe_recall.csv, artifacts/57_additions.csv; large caches (tier key
hashes, standardised keys) in /home/user/benchdata/dbaudit/cache. CPU, about 10 min cold, seconds warm.
Usage: python enveda-casmi26/57_candidate_recall_audit.py
Data (read as data only): /home/user/ext (PubChem tier, prvsiyan COCONUT), /home/user/benchdata/pool,
/home/user/benchdata/e180extra, /home/user/benchdata/dbaudit/* (Kaggle datasets xianyu3/casmi26-licensed-coconut-
candidates-202610, xjoannax88/casmi-pubchemlite-candidates, deadsignalslab/casmi26-chebi-lmsd-regenerated,
dmitriigluzdov/casmi26-natural-product-knowledge-table; coconut_csv_lite-10-2026.zip from coconut.s3.uni-jena.de).
"""
import json
import pickle
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path(__file__).resolve().parent / "artifacts"
DB = Path("/home/user/benchdata/dbaudit")
CACHE = DB / "cache"
B477 = Path("/home/user/work/out_bench477")
LO, HI, PPM = 150.0, 1250.0, 10.0           # PubChem tier mass range; pipeline window +-10 ppm
SHARE = {2: 0.20, 3: 0.52}                    # community class mix of the test (28/20/52)
RR_NEW = 0.26                                  # honest MRR of PubChem-reachable unseen answers (unseen1 class 2)
CACHE.mkdir(parents=True, exist_ok=True)


def u64(keys):
    """InChIKey-14 (A-Z) -> exact base-26 integer mod 2^64 (collision odds ~1e-5 over the whole tier)."""
    a = np.frombuffer(np.asarray(keys, dtype="S14").tobytes(), dtype=np.uint8).reshape(-1, 14).astype(np.uint64)
    h = np.zeros(len(a), np.uint64)
    for j in range(14):
        h = h * np.uint64(26) + (a[:, j] - np.uint64(65))
    return h


def tier_hashes():
    p = CACHE / "tier_u64_sorted.npy"
    if not p.exists():
        k = np.load("/home/user/ext/pop/pc_ik14.npy", mmap_mode="r")
        h = np.concatenate([u64(k[i:i + 5_000_000]) for i in range(0, len(k), 5_000_000)])
        h.sort()
        np.save(p, h)
    return np.load(p, mmap_mode="r")


TH = tier_hashes()


def in_tier(keys):
    q = u64(pd.Series(keys).fillna("AAAAAAAAAAAAAA").values)
    i = np.searchsorted(TH, q).clip(0, len(TH) - 1)
    return TH[i] == q


_G = {}


def std(smi):
    """raw ik14, largest-fragment + uncharged parent ik14, parent metric key, parent mass, parent charge."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Descriptors import ExactMolWt
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if not _G:
        _G.update(T=rdMolStandardize.TautomerEnumerator(), U=rdMolStandardize.Uncharger(),
                  L=rdMolStandardize.LargestFragmentChooser(preferOrganic=True))
    try:
        m = Chem.MolFromSmiles(smi)
        raw0 = Chem.MolToInchiKey(m)[:14]
        m = _G["U"].uncharge(_G["L"].choose(m))
        raw, mass, chg = Chem.MolToInchiKey(m)[:14], ExactMolWt(m), Chem.GetFormalCharge(m)
        if not LO - 50 <= mass <= HI + 50:     # outside the tier range: skip the (slow) tautomer step
            return raw0, raw, raw, mass, chg
        Chem.RemoveStereochemistry(m)
        return raw0, raw, Chem.MolToInchiKey(_G["T"].Canonicalize(m))[:14], mass, chg
    except Exception:
        return None, None, None, np.nan, 0


def std_frame(smiles, tag):
    p = CACHE / f"std_{tag}.parquet"
    if not p.exists():
        with Pool(4) as pool:
            r = pool.map(std, list(smiles), chunksize=50)
        pd.DataFrame(r, columns=["raw0", "raw", "mkey", "pmass", "pcharge"]).to_parquet(p)
    return pd.read_parquet(p)


def hit(df, keyset):
    return df.raw0.isin(keyset) | df.raw.isin(keyset) | df.mkey.isin(keyset)


def hit_tier(df):
    return in_tier(df.raw0) | in_tier(df.raw) | in_tier(df.mkey)


# ---------------------------------------------------------------- current sources (0.477 / 0.479 line)
cm = pickle.load(open("/home/user/ext/coconut/coco_meta.pkl", "rb"))
pm = pd.read_parquet("/home/user/benchdata/pool/pool_meta.parquet", columns=["key", "src", "mass"])
ts = pd.read_parquet("/home/user/benchdata/pool/train_structs.parquet", columns=["key", "rawkey", "mass"])
bio = pd.read_parquet(DB / "chebi/bio_structures.parquet")
pcl = pd.read_parquet(DB / "pclite/pubchemlite_structures.parquet", columns=["key", "exact_mass", "charge"])
SRC = {
    "train": set(ts.key) | set(ts.rawkey),
    "v4pool_coconut": set(pm.key[pm.src >= 2]),
    "e2_coconut_prvsiyan": set(cm["keys"]),
    "e2_chebi_lmsd": set(bio.inchikey14),
    "e2_pubchemlite": set(pcl.key),
}
E2 = set().union(*SRC.values()) | set(pm.key)          # engine-2 pool (and engine-1 v4 pool, a subset)
e2_mass = pd.concat([pd.DataFrame({"k": cm["keys"], "m": np.load("/home/user/ext/coconut/coco_mass.npy")}),
                     pd.DataFrame({"k": bio.inchikey14, "m": bio.monoisotopic_mass}),
                     pd.DataFrame({"k": pcl.key, "m": pcl.exact_mass}),
                     pd.DataFrame({"k": ts.rawkey, "m": ts.mass})]).drop_duplicates("k").m.dropna().sort_values().values
pc_mass = np.load("/home/user/ext/pubchem/pc_mass.npy", mmap_mode="r")


def window_count(sorted_mass, targets):
    t = np.asarray(targets, float)
    return np.searchsorted(sorted_mass, t * (1 + PPM / 1e6), "right") - np.searchsorted(sorted_mass, t * (1 - PPM / 1e6), "left")


# ---------------------------------------------------------------- additions
z = zipfile.ZipFile(DB / "coconut2610/coconut_csv_lite-10-2026.zip")
co = pd.read_csv(z.open("coconut_csv_lite-10-2026.csv"), usecols=["identifier", "canonical_smiles", "standard_inchi_key"])
co["ik14"] = co.standard_inchi_key.str[:14]
co = co.drop_duplicates("ik14").reset_index(drop=True)
xy = pd.read_json(DB / "coco202610/candidates.jsonl", lines=True)
npt = pd.read_parquet(DB / "nptable/natural_products.parquet")
ADD = {
    "coconut_2026_10_full": co.rename(columns={"canonical_smiles": "smiles"})[["ik14", "smiles"]],
    "coconut_2026_10_licensed_chebi_wikidata": xy.rename(columns={"key14": "ik14"})[["ik14", "smiles"]],
    "lotus_2026_04": npt[npt.in_lotus][["ik14", "smiles"]],
    "npatlas_2024_09": npt[npt.in_npatlas][["ik14", "smiles"]],
}
add_sets, add_new_e2_mass, add_rows = {}, {}, []
for name, a in ADD.items():
    a = a.drop_duplicates("ik14").reset_index(drop=True)
    pre = a.ik14.isin(E2)                                         # raw-key hit: no RDKit needed
    s = std_frame(a.smiles[~pre], f"add_{name}")
    s.index = a.index[~pre]
    full = pd.DataFrame({"raw0": a.ik14, "raw": a.ik14, "mkey": a.ik14, "pmass": np.nan, "pcharge": 0})
    full.loc[s.index, ["raw0", "raw", "mkey", "pmass", "pcharge"]] = s.values
    add_sets[name] = set(full.raw0.dropna()) | set(full.raw.dropna()) | set(full.mkey.dropna())
    r = full.loc[~pre]
    new_e2 = r[~hit(r, E2) & r.pmass.between(LO, HI)]
    new_all = new_e2[~hit_tier(new_e2)]
    neu_e2 = new_e2[new_e2.pcharge == 0]
    neu_all = new_all[new_all.pcharge == 0]
    add_new_e2_mass[name] = np.sort(neu_e2.pmass.values.astype(float))
    add_rows.append(dict(addition=name, structures=len(a), raw_hit_engine2_pool=int(pre.sum()),
                         new_to_engine2_pool_150_1250=len(new_e2), of_which_neutral=len(neu_e2),
                         neutral_already_in_pubchem_tier=len(neu_e2) - len(neu_all),
                         new_to_every_source=len(new_all), new_to_every_source_neutral=len(neu_all),
                         neutral_new_examples=";".join(neu_all.mkey.astype(str).head(5))))
    print(add_rows[-1], flush=True)
adds = pd.DataFrame(add_rows)

# ---------------------------------------------------------------- probes: structures absent from train.parquet
rows = [json.loads(line) for line in open("/home/user/benchdata/e180extra/e180_not_in_train.jsonl")]
p1 = pd.DataFrame({"smiles": [r["smiles"] for r in rows], "ik": [r["inchikey"][:14] for r in rows]}).drop_duplicates("ik")
p2 = npt[npt.in_npatlas & (npt.npatlas_first_year >= 2021) & ~npt.in_casmi_train]
p3 = npt[~npt.in_casmi_train].sample(5000, random_state=0)
PROBES = {"P1_enveda180_not_in_train": p1.smiles, "P2_npatlas_2021_2024_not_in_train": p2.smiles,
          "P3_lotus_npatlas_not_in_train_5k": p3.smiles}
rec_rows, dil_rows = [], []
for pname, smi in PROBES.items():
    P = std_frame(smi.reset_index(drop=True), pname)
    P = P[~hit(P, SRC["train"]) & P.pmass.between(LO, HI)].reset_index(drop=True)
    cur_e2, cur_t = hit(P, E2), hit_tier(P)
    row = dict(probe=pname, n=len(P), **{k: round(float(hit(P, v).mean()), 4) for k, v in SRC.items()},
               pubchem_tier=round(float(cur_t.mean()), 4), engine2_pool=round(float(cur_e2.mean()), 4),
               current_union=round(float((cur_e2 | cur_t).mean()), 4), missing_from_all=int((~(cur_e2 | cur_t)).sum()))
    for name, s in add_sets.items():
        h = hit(P, s)
        row[f"{name}"] = round(float(h.mean()), 4)
        row[f"gain_union_{name}"] = int((h & ~(cur_e2 | cur_t)).sum())       # newly reachable at all
        row[f"gain_e2_{name}"] = int((h & ~cur_e2).sum())                   # moves into engine 2's pool
    rec_rows.append(row)
    # dilution: candidates per +-10 ppm window at the probe masses
    d = dict(probe=pname, tier_window_median=float(np.median(window_count(pc_mass, P.pmass))),
             e2_window_median=float(np.median(window_count(e2_mass, P.pmass))),
             e2_window_mean=float(np.mean(window_count(e2_mass, P.pmass))))
    for name, m in add_new_e2_mass.items():
        d[f"extra_e2_decoys_mean_{name}"] = round(float(np.mean(window_count(m, P.pmass))), 3)
    dil_rows.append(d)
    print(row, d, flush=True)
rec = pd.DataFrame(rec_rows)

# ---------------------------------------------------------------- where the 0.477's final 25 come from (bench477)
sub = pd.read_csv(B477 / "submission.csv")
lists = {m: [x for x in s.split(";") if x] for m, s in zip(sub.molecule_id, sub.smiles.fillna(""))}
fc = pd.DataFrame([(m, i + 1, x) for m, l in lists.items() for i, x in enumerate(l)], columns=["mid", "rank", "smiles"])
F = std_frame(fc.smiles, "bench477_final")
fc["v4pool"] = hit(F, set(pm.key)).values
fc["e2_extra"] = (~fc.v4pool & hit(F, E2).values)
fc["tier_only"] = (~fc.v4pool & ~fc.e2_extra & hit_tier(F))
fc["no_database"] = ~(fc.v4pool | fc.e2_extra | fc.tier_only)
comp = {c: round(float(fc[c].mean()), 4) for c in ["v4pool", "e2_extra", "tier_only", "no_database"]}
comp_top1 = {c: round(float(fc[fc["rank"] == 1][c].mean()), 4) for c in ["v4pool", "e2_extra", "tier_only", "no_database"]}
sizes = pd.Series({m: len(l) for m, l in lists.items()})

# ---------------------------------------------------------------- expected public gain (upper bounds)
est = {}
for name in ADD:
    g2 = rec.set_index("probe")[f"gain_union_{name}"] / rec.set_index("probe").n
    e2g = rec.set_index("probe")[f"gain_e2_{name}"] / rec.set_index("probe").n
    est[name] = dict(
        class2_newly_reachable_rate_P2=round(float(g2["P2_npatlas_2021_2024_not_in_train"]), 4),
        class2_moved_into_engine2_rate_P2=round(float(e2g["P2_npatlas_2021_2024_not_in_train"]), 4),
        mrr_gain_upper_bound=round(SHARE[2] * float(max(g2.max(), e2g.max())) * RR_NEW, 5),
        class3_reach="0 by the host's definition (absent from PubChem and COCONUT); new neutral structures outside "
                     "every current source: " + str(int(adds.set_index("addition").loc[name, "new_to_every_source_neutral"])))
out = dict(
    current_sources={
        "engine1_v4pool": "train.parquet structures U prvsiyan COCONUT (710,701 metric keys), +-10 ppm window "
                          "(30 ppm fallback), no formula filter, plus generated derivatives of strong analogs",
        "engine2_pool": f"prvsiyan COCONUT {len(cm['keys']):,} + ChEBI/LMSD (sergiomalv, 98,433 regenerated) + "
                        "PubChemLite CC0 (franciscoangulo; proxy here xjoannax88 347,250) + train; +-10 ppm (30 fallback), "
                        "no formula filter; top 25 into weighted RRF with engine 1",
        "pubchem_channel": f"PubChem tier {len(pc_mass):,} rows 150-1250 Da (+ delta_pc/growth_pc supplements of the "
                           "popularity dataset, not local); +-10 ppm; pass-1 top 5,000 by partial f.z plus the 200 most "
                           "documented; 25 non-pool structures by z(f.z) + 0.15 popularity; enter the final list only "
                           "through the gated merge (promotion rel > 150, candidate-level gate)",
        "post": "neural RRF, leader-safe consensus, fragment check, merge (0.477); 0.479 adds K14 dedup and fills "
                "empty slots from scored outsiders (fusion overflow, PubChem, engine 2, main tail); no new database",
        "bench477_final_list_sizes": sizes.describe().round(3).to_dict(),
        "bench477_final_candidate_origin_all_ranks": comp, "bench477_final_candidate_origin_rank1": comp_top1},
    additions=adds.to_dict("records"), probe_recall=rec.to_dict("records"), dilution=dil_rows, expected=est)
ART.mkdir(exist_ok=True)
json.dump(out, open(ART / "57_summary.json", "w"), indent=1, default=float)
rec.to_csv(ART / "57_probe_recall.csv", index=False)
adds.to_csv(ART / "57_additions.csv", index=False)
print(json.dumps({k: out[k] for k in ("expected",)}, indent=1))
print(json.dumps(out["current_sources"], indent=1, default=float))
for d in dil_rows:
    print(d)
