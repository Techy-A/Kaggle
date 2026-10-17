"""Stream the Enveda-180 Zenodo release (CC BY 4.0, record 21346580) and keep only spectra whose structure is NOT in
train.parquet: molecules no competition model has seen, i.e. a leak-free pseudo-test. Reads gzip JSON lines from stdin."""
import gzip, json, sys
import pyarrow.parquet as pq
train = set(pq.read_table('/home/user/enveda_data/train.parquet', columns=['inchikey14']).column(0).to_pylist())
out = open(sys.argv[1], 'w'); n = kept = 0; adducts = {}
with gzip.open(sys.stdin.buffer, 'rt') as f:
    for line in f:
        n += 1
        r = json.loads(line)
        ik = (r.get('inchikey') or '')[:14]
        if ik and ik not in train:
            out.write(line); kept += 1; adducts[r.get('adduct')] = adducts.get(r.get('adduct'), 0) + 1
        if n % 200000 == 0:
            print(n, 'read', kept, 'kept', flush=True)
out.close()
print('done', n, 'read', kept, 'kept'); print(sorted(adducts.items(), key=lambda x: -x[1])[:15])
