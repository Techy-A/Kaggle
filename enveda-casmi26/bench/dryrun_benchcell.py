import json, os, sys, time
nb = json.load(open('/home/user/work/push_bench/notebook.ipynb'))
src = ''.join(nb['cells'][14]['source'])
KW = os.path.abspath('kw')
src = src.replace('/kaggle/working', KW)
g = dict(os=os, time=time, V4CODE='/home/user/benchdata/v4b/code', COMP='/home/user/enveda_data',
         POOL_DIR=os.path.abspath('pool'), E004_PREFLIGHT={}, ICE_BUDGET=300, IS_RERUN=False)
exec(compile(src, 'bench_cell', 'exec'), g)
import pandas as pd, numpy as np, pyarrow.parquet as pq
ans = pd.read_csv(KW + '/bench_answers.csv'); te = pd.read_parquet(KW + '/bench_comp/test.parquet')
t0 = pd.read_parquet('/home/user/enveda_data/test.parquet')
print('answers', ans.cls.value_counts().to_dict(), 'metric None', ans.metric_key.isna().sum())
print('test cols equal', list(te.columns) == list(t0.columns), te.dtypes.equals(t0.dtypes), te.shape, te.molecule_id.nunique())
tr = pq.ParquetFile(KW + '/bench_comp/train.parquet')
ik = pq.read_table(KW + '/bench_comp/train.parquet', columns=['inchikey14', 'ingest_lib']).to_pandas()
c23 = set(ans[ans.cls >= 2].inchikey14); c1 = set(ans[ans.cls == 1].inchikey14)
print('train rows', len(ik), '| np rows left', (ik.ingest_lib == 'enveda-np-examples').sum(),
      '| class2/3 ik left', ik.inchikey14.isin(c23).sum(), '| class1 structures still with spectra', ik[ik.inchikey14.isin(c1)].inchikey14.nunique(), '/', len(c1))
pm = pd.read_parquet(KW + '/bench_pool/pool_meta.parquet'); keep = np.load(KW + '/bench_pool/keep_mask.npy')
d3 = set(json.load(open(KW + '/bench_drop3.json')))
print('pool kept', len(pm), 'dropped', (~keep).sum(), '| class3 keys left in pool', pm.key.isin(d3).sum(), '| pid ok', (pm.pid.values == np.arange(len(pm))).all())
fo = np.load(KW + '/bench_pool/pool_frag_off.npy'); fm = np.load(KW + '/bench_pool/pool_frag_mass.npy'); fp = np.load(KW + '/bench_pool/pool_fp.npy')
off0 = np.load('pool/pool_frag_off.npy'); fm0 = np.load('pool/pool_frag_mass.npy'); fp0 = np.load('pool/pool_fp.npy')
rows = np.flatnonzero(keep); j = len(rows) // 2
print('frag check', np.array_equal(fm[fo[j]:fo[j + 1]], fm0[off0[rows[j]]:off0[rows[j] + 1]]), fo[-1] == len(fm), '| fp check', np.array_equal(fp[j], fp0[rows[j]]), fp.shape[0] == len(pm))
print('class-3 answers that were in the v4 pool:', len(set(ans[ans.cls == 3].metric_key) & set(pd.read_parquet('pool/pool_meta.parquet').key)), 'of', (ans.cls == 3).sum())
print('preflight ids', len(g['E004_PREFLIGHT']['input_ids']), g['COMP'], g['IS_RERUN'], g['ICE_BUDGET'])
