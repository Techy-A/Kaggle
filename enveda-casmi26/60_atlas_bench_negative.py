"""60: ICEBERG atlas on the 18 negative-only bench molecules (3 Nov). What: per molecule, download the atlas [M-H]- predictions for its formula and rank the true structure by binned cosine to the bench spectra. Why: step 3 of the atlas plan (59 validated the signal on train). Output: artifacts/60_atlas_bench.csv, result in CASMI26.md. Usage: python 60_atlas_bench_negative.py (network + CPU; needs the bench477 outputs)."""
import numpy as np, pandas as pd, requests, re, json
from collections import defaultdict
B='/home/user/work/out_bench477/'
neg=pd.read_csv('/home/user/work/neg_bench_ids.csv').iloc[:,0].tolist()
ans=pd.read_csv(B+'bench_answers.csv').set_index('molecule_id')
te=pd.read_parquet(B+'bench_comp/test.parquet')
d=json.load(open('/home/user/kaggle/enveda-casmi26/artifacts/bench477_results.json'));rrb=d['baseline0465']['rr']
BIN=0.01
def vec(mz,it):
    s=defaultdict(float)
    for x,y in zip(np.round(np.asarray(mz)/BIN).astype(np.int64),np.sqrt(np.asarray(it,float))): s[x]=max(s[x],y)
    return s
def cos(a,b):
    n=sum(v*v for v in a.values())**.5*sum(v*v for v in b.values())**.5
    return sum(v*b.get(k,0) for k,v in a.items())/n if n else 0
rows=[]
for m in neg:
    r=ans.loc[m];ik=r.inchikey14;f=r.molecular_formula
    q=te[te.molecule_id==m];a=defaultdict(float)
    for _,x in q.iterrows():
        for k,v in vec(x.ms2_mzs,x.ms2_normalized_intensities).items(): a[k]=max(a[k],v)
    ad=sorted(set(q.adduct));adduct='[M-H]-' if '[M-H]-' in ad else None
    if adduct is None: rows.append((m,r.cls,f,ad,None,None,None,rrb[m]));print(m,r.cls,f,ad,'no [M-H]- spectrum (formate adduct only)');continue
    txt=requests.get('https://iceberg-ms.mit.edu/download_mgf',params={'formula':f,'adduct':adduct},timeout=300).text
    sc={}
    for rec in txt.split('END IONS'):
        if 'BEGIN IONS' not in rec: continue
        k2=re.search(r'INCHIKEY=(\S+)',rec).group(1)[:14];pk=re.findall(r'^(\d+\.\d+) ([\d.eE+-]+)$',rec,re.M)
        sc[k2]=max(sc.get(k2,0),cos(a,vec([float(p[0]) for p in pk],[float(p[1]) for p in pk])))
    rank=(1+sum(v>sc[ik] for v in sc.values())) if ik in sc else None
    rows.append((m,r.cls,f,ad,len(sc),rank,ik in sc,rrb[m]));print(m,r.cls,f,'n',len(sc),'atlas rank',rank,'baseline rr',round(rrb[m],3),flush=True)
df=pd.DataFrame(rows,columns=['mol','cls','f','ads','n','atlas_rank','in_atlas','base_rr']);df.to_csv('atlas_bench.csv',index=False)
print(df.groupby('cls').agg(n=('mol','count'),in_atlas=('in_atlas','sum'),top1=('atlas_rank',lambda x:(x==1).sum()),top5=('atlas_rank',lambda x:(x<=5).sum()),base_hits=('base_rr',lambda x:(x>0).sum())))
