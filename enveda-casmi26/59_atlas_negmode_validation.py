"""59: ICEBERG atlas negative-mode validation (3 Nov). What: 80 train molecules with only [M-H]- spectra; download the public atlas [M-H]- predictions for each formula and rank same-formula isomers by binned cosine to the real spectrum. Why: 27 of 400 test molecules (6.75%) are negative-only and get no ICEBERG/GLACIER evidence. Output: atlas_val.csv; result in CASMI26.md. Usage: python 59_atlas_negmode_validation.py (network + CPU).
near the top among all same-formula PubChem isomers? CPU/network only."""
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds, requests, re, json, time, sys
from collections import defaultdict
rng=np.random.default_rng(0)
mt=pd.read_parquet('/home/user/enveda_data/train_meta.parquet',columns=['inchikey14','molecular_formula','adduct','precursor_mz'])
g=mt.groupby('inchikey14').agg(f=('molecular_formula','first'),ads=('adduct',lambda x:set(x)),mz=('precursor_mz','median'))
neg=g[g.ads.map(lambda a:a=={'[M-H]-'}) & g.mz.between(200,600)]
neg=neg[~neg.f.str.contains('Cl|Br|I|Si|B|Se|As|Pt|Fe|Na|K')]
print('neg-only molecules',len(neg))
sample=neg.sample(80,random_state=1)
d=ds.dataset('/home/user/enveda_data/train.parquet')
q=d.to_table(columns=['inchikey14','ms2_mzs','ms2_normalized_intensities','collision_energy_ev'],filter=ds.field('inchikey14').isin(list(sample.index))).to_pandas()
print('query rows',len(q))
BIN=0.01
def vec(mz,it):
    b=np.round(np.asarray(mz)/BIN).astype(np.int64);w=np.sqrt(np.asarray(it,float))
    s=defaultdict(float)
    for x,y in zip(b,w): s[x]=max(s[x],y)
    return s
def cos(a,b):
    n=sum(v*v for v in a.values())**.5*sum(v*v for v in b.values())**.5
    return sum(v*b.get(k,0) for k,v in a.items())/n if n else 0
def parse(txt):
    for rec in txt.split('END IONS'):
        if 'BEGIN IONS' not in rec: continue
        ik=re.search(r'INCHIKEY=(\S+)',rec).group(1)[:14]
        peaks=re.findall(r'^(\d+\.\d+) ([\d.eE+-]+)$',rec,re.M)
        yield ik,peaks
res=[]
for ik,row in sample.iterrows():
    qs=q[q.inchikey14==ik]
    a=defaultdict(float)
    for _,r in qs.iterrows():
        for k,v in vec(r.ms2_mzs,r.ms2_normalized_intensities).items(): a[k]=max(a[k],v)
    t0=time.time()
    try: txt=requests.get('https://iceberg-ms.mit.edu/download_mgf',params={'formula':row.f,'adduct':'[M-H]-'},timeout=300).text
    except Exception as e: print(ik,'ERR',e); continue
    sc={}
    for k2,peaks in parse(txt):
        mz=[float(p[0]) for p in peaks];it=[float(p[1]) for p in peaks]
        b=vec(mz,it); sc[k2]=max(sc.get(k2,0),cos(a,b))
    if ik not in sc: res.append((ik,row.f,len(sc),None));print(ik,row.f,len(sc),'truth not in atlas');continue
    rank=1+sum(v>sc[ik] for v in sc.values())
    res.append((ik,row.f,len(sc),rank));print(ik,row.f,len(sc),rank,round(time.time()-t0,1),flush=True)
df=pd.DataFrame(res,columns=['ik','f','n','rank']);df.to_csv('atlas_val.csv',index=False)
ok=df.dropna();print('in atlas',len(ok),'of',len(df),'| MRR',(1/ok['rank']).mean(),'top1',(ok['rank']==1).mean(),'top5',(ok['rank']<=5).mean(),'| random MRR',(1/ok.n).mean()*0+np.mean([np.log(n)/n if n>1 else 1 for n in ok.n]))
