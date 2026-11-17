"""Post-hoc Takens (chapman8), so folds 1-8. A: homofilia; B: informacao nos vaos."""
import sys, json, zipfile, numpy as np, pandas as pd
from numpy.lib import format as npf
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
from sklearn.metrics import roc_auc_score
from evaluate_augmented import BOOSTING
from evaluate_views import BoostingPorRotulo
from threadpoolctl import threadpool_limits
SP='results/external/chapman8/posthoc_takens/'
def membro(caminho, nome):
    z=zipfile.ZipFile(caminho); info=z.getinfo(nome)
    f=open(caminho,'rb'); f.seek(info.header_offset); h=f.read(30)
    n,e=int.from_bytes(h[26:28],'little'),int.from_bytes(h[28:30],'little')
    f.seek(info.header_offset+30+n+e)
    v=npf.read_magic(f); shape,fo,dt=(npf.read_array_header_1_0 if v==(1,0) else npf.read_array_header_2_0)(f)
    return np.memmap(caminho,dtype=dt,mode='r',offset=f.tell(),shape=shape,order='F' if fo else 'C')
def abrir(c):
    return {k:membro(c,k+'.npy') for k in ('ecg_id','sinais','arestas','offsets')}
T=abrir('data/features/graphs/takens_edges_chapman.npz'); H=abrir('data/features/graphs/hvg_edges_chapman.npz')
assert np.array_equal(T['ecg_id'],H['ecg_id'])
lab=pd.read_csv('data/external/chapman8/labels.csv').set_index('ecg_id')
rot=json.load(open('data/external/chapman8/prep.json'))['labels']
ids=np.asarray(T['ecg_id']); pos={e:i for i,e in enumerate(ids)}
fold=lab['strat_fold']; usar=fold.index[fold.between(1,8)]
linhas=np.array([pos[e] for e in usar])
BINS=np.unique(np.round(np.geomspace(1,1000,26)).astype(int)); BINS[-1]=1001
def feats(c, l):
    sig=np.asarray(c['sinais'][l]); out_h=[]; out_m=[]; hom=[]
    for d in range(12):
        g=l*12+d; a=np.asarray(c['arestas'][c['offsets'][g]:c['offsets'][g+1]]).astype(np.int64)
        x=sig[d]; x=(x-x.mean())/(x.std()+1e-9)
        if len(a)==0:
            out_h.append(np.zeros(len(BINS)-1)); out_m.append(np.full(6,np.nan)); hom.append(np.nan); continue
        s=a[:,1]-a[:,0]
        out_h.append(np.histogram(s,BINS)[0]/len(s))
        o=np.concatenate([a[:,0],a[:,1]]); dd=np.concatenate([a[:,1],a[:,0]]); ls=np.log(np.concatenate([s,s]))
        grau=np.bincount(dd,minlength=1000); ok=grau>0
        mlog=np.bincount(dd,ls,1000)[ok]/grau[ok]                      # o que a GNN ve: media do log do vao por no
        out_m.append([mlog.mean(),mlog.std(),*np.percentile(mlog,[10,50,90]),ok.mean()])
        xv=np.bincount(dd,x[o],1000)[ok]/grau[ok]
        hom.append(np.corrcoef(x[ok],xv)[0,1])
    return np.concatenate(out_h),np.concatenate(out_m),np.array(hom)
from joblib import Parallel, delayed
def bloco(c, ls): return [feats(c,l) for l in ls]
partes=np.array_split(linhas,48)
R={}
for nome,c in (('takens',T),('hvg',H)):
    r=sum(Parallel(n_jobs=12)(delayed(bloco)(c,p) for p in partes),[])
    R[nome]=[np.stack([x[k] for x in r]) for k in range(3)]
    print(nome,'homofilia (corr x_no vs media x_vizinhos), mediana por derivacao:',np.round(np.nanmedian(R[nome][2],0),3),flush=True)
np.savez(SP+'takens_posthoc_feats.npz',ecg_id=usar.to_numpy(),**{f'{n}_{k}':R[n][i] for n in R for i,k in enumerate(('hist','mlog','hom'))})
f=fold.loc[usar].to_numpy(); y=lab.loc[usar,rot].to_numpy(np.int8)
tr,va=f<=7,f==8
def av(X):
    X=X.astype(np.float32)
    with threadpool_limits(4):
        m=BoostingPorRotulo(max_iter=2000,**BOOSTING).fit(X[tr],y[tr],X[va],y[va])
        s=m.predict_proba(X[va])
    return float(np.mean([roc_auc_score(y[va][:,i],s[:,i]) for i in range(len(rot))]))
res={}
for nome in ('takens','hvg'):
    res[f'{nome}: media log vao por no (o que a GNN ve), 6x12']=av(R[nome][1])
    res[f'{nome}: histograma completo dos vaos, {len(BINS)-1}x12']=av(R[nome][0])
for k,v in res.items(): print(f'{v:.4f}  {k}',flush=True)
json.dump(res,open(SP+'takens_posthoc_B.json','w'),indent=2)
