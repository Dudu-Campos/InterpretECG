"""Suavizacao excessiva: variancia que sobra apos k passos de media sobre vizinhos (operador da GNN), e alcance em k saltos."""
import sys, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, 'scripts')
from posthoc_takens_io import abrir
T=abrir('data/features/graphs/takens_edges_chapman.npz'); H=abrir('data/features/graphs/hvg_edges_chapman.npz')
lab=pd.read_csv('data/external/chapman8/labels.csv').set_index('ecg_id')
ids=np.asarray(T['ecg_id']); f=lab['strat_fold'].reindex(ids).fillna(99).to_numpy()
rng=np.random.default_rng(0); amostra=rng.choice(np.where(f<=7)[0],150,replace=False)
res={n:{'var':[], 'alc':[]} for n in ('takens','hvg')}
for l in amostra:
    for d in range(12):
        x=np.asarray(T['sinais'][l,d],float); x=(x-x.mean())/(x.std()+1e-9)
        for n,c in (('takens',T),('hvg',H)):
            g=l*12+d; a=np.asarray(c['arestas'][c['offsets'][g]:c['offsets'][g+1]]).astype(int)
            if len(a)==0: continue
            A=sp.coo_matrix((np.ones(2*len(a)),(np.r_[a[:,0],a[:,1]],np.r_[a[:,1],a[:,0]])),shape=(1000,1000)).tocsr()
            gr=np.asarray(A.sum(1)).ravel(); ok=gr>0
            P=sp.diags(np.where(ok,1/np.maximum(gr,1),0))@A
            v=[];h=x.copy()
            for k in range(4):
                h=np.where(ok,P@h,h); v.append(h[ok].var()/x[ok].var())
            res[n]['var'].append(v)
            R=(A+sp.eye(1000)).astype(bool).astype(float); M=R.copy(); alc=[]
            for k in range(4):
                alc.append((M>0).sum(1).mean()/1000); M=M@R
            res[n]['alc'].append(alc)
for n in res:
    print(n,'variancia restante apos k=1..4 medias:',np.round(np.nanmedian(res[n]['var'],0),3), 'nan:',int(np.isnan(np.array(res[n]['var'])[:,0]).sum()),
          '| fracao do grafo alcancada em k=1..4 saltos:',np.round(np.median(res[n]['alc'],0),3))
