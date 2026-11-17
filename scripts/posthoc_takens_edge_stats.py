import sys, json, numpy as np, pandas as pd
sys.path.insert(0,'scripts'); sys.path.insert(0,'src')
from build_hvg_cache import arestas_takens, EMBEDDING
from interpretecg.data import LEADS
from interpretecg.views.series_to_graph.visibility import _hvg_edges
from interpretecg.external import carregar
from joblib import Parallel, delayed
ids, sig, lab, nomes = carregar('chapman8')
f = lab['strat_fold'].to_numpy()
rng = np.random.default_rng(0)
idx = rng.choice(np.where(f <= 7)[0], 240, replace=False)
def um(i):
    out=[]
    for d in range(12):
        x=sig[i,d]
        t=arestas_takens(x,d); h=np.array(list(_hvg_edges(x.astype(float))))
        m,tau=EMBEDDING[LEADS[d]]
        st=(t[:,1]-t[:,0]).astype(int) if len(t) else np.array([1])
        sh=h[:,1]-h[:,0]
        out.append(dict(i=int(i),lead=LEADS[d],m=m,tau=tau,win=(m-1)*tau,n_t=len(t),n_h=len(h),
            t_le5=np.mean(st<=5),t_lewin=np.mean(st<=(m-1)*tau),t_med=np.median(st),
            t_meanlog=np.mean(np.log(st)),h_meanlog=np.mean(np.log(sh)),
            nodes_t_with_edges=len(np.unique(t)) if len(t) else 0,
            flat=np.ptp(x)==0))
    return out
r=sum(Parallel(n_jobs=12)(delayed(um)(i) for i in idx),[])
d=pd.DataFrame(r)
d.to_csv('results/external/chapman8/posthoc_takens/takens_spans.csv',index=False)
pd.set_option('display.width',200)
print(d.groupby('lead')[['m','tau','win','n_t','n_h','t_le5','t_lewin','t_med','t_meanlog','h_meanlog','nodes_t_with_edges']].median().round(3))
print('geral: arestas takens mediana',d.n_t.median(),'max',d.n_t.max(),'| hvg mediana',d.n_h.median())
print('frac arestas com vao<=5:',d.t_le5.median().round(3),' vao<=janela de embedding:',d.t_lewin.median().round(3))
print('derivacoes com >100k arestas:',(d.n_t>1e5).sum(),'de',len(d),'; planas:',d.flat.sum())
