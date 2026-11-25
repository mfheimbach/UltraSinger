import sys,json,math; sys.path.insert(0,'/tmp/lb'); import dp; from dp import *
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
d=pd.DataFrame(json.load(open("/tmp/lb/f_dev.json"))+json.load(open("/tmp/lb/f_usdb.json")))
songs={}
for run,dd in [("devfs_best","eval_data/dev"),("fullsong2_game_all","eval_data/usdb")]:
    for s,gt,pr in load(run,dd): songs[s]=(gt,[w for w in words_of(pr) if w[2]])
# relabel: each GT line start -> nearest pred word (within 0.5 s)
d["y"]=0
for s,(gt,ws) in songs.items():
    g=[x.start for x in gt.notes[1:] if x.line_start]; idx=d.index[d.song==s]; st=d.loc[idx,"start"].values
    for t in g:
        k=np.argmin(np.abs(st-t))
        if abs(st[k]-t)<0.5: d.loc[idx[k],"y"]=1
d["lpp"]=np.log((d.pp+1e-3)/(1-d.pp+1e-3))
FEATS={"a":0,"b":0,"pause+punct":["lg","lpp"]}
def dp_plan(ws, bscore, alpha, c_mu=23,c_sd=9,d_mu=1.9,d_lsd=0.57,cmax=45,maxw=16):
    n=len(ws); pre=[0]
    for w in ws: pre.append(pre[-1]+len(w[2]))
    best=[-1e18]*(n+1); back=[0]*(n+1); best[0]=0
    for b in range(1,n+1):
        for a in range(max(0,b-maxw),b):
            chars=pre[b]-pre[a]+(b-1-a)
            if chars>cmax and b-1>a: continue
            dur=max(0.05,ws[b-1][1]-ws[a][0])
            sc=best[a]+alpha*(-0.5*((chars-c_mu)/c_sd)**2-0.5*((math.log(dur)-math.log(d_mu))/d_lsd)**2)+(bscore[a] if a>0 else 0)
            if sc>best[b]: best[b],back[b]=sc,a
    st=[];b=n
    while b>0:
        a=back[b]
        if a>0: st.append(a)
        b=a
    return sorted(st)
usdb=set(d[d.song.isin([s for s,_,_ in load("fullsong2_game_all","eval_data/usdb")])].song)
for name,cols in list(FEATS.items())[2:]:
    for alpha in [0.5]:
        res={}; aucs=[]
        for s in songs:
            tr=d[d.song!=s]; te=d[d.song==s]
            m=LogisticRegression(class_weight=None,max_iter=1000).fit(tr[cols],tr.y)
            p=m.predict_proba(te[cols])[:,1]; aucs.append(roc_auc_score(te.y,p) if te.y.nunique()>1 else np.nan)
            gt,ws=songs[s]
            bscore=[0]+list(np.log(p/(1-p)))  # break log-odds before word i
            st=dp_plan(ws,bscore,alpha)
            res[s]=f1(gt,[ws[i][0] for i in st])[0]
        u=np.mean([v for k,v in res.items() if k in usdb]); dv=np.mean([v for k,v in res.items() if k not in usdb])
        print(f"{name:12s} alpha {alpha:<5} AUC {np.nanmean(aucs):.3f} | line F1 USDB {u:.3f} dev {dv:.3f}")
m=LogisticRegression(max_iter=1000).fit(d[["lg","lpp"]],d.y)
print("FINAL coef", dict(zip(["lg","lpp"],m.coef_[0].round(3))), "intercept", round(m.intercept_[0],3), "base rate", round(d.y.mean(),3))
