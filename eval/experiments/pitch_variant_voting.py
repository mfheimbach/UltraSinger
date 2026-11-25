import sys,os,glob,itertools; sys.path.insert(0,'/home/claude/ultrasinger/eval'); import numpy as np, usdx
from collections import Counter
R={}
for d in ["usdb_clips","dev_clips"]:
  for song in sorted(os.listdir(f"/home/claude/ultrasinger/eval_data/{d}")):
    if "Systemabsturz" in song: continue
    f=f"/tmp/tta/tracks/{song[:40]}.npz"
    gp=usdx.find_song_txt(f"/home/claude/ultrasinger/eval_data/{d}/{song}")
    if not os.path.exists(f) or not gp: continue
    z=np.load(f); keys=sorted({k.split("__")[0] for k in z.files})
    gt=[n for n in usdx.read(gp).notes if n.kind in (":","*") and n.end-n.start>0.05]
    P={}
    for k in keys:
        t,m=z[f"{k}__0"],z[f"{k}__1"]; v=[]
        for n in gt:
            sel=m[(t>=n.start)&(t<n.end)]; sel=sel[~np.isnan(sel)]
            v.append(np.round(np.median(sel)) if len(sel) else np.nan)
        P[k]=np.array(v)
    g=np.array([n.pitch+48 for n in gt],float)
    ok=~np.isnan(P["orig"])
    shift=max(range(12),key=lambda s:np.mean(((P["orig"][ok]-g[ok]-s)%12)==0))
    R[song]={k:(((v-g-shift)%12)==0)&~np.isnan(v) for k,v in P.items()} | {"_P":P,"_g":g-(-shift)}
keys=[k for k in next(iter(R.values())) if not k.startswith("_")]
def acc(k): return np.mean([np.mean(r[k]) for r in R.values()])
base=acc("orig")
print(f"{len(R)} clips, note accuracy orig {base:.3f}")
rows=[]
for k in keys:
    a=acc(k); o=np.concatenate([r["orig"] for r in R.values()]); v=np.concatenate([r[k] for r in R.values()])
    rows.append((k,a,np.mean(v[~o]),np.mean(~v[o]), np.mean([np.mean(r[k])>=np.mean(r["orig"]) for r in R.values()])))
print("variant   acc    fixes(orig wrong->right)  breaks(orig right->wrong)  clips>=orig")
for k,a,fx,br,w in sorted(rows,key=lambda x:-x[1]): print(f"{k:8s} {a:.3f}   {fx:.3f}   {br:.3f}   {w:.0%}")
def vote(ks):
    out=[]
    for r in R.values():
        P=r["_P"]; g=r["_g"]; res=[]
        for i in range(len(g)):
            c=Counter(P[k][i] for k in ks if not np.isnan(P[k][i]))
            if not c: res.append(False); continue
            top=max(c.values()); cands=[p for p,n in c.items() if n==top]
            p=P["orig"][i] if P["orig"][i] in cands else cands[0]
            res.append(((p-g[i])%12)==0)
        out.append(np.mean(res))
    return np.mean(out)
good=[k for k,a,fx,br,w in rows if k!="orig" and a>=base-0.005]
print("candidates within 0.005 of orig:", good)
res=[]
for n in [2,4]:
    for combo in itertools.combinations(good,n):
        res.append((vote(("orig",)+combo),combo))
for a,c in sorted(res,reverse=True)[:8]: print(f"vote orig+{'+'.join(c)}: {a:.3f}")
