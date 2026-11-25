import sys,os,glob; sys.path.insert(0,'/home/claude/ultrasinger/eval'); import numpy as np, usdx
from metrics import _grid,_rasterize
U="/home/claude/ultrasinger"
key=lambda s:"".join(c for c in s if c.isalnum())[:12]
def gold_frames(song,t):
    g=np.zeros(t.size,bool)
    for n in song.notes:
        if n.golden: a,b=np.searchsorted(t,[n.start,n.end]); g[a:b]=True
    return g
res={}
stats=[]
for data,run in [("usdb","plf_planned"),("dev","plfdev_planned")]:
    for song in sorted(os.listdir(f"{U}/eval_data/{data}")):
        gp=usdx.find_song_txt(f"{U}/eval_data/{data}/{song}")
        pr=[p for p in glob.glob(f"{U}/eval_work/runs/{run}/*/*/*.txt") if key(os.path.basename(os.path.dirname(os.path.dirname(p))))==key(song)]
        if not gp or not pr or "Systemabsturz" in song: continue
        gt=usdx.read(gp); pred=usdx.read(pr[0])
        stats.append((song[:20], np.mean([n.golden for n in gt.notes]), np.mean([n.end-n.start for n in gt.notes if n.golden]) if any(n.golden for n in gt.notes) else 0, np.mean([n.end-n.start for n in gt.notes])))
        t=_grid([gt,pred]); gg=gold_frames(gt,t)
        durs=np.array([n.end-n.start for n in pred.notes])
        for share in [0.03,0.05,0.08,0.12]:
            thr=np.quantile(durs,1-share); pg=np.zeros(t.size,bool)
            for n,d in zip(pred.notes,durs):
                if d>=thr: a,b=np.searchsorted(t,[n.start,n.end]); pg[a:b]=True
            tp=(gg&pg).sum(); p=tp/max(1,pg.sum()); r=tp/max(1,gg.sum())
            res.setdefault(share,[]).append((p,r,2*p*r/max(1e-9,p+r)))
for s in stats: print(f"{s[0]:20s} golden share {s[1]:.2f}  golden note dur {s[2]:.2f}s vs all {s[3]:.2f}s")
for share,v in res.items():
    v=np.array(v); print(f"longest {share:.0%} of notes golden: precision {v[:,0].mean():.2f} recall {v[:,1].mean():.2f} F1 {v[:,2].mean():.2f}")
