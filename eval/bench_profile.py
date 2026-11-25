"""Profile full songs in ~20 s windows (starting at chart line starts): chart/audio features and
current-best performance per window. Writes /tmp/bench/windows.csv."""
import sys,os,glob; sys.path.insert(0,'/home/claude/ultrasinger/eval')
import numpy as np, pandas as pd, librosa, usdx
from metrics import _grid,_rasterize,_chroma_dist
U="/home/claude/ultrasinger"
WIN=20.0
rows=[]
for data,run in [("usdb","plf_planned"),("dev","plfdev_planned")]:
    for song in sorted(os.listdir(f"{U}/eval_data/{data}")):
        if "Systemabsturz" in song: continue
        gp=usdx.find_song_txt(f"{U}/eval_data/{data}/{song}")
        key=lambda s:"".join(c for c in s if c.isalnum())[:12]
        pr=[p for p in glob.glob(f"{U}/eval_work/runs/{run}/*/*/*.txt") if key(os.path.basename(os.path.dirname(os.path.dirname(p))))==key(song)]
        voc=[p for p in glob.glob(f"{U}/eval_work/cache/*/separated/htdemucs/*/vocals.wav") if key(os.path.basename(os.path.dirname(p)))==key(song)]
        if not gp or not pr or not voc: print("skip",song,bool(gp),bool(pr),bool(voc)); continue
        gt=usdx.read(gp); pred=usdx.read(pr[0]); lang=gt.language
        mixf=voc[0].replace("vocals.wav","no_vocals.wav")
        v,_=librosa.load(voc[0],sr=16000); acc,_=librosa.load(mixf,sr=16000)
        song_rms=np.sqrt(np.mean(v**2))+1e-9
        t=_grid([gt,pred]); gv,gpch,gg,gpitch=_rasterize(gt,t); pv,_,_,ppitch=_rasterize(pred,t)
        have=gpch&~np.isnan(ppitch); dist=np.full(t.size,np.inf); dist[have]=_chroma_dist(gpitch[have],ppitch[have])
        # windows: greedy, start at line starts
        starts=[n.start for k,n in enumerate(gt.notes) if k==0 or n.line_start]
        w0=starts[0]-0.3; total=gt.notes[-1].end
        while w0<total-5:
            w1=w0+WIN
            nxt=[s for s in starts if s>=w1-3]
            w1=nxt[0]-0.3 if nxt and nxt[0]-0.3<w0+WIN+5 else w0+WIN
            sel=[n for n in gt.notes if n.start>=w0 and n.end<=w1]
            if len(sel)>=8:
                m=(t>=w0)&(t<w1); gp_=gpch&m
                pitches=np.array([n.pitch for n in sel if n.pitched],float)
                durs=np.array([n.end-n.start for n in sel])
                a,b=int(w0*16000),int(w1*16000)
                vr=np.sqrt(np.mean(v[a:b]**2))+1e-9; ar=np.sqrt(np.mean(acc[a:b]**2))+1e-9
                cent=np.mean(librosa.feature.spectral_centroid(y=v[a:b],sr=16000)) if b>a else np.nan
                words=sum(1 for n in sel if n.text.endswith(" ") or n.text.startswith(" "))
                rows.append(dict(song=song,set=data,lang=lang,start=round(w0,2),end=round(w1,2),
                    sung_share=(gv&m).sum()/m.sum(), note_rate=len(sel)/(w1-w0), note_dur=np.median(durs),
                    short_share=np.mean(durs<0.2), pitch_std=np.std(pitches) if len(pitches) else 0,
                    pitch_range=np.ptp(pitches) if len(pitches) else 0, pitch_mean=np.mean(pitches) if len(pitches) else 0,
                    golden_share=np.mean([n.golden for n in sel]), unpitched_share=np.mean([not n.pitched for n in sel]),
                    words_per_s=words/(w1-w0), vocal_level=20*np.log10(vr/song_rms), vocal_to_backing=20*np.log10(vr/ar),
                    vocal_centroid=cent, position=w0/total,
                    usdx_medium=((gp_&(dist<=1)).sum()/gp_.sum()) if gp_.sum() else np.nan,
                    usdx_hard=((gp_&(dist==0)).sum()/gp_.sum()) if gp_.sum() else np.nan,
                    missed=((gp_&~pv).sum()/gp_.sum()) if gp_.sum() else np.nan))
            w0=w1
d=pd.DataFrame(rows); d.to_csv("/tmp/bench/windows.csv",index=False)
print(len(d),"windows from",d.song.nunique(),"songs"); print(d.describe().T[["mean","std","min","max"]].round(2))
