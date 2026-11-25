"""ROVER-style word voting over cached transcriptions of the clips; WER vs chart lyrics."""
import sys,os,glob,json,re,difflib,itertools; sys.path.insert(0,'/home/claude/ultrasinger/eval')
import numpy as np, jiwer, usdx
def norm(w): return re.sub(r"[^\w']","",w.lower())
def load(f): return [norm(x["word"]) for x in json.load(open(f))["transcribed_data"] if norm(x["word"])]
def rover(hyps):
    """pivot = first hypothesis; vote each pivot word against the aligned words of the others;
    drop a pivot word if most hypotheses have nothing there"""
    piv=hyps[0]; votes=[[w] for w in piv]; gaps=[0]*len(piv)
    for h in hyps[1:]:
        for t,i1,i2,j1,j2 in difflib.SequenceMatcher(a=piv,b=h,autojunk=False).get_opcodes():
            if t=="equal":
                for k in range(i2-i1): votes[i1+k].append(h[j1+k])
            elif t=="replace" and i2-i1==j2-j1:
                for k in range(i2-i1): votes[i1+k].append(h[j1+k])
            elif t in ("delete","replace"):
                for k in range(i1,i2): gaps[k]+=1
    out=[]
    for v,g in zip(votes,gaps):
        if g > len(hyps)/2: continue
        out.append(max(set(v), key=lambda w:(v.count(w), w==v[0])))
    return out
names={"turbo":"whisper_large-v3-turbo_cpu_None_4_None_None.json","v3":"whisper_large-v3_cpu_None_4_None_None.json",
       "v2":"whisper_large-v2_cpu_None_4_None_None.json","c15":"whisper_large-v3-turbo_cpu_None_4_None_None_chunk15.json",
       "c10":"whisper_large-v3-turbo_cpu_None_4_None_None_chunk10.json","medium":"../whisper_medium_cpu_None_4_None_None.json"}
res={}
for d in ["usdb_clips","dev_clips"]:
    for song in sorted(os.listdir(f"/home/claude/ultrasinger/eval_data/{d}")):
        if "Systemabsturz" in song or "Kouji" in song: continue
        gp=usdx.find_song_txt(f"/home/claude/ultrasinger/eval_data/{d}/{song}")
        if not gp: continue
        c=glob.glob(f"/home/claude/ultrasinger/eval_work_clips/cache/{glob.escape(song.replace('&','_').replace(',','_'))}/demucs_htdemucs_ft/")
        if not c: c=[x for x in glob.glob("/home/claude/ultrasinger/eval_work_clips/cache/*/demucs_htdemucs_ft/") if os.path.basename(os.path.dirname(os.path.dirname(x)))[:12]==song[:12]]
        if not c: print("no cache",song); continue
        H={k:load(c[0]+f) for k,f in names.items() if os.path.exists(c[0]+f)}
        if len(H)<6: print("missing",song,list(H)); continue
        g=" ".join(norm(w) for w in usdx.read(gp).words() if norm(w))
        r={k:jiwer.wer(g," ".join(v)) for k,v in H.items()}
        for combo in [("turbo","v3","v2"),("turbo","v3","v2","medium"),("turbo","v3","v2","c15","c10"),("turbo","v3","v2","c15","c10","medium"),("turbo","c15","c10")]:
            r["rover:"+"+".join(combo)]=jiwer.wer(g," ".join(rover([H[k] for k in combo])))
        r["oracle_best_single"]=min(r[k] for k in H)
        res[song[:18]]=r
import pandas as pd; pd.set_option("display.width",250)
d=pd.DataFrame(res).T; print(d.round(2).to_string()); print(d.mean().round(3).to_string())
