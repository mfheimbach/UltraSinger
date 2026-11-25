"""Does wav2vec2 alignment confidence (or text compression) pick the better transcription per clip?"""
import sys,os,glob,json,re,zlib; sys.path.insert(0,'/home/claude/ultrasinger/eval')
import numpy as np, jiwer, usdx
def norm(w): return re.sub(r"[^\w']","",w.lower())
names={"turbo":"whisper_large-v3-turbo_cpu_None_4_None_None.json","v3":"whisper_large-v3_cpu_None_4_None_None.json",
       "v2":"whisper_large-v2_cpu_None_4_None_None.json","c15":"whisper_large-v3-turbo_cpu_None_4_None_None_chunk15.json",
       "c10":"whisper_large-v3-turbo_cpu_None_4_None_None_chunk10.json","medium":"../whisper_medium_cpu_None_4_None_None.json"}
rows=[]
for d in ["usdb_clips","dev_clips"]:
    for song in sorted(os.listdir(f"/home/claude/ultrasinger/eval_data/{d}")):
        if "Systemabsturz" in song or "Kouji" in song: continue
        gp=usdx.find_song_txt(f"/home/claude/ultrasinger/eval_data/{d}/{song}")
        c=[x for x in glob.glob("/home/claude/ultrasinger/eval_work_clips/cache/*/demucs_htdemucs_ft/") if "".join(ch for ch in os.path.basename(os.path.dirname(os.path.dirname(x))) if ch.isalnum())[:12]=="".join(ch for ch in song if ch.isalnum())[:12]]
        if not gp or not c: continue
        g=" ".join(norm(w) for w in usdx.read(gp).words() if norm(w))
        H={}
        for k,f in names.items():
            if not os.path.exists(c[0]+f): continue
            td=json.load(open(c[0]+f))["transcribed_data"]
            words=[x["word"] for x in td if norm(x["word"])]
            conf=np.mean([x.get("confidence") or 0 for x in td]) if td else 0
            text=" ".join(norm(w) for w in words)
            comp=len(text)/max(1,len(zlib.compress(text.encode())))
            H[k]=dict(wer=jiwer.wer(g,text),conf=conf,comp=comp,n=len(words))
        if len(H)<6: continue
        pick_conf=max(H,key=lambda k:H[k]["conf"]); pick_comp=min(H,key=lambda k:H[k]["comp"])
        pick_both=max(H,key=lambda k:H[k]["conf"]-0.1*H[k]["comp"])
        rows.append(dict(song=song[:18],turbo=H["turbo"]["wer"],best=min(v["wer"] for v in H.values()),by_conf=H[pick_conf]["wer"],conf_pick=pick_conf,by_comp=H[pick_comp]["wer"],
                         corr=np.corrcoef([v["conf"] for v in H.values()],[v["wer"] for v in H.values()])[0,1]))
import pandas as pd; d=pd.DataFrame(rows); print(d.round(2).to_string()); print(d.mean(numeric_only=True).round(3).to_dict())
