"""Compute RMVPE tracks for many preprocessing variants of each clip's vocals (cached as npz)."""
import sys,os,glob,contextlib,io; sys.path.insert(0,'/home/claude/ultrasinger/src')
import numpy as np, librosa, scipy.signal as sg
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from modules.Pitcher import alt_pitchers as ap
sess=ort.InferenceSession(hf_hub_download(ap.RMVPE_REPO, ap.RMVPE_FILE), providers=["CPUExecutionProvider"])
SR=16000
def rmvpe(y):
    f0,conf=ap._rmvpe_pass(sess,y); t=np.arange(len(f0))*0.01
    return t, np.where(f0>0, librosa.hz_to_midi(np.maximum(f0,1e-3)), np.nan), conf
def variants(y):
    out={}
    out["orig"]=rmvpe(y)
    for s in [-2,-1,-0.5,0.5,1,2]:
        t,m,c=rmvpe(librosa.effects.pitch_shift(y,sr=SR,n_steps=s)); out[f"pv{s:+g}"]=(t,m-s,c)
        r=2**(s/12)  # resample: playing r times faster raises pitch by s semitones; time scales by 1/r
        ys=librosa.resample(y,orig_sr=SR,target_sr=SR/r)  # fewer samples when r>1
        t,m,c=rmvpe(ys); out[f"rs{s:+g}"]=(t*r,m-s,c)  # time back to the original scale
    b,a=sg.butter(4,150/(SR/2),"high"); out["hp150"]=rmvpe(sg.filtfilt(b,a,y).astype(np.float32))
    b,a=sg.butter(4,4000/(SR/2),"low"); out["lp4k"]=rmvpe(sg.filtfilt(b,a,y).astype(np.float32))
    out["preemph"]=rmvpe(librosa.effects.preemphasis(y).astype(np.float32))
    return out
for vocal in sys.argv[1:]:
    song=os.path.basename(os.path.dirname(vocal)); dst=f"/tmp/tta/tracks/{song[:40]}.npz"
    if os.path.exists(dst): continue
    y,_=librosa.load(vocal,sr=SR)
    with contextlib.redirect_stdout(io.StringIO()): V=variants(y)
    os.makedirs("/tmp/tta/tracks",exist_ok=True)
    np.savez(dst, **{f"{k}__{i}":v[i] for k,v in V.items() for i in range(3)})
    print("done",song,flush=True)
