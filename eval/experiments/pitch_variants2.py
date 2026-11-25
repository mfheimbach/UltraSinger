"""Variants that add something not already in the signal: stereo channels, another separator,
nonlinear distortion, time-stretch, harmonic part (HPSS)."""
import sys,os,glob,contextlib,io; sys.path.insert(0,'/home/claude/ultrasinger/src')
import numpy as np, librosa
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from modules.Pitcher import alt_pitchers as ap
sess=ort.InferenceSession(hf_hub_download(ap.RMVPE_REPO, ap.RMVPE_FILE), providers=["CPUExecutionProvider"])
SR=16000
def rmvpe(y):
    f0,conf=ap._rmvpe_pass(sess,y.astype(np.float32)); t=np.arange(len(f0))*0.01
    return t, np.where(f0>0, librosa.hz_to_midi(np.maximum(f0,1e-3)), np.nan), conf
for vocal in sys.argv[1:]:
    song=os.path.basename(os.path.dirname(vocal)); dst=f"/tmp/tta/tracks2/{song[:40]}.npz"
    if os.path.exists(dst): continue
    st,_=librosa.load(vocal,sr=SR,mono=False); y=st.mean(0)
    V={}
    with contextlib.redirect_stdout(io.StringIO()):
        V["left"]=rmvpe(st[0]); V["right"]=rmvpe(st[1])
        side=(st[0]-st[1])/2
        if np.sqrt(np.mean(side**2))>1e-4: V["side"]=rmvpe(side)
        alt=vocal.replace("demucs_htdemucs_ft/separated/htdemucs_ft","separated/htdemucs")
        if os.path.exists(alt): V["htdemucs"]=rmvpe(librosa.load(alt,sr=SR)[0])
        peak=np.max(np.abs(y))+1e-9
        V["tanh5"]=rmvpe(np.tanh(5*y/peak))
        V["clip30"]=rmvpe(np.clip(y,-0.3*peak,0.3*peak))
        for r in [0.9,1.1]:  # time-stretch by r (faster for r>1): times back by *r
            t,m,c=rmvpe(librosa.effects.time_stretch(y,rate=r)); V[f"ts{r}"]=(t*r,m,c)
        V["harm"]=rmvpe(librosa.effects.harmonic(y,margin=2))
        V["orig"]=rmvpe(y)
    os.makedirs("/tmp/tta/tracks2",exist_ok=True)
    np.savez(dst, **{f"{k}__{i}":v[i] for k,v in V.items() for i in range(3)})
    print("done",song,sorted(V),flush=True)
