import sys,os,glob; sys.path.insert(0,'/home/claude/ultrasinger/eval'); import numpy as np, pandas as pd, usdx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
d=pd.read_csv("/tmp/bench/windows.csv")
F=["sung_share","note_rate","note_dur","short_share","pitch_std","pitch_range","pitch_mean","golden_share","words_per_s","vocal_level","vocal_to_backing","vocal_centroid","position"]
X=StandardScaler().fit_transform(d[F].fillna(0)); pca=PCA().fit(X); Z=pca.transform(X)
print("explained var:", pca.explained_variance_ratio_[:6].round(2), "cum", pca.explained_variance_ratio_[:6].cumsum().round(2))
for i in range(4):
    load=sorted(zip(F,pca.components_[i]),key=lambda x:-abs(x[1]))[:4]
    print(f"PC{i+1}:", ", ".join(f"{f} {w:+.2f}" for f,w in load))
# where are the current clips? mark windows overlapping the clip ranges
U="/home/claude/ultrasinger"; d["in_clip"]=False
sys.path.insert(0,U+"/eval"); import make_clips
for data in ["usdb","dev"]:
    for song in os.listdir(f"{U}/eval_data/{data}"):
        gp=usdx.find_song_txt(f"{U}/eval_data/{data}/{song}")
        if not gp: continue
        a,b,_=make_clips.pick_window(usdx.read(gp),45.0)
        m=(d.song==song)&(d.end>a+5)&(d.start<b-5); d.loc[m,"in_clip"]=True
print("windows inside current clips:", d.in_clip.sum(), "of", len(d))
for i in range(3):
    print(f"PC{i+1} mean: clips {Z[d.in_clip,i].mean():+.2f}  rest {Z[~d.in_clip,i].mean():+.2f}")
print("features, clips vs rest:"); print(d.groupby("in_clip")[F+["usdx_medium","usdx_hard","missed"]].mean().T.round(2))
# PCR: what drives low scores? (cross-validated by song)
y=d.usdx_medium.values; g=d.song.values
for k in [2,4,6]:
    p=cross_val_predict(LinearRegression(),Z[:,:k],y,groups=g,cv=GroupKFold(5))
    print(f"PCR k={k}: CV R2 {1-np.sum((y-p)**2)/np.sum((y-y.mean())**2):.2f}")
sv=d.groupby("song").usdx_medium.var().mean(); tv=d.usdx_medium.var()
print(f"usdx_medium variance: total {tv:.4f}, within songs {sv:.4f} ({sv/tv:.0%}), between songs {1-sv/tv:.0%}")
print(d.groupby("song").usdx_medium.agg(["mean","min","max","count"]).round(2).rename(index=lambda x:x[:22]).to_string())
lr=LinearRegression().fit(StandardScaler().fit_transform(d[F].fillna(0)),y)
print("standardized coefs on usdx_medium:", ", ".join(f"{f} {c:+.3f}" for f,c in sorted(zip(F,lr.coef_),key=lambda x:x[1])))
d[[f"PC{i+1}" for i in range(4)]]=Z[:,:4]; d.to_csv("/tmp/bench/windows_pca.csv",index=False)
