"""Build a compact, diverse workbench of ~20 s windows from the full test songs.

Input: the window table from eval/bench_profile.py (chart/audio features + current scores per
window). Selection:
  - diversity: k-means on the first 4 principal components of the window features, one window
    per cluster (closest to the centre, preferring songs not used yet)
  - weak spots: the lowest-scoring windows, at most one per song
Outlier songs (e.g. the Japanese one) are excluded. Each window becomes a normal song folder
(audio.wav + song.txt) like eval/make_clips.py.

  python eval/make_bench.py --windows windows.csv --out eval_data/bench [--clusters 8 --weak 8]
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import usdx  # noqa: E402
from make_clips import write_chart  # noqa: E402

FEATURES = ["sung_share", "note_rate", "note_dur", "short_share", "pitch_std", "pitch_range", "pitch_mean",
            "golden_share", "words_per_s", "vocal_level", "vocal_to_backing", "vocal_centroid", "position"]


def select(d: pd.DataFrame, clusters: int, weak: int, seed: int = 0) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    z = PCA(4).fit_transform(StandardScaler().fit_transform(d[FEATURES].fillna(0)))
    km = KMeans(clusters, n_init=20, random_state=seed).fit(z)
    chosen, used = [], set()
    for c in range(clusters):
        idx = np.where(km.labels_ == c)[0]
        dist = np.linalg.norm(z[idx] - km.cluster_centers_[c], axis=1)
        order = idx[np.argsort(dist)]
        pick = next((i for i in order if d.song.iloc[i] not in used), order[0])
        chosen.append((pick, f"cluster {c}"))
        used.add(d.song.iloc[pick])
    weak_used = set()
    for i in np.argsort(d.usdx_medium.values):
        if len([c for c in chosen if c[1] == "weak"]) >= weak:
            break
        if d.song.iloc[i] in weak_used or i in [c[0] for c in chosen]:
            continue
        chosen.append((i, "weak"))
        weak_used.add(d.song.iloc[i])
    out = d.iloc[[i for i, _ in chosen]].copy()
    out["reason"] = [r for _, r in chosen]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--songs", nargs="+", default=[os.path.join(HERE, "..", "eval_data", s) for s in ("usdb", "dev")])
    ap.add_argument("--out", required=True)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--weak", type=int, default=8)
    ap.add_argument("--exclude", nargs="*", default=["Kouji Wada", "Systemabsturz"])
    args = ap.parse_args()

    d = pd.read_csv(args.windows)
    d = d[~d.song.apply(lambda s: any(x in s for x in args.exclude))].reset_index(drop=True)
    sel = select(d, args.clusters, args.weak)
    os.makedirs(args.out, exist_ok=True)
    for _, w in sel.iterrows():
        src_dir = next(os.path.join(s, w.song) for s in args.songs if os.path.isdir(os.path.join(s, w.song)))
        song = usdx.read(usdx.find_song_txt(src_dir))
        kept = [j for j, n in enumerate(song.notes) if n.start >= w.start and n.end <= w.end]
        name = f"{w.song} @{int(w.start)}s"
        dst = os.path.join(args.out, name)
        os.makedirs(dst, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{w.start:.3f}", "-t", f"{w.end - w.start:.3f}",
                        "-i", song.audio_path(), "-vn", "-ac", "2", "-ar", "44100", os.path.join(dst, "audio.wav")],
                       check=True)
        write_chart(song, os.path.join(dst, "song.txt"), "audio.wav", w.start, kept)
        with open(os.path.join(dst, "window.json"), "w") as f:
            json.dump({"song": w.song, "start": w.start, "end": w.end, "reason": w.reason,
                       "usdx_medium_at_selection": round(float(w.usdx_medium), 3)}, f)
        print(f"{name:55s} {w.reason:10s} {w.end - w.start:4.1f}s  score then {w.usdx_medium:.2f}")


if __name__ == "__main__":
    main()
