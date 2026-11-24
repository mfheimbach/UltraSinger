"""Fit the line-break model in src/modules/Ultrastar/line_breaks.py.

Uses finished eval runs (our predicted charts) and the ground-truth charts: every pause
between two predicted words is a sample, labelled 1 if a chart line starts there (nearest
predicted word within 0.5 s). Features: log pause, logit of the punctuation model.
Reports leave-one-song-out line F1 of the full planner and the final coefficients.

  python eval/fit_line_breaks.py --run eval_work/runs/<run> eval_data/<set> [--run ...]
"""

import argparse
import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

import usdx  # noqa: E402
from metrics import line_metrics  # noqa: E402
from modules.Midi.MidiSegment import MidiSegment  # noqa: E402
from modules.Ultrastar import line_breaks as lb  # noqa: E402


def segments_from_chart(song: usdx.Song) -> list[MidiSegment]:
    return [MidiSegment(note="C4", start=n.start, end=n.end, word=n.text) for n in song.notes]


def find_pred(run_dir: str, song_dir: str) -> str | None:
    name = os.path.basename(song_dir)
    key = lambda s: "".join(c for c in s if c.isalnum())[:12]
    for path in glob.glob(os.path.join(run_dir, "**", "*.txt"), recursive=True):
        if "jitter_0" in path or "jitter_1.01" in path:
            continue
        if key(os.path.basename(path)) == key(name):
            return path
    return None


def collect(runs):
    songs = []
    for run_dir, data_dir in runs:
        for name in sorted(os.listdir(data_dir)):
            gt_path = usdx.find_song_txt(os.path.join(data_dir, name))
            pred = find_pred(run_dir, os.path.join(data_dir, name))
            if gt_path and pred:
                songs.append((name, usdx.read(gt_path), usdx.read(pred)))
    return songs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("RUN_DIR", "DATA_DIR"), required=True)
    ap.add_argument("--exclude", nargs="*", default=["Systemabsturz"])
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression

    songs = [s for s in collect(args.run) if not any(x in s[0] for x in args.exclude)]
    data = {}
    for name, gt, pred in songs:
        segs = segments_from_chart(pred)
        words = lb._words(segs)
        p = np.clip(lb.punctuation_probabilities([w[4] for w in words], lb_model), 1e-3, 1 - 1e-3)
        x, y = [], np.zeros(len(words) - 1)
        for i in range(1, len(words)):
            pause = max(0.0, words[i][2] - words[i - 1][3])
            x.append([math.log(pause + 0.02), math.log(p[i - 1] / (1 - p[i - 1]))])
        starts = np.array([w[2] for w in words[1:]])
        for t in [n.start for n in gt.notes[1:] if n.line_start]:
            k = int(np.argmin(np.abs(starts - t)))
            if abs(starts[k] - t) < 0.5:
                y[k] = 1
        data[name] = (gt, pred, np.array(x), y)

    f1s = []
    for name in data:
        X = np.vstack([data[s][2] for s in data if s != name])
        Y = np.concatenate([data[s][3] for s in data if s != name])
        m = LogisticRegression(max_iter=1000).fit(X, Y)
        lb.INTERCEPT, lb.W_PAUSE, lb.W_PUNCT = m.intercept_[0], *m.coef_[0]
        gt, pred, _, _ = data[name]
        segs = segments_from_chart(pred)
        lb.punctuation_probabilities = cached(data[name][2])
        lb.plan_line_breaks(segs, lb_model)
        lb.punctuation_probabilities = original_pp
        planned = usdx.Song(path="", notes=[usdx.Note(kind=":", start=s.start, end=s.end, pitch=0, text=s.word,
                                                       line_start=s.line_start) for s in segs])
        f = line_metrics(gt, planned).get("line_f1", 0.0)
        orig = line_metrics(gt, pred).get("line_f1", 0.0)
        f1s.append((f, orig))
        print(f"{name[:30]:30s} planned {f:.2f}  as written {orig:.2f}")
    print(f"LOSO line F1: planned {np.mean([a for a, _ in f1s]):.3f}  as written {np.mean([b for _, b in f1s]):.3f}")
    X = np.vstack([v[2] for v in data.values()])
    Y = np.concatenate([v[3] for v in data.values()])
    m = LogisticRegression(max_iter=1000).fit(X, Y)
    print(f"INTERCEPT, W_PAUSE, W_PUNCT = {m.intercept_[0]:.3f}, {m.coef_[0][0]:.3f}, {m.coef_[0][1]:.3f}")


lb_model = "oliverguhr/fullstop-punctuation-multilingual-base"
original_pp = lb.punctuation_probabilities


def cached(x):
    """Replay the punctuation probabilities already computed for a song (logit stored in x[:, 1])."""
    p = 1 / (1 + np.exp(-x[:, 1]))
    return lambda words, *a, **k: np.append(p, 0.5)


if __name__ == "__main__":
    main()
