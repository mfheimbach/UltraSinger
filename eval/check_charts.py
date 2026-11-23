"""Sanity-check ground-truth charts against their audio, independent of UltraSinger.

    python eval/check_charts.py --songs eval_data/usdb [--work eval_work]

Timing: cross-correlates the chart's note activity with the energy envelope of
the separated vocals (from the eval cache, so run the songs once first). A best
lag far from 0 means the chart's GAP doesn't fit this audio version.
Pitch: use a repitch run's `transpose` column.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import usdx  # noqa: E402
from run_eval import discover, slug  # noqa: E402

HOP = 0.01


def vocal_activity(wav_path: str) -> np.ndarray:
    import librosa
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    hop = int(sr * HOP)
    rms = librosa.feature.rms(y=y, frame_length=hop * 4, hop_length=hop)[0]
    db = 20 * np.log10(rms + 1e-9)
    return (db > np.percentile(db, 60)).astype(float)  # loudest 40 % of frames ~ singing


def chart_activity(song: usdx.Song, n: int) -> np.ndarray:
    act = np.zeros(n)
    for note in song.notes:
        a, b = int(note.start / HOP), int(note.end / HOP)
        act[max(a, 0):min(b, n)] = 1.0
    return act


def best_lag(chart: np.ndarray, vocal: np.ndarray, max_s: float = 2.0) -> tuple[float, float]:
    """Lag (s) to add to the chart to best match the vocals, and the overlap gain vs lag 0."""
    k = int(max_s / HOP)
    c = chart - chart.mean()
    v = vocal - vocal.mean()
    scores = {lag: float(np.dot(np.roll(c, lag), v)) for lag in range(-k, k + 1, 2)}
    lag = max(scores, key=scores.get)
    gain = scores[lag] / scores[0] if scores[0] > 0 else float("inf")
    return lag * HOP, gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "eval_work"))
    args = ap.parse_args()

    print("| song | chart lag | overlap gain | verdict |")
    print("|---|---|---|---|")
    for name, txt in discover(args.songs, None):
        vocals = glob.glob(os.path.join(args.work, "cache", slug(name), "separated", "*", "*", "vocals.wav"))
        if not vocals:
            print(f"| {name} | - | - | no cached vocals, run it once |")
            continue
        vocal = vocal_activity(vocals[0])
        chart = chart_activity(usdx.read(txt), len(vocal))
        lag, gain = best_lag(chart, vocal)
        verdict = "ok" if abs(lag) <= 0.15 else ("check GAP" if gain > 1.15 else "unclear")
        print(f"| {name} | {lag * 1000:+.0f} ms | {gain:.2f} | {verdict} |")


if __name__ == "__main__":
    main()
