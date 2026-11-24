"""Where do the remaining errors sit? Frame-level breakdown of a run against the charts.

    python eval/error_analysis.py --songs eval_data/usdb_clips --run eval_work_clips/runs/<run> [--jitter 1.0]

For every pitched chart note, each 10 ms frame is classified as
  hit (exact pitch class), off1 (+-1 semitone), off2, off3plus, or missed (no note in the output),
and the shares are reported overall and sliced by note duration, register (relative to the
song's median chart pitch), position in the phrase, position within the note, language,
golden notes and song. Octaves are ignored throughout, as in USDX scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import usdx  # noqa: E402
from run_eval import discover, slug  # noqa: E402

HOP = 0.01
CLASSES = ["hit", "off1", "off2", "off3plus", "missed"]


def pred_grid(pred: usdx.Song, t: np.ndarray) -> np.ndarray:
    p = np.full(t.size, np.nan)
    for n in pred.notes:
        if not n.pitched:
            continue
        a, b = np.searchsorted(t, [n.start, n.end])
        p[a:b] = n.pitch
    return p


def classify(gt_pitch: int, pred: np.ndarray) -> list[str]:
    out = []
    for v in pred:
        if np.isnan(v):
            out.append("missed")
            continue
        d = abs(int(v) - gt_pitch) % 12
        d = min(d, 12 - d)
        out.append("hit" if d == 0 else "off1" if d == 1 else "off2" if d == 2 else "off3plus")
    return out


def bucket_duration(d: float) -> str:
    return "<0.15s" if d < 0.15 else "0.15-0.3s" if d < 0.3 else "0.3-0.6s" if d < 0.6 else ">=0.6s"


def bucket_register(rel: float) -> str:
    return "low (<-4)" if rel < -4 else "high (>+4)" if rel > 4 else "mid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--jitter", default=None, help="jitter subfolder to read, e.g. 1.0")
    ap.add_argument("--exclude", nargs="*", default=["Systemabsturz"])
    args = ap.parse_args()

    tables: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    dist_hist = np.zeros(7)
    missed_kind = defaultdict(int)

    for name, txt in discover(args.songs, None, args.exclude):
        sub = os.path.join(f"jitter_{args.jitter}", slug(name)) if args.jitter else slug(name)
        info_path = os.path.join(args.run, sub, "run_info.json")
        if not os.path.exists(info_path):
            continue
        gt = usdx.read(txt)
        pred = usdx.read(json.load(open(info_path))["txt"])
        t = np.arange(0, max(n.end for n in gt.notes) + 1, HOP)
        pg = pred_grid(pred, t)
        pitched = [n for n in gt.notes if n.pitched]
        med = float(np.median([n.pitch for n in pitched]))
        lang = (gt.language or "?").split()[0]

        # phrase position: first / last note of a line, else middle
        for k, n in enumerate(gt.notes):
            if not n.pitched:
                continue
            nxt_line = k + 1 < len(gt.notes) and gt.notes[k + 1].line_start
            phrase = "first in line" if n.line_start else "last in line" if (nxt_line or k + 1 == len(gt.notes)) else "middle"
            a, b = np.searchsorted(t, [n.start, n.end])
            frames = pg[a:b]
            classes = classify(n.pitch, frames)
            # position within the note: first 60 ms, last 60 ms, core
            pos = ["onset 60ms" if (a + i - a) < 6 else "end 60ms" if (b - (a + i)) <= 6 else "core"
                   for i in range(len(classes))]
            labels = {
                "all": "all",
                "duration": bucket_duration(n.end - n.start),
                "register": bucket_register(n.pitch - med),
                "phrase": phrase,
                "language": lang,
                "golden": "golden" if n.golden else "normal",
                "song": name[:32],
            }
            for c, p in zip(classes, pos):
                for dim, val in labels.items():
                    tables[dim][val][c] += 1
                tables["within note"][p][c] += 1
            voiced_idx = np.flatnonzero(~np.isnan(pg))
            for i, c in enumerate(classes):
                if c != "missed":
                    continue
                if voiced_idx.size == 0:
                    missed_kind[">500 ms (no note: missing word/phrase)"] += 1
                    continue
                dist = np.min(np.abs(voiced_idx - (a + i))) * HOP
                missed_kind["<=150 ms (timing)" if dist <= 0.15 else "150-500 ms" if dist <= 0.5
                            else ">500 ms (no note: missing word/phrase)"] += 1
            for v in frames:
                if not np.isnan(v):
                    d = abs(int(v) - n.pitch) % 12
                    dist_hist[min(d, 12 - d)] += 1

    def row(counts):
        total = sum(counts.values())
        return total, [counts.get(c, 0) / total if total else 0 for c in CLASSES]

    print(f"# Error analysis: {os.path.basename(os.path.normpath(args.run))}\n")
    for dim in ["all", "duration", "register", "phrase", "within note", "language", "golden", "song"]:
        print(f"## {dim}\n")
        print("| " + dim + " | frames | " + " | ".join(CLASSES) + " |")
        print("|---" * (len(CLASSES) + 2) + "|")
        for val, counts in sorted(tables[dim].items(), key=lambda kv: -sum(kv[1].values())):
            total, shares = row(counts)
            print(f"| {val} | {total} | " + " | ".join(f"{s:.2f}" for s in shares) + " |")
        print()
    tot = sum(missed_kind.values())
    if tot:
        print("## missed frames: how far is the nearest output note?\n")
        print("| kind | share of missed frames |")
        print("|---|---|")
        for k in ["<=150 ms (timing)", "150-500 ms", ">500 ms (no note: missing word/phrase)"]:
            print(f"| {k} | {missed_kind[k] / tot:.2f} |")
        print()
    h = dist_hist / dist_hist.sum()
    print("## pitch distance where both have a note (semitones, octave-folded)\n")
    print("| " + " | ".join(str(i) for i in range(7)) + " |")
    print("|---" * 7 + "|")
    print("| " + " | ".join(f"{x:.3f}" for x in h) + " |")


if __name__ == "__main__":
    main()
