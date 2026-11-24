"""Compare per-song and mean metrics of two or more runs.

    python eval/compare.py eval_work/runs/baseline eval_work/runs/my_change [--metrics usdx_medium wer]
"""

from __future__ import annotations

import argparse
import csv
import math
import os


def load(run_dir: str) -> dict[str, dict]:
    with open(os.path.join(run_dir, "metrics.csv"), encoding="utf-8") as f:
        return {r["song"]: r for r in csv.DictReader(f)}


def num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--metrics", nargs="*")
    args = ap.parse_args()

    data = [load(r) for r in args.runs]
    names = [os.path.basename(os.path.normpath(r)) for r in args.runs]
    songs = sorted(set.intersection(*(set(d) for d in data)))
    cols = [c for c in data[0][songs[0]] if c not in ("song", "language", "error")] if songs else []
    metrics = args.metrics or [m for m in ("usdx_medium", "usdx_hard", "chroma_acc", "voicing_f1", "onset_f1_100ms",
                                           "note_count_ratio", "wer", "note_chroma_acc", "octave_errors") if m in cols]

    for m in metrics:
        print(f"\n## {m}")
        print("| song | " + " | ".join(names) + (" | delta |" if len(names) == 2 else " |"))
        print("|---" * (len(names) + 1 + (len(names) == 2)) + "|")
        means = [[] for _ in names]
        for s in songs:
            vals = [num(d[s].get(m)) for d in data]
            for i, v in enumerate(vals):
                if v is not None:
                    means[i].append(v)
            cells = [f"{v:.3f}" if v is not None else "-" for v in vals]
            if len(vals) == 2 and None not in vals:
                cells.append(f"{vals[1] - vals[0]:+.3f}")
            print(f"| {s} | " + " | ".join(cells) + " |")
        mv = [sum(x) / len(x) if x else None for x in means]
        cells = [f"**{v:.3f}**" if v is not None else "-" for v in mv]
        if len(mv) == 2 and None not in mv:
            cells.append(f"**{mv[1] - mv[0]:+.3f}**")
        print("| **mean** | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
