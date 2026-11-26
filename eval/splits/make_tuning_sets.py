"""Tuning sets from the SingStar train split: every train song, full length.

- tune_train: the songs of ~75% of the train artist groups (what the tuner searches on)
- tune_holdout: the other ~25% (artist-disjoint; scored once at the end of a tuning run)
- test, reserve, hard and excluded songs are never touched
- full songs (default): each entry is a link to the song folder in raw/, nothing is copied. Full songs
  are what the pipeline is for: lyrics lookup, whole-song line breaks, key and punctuation context
  only work on them, and a song's score isn't one random 30 s of it
- --windows: one random 30 s window per song (>= 40% sung) instead, written as audio + chart; for
  CPU-only machines (about 7x less audio)
- stable: group side (and window) are drawn per group / per song from a fixed seed, so adding songs
  later doesn't move anything already drawn

Songs come from eval_singstar/raw/<edition>/<song>/ (on Mo's machine a link to the collection).

    python eval/splits/make_tuning_sets.py --check     # list songs missing under raw/
    python eval/splits/make_tuning_sets.py             # eval_singstar/tune_train, tune_holdout (links)
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "eval"))
import usdx  # noqa: E402
from make_clips import write_chart  # noqa: E402

SEED, SECONDS, MIN_SUNG, LEAD_IN, HOLDOUT_SHARE = 20261126, 30.0, 0.40, 0.5, 0.25
RAW = os.path.join(ROOT, "eval_singstar", "raw")


def window_options(song):
    """Every line start whose 30 s window is at least MIN_SUNG sung: (start, note indices, share)."""
    notes, options = song.notes, []
    for i, n in enumerate(notes):
        if not n.line_start:
            continue
        start = max(0.0, n.start - LEAD_IN)
        kept = [j for j in range(i, len(notes)) if notes[j].end <= start + SECONDS]
        sung = sum(notes[j].end - notes[j].start for j in kept)
        if kept and sung / SECONDS >= MIN_SUNG:
            options.append((start, kept, sung / SECONDS))
    return options


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only list train songs missing under raw/")
    ap.add_argument("--skip-missing", action="store_true", help="skip songs not under raw/ (testing)")
    ap.add_argument("--windows", action="store_true", help="30 s windows instead of full songs")
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "singstar_split.csv"), encoding="utf-8"))
            if r["split"] == "train"]
    missing = [r for r in rows if not os.path.isdir(os.path.join(RAW, r["edition"], r["song"]))]
    if args.check or (missing and not args.skip_missing):
        print(f"{len(rows)} train songs, {len(missing)} missing under {RAW}")
        for r in missing[:20]:
            print("  missing:", r["edition"], "/", r["song"])
        sys.exit(1 if missing else 0)
    out = {side: os.path.join(ROOT, "eval_singstar", side) for side in ("tune_train", "tune_holdout")}
    for d in out.values():
        os.makedirs(d, exist_ok=True)
    counts = {side: 0 for side in out}
    for r in rows:
        if r in missing:
            continue
        side = "tune_holdout" if random.Random(f"{SEED}:group:{r['group']}").random() < HOLDOUT_SHARE else "tune_train"
        src = os.path.join(RAW, r["edition"], r["song"])
        if not args.windows:
            link = os.path.join(out[side], r["song"])
            if not os.path.lexists(link):
                os.symlink(os.path.relpath(src, out[side]), link)
            counts[side] += 1
            continue
        song = usdx.read(usdx.find_song_txt(os.path.join(RAW, r["edition"], r["song"])))
        options = window_options(song)
        if not options:
            print(f"  no window with >= {MIN_SUNG:.0%} singing: {r['song']}")
            continue
        start, kept, share = random.Random(f"{SEED}:song:{r['song']}").choice(options)
        end = min(start + SECONDS, song.notes[kept[-1]].end + 1.0)
        dst = os.path.join(out[side], f"{r['song']} @{int(start)}s")
        if not os.path.isfile(os.path.join(dst, "song.txt")):
            os.makedirs(dst, exist_ok=True)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                            "-i", song.audio_path(), "-vn", "-ac", "2", "-ar", "44100",
                            os.path.join(dst, "audio.wav")], check=True)
            write_chart(song, os.path.join(dst, "song.txt"), "audio.wav", start, kept)
            with open(os.path.join(dst, "window.json"), "w", encoding="utf-8") as f:
                json.dump({"song": r["song"], "bucket": r["bucket"], "group": r["group"], "start": round(start, 3),
                           "end": round(end, 3), "sung_share": round(share, 2)}, f)
        counts[side] += 1
    print(", ".join(f"{side}: {n} windows" for side, n in counts.items()))


if __name__ == "__main__":
    main()
