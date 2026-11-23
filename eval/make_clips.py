"""Cut each song down to its densest stretch of singing.

    python eval/make_clips.py --songs eval_data/dev --out eval_data/dev_clips [--seconds 45]

Writes one folder per song with the cropped audio (wav) and a matching
chart. Only notes fully inside the window are kept, and the window starts
at a line start, so no phrase is cut in half. The result is a normal song
folder: run_eval.py works on it unchanged.

Useful for trying large models on small machines and for quick iteration.
Clips miss song-level context (intros, BPM over the whole song, language
detection on the first 30 s), so confirm wins on full songs.
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import usdx  # noqa: E402
from run_eval import discover  # noqa: E402

DROP_TAGS = {"VIDEO", "VIDEOGAP", "START", "END", "PREVIEWSTART", "MEDLEYSTARTBEAT", "MEDLEYENDBEAT",
             "MP3", "AUDIO", "VOCALS", "INSTRUMENTAL", "GAP", "COVER", "BACKGROUND", "AUDIOURL", "VIDEOURL"}
LEAD_IN = 0.5  # seconds of audio before the first kept note


def pick_window(song: usdx.Song, seconds: float) -> tuple[float, float, list[int]]:
    """Window (start, end, kept note indices) with the most sung time."""
    notes = song.notes
    best = (-1.0, 0.0, 0.0, [])
    for i, n in enumerate(notes):
        if not n.line_start:
            continue
        start = max(0.0, n.start - LEAD_IN)
        end = start + seconds
        kept = [j for j in range(i, len(notes)) if notes[j].end <= end]
        sung = sum(notes[j].end - notes[j].start for j in kept)
        if sung > best[0]:
            best = (sung, start, end, kept)
    _, start, end, kept = best
    if kept:
        end = min(end, notes[kept[-1]].end + 1.0)
    return start, end, kept


def write_chart(src: usdx.Song, dst_txt: str, audio_name: str, clip_start: float, kept: list[int]) -> None:
    bpm = float(src.headers["BPM"].replace(",", "."))
    beat_s = 60.0 / (bpm * 4)
    gap = float(src.headers.get("GAP", "0").replace(",", ".")) / 1000.0

    # original beat of a note: (start - gap) / beat_s. New gap so times shift by -clip_start.
    new_gap = gap - clip_start
    shift = 0
    if new_gap < 0:
        shift = math.ceil(-new_gap / beat_s)
        new_gap += shift * beat_s

    lines = []
    for key, value in src.headers.items():
        if key in DROP_TAGS:
            continue
        lines.append(f"#{key}:{value.strip()}")
    lines.append(f"#MP3:{audio_name}")
    lines.append(f"#AUDIO:{audio_name}")
    lines.append(f"#GAP:{round(new_gap * 1000)}")

    prev_end_beat = None
    for k, j in enumerate(kept):
        n = src.notes[j]
        b = round((n.start - gap) / beat_s) - shift
        d = max(1, round((n.end - n.start) / beat_s))
        if k > 0 and n.line_start:
            lines.append(f"- {max(prev_end_beat, b - 1)}")
        lines.append(f"{n.kind} {b} {d} {n.pitch} {n.text}")
        prev_end_beat = b + d
    lines.append("E")
    with open(dst_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=45.0)
    args = ap.parse_args()

    for name, txt in discover(args.songs, None):
        song = usdx.read(txt)
        if song.is_duet or song.headers.get("RELATIVE", "").strip().lower() == "yes":
            print(f"skip {name}: duet/relative charts are not clipped")
            continue
        start, end, kept = pick_window(song, args.seconds)
        if not kept:
            print(f"skip {name}: no notes")
            continue
        dst = os.path.join(args.out, name)
        os.makedirs(dst, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                        "-i", song.audio_path(), "-vn", "-ac", "2", "-ar", "44100", os.path.join(dst, "audio.wav")],
                       check=True)
        write_chart(song, os.path.join(dst, "song.txt"), "audio.wav", start, kept)
        print(f"{name}: {start:.1f}-{end:.1f}s, {len(kept)} notes")


if __name__ == "__main__":
    main()
