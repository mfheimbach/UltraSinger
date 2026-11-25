"""Run a preset on one song.

  python src/pipeline/run.py song.mp3 --preset presets/best.yaml --out output/ [--set step.param=value ...]
  python src/pipeline/run.py --list        # registered stages
  python src/pipeline/run.py --check presets/best.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline import stages  # noqa: E402,F401  (registers the stages)
from pipeline.core import Context, describe, load_preset  # noqa: E402

INITIAL = {"audio.mix", "media"}


def media_from_filename(path: str, artist: str | None = None, title: str | None = None):
    from modules.ProcessData import MediaInfo
    stem = os.path.splitext(os.path.basename(path))[0]
    if not (artist and title):
        artist, title = stem.split(" - ", 1) if " - " in stem else ("Unknown", stem)
    return MediaInfo(artist=artist, title=title, audio_extension=os.path.splitext(path)[1].lstrip(".") or "wav")


def run_song(audio_path: str, preset, overrides: list[str] | None = None, output_dir: str = "output",
             cache_dir: str | None = None, device: str = "cpu", artist: str | None = None,
             title: str | None = None, copy_audio: bool = True, verbose: bool = True,
             cache_by: str = "lineage") -> dict:
    """Run a preset on one audio file. Returns the artifact store (``file.txt`` = the chart)."""
    pipeline = load_preset(preset, overrides)
    media = media_from_filename(audio_path, artist, title)
    basename = f"{media.artist} - {media.title}"
    song_dir = os.path.join(output_dir, basename)
    os.makedirs(song_dir, exist_ok=True)
    if copy_audio:
        target = os.path.join(song_dir, f"{basename}.{media.audio_extension}")
        if os.path.abspath(target) != os.path.abspath(audio_path):
            shutil.copy(audio_path, target)
    ctx = Context(cache_dir=cache_dir or os.path.join(song_dir, "cache"), output_dir=song_dir, device=device,
                  basename=basename, verbose=verbose, cache_by=cache_by)
    store = {"audio.mix": os.path.abspath(audio_path), "media": media}
    store = pipeline.run(ctx, store)
    # provenance of the final chart: step order + settings, next to the .txt
    import json
    with open(os.path.join(song_dir, "lineage.json"), "w", encoding="utf-8") as f:
        json.dump({"preset": pipeline.name, "overrides": list(overrides or []),
                   "chain": store["_lineage"].get("file.txt", [])}, f, indent=1, default=str)
    return store


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--preset", default=os.path.join(os.path.dirname(SRC), "presets", "best.yaml"))
    ap.add_argument("--set", action="append", default=[], metavar="STEP.PARAM=VALUE")
    ap.add_argument("--out", default="output")
    ap.add_argument("--cache")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", metavar="PRESET")
    args = ap.parse_args(argv)
    if args.list:
        print(describe())
        return
    if args.check:
        problems = load_preset(args.check, args.set).validate(INITIAL)
        print("ok" if not problems else "\n".join(problems))
        return
    store = run_song(args.audio, args.preset, args.set, args.out, args.cache, args.device)
    print(store.get("file.txt"))


if __name__ == "__main__":
    main()
