"""Run a preset on one song.

Used by the UltraSinger CLI (src/UltraSinger.py) and the eval harness. For development:

  python src/pipeline/run.py song.mp3 --preset presets/best.yaml --out output/ [--set step.param=value ...]
  python src/pipeline/run.py --list        # registered stages
  python src/pipeline/run.py --check presets/best.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pipeline import stages  # noqa: E402,F401  (registers the stages)
from pipeline.core import Context, describe, load_preset  # noqa: E402

PRESETS = os.path.join(os.path.dirname(SRC), "presets")
INITIAL = {"audio.mix", "media", "chart", "lyrics.text"}


def preset_path(name_or_path: str) -> str:
    """'best' -> presets/best.yaml; paths are returned unchanged."""
    if os.path.exists(name_or_path):
        return name_or_path
    candidate = os.path.join(PRESETS, name_or_path if name_or_path.endswith(".yaml") else name_or_path + ".yaml")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"preset not found: {name_or_path} (looked in {PRESETS})")


def run_pipeline(song, preset, overrides: list[str] | None = None, cache_dir: str | None = None,
                 device: str = "cpu", verbose: bool = True, cache_by: str = "lineage",
                 extra: dict | None = None, debug: bool = False, fresh: tuple[str, ...] = ()) -> dict:
    """Run a preset on a prepared song (modules.media_input.SongInput). Returns the artifact store."""
    from pipeline.log import run_logging
    pipeline = load_preset(preset_path(preset) if isinstance(preset, str) else preset, overrides)
    ctx = Context(cache_dir=cache_dir or os.path.join(song.song_dir, "cache"), output_dir=song.song_dir,
                  device=device, basename=song.basename, verbose=verbose, cache_by=cache_by,
                  extra={"fresh": set(fresh), "code_version": _git_version()})
    store = {"audio.mix": os.path.abspath(song.audio), "media": song.media}
    if song.chart:
        store["chart"] = song.chart
    store.update(extra or {})
    with run_logging(os.path.join(song.song_dir, "run_log.jsonl"), verbose, debug):
        _log_estimate(pipeline, ctx, store)
        store = pipeline.run(ctx, store)
    with open(os.path.join(song.song_dir, "lineage.json"), "w", encoding="utf-8") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "code": _git_version(),
                   "preset": pipeline.name, "overrides": list(overrides or []),
                   "steps": [st.id for st in pipeline.stages],
                   "chain": store["_lineage"].get("file.txt", [])}, f, indent=1, default=str)
    return store


def _log_estimate(pipeline, ctx, store) -> None:
    """Expected runtime from earlier runs (stages that will come from the cache are not excluded)."""
    from pipeline import runtimes
    from pipeline.log import get
    secs = runtimes.audio_seconds(store["audio.mix"])
    ctx.extra["audio_seconds"] = secs
    if not secs:
        return
    rows, total = runtimes.estimate(pipeline.stages, secs, ctx.device)
    known = [r for r in rows if r[1] is not None]
    if known:
        unknown = [sid for sid, est, _ in rows if est is None]
        get("run").info(f"{secs:.0f} s of audio: up to ~{total / 60:.1f} min without cache "
                        f"(from {sum(n for *_, n in known)} earlier stage runs"
                        + (f"; no data for {', '.join(unknown)}" if unknown else "") + ")")


def _git_version() -> str | None:
    """Commit (+ '-dirty' with local changes) of the code that made a chart, if run from git."""
    import subprocess
    try:
        root = os.path.dirname(SRC)
        rev = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (rev + ("-dirty" if dirty else "")) or None
    except (OSError, subprocess.SubprocessError):
        return None


def media_from_filename(path: str, artist: str | None = None, title: str | None = None):
    from modules.ProcessData import MediaInfo
    stem = os.path.splitext(os.path.basename(path))[0]
    if not (artist and title):
        artist, title = stem.split(" - ", 1) if " - " in stem else ("Unknown", stem)
    return MediaInfo(artist=artist, title=title, audio_extension=os.path.splitext(path)[1].lstrip(".") or "wav")


def run_song(audio_path: str, preset, overrides: list[str] | None = None, output_dir: str = "output",
             cache_dir: str | None = None, device: str = "cpu", artist: str | None = None,
             title: str | None = None, copy_audio: bool = True, verbose: bool = True,
             cache_by: str = "lineage", chart: str | None = None, extra: dict | None = None,
             fresh: tuple[str, ...] = ()) -> dict:
    """Run a preset on one audio file without any lookup (eval harness, scripts)."""
    from modules.media_input import SongInput
    media = media_from_filename(audio_path, artist, title)
    basename = f"{media.artist} - {media.title}"
    song_dir = os.path.join(output_dir, basename)
    os.makedirs(song_dir, exist_ok=True)
    if copy_audio:
        target = os.path.join(song_dir, f"{basename}.{media.audio_extension}")
        if os.path.abspath(target) != os.path.abspath(audio_path):
            shutil.copy(audio_path, target)
    song = SongInput(basename, song_dir, audio_path, media, chart=chart)
    return run_pipeline(song, preset, overrides, cache_dir, device, verbose, cache_by, extra, fresh=fresh)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--preset", default="best")
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
        problems = load_preset(preset_path(args.check), args.set).validate(INITIAL)
        print("ok" if not problems else "\n".join(problems))
        return
    store = run_song(args.audio, args.preset, args.set, args.out, args.cache, args.device)
    print(store.get("file.txt"))


if __name__ == "__main__":
    main()
