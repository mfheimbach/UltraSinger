"""UltraSinger-ng: create UltraStar karaoke files from songs.

  python src/UltraSinger.py -i "Artist - Title.mp3"
  python src/UltraSinger.py -i https://www.youtube.com/watch?v=...
  python src/UltraSinger.py -i chart.txt                      # re-pitch an existing chart
  python src/UltraSinger.py -i song.mp3 --preset cpu_fullsong --set transcribe.whisper.language=de
  python src/UltraSinger.py --list                            # stages and their parameters

Every processing decision lives in a preset (presets/*.yaml): an ordered list of stages with
parameters. --set changes one parameter (step.param=value) or drops a step (-step).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from version import APP_VERSION  # noqa: E402

log = logging.getLogger("ultrasinger.cli")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="UltraSinger", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--input", help="audio/video file, YouTube URL, or UltraStar txt (re-pitch)")
    ap.add_argument("-o", "--output", help="output folder (default: 'output' next to the input)")
    ap.add_argument("--preset", help="preset name or path (default: best; repitch for a txt input)")
    ap.add_argument("--set", action="append", default=[], metavar="STEP.PARAM=VALUE",
                    help="override a parameter, or '-step' to drop a step (repeatable)")
    ap.add_argument("--language", help="language code (skips detection), e.g. en, de")
    ap.add_argument("--midi", action="store_true", help="also write a MIDI file")
    ap.add_argument("--no-stems", action="store_true", help="don't write vocal/instrumental audio")
    ap.add_argument("--no-lookup", action="store_true", help="no MusicBrainz lookup: metadata from the file name")
    ap.add_argument("--cpu", action="store_true", help="never use the GPU")
    ap.add_argument("--cache", help="persistent cache folder (default: inside the song folder, removed at the end)")
    ap.add_argument("--keep-cache", action="store_true", help="keep the song's cache folder")
    ap.add_argument("--cookiefile", help="cookies.txt for YouTube")
    ap.add_argument("--ffmpeg", default="", help="path to ffmpeg if it is not on PATH")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument("--list", action="store_true", help="list stages and parameters")
    ap.add_argument("--check", action="store_true", help="validate the preset (with --set) and exit")
    ap.add_argument("--version", action="version", version=f"UltraSinger-ng {APP_VERSION}")
    return ap


def resolve_preset(args) -> dict:
    """Preset as a dict, with the CLI shortcuts applied."""
    import yaml
    from pipeline.run import preset_path
    name = args.preset or ("repitch" if (args.input or "").lower().endswith(".txt") else "best")
    with open(preset_path(name), encoding="utf-8") as f:
        preset = yaml.safe_load(f)
    steps = preset["steps"]
    uses = [s["use"] for s in steps]
    if args.language:
        for s in steps:
            if s["use"] in ("transcribe.whisper", "transcribe.qwen3", "words.align_text"):
                s.setdefault("params", {})["language"] = args.language
    if args.no_stems:
        for s in steps:
            if s["use"] == "write.ultrastar":
                s.setdefault("params", {})["stems"] = False
    if args.midi and "write.midi" not in uses:
        steps.append({"use": "write.midi"})
    return preset


def check_ffmpeg(user_path: str) -> None:
    from modules.ffmpeg_helper import get_ffmpeg_and_ffprobe_paths, is_ffmpeg_available
    if not is_ffmpeg_available(user_path):
        log.error("FFmpeg is not available: install it or pass --ffmpeg PATH")
        sys.exit(1)
    ffmpeg, ffprobe = get_ffmpeg_and_ffprobe_paths()
    log.debug(f"ffmpeg {ffmpeg}, ffprobe {ffprobe}")


def pick_device(force_cpu: bool) -> str:
    if force_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu"
    from modules.DeviceDetection.device_detection import check_gpu_support
    return check_gpu_support()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from pipeline import log as plog
    from pipeline.core import describe, load_preset
    from pipeline.run import INITIAL, run_pipeline
    plog.setup(verbose=not args.quiet)

    if args.list:
        print(describe())
        return 0
    preset = resolve_preset(args)
    try:
        pipeline = load_preset(preset, args.set)
    except (ValueError, KeyError) as err:
        log.error(str(err))
        return 1
    if args.check:
        problems = pipeline.validate(INITIAL)
        print("ok" if not problems else "\n".join(problems))
        return 0 if not problems else 1
    if not args.input:
        build_parser().print_help()
        return 1

    log.info(f"UltraSinger-ng {APP_VERSION} - preset '{preset.get('name')}'")
    check_ffmpeg(args.ffmpeg)
    device = pick_device(args.cpu)

    from modules import media_input
    output_root = args.output or os.path.join(
        os.getcwd() if args.input.startswith(("http://", "https://")) else os.path.dirname(os.path.abspath(args.input)),
        "output")
    song = media_input.prepare(args.input, output_root, lookup=not args.no_lookup, cookiefile=args.cookiefile)
    log.info(f"song folder: {song.song_dir}")

    initial = {"audio.mix", "media"} | ({"chart"} if song.chart else set())
    problems = pipeline.validate(initial)
    if problems:
        log.error("the preset needs inputs this song does not provide:\n  " + "\n  ".join(problems))
        return 1

    cache_dir = args.cache or os.path.join(song.song_dir, "cache")
    store = run_pipeline(song, preset, args.set, cache_dir, device, verbose=not args.quiet)
    if not args.cache and not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)
    log.info(f"done: {store.get('file.txt')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
