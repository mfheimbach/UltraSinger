"""Run a pipeline preset on one ground-truth song, offline and with a persistent cache.

Called by run_eval.py in a fresh subprocess per song (keeps memory bounded) and imported by
batch_eval.py (one process, many configs).

    python eval/run_song.py --gt song.txt --out OUT --cache CACHE --preset bench [--set step.param=value ...]

Modes
  full     audio in, UltraStar txt out (the preset as given)
  repitch  the chart's syllables and timing, only the pitches detected (preset 'repitch')
Options
  --gt-language    take the language from the chart (short windows break Whisper's detection)
  --oracle-lyrics  align the chart's words instead of transcribing (preset 'oracle_lyrics')
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import usdx  # noqa: E402

LANG_ALIASES = {"englisch": "en", "deutsch": "de", "german": "de", "english": "en", "spanisch": "es",
                "französisch": "fr", "italienisch": "it"}
LANGUAGE_STEPS = ("transcribe.whisper", "transcribe.qwen3", "words.align_text")


def language_code(name: str) -> str:
    key = (name or "").strip().lower()
    if "roman" in key:
        return "en"  # romanized lyrics: Latin-alphabet aligner fits better than the native-script one
    if key in LANG_ALIASES:
        return LANG_ALIASES[key]
    import langcodes
    try:
        return langcodes.find(key).language
    except LookupError:
        return "en"


def run_preset(gt: "usdx.Song", out: str, cache: str, preset: str, overrides: list[str],
               gt_language: bool = False, verbose: bool = False, cache_by: str = "lineage",
               mode: str = "full", gt_path: str | None = None, oracle_lyrics: bool = False,
               fresh: tuple[str, ...] = ()) -> str:
    """Run a preset on the chart's audio. Returns the written txt (also recorded in run_info.json)."""
    import yaml
    from pipeline.run import preset_path, run_song

    if mode == "repitch":
        preset = "repitch"
    if oracle_lyrics:
        preset = "oracle_lyrics"
    with open(preset_path(preset), encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    steps = {s.get("id") or s["use"] for s in spec["steps"]}
    overrides = [o for o in overrides if o != "gt_language=true"]
    if gt_language or oracle_lyrics:
        code = language_code(gt.language)
        overrides += [f"{s}.language={code}" for s in LANGUAGE_STEPS if s in steps]
    extra = {"lyrics.text": " ".join(gt.words())} if oracle_lyrics else None
    store = run_song(os.path.abspath(gt.audio_path()), spec, overrides, os.path.abspath(out),
                     os.path.abspath(cache), "cpu", gt.artist, gt.title, copy_audio=False, verbose=verbose,
                     cache_by=cache_by, chart=os.path.abspath(gt_path) if mode == "repitch" else None,
                     extra=extra, fresh=fresh)
    txt = store["file.txt"]
    # artifacts for candidate analysis (eval/selection): everything but the file outputs
    import pickle
    keep = {k: v for k, v in store.items() if not k.startswith(("file.", "_"))}
    with open(os.path.join(out, "store.pkl"), "wb") as f:
        pickle.dump(keep, f)
    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"txt": os.path.abspath(txt), "mode": mode, "preset": spec.get("name"),
                   "overrides": overrides}, f, indent=2)
    return txt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "repitch"], default="full")
    ap.add_argument("--gt", required=True, help="ground-truth UltraStar txt (metadata + audio reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--preset", default="best")
    ap.add_argument("--set", action="append", default=[], metavar="STEP.PARAM=VALUE")
    ap.add_argument("--gt-language", action="store_true")
    ap.add_argument("--oracle-lyrics", action="store_true")
    ap.add_argument("--cache-by", choices=["lineage", "content"], default="lineage")
    args = ap.parse_args()
    run_preset(usdx.read(args.gt), args.out, args.cache, args.preset, args.set,
               gt_language=args.gt_language or "gt_language=true" in args.set, verbose=True,
               cache_by=args.cache_by, mode=args.mode, gt_path=args.gt, oracle_lyrics=args.oracle_lyrics)


if __name__ == "__main__":
    main()
