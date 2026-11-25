"""Run UltraSinger on one song, offline and with a persistent cache.

Called by run_eval.py in a fresh subprocess per song, because UltraSinger
keeps its settings in module-level state.

    python eval/run_song.py --mode full --gt song.txt --out OUT --cache CACHE [--set key=value ...]

Modes
  full     audio in, UltraStar txt out (the normal pipeline)
  repitch  ground-truth txt in, only the pitches are re-detected
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from enum import Enum

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import usdx  # noqa: E402


def _coerce(current, raw: str):
    if isinstance(current, Enum):
        return type(current)(raw)
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if raw.lower() in ("none", "null"):
        return None
    return raw


def apply_overrides(settings, overrides: list[str]) -> None:
    for item in overrides:
        key, raw = item.split("=", 1)
        if not hasattr(settings, key):
            raise SystemExit(f"Unknown setting: {key}")
        setattr(settings, key, _coerce(getattr(settings, key), raw))


LANG_ALIASES = {"englisch": "en", "deutsch": "de", "german": "de", "english": "en", "spanisch": "es",
                "französisch": "fr", "italienisch": "it"}


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


def share_separation_cache(cache: str, gt: usdx.Song) -> None:
    """Demucs output is cached under the input file's basename, which differs
    between modes (full copies the audio as 'Artist - Title.ext', repitch reads
    it in place). Alias the two so separation runs once per song."""
    names = {os.path.splitext(os.path.basename(gt.audio_path()))[0], f"{gt.artist} - {gt.title}"}
    root = os.path.join(cache, "separated")
    if not os.path.isdir(root):
        return
    for model in os.listdir(root):
        mdir = os.path.join(root, model)
        existing = [n for n in names if os.path.isdir(os.path.join(mdir, n))]
        if not existing:
            continue
        for n in names - set(existing):
            os.symlink(existing[0], os.path.join(mdir, n))


def run_configured(us, gt: "usdx.Song", gt_path: str, mode: str, out: str, cache: str,
                   overrides: list[str], oracle_lyrics: bool = False) -> str:
    """Configure UltraSinger's module-level settings for one song and run it.
    Resets settings first, so it can be called repeatedly in one process."""
    from modules.musicbrainz_client import SongInfo
    from modules.Ultrastar.ultrastar_txt import FormatVersion

    # Offline + no side quests: no MusicBrainz lookups, no sheet music, no support banner
    us.search_musicbrainz = lambda title, artist: SongInfo(title=gt.title, artist=gt.artist)
    us.create_sheet = lambda *a, **k: None
    us.print_support = lambda *a, **k: None

    s = us.settings
    s.__dict__.clear()  # back to the class defaults
    s.output_folder_path = os.path.abspath(out)
    s.cache_override_path = os.path.abspath(cache)
    s.keep_cache = True
    s.create_midi = False
    s.create_plot = False
    s.calculate_score = False
    s.interactive_mode = False

    if mode == "full":
        # language detection is part of what we evaluate; pin it with --set language=xx
        s.input_file_path = os.path.abspath(gt.audio_path())
    else:
        s.input_file_path = os.path.abspath(gt_path)
        # re-pitch never needs instrumental/vocal stems written next to the output
        s.format_version = FormatVersion.V1_0_0
        s.create_karaoke = False

    if oracle_lyrics:
        lyrics_path = os.path.join(os.path.abspath(cache), "oracle_lyrics.txt")
        os.makedirs(os.path.dirname(lyrics_path), exist_ok=True)
        with open(lyrics_path, "w", encoding="utf-8") as f:
            f.write(" ".join(gt.words()))
        s.lyrics_file = lyrics_path
        s.language = language_code(gt.language)

    # pseudo-setting: take the language from the chart (short windows make Whisper's language
    # detection unreliable; on full songs the pipeline detects it itself)
    if "gt_language=true" in overrides:
        overrides = [o for o in overrides if o != "gt_language=true"]
        s.language = language_code(gt.language)
    apply_overrides(s, overrides)
    # Intermediate audio (denoised/mono/muted) and the transcription cache are not keyed
    # by separation model, so non-default models get their own cache folder.
    if s.separator_model:
        tag = os.path.splitext(s.separator_model)[0] + (f"+{os.path.splitext(s.lead_vocal_model)[0]}" if s.lead_vocal_model else "")
        s.cache_override_path = os.path.join(s.cache_override_path, f"sep_{tag}")
    elif s.demucs_model.value != "htdemucs" or s.lead_vocal_model:
        tag = s.demucs_model.value + (f"+{os.path.splitext(s.lead_vocal_model)[0]}" if s.lead_vocal_model else "")
        s.cache_override_path = os.path.join(s.cache_override_path, f"demucs_{tag}")
    os.makedirs(s.output_folder_path, exist_ok=True)
    os.makedirs(s.cache_override_path, exist_ok=True)
    share_separation_cache(s.cache_override_path, gt)

    us.check_requirements()
    txt_path, _simple, _accurate = us.run()

    # the eval only needs the txt; drop audio/video copies UltraSinger writes next to it
    for root, _dirs, files in os.walk(out):
        for name in files:
            if name.lower().endswith((".ogg", ".m4a", ".mp3", ".wav", ".mp4", ".flac", ".jpg", ".png")):
                os.remove(os.path.join(root, name))

    with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump({"txt": os.path.abspath(txt_path), "mode": mode, "overrides": overrides}, f, indent=2)
    return txt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "repitch"], default="full")
    ap.add_argument("--gt", required=True, help="ground-truth UltraStar txt (metadata + audio reference)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--oracle-lyrics", action="store_true",
                    help="align the chart's own lyrics (no timing) instead of transcribing: upper bound for lyrics lookup")
    args = ap.parse_args()

    import UltraSinger as us
    run_configured(us, usdx.read(args.gt), args.gt, args.mode, args.out, args.cache, args.set, args.oracle_lyrics)


if __name__ == "__main__":
    main()
