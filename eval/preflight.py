"""Check everything a long tuning run needs before it starts, in one go.

    python eval/preflight.py tune/staged_singstar.yaml [--device auto] [--no-smoke]

Prints one line per check (OK / FAIL / INFO) and exits non-zero if anything fails. The smoke test
runs the base pipeline once on one training song (downloads models on first use).
"""

import argparse
import importlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok is not False)
    tag = "OK  " if ok is True else ("INFO" if ok is None else "FAIL")
    print(f"[{tag}] {name}" + (f": {detail}" if detail else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--work", default="eval_work_ng")
    ap.add_argument("--no-smoke", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)

    # packages
    for mod in ("torch", "whisperx", "demucs", "onnxruntime", "onnx", "optuna", "mir_eval", "jiwer", "librosa",
                "pretty_midi", "transformers", "hyphen", "Levenshtein", "scipy", "pydub", "soundfile", "yaml"):
        try:
            importlib.import_module(mod)
            check(f"import {mod}", True)
        except Exception as e:  # noqa: BLE001
            check(f"import {mod}", False, f"{type(e).__name__}: {e}")
    check("ffmpeg on PATH", shutil.which("ffmpeg") is not None)

    # device
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda":
        ok = torch.cuda.is_available()
        check("CUDA", ok, torch.cuda.get_device_name(0) + f", {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB"
              if ok else "torch sees no GPU")
        try:
            import onnxruntime as ort
            check("onnxruntime CUDA provider", "CUDAExecutionProvider" in ort.get_available_providers() or None,
                  "missing: GAME/RMVPE run on CPU (works, slower; pip install onnxruntime-gpu)"
                  if "CUDAExecutionProvider" not in ort.get_available_providers() else "")
        except Exception:  # noqa: BLE001
            pass
    else:
        check("device", None, "cpu (a full staged run takes ~8-9 h on a laptop CPU)")

    # spec + search space
    import yaml
    import tune
    from pipeline.core import load_preset, resolve_preset
    from pipeline.run import preset_path
    spec = yaml.safe_load(open(args.spec, encoding="utf-8"))
    try:
        tune.validate_spec(spec)
        check("spec complete (incl. holdout)", True)
    except SystemExit as e:
        check("spec complete", False, str(e))
        return finish()
    bad = []
    groups = spec.get("stages") or [{"name": "all", "search": spec["search"]}]
    for g in groups:
        for d in tune.parse_search(g["search"]):
            for v in (d.choices if d.kind == "choice" else [d.low, d.high]):
                ov = list(spec.get("overrides") or []) + ([f"{d.parent[0]}={str(d.parent[1]).lower()}"]
                                                          if d.parent else []) + [d.override(v)]
                try:
                    load_preset(preset_path(spec["base"]), ov).validate({"audio.mix", "media"})
                except Exception as e:  # noqa: BLE001
                    bad.append(f"{d.key}={v}: {str(e)[:120]}")
    check(f"every search value builds a valid pipeline ({len(groups)} groups)", not bad, "; ".join(bad[:3]))

    # data
    from run_eval import discover
    for role in ("train", "holdout"):
        path = spec["data"][role]
        songs = discover(path, None) if os.path.isdir(path) else []
        missing_audio, bad_lang, langs = [], [], {}
        import usdx
        from run_song import language_code
        for name, txt in songs:
            song = usdx.read(txt)
            a = song.audio_path()
            if not a or not os.path.isfile(a):
                missing_audio.append(name)
            if spec["data"]["gt_language"]:
                try:
                    code = language_code(song.language)
                    langs[code] = langs.get(code, 0) + 1
                except ValueError as e:
                    bad_lang.append(f"{name}: {e}")
        check(f"data {role} ({path})", bool(songs) and not missing_audio and not bad_lang,
              f"{len(songs)} songs, languages {dict(sorted(langs.items()))}"
              + (f", audio missing: {missing_audio[:3]}" if missing_audio else "")
              + (f", language problems: {bad_lang[:3]}" if bad_lang else ""))

    # models and disk
    params = resolve_preset(preset_path("best"))["params"]
    game = os.path.expanduser(params["segment.game"]["model_dir"])
    check("GAME model", os.path.isfile(os.path.join(game, "config.json")),
          game + ("" if os.path.isfile(os.path.join(game, "config.json")) else
                  " missing: download GAME-1.0.3-small-onnx from github.com/openvpi/GAME releases and unpack there"))
    os.makedirs(args.work, exist_ok=True)
    free = shutil.disk_usage(args.work).free / 1e9
    check("disk space in work dir", free > 20, f"{free:.0f} GB free (cache grows to ~10-15 GB)")
    check("models downloaded on first use", None, "Whisper large-v3-turbo, wav2vec2 aligners, demucs, RMVPE, "
          "punctuation model (~5 GB total, Hugging Face / torch hub)")

    # smoke: one song through the base pipeline
    if not args.no_smoke and all(RESULTS):
        tune.DEVICE = device
        try:
            songs = discover(spec["data"]["train"], None)
            ev = tune.Evaluator(spec, args.work, songs[:1])
            ev.base_overrides = list(spec.get("overrides") or [])
            name = songs[0][0]
            res = ev.score(name, (), [], tune._write_index(ev, name))
            check(f"smoke: base pipeline on '{name}'", True, ", ".join(f"{k} {v:.3f}" for k, v in res.items()
                                                                       if isinstance(v, float)))
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("smoke: base pipeline", False, f"{type(e).__name__}: {e}")
    return finish()


def finish():
    ok = all(RESULTS)
    print("\nall checks passed: start the run" if ok else "\nfix the FAIL lines first")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
