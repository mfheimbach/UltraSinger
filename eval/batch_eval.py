"""Evaluate many configurations in one process: fast sweeps of late pipeline stages.

    python eval/batch_eval.py --songs eval_data/usdb_clips --work eval_work_clips --prefix lq_usdb \\
        --base demucs_model=htdemucs_ft whisper_model=large-v3-turbo \\
        --config t0: --config t25:key_quantize_min_offset=0.25 ...

Each --config is LABEL:KEY=VAL[,KEY=VAL...] on top of --base. Results land in
<work>/runs/<prefix>_<label>/ exactly like run_eval.py output (metrics.csv, summary.md),
so compare.py works on them.

Faster than run_eval.py because libraries load once, and BPM / key detection are
memoized per audio file. Expensive stages (separation, transcription, pitch) come
from UltraSinger's caches, so run each upstream configuration once with run_eval.py
first; this tool is for sweeping what comes after.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import usdx  # noqa: E402
from run_eval import discover, score_one, slug, write_outputs  # noqa: E402
from run_song import run_configured  # noqa: E402


def memoize_by_file(fn):
    cache = {}

    @functools.wraps(fn)
    def wrapper(path, *a, **k):
        key = (os.path.abspath(path), os.path.getmtime(path), a, tuple(sorted(k.items())))
        if key not in cache:
            cache[key] = fn(path, *a, **k)
        return cache[key]
    return wrapper


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", required=True)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "eval_work"))
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--base", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--config", action="append", required=True, metavar="LABEL:KEY=VAL,...")
    ap.add_argument("--exclude", nargs="*")
    ap.add_argument("--jitter", nargs="*", type=float, default=None,
                    help="BPM scales to average over, e.g. 0.99 1 1.01. UltraSinger's note segmentation "
                         "is very sensitive to tiny BPM changes; averaging removes that noise from comparisons")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    work = os.path.abspath(args.work)

    import UltraSinger as us
    us.get_bpm_from_file = memoize_by_file(us.get_bpm_from_file)
    us.detect_key_from_audio = memoize_by_file(us.detect_key_from_audio)
    import modules.Audio.bpm as bpm_mod
    bpm_mod.get_bpm_with_beat_this = memoize_by_file(bpm_mod.get_bpm_with_beat_this)

    songs = discover(args.songs, None, args.exclude)
    configs = []
    for c in args.config:
        label, _, rest = c.partition(":")
        configs.append((label, args.base + [x for x in rest.split(",") if x]))

    for label, overrides in configs:
        name = f"{args.prefix}_{label}"
        run_dir = os.path.join(work, "runs", name)
        os.makedirs(run_dir, exist_ok=True)
        config = {"name": name, "mode": "full", "overrides": overrides, "oracle_lyrics": False}
        json.dump(config, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
        t0 = time.time()
        rows = []
        scales = args.jitter or [None]
        for song_name, txt in songs:
            per_scale = []
            for scale in scales:
                sub = slug(song_name) if scale is None else os.path.join(f"jitter_{scale}", slug(song_name))
                out = os.path.join(run_dir, sub)
                shutil.rmtree(out, ignore_errors=True)
                os.makedirs(out)
                log = io.StringIO()
                ov = overrides + ([f"bpm_scale={scale}"] if scale is not None else [])
                try:
                    with contextlib.redirect_stdout(log if not args.verbose else sys.stdout):
                        run_configured(us, usdx.read(txt), txt, "full", out,
                                       os.path.join(work, "cache", slug(song_name)), ov)
                    per_scale.append(score_one("full", song_name, txt, os.path.dirname(out)))
                except Exception as e:  # keep going
                    per_scale.append({"song": song_name, "error": repr(e)})
                open(os.path.join(out, "run.log"), "w", encoding="utf-8").write(log.getvalue())
            good = [r for r in per_scale if "error" not in r]
            if not good:
                rows.append(per_scale[0])
                continue
            avg = {"song": song_name, "language": good[0].get("language", "")}
            for k in good[0]:
                vals = [r[k] for r in good if isinstance(r.get(k), (int, float))]
                if vals and k not in avg:
                    avg[k] = sum(vals) / len(vals)
            rows.append(avg)
        with contextlib.redirect_stdout(io.StringIO()):
            write_outputs(run_dir, "full", rows, config)
        ok = [r for r in rows if "error" not in r]
        mean = lambda k: sum(r[k] for r in ok if r.get(k) == r.get(k)) / max(1, len(ok))
        print(f"{name}: {time.time() - t0:.0f}s  usdx_medium {mean('usdx_medium'):.3f}  "
              f"usdx_hard {mean('usdx_hard'):.3f}  pitch_mae {mean('pitch_mae'):.3f}  "
              f"errors {len(rows) - len(ok)}", flush=True)


if __name__ == "__main__":
    main()
