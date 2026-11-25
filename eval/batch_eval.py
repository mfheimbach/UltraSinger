"""Evaluate many preset variants in one process (libraries load once; stage caches are shared).

    python eval/batch_eval.py --songs eval_singstar/bench --work eval_work_ss --prefix gp \
        --preset bench --gt-language --config base: --config t3:segment.game.boundary_threshold=0.3 ...

Each --config is LABEL:STEP.PARAM=VAL[,STEP.PARAM=VAL...] (or -step) on top of --base. Results land in
<work>/runs/<prefix>_<label>/ like run_eval.py output (metrics.csv, summary.md), so compare.py works
on them. --jitter averages over BPM scales (tempo.scale); with GAME segmentation it barely matters.
"""

from __future__ import annotations

import argparse
import contextlib
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
from run_song import run_preset  # noqa: E402


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
    ap.add_argument("--preset", default="bench", help="pipeline preset (presets/*.yaml)")
    ap.add_argument("--gt-language", action="store_true", help="take the language from the chart")
    ap.add_argument("--cache-by", choices=["lineage", "content"], default="lineage",
                    help="lineage (default): step order is part of the cache key; content: steps whose inputs "
                         "are identical share cache entries regardless of how they were produced")
    args = ap.parse_args()
    work = os.path.abspath(args.work)

    songs = discover(args.songs, None, args.exclude)
    configs = []
    for c in args.config:
        label, _, rest = c.partition(":")
        configs.append((label, args.base + [x for x in rest.split(",") if x]))

    for label, overrides in configs:
        name = f"{args.prefix}_{label}"
        run_dir = os.path.join(work, "runs", name)
        os.makedirs(run_dir, exist_ok=True)
        config = {"name": name, "mode": "full", "overrides": overrides, "oracle_lyrics": False,
                  "preset": args.preset, "gt_language": args.gt_language}
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
                try:
                    with contextlib.redirect_stdout(log if not args.verbose else sys.stdout):
                        ov = [o for o in overrides if o != "gt_language=true"]
                        ov += [f"tempo.scale={scale}"] if scale is not None else []
                        run_preset(usdx.read(txt), out, os.path.join(work, "cache", slug(song_name)),
                                   args.preset, ov, gt_language=args.gt_language or "gt_language=true" in overrides,
                                   verbose=args.verbose, cache_by=args.cache_by)
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
