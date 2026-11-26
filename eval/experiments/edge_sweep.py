"""Why do tuned values sit at the edge of their search range? Sweep one parameter at a time past the
range, everything else at the tuned config, and show every metric (not just the objective).

    python eval/experiments/edge_sweep.py tune/staged_singstar.yaml --start tune/out/staged_singstar/preset.yaml
"""

import argparse
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tune  # noqa: E402
from run_eval import discover  # noqa: E402

SWEEPS = {
    "notes.calibrate.end_extension": [-0.1, -0.05, 0.0, 0.03, 0.06, 0.1, 0.2, 0.3],
    "notes.from_pitch.key_min_offset": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "notes.snap_edges.conf": [0.0, 0.03, 0.05, 0.1, 0.2, 0.4],
    "notes.calibrate.offset": [-0.15, -0.1, -0.07, -0.04, 0.0],
}
SHOW = ["usdx_medium", "usdx_medium_precision", "usdx_hard", "voicing_recall", "voicing_precision", "voicing_f1",
        "start_score", "end_score", "line_f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--start", required=True)
    ap.add_argument("--work", default="eval_work_ng")
    ap.add_argument("--only", nargs="*", help="sweep only these keys")
    args = ap.parse_args()
    spec = yaml.safe_load(open(args.spec))
    songs = discover(spec["data"]["train"], None)
    ev = tune.Evaluator(spec, args.work, songs)
    ev.base_overrides = list(yaml.safe_load(open(args.start))["overrides"])
    ev.metric_keys = list(dict.fromkeys(ev.metric_keys + SHOW))
    names = [n for n, _ in songs]
    print("start:", ev.base_overrides)
    for key, values in SWEEPS.items():
        if args.only and key not in args.only:
            continue
        dims = tune.parse_search({key: values})
        exp, cheap, tail = tune.split_dims(ev.pipeline([], names[0]), dims, 3.0, 1.0)
        print(f"\n## {key}\n")
        print("| value | objective | " + " | ".join(SHOW) + " |")
        print("|---" * (len(SHOW) + 2) + "|")
        for v in values:
            ov = dims[0].override(v)
            per = [ev.score(n, tuple(ov for d in exp), [ov] if cheap else [], tail) for n in names]
            obj = ev.objective(per)
            means = [np.nanmean([d.get(k) if d.get(k) is not None else np.nan for d in per]) for k in SHOW]
            print(f"| {v} | {obj:.4f} | " + " | ".join(f"{m:.3f}" for m in means) + " |", flush=True)


if __name__ == "__main__":
    main()
