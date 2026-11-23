"""Coordinate-descent tuning over pipeline stages that build on each other.

    python eval/optimize.py --songs eval_data/dev_clips --stages eval/stages_cpu.json [--work eval_work_clips]

Each stage is a list of candidate Settings overrides; the first candidate is the
starting point. A pass re-tunes every stage in order while the others stay at
their current best. Passes repeat until one changes nothing (or --max-passes).
Every configuration is evaluated once: runs are named by a hash of their
settings and reused across passes and across invocations, and UltraSinger's
own caches make re-runs cheap when only late stages differ.

Objective: weighted sum of mean metrics over "clean" songs (charts not flagged
as transposed), default 0.5*usdx_medium + 0.5*(1 - wer).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_objective(text: str) -> list[tuple[str, float]]:
    out = []
    for part in text.split(","):
        name, w = part.split(":")
        out.append((name.strip(), float(w)))
    return out


def config_id(overrides: dict) -> str:
    blob = json.dumps(overrides, sort_keys=True)
    return "opt_" + hashlib.sha1(blob.encode()).hexdigest()[:10]


def to_sets(overrides: dict) -> list[str]:
    args = []
    for k, v in sorted(overrides.items()):
        args += ["--set", f"{k}={str(v).lower() if isinstance(v, bool) else v}"]
    return args


def objective(run_dir: str, weights) -> tuple[float, dict]:
    with open(os.path.join(run_dir, "metrics.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    clean = [r for r in rows if not r.get("error") and r.get("transpose") in (None, "", "0")]
    means = {}
    score = 0.0
    for name, w in weights:
        vals = []
        for r in clean:
            try:
                v = float(r.get(name, "nan"))
            except ValueError:
                continue
            if not math.isnan(v):
                vals.append(v)
        m = sum(vals) / len(vals) if vals else float("nan")
        means[name] = m
        score += w * m if not math.isnan(m) else 0.0  # "wer:-0.5" with --offset 0.5 == 0.5*(1-wer)
    return score, means


def evaluate(args, overrides: dict, weights, log) -> float:
    name = config_id(overrides)
    run_dir = os.path.join(args.work, "runs", name)
    if not os.path.exists(os.path.join(run_dir, "metrics.csv")):
        t0 = time.time()
        cmd = [sys.executable, os.path.join(HERE, "run_eval.py"), "--songs", args.songs, "--name", name,
               "--work", args.work, *to_sets(overrides)]
        if args.exclude:
            cmd += ["--exclude", *args.exclude]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, check=False)
        log(f"  ran {name} in {time.time() - t0:.0f}s")
    score, means = objective(run_dir, weights)
    score += args.offset
    log(f"  {score:.4f}  {json.dumps(overrides, sort_keys=True)}  {name}  "
        + " ".join(f"{k}={v:.3f}" for k, v in means.items()))
    return score


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", required=True)
    ap.add_argument("--stages", required=True,
                    help="JSON: [[stage_name, [override, ...]], ...]; a one-candidate stage just fixes settings")
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "eval_work"))
    ap.add_argument("--objective", default="usdx_medium:0.5,wer:-0.5")
    ap.add_argument("--offset", type=float, default=0.5, help="constant added to the objective (0.5 makes the default 0.5*usdx + 0.5*(1-wer))")
    ap.add_argument("--min-gain", type=float, default=0.002, help="smaller gains don't count as improvement")
    ap.add_argument("--max-passes", type=int, default=4)
    ap.add_argument("--exclude", nargs="*")
    args = ap.parse_args()
    args.work = os.path.abspath(args.work)
    args.songs = os.path.abspath(args.songs)

    stages = json.load(open(args.stages, encoding="utf-8"))
    weights = parse_objective(args.objective)
    log_path = os.path.join(args.work, f"optimize_{os.path.splitext(os.path.basename(args.stages))[0]}.log")
    os.makedirs(args.work, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    choice = {name: 0 for name, _ in stages}

    def merged(override_for=None, idx=None):
        out = {}
        for name, cands in stages:
            i = idx if name == override_for else choice[name]
            out.update(cands[i])
        return out

    log(f"# optimize {args.stages} on {args.songs}  objective={args.objective} +{args.offset}")
    best = evaluate(args, merged(), weights, log)
    for p in range(1, args.max_passes + 1):
        log(f"\n## pass {p} (start {best:.4f})")
        changed = False
        for name, cands in stages:
            log(f"stage {name}")
            scores = []
            for i in range(len(cands)):
                scores.append(best if i == choice[name] else evaluate(args, merged(name, i), weights, log))
            i_best = max(range(len(cands)), key=lambda i: scores[i])
            if i_best != choice[name] and scores[i_best] > best + args.min_gain:
                log(f"  -> {name}: {cands[choice[name]]} => {cands[i_best]}  ({best:.4f} -> {scores[i_best]:.4f})")
                choice[name], best, changed = i_best, scores[i_best], True
            else:
                log(f"  -> {name}: keep {cands[choice[name]]}")
        if not changed:
            log(f"\nconverged after pass {p}")
            break
    log(f"\nbest {best:.4f}: {json.dumps(merged(), sort_keys=True)}  ({config_id(merged())})")


if __name__ == "__main__":
    main()
