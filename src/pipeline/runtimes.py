"""Runtime records and estimates.

Every computed stage appends one line to a runtime log (default ~/.cache/ultrasinger/runtimes.jsonl,
override with ULTRASINGER_RUNTIME_LOG): stage, parameters, device, CPU count, audio length, seconds,
peak memory, code version. ``estimate()`` fits seconds = a + b * audio_seconds per stage (same
stage, device and parameters; falls back to same stage and device) and sums a preset.

  python src/pipeline/runtimes.py best --seconds 210 [--device cpu]
"""

from __future__ import annotations

import json
import os
import time

DEFAULT_LOG = os.path.join(os.path.expanduser("~"), ".cache", "ultrasinger", "runtimes.jsonl")


def log_path() -> str:
    return os.environ.get("ULTRASINGER_RUNTIME_LOG", DEFAULT_LOG)


def record(entry: dict) -> None:
    path = log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}, default=str) + "\n")
    except OSError:
        pass  # runtime records are a convenience, never a reason to fail a run


def load() -> list[dict]:
    try:
        with open(log_path(), encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return []


def audio_seconds(path) -> float | None:
    try:
        import soundfile
        return float(soundfile.info(path).duration)
    except Exception:
        try:
            import librosa
            return float(librosa.get_duration(path=path))
        except Exception:
            return None


def _fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Least squares seconds = a + b * audio_seconds; with one length only: proportional."""
    xs = [x for x, _ in points]
    if len(points) == 1 or max(xs) - min(xs) < 1.0:
        ratio = sorted(y / max(x, 1e-6) for x, y in points)[len(points) // 2]
        return 0.0, ratio
    n = len(points)
    mx, my = sum(xs) / n, sum(y for _, y in points) / n
    b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    if a < 0:  # no negative fixed cost: refit through the origin
        a, b = 0.0, sum(x * y for x, y in points) / sum(x * x for x in xs)
    return a, max(b, 0.0)


def estimate(stages, seconds: float, device: str = "cpu", records: list[dict] | None = None):
    """Per stage (id, estimated seconds or None, number of records used) and the total."""
    records = load() if records is None else records
    rows, total = [], 0.0
    for st in stages:
        params = json.dumps(_params(st), sort_keys=True, default=str)
        same = [r for r in records if r.get("stage") == st.name and r.get("device") == device
                and r.get("audio_seconds") and json.dumps(r.get("params"), sort_keys=True, default=str) == params]
        pool = same or [r for r in records if r.get("stage") == st.name and r.get("device") == device
                        and r.get("audio_seconds")]
        if not pool:
            rows.append((st.id, None, 0))
            continue
        a, b = _fit([(r["audio_seconds"], r["seconds"]) for r in pool[-50:]])
        est = a + b * seconds
        rows.append((st.id, est, len(pool)))
        total += est
    return rows, total


def _params(st) -> dict:
    import dataclasses
    return dataclasses.asdict(st.params)


def main(argv=None):
    import argparse
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if here not in sys.path:
        sys.path.insert(0, here)
    from pipeline import stages  # noqa: F401
    from pipeline.core import load_preset
    from pipeline.run import preset_path
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("--seconds", type=float, required=True, help="audio length")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args(argv)
    p = load_preset(preset_path(args.preset), args.set)
    rows, total = estimate(p.stages, args.seconds, args.device)
    for sid, est, n in rows:
        print(f"{sid:28s} {'?' if est is None else f'{est:7.1f}s'}  ({n} records)")
    unknown = [sid for sid, est, _ in rows if est is None]
    print(f"{'total':28s} {total:7.1f}s" + (f"  + unknown: {', '.join(unknown)}" if unknown else ""))


if __name__ == "__main__":
    main()
