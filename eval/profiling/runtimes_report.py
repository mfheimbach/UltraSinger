"""Where the time goes: stage runtimes from the runtime log (~/.cache/ultrasinger/runtimes.jsonl).

  python eval/profiling/runtimes_report.py [--device cuda] [--since 2026-11-26] [--log PATH]

Per stage: runs, median seconds per song and per minute of audio, median peak memory, and its share
of all recorded stage time. A slow stage with many runs is where caching or speed-ups pay most.
"""

from __future__ import annotations

import argparse
import os
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from pipeline import runtimes  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--since", help="only records from this date/time on (YYYY-MM-DD[ HH:MM])")
    ap.add_argument("--log", help="runtime log path (default: ULTRASINGER_RUNTIME_LOG or ~/.cache/...)")
    a = ap.parse_args()
    if a.log:
        os.environ["ULTRASINGER_RUNTIME_LOG"] = a.log
    recs = [r for r in runtimes.load() if r.get("seconds") is not None
            and (not a.device or r.get("device") == a.device) and (not a.since or r.get("t", "") >= a.since)]
    if not recs:
        raise SystemExit(f"no records in {runtimes.log_path()} for these filters")
    by = defaultdict(list)
    for r in recs:
        by[(r["stage"], r.get("device"))].append(r)
    total = sum(r["seconds"] for r in recs)
    rows = []
    for (stage, dev), rs in by.items():
        secs = [r["seconds"] for r in rs]
        per_min = [r["seconds"] / r["audio_seconds"] * 60 for r in rs if r.get("audio_seconds")]
        mem = [r["peak_rss_mb"] for r in rs if r.get("peak_rss_mb")]
        rows.append((sum(secs), stage, dev, len(rs), st.median(secs),
                     st.median(per_min) if per_min else float("nan"), st.median(mem) if mem else float("nan")))
    print(f"{len(recs)} records, {total / 3600:.1f} h of stage time ({runtimes.log_path()})\n")
    print(f"{'stage':28s} {'device':6s} {'runs':>6s} {'s/song':>8s} {'s/min audio':>11s} {'peak MB':>8s} {'share':>6s}")
    for tot, stage, dev, n, med, pm, mem in sorted(rows, reverse=True):
        print(f"{stage:28s} {str(dev):6s} {n:6d} {med:8.1f} {pm:11.1f} {mem:8.0f} {tot / total:6.1%}")


if __name__ == "__main__":
    main()
