"""Clean a stage cache: remove entries not written or used since a cut-off.

    python -m pipeline.cache_gc eval_work_ng/cache --older-than-hours 48            # dry run
    python -m pipeline.cache_gc eval_work_ng/cache --older-than-hours 48 --delete

Cache files are only ever written (a hit doesn't touch them), so "old" means: produced by code or
settings that no current run asked for since the cut-off. Separation folders (separation_*) are kept
by default: they are content-addressed, expensive to recompute and reused across settings.
"""

from __future__ import annotations

import argparse
import os
import time


def stale_files(cache_dir: str, cutoff: float, keep_prefixes: tuple[str, ...]) -> list[str]:
    out = []
    for root, dirs, files in os.walk(cache_dir):
        dirs[:] = [d for d in dirs if not d.startswith(keep_prefixes)]
        for f in files:
            p = os.path.join(root, f)
            if os.path.getmtime(p) < cutoff:
                out.append(p)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir")
    ap.add_argument("--older-than-hours", type=float, required=True)
    ap.add_argument("--keep", nargs="*", default=["separation_"], help="folder name prefixes never touched")
    ap.add_argument("--delete", action="store_true", help="actually delete (default: list only)")
    args = ap.parse_args()
    cutoff = time.time() - args.older_than_hours * 3600
    files = stale_files(args.cache_dir, cutoff, tuple(args.keep))
    size = sum(os.path.getsize(p) for p in files)
    print(f"{len(files)} files, {size / 1e9:.2f} GB older than {args.older_than_hours:g} h "
          f"(kept: {', '.join(args.keep) or '-'})")
    if not args.delete:
        print("dry run; add --delete to remove them")
        return
    for p in files:
        os.remove(p)
    for root, dirs, files_ in os.walk(args.cache_dir, topdown=False):   # drop emptied folders
        if root != args.cache_dir and not os.listdir(root):
            os.rmdir(root)
    print("deleted")


if __name__ == "__main__":
    main()
