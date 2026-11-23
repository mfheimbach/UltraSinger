"""Evaluate UltraSinger on a folder of ground-truth songs.

    python eval/run_eval.py --songs DATA/songs --name baseline [--mode full|repitch] [--set key=value ...]

DATA/songs contains one folder per song, each with an UltraStar txt and
the audio it references (the layout USDB Syncer produces).

Outputs (under --work, default ./eval_work, git-ignored):
  cache/<song>/          separation, transcription and pitch caches, shared by all runs
  runs/<name>/<song>/    UltraSinger output + run.log
  runs/<name>/metrics.csv
  runs/<name>/summary.md

Re-score an existing run without re-running UltraSinger: add --score-only.
Compare runs: python eval/compare.py runs/a runs/b
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import metrics  # noqa: E402
import usdx  # noqa: E402

HEADLINE = {
    "full": ["usdx_medium", "chroma_acc", "voicing_f1", "onset_f1_100ms", "note_count_ratio", "wer", "best_offset_ms", "pitch_shift"],
    "repitch": ["note_chroma_acc", "note_chroma_acc_pm1", "note_chroma_acc_dur", "octave_errors", "transpose"],
}


def slug(name: str) -> str:
    return re.sub(r"[^\w\-. ]", "_", name).strip()


def _song_folders(root: str):
    """Every folder below root that holds an UltraStar txt (USDB Syncer may nest by artist)."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        if any(f.lower().endswith(".txt") for f in filenames) and usdx.find_song_txt(dirpath):
            yield dirpath


def discover(songs_dir: str, only: list[str] | None, exclude: list[str] | None = None) -> list[tuple[str, str]]:
    out = []
    songs_dir = os.path.abspath(songs_dir)
    for folder in _song_folders(songs_dir):
        name = os.path.basename(folder)
        if only and not any(o.lower() in name.lower() for o in only):
            continue
        if exclude and any(x.lower() in name.lower() for x in exclude):
            continue
        txt = usdx.find_song_txt(folder)
        song = usdx.read(txt)
        audio = song.audio_path()
        if not audio or not os.path.exists(audio):
            print(f"skip {name}: audio missing ({audio})")
            continue
        if song.is_duet:
            print(f"note {name}: duet, only P1 is scored")
        out.append((name, txt))
    return out


def run_one(mode, name, txt, run_dir, cache_dir, overrides, timeout, extra=()) -> tuple[bool, float]:
    out = os.path.join(run_dir, slug(name))
    shutil.rmtree(out, ignore_errors=True)  # UltraSinger appends (1), (2) to existing folders
    os.makedirs(out)
    cmd = [sys.executable, os.path.join(HERE, "run_song.py"), "--mode", mode, "--gt", txt,
           "--out", out, "--cache", os.path.join(cache_dir, slug(name))]
    for o in overrides:
        cmd += ["--set", o]
    cmd += list(extra)
    t0 = time.time()
    with open(os.path.join(out, "run.log"), "w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                  cwd=os.path.join(os.path.dirname(HERE), "src"))
            ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            log.write("\nTIMEOUT\n")
            ok = False
    return ok, time.time() - t0


def score_one(mode, name, txt, run_dir) -> dict:
    info_path = os.path.join(run_dir, slug(name), "run_info.json")
    if not os.path.exists(info_path):
        return {"song": name, "error": "no output"}
    info = json.load(open(info_path, encoding="utf-8"))
    gt, pred = usdx.read(txt), usdx.read(info["txt"])
    res = metrics.evaluate(gt, pred) if mode == "full" else metrics.evaluate_repitch(gt, pred)
    return {"song": name, "language": gt.language, **res}


def write_outputs(run_dir, mode, rows, config):
    keys = ["song", "language"]
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    cols = [c for c in HEADLINE[mode] if any(c in r for r in rows)]
    lines = [f"# {config['name']} ({mode})", "", f"overrides: `{' '.join(config['overrides']) or '-'}`", ""]
    lines.append("| song | " + " | ".join(cols) + " |")
    lines.append("|---" * (len(cols) + 1) + "|")
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.3f}" if isinstance(v, float) else (str(v) if v is not None else r.get("error", "")))
        lines.append(f"| {r['song']} | " + " | ".join(cells) + " |")
    means = []
    for c in cols:
        vals = [r[c] for r in rows if isinstance(r.get(c), (int, float)) and r[c] == r[c]]
        means.append(f"**{sum(vals) / len(vals):.3f}**" if vals else "")
    lines.append("| **mean** | " + " | ".join(means) + " |")
    clean = [r for r in rows if not r.get("transpose") and "error" not in r]
    if len(clean) < len(rows):
        cm = []
        for c in cols:
            vals = [r[c] for r in clean if isinstance(r.get(c), (int, float)) and r[c] == r[c]]
            cm.append(f"**{sum(vals) / len(vals):.3f}**" if vals else "")
        lines.append(f"| **mean, clean ({len(clean)})** | " + " | ".join(cm) + " |")
    shifted = [r["song"] for r in rows if r.get("transpose")]
    if shifted:
        lines += ["", f"Transposed vs. chart (check the ground truth): {', '.join(shifted)}"]
    open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--songs", required=True)
    ap.add_argument("--name", required=True, help="run name, e.g. baseline or swiftf0_fmax800")
    ap.add_argument("--mode", choices=["full", "repitch"], default="full")
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "eval_work"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="UltraSinger Settings override")
    ap.add_argument("--only", nargs="*", help="substring filter on song folder names")
    ap.add_argument("--exclude", nargs="*", help="substring filter of songs to leave out")
    ap.add_argument("--oracle-lyrics", action="store_true", help="align the chart's lyrics instead of transcribing")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    args.work = os.path.abspath(args.work)
    run_dir = os.path.join(args.work, "runs", args.name)
    cache_dir = os.path.join(args.work, "cache")
    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    if args.score_only and os.path.exists(config_path):
        config = json.load(open(config_path))  # keep the settings the run was made with
        args.mode = config.get("mode", args.mode)
    else:
        config = {"name": args.name, "mode": args.mode, "overrides": args.set, "oracle_lyrics": args.oracle_lyrics}
        json.dump(config, open(config_path, "w"), indent=2)

    songs = discover(args.songs, args.only, args.exclude)
    print(f"{len(songs)} songs")
    rows = []
    for i, (name, txt) in enumerate(songs, 1):
        if not args.score_only:
            extra = ["--oracle-lyrics"] if args.oracle_lyrics else []
            ok, dt = run_one(args.mode, name, txt, run_dir, cache_dir, args.set, args.timeout, extra)
            print(f"[{i}/{len(songs)}] {name}: {'ok' if ok else 'FAILED'} in {dt:.0f}s")
        try:
            row = score_one(args.mode, name, txt, run_dir)
        except Exception as e:  # keep going; one broken file shouldn't sink the run
            row = {"song": name, "error": repr(e)}
        rows.append(row)
    write_outputs(run_dir, args.mode, rows, config)


if __name__ == "__main__":
    main()
