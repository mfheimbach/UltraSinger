"""Fast in-process sweep over NotePitchOptions, using cached pitch tracks.

Needs one prior repitch run per song (fills eval_work/cache/<song>/swiftf0_True_sep*.json).
Reproduces the repitch pipeline's note step exactly, minus the subprocess
and model loading, so each configuration costs seconds.

    python eval/sweep_note_pitch.py --songs eval_data/dev [--exclude Systemabsturz]
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import librosa  # noqa: E402

import metrics  # noqa: E402
import usdx  # noqa: E402
from run_eval import discover, slug  # noqa: E402
from modules.Midi.midi_creator import NotePitchOptions, create_midi_notes_from_pitched_data  # noqa: E402
from modules.Pitcher.pitched_data import PitchedData  # noqa: E402

GRID = {
    "method": ["mode", "median", "chroma_mode"],
    "conf_threshold": [0.4, 0.7, 0.9],
    "edge_trim": [0.0, 0.15, 0.3],
    "tuning_correction": [False, True],
}


def load_cases(songs_dir, work, exclude, separated=True):
    cases = []
    for name, txt in discover(songs_dir, None, exclude):
        cache = os.path.join(work, "cache", slug(name), f"swiftf0_True_sep{separated}.json")
        if not os.path.exists(cache):
            print(f"skip {name}: no cached pitch ({cache})")
            continue
        pitched = PitchedData.from_json(open(cache).read())
        cases.append((name, usdx.read(txt), pitched))
    return cases


def score(cases, options: NotePitchOptions) -> dict:
    per_song = {}
    for name, gt, pitched in cases:
        segs = create_midi_notes_from_pitched_data(
            [n.start for n in gt.notes], [n.end for n in gt.notes], [n.text for n in gt.notes],
            pitched, options=options)
        pred = copy.deepcopy(gt)
        for n, s in zip(pred.notes, segs):
            n.pitch = int(librosa.note_to_midi(s.note)) - 48  # UltraSinger's UltraStar convention
        per_song[name] = metrics.evaluate_repitch(gt, pred)
    keys = ["note_chroma_acc", "note_chroma_acc_pm1", "note_chroma_acc_dur", "octave_errors"]
    mean = {k: sum(r[k] for r in per_song.values()) / len(per_song) for k in keys}
    return {"mean": mean, "per_song": per_song}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "eval_work"))
    ap.add_argument("--exclude", nargs="*")
    ap.add_argument("--out", default=None, help="json file for all results")
    args = ap.parse_args()

    import contextlib
    import io
    cases = load_cases(args.songs, args.work, args.exclude)
    print(f"{len(cases)} songs")

    results = []
    keys = list(GRID)
    for values in itertools.product(*(GRID[k] for k in keys)):
        opts = NotePitchOptions(**dict(zip(keys, values)))
        with contextlib.redirect_stdout(io.StringIO()):  # silence UltraSinger's per-call logging
            res = score(cases, opts)
        results.append({"options": dict(zip(keys, values)), **res})

    results.sort(key=lambda r: -r["mean"]["note_chroma_acc"])
    base = next(r for r in results if r["options"] == {"method": "mode", "conf_threshold": 0.4,
                                                        "edge_trim": 0.0, "tuning_correction": False})
    print(f"\nbaseline: {base['mean']['note_chroma_acc']:.3f}  (±1: {base['mean']['note_chroma_acc_pm1']:.3f})\n")
    print("| method | conf | trim | tuning | chroma_acc | ±1 | dur-weighted |")
    print("|---|---|---|---|---|---|---|")
    for r in results[:12]:
        o, m = r["options"], r["mean"]
        print(f"| {o['method']} | {o['conf_threshold']} | {o['edge_trim']} | {o['tuning_correction']} | "
              f"{m['note_chroma_acc']:.3f} | {m['note_chroma_acc_pm1']:.3f} | {m['note_chroma_acc_dur']:.3f} |")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
