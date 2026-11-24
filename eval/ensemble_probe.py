"""Probe ensemble potential for note pitch: RMVPE vs SwiftF0 vs GAME on chart note timings.

    python eval/ensemble_probe.py --songs eval_data/usdb

Needs cached pitch tracks from repitch runs with pitch_audio=vocals for rmvpe and swiftf0
(eval_work/cache/<song>/{rmvpe,swiftf0}_vocals_True_sep*.json) and the separated vocals.
Reports per-detector note pitch accuracy (chroma, exact), majority vote, agreement-based
accuracy and the oracle (any detector right) as the ensemble ceiling.
"""

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)

import librosa  # noqa: E402

import usdx  # noqa: E402
from run_eval import discover, slug  # noqa: E402
from modules.Midi.midi_creator import NotePitchOptions, note_from_frames, estimate_tuning_offset  # noqa: E402
from modules.Midi.game_segmenter import GameModel  # noqa: E402
from modules.Pitcher.pitched_data import PitchedData  # noqa: E402


def track_notes(pitched: PitchedData, notes, opts):
    t = np.asarray(pitched.times)
    f = np.asarray(pitched.frequencies)
    c = np.asarray(pitched.confidence)
    tuning = estimate_tuning_offset(pitched, opts.conf_threshold)
    out = []
    for n in notes:
        a, b = np.searchsorted(t, [n.start, n.end])
        if b <= a:
            b = min(a + 1, len(t))
        note, _hint = note_from_frames(f[a:b], c[a:b], opts, tuning)
        out.append(int(librosa.note_to_midi(note)))
    return np.array(out)


def game_notes(model: GameModel, vocals: str, notes):
    y, sr = librosa.load(vocals, sr=model.sr, mono=True)
    out = np.full(len(notes), -1)
    # windows of <= 30 s cut between notes
    i = 0
    while i < len(notes):
        j = i
        while j + 1 < len(notes) and notes[j + 1].end - notes[i].start < 30:
            j += 1
        c0 = max(0.0, notes[i].start - 0.2)
        c1 = min(len(y) / sr, notes[j].end + 0.2)
        regions, cursor, idx = [], c0, []
        for k in range(i, j + 1):
            a, b = max(notes[k].start, cursor), min(notes[k].end, c1)
            if b <= a:
                continue
            if a > cursor:
                regions.append(a - cursor)
                idx.append(None)
            regions.append(b - a)
            idx.append(k)
            cursor = b
        if c1 > cursor:
            regions.append(c1 - cursor)
            idx.append(None)
        wave = y[int(c0 * sr): int(c1 * sr)].astype(np.float32)
        dur = np.array([len(wave) / sr], dtype=np.float32)
        x_seg, x_est, mask_t = model.encoder.run(None, {"waveform": wave[None], "duration": dur})
        known = model.dur2bd.run(None, {"durations": np.array([regions], dtype=np.float32), "maskT": mask_t})[0]
        durs, mask_n = model.bd2dur.run(None, {"boundaries": known, "maskT": mask_t})
        _pres, scores = model.estimator.run(None, {"x_est": x_est, "boundaries": known, "maskT": mask_t,
                                                   "maskN": mask_n, "threshold": np.array(0.2, dtype=np.float32)})
        n = int(mask_n[0].sum())
        for r, k in zip(range(min(n, len(idx))), idx):
            if k is not None:
                out[k] = int(round(scores[0][r]))
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--work", default=os.path.join(os.path.dirname(HERE), "eval_work"))
    ap.add_argument("--game", default=os.path.expanduser("~/.cache/game/GAME-1.0.3-small-onnx"))
    args = ap.parse_args()

    opts = NotePitchOptions(method="chroma_mode", tuning_correction=True)
    model = GameModel(args.game)
    totals = {}
    for name, txt in discover(args.songs, None, ["Systemabsturz"]):
        cache = os.path.join(args.work, "cache", slug(name))
        rm = glob.glob(os.path.join(cache, "rmvpe_vocals_True_sep*.json"))
        sw = glob.glob(os.path.join(cache, "swiftf0_vocals_True_sep*.json"))
        vocals = glob.glob(os.path.join(cache, "separated", "htdemucs", "*", "vocals.wav"))
        if not (rm and sw and vocals):
            print(f"skip {name}: missing caches")
            continue
        gt = usdx.read(txt)
        notes = [n for n in gt.notes if n.pitched]
        g = np.array([n.pitch for n in notes]) + 48  # UltraSinger convention -> MIDI
        preds = {
            "rmvpe": track_notes(PitchedData.from_json(open(rm[0]).read()), notes, opts),
            "swiftf0": track_notes(PitchedData.from_json(open(sw[0]).read()), notes, opts),
            "game": game_notes(model, vocals[0], notes),
        }
        ok = {k: (v - g) % 12 == 0 for k, v in preds.items()}
        r, s, m = preds["rmvpe"], preds["swiftf0"], preds["game"]
        vote = np.where((s % 12) == (m % 12), s, r)  # majority if two agree (by pitch class), else RMVPE
        ok["vote"] = (vote - g) % 12 == 0
        ok["oracle"] = ok["rmvpe"] | ok["swiftf0"] | ok["game"]
        agree3 = ((r - s) % 12 == 0) & ((r - m) % 12 == 0)
        ok["_agree3_share"] = agree3
        ok["_acc_when_agree3"] = ok["rmvpe"][agree3] if agree3.any() else np.array([])
        ok["_acc_when_disagree"] = ok["rmvpe"][~agree3] if (~agree3).any() else np.array([])
        for k, v in ok.items():
            totals.setdefault(k, []).append(float(np.mean(v)) if len(v) else float("nan"))
        print(f"{name[:34]:34s} " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in ok.items()
                                               if not k.startswith("_") and len(v)))
    print("\nmean over songs:")
    for k, v in totals.items():
        print(f"  {k:20s} {np.nanmean(v):.3f}")


if __name__ == "__main__":
    main()
