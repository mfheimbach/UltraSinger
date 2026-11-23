"""Compare a generated UltraStar song against a ground-truth one.

Three families of metrics:

* frame-level (10 ms grid): voicing and pitch, robust to different note
  segmentation. Pitch metrics are chroma based (octave ignored), as in
  USDX scoring and because charts often move the singer's octave.
* note-level: onset matching with a tolerance.
* lyrics: word and character error rate.

`usdx_medium` approximates what a singer would score on the real chart
if they sang the generated chart perfectly (USDX medium difficulty:
pitch within +-1 semitone, octave ignored, golden notes count double).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from usdx import Song

HOP = 0.01


def _grid(songs: list[Song]) -> np.ndarray:
    end = max((n.end for s in songs for n in s.notes), default=0.0) + 1.0
    return np.arange(0.0, end, HOP)


def _rasterize(song: Song, t: np.ndarray):
    voiced = np.zeros(t.size, bool)
    pitched = np.zeros(t.size, bool)
    golden = np.zeros(t.size, bool)
    pitch = np.full(t.size, np.nan)
    for n in song.notes:
        a, b = np.searchsorted(t, [n.start, n.end])
        voiced[a:b] = True
        if n.pitched:
            pitched[a:b] = True
            pitch[a:b] = n.pitch
        if n.golden:
            golden[a:b] = True
    return voiced, pitched, golden, pitch


def _chroma_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a - b) % 12
    return np.minimum(d, 12 - d)


def _safe(num, den):
    return float(num) / float(den) if den else float("nan")


def frame_metrics(gt: Song, pred: Song) -> dict:
    t = _grid([gt, pred])
    gv, gp, gg, gpitch = _rasterize(gt, t)
    pv, _pp, _pg, ppitch = _rasterize(pred, t)

    both = gv & pv
    rec = _safe(both.sum(), gv.sum())
    prec = _safe(both.sum(), pv.sum())
    f1 = _safe(2 * rec * prec, rec + prec) if rec == rec and prec == prec else float("nan")

    # pitch on GT-pitched frames; missing prediction counts as wrong (mir_eval style)
    have = gp & ~np.isnan(ppitch)
    dist = np.full(t.size, np.inf)
    dist[have] = _chroma_dist(gpitch[have], ppitch[have])
    exact = gp & ~np.isnan(ppitch) & (gpitch == ppitch)

    weight = np.where(gg, 2.0, 1.0)
    usdx_hits = (gp & (dist <= 1)) * weight
    usdx_total = gp * weight

    return {
        "transpose": _mode_shift(gpitch[have], ppitch[have]),
        "voicing_recall": rec,
        "voicing_precision": prec,
        "voicing_f1": f1,
        "pitch_exact": _safe(exact.sum(), gp.sum()),
        "chroma_acc": _safe((gp & (dist == 0)).sum(), gp.sum()),
        "chroma_acc_pm1": _safe((gp & (dist <= 1)).sum(), gp.sum()),
        # same, but only where something was predicted: isolates pitch from voicing errors
        "chroma_acc_cond": _safe((have & (dist == 0)).sum(), have.sum()),
        "usdx_medium": _safe(usdx_hits.sum(), usdx_total.sum()),
    }


def _mode_shift(gt_pitch: np.ndarray, pred_pitch: np.ndarray) -> int:
    """Most common chroma shift (semitones, -5..6) between prediction and ground truth.
    Nonzero with a clear majority usually means the chart is transposed relative
    to this audio file (wrong version / pitched master), i.e. a ground-truth problem."""
    if gt_pitch.size == 0:
        return 0
    d = (pred_pitch - gt_pitch).astype(int) % 12
    counts = np.bincount(d, minlength=12)
    k = int(counts.argmax())
    # only flag a clear shift: the mode must beat "no shift" by a wide margin
    if k == 0 or counts[k] < 2 * counts[0]:
        return 0
    return k - 12 if k > 6 else k


def best_offset(gt: Song, pred: Song, max_shift: float = 1.0, step: float = 0.02) -> float:
    """How late (s) the prediction is overall, found by maximising voicing overlap.
    A large value points at a GAP/offset problem rather than note errors."""
    t = _grid([gt, pred])
    gv = _rasterize(gt, t)[0]
    pv = _rasterize(pred, t)[0]
    k = int(round(step / HOP))
    best, best_s = -1, 0.0
    for s in range(-int(max_shift / HOP), int(max_shift / HOP) + 1, k):
        shifted = np.roll(pv, s)
        ov = (gv & shifted).sum()
        if ov > best:
            best, best_s = ov, -s * HOP
    return best_s


def note_metrics(gt: Song, pred: Song) -> dict:
    import mir_eval

    def arr(s: Song):
        iv = np.array([[n.start, max(n.end, n.start + 1e-3)] for n in s.notes]) if s.notes else np.zeros((0, 2))
        return iv

    g_iv, p_iv = arr(gt), arr(pred)
    out = {
        "notes_gt": len(gt.notes),
        "notes_pred": len(pred.notes),
        "note_count_ratio": _safe(len(pred.notes), len(gt.notes)),
    }
    if len(g_iv) == 0 or len(p_iv) == 0:
        return out

    for tol in (0.05, 0.1):
        p, r, f = mir_eval.transcription.onset_precision_recall_f1(g_iv, p_iv, onset_tolerance=tol)
        out[f"onset_f1_{int(tol * 1000)}ms"] = f
        out[f"onset_p_{int(tol * 1000)}ms"] = p
        out[f"onset_r_{int(tol * 1000)}ms"] = r

    matches = mir_eval.transcription.match_note_onsets(g_iv, p_iv, onset_tolerance=0.1)
    if matches:
        errs = [abs(g_iv[i, 0] - p_iv[j, 0]) for i, j in matches]
        out["onset_err_median_ms"] = float(np.median(errs) * 1000)
        # note-level pitch: matched notes with equal chroma
        gpn = np.array([gt.notes[i].pitch for i, _ in matches])
        ppn = np.array([pred.notes[j].pitch for _, j in matches])
        pitched = np.array([gt.notes[i].pitched for i, _ in matches])
        hit = (_chroma_dist(gpn, ppn) == 0) & pitched
        n_gt = sum(n.pitched for n in gt.notes)
        n_pred = sum(n.pitched for n in pred.notes)
        out["note_f1_onset_chroma"] = _safe(2 * hit.sum(), n_gt + n_pred)
    return out


def _norm_words(words: list[str]) -> str:
    out = []
    for w in words:
        w = unicodedata.normalize("NFC", w.lower())
        w = re.sub(r"[^\w']", "", w).strip("'")
        if w:
            out.append(w)
    return " ".join(out)


def lyric_metrics(gt: Song, pred: Song) -> dict:
    import jiwer

    ref, hyp = _norm_words(gt.words()), _norm_words(pred.words())
    if not ref:
        return {}
    return {"wer": jiwer.wer(ref, hyp or "<empty>"), "cer": jiwer.cer(ref, hyp or "<empty>")}


def evaluate(gt: Song, pred: Song) -> dict:
    res = {}
    res.update(frame_metrics(gt, pred))
    res["best_offset_ms"] = best_offset(gt, pred) * 1000
    res.update(note_metrics(gt, pred))
    res.update(lyric_metrics(gt, pred))
    return res


def evaluate_repitch(gt: Song, pred: Song) -> dict:
    """Re-pitch mode: timings are identical, so compare note by note."""
    if len(gt.notes) != len(pred.notes):
        return {"error": f"note count mismatch {len(gt.notes)} vs {len(pred.notes)}"}
    g = np.array([n.pitch for n in gt.notes])
    p = np.array([n.pitch for n in pred.notes])
    dur = np.array([n.end - n.start for n in gt.notes])
    m = np.array([n.pitched for n in gt.notes])
    d = _chroma_dist(g[m], p[m])
    w = dur[m]
    return {
        "transpose": _mode_shift(g[m], p[m]),
        "notes": int(m.sum()),
        "note_chroma_acc": float((d == 0).mean()),
        "note_chroma_acc_pm1": float((d <= 1).mean()),
        "note_chroma_acc_dur": float((w * (d == 0)).sum() / w.sum()),
        "note_exact": float((g[m] == p[m]).mean()),
        "octave_errors": float(((d == 0) & (g[m] != p[m])).mean()),
    }
