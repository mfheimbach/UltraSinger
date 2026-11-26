"""Reference-free features of one candidate run (no chart needed), for learning to pick between
parallel options (separation seeds/models, transcribers, GAME seeds, ...).

A candidate is a run directory written by run_song.py: the chart (*.txt) plus store.pkl with the
pipeline artifacts. Feature groups (prefix):
  sep_   separated stems vs the mix (reconstruction, level, harmonicity, flatness, leakage)
  pit_   pitch track (voiced share, confidence, jitter, octave jumps)
  txt_   transcript (alignment confidence, density, repetitions, odd words)
  not_   notes (count, durations, agreement with the pitch track, coverage of voiced frames)
  agr_   agreement with the other candidates of the same group (needs the group, see agreement())
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
from functools import lru_cache

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)
sys.path.insert(0, os.path.join(os.path.dirname(EVAL), "src"))

import usdx  # noqa: E402

SR = 16000


def load_candidate(run_song_dir: str) -> dict:
    with open(os.path.join(run_song_dir, "store.pkl"), "rb") as f:
        store = pickle.load(f)
    charts = [p for p in glob.glob(os.path.join(run_song_dir, "**", "*.txt"), recursive=True)]
    store["_chart"] = usdx.read(charts[0]) if charts else None
    return store


@lru_cache(maxsize=64)
def _audio(path: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y


def _frames_rms(y: np.ndarray, hop: int = 800) -> np.ndarray:  # 50 ms
    n = len(y) // hop
    return np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)


def separation_features(store: dict) -> dict:
    mix, voc, inst = store.get("audio.mix"), store.get("audio.vocals"), store.get("audio.instrumental")
    if not (mix and voc and inst):
        return {}
    import librosa
    m, v, i = _audio(mix), _audio(voc), _audio(inst)
    n = min(len(m), len(v), len(i))
    m, v, i = m[:n], v[:n], i[:n]
    out = {"sep_recon_err_db": 10 * np.log10(np.sum((m - v - i) ** 2) / (np.sum(m ** 2) + 1e-12) + 1e-12),
           "sep_voc_level_db": 10 * np.log10(np.sum(v ** 2) / (np.sum(m ** 2) + 1e-12) + 1e-12)}
    rv, ri = _frames_rms(v), _frames_rms(i)
    active = rv > rv.max() * 10 ** (-35 / 20)
    out["sep_voc_active"] = float(active.mean())
    out["sep_env_corr"] = float(np.corrcoef(rv, ri)[0, 1]) if rv.std() > 0 and ri.std() > 0 else 0.0
    # quiet parts of the vocal stem: how much is left there (bleed) relative to the sung parts
    quiet = ~active
    out["sep_bleed_db"] = float(20 * np.log10((np.median(rv[quiet]) + 1e-9) / (np.median(rv[active]) + 1e-9))) \
        if quiet.any() and active.any() else 0.0
    S = np.abs(librosa.stft(v, n_fft=1024, hop_length=512))
    H, P = librosa.decompose.hpss(S)
    frame_active = np.repeat(active, int(np.ceil(S.shape[1] / max(len(active), 1))))[: S.shape[1]]
    if frame_active.any():
        out["sep_harmonic_share"] = float(np.sum(H[:, frame_active] ** 2) /
                                          (np.sum(H[:, frame_active] ** 2) + np.sum(P[:, frame_active] ** 2) + 1e-12))
        flat = librosa.feature.spectral_flatness(S=S)[0]
        out["sep_flatness"] = float(np.mean(flat[frame_active]))
    return out


def pitch_features(store: dict) -> dict:
    p = store.get("pitch")
    if p is None:
        return {}
    f = np.asarray(p.frequencies, float)
    c = np.asarray(p.confidence, float)
    voiced = f > 0
    out = {"pit_voiced_frac": float(voiced.mean()),
           "pit_conf_voiced": float(c[voiced].mean()) if voiced.any() else 0.0}
    midi = 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0)
    both = voiced[1:] & voiced[:-1]
    d = np.abs(np.diff(midi))[both]
    out["pit_jitter"] = float(np.median(d)) if d.size else 0.0
    out["pit_octave_jumps"] = float(np.mean(d > 9)) if d.size else 0.0
    out["pit_range"] = float(np.percentile(midi[voiced], 95) - np.percentile(midi[voiced], 5)) if voiced.sum() > 20 else 0.0
    return out


def text_features(store: dict) -> dict:
    words = store.get("words") or []
    if not words:
        return {"txt_n_words": 0}
    texts = [w.word.strip().lower() for w in words]
    conf = np.array([getattr(w, "confidence", 0) or 0 for w in words], float)
    voiced_sec = 0.0
    p = store.get("pitch")
    if p is not None:
        voiced_sec = float(np.sum(np.asarray(p.frequencies) > 0) * 0.01)
    out = {"txt_n_words": len(words),
           "txt_words_per_voiced_sec": len(words) / voiced_sec if voiced_sec > 0 else 0.0,
           "txt_align_conf": float(conf.mean()),
           "txt_align_conf_p10": float(np.percentile(conf, 10)),
           "txt_repeat_rate": float(np.mean([a == b for a, b in zip(texts, texts[1:])])) if len(texts) > 1 else 0.0,
           "txt_max_word_len": max(len(t) for t in texts),
           "txt_unique_share": len(set(texts)) / len(texts),
           "txt_word_dur_median": float(np.median([w.end - w.start for w in words]))}
    return out


LM_NAME = os.environ.get("SELECTION_LM", "Qwen/Qwen2.5-0.5B")
_LM = {}


def lm_nll(text: str) -> float | None:
    """Mean token negative log-likelihood of the lyrics under a small multilingual LM: how plausible
    the transcript reads as text (garbled or hallucinated transcripts score high). Cached."""
    import hashlib
    import json
    cache_path = os.path.join(EVAL, "selection", ".lm_cache.json")
    if "cache" not in _LM:
        try:
            with open(cache_path, encoding="utf-8") as f:
                _LM["cache"] = json.load(f)
        except (OSError, ValueError):
            _LM["cache"] = {}
    key = hashlib.sha1(f"{LM_NAME}|{text}".encode()).hexdigest()
    if key in _LM["cache"]:
        return _LM["cache"][key]
    if not text.strip():
        return None
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if "model" not in _LM:
        _LM["tok"] = AutoTokenizer.from_pretrained(LM_NAME)
        _LM["model"] = AutoModelForCausalLM.from_pretrained(LM_NAME, torch_dtype=torch.float32).eval()
    ids = _LM["tok"](text, return_tensors="pt").input_ids[:, :512]
    with torch.no_grad():
        loss = _LM["model"](ids, labels=ids).loss.item()
    _LM["cache"][key] = loss
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(_LM["cache"], f)
    return loss


def lm_features(store: dict) -> dict:
    words = store.get("words") or []
    text = " ".join(w.word.strip() for w in words)
    nll = lm_nll(text)
    return {} if nll is None else {"txt_lm_nll": nll}


def note_features(store: dict) -> dict:
    song = store.get("_chart")
    p = store.get("pitch")
    if song is None or not song.notes:
        return {"not_n_notes": 0}
    notes = song.notes
    dur = np.array([n.end - n.start for n in notes])
    span = notes[-1].end - notes[0].start
    out = {"not_n_notes": len(notes), "not_per_sec": len(notes) / max(span, 1e-6),
           "not_dur_median": float(np.median(dur)), "not_short_frac": float(np.mean(dur < 0.1))}
    if p is not None:
        f = np.asarray(p.frequencies, float)
        t = np.asarray(p.times, float)
        midi = 69 + 12 * np.log2(np.maximum(f, 1e-3) / 440.0)
        voiced = f > 0
        covered = np.zeros_like(voiced)
        devs = []
        for n in notes:
            sel = (t >= n.start) & (t < n.end)
            covered |= sel
            sv = sel & voiced
            if sv.sum() >= 3:
                # chart pitch is (octave-free in scoring) -> compare chroma distance
                d = (np.median(midi[sv]) - n.pitch) % 12
                devs.append(min(d, 12 - d))
        out["not_pitch_dev"] = float(np.mean(devs)) if devs else 6.0
        out["not_voiced_covered"] = float((covered & voiced).sum() / max(voiced.sum(), 1))
        out["not_frames_voiced"] = float((covered & voiced).sum() / max(covered.sum(), 1))
    return out


def features(store: dict) -> dict:
    out = {}
    fns = [separation_features, pitch_features, text_features, note_features]
    if os.environ.get("SELECTION_NO_LM") is None:
        fns.append(lm_features)
    for fn in fns:
        try:
            out.update(fn(store))
        except Exception as err:  # one broken group must not lose the others
            out[f"{fn.__name__}_error"] = str(err)[:80]
    return out


def agreement(stores: dict[str, dict]) -> dict[str, dict]:
    """Per candidate: how well it agrees with the others of its group (pseudo-reference scores)."""
    import metrics
    names = list(stores)
    out = {n: {} for n in names}
    for a in names:
        sa = stores[a]
        usdx_vs, wer_vs, pitch_vs = [], [], []
        for b in names:
            if a == b:
                continue
            sb = stores[b]
            if sa.get("_chart") is not None and sb.get("_chart") is not None and sb["_chart"].notes:
                fm = metrics.frame_metrics(sb["_chart"], sa["_chart"])
                usdx_vs.append(fm.get("usdx_medium", np.nan))
                lm = metrics.lyric_metrics(sb["_chart"], sa["_chart"])
                wer_vs.append(lm.get("wer", np.nan))
            pa, pb = sa.get("pitch"), sb.get("pitch")
            if pa is not None and pb is not None:
                fa, fb = np.asarray(pa.frequencies), np.asarray(pb.frequencies)
                k = min(len(fa), len(fb))
                both = (fa[:k] > 0) & (fb[:k] > 0)
                if both.any():
                    pitch_vs.append(float(np.median(np.abs(12 * np.log2(fa[:k][both] / fb[:k][both])))))
        out[a] = {"agr_usdx": float(np.nanmean(usdx_vs)) if usdx_vs else np.nan,
                  "agr_wer": float(np.nanmean(wer_vs)) if wer_vs else np.nan,
                  "agr_pitch": float(np.nanmean(pitch_vs)) if pitch_vs else np.nan}
    return out
