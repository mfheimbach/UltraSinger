"""Model load vs inference time on this machine (what keeping models loaded would save).

  python eval/profiling/model_probe.py game    VOCALS.wav [--seconds 120]
  python eval/profiling/model_probe.py whisper VOCALS.wav [--batch 4 16] [--compute float16 int8]
  python eval/profiling/model_probe.py punct   [--words 600]

game: session load per model, active onnxruntime providers (CUDA or silently CPU?), inference per
30 s chunk for steps 4/8 and with deterministic compute on/off. whisper: model and aligner load vs
transcription per batch size / precision. punct: punctuation model load vs one song's inference.
Use a vocals file from a song folder (cache/*_vocals*.wav or the [Vocals].opus output).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def timed(label, fn, *a, **kw):
    t = time.time()
    out = fn(*a, **kw)
    print(f"  {label:44s} {time.time() - t:7.2f} s", flush=True)
    return out


def probe_game(a):
    import librosa
    import numpy as np
    import onnxruntime as ort
    import yaml
    from modules.Midi import game_segmenter as gs
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "presets", "best.yaml"), encoding="utf-8") as f:
        p = yaml.safe_load(f)["params"]["segment.game"]
    model_dir = os.path.expanduser(p["model_dir"])
    print(f"onnxruntime {ort.__version__}, available providers: {ort.get_available_providers()}")
    for deterministic in (True, False):
        orig = ort.SessionOptions

        def opts(det=deterministic):
            o = orig()
            o.use_deterministic_compute = det
            return o
        ort.SessionOptions = lambda: opts()          # probe only: switch the flag GameModel sets
        try:
            m = timed(f"load 5 sessions (deterministic={deterministic})", gs.GameModel, model_dir, a.device, 0)
        finally:
            ort.SessionOptions = orig
        if deterministic:
            print(f"  active providers: {m.encoder.get_providers()}")
        y, _ = librosa.load(a.audio, sr=m.sr, mono=True, duration=a.seconds)
        chunk = int(30 * m.sr)
        for steps in (8, 4):
            t = time.time()
            for i in range(0, len(y), chunk):
                w = y[i:i + chunk]
                d = len(w) / m.sr          # realistic known regions: ~1 s pieces (syllables and gaps)
                regions = [1.0] * int(d) + ([d - int(d)] if d - int(d) > 1e-3 else [])
                m.run(w, regions, 0, p["boundary_threshold"], p["radius"], steps, p["presence_threshold"])
            print(f"  inference {len(y) / m.sr:.0f} s audio, steps {steps:<2d} (det={deterministic})"
                  f"{'':6s} {time.time() - t:7.2f} s", flush=True)


def probe_whisper(a):
    import whisperx
    audio = whisperx.load_audio(a.audio)
    print(f"audio {len(audio) / 16000:.0f} s")
    for compute in a.compute:
        model = timed(f"load large-v3-turbo ({compute})", whisperx.load_model, "large-v3-turbo",
                      language=a.language, device=a.device, compute_type=compute)
        for b in a.batch:
            res = timed(f"transcribe batch {b} ({compute})", model.transcribe, audio, batch_size=b, language=a.language)
        del model
    ma, meta = timed("load aligner", whisperx.load_align_model, language_code=a.language, device=a.device)
    timed("align", whisperx.align, res["segments"], ma, meta, audio, a.device, return_char_alignments=False)


def probe_punct(a):
    from modules.Ultrastar import line_breaks as lb
    words = ("and I said hello there how are you today I am fine thank you " * 100).split()[:a.words]
    lb._PUNCT_MEMO.clear() if hasattr(lb, "_PUNCT_MEMO") else None
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "presets", "best.yaml"), encoding="utf-8") as f:
        p = yaml.safe_load(f)["params"]["notes.line_breaks"]
    name = p["model"]
    timed(f"load + infer {len(words)} words (cold)", lb._punctuation_probabilities, words, name, a.device,
          p["punct_chunk"], p["punct_context"])
    import transformers
    timed("load model only", transformers.AutoModelForTokenClassification.from_pretrained, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["game", "whisper", "punct"])
    ap.add_argument("audio", nargs="?")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--batch", type=int, nargs="+", default=[4, 16])
    ap.add_argument("--compute", nargs="+", default=["float16", "int8"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--words", type=int, default=600)
    a = ap.parse_args()
    if a.what != "punct" and not a.audio:
        ap.error("needs a vocals audio file")
    {"game": probe_game, "whisper": probe_whisper, "punct": probe_punct}[a.what](a)


if __name__ == "__main__":
    main()
