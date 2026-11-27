"""Note segmentation with GAME (openvpi, Generative Adaptive MIDI Extractor), ONNX models.

Given the audio and syllable timings, GAME predicts notes (boundaries, presence, pitch)
that never cross the given syllable boundaries ("align" mode). Each syllable is then
split into its voiced notes.

Models: https://github.com/openvpi/GAME/releases (ONNX zips; unpack and point
segment.game.model_dir at the folder with config.json). Model license: CC BY-NC-SA 4.0.
"""


import logging
import json
import os

import librosa
import numpy as np

from modules.console_colors import blue_highlighted

log = logging.getLogger("ultrasinger.game_segmenter")



class GameModel:
    def __init__(self, model_dir: str, device: str, seed: int):
        import onnxruntime as ort
        from modules.onnx_session import session
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        opts = ort.SessionOptions()
        opts.use_deterministic_compute = True

        def load(name):
            # The segmenter samples noise (RandomUniformLike without a seed): identical input gave
            # 59-65 notes from run to run. Seed it so results are reproducible (and cacheable).
            import onnx
            model = onnx.load(os.path.join(model_dir, f"{name}.onnx"))
            # onnxruntime's engine (minstd) treats seeds 0 and 1 alike, so seed n is passed as n + 1
            for node in model.graph.node:
                if node.op_type.startswith("Random"):
                    node.attribute.append(onnx.helper.make_attribute("seed", float(seed + 1)))
            return session(model.SerializeToString(), device, opts)
        self.encoder, self.dur2bd, self.segmenter = load("encoder"), load("dur2bd"), load("segmenter")
        self.bd2dur, self.estimator = load("bd2dur"), load("estimator")
        self.sr = self.config["samplerate"]

    def language_id(self, code: str | None) -> int:
        langs = self.config.get("languages") or {}
        return int(langs.get((code or "").lower(), 0))

    def run(self, wave: np.ndarray, regions: list[float], language: int,
            boundary_threshold: float, radius: int, steps: int, presence_threshold: float):
        """wave: mono at self.sr; regions: known region durations (s) covering the audio.
        Returns note durations (s), presence flags, pitches (MIDI float)."""
        duration = np.array([len(wave) / self.sr], dtype=np.float32)
        x_seg, x_est, mask_t = self.encoder.run(None, {"waveform": wave[None].astype(np.float32), "duration": duration})
        known = self.dur2bd.run(None, {"durations": np.array([regions], dtype=np.float32), "maskT": mask_t})[0]
        bounds = known
        lang = np.array([language], dtype=np.int64)
        for i in range(steps):
            t = np.array([i / steps], dtype=np.float32)
            bounds = self.segmenter.run(None, {
                "x_seg": x_seg, "language": lang, "known_boundaries": known, "prev_boundaries": bounds,
                "t": t, "maskT": mask_t, "threshold": np.array(boundary_threshold, dtype=np.float32),
                "radius": np.array(radius, dtype=np.int64)})[0]
        durs, mask_n = self.bd2dur.run(None, {"boundaries": bounds, "maskT": mask_t})
        presence, scores = self.estimator.run(None, {
            "x_est": x_est, "boundaries": bounds, "maskT": mask_t, "maskN": mask_n,
            "threshold": np.array(presence_threshold, dtype=np.float32)})
        n = int(mask_n[0].sum())
        return durs[0][:n], presence[0][:n], scores[0][:n]


def _chunks(syllables, total: float, max_seconds: float):
    """Group syllables into chunks of about max_seconds, cut in the middle of gaps (0 = no chunking)."""
    if max_seconds <= 0:
        return [(0.0, total, list(syllables))]
    chunks, start, cur = [], 0.0, []
    for i, s in enumerate(syllables):
        cur.append(s)
        nxt = syllables[i + 1].start if i + 1 < len(syllables) else total
        if nxt - start >= max_seconds or i + 1 == len(syllables):
            end = total if i + 1 == len(syllables) else (s.end + nxt) / 2
            chunks.append((start, end, cur))
            start, cur = end, []
    return chunks


def game_notes_per_syllable(audio_path: str, transcribed_data, language: str, model_dir: str,
                            device: str, boundary_threshold: float, steps: int, chunk_seconds: float,
                            seed: int, radius: int, presence_threshold: float):
    """For each syllable: list of (start, end, midi_float) voiced notes predicted by GAME."""
    log.info(f"Segmenting notes with {blue_highlighted('GAME')} ({os.path.basename(model_dir)})")
    model = GameModel(model_dir, device, seed)
    y, _ = librosa.load(audio_path, sr=model.sr, mono=True)
    total = len(y) / model.sr
    lang = model.language_id(language)
    # Work on timings rounded to 1 ms: fresh transcriptions (float32) and cached JSON (float64)
    # differ by ~1e-7 s, which shifts chunk cuts by a sample and changes GAME's output.
    class _Syl:
        __slots__ = ("orig", "start", "end")

        def __init__(self, orig):
            self.orig, self.start, self.end = orig, round(float(orig.start), 3), round(float(orig.end), 3)

    syllables = [_Syl(s) for s in transcribed_data if round(float(s.end), 3) > round(float(s.start), 3)]
    result = {id(s.orig): [] for s in syllables}

    for c0, c1, chunk in _chunks(syllables, total, chunk_seconds):
        wave = y[int(c0 * model.sr): int(c1 * model.sr)]
        # known regions: gap, syllable, gap, syllable, ..., gap (relative to chunk start)
        edges, regions, cursor = [], [], c0
        for s in chunk:
            a, b = max(s.start, cursor), min(s.end, c1)
            if b <= a:
                continue
            if a > cursor:
                regions.append(a - cursor)
            edges.append((s, a, b))
            regions.append(b - a)
            cursor = b
        if c1 > cursor:
            regions.append(c1 - cursor)
        if not edges:
            continue
        durs, presence, scores = model.run(wave, regions, lang, boundary_threshold, radius, steps, presence_threshold)
        t = c0
        notes = []
        for d, p, sc in zip(durs, presence, scores):
            notes.append((t, t + float(d), bool(p), float(sc)))
            t += float(d)
        for s, a, b in edges:
            for n0, n1, p, sc in notes:
                mid = (n0 + n1) / 2
                if p and a <= mid < b:
                    result[id(s.orig)].append((max(n0, a), min(n1, b), sc))
    return result


def split_syllables_by_game(transcribed_data, notes_per_syllable):
    """Split each syllable at GAME's note boundaries (first part keeps the text, others '~').
    Returns the new transcribed data and the GAME pitch per segment (same order)."""
    import copy
    from modules.Speech_Recognition.TranscribedData import TranscribedData

    new_data, pitches = [], []
    for data in transcribed_data:
        notes = notes_per_syllable.get(id(data), [])
        if len(notes) <= 1:
            new_data.append(data)
            pitches.append(notes[0][2] if notes else None)
            continue
        has_space = str(data.word).endswith(" ")
        for n, (n0, n1, sc) in enumerate(notes):
            seg = copy.deepcopy(data) if n == 0 else TranscribedData()
            seg.start = data.start if n == 0 else n0
            seg.end = data.end if n == len(notes) - 1 else notes[n + 1][0]
            seg.word = (data.word[:-1] if has_space else data.word) if n == 0 else "~"
            seg.is_hyphen = True
            seg.is_word_end = False
            new_data.append(seg)
            pitches.append(sc)
        if has_space:
            new_data[-1].word += " "
            new_data[-1].is_word_end = True
    return new_data, pitches
