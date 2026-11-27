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
                            seed: int, radius: int, presence_threshold: float, gpu_max_chunk_seconds: float):
    """For each syllable: list of (start, end, midi_float) voiced notes predicted by GAME.

    On CUDA, GAME runs in a short-lived subprocess per song: onnxruntime's GPU memory pool only grows
    and memory of earlier sessions was not fully returned, so a long run ran out of memory; a process
    that exits frees everything. (A fresh model per song is needed anyway: the seeded Random ops keep
    their state inside a session, so reusing one would make a song's notes depend on earlier songs.)"""
    log.info(f"Segmenting notes with {blue_highlighted('GAME')} ({os.path.basename(model_dir)})")
    # Work on timings rounded to 1 ms: fresh transcriptions (float32) and cached JSON (float64)
    # differ by ~1e-7 s, which shifts chunk cuts by a sample and changes GAME's output.
    kept = [s for s in transcribed_data if round(float(s.end), 3) > round(float(s.start), 3)]
    timings = [(round(float(s.start), 3), round(float(s.end), 3)) for s in kept]
    args = (audio_path, timings, language, model_dir, device, boundary_threshold, steps, chunk_seconds,
            seed, radius, presence_threshold, gpu_max_chunk_seconds)
    if device == "cuda":
        # GPU memory for the subprocess: release models other modules keep loaded (Whisper); in a
        # song's pipeline GAME runs after Whisper anyway, so keeping it loaded would save nothing
        from modules import gpu_memory
        gpu_memory.release_all()
        per_syllable, warnings = _in_subprocess(args)
    else:
        per_syllable, warnings = notes_for_timings(*args)
    for w in warnings:
        log.warning(w)
    result = {id(s): [] for s in transcribed_data}
    for s, notes in zip(kept, per_syllable):
        result[id(s)] = [tuple(n) for n in notes]
    return result


def _in_subprocess(args):
    import pickle
    import subprocess
    import sys
    import tempfile
    src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in.pkl"), os.path.join(tmp, "out.pkl")
        with open(inp, "wb") as f:
            pickle.dump(args, f)
        env = {**os.environ, "PYTHONPATH": src + os.pathsep + os.environ.get("PYTHONPATH", "")}
        proc = subprocess.run([sys.executable, "-m", "modules.Midi.game_segmenter", inp, out], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(f"GAME subprocess failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
        with open(out, "rb") as f:
            return pickle.load(f)


def notes_for_timings(audio_path, timings, language, model_dir, device, boundary_threshold, steps,
                      chunk_seconds, seed, radius, presence_threshold, gpu_max_chunk_seconds):
    """GAME over the audio with known syllable timings [(start, end)]. Returns (notes per timing as
    [(start, end, midi)], warnings). On CUDA, chunks longer than gpu_max_chunk_seconds run on CPU:
    decided by length only, so the notes never depend on how much GPU memory happened to be free."""
    from modules import gpu_memory
    gpu_memory.free_torch_cache()
    warnings = []
    model = GameModel(model_dir, device, seed)
    cpu_model, long_chunks = None, 0
    y, _ = librosa.load(audio_path, sr=model.sr, mono=True)
    total = len(y) / model.sr
    lang = model.language_id(language)

    class _Syl:
        __slots__ = ("i", "start", "end")

        def __init__(self, i, t):
            self.i, (self.start, self.end) = i, t

    syllables = [_Syl(i, t) for i, t in enumerate(timings)]
    result = [[] for _ in timings]
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
        if len(regions) == 1:
            # one region (a syllable filling the whole chunk) gives zero known boundaries, which
            # GAME's CUDA kernels reject; split it in two (a known boundary mid-syllable)
            warnings.append("FALLBACK: GAME chunk with a single known region split in two")
            regions = [regions[0] / 2, regions[0] / 2]
        on_cpu = device == "cuda" and c1 - c0 > gpu_max_chunk_seconds
        if on_cpu:
            long_chunks += 1
            if cpu_model is None:
                cpu_model = GameModel(model_dir, "cpu", seed)
        try:
            durs, presence, scores = (cpu_model if on_cpu else model).run(
                wave, regions, lang, boundary_threshold, radius, steps, presence_threshold)
        except Exception as e:  # noqa: BLE001  (onnxruntime raises a generic RuntimeException)
            if "Failed to allocate memory" in str(e):
                raise RuntimeError(f"GAME: a {c1 - c0:.0f} s chunk does not fit in GPU memory; lower "
                                   f"segment.game.gpu_max_chunk_seconds ({gpu_max_chunk_seconds:g})") from e
            raise
        t = c0
        notes = []
        for d, p, sc in zip(durs, presence, scores):
            notes.append((t, t + float(d), bool(p), float(sc)))
            t += float(d)
        for s, a, b in edges:
            for n0, n1, p, sc in notes:
                mid = (n0 + n1) / 2
                if p and a <= mid < b:
                    result[s.i].append((max(n0, a), min(n1, b), sc))
    if long_chunks:
        log.info(f"GAME: {long_chunks} chunk(s) longer than {gpu_max_chunk_seconds:g} s ran on CPU")
    return result, warnings


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


if __name__ == "__main__":   # subprocess entry (see _in_subprocess)
    import pickle
    import sys
    with open(sys.argv[1], "rb") as f:
        _args = pickle.load(f)
    _res = notes_for_timings(*_args)
    with open(sys.argv[2], "wb") as f:
        pickle.dump(_res, f)
