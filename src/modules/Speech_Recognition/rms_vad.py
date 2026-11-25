"""RMS-energy voice activity detection for separated vocals (Syed et al. 2025, "Exploiting Music
Source Separation for Automatic Lyrics Transcription with Whisper"): RMS normalised by its
maximum over the whole file, threshold 0.1, gaps under 1 s bridged; segments over 30 s are
split at their quietest frame (Whisper sees at most 30 s)."""

import numpy as np
from whisperx.diarize import Segment as SegmentX
from whisperx.vads.vad import Vad


class RmsVad(Vad):
    def __init__(self, threshold: float = 0.1, min_silence: float = 1.0, frame: float = 0.025, hop: float = 0.01,
                 max_len: float = 30.0):
        super().__init__(0.5)
        self.threshold, self.min_silence, self.frame, self.hop, self.max_len = threshold, min_silence, frame, hop, max_len

    def __call__(self, audio, **kwargs):
        wav = audio["waveform"]
        wav = (wav.numpy() if hasattr(wav, "numpy") else np.asarray(wav)).reshape(-1)
        sr = audio["sample_rate"]
        n, h = int(self.frame * sr), int(self.hop * sr)
        if len(wav) < n:
            return []
        frames = np.lib.stride_tricks.sliding_window_view(wav, n)[::h]
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        active = rms / (rms.max() + 1e-12) > self.threshold
        segments, start, last = [], None, None
        for i, a in enumerate(active):
            if not a:
                continue
            t = i * self.hop
            if start is None:
                start = t
            elif t - last > self.min_silence:
                segments.append((start, last + self.frame))
                start = t
            last = t
        if start is not None:
            segments.append((start, last + self.frame))
        out, todo = [], list(segments)
        while todo:
            s, e = todo.pop(0)
            if e - s <= self.max_len:
                out.append((s, e))
                continue
            a, b = int((s + 0.25 * (e - s)) / self.hop), int((s + 0.75 * (e - s)) / self.hop)
            cut = (a + int(np.argmin(rms[a:b]))) * self.hop
            todo[:0] = [(s, cut), (cut, e)]
        return [SegmentX(s, e, "UNKNOWN") for s, e in out]

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments_list, chunk_size, onset: float = 0.5, offset=None):
        if not segments_list:
            return []
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
