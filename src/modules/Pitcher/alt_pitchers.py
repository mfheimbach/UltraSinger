"""RMVPE pitch detector (ONNX): PitchedData on a 10 ms grid.

Needs onnxruntime (already used by SwiftF0); downloads rmvpe.onnx from Hugging Face on first use.
(FCPE was tried and dropped.)
"""

import logging


import librosa
import numpy as np

from modules.console_colors import blue_highlighted
from modules.Pitcher.pitched_data import PitchedData

log = logging.getLogger("ultrasinger.alt_pitchers")

RMVPE_REPO = "lj1995/VoiceConversionWebUI"
RMVPE_FILE = "rmvpe.onnx"


# ---------------------------------------------------------------- RMVPE
# Preprocessing and decoding follow RVC's infer/lib/rmvpe.py:
# 16 kHz, log-mel (128 bins, n_fft/win 1024, hop 160, 30-8000 Hz, HTK, magnitude STFT),
# 360-bin salience (20 cents/bin), local weighted average around the peak.

# Fixed by the model (not tunable): input format and salience bins.
SR, N_FFT, HOP, N_MELS, FMIN, FMAX = 16000, 1024, 160, 128, 30, 8000
FRAME_SECONDS = HOP / SR                 # 10 ms
LOG_FLOOR = 1e-5                         # log-mel floor
PEAK_WINDOW = 9                          # salience bins averaged around the peak (+-4)
FRAMES_MULTIPLE = 32                     # the network needs frame counts in multiples of 32
CHUNK_FRAMES = FRAMES_MULTIPLE * 250     # ~80 s per pass keeps memory bounded (full songs on 8 GB)
_CENTS_MAPPING = 20 * np.arange(360) + 1997.3794084376191


def _rmvpe_mel(y: np.ndarray) -> np.ndarray:
    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", center=True)
    mag = np.abs(stft)
    mel_basis = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=True)
    mel = mel_basis @ mag
    return np.log(np.clip(mel, LOG_FLOOR, None)).astype(np.float32)


def _rmvpe_decode(hidden: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.argmax(hidden, axis=1)
    half = PEAK_WINDOW // 2
    padded = np.pad(hidden, ((0, 0), (half, half)))
    idx = center[:, None] + np.arange(PEAK_WINDOW)[None, :]
    rows = np.arange(hidden.shape[0])[:, None]
    sal = padded[rows, idx]
    cents_pad = np.pad(_CENTS_MAPPING, (half, half))
    cents = cents_pad[idx]
    cents = (sal * cents).sum(1) / np.maximum(sal.sum(1), 1e-9)
    conf = hidden.max(axis=1)
    f0 = 10 * 2 ** (cents / 1200)
    f0[conf <= threshold] = 0.0
    return f0, conf


def get_pitch_with_rmvpe(audio_path: str, device: str, threshold: float) -> PitchedData:
    """RMVPE pitch (10 ms frames). Pitch-shifted extra passes with median voting were tested and
    rejected (worse in the pipeline), so there is one pass."""
    from huggingface_hub import hf_hub_download
    from modules.onnx_session import session as onnx_session

    log.info(f"Pitching with {blue_highlighted('RMVPE')}")
    model_path = hf_hub_download(RMVPE_REPO, RMVPE_FILE)
    session = onnx_session(model_path, device)

    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    f0, conf = _rmvpe_pass(session, y, threshold)
    midi = np.where(f0 > 0, librosa.hz_to_midi(np.maximum(f0, 1e-3)), np.nan)
    f0 = np.where(~np.isnan(midi), librosa.midi_to_hz(np.nan_to_num(midi)), 0.0)
    times = np.arange(len(f0)) * FRAME_SECONDS
    return PitchedData(times.tolist(), f0.astype(float).tolist(), conf.astype(float).tolist())


def _rmvpe_pass(session, y: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mel = _rmvpe_mel(y)  # (128, T)
    n_frames = mel.shape[-1]
    hidden_parts = []
    for start in range(0, n_frames, CHUNK_FRAMES):
        part = mel[:, start:start + CHUNK_FRAMES]
        n = part.shape[-1]
        pad = FRAMES_MULTIPLE * ((n - 1) // FRAMES_MULTIPLE + 1) - n
        if pad:
            part = np.pad(part, ((0, 0), (0, pad)), mode="constant")
        inp = session.get_inputs()[0].name
        out = session.run(None, {inp: part[None]})[0][0]  # (T', 360)
        hidden_parts.append(out[:n])
    hidden = np.concatenate(hidden_parts, axis=0)
    return _rmvpe_decode(hidden, threshold)
