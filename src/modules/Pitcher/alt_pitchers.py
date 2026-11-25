"""Alternative pitch detectors: RMVPE (ONNX) and FCPE (torchfcpe).

Both return PitchedData on a 10 ms grid, like SwiftF0's output is consumed.
Optional dependencies: RMVPE needs onnxruntime (already used by SwiftF0) and
downloads rmvpe.onnx from Hugging Face on first use; FCPE needs `pip install torchfcpe`.
"""

import os

import librosa
import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
from modules.Pitcher.pitched_data import PitchedData

RMVPE_REPO = "lj1995/VoiceConversionWebUI"
RMVPE_FILE = "rmvpe.onnx"


# ---------------------------------------------------------------- RMVPE
# Preprocessing and decoding follow RVC's infer/lib/rmvpe.py:
# 16 kHz, log-mel (128 bins, n_fft/win 1024, hop 160, 30-8000 Hz, HTK, magnitude STFT),
# 360-bin salience (20 cents/bin), local weighted average around the peak.

_CENTS_MAPPING = 20 * np.arange(360) + 1997.3794084376191


def _rmvpe_mel(y: np.ndarray) -> np.ndarray:
    stft = librosa.stft(y, n_fft=1024, hop_length=160, win_length=1024, window="hann", center=True)
    mag = np.abs(stft)
    mel_basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    mel = mel_basis @ mag
    return np.log(np.clip(mel, 1e-5, None)).astype(np.float32)


def _rmvpe_decode(hidden: np.ndarray, threshold: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    center = np.argmax(hidden, axis=1)
    padded = np.pad(hidden, ((0, 0), (4, 4)))
    idx = center[:, None] + np.arange(9)[None, :]
    rows = np.arange(hidden.shape[0])[:, None]
    sal = padded[rows, idx]
    cents_pad = np.pad(_CENTS_MAPPING, (4, 4))
    cents = cents_pad[idx]
    cents = (sal * cents).sum(1) / np.maximum(sal.sum(1), 1e-9)
    conf = hidden.max(axis=1)
    f0 = 10 * 2 ** (cents / 1200)
    f0[conf <= threshold] = 0.0
    return f0, conf


def get_pitch_with_rmvpe(audio_path: str, device: str = "cpu", shifts: tuple[float, ...] = (0,)) -> PitchedData:
    """RMVPE pitch. With several shifts (semitones), the audio is also pitch-shifted, each pass is
    shifted back, and the per-frame median is taken (frames voiced in most passes)."""
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    print(f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('RMVPE')}"
          + (f" (pitch-shift passes {list(shifts)})" if tuple(shifts) != (0,) else ""))
    model_path = hf_hub_download(RMVPE_REPO, RMVPE_FILE)
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    passes = []
    for shift in shifts:
        ys = y if shift == 0 else librosa.effects.pitch_shift(y, sr=16000, n_steps=shift)
        f0, conf = _rmvpe_pass(session, ys)
        midi = np.where(f0 > 0, librosa.hz_to_midi(np.maximum(f0, 1e-3)) - shift, np.nan)
        passes.append((midi, conf))
    n = min(len(m) for m, _ in passes)
    midi = np.vstack([m[:n] for m, _ in passes])
    conf = np.median(np.vstack([c[:n] for _, c in passes]), axis=0)
    voiced = np.sum(~np.isnan(midi), axis=0) * 2 > len(passes)
    with np.errstate(all="ignore"):
        med = np.nanmedian(midi, axis=0)
    f0 = np.where(voiced, librosa.midi_to_hz(np.nan_to_num(med)), 0.0)
    times = np.arange(n) * 0.01
    return PitchedData(times.tolist(), f0.astype(float).tolist(), conf.astype(float).tolist())


def _rmvpe_pass(session, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mel = _rmvpe_mel(y)  # (128, T)
    n_frames = mel.shape[-1]
    hidden_parts = []
    chunk = 32 * 250  # ~80 s per pass keeps memory bounded (full songs on 8 GB)
    for start in range(0, n_frames, chunk):
        part = mel[:, start:start + chunk]
        n = part.shape[-1]
        pad = 32 * ((n - 1) // 32 + 1) - n
        if pad:
            part = np.pad(part, ((0, 0), (0, pad)), mode="constant")
        inp = session.get_inputs()[0].name
        out = session.run(None, {inp: part[None]})[0][0]  # (T', 360)
        hidden_parts.append(out[:n])
    hidden = np.concatenate(hidden_parts, axis=0)
    return _rmvpe_decode(hidden)


# ---------------------------------------------------------------- FCPE

def get_pitch_with_fcpe(audio_path: str, device: str = "cpu") -> PitchedData:
    import torch
    from torchfcpe import spawn_bundled_infer_model

    print(f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('FCPE')}")
    model = spawn_bundled_infer_model(device=device)
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    n_frames = len(y) // 160 + 1
    audio = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        f0 = model.infer(audio, sr=16000, decoder_mode="local_argmax", threshold=0.006,
                         f0_min=50, f0_max=1100, interp_uv=False,
                         output_interp_target_length=n_frames)
    f0 = f0.squeeze().cpu().numpy().astype(float)
    conf = (f0 > 0).astype(float)  # FCPE's voiced/unvoiced decision as confidence
    times = np.arange(len(f0)) * 0.01
    return PitchedData(times.tolist(), f0.tolist(), conf.tolist())
