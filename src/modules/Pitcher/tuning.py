"""Global tuning deviation of a pitch track."""

import librosa
import numpy as np

from modules.Pitcher.pitched_data import PitchedData


def estimate_tuning_offset(pitched_data: PitchedData, conf_threshold: float, min_frames: int) -> float:
    """Global tuning deviation in semitones (-0.5..0.5), circular mean over confident frames."""
    f = np.asarray(pitched_data.frequencies, dtype=float)
    c = np.asarray(pitched_data.confidence, dtype=float)
    m = (c > conf_threshold) & (f > 0)
    if m.sum() < min_frames:
        return 0.0
    frac = librosa.hz_to_midi(f[m]) % 1.0
    angle = np.angle(np.mean(np.exp(2j * np.pi * frac)))
    return float(angle / (2 * np.pi))
