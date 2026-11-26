"""Pitcher module"""

import logging
import numpy as np

from scipy.io import wavfile
from swift_f0 import SwiftF0

from modules.console_colors import blue_highlighted
from modules.Pitcher.pitched_data import PitchedData

log = logging.getLogger("ultrasinger.pitcher")

_detectors: dict = {}


def _get_detector(fmin: float, fmax: float, confidence_threshold: float):
    """SwiftF0 detector per setting (lazy). Music/speech range 46.875-2093.75 Hz; speech only 65-400."""
    key = (fmin, fmax, confidence_threshold)
    if key not in _detectors:
        _detectors[key] = SwiftF0(fmin=fmin, fmax=fmax, confidence_threshold=confidence_threshold)
    return _detectors[key]


def get_pitch_with_file(filename: str, fmin: float, fmax: float, confidence_threshold: float) -> PitchedData:
    """Pitch detection using SwiftF0"""

    log.info(f"Pitching with {blue_highlighted('SwiftF0')}"
    )
    sample_rate, audio = wavfile.read(filename)

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Normalize audio to float32 based on dtype
    if audio.dtype == np.uint8:
        # uint8: range [0, 255] -> subtract 128 and divide by 128
        audio = (audio.astype(np.float32) - 128.0) / 128.0   # format definition, not tunable
    elif audio.dtype in [np.int16, np.int32, np.int64]:
        # Signed integers: use iinfo to get max value and normalize
        dtype_info = np.iinfo(audio.dtype)
        max_val = max(abs(dtype_info.min), abs(dtype_info.max))
        audio = audio.astype(np.float32) / float(max_val)
    elif audio.dtype == np.float64:
        # float64: cast to float32
        audio = audio.astype(np.float32)
    elif audio.dtype != np.float32:
        # Fallback for other types: assume int16 range
        audio = audio.astype(np.float32) / 32768.0   # int16 full scale

    return get_pitch_with_swift_f0(audio, sample_rate, fmin, fmax, confidence_threshold)


def get_pitch_with_swift_f0(
    audio: np.ndarray, sample_rate: int, fmin: float, fmax: float, confidence_threshold: float
) -> PitchedData:
    """Pitch detection using SwiftF0

    SwiftF0 processes audio at 16kHz with 256-sample hop size internally.
    Returns frames at approximately 62.5 ms intervals.
    """
    detector = _get_detector(fmin, fmax, confidence_threshold)

    # Detect pitch
    result = detector.detect_from_array(audio, sample_rate)

    # Convert to PitchedData format
    times = [float(t) for t in result.timestamps]
    frequencies = [float(f) for f in result.pitch_hz]
    confidence = [float(c) for c in result.confidence]

    return PitchedData(times, frequencies, confidence)
