"""Key detection from audio (chroma correlation with major/minor templates)."""


import logging
import librosa
import numpy as np

from modules.console_colors import blue_highlighted

log = logging.getLogger("ultrasinger.key_detector")

# scales (in semitones relative to root note)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def detect_key_from_audio(audio_path: str, analysis_seconds: float) -> tuple[str, str]:
    """
    Detect the key and mode (major/minor) of a song from audio file.

    Args:
        audio_path: Path to audio file

    Returns:
        Tuple of (key_note, mode) e.g., ('C', 'major') or ('A', 'minor')
    """
    log.info("Detecting musical key")

    y, sr = librosa.load(audio_path, sr=None, duration=analysis_seconds)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    chroma_avg = chroma_avg / np.sum(chroma_avg)

    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])  # Major scale pattern
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])  # Minor scale pattern
    major_template = major_template / np.sum(major_template)
    minor_template = minor_template / np.sum(minor_template)

    best_correlation = -1
    best_key = None
    best_mode = None

    # Try all 12 keys
    for shift in range(12):
        shifted_major = np.roll(major_template, shift)
        shifted_minor = np.roll(minor_template, shift)

        major_corr = np.corrcoef(chroma_avg, shifted_major)[0, 1]
        minor_corr = np.corrcoef(chroma_avg, shifted_minor)[0, 1]

        if major_corr > best_correlation:
            best_correlation = major_corr
            best_key = NOTE_NAMES[shift]
            best_mode = 'major'

        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_key = NOTE_NAMES[shift]
            best_mode = 'minor'

    log.info(f"Detected key: {blue_highlighted(best_key)} {blue_highlighted(best_mode)}")
    return best_key, best_mode
