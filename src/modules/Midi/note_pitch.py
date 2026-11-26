"""One pitch per syllable piece from a pitch track (notes.from_pitch)."""

import logging
import math
from collections import Counter
from dataclasses import dataclass

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.tuning import estimate_tuning_offset
from modules.Speech_Recognition.TranscribedData import TranscribedData

log = logging.getLogger("ultrasinger.note_pitch")

MIDI_OCTAVES = 11   # MIDI pitch range 0..131 for the octave search (definition, not tunable)
_FALLBACKS = {"low_confidence": 0, "unvoiced": 0}


@dataclass
class NotePitchOptions:
    """How pitch frames inside a note are combined into one note (values from the preset)."""
    method: str              # mode | median | chroma_mode
    conf_threshold: float
    edge_trim: float         # fraction of frames dropped at each end (attack/release, slides)
    tuning_correction: bool  # compensate recordings not tuned to A440
    tuning_min_frames: int   # confident frames needed to estimate the tuning offset
    fmin: float              # frames outside fmin..fmax (Hz) are ignored
    fmax: float


def note_from_frames(freqs, confs, options: NotePitchOptions, tuning_offset: float) -> tuple[str, float]:
    """Returns the note name and the unrounded MIDI pitch (median of the used frames)."""
    f = np.asarray(freqs, dtype=float)
    c = np.asarray(confs, dtype=float)
    keep = (c > options.conf_threshold) & (f > options.fmin) & (f < options.fmax) & (f > 0)
    if not keep.any():
        # FALLBACK (counted and reported per run by create_midi_notes_from_pitched_data)
        _FALLBACKS["low_confidence"] += 1
        keep = f > 0
        if not keep.any():
            _FALLBACKS["unvoiced"] += 1
            return ("C4", None) if not len(freqs) or freqs[0] <= 0 else (librosa.hz_to_note(float(freqs[0])), None)
    idx = np.flatnonzero(keep)
    if options.edge_trim > 0 and idx.size > 2:
        cut = int(idx.size * options.edge_trim)
        if idx.size - 2 * cut >= 1:
            idx = idx[cut: idx.size - cut]
    m = librosa.hz_to_midi(f[idx]) - tuning_offset
    w = c[idx]
    if options.method == "median":
        midi = int(np.round(np.median(m)))
    elif options.method == "chroma_mode":
        r = np.round(m).astype(int)
        chroma = int(np.bincount(r % 12, weights=w, minlength=12).argmax())
        med = np.median(m)
        midi = int(min((o * 12 + chroma for o in range(MIDI_OCTAVES)), key=lambda x: abs(x - med)))
    else:  # mode: identical to the original Counter over note names
        midi = Counter(np.round(m).astype(int).tolist()).most_common(1)[0][0]
    return librosa.midi_to_note(midi, unicode=False), float(np.median(m))


def find_nearest_index(array: list[float], value: float) -> int:
    """Nearest index in array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1

    return idx


def notes_from_pitch(syllables: list[TranscribedData], pitched_data: PitchedData,
                     options: NotePitchOptions) -> list[MidiSegment]:
    """One MidiSegment per syllable: the pitch of its frames (options), unquantized; the unrounded
    pitch is kept on the note for notes.quantize_key."""
    tuning_offset = 0.0
    if options.tuning_correction:
        tuning_offset = estimate_tuning_offset(pitched_data, options.conf_threshold, options.tuning_min_frames)
        log.info(f"Tuning offset {tuning_offset * 100:+.0f} cents")
    _FALLBACKS.update(low_confidence=0, unvoiced=0)
    notes = []
    for s in syllables:
        start = find_nearest_index(pitched_data.times, s.start)
        end = find_nearest_index(pitched_data.times, s.end)
        if start == end:
            freqs, confs = [pitched_data.frequencies[start]], [pitched_data.confidence[start]]
        else:
            freqs, confs = pitched_data.frequencies[start:end], pitched_data.confidence[start:end]
        note, pitch_hint = note_from_frames(freqs, confs, options, tuning_offset)
        notes.append(MidiSegment(note, s.start, s.end, str(s.word), pitch=pitch_hint))
    if _FALLBACKS["low_confidence"]:
        log.warning(f"FALLBACK: {_FALLBACKS['low_confidence']} of {len(notes)} notes had no confident "
                    f"pitch frames and used all voiced frames; {_FALLBACKS['unvoiced']} had none and got C4")
    return notes
