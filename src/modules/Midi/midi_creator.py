"""Midi creator module"""

import math
import os
from collections import Counter

import librosa
import numpy as np
import pretty_midi
import unidecode

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import (
    ULTRASINGER_HEAD, blue_highlighted,
)
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Audio.key_detector import quantize_note_to_key
from dataclasses import dataclass


@dataclass
class NotePitchOptions:
    """How pitch frames inside a note are combined into one note.

    The defaults reproduce the original behaviour (most frequent note name
    over frames with confidence > 0.4)."""
    method: str = "mode"            # mode | median | chroma_mode
    conf_threshold: float = 0.4
    edge_trim: float = 0.0          # fraction of frames dropped at each end (attack/release, slides)
    tuning_correction: bool = False  # compensate recordings not tuned to A440
    fmin: float = 0.0
    fmax: float = 1e9


def estimate_tuning_offset(pitched_data: PitchedData, conf_threshold: float = 0.4) -> float:
    """Global tuning deviation in semitones (-0.5..0.5), circular mean over confident frames."""
    f = np.asarray(pitched_data.frequencies, dtype=float)
    c = np.asarray(pitched_data.confidence, dtype=float)
    m = (c > conf_threshold) & (f > 0)
    if m.sum() < 10:
        return 0.0
    frac = librosa.hz_to_midi(f[m]) % 1.0
    angle = np.angle(np.mean(np.exp(2j * np.pi * frac)))
    return float(angle / (2 * np.pi))


def note_from_frames(freqs, confs, options: NotePitchOptions, tuning_offset: float = 0.0) -> tuple[str, float]:
    """Returns the note name and the unrounded MIDI pitch (median of the used frames)."""
    f = np.asarray(freqs, dtype=float)
    c = np.asarray(confs, dtype=float)
    keep = (c > options.conf_threshold) & (f > options.fmin) & (f < options.fmax) & (f > 0)
    if not keep.any():
        keep = f > 0
        if not keep.any():
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
        midi = int(min((o * 12 + chroma for o in range(11)), key=lambda x: abs(x - med)))
    else:  # mode: identical to the original Counter over note names
        midi = Counter(np.round(m).astype(int).tolist()).most_common(1)[0][0]
    return librosa.midi_to_note(midi, unicode=False), float(np.median(m))

def create_midi_instrument(midi_segments: list[MidiSegment]) -> object:
    """Converts an Ultrastar data to a midi instrument"""

    print(f"{ULTRASINGER_HEAD} Creating midi instrument")

    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    velocity = 100

    for i, midi_segment in enumerate(midi_segments):
        note = pretty_midi.Note(velocity, librosa.note_to_midi(midi_segment.note), midi_segment.start, midi_segment.end)
        instrument.notes.append(note)

    return instrument

def sanitize_for_midi(text):
    """
    Sanitize text for MIDI compatibility.
    Uses unidecode to approximate characters to ASCII.
    """
    return unidecode.unidecode(text)

def __create_midi(instruments: list[object], bpm: float, midi_output: str, midi_segments: list[MidiSegment]) -> None:
    """Write instruments to midi file"""

    print(f"{ULTRASINGER_HEAD} Creating midi file -> {midi_output}")

    midi_data = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for i, midi_segment in enumerate(midi_segments):
        sanitized_word = sanitize_for_midi(midi_segment.word)
        midi_data.lyrics.append(pretty_midi.Lyric(text=sanitized_word, time=midi_segment.start))
    for instrument in instruments:
        midi_data.instruments.append(instrument)
    midi_data.write(midi_output)


class MidiCreator:
    """Docstring"""


def convert_frequencies_to_notes(frequency: [str]) -> list[list[str]]:
    """Converts frequencies to notes"""
    notes = []
    for freq in frequency:
        notes.append(librosa.hz_to_note(float(freq)))
    return notes


def most_frequent(array: [str]) -> list[tuple[str, int]]:
    """Get most frequent item in array"""
    return Counter(array).most_common(1)


def find_nearest_index(array: list[float], value: float) -> int:
    """Nearest index in array"""
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (
        idx == len(array)
        or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])
    ):
        return idx - 1

    return idx


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], words: list[str],
                                         pitched_data: PitchedData, allowed_notes: set[str] = None,
                                         options: NotePitchOptions = None) -> list[MidiSegment]:
    """Create midi notes from pitched data

    Args:
        start_times: List of start times
        end_times: List of end times
        words: List of words/syllables
        pitched_data: Pitched data containing frequencies and confidence
        allowed_notes: Optional set of allowed note names for key quantization

    Returns:
        List of MidiSegments
    """
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    tuning_offset = 0.0
    if options is not None and options.tuning_correction:
        tuning_offset = estimate_tuning_offset(pitched_data, options.conf_threshold)
        print(f"{ULTRASINGER_HEAD} Tuning offset {tuning_offset * 100:+.0f} cents")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])

        midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word, allowed_notes,
                                                          options, tuning_offset)
        midi_segments.append(midi_segment)

        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_segments


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str,
                                        allowed_notes: set[str] = None, options: NotePitchOptions = None,
                                        tuning_offset: float = 0.0) -> MidiSegment:
    """Create midi note from pitched data

    Args:
        start_time: Start time of the note
        end_time: End time of the note
        pitched_data: Pitched data containing frequencies and confidence
        word: The word/syllable for this note
        allowed_notes: Optional set of allowed note names (without octave) for key quantization

    Returns:
        One MidiSegment
    """

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start == end:
        freqs = [pitched_data.frequencies[start]]
        confs = [pitched_data.confidence[start]]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    pitch_hint = None
    if options is None:
        conf_f = get_frequencies_with_high_confidence(freqs, confs)
        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]
    else:
        note, pitch_hint = note_from_frames(freqs, confs, options, tuning_offset)

    if allowed_notes is not None:
        note = quantize_note_to_key(note, allowed_notes, pitch_hint)

    return MidiSegment(note, start_time, end_time, word)


def create_midi_segments_from_transcribed_data(transcribed_data: list[TranscribedData], pitched_data: PitchedData,
                                                allowed_notes: set[str] = None,
                                                options: NotePitchOptions = None) -> list[MidiSegment]:
    """Create MIDI segments from transcribed data

    Args:
        transcribed_data: List of transcribed data segments
        pitched_data: Pitched data containing frequencies and confidence
        allowed_notes: Optional set of allowed note names for key quantization

    Returns:
        List of MidiSegments
    """
    start_times = []
    end_times = []
    words = []

    if transcribed_data:
        for i, midi_segment in enumerate(transcribed_data):
            start_times.append(midi_segment.start)
            end_times.append(midi_segment.end)
            words.append(midi_segment.word)
        midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words,
                                                            pitched_data, allowed_notes, options)
        return midi_segments


def create_repitched_midi_segments_from_ultrastar_txt(pitched_data: PitchedData, ultrastar_txt: UltrastarTxtValue,
                                                      options: NotePitchOptions = None) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    for i, note_lines in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_times.append(note_lines.startTime)
        end_times.append(note_lines.endTime)
        words.append(note_lines.word)
    midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words, pitched_data, options=options)
    return midi_segments


def create_midi_file(
        real_bpm: float,
        song_output: str,
        midi_segments: list[MidiSegment],
        basename_without_ext: str,
) -> None:
    """Create midi file"""
    print(f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}")

    voice_instrument = [
        create_midi_instrument(midi_segments)
    ]

    midi_output = os.path.join(song_output, f"{basename_without_ext}.mid")
    __create_midi(voice_instrument, real_bpm, midi_output, midi_segments)
