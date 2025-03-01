﻿"""Midi creator module"""

import math
import os
from collections import Counter
from Settings import Settings

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


def create_midi_notes_from_pitched_data(start_times: list[float], end_times: list[float], words: list[str], pitched_data: PitchedData) -> list[
    MidiSegment]:
    """Create midi notes from pitched data"""
    print(f"{ULTRASINGER_HEAD} Creating midi_segments")

    midi_segments = []

    for index, start_time in enumerate(start_times):
        end_time = end_times[index]
        word = str(words[index])

        midi_segment = create_midi_note_from_pitched_data(start_time, end_time, pitched_data, word)
        midi_segments.append(midi_segment)

        # todo: Progress?
        # print(filename + " f: " + str(mean))
    return midi_segments


def create_midi_note_from_pitched_data(start_time: float, end_time: float, pitched_data: PitchedData, word: str) -> MidiSegment:
    """Create midi note from pitched data"""

    start = find_nearest_index(pitched_data.times, start_time)
    end = find_nearest_index(pitched_data.times, end_time)

    if start == end:
        freqs = [pitched_data.frequencies[start]]
        confs = [pitched_data.confidence[start]]
    else:
        freqs = pitched_data.frequencies[start:end]
        confs = pitched_data.confidence[start:end]

    conf_f = get_frequencies_with_high_confidence(freqs, confs)

    notes = convert_frequencies_to_notes(conf_f)

    note = most_frequent(notes)[0][0]

    return MidiSegment(note, start_time, end_time, word)


def create_midi_segments_from_transcribed_data(transcribed_data: list[TranscribedData], pitched_data: PitchedData) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    if transcribed_data:
        for i, midi_segment in enumerate(transcribed_data):
            start_times.append(midi_segment.start)
            end_times.append(midi_segment.end)
            words.append(midi_segment.word)
        midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words,
                                                            pitched_data)
        return midi_segments


def create_midi_segments_with_probability(process_data):
    """Create MIDI segments using probabilistic pitch processing."""
    print(f"{ULTRASINGER_HEAD} Creating MIDI segments with {blue_highlighted('word-based pitch processing')}")
    
    # Import here to avoid circular imports
    from modules.Pitcher.pitch_probability import process_with_lyrics, enhance_with_original_audio
    
    # Use both data sources if available
    if hasattr(process_data, 'has_original_pitched_data') and process_data.has_original_pitched_data:
        try:
            # First enhance the pitched data with original audio
            enhanced_data = enhance_with_original_audio(
                process_data.pitched_data, 
                process_data.original_pitched_data,
                process_data.media_info.bpm
            )
            
            # Then create segments based on words, using enhanced pitch data
            midi_segments = process_with_lyrics(
                enhanced_data, 
                process_data.transcribed_data,
                process_data.media_info.bpm
            )
            print(f"{ULTRASINGER_HEAD} Successfully processed with enhanced audio data")
            
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} Error during enhanced processing: {str(e)}")
            print(f"{ULTRASINGER_HEAD} Falling back to standard processing")
            midi_segments = process_with_lyrics(
                process_data.pitched_data,
                process_data.transcribed_data,
                process_data.media_info.bpm
            )
    else:
        # Fall back to using just the separated vocal data
        midi_segments = process_with_lyrics(
            process_data.pitched_data,
            process_data.transcribed_data,
            process_data.media_info.bpm
        )
    
    return midi_segments

def create_repitched_midi_segments_from_ultrastar_txt(pitched_data: PitchedData, ultrastar_txt: UltrastarTxtValue) -> list[MidiSegment]:
    start_times = []
    end_times = []
    words = []

    for i, note_lines in enumerate(ultrastar_txt.UltrastarNoteLines):
        start_times.append(note_lines.startTime)
        end_times.append(note_lines.endTime)
        words.append(note_lines.word)
    midi_segments = create_midi_notes_from_pitched_data(start_times, end_times, words, pitched_data)
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

def attach_lyrics_to_notes(midi_segments, transcribed_data):
    """Attach transcribed lyrics to note segments based on timing."""
    if not transcribed_data or not midi_segments:
        return midi_segments
        
    print(f"{ULTRASINGER_HEAD} Attaching {len(transcribed_data)} words to {len(midi_segments)} notes")
    
    # Create a mapping of time ranges for each note segment
    note_time_ranges = []
    for i, segment in enumerate(midi_segments):
        note_time_ranges.append((segment.start, segment.end, i))
    
    # Create a copy of segments with placeholder words
    result_segments = []
    for segment in midi_segments:
        result_segments.append(MidiSegment(
            note=segment.note,
            start=segment.start,
            end=segment.end,
            word="♪"  # Placeholder
        ))
    
    # Match each word to a note based on timing overlap
    for word_data in transcribed_data:
        word = word_data.word
        word_start = word_data.start
        word_end = word_data.end
        
        best_overlap = 0
        best_index = -1
        
        for start, end, idx in note_time_ranges:
            # Calculate overlap
            overlap_start = max(start, word_start)
            overlap_end = min(end, word_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = idx
        
        if best_index >= 0:
            result_segments[best_index].word = word
    
    return result_segments