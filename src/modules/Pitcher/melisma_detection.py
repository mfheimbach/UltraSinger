"""Melisma detection for UltraSinger.

This module provides functions to detect melismas (vocal runs) in singing,
where a single syllable spans multiple notes.
"""

import re
from typing import List, Optional, Dict, Tuple

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def is_continuation_marker(word: str) -> bool:
    """
    Check if a word is a continuation marker.
    
    Args:
        word: Word text to check
        
    Returns:
        True if the word is a continuation marker
    """
    return word.strip() == "~" or word.startswith("~")


def map_words_to_notes(
    transcribed_data: List[TranscribedData],
    midi_segments: List[MidiSegment]
) -> Dict[int, List[int]]:
    """
    Map words in transcribed data to corresponding notes.
    
    Args:
        transcribed_data: List of transcribed words with timing
        midi_segments: List of MIDI note segments
        
    Returns:
        Dictionary mapping word indices to lists of note indices
    """
    word_to_notes = {}
    
    for word_idx, word_data in enumerate(transcribed_data):
        notes_for_word = []
        
        # Find all notes that overlap with this word's timespan
        for note_idx, segment in enumerate(midi_segments):
            # Check for overlap
            if (segment.start < word_data.end and segment.end > word_data.start):
                notes_for_word.append(note_idx)
        
        word_to_notes[word_idx] = notes_for_word
    
    return word_to_notes


def detect_rapid_pitch_changes(
    midi_segments: List[MidiSegment],
    bpm: float,
    threshold: float = 0.15
) -> List[Tuple[int, int]]:
    """
    Detect sequences of rapid pitch changes that might indicate melismas.
    
    Args:
        midi_segments: List of MIDI segments
        bpm: Beats per minute
        threshold: Duration threshold for "rapid" notes in seconds
        
    Returns:
        List of (start_idx, end_idx) tuples for rapid note sequences
    """
    if not midi_segments or len(midi_segments) < 3:
        return []
    
    # Calculate eighth note duration as reference
    eighth_note_duration = 60.0 / bpm / 2
    
    # Adjust threshold based on tempo if needed
    adjusted_threshold = min(threshold, eighth_note_duration * 0.8)
    
    # Find sequences of rapid notes
    rapid_sequences = []
    current_sequence = []
    
    for i, segment in enumerate(midi_segments):
        duration = segment.end - segment.start
        
        if duration < adjusted_threshold:
            if not current_sequence:
                current_sequence = [i]
            else:
                current_sequence.append(i)
        else:
            if len(current_sequence) >= 3:  # Need at least 3 notes to be a melisma
                rapid_sequences.append((current_sequence[0], current_sequence[-1]))
            current_sequence = []
    
    # Handle the last sequence
    if len(current_sequence) >= 3:
        rapid_sequences.append((current_sequence[0], current_sequence[-1]))
    
    return rapid_sequences


def detect_melismas(
    midi_segments: List[MidiSegment],
    bpm: float,
    transcribed_data: Optional[List[TranscribedData]] = None,
    settings: Optional[object] = None
) -> List[bool]:
    """
    Detect melismas (vocal runs) in a sequence of notes.
    
    Args:
        midi_segments: List of MIDI note segments
        bpm: Beats per minute
        transcribed_data: Optional list of transcribed words with timing
        settings: Optional settings object
        
    Returns:
        List of booleans indicating whether each note is part of a melisma
    """
    if not midi_segments:
        return []
    
    # Initialize result array
    is_melisma = [False] * len(midi_segments)
    
    print(f"{ULTRASINGER_HEAD} Detecting melismas in {len(midi_segments)} notes")
    
    # Method 1: Use transcription data if available
    if transcribed_data:
        word_to_notes = map_words_to_notes(transcribed_data, midi_segments)
        
        # Mark notes as melisma if multiple notes share the same syllable
        for word_idx, note_indices in word_to_notes.items():
            # Skip words with continuation markers as they're part of the previous word
            if word_idx < len(transcribed_data) and not is_continuation_marker(transcribed_data[word_idx].word):
                if len(note_indices) > 1:
                    # Multiple notes for one syllable = melisma
                    for note_idx in note_indices:
                        if note_idx < len(is_melisma):
                            is_melisma[note_idx] = True
    
    # Method 2: Look for continuation markers in note words
    for i, segment in enumerate(midi_segments):
        if is_continuation_marker(segment.word):
            is_melisma[i] = True
            # Also mark the previous note if it exists
            if i > 0:
                is_melisma[i-1] = True
    
    # Method 3: Look for rapid note sequences
    rapid_sequences = detect_rapid_pitch_changes(midi_segments, bpm)
    for start_idx, end_idx in rapid_sequences:
        for i in range(start_idx, end_idx + 1):
            is_melisma[i] = True
    
    # Method 4: Look for sequences with the same text
    for i in range(1, len(midi_segments) - 1):
        # If notes have the same text and they're short, they might be a melisma
        if (midi_segments[i].word == midi_segments[i-1].word and 
            midi_segments[i].word == midi_segments[i+1].word):
            
            # Check if they're relatively short notes
            if (midi_segments[i].end - midi_segments[i].start < 0.2 and
                midi_segments[i-1].end - midi_segments[i-1].start < 0.2 and
                midi_segments[i+1].end - midi_segments[i+1].start < 0.2):
                
                is_melisma[i-1] = True
                is_melisma[i] = True
                is_melisma[i+1] = True
    
    melisma_count = sum(1 for x in is_melisma if x)
    print(f"{ULTRASINGER_HEAD} Detected {melisma_count} notes as part of melismas")
    
    return is_melisma