"""Note significance scoring for UltraSinger.

This module provides functions to calculate the musical significance of notes
for improved note regularization and merging decisions.
"""

import numpy as np
import librosa
import re
from typing import List, Optional, Tuple, Dict

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Pitcher.pitched_data import PitchedData
from modules.Midi.note_length_calculator import (
    get_thirtytwo_note_second,
    get_sixteenth_note_second,
    get_eighth_note_second,
    get_quarter_note_second
)
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted


def calculate_semitone_difference(note1: str, note2: str) -> float:
    """
    Calculate the semitone difference between two notes.
    
    Args:
        note1: First note (e.g., 'C4')
        note2: Second note (e.g., 'D4')
        
    Returns:
        Absolute semitone difference
    """
    # Extract note name and octave
    pattern = r'([A-G][#b]?)(\d+)'
    match1 = re.match(pattern, note1)
    match2 = re.match(pattern, note2)
    
    if not match1 or not match2:
        return 0.0
    
    note_name1, octave1 = match1.groups()
    note_name2, octave2 = match2.groups()
    
    octave1 = int(octave1)
    octave2 = int(octave2)
    
    # Define semitone positions for each note
    note_values = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 
                   'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 
                   'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11}
    
    # Calculate absolute semitone position
    semitones1 = octave1 * 12 + note_values.get(note_name1, 0)
    semitones2 = octave2 * 12 + note_values.get(note_name2, 0)
    
    # Calculate difference
    return abs(semitones1 - semitones2)


def calculate_duration_significance(segment: MidiSegment, bpm: float) -> float:
    """
    Calculate the significance score based on note duration.
    
    Args:
        segment: The note segment to evaluate
        bpm: Beats per minute of the song
        
    Returns:
        Score between 0.0 and 1.0 where higher values indicate more significant
    """
    duration = segment.end - segment.start
    
    # Reference durations in seconds based on BPM
    thirty_second = get_thirtytwo_note_second(bpm)
    sixteenth = get_sixteenth_note_second(bpm)
    eighth = get_eighth_note_second(bpm)
    quarter = get_quarter_note_second(bpm)
    
    # Very short notes (less than a 32nd note) have low significance
    if duration < thirty_second * 0.8:
        return 0.1
    
    # Score increases with duration up to a quarter note
    if duration < sixteenth:
        # Map 32nd to 16th note range to 0.1-0.3
        return 0.1 + 0.2 * ((duration - thirty_second) / (sixteenth - thirty_second))
    elif duration < eighth:
        # Map 16th to 8th note range to 0.3-0.6
        return 0.3 + 0.3 * ((duration - sixteenth) / (eighth - sixteenth))
    elif duration < quarter:
        # Map 8th to quarter note range to 0.6-0.9
        return 0.6 + 0.3 * ((duration - eighth) / (quarter - eighth))
    else:
        # Quarter note or longer is highly significant
        return 0.9 + min(0.1, 0.1 * (duration - quarter) / quarter)


def calculate_pitch_change_significance(
    segment: MidiSegment,
    prev_segment: Optional[MidiSegment],
    next_segment: Optional[MidiSegment]
) -> float:
    """
    Calculate significance based on pitch changes to and from this note.
    
    Args:
        segment: The current note segment
        prev_segment: The previous note segment (or None)
        next_segment: The next note segment (or None)
        
    Returns:
        Score between 0.0 and 1.0 where higher values indicate more significant
    """
    max_diff = 0.0
    
    # Check pitch difference with previous note
    if prev_segment:
        prev_diff = calculate_semitone_difference(segment.note, prev_segment.note)
        max_diff = max(max_diff, prev_diff)
    
    # Check pitch difference with next note
    if next_segment:
        next_diff = calculate_semitone_difference(segment.note, next_segment.note)
        max_diff = max(max_diff, next_diff)
    
    # Normalize to [0, 1] range
    # Notes with large interval jumps (>= 5 semitones) are very significant
    return min(1.0, max_diff / 5.0)


def calculate_beat_alignment(segment: MidiSegment, bpm: float) -> float:
    """
    Calculate how well a note aligns with the beat structure.
    
    Args:
        segment: The note segment to evaluate
        bpm: Beats per minute of the song
        
    Returns:
        Score between 0.0 and 1.0 where higher values indicate better alignment
    """
    # Calculate beat duration in seconds
    beat_duration = 60.0 / bpm
    
    # Find the nearest beat to the start of the segment
    beats_since_start = segment.start / beat_duration
    nearest_beat_time = round(beats_since_start) * beat_duration
    
    # Calculate distance to nearest beat in seconds
    distance = abs(segment.start - nearest_beat_time)
    
    # Notes starting very close to a beat get high significance
    # Normalize to [0, 1] range where 0 means half a beat away and 1 means perfectly aligned
    max_distance = beat_duration / 2
    
    if distance > max_distance:
        return 0.0
    
    return 1.0 - (distance / max_distance)


def calculate_vad_confidence(
    segment: MidiSegment,
    vad_timestamps: np.ndarray,
    vad_scores: np.ndarray
) -> float:
    """
    Calculate the average VAD confidence over the note's duration.
    
    Args:
        segment: The note segment to evaluate
        vad_timestamps: Array of VAD timestamps
        vad_scores: Array of VAD confidence scores
        
    Returns:
        Score between 0.0 and 1.0 where higher values indicate higher confidence
    """
    if len(vad_timestamps) == 0 or len(vad_scores) == 0:
        return 0.5  # Default mid-value if no VAD data
    
    # Find all VAD scores that fall within the segment timespan
    start_idx = np.searchsorted(vad_timestamps, segment.start) - 1
    end_idx = np.searchsorted(vad_timestamps, segment.end)
    
    # Handle boundary cases
    start_idx = max(0, start_idx)
    end_idx = min(len(vad_timestamps) - 1, end_idx)
    
    if start_idx >= end_idx:
        # If no VAD points fall within the segment, use the nearest point
        nearest_idx = start_idx if start_idx < len(vad_timestamps) else end_idx
        return vad_scores[nearest_idx]
    
    # Calculate weighted average of VAD scores within the segment
    relevant_timestamps = vad_timestamps[start_idx:end_idx+1]
    relevant_scores = vad_scores[start_idx:end_idx+1]
    
    # Calculate time weights (longer coverage = higher weight)
    weights = np.ones_like(relevant_scores)
    for i in range(len(relevant_timestamps) - 1):
        curr_time = max(relevant_timestamps[i], segment.start)
        next_time = min(relevant_timestamps[i+1], segment.end)
        weights[i] = next_time - curr_time
    
    # Last point may be outside the segment
    if relevant_timestamps[-1] < segment.end:
        weights[-1] = segment.end - relevant_timestamps[-1]
    
    # Normalize weights
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
        return float(np.sum(relevant_scores * weights))
    
    return 0.5  # Default if something went wrong


def calculate_word_boundary_score(segment: MidiSegment, transcribed_data: List[TranscribedData]) -> float:
    """
    Calculates significance based on alignment with word boundaries.
    
    Args:
        segment: The note segment to evaluate
        transcribed_data: List of transcribed words with timing
        
    Returns:
        Score between 0.0 and 1.0 where higher values indicate better alignment
    """
    if not transcribed_data:
        return 0.5  # Default mid-value
    
    # Find all transcribed words that overlap with this segment
    overlapping_words = []
    for word_data in transcribed_data:
        # Check for overlap
        if (segment.start <= word_data.end and segment.end >= word_data.start):
            overlapping_words.append(word_data)
    
    if not overlapping_words:
        return 0.0  # No overlapping words found
    
    # Calculate boundary scores
    max_score = 0.0
    for word_data in overlapping_words:
        # Check if segment starts or ends near word boundary
        start_dist = abs(segment.start - word_data.start)
        end_dist = abs(segment.end - word_data.end)
        
        # Time tolerance for boundary matching (100ms)
        tolerance = 0.1
        
        if start_dist < tolerance or end_dist < tolerance:
            # Note aligns with word boundary
            boundary_score = 1.0 - min(start_dist, end_dist) / tolerance
            max_score = max(max_score, boundary_score)
        
        # Special case: continuation marker
        if word_data.word.strip() == "~":
            # Lower score for continuation markers unless they contain a pitch change
            max_score = max(max_score, 0.3)
    
    return max_score


def calculate_note_significance(
    segment: MidiSegment,
    prev_segment: Optional[MidiSegment],
    next_segment: Optional[MidiSegment],
    bpm: float,
    transcribed_data: Optional[List[TranscribedData]] = None,
    vad_timestamps: Optional[np.ndarray] = None,
    vad_scores: Optional[np.ndarray] = None,
    settings: Optional[object] = None
) -> float:
    """
    Calculate a comprehensive significance score for a note based on musical context.
    
    Args:
        segment: The note segment to evaluate
        prev_segment: The previous note segment (or None)
        next_segment: The next note segment (or None)
        bpm: Beats per minute of the song
        transcribed_data: Optional list of transcribed words with timing
        vad_timestamps: Optional array of VAD timestamps
        vad_scores: Optional array of VAD confidence scores
        settings: Optional settings object with weighting parameters
        
    Returns:
        Score between 0.0 and 1.0 where higher values indicate more musical significance
    """
    # Get default weights if settings not provided
    weights = {
        'duration': 0.35,
        'pitch_change': 0.25,
        'beat_alignment': 0.15,
        'word_boundary': 0.15,
        'vad': 0.10
    }
    
    if settings and hasattr(settings, 'NOTE_SIGNIFICANCE_WEIGHTS'):
        weights = settings.NOTE_SIGNIFICANCE_WEIGHTS
    
    # Calculate individual component scores
    duration_score = calculate_duration_significance(segment, bpm)
    
    pitch_change_score = calculate_pitch_change_significance(
        segment, prev_segment, next_segment
    )
    
    beat_score = calculate_beat_alignment(segment, bpm)
    
    word_score = 0.5  # Default
    if transcribed_data:
        word_score = calculate_word_boundary_score(segment, transcribed_data)
    
    vad_score = 0.5  # Default
    if vad_timestamps is not None and vad_scores is not None:
        vad_score = calculate_vad_confidence(segment, vad_timestamps, vad_scores)
    
    # Calculate weighted score
    significance = (
        weights['duration'] * duration_score +
        weights['pitch_change'] * pitch_change_score +
        weights['beat_alignment'] * beat_score +
        weights['word_boundary'] * word_score +
        weights['vad'] * vad_score
    )
    
    # Ensure result is in [0,1] range
    return max(0.0, min(1.0, significance))


def score_all_notes(
    midi_segments: List[MidiSegment],
    bpm: float,
    transcribed_data: Optional[List[TranscribedData]] = None,
    vad_results: Optional[Dict] = None,
    settings: Optional[object] = None
) -> List[float]:
    """
    Calculate significance scores for all notes in a sequence.
    
    Args:
        midi_segments: List of MIDI note segments
        bpm: Beats per minute of the song
        transcribed_data: Optional list of transcribed words with timing
        vad_results: Optional dictionary with VAD results (timestamps and scores)
        settings: Optional settings object
        
    Returns:
        List of significance scores corresponding to each note
    """
    # Get VAD data if available
    vad_timestamps = None
    vad_scores = None
    if vad_results and len(vad_results) > 0:
        # Use the first available VAD result
        model = next(iter(vad_results.keys()))
        vad_timestamps, vad_scores = vad_results[model]
    
    scores = []
    
    for i, segment in enumerate(midi_segments):
        # Get adjacent segments
        prev_segment = midi_segments[i-1] if i > 0 else None
        next_segment = midi_segments[i+1] if i < len(midi_segments) - 1 else None
        
        # Calculate significance
        score = calculate_note_significance(
            segment,
            prev_segment,
            next_segment,
            bpm,
            transcribed_data,
            vad_timestamps,
            vad_scores,
            settings
        )
        
        scores.append(score)
    
    return scores


def visualize_note_significance(
    midi_segments: List[MidiSegment],
    significance_scores: List[float],
    output_path: str,
    melismas: Optional[List[bool]] = None,
    regularized_segments: Optional[List[MidiSegment]] = None
) -> None:
    """
    Create a visualization of note significance scores.
    
    Args:
        midi_segments: Original MIDI note segments
        significance_scores: Significance scores for each note
        output_path: Path to save the visualization
        melismas: Optional list indicating which notes are part of melismas
        regularized_segments: Optional list of regularized note segments
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Note pitches with significance scores
    plt.subplot(2, 1, 1)
    
    # Convert notes to MIDI numbers for visualization
    midi_numbers = []
    for segment in midi_segments:
        try:
            midi_num = librosa.note_to_midi(segment.note)
            midi_numbers.append(midi_num)
        except:
            # Fallback for any parsing issues
            midi_numbers.append(60)  # Default to middle C
    
    # Create x-coordinates (note start times)
    x_coords = [segment.start for segment in midi_segments]
    
    # Create colormap based on significance
    colors = plt.cm.viridis(significance_scores)
    
    # Plot notes
    for i, (x, midi_num, score) in enumerate(zip(x_coords, midi_numbers, significance_scores)):
        width = midi_segments[i].end - midi_segments[i].start
        plt.barh(
            midi_num, 
            width, 
            left=x, 
            height=0.8, 
            color=colors[i], 
            alpha=0.7
        )
        
        # Annotate with significance score
        plt.text(
            x + width/2,
            midi_num + 0.4,
            f"{score:.2f}",
            ha='center',
            va='center',
            fontsize=8
        )
        
        # Mark melismas if provided
        if melismas and melismas[i]:
            plt.axvspan(x, x + width, color='red', alpha=0.1)
    
    plt.ylabel('MIDI Note Number')
    plt.title('Note Significance Scores')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, label='Significance Score')
    
    # Plot 2: Before/After comparison if regularized segments provided
    if regularized_segments:
        plt.subplot(2, 1, 2)
        
        # Convert original and regularized segments to visual representation
        orig_segments = [(s.start, s.end, librosa.note_to_midi(s.note)) for s in midi_segments]
        reg_segments = [(s.start, s.end, librosa.note_to_midi(s.note)) for s in regularized_segments]
        
        # Plot original segments (light blue)
        for start, end, pitch in orig_segments:
            plt.barh(
                pitch - 0.2,
                end - start,
                left=start,
                height=0.4,
                color='lightblue',
                alpha=0.7,
                label='Original' if (start, end, pitch) == orig_segments[0] else None
            )
        
        # Plot regularized segments (dark blue)
        for start, end, pitch in reg_segments:
            plt.barh(
                pitch + 0.2,
                end - start,
                left=start,
                height=0.4,
                color='darkblue',
                alpha=0.7,
                label='Regularized' if (start, end, pitch) == reg_segments[0] else None
            )
        
        plt.ylabel('MIDI Note Number')
        plt.xlabel('Time (seconds)')
        plt.title('Before/After Regularization')
        plt.legend()
    else:
        plt.xlabel('Time (seconds)')
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()