"""Note regularization for UltraSinger.

This module provides functions to intelligently regularize notes
based on musical significance and context.
"""

import numpy as np
import os
from typing import List, Optional, Tuple, Dict

from modules.Midi.MidiSegment import MidiSegment
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Pitcher.pitched_data import PitchedData
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted

# Import our other modules
from modules.Pitcher.note_significance import score_all_notes, calculate_semitone_difference
from modules.Pitcher.melisma_detection import detect_melismas


def should_merge_notes(
    segment1: MidiSegment,
    segment2: MidiSegment,
    score1: float,
    score2: float,
    is_melisma1: bool,
    is_melisma2: bool,
    settings: Optional[object] = None
) -> bool:
    """
    Determine if two adjacent notes should be merged based on their significance scores
    and musical context.
    
    Args:
        segment1: First note segment
        segment2: Second note segment
        score1: Significance score of first note
        score2: Significance score of second note
        is_melisma1: Whether first note is part of a melisma
        is_melisma2: Whether second note is part of a melisma
        settings: Optional settings object
        
    Returns:
        True if notes should be merged, False otherwise
    """
    # Get thresholds from settings or use defaults
    low_significance_threshold = 0.3
    if settings and hasattr(settings, 'NOTE_LOW_SIGNIFICANCE_THRESHOLD'):
        low_significance_threshold = settings.NOTE_LOW_SIGNIFICANCE_THRESHOLD
    
    high_significance_threshold = 0.6
    if settings and hasattr(settings, 'NOTE_HIGH_SIGNIFICANCE_THRESHOLD'):
        high_significance_threshold = settings.NOTE_HIGH_SIGNIFICANCE_THRESHOLD
    
    max_pitch_diff_for_merge = 2.0  # Maximum semitone difference to allow merging
    if settings and hasattr(settings, 'MAX_PITCH_DIFF_FOR_MERGE'):
        max_pitch_diff_for_merge = settings.MAX_PITCH_DIFF_FOR_MERGE
    
    # Calculate pitch difference
    pitch_diff = calculate_semitone_difference(segment1.note, segment2.note)
    
    # Gap between notes
    gap = segment2.start - segment1.end
    
    # Case 1: Both notes have low significance
    if score1 < low_significance_threshold and score2 < low_significance_threshold:
        # If pitches are similar, merge them
        if pitch_diff <= max_pitch_diff_for_merge:
            return True
    
    # Case 2: One note has low significance and one is part of a melisma
    if ((score1 < low_significance_threshold and is_melisma2) or
        (score2 < low_significance_threshold and is_melisma1)):
        # If pitches are similar, merge them
        if pitch_diff <= max_pitch_diff_for_merge:
            return True
    
    # Case 3: Both notes are part of the same melisma
    if is_melisma1 and is_melisma2:
        # For melismas, be more conservative about merging to preserve vocal runs
        if score1 < low_significance_threshold and score2 < low_significance_threshold:
            if pitch_diff <= max_pitch_diff_for_merge:
                return True
    
    # Case 4: Notes have a tiny gap between them
    if 0 < gap < 0.05:
        # For very small gaps with similar pitches, merge them
        if pitch_diff <= max_pitch_diff_for_merge:
            return True
    
    # Case 5: Notes are very short
    duration1 = segment1.end - segment1.start
    duration2 = segment2.end - segment2.start
    
    if duration1 < 0.1 and duration2 < 0.1:
        # Very short notes with similar pitches should be merged
        if pitch_diff <= max_pitch_diff_for_merge:
            return True
    
    # Default: don't merge
    return False


def merge_segments(segment1: MidiSegment, segment2: MidiSegment) -> MidiSegment:
    """
    Merge two MIDI segments into one.
    
    Args:
        segment1: First segment
        segment2: Second segment
        
    Returns:
        Merged segment
    """
    # Choose the note with higher confidence (for now, we'll use the longer note)
    duration1 = segment1.end - segment1.start
    duration2 = segment2.end - segment2.start
    
    dominant_note = segment1.note if duration1 >= duration2 else segment2.note
    
    # Combine the words, handling continuation markers
    if segment2.word.strip() == "~" or segment2.word.startswith("~"):
        # Keep first word if second is continuation
        combined_word = segment1.word
    elif segment1.word.strip().endswith(" ") and not segment2.word.strip().startswith(" "):
        # Handle space at end of first word
        combined_word = segment1.word + segment2.word
    else:
        # Default case
        combined_word = segment1.word + segment2.word
    
    # Create merged segment
    return MidiSegment(
        note=dominant_note,
        start=segment1.start,
        end=segment2.end,
        word=combined_word
    )


def post_process_regularized_notes(
    segments: List[MidiSegment],
    bpm: float
) -> List[MidiSegment]:
    """
    Apply post-processing to regularized notes to fix any remaining issues.
    
    Args:
        segments: List of MIDI segments
        bpm: Beats per minute
        
    Returns:
        List of post-processed MIDI segments
    """
    if not segments:
        return []
    
    processed = []
    
    for i, segment in enumerate(segments):
        # Fix any overlaps with previous notes
        if processed and segment.start < processed[-1].end:
            # Handle overlap
            overlap = processed[-1].end - segment.start
            
            # If overlap is significant, adjust this note's start time
            if overlap > 0.05:
                segment = MidiSegment(
                    note=segment.note,
                    start=processed[-1].end,
                    end=segment.end,
                    word=segment.word
                )
        
        # Fix very short notes
        duration = segment.end - segment.start
        min_duration = 0.1  # Absolute minimum duration
        
        if duration < min_duration:
            # Extend the note if possible
            if i < len(segments) - 1:
                next_start = segments[i + 1].start
                segment = MidiSegment(
                    note=segment.note,
                    start=segment.start,
                    end=min(next_start, segment.start + min_duration),
                    word=segment.word
                )
            else:
                # Just extend the note
                segment = MidiSegment(
                    note=segment.note,
                    start=segment.start,
                    end=segment.start + min_duration,
                    word=segment.word
                )
        
        processed.append(segment)
    
    return processed


def regularize_notes(
    midi_segments: List[MidiSegment],
    bpm: float,
    transcribed_data: Optional[List[TranscribedData]] = None,
    pitched_data: Optional[PitchedData] = None,
    vad_results: Optional[Dict] = None,
    settings: Optional[object] = None
) -> List[MidiSegment]:
    """
    Regularize notes based on musical significance and context.
    
    Args:
        midi_segments: List of MIDI note segments
        bpm: Beats per minute
        transcribed_data: Optional list of transcribed words with timing
        pitched_data: Optional pitched data
        vad_results: Optional VAD results
        settings: Optional settings object
        
    Returns:
        List of regularized MIDI segments
    """
    if not midi_segments:
        return []
    
    print(f"{ULTRASINGER_HEAD} Regularizing {len(midi_segments)} notes with significance-based approach")
    
    # Step 1: Calculate significance scores for all notes
    significance_scores = score_all_notes(
        midi_segments, bpm, transcribed_data, vad_results, settings
    )
    
    # Step 2: Detect melismas
    is_melisma = detect_melismas(midi_segments, bpm, transcribed_data, settings)
    
    # Step 3: Create debug visualization if enabled
    debug_level = 0
    if settings and hasattr(settings, 'DEBUG_LEVEL'):
        debug_level = settings.DEBUG_LEVEL
    
    if debug_level >= 2 and hasattr(settings, 'output_folder_path'):
        from modules.Pitcher.note_significance import visualize_note_significance
        
        output_path = os.path.join(settings.output_folder_path, "cache", "note_significance.png")
        visualize_note_significance(
            midi_segments,
            significance_scores,
            output_path,
            is_melisma
        )
    
    # Step 4: Iteratively merge notes
    result_segments = []
    i = 0
    
    while i < len(midi_segments):
        current_segment = midi_segments[i]
        current_score = significance_scores[i]
        current_is_melisma = is_melisma[i]
        
        # If this is the last note, just add it and we're done
        if i == len(midi_segments) - 1:
            result_segments.append(current_segment)
            break
        
        # Check if we should merge with the next note
        next_segment = midi_segments[i + 1]
        next_score = significance_scores[i + 1]
        next_is_melisma = is_melisma[i + 1]
        
        if should_merge_notes(
            current_segment,
            next_segment,
            current_score,
            next_score,
            current_is_melisma,
            next_is_melisma,
            settings
        ):
            # Merge the notes
            merged_segment = merge_segments(current_segment, next_segment)
            
            # Skip the next note since we merged it
            i += 2
            
            # Add the merged segment
            result_segments.append(merged_segment)
        else:
            # Just add the current note and move on
            result_segments.append(current_segment)
            i += 1
    
    # Step 5: Fix any issues with the regularized notes
    result_segments = post_process_regularized_notes(result_segments, bpm)
    
    original_count = len(midi_segments)
    final_count = len(result_segments)
    reduction = original_count - final_count
    reduction_percent = reduction / original_count * 100 if original_count > 0 else 0
    
    print(f"{ULTRASINGER_HEAD} Regularization complete: {original_count} → {final_count} notes ({reduction_percent:.1f}% reduction)")
    
    # Create "after" visualization if debug enabled
    if debug_level >= 2 and hasattr(settings, 'output_folder_path'):
        from modules.Pitcher.note_significance import visualize_note_significance
        
        # Recalculate scores for the regularized segments
        new_scores = score_all_notes(
            result_segments, bpm, transcribed_data, vad_results, settings
        )
        
        output_path = os.path.join(settings.output_folder_path, "cache", "regularized_notes.png")
        visualize_note_significance(
            midi_segments,
            significance_scores,
            output_path,
            is_melisma,
            result_segments
        )
    
    return result_segments