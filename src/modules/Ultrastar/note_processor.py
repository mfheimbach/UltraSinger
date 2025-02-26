"""Note processor for Ultrastar."""

from typing import List
import math
import re

from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_length_calculator import get_sixteenth_note_second, get_whole_note_second
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted

def optimize_midi_segments(midi_segments: List[MidiSegment], real_bpm: float) -> List[MidiSegment]:
    """
    Apply multiple optimization steps to MIDI segments with iterative approach.
    
    Pipeline: unify pitches → normalize → resolve overlaps → eliminate short notes → validate
    
    Args:
        midi_segments: List of MidiSegment objects
        real_bpm: Real BPM value of the song
        
    Returns:
        List of optimized MidiSegment objects
    """
    if not midi_segments:
        return []
        
    print(f"{ULTRASINGER_HEAD} Running note optimization on {len(midi_segments)} notes with BPM: {blue_highlighted(str(round(real_bpm, 1)))}")
    segments = midi_segments.copy()
    original_count = len(segments)
    
    # Calculate duration thresholds based on BPM
    min_duration = get_sixteenth_note_second(real_bpm)
    max_duration = get_whole_note_second(real_bpm) * 4  # 4 whole notes
    
    print(f"{ULTRASINGER_HEAD} Duration thresholds: min={blue_highlighted(f'{min_duration:.3f}s')}, max={blue_highlighted(f'{max_duration:.3f}s')}")
    
    # Step 1: Unify pitch clusters first
    segments_before = len(segments)
    segments = unify_pitch_clusters(segments, real_bpm)
    print(f"{ULTRASINGER_HEAD} Pitch clustering: {segments_before} → {len(segments)} notes")
    
    # Step 2: Normalize notes (merge short with adjacent)
    segments_before = len(segments)
    segments = normalize_note_durations(segments, real_bpm)
    print(f"{ULTRASINGER_HEAD} Duration normalization: {segments_before} → {len(segments)} notes")
    
    # Step 3: Resolve overlaps
    segments_before = len(segments)
    segments = resolve_overlaps(segments, min_duration)
    print(f"{ULTRASINGER_HEAD} Overlap resolution: {segments_before} → {len(segments)} notes")
    
    # Step 4: Eliminate remaining short notes
    segments_before = len(segments)
    segments = eliminate_very_short_notes(segments, min_duration/2)
    print(f"{ULTRASINGER_HEAD} Short note elimination: {segments_before} → {len(segments)} notes")
    
    # Step 5: Validate structure (note/lyric alignment)
    segments_before = len(segments)
    segments = validate_structure(segments)
    print(f"{ULTRASINGER_HEAD} Structure validation: {segments_before} → {len(segments)} notes")
    
    # Final metrics
    final_count = len(segments)
    print(f"{ULTRASINGER_HEAD} Note optimization complete: {original_count} → {final_count} notes ({original_count - final_count} removed)")
    
    return segments
    
def eliminate_very_short_notes(midi_segments: List[MidiSegment], min_threshold: float) -> List[MidiSegment]:
    """Eliminate notes that are shorter than the minimum threshold."""
    return [segment for segment in midi_segments if segment.end - segment.start >= min_threshold]

def normalize_note_durations(midi_segments: List[MidiSegment], real_bpm: float) -> List[MidiSegment]:
    """
    Fix notes with abnormal durations (too short or too long).
    
    Args:
        midi_segments: List of MidiSegment objects
        real_bpm: Real BPM value of the song
        
    Returns:
        List of MidiSegment objects with normalized durations
    """
    if not midi_segments:
        return midi_segments
    
    # Calculate duration thresholds based on BPM
    min_duration = get_sixteenth_note_second(real_bpm)
    max_duration = get_whole_note_second(real_bpm) * 4  # 4 whole notes
    
    # Identify short notes for potential merging
    short_notes = []
    for i, segment in enumerate(midi_segments):
        duration = segment.end - segment.start
        if duration < min_duration:
            short_notes.append(i)
    
    # First pass: Attach short notes to adjacent long notes
    segments = attach_short_to_long(midi_segments, short_notes, min_duration)
    
    # Second pass: Connect pairs of short notes
    segments = connect_short_pairs(segments, min_duration)
    
    # Third pass: Attach remaining short notes to newly created long notes
    updated_short_notes = []
    for i, segment in enumerate(segments):
        duration = segment.end - segment.start
        if duration < min_duration:
            updated_short_notes.append(i)
    
    segments = attach_short_to_long(segments, updated_short_notes, min_duration)
    
    # Finally, fix any excessively long notes
    segments = limit_long_notes(segments, max_duration)
    
    return segments

def count_short_notes(midi_segments: List[MidiSegment], min_duration: float) -> int:
    """Count how many notes are shorter than the minimum duration."""
    return sum(1 for segment in midi_segments if segment.end - segment.start < min_duration)

def force_resolve_short_notes(midi_segments: List[MidiSegment], min_duration: float) -> List[MidiSegment]:
    """
    Force resolution of remaining short notes by either merging or removing them.
    
    Args:
        midi_segments: List of MidiSegment objects
        min_duration: Minimum acceptable note duration
        
    Returns:
        List of updated MidiSegment objects with no very short notes
    """
    result = []
    elimination_threshold = min_duration / 2  # Notes shorter than this will be eliminated if they can't be merged
    i = 0
    
    while i < len(midi_segments):
        current = midi_segments[i]
        duration = current.end - current.start
        
        if duration < min_duration:
            # Try to merge with adjacent notes if possible
            merged_successfully = False
            
            # Try to merge with next note
            if i < len(midi_segments) - 1:
                next_note = midi_segments[i+1]
                gap = next_note.start - current.end
                
                if gap < min_duration / 2 and is_pitch_similar(current.note, next_note.note, 3):
                    # Merge into next note
                    next_note.start = current.start
                    # Keep next note's word if current is a continuation
                    if current.word.strip() == "~":
                        pass  # Keep next note's word
                    elif not next_note.word.startswith("~"):
                        next_note.word = current.word + next_note.word
                    
                    # Skip this note (don't add to result)
                    merged_successfully = True
                    i += 1  # Advance to skip the merged note
            
            # If not merged and too short, eliminate
            if not merged_successfully and duration < elimination_threshold:
                # Skip this note completely (don't add to result)
                pass
            elif not merged_successfully:
                # Note is short but above elimination threshold and couldn't be merged
                result.append(current)
        else:
            # Note is already long enough
            result.append(current)
        
        i += 1
    
    return result

def attach_short_to_long(midi_segments: List[MidiSegment], short_note_indices: List[int], min_duration: float, max_pitch_diff: int = 3) -> List[MidiSegment]:
    """
    Attach short notes to adjacent long notes with similar pitch.
    
    Args:
        midi_segments: List of MidiSegment objects
        short_note_indices: Indices of short notes to process
        min_duration: Minimum acceptable note duration
        max_pitch_diff: Maximum semitone difference for pitch matching
        
    Returns:
        List of updated MidiSegment objects
    """
    if not short_note_indices:
        return midi_segments
    
    result = midi_segments.copy()
    processed = set()
    elimination_threshold = min_duration / 2  # Notes shorter than this will be candidates for elimination
    
    for idx in short_note_indices:
        if idx in processed:
            continue
            
        short_note = result[idx]
        duration = short_note.end - short_note.start
        is_continuation = short_note.word.strip() == "~"
        
        # Adjust pitch difference for continuations
        pitch_diff = max_pitch_diff
        if is_continuation:
            pitch_diff = 5  # Allow larger pitch difference for continuations
        
        # Try to attach to previous note
        if idx > 0:
            prev_note = result[idx-1]
            prev_duration = prev_note.end - prev_note.start
            gap = short_note.start - prev_note.end
            
            # Check if we can attach to previous note
            if (prev_duration >= min_duration / 2 and  # Lower threshold for prev note
                gap < min_duration / 2 and            # Gap threshold
                is_pitch_similar(prev_note.note, short_note.note, pitch_diff)):
                
                # Extend previous note
                prev_note.end = short_note.end
                result[idx-1] = prev_note
                processed.add(idx)
                continue
        
        # Try to attach to next note
        if idx < len(result) - 1:
            next_note = result[idx+1]
            next_duration = next_note.end - next_note.start
            gap = next_note.start - short_note.end
            
            # Adjust pitch diff if next note is a continuation
            next_pitch_diff = pitch_diff
            if next_note.word.strip().startswith("~"):
                next_pitch_diff = 5
                
            # Check if we can attach to next note
            if (next_duration >= min_duration / 2 and  # Lower threshold
                gap < min_duration / 2 and            # Gap threshold
                is_pitch_similar(next_note.note, short_note.note, next_pitch_diff)):
                
                # Extend next note backward
                next_note.start = short_note.start
                result[idx+1] = next_note
                processed.add(idx)
                continue
                
        # If note is very short and couldn't be merged, mark for elimination
        if duration < elimination_threshold:
            processed.add(idx)
    
    # Remove processed notes
    return [note for i, note in enumerate(result) if i not in processed]

def connect_short_pairs(midi_segments: List[MidiSegment], min_duration: float, max_pitch_diff: int = 3) -> List[MidiSegment]:
    """
    Connect pairs of short notes with similar pitches.
    
    Args:
        midi_segments: List of MidiSegment objects
        min_duration: Minimum acceptable note duration
        max_pitch_diff: Maximum semitone difference for pitch matching
        
    Returns:
        List of updated MidiSegment objects
    """
    if len(midi_segments) < 2:
        return midi_segments
    
    result = []
    skip_next = False
    
    for i in range(len(midi_segments) - 1):
        if skip_next:
            skip_next = False
            continue
            
        current = midi_segments[i]
        next_note = midi_segments[i+1]
        
        current_duration = current.end - current.start
        next_duration = next_note.end - next_note.start
        
        # If both notes are short and close together with similar pitch
        if (current_duration < min_duration and 
            next_duration < min_duration and 
            next_note.start - current.end < min_duration / 2 and
            is_pitch_similar(current.note, next_note.note, max_pitch_diff)):
            
            # Create merged note with dominant pitch
            merged = MidiSegment(
                note=get_dominant_pitch([current, next_note]),
                start=current.start,
                end=next_note.end,
                word=current.word + next_note.word if not next_note.word.startswith("~") else current.word
            )
            
            result.append(merged)
            skip_next = True
        else:
            result.append(current)
            
    # Add the last element if not processed
    if not skip_next and midi_segments:
        result.append(midi_segments[-1])
    
    return result

def limit_long_notes(midi_segments: List[MidiSegment], max_duration: float) -> List[MidiSegment]:
    """
    Limit excessively long notes.
    
    Args:
        midi_segments: List of MidiSegment objects
        max_duration: Maximum acceptable note duration
        
    Returns:
        List of updated MidiSegment objects
    """
    result = []
    
    for segment in midi_segments:
        duration = segment.end - segment.start
        
        if duration > max_duration:
            segment.end = segment.start + max_duration
        
        result.append(segment)
    
    return result

def is_pitch_similar(note1: str, note2: str, max_semitones: int = 3) -> bool:
    """
    Check if two note pitches are within a certain semitone range.
    
    Args:
        note1: First note (e.g., 'C4')
        note2: Second note (e.g., 'D4')
        max_semitones: Maximum semitone difference allowed
        
    Returns:
        True if notes are within range, False otherwise
    """
    # Extract note name and octave
    pattern = r'([A-G][#b]?)(\d+)'
    match1 = re.match(pattern, note1)
    match2 = re.match(pattern, note2)
    
    if not match1 or not match2:
        return False
    
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
    diff = abs(semitones1 - semitones2)
    
    return diff <= max_semitones

def get_dominant_pitch(notes: List[MidiSegment]) -> str:
    """
    Get the dominant pitch from a list of notes.
    
    Args:
        notes: List of MidiSegment objects
        
    Returns:
        Dominant pitch (from longest note, or most common)
    """
    if not notes:
        return ""
    
    # First try to get pitch from longest note
    longest_note = max(notes, key=lambda n: n.end - n.start)
    
    # If longest note has meaningful duration, use its pitch
    if longest_note.end - longest_note.start > 0.1:
        return longest_note.note
    
    # Otherwise count pitches and use most common
    pitch_counts = {}
    for note in notes:
        pitch_counts[note.note] = pitch_counts.get(note.note, 0) + 1
    
    return max(pitch_counts.items(), key=lambda x: x[1])[0]

def unify_pitch_clusters(midi_segments: List[MidiSegment], real_bpm: float) -> List[MidiSegment]:
    """
    Unify pitch clusters to handle fragmented melismas.
    
    Args:
        midi_segments: List of MidiSegment objects
        real_bpm: Real BPM value of the song
        
    Returns:
        List of updated MidiSegment objects
    """
    if len(midi_segments) < 3:
        return midi_segments
    
    min_duration = get_sixteenth_note_second(real_bpm)
    gap_threshold = min_duration / 2
    
    # Identify clusters of short notes with continuations
    clusters = []
    current_cluster = []
    
    for i, segment in enumerate(midi_segments):
        duration = segment.end - segment.start
        is_short = duration < min_duration
        is_continuation = segment.word.strip() == "~"
        
        # Check if we should add to current cluster
        if (is_short and (is_continuation or current_cluster)):
            # Check gap with previous note if cluster not empty
            if current_cluster:
                prev_idx = current_cluster[-1]
                prev_segment = midi_segments[prev_idx]
                gap = segment.start - prev_segment.end
                
                if gap <= gap_threshold:
                    current_cluster.append(i)
                else:
                    # Gap too large, end current cluster
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [i] if is_short else []
            else:
                current_cluster.append(i)
        else:
            # End current cluster
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            current_cluster = [i] if is_short else []
    
    # Add final cluster if needed
    if len(current_cluster) >= 3:
        clusters.append(current_cluster)
    
    # Process each cluster
    result = midi_segments.copy()
    processed_indices = set()
    
    for cluster in clusters:
        # Get cluster notes
        cluster_notes = [result[idx] for idx in cluster]
        
        # Find dominant pitch
        dominant_pitch = get_dominant_pitch(cluster_notes)
        
        # Update all notes in cluster to use dominant pitch
        for idx in cluster:
            if result[idx].note != dominant_pitch:
                result[idx].note = dominant_pitch
        
        processed_indices.update(cluster)
    
    # Now merge adjacent same-pitch notes in processed clusters
    final_result = []
    i = 0
    
    while i < len(result):
        if i not in processed_indices:
            final_result.append(result[i])
            i += 1
            continue
        
        # Start of a potential merge
        current = result[i]
        merged_note = MidiSegment(
            note=current.note,
            start=current.start,
            end=current.end,
            word=current.word
        )
        
        j = i + 1
        while j < len(result) and j in processed_indices:
            next_note = result[j]
            gap = next_note.start - merged_note.end
            
            if gap <= gap_threshold and next_note.note == merged_note.note:
                # Merge with next note
                merged_note.end = next_note.end
                # Handle word merging for continuations
                if not next_note.word.startswith("~"):
                    merged_note.word += next_note.word
                j += 1
            else:
                break
        
        final_result.append(merged_note)
        i = j
    
    return final_result

def resolve_overlaps(midi_segments: List[MidiSegment], min_duration: float) -> List[MidiSegment]:
    """
    Resolve overlaps conservatively.
    
    Args:
        midi_segments: List of MidiSegment objects
        min_duration: Minimum acceptable note duration
        
    Returns:
        List of updated MidiSegment objects
    """
    if len(midi_segments) < 2:
        return midi_segments
    
    result = [midi_segments[0]]
    
    for i in range(1, len(midi_segments)):
        current = midi_segments[i]
        previous = result[-1]
        
        # Check for overlap
        overlap = previous.end - current.start
        
        if overlap > 0:
            # Rules for resolving overlaps:
            # 1. Longer note wins
            # 2. If equal length, first note wins
            # 3. If cut note becomes too short, delete it
            
            prev_duration = previous.end - previous.start
            curr_duration = current.end - current.start
            
            if prev_duration > curr_duration:
                # Previous note is longer, it wins
                current.start = previous.end
                
                # Check if current became too short
                new_duration = current.end - current.start
                if new_duration < min_duration / 2:
                    # Skip this note entirely
                    continue
            else:
                # Current note is longer or equal, it wins
                previous.end = current.start
                result[-1] = previous
        
        result.append(current)
    
    return result

def validate_structure(midi_segments: List[MidiSegment]) -> List[MidiSegment]:
    """
    Validate structure (note/lyric alignment).
    
    Args:
        midi_segments: List of MidiSegment objects
        
    Returns:
        List of updated MidiSegment objects
    """
    if len(midi_segments) < 2:
        return midi_segments
    
    result = midi_segments.copy()
    
    # Check for missing continuations in melismas
    for i in range(1, len(result)):
        prev = result[i-1]
        curr = result[i]
        
        # Check if previous word should have a continuation but doesn't
        if (not prev.word.endswith(" ") and  # Not end of word
            not prev.word.endswith("~") and  # Doesn't already have continuation
            not curr.word.startswith("~")):  # Next note doesn't continue it
            
            # Add continuation marker
            prev.word += "~"
            result[i-1] = prev
    
    # Check for word/note alignment
    for i in range(len(result) - 1):
        curr = result[i]
        next_note = result[i+1]
        
        # Check for mid-syllable breaks
        if curr.word.endswith("~") and next_note.word.startswith(" "):
            # Remove space from next note
            next_note.word = next_note.word.lstrip()
            result[i+1] = next_note
    
    return result