"""Probabilistic pitch processor for UltraSinger.

Uses pitch detection confidence to make better decisions about notes.
"""

from typing import List, Tuple, Dict
import numpy as np
import librosa
import math

from modules.Midi.MidiSegment import MidiSegment
from modules.Pitcher.pitched_data import PitchedData
from modules.Midi.note_length_calculator import get_sixteenth_note_second, get_eighth_note_second
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, green_highlighted


class ProbabilisticPitchProcessor:
    """
    Processes pitch data using probability and confidence scores
    to produce more natural-sounding note segments.
    """
    
    def __init__(self, real_bpm: float):
        """
        Initialize with song's BPM.
        
        Args:
            real_bpm: Real BPM value of the song
        """
        self.real_bpm = real_bpm
        self.min_duration = get_sixteenth_note_second(real_bpm)
        self.preferred_duration = get_eighth_note_second(real_bpm)
        
        # Parameters for confidence weighting
        self.conf_threshold_low = 0.3   # Minimum confidence to consider
        self.conf_threshold_high = 0.7  # High confidence threshold
        self.duration_weight = 0.6      # Weight for duration factor
        self.confidence_weight = 0.4    # Weight for confidence factor
        
        print(f"{ULTRASINGER_HEAD} Probabilistic pitch processor initialized with BPM: {blue_highlighted(str(round(real_bpm, 1)))}")
        print(f"{ULTRASINGER_HEAD} Duration thresholds: min={blue_highlighted(f'{self.min_duration:.3f}s')}, preferred={blue_highlighted(f'{self.preferred_duration:.3f}s')}")
    
    def process_pitch_data(self, pitched_data: PitchedData) -> List[MidiSegment]:
        """
        Process pitch data using confidence to create better note segments.
        
        Args:
            pitched_data: PitchedData object with time, frequency and confidence values
            
        Returns:
            List of MidiSegment objects
        """
        print(f"{ULTRASINGER_HEAD} Processing {len(pitched_data.times)} pitch data points")
        
        # Step 1: Find stable pitch regions
        stable_regions = self._find_stable_pitch_regions(pitched_data)
        print(f"{ULTRASINGER_HEAD} Found {len(stable_regions)} initial pitch regions")
        
        # Step 2: Filter regions by confidence-weighted probability
        filtered_regions = self._filter_by_probability(stable_regions, pitched_data)
        print(f"{ULTRASINGER_HEAD} After probability filtering: {len(filtered_regions)} regions")
        
        # Step 3: Merge adjacent compatible regions
        merged_regions = self._merge_compatible_regions(filtered_regions)
        print(f"{ULTRASINGER_HEAD} After merging: {len(merged_regions)} note segments")
        
        # Step 4: Validate final results
        validated_regions = self._validate_regions(merged_regions)
        print(f"{ULTRASINGER_HEAD} Final validated note segments: {len(validated_regions)}")
        
        # Create MidiSegment objects
        midi_segments = []
        for i, (start, end, note, avg_conf) in enumerate(validated_regions):
            # Generate placeholder words - these will be replaced with actual lyrics later
            word = "♪" if i < len(validated_regions) - 1 else "♪ "
            
            midi_segments.append(MidiSegment(
                note=note,
                start=start,
                end=end,
                word=word
            ))
        
        return midi_segments
    
    def process_with_lyrics(self, pitched_data: PitchedData, transcribed_data) -> List[MidiSegment]:
        """Use transcribed words as the primary structure, determining pitch for each word."""
        # Check if we have transcribed data
        if not transcribed_data or len(transcribed_data) == 0:
            print(f"{ULTRASINGER_HEAD} No transcribed data available, falling back to pitch-only processing")
            return self.process_pitch_data(pitched_data)
        
        print(f"{ULTRASINGER_HEAD} Processing {len(transcribed_data)} words")
        
        # Create output segments based on the transcribed words
        midi_segments = []
        
        for word_data in transcribed_data:
            # Skip empty words
            if not word_data.word.strip():
                continue
                
            word = word_data.word
            word_start = word_data.start
            word_end = word_data.end
            
            # Find relevant pitch data within this word's timeframe
            relevant_times = []
            relevant_freqs = []
            relevant_confs = []
            
            for i, time in enumerate(pitched_data.times):
                if word_start <= time <= word_end:
                    # Only include pitch data with reasonable confidence
                    if pitched_data.confidence[i] > 0.3:
                        relevant_times.append(time)
                        relevant_freqs.append(pitched_data.frequencies[i])
                        relevant_confs.append(pitched_data.confidence[i])
            
            # If we have pitch data for this word
            if relevant_freqs:
                # Get dominant pitch using weighted confidence
                total_weight = sum(relevant_confs)
                if total_weight > 0:  # Avoid division by zero
                    conf_weighted_freqs = [f * c for f, c in zip(relevant_freqs, relevant_confs)]
                    avg_freq = sum(conf_weighted_freqs) / total_weight
                    note = librosa.hz_to_note(avg_freq)
                else:
                    # Default to middle C if we can't determine a note
                    note = "C4"
                    
                # Create MIDI segment with the word
                midi_segments.append(MidiSegment(
                    note=note,
                    start=word_start,
                    end=word_end,
                    word=word
                ))
            else:
                # If we have no pitch data for this word, use a reasonable default
                note = "C4"  # Default to middle C
                midi_segments.append(MidiSegment(
                    note=note,
                    start=word_start,
                    end=word_end,
                    word=word
                ))
        
        print(f"{ULTRASINGER_HEAD} Created {len(midi_segments)} word-aligned notes")
        return midi_segments
    
    def enhance_with_original_audio(self, pitched_data_separated: PitchedData, 
                                   pitched_data_original: PitchedData) -> PitchedData:
        """
        Enhance confidence scores by comparing separated and original audio.
        
        Args:
            pitched_data_separated: Pitch data from separated vocals
            pitched_data_original: Pitch data from original unseparated audio
            
        Returns:
            Enhanced PitchedData with improved confidence scores
        """
        # Create enhanced data structure
        enhanced_data = PitchedData(
            times=pitched_data_separated.times.copy(),
            frequencies=pitched_data_separated.frequencies.copy(),
            confidence=pitched_data_separated.confidence.copy()
        )
        
        # Find matching times in both datasets
        for i, time in enumerate(pitched_data_separated.times):
            # Find closest time in original data
            original_idx = self._find_closest_time(time, pitched_data_original.times)
            
            if original_idx is not None:
                sep_freq = pitched_data_separated.frequencies[i]
                orig_freq = pitched_data_original.frequencies[original_idx]
                sep_conf = pitched_data_separated.confidence[i]
                orig_conf = pitched_data_original.confidence[original_idx]
                
                # If pitches match within a semitone, boost confidence
                if self._is_pitch_similar(sep_freq, orig_freq):
                    # Weighted average favoring the higher confidence
                    enhanced_conf = max(sep_conf, orig_conf) * 0.7 + min(sep_conf, orig_conf) * 0.3
                    enhanced_data.confidence[i] = min(1.0, enhanced_conf * 1.2)  # Boost but cap at 1.0
                else:
                    # Pitches disagree, reduce confidence
                    enhanced_data.confidence[i] = sep_conf * 0.8
        
        print(f"{ULTRASINGER_HEAD} Enhanced pitch confidence using original audio comparison")
        return enhanced_data
    
    def _find_closest_time(self, target_time: float, time_list) -> int:
        """Find index of closest time in a list."""
        if not time_list or len(time_list) == 0:
            return None
            
        # Use numpy operations for efficient search
        time_array = np.array(time_list)
        idx = np.abs(time_array - target_time).argmin()
        if abs(time_list[idx] - target_time) < 0.05:  # Within 50ms
            return idx
        return None
            
    def _is_pitch_similar(self, freq1: float, freq2: float, tolerance_semitones: float = 1.0) -> bool:
        """Check if two frequencies are within a certain semitone range."""
        if freq1 <= 0 or freq2 <= 0:
            return False
            
        # Convert ratio to semitones
        ratio = freq1 / freq2 if freq1 <= freq2 else freq2 / freq1
        semitone_diff = 12 * math.log2(ratio)
        
        return semitone_diff < tolerance_semitones
    
    def _find_stable_pitch_regions(self, pitched_data: PitchedData) -> List[Tuple[float, float, str, float]]:
        """
        Find regions of stable pitch.
        
        Returns list of (start_time, end_time, note, avg_confidence) tuples
        """
        if not pitched_data.times or len(pitched_data.times) == 0:
            return []
            
        regions = []
        current_start = 0
        current_note = self._freq_to_note(pitched_data.frequencies[0])
        
        for i in range(1, len(pitched_data.times)):
            # Skip very low confidence points
            if pitched_data.confidence[i] < self.conf_threshold_low:
                continue
                
            note = self._freq_to_note(pitched_data.frequencies[i])
            
            # If note changed or long gap, end the current region
            time_gap = pitched_data.times[i] - pitched_data.times[i-1]
            if note != current_note or time_gap > self.min_duration:
                if i - current_start >= 3:  # Require at least 3 points for a region
                    avg_conf = np.mean([pitched_data.confidence[j] for j in range(current_start, i)])
                    start_time = pitched_data.times[current_start]
                    end_time = pitched_data.times[i-1]
                    
                    regions.append((start_time, end_time, current_note, avg_conf))
                
                current_start = i
                current_note = note
        
        # Add the final region
        if len(pitched_data.times) - current_start >= 3:
            avg_conf = np.mean([pitched_data.confidence[j] for j in range(current_start, len(pitched_data.times))])
            start_time = pitched_data.times[current_start]
            end_time = pitched_data.times[-1]
            
            regions.append((start_time, end_time, current_note, avg_conf))
        
        return regions
    
    def _filter_by_probability(self, regions: List[Tuple[float, float, str, float]], 
                               pitched_data: PitchedData) -> List[Tuple[float, float, str, float]]:
        """
        Filter regions using confidence-weighted probability.
        
        Higher confidence allows shorter notes, lower confidence requires longer notes.
        """
        filtered_regions = []
        
        for start_time, end_time, note, avg_conf in regions:
            duration = end_time - start_time
            
            # Calculate probability of keeping based on duration and confidence
            duration_factor = min(1.0, duration / self.preferred_duration)
            confidence_factor = (avg_conf - self.conf_threshold_low) / (self.conf_threshold_high - self.conf_threshold_low)
            confidence_factor = max(0.0, min(1.0, confidence_factor))
            
            # Weighted probability
            probability = (duration_factor * self.duration_weight + 
                          confidence_factor * self.confidence_weight)
            
            # Continuous threshold function based on duration
            # Starts high for very short notes, decreases smoothly
            duration_ratio = min(1.0, duration / self.min_duration)
            threshold = 0.8 * math.exp(-2.0 * duration_ratio) + 0.4
            
            # Apply threshold
            if probability >= threshold:
                filtered_regions.append((start_time, end_time, note, avg_conf))
        
        return filtered_regions
        
    
    def _merge_compatible_regions(self, regions: List[Tuple[float, float, str, float]]) -> List[Tuple[float, float, str, float]]:
        """Merge adjacent compatible regions."""
        if not regions or len(regions) == 0:
            return []
            
        regions.sort(key=lambda r: r[0])  # Sort by start time
        merged = [regions[0]]
        
        for current_start, current_end, current_note, current_conf in regions[1:]:
            prev_start, prev_end, prev_note, prev_conf = merged[-1]
            
            # Check if regions can be merged
            gap = current_start - prev_end
            
            if (gap < self.min_duration / 2 and 
                current_note == prev_note):
                
                # Merge regions
                weighted_conf = (prev_conf * (prev_end - prev_start) + 
                               current_conf * (current_end - current_start)) / (current_end - prev_start)
                
                merged[-1] = (prev_start, current_end, current_note, weighted_conf)
            else:
                merged.append((current_start, current_end, current_note, current_conf))
        
        return merged
    
    def _validate_regions(self, regions: List[Tuple[float, float, str, float]]) -> List[Tuple[float, float, str, float]]:
        """Validate and clean up regions."""
        if not regions or len(regions) == 0:
            return []
            
        validated = []
        
        for i, (start, end, note, conf) in enumerate(regions):
            # Ensure minimum duration
            if end - start < self.min_duration / 2:
                continue
                
            # Fix overlaps with previous note
            if validated and start < validated[-1][1]:
                # Resolve overlap by adjusting start time
                start = validated[-1][1]
                
                # Skip if resulting duration is too short
                if end - start < self.min_duration / 2:
                    continue
            
            validated.append((start, end, note, conf))
        
        return validated
    
    def _freq_to_note(self, freq: float) -> str:
        """Convert frequency to note name (e.g., 'C4')."""
        if freq <= 0:
            return "C4"  # Default note for silence/errors
        
        return librosa.hz_to_note(freq)


# Module-level functions that wrap the class for backward compatibility

def process_pitch_data(pitched_data: PitchedData, real_bpm: float) -> List[MidiSegment]:
    """
    Process pitched data with probabilistic model to create better note segments.
    
    Args:
        pitched_data: PitchedData object with time, frequency and confidence values
        real_bpm: Real BPM value of the song
        
    Returns:
        List of MidiSegment objects
    """
    processor = ProbabilisticPitchProcessor(real_bpm)
    return processor.process_pitch_data(pitched_data)


def process_with_lyrics(pitched_data: PitchedData, transcribed_data, real_bpm: float) -> List[MidiSegment]:
    """
    Process pitched data and align with transcribed lyrics.
    
    Args:
        pitched_data: PitchedData object with time, frequency and confidence values
        transcribed_data: Transcription data with word timings
        real_bpm: Real BPM value of the song
        
    Returns:
        List of MidiSegment objects with properly aligned lyrics
    """
    processor = ProbabilisticPitchProcessor(real_bpm)
    return processor.process_with_lyrics(pitched_data, transcribed_data)


def enhance_with_original_audio(pitched_data_separated: PitchedData, 
                               pitched_data_original: PitchedData,
                               real_bpm: float) -> PitchedData:
    """
    Process pitch data from both separated and original audio.
    
    Args:
        pitched_data_separated: Pitch data from separated vocals
        pitched_data_original: Pitch data from original unseparated audio
        real_bpm: Real BPM value of the song
        
    Returns:
        Enhanced PitchedData with improved confidence scores
    """
    processor = ProbabilisticPitchProcessor(real_bpm)
    return processor.enhance_with_original_audio(pitched_data_separated, pitched_data_original)