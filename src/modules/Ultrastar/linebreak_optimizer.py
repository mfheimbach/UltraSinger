"""Enhanced linebreak optimizer for UltraStar."""

import math
import re
from typing import List, Tuple

from modules.Midi.MidiSegment import MidiSegment
from modules.Ultrastar.coverter.ultrastar_converter import second_to_beat
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


class LinebreakOptimizer:
    """
    Enhanced linebreak optimizer with BPM adaptation and dynamic scoring.
    """

    def __init__(self, optimal_syllables=10, max_line_duration=5.0, min_line_duration=2.0, 
                 normal_bpm=120.0, syllable_tolerance=3.0, time_tolerance=1.0):
        """Initialize with configurable parameters."""
        self.optimal_syllables = optimal_syllables
        self.max_line_duration = max_line_duration
        self.min_line_duration = min_line_duration
        self.normal_bpm = normal_bpm
        self.syllable_tolerance = syllable_tolerance
        self.time_tolerance = time_tolerance
        
    def calculate_linebreaks(self, midi_segments: List[MidiSegment], gap: float, real_bpm: float, multiplication: float) -> List[Tuple[int, int]]:
        """
        Calculate optimal linebreaks with BPM adaptation and dynamic scoring.
        
        Args:
            midi_segments: List of MidiSegment objects
            gap: Gap value in seconds
            real_bpm: Real BPM value of the song
            multiplication: Multiplication factor
            
        Returns:
            List of tuples (position_index, show_next_beat)
        """
        print(f"{ULTRASINGER_HEAD} Optimizing linebreaks with BPM adaptation (Song BPM: {blue_highlighted(str(round(real_bpm, 1)))}, Reference BPM: {blue_highlighted(str(self.normal_bpm))})")
        
        if not midi_segments:
            return []
        
        # Calculate BPM factor for adaptation
        bpm_factor = math.sqrt(self.normal_bpm / real_bpm)
        adapted_syllables = max(5, min(15, round(self.optimal_syllables * bpm_factor)))
        
        print(f"{ULTRASINGER_HEAD} BPM-adapted target: {blue_highlighted(str(adapted_syllables))} syllables per line")
            
        # Find potential break points and score them
        scored_positions = []
        current_line_start = 0
        current_syllables = 0
        
        for i, segment in enumerate(midi_segments):
            # Skip if this is not a word end (middle of a word)
            if not segment.word.endswith(" ") and i < len(midi_segments) - 1:
                continue
                
            # Count syllables in this word
            word = segment.word.strip()
            syllables = self._count_syllables(word)
            current_syllables += syllables
            
            # Calculate current line duration
            line_duration = segment.end - midi_segments[current_line_start].start
            
            # Skip if not enough syllables or too short duration unless at end of sentence
            if (current_syllables < adapted_syllables / 2 or 
                line_duration < self.min_line_duration) and not word.endswith((".", "!", "?")):
                continue
                
            # Skip last position
            if i >= len(midi_segments) - 1:
                continue
                
            # Calculate score using bell curve functions
            syllable_score = self._calculate_syllable_score(current_syllables, adapted_syllables)
            duration_score = self._calculate_duration_score(line_duration)
            
            # Add bonuses for natural breaks
            punctuation_bonus = 0.5 if word.endswith((".", ",", "!", "?", ";", ":")) else 0.0
            
            # Combined score
            score = (syllable_score * 0.4) + (duration_score * 0.4) + punctuation_bonus
            
            # Calculate beat position for display
            show_next = second_to_beat(segment.end - gap, real_bpm) * multiplication
            
            scored_positions.append((i, round(show_next), score, current_syllables, line_duration))
            
        # Sort by score (highest first)
        scored_positions.sort(key=lambda x: x[2], reverse=True)
        
        # Select best breaks ensuring minimum spacing
        selected_breaks = []
        used_positions = set()
        min_spacing_syllables = max(3, adapted_syllables // 3)
        
        # First pass: select high-scoring obvious breaks
        for pos, beat, score, syllables, duration in scored_positions:
            if score > 0.8:  # Obvious good breaks
                selected_breaks.append((pos, beat))
                used_positions.add(pos)
                
                # Mark nearby positions as used to ensure spacing
                for nearby in range(max(0, pos - min_spacing_syllables), min(len(midi_segments), pos + min_spacing_syllables + 1)):
                    used_positions.add(nearby)
        
        # Second pass: add breaks to ensure no excessively long lines
        current_line_start = 0
        current_syllables = 0
        current_duration = 0
        
        for i, segment in enumerate(midi_segments):
            if i in used_positions or i == len(midi_segments) - 1:
                # Reset counters at existing breaks
                current_line_start = i + 1
                current_syllables = 0
                current_duration = 0
                continue
                
            # Count syllables in this word
            word = segment.word.strip()
            syllables = self._count_syllables(word)
            current_syllables += syllables
            
            # Update duration
            if i > 0:
                current_duration = segment.end - midi_segments[current_line_start].start
            
            # Check if we need a forced break due to long line
            if ((current_syllables >= adapted_syllables * 1.5 or 
                current_duration >= self.max_line_duration * 0.9) and 
                segment.word.endswith(" ")):
                
                # Find this position in scored_positions to get the beat
                for pos, beat, _, _, _ in scored_positions:
                    if pos == i:
                        selected_breaks.append((i, beat))
                        used_positions.add(i)
                        
                        # Reset counters
                        current_line_start = i + 1
                        current_syllables = 0
                        current_duration = 0
                        break
        
        # Sort breaks by position
        selected_breaks.sort()
        
        return selected_breaks
    
    def _calculate_syllable_score(self, current_syllables: int, optimal_syllables: int) -> float:
        """Calculate score using bell curve based on syllable count."""
        # Bell curve function: exp(-((x-optimal)/tolerance)²)
        return math.exp(-((current_syllables - optimal_syllables) / self.syllable_tolerance) ** 2)
    
    def _calculate_duration_score(self, duration: float) -> float:
        """Calculate score using bell curve based on line duration."""
        optimal_duration = (self.min_line_duration + self.max_line_duration) / 2
        return math.exp(-((duration - optimal_duration) / self.time_tolerance) ** 2)
        
    def _count_syllables(self, word: str) -> int:
        """
        Count the number of syllables in a word.
        """
        # Handle empty words
        if not word:
            return 0
            
        # Handle hyphenated words (marked with ~)
        if word.startswith("~"):
            return 1
            
        # Clean word
        word = re.sub(r'[^\w\s]', '', word.lower())
        
        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_is_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
            
        # Adjust for common patterns
        if word.endswith('e'):
            count -= 1
            
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
            
        # Ensure at least 1 syllable for any non-empty word
        return max(count, 1)