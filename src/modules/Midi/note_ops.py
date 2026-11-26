"""Note-level operations on MidiSegment lists. All behaviour comes from explicit parameters."""

import librosa
import numpy as np

from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_length_calculator import get_sixteenth_note_second
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscribedData import TranscribedData


def apply_game_pitches(midi_segments, game_pitches, allowed_notes, key_min_offset: float):
    from modules.Audio.key_detector import quantize_note_to_key
    if len(midi_segments) != len(game_pitches):
        import logging
        logging.getLogger("ultrasinger.notes").warning("GAME pitch count mismatch, keeping pitch-track notes")
        return
    for seg, midi in zip(midi_segments, game_pitches):
        if midi is None:
            continue
        note = librosa.midi_to_note(int(round(midi)), unicode=False)
        if allowed_notes is not None:
            note = quantize_note_to_key(note, allowed_notes, midi, key_min_offset)
        seg.note = note


def snap_note_edges(midi_segments: list[MidiSegment], pitched_data: PitchedData, max_shift: float,
                    conf_threshold: float, min_gap: float, min_len: float) -> None:
    """Move note starts/ends that border a pause onto the voicing on/offset (within max_shift).
    Edges between touching notes are pitch changes and stay where they are."""
    times = np.asarray(pitched_data.times)
    voiced = (np.asarray(pitched_data.confidence) >= conf_threshold) & (np.asarray(pitched_data.frequencies) > 0)
    if len(times) < 2:
        return
    hop = float(np.median(np.diff(times)))

    def is_voiced(t):
        i = int(np.searchsorted(times, t))
        return 0 <= i < len(voiced) and voiced[i]

    for k, seg in enumerate(midi_segments):
        prev_end = midi_segments[k - 1].end if k > 0 else 0.0
        next_start = midi_segments[k + 1].start if k + 1 < len(midi_segments) else float("inf")
        if seg.start - prev_end > min_gap:
            t = seg.start
            if is_voiced(t):  # singing already started: move the start back to the onset
                while t - hop > max(prev_end + min_gap, seg.start - max_shift) and is_voiced(t - hop):
                    t -= hop
            else:  # nothing sung yet: move the start forward to the onset
                limit = min(seg.start + max_shift, seg.end - min_len)
                while t + hop < limit and not is_voiced(t):
                    t += hop
                if not is_voiced(t):
                    t = seg.start
            seg.start = t
        if next_start - seg.end > min_gap:
            t = seg.end
            if is_voiced(t):
                while t + hop < min(next_start - min_gap, seg.end + max_shift) and is_voiced(t + hop):
                    t += hop
            else:
                limit = max(seg.end - max_shift, seg.start + min_len)
                while t - hop > limit and not is_voiced(t):
                    t -= hop
                if not is_voiced(t):
                    t = seg.end
            seg.end = t


def apply_timing_calibration(midi_segments: list[MidiSegment], offset: float, end_extension: float) -> None:
    """Shift all notes by offset and lengthen them by end_extension, without overlaps."""
    for i, seg in enumerate(midi_segments):
        seg.start = max(0.0, seg.start + offset)
        end = seg.end + offset + end_extension
        if i + 1 < len(midi_segments):
            end = min(end, midi_segments[i + 1].start + offset)
        seg.end = max(end, seg.start + 0.01)


def merge_syllable_segments(midi_segments: list[MidiSegment],
                            transcribed_data: list[TranscribedData],
                            real_bpm: float, slides: bool, same_pitch_syllables: bool,
                            slide_max_semitones: int) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Merge sub-segments of a syllable where the pitch is the same

    This function handles three cases:
    1. Merge consecutive ~ segments with the SAME pitch (same note held)
    2. Detect and merge SLIDES: short ~ segments with ±1-2 semitone jumps between syllables
    3. Merge ANY consecutive segments (including regular syllables) with the SAME pitch into one word
    """

    sixteenth_note = get_sixteenth_note_second(real_bpm)
    # Slides are typically very short (1-2 16th notes)
    max_slide_duration = sixteenth_note * 2

    new_data = []
    new_midi_notes = []

    previous_data = None

    for i, data in enumerate(transcribed_data):
        # Check if previous element exists
        is_same_note = i > 0 and midi_segments[i].note == midi_segments[i - 1].note
        has_breath_pause = False

        if previous_data is not None:
            has_breath_pause = (data.start - previous_data.end) > sixteenth_note

        # Check if current word is a ~ segment (continuation marker)
        current_word_stripped = str(data.word).strip()
        is_tilde_segment = current_word_stripped == "~"

        # Slide detection: Detect short ~ segments with small pitch jumps
        is_potential_slide = False
        if i > 0 and is_tilde_segment:
            duration = data.end - data.start

            # Calculate pitch jump in semitones
            try:
                prev_midi = librosa.note_to_midi(midi_segments[i - 1].note)
                curr_midi = librosa.note_to_midi(midi_segments[i].note)
                semitone_diff = abs(curr_midi - prev_midi)

                # Slide: Short duration AND small pitch jump (1-2 semitones)
                is_potential_slide = (slides and
                                     duration <= max_slide_duration and
                                     semitone_diff <= slide_max_semitones and
                                     semitone_diff > 0)
            except:
                # Ignore slide detection on error
                pass

        # Check if current segment should be merged with previous due to same pitch
        should_merge_same_pitch = False
        if (same_pitch_syllables and
            i > 0 and
            previous_data is not None and
            not is_tilde_segment and  # Not a ~ segment
            is_same_note and
            not has_breath_pause and
            not previous_data.is_word_end):  # Don't merge across word boundaries
            should_merge_same_pitch = True

        should_merge_tilde = (is_tilde_segment and
                             previous_data is not None and
                             (is_same_note or is_potential_slide) and
                             not has_breath_pause)

        if should_merge_tilde:
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # For slides: Keep the original note (not the transition note)
            if is_potential_slide and not is_same_note:
                new_midi_notes[-1].note = midi_segments[i - 1].note

            # Take over space and word_end flag from current segment
            # "~ " means end of word - add space to previous segment
            if str(data.word).endswith(" "):
                if not new_data[-1].word.endswith(" "):
                    new_data[-1].word += " "
                if not new_midi_notes[-1].word.endswith(" "):
                    new_midi_notes[-1].word += " "
                new_data[-1].is_word_end = True

        elif should_merge_same_pitch:
            # Merge regular syllable with previous syllable (same pitch)
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # Check if current word has space at end before stripping
            has_space = str(data.word).endswith(" ")

            # Concatenate the words (remove trailing space from previous, add current word)
            prev_word = new_data[-1].word.rstrip()
            curr_word = data.word.rstrip()

            # Remove ~ from BOTH previous and current word if present
            if prev_word == "~":
                prev_word = ""
            if curr_word.startswith("~"):
                curr_word = curr_word[1:]

            # Concatenate
            new_data[-1].word = prev_word + curr_word
            new_midi_notes[-1].word = prev_word + curr_word

            # Preserve space at end if current segment had it
            if has_space:
                new_data[-1].word += " "
                new_midi_notes[-1].word += " "

            if data.is_word_end:
                new_data[-1].is_word_end = True
                new_data[-1].is_hyphen = False
            else:
                # Keep hyphen status if either segment was hyphenated
                new_data[-1].is_hyphen = new_data[-1].is_hyphen or data.is_hyphen

        else:
            # Add as new segment
            new_data.append(data)
            new_midi_notes.append(midi_segments[i])

        previous_data = data

    return new_midi_notes, new_data

