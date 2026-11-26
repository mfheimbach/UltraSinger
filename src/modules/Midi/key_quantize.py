"""Notes of a key and quantization of out-of-key notes (music theory, no audio)."""

import librosa

from modules.Audio.key_detector import MAJOR_SCALE, MINOR_SCALE, NOTE_NAMES

OCTAVE_RANGE = (0, 8)   # octaves tried for candidates (note names C0..B8)


def get_allowed_notes_for_key(key_note: str, mode: str) -> set[str]:
    """
    Get all allowed note names for a given key across all octaves.

    Args:
        key_note: Root note (e.g., 'C', 'D#', 'F')
        mode: 'major' or 'minor'

    Returns:
        Set of allowed note names (e.g., {'C', 'D', 'E', ...})
    """
    root_idx = NOTE_NAMES.index(key_note)
    scale = MAJOR_SCALE if mode == 'major' else MINOR_SCALE

    allowed_notes = set()
    for interval in scale:
        note_idx = (root_idx + interval) % 12
        allowed_notes.add(NOTE_NAMES[note_idx])

    return allowed_notes


def quantize_note_to_key(note: str, allowed_notes: set[str], pitch_hint: float | None,
                         min_offset: float) -> str:
    """
    Quantize a note to the nearest allowed note in the key.

    Args:
        note: MIDI note name (e.g., 'C4', 'D#5')
        allowed_notes: Set of allowed note names without octave
        pitch_hint: Optional unrounded MIDI pitch of the note. Breaks ties between
            two equally distant allowed notes; without it the lower one wins.
        min_offset: Limited quantization. With a pitch_hint, an out-of-key note is
            only moved if the measured pitch is at least this many semitones away
            from the note's centre (i.e. ambiguous). 0 = always quantize.

    Returns:
        Quantized note name
    """
    # Parse note and octave
    if len(note) == 0:
        return note

    # Extract note name and octave
    if len(note) >= 2 and note[-2] == '#':
        note_name = note[:-1]
        octave = note[-1]
    else:
        note_name = note[:-1] if note[-1].isdigit() else note
        octave = note[-1] if note[-1].isdigit() else ''

    # If already in key, return as is
    if note_name in allowed_notes:
        return note

    # Find nearest allowed note
    try:
        note_midi = librosa.note_to_midi(note)
    except:
        return note

    # Limited quantization: a note measured close to its own centre is a deliberate
    # chromatic note, leave it alone
    if pitch_hint is not None and min_offset > 0 and abs(pitch_hint - round(pitch_hint)) < min_offset:
        return note

    # Try all allowed notes in nearby octaves
    min_distance = float('inf')
    best_note = note

    best_midi = None
    reference = pitch_hint if pitch_hint is not None else note_midi

    # sorted(): set order depends on the per-process hash seed, so iterating the
    # set directly made tie-breaking (and the output) differ from run to run
    for allowed_note_name in sorted(allowed_notes):
        # Try same octave and adjacent octaves
        for oct_offset in [-1, 0, 1]:
            if not octave:
                test_octave = 4 + oct_offset
            else:
                test_octave = int(octave) + oct_offset

            if test_octave < OCTAVE_RANGE[0] or test_octave > OCTAVE_RANGE[1]:
                continue

            test_note = f"{allowed_note_name}{test_octave}"

            try:
                test_midi = librosa.note_to_midi(test_note)
                distance = abs(test_midi - note_midi)

                better = distance < min_distance
                if not better and distance == min_distance and best_midi is not None:
                    # tie: prefer the candidate closer to the measured pitch, then the lower one
                    d_new, d_old = abs(test_midi - reference), abs(best_midi - reference)
                    better = d_new < d_old or (d_new == d_old and test_midi < best_midi)
                if better:
                    min_distance = distance
                    best_note = test_note
                    best_midi = test_midi
            except:
                continue

    #log.info(f"Moved {blue_highlighted(note)} to {blue_highlighted(best_note)}")
    return best_note