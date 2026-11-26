"""MIDI file of the notes (write.midi)."""

import logging
import os

import librosa
import pretty_midi
import unidecode

from modules.Midi.MidiSegment import MidiSegment
from modules.console_colors import blue_highlighted

log = logging.getLogger("ultrasinger.midi_writer")


def write_midi_file(real_bpm: float, song_output: str, midi_segments: list[MidiSegment], basename_without_ext: str,
                    velocity: int) -> str:
    """Notes + lyrics (ASCII-approximated for MIDI) -> <basename>.mid. Returns the path."""
    log.info(f"Creating Midi with {blue_highlighted('pretty_midi')}")
    instrument = pretty_midi.Instrument(program=0, name="Vocals")
    for seg in midi_segments:
        instrument.notes.append(pretty_midi.Note(velocity, librosa.note_to_midi(seg.note), seg.start, seg.end))
    midi = pretty_midi.PrettyMIDI(initial_tempo=real_bpm)
    for seg in midi_segments:
        midi.lyrics.append(pretty_midi.Lyric(text=unidecode.unidecode(seg.word), time=seg.start))
    midi.instruments.append(instrument)
    path = os.path.join(song_output, f"{basename_without_ext}.mid")
    midi.write(path)
    return path
