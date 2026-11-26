from dataclasses import dataclass


@dataclass
class MidiSegment:
  note: str
  start: float
  end: float
  word: str
  line_start: bool = False  # first note of a planned line
  pitch: float | None = None  # unrounded MIDI pitch the note came from (key quantization uses it)
