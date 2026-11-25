"""Standalone UltraStar txt reader for evaluation.

Deliberately independent of UltraSinger's own parser, so a bug there
can't hide in both the prediction and the ground truth.

Times are in seconds. Pitches are kept in the file's own UltraStar
units; both ground truth and prediction use the same convention, and
the chroma metrics ignore octaves anyway.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

NOTE_RE = re.compile(r"^([:*FRG])[ \t]+(-?\d+)[ \t]+(\d+)[ \t]+(-?\d+)(?:[ \t](.*))?$")
# Note types whose pitch counts. F = freestyle, R/G = rap (pitch not scored in USDX)
PITCHED_TYPES = {":", "*"}
GOLDEN_TYPES = {"*", "G"}


@dataclass
class Note:
    kind: str
    start: float
    end: float
    pitch: int
    text: str
    line_start: bool = False

    @property
    def pitched(self) -> bool:
        return self.kind in PITCHED_TYPES

    @property
    def golden(self) -> bool:
        return self.kind in GOLDEN_TYPES


@dataclass
class Song:
    path: str
    headers: dict = field(default_factory=dict)
    notes: list[Note] = field(default_factory=list)
    line_breaks: list[float] = field(default_factory=list)
    is_duet: bool = False

    @property
    def artist(self) -> str:
        return self.headers.get("ARTIST", "").strip()

    @property
    def title(self) -> str:
        return self.headers.get("TITLE", "").strip()

    @property
    def language(self) -> str:
        return self.headers.get("LANGUAGE", "").strip()

    def audio_path(self) -> str | None:
        name = self.headers.get("AUDIO") or self.headers.get("MP3")
        if not name:
            return None
        return os.path.join(os.path.dirname(self.path), name.strip())

    def words(self) -> list[str]:
        """Reassemble words from syllables. UltraStar marks word starts with a leading space."""
        words, cur = [], ""
        for n in self.notes:
            t = n.text.replace("~", "")
            if (t.startswith(" ") or n.line_start) and cur:
                words.append(cur)
                cur = ""
            cur += t.strip()
            if t.endswith(" ") and cur:
                words.append(cur)
                cur = ""
        if cur:
            words.append(cur)
        return [w for w in words if w]


def _num(value: str) -> float:
    return float(value.strip().replace(",", "."))


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def read(path: str) -> Song:
    song = Song(path=path)
    text = _read_text(path)
    lines = text.splitlines()

    for line in lines:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            song.headers[key.strip().upper()] = value.rstrip("\r\n")

    bpm = _num(song.headers.get("BPM", "0"))
    if bpm <= 0:
        raise ValueError(f"{path}: missing or invalid #BPM")
    gap = _num(song.headers.get("GAP", "0")) / 1000.0
    relative = song.headers.get("RELATIVE", "").strip().lower() in ("yes", "true", "1")
    beat_s = 60.0 / (bpm * 4)  # UltraStar BPM is quarter of the beat grid

    offset = 0  # beat offset for relative mode
    player = 1
    new_line = True
    for line in lines:
        if not line or line.startswith("#"):
            continue
        if line.startswith("P"):
            song.is_duet = True
            m = re.match(r"P\s*(\d+)", line)
            player = int(m.group(1)) if m else player
            continue
        if player != 1:
            continue  # duets: evaluate the first singer only
        if line.startswith("E"):
            break
        if line.startswith("-"):
            parts = line[1:].split()
            new_line = True
            if parts:
                song.line_breaks.append(gap + (int(parts[0]) + offset) * beat_s)
                if relative:
                    offset += int(parts[1]) if len(parts) > 1 else int(parts[0])
            continue
        m = NOTE_RE.match(line.rstrip("\r\n"))
        if not m:
            continue
        kind, start, dur, pitch, syl = m.groups()
        start_b = int(start) + offset
        song.notes.append(
            Note(
                kind=kind,
                start=gap + start_b * beat_s,
                end=gap + (start_b + int(dur)) * beat_s,
                pitch=int(pitch),
                text=syl or "",
                line_start=new_line,
            )
        )
        new_line = False
    song.notes.sort(key=lambda n: n.start)
    return song


def find_song_txt(folder: str) -> str | None:
    """Pick the UltraStar txt in a song folder (skips license/instrumental helpers)."""
    candidates = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".txt"):
            continue
        p = os.path.join(folder, name)
        head = _read_text(p)[:2000]
        if "#TITLE" in head.upper() and "#BPM" in head.upper():
            candidates.append(p)
    if not candidates:
        return None
    # Prefer song.txt / non-instrumental variants
    candidates.sort(key=lambda p: ("instrumental" in p.lower(), "[" in os.path.basename(p), len(p)))
    return candidates[0]
