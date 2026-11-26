"""Artifact contracts: invariants checked after every stage.

A reordered or new stage that breaks an invariant (unsorted notes, negative durations, overlaps,
a pitch track with mismatched arrays) is reported right where it happens instead of surfacing
as a mysterious score drop later. Default: violations are logged as warnings. Strict mode
(``ULTRASINGER_STRICT=1`` or ``Context.extra["strict"]``) raises instead; the tests use it.
"""

from __future__ import annotations

import math
import os

EPS = 1e-6


class ContractError(ValueError):
    pass


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def check_timed(items, what: str, allow_overlap: bool) -> list[str]:
    """Lists of objects with .start/.end in seconds (TranscribedData, MidiSegment)."""
    problems = []
    if not isinstance(items, list):
        return [f"{what}: expected a list, got {type(items).__name__}"]
    prev = None
    for i, it in enumerate(items):
        s, e = getattr(it, "start", None), getattr(it, "end", None)
        if not (_finite(s) and _finite(e)):
            problems.append(f"{what}[{i}]: non-numeric timing ({s}, {e})")
            continue
        if s < -EPS:
            problems.append(f"{what}[{i}]: negative start {s:.3f}")
        if e < s - EPS:
            problems.append(f"{what}[{i}]: end {e:.3f} before start {s:.3f}")
        if prev is not None:
            if s < prev.start - EPS:
                problems.append(f"{what}[{i}]: not sorted (start {s:.3f} < previous {prev.start:.3f})")
            elif not allow_overlap and s < prev.end - 1e-3:
                problems.append(f"{what}[{i}]: overlaps previous ({s:.3f} < {prev.end:.3f})")
        prev = it
        if len(problems) > 20:
            problems.append("... (more)")
            break
    return problems


def check_notes(notes) -> list[str]:
    problems = check_timed(notes, "notes", allow_overlap=False)
    for i, n in enumerate(notes if isinstance(notes, list) else []):
        if not getattr(n, "note", None):
            problems.append(f"notes[{i}]: no pitch")
        if getattr(n, "end", 0) - getattr(n, "start", 0) <= 0:
            problems.append(f"notes[{i}]: zero or negative length")
        if len(problems) > 20:
            break
    return problems


def check_pitch(p) -> list[str]:
    try:
        n = len(p.times)
        if len(p.frequencies) != n or len(p.confidence) != n:
            return [f"pitch: array lengths differ ({n}, {len(p.frequencies)}, {len(p.confidence)})"]
        t = list(p.times)
        if any(b < a for a, b in zip(t, t[1:])):
            return ["pitch: times not increasing"]
    except AttributeError as err:
        return [f"pitch: {err}"]
    return []


def check_tempo(t) -> list[str]:
    return [] if _finite(t) and float(t) > 0 else [f"tempo: {t!r} is not a positive number"]


def check_file(path) -> list[str]:
    return [] if isinstance(path, str) and os.path.isfile(path) else [f"file {path!r} does not exist"]


def check(artifact: str, value) -> list[str]:
    """Problems of one artifact (empty list = fine). Unknown kinds pass."""
    kind = artifact.split(".", 1)[0]
    if value is None:
        return []
    if kind == "words":
        # words may overlap (aligner output) and may be untimed (start = end = None) until
        # words.place_untimed; order must hold among the timed ones
        return check_timed([w for w in value if not (w.start is None and w.end is None)], kind, allow_overlap=True)
    if kind == "syllables":
        # syllables may overlap until syllables.trim_overlaps; order must hold
        return check_timed(value, kind, allow_overlap=True)
    if kind == "notes":
        return check_notes(value)
    if kind == "pitch":
        return check_pitch(value)
    if kind == "tempo":
        return check_tempo(value)
    if kind in ("audio", "file"):
        return check_file(value)
    return []


def strict_mode(ctx) -> bool:
    return bool(ctx.extra.get("strict")) or os.environ.get("ULTRASINGER_STRICT", "") not in ("", "0")
