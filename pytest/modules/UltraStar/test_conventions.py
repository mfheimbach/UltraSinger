"""Tests for the community conventions (modules/Ultrastar/conventions.py)."""

import tempfile

from modules.Ultrastar.conventions import apply_conventions

CHART = "\n".join([
    "#TITLE:T", "#ARTIST:A", "#BPM:300", "#GAP:0",
    ": 0 4 10 it's ", ": 4 4 12 me", "- 8",
    ": 40 2 10 now ", ": 42 3 10 we", "- 45",
    ": 545 4 10 la", "E", ""])


def _best_rules() -> dict:
    """The line-break rules as the preset sets them (values live in presets/best.yaml)."""
    from pipeline.core import resolve_preset
    from pipeline.run import preset_path
    p = resolve_preset(preset_path("best"))["params"]["write.conventions"]
    return {k[len("break_"):]: v for k, v in p.items() if k.startswith("break_")}


def _run(text, **kw):
    with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(text)
        path = f.name
    args = {"touching": True, "line_breaks": True, "apostrophes": True, "capitalise": True,
            "break_rules": _best_rules(), **kw}
    counts = apply_conventions(path, **args)
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines(), counts


def test_touching_notes_get_one_free_beat():
    lines, counts = _run(CHART, line_breaks=False, apostrophes=False, capitalise=False)
    assert ": 0 3 10 it's " in lines and ": 40 1 10 now " in lines
    assert counts["touching_notes"] == 2


def test_line_break_timing_yass():
    lines, _ = _run(CHART, touching=False, apostrophes=False, capitalise=False)
    # gap 8 -> 40 is 32 beats (1.6 s at BPM 300): more than 16 beats -> end + 10
    assert "- 18" in lines
    # gap 45 -> 545 is 25 s: 2 s after the line end (2 s = 40 beats)
    assert "- 85" in lines


def test_apostrophes_and_capitalisation():
    lines, counts = _run(CHART, touching=False, line_breaks=False)
    assert ": 0 4 10 It’s " in lines and ": 40 2 10 Now " in lines and ": 545 4 10 La" in lines
    assert counts["apostrophes"] == 1 and counts["capitalised_lines"] == 3
