"""Unit tests for the pipeline's own module logic (note operations, line breaks, writer grid, key
quantization, contracts, runtime estimates, log de-duplication). Synthetic inputs, no models."""

import logging
import os
import tempfile

import pytest

os.environ["ULTRASINGER_RUNTIME_LOG"] = os.path.join(tempfile.gettempdir(), "ultrasinger_test_runtimes.jsonl")

from modules.Midi.MidiSegment import MidiSegment  # noqa: E402
from modules.Midi import note_ops  # noqa: E402
from modules.Speech_Recognition.TranscribedData import TranscribedData  # noqa: E402
from modules.Ultrastar import line_breaks as lb  # noqa: E402
from pipeline import contracts  # noqa: E402


def seg(note, start, end, word):
    return MidiSegment(note, start, end, word)


def syl(word, start, end):
    t = TranscribedData()
    t.word, t.start, t.end = word, start, end
    return t


# ----------------------------------------------------------------------------- note operations

def test_trim_overlaps_cuts_ends_to_next_start_and_keeps_input():
    items = [syl("a", 0.0, 1.5), syl("b", 1.0, 2.0), syl("c", 2.5, 3.0)]
    out = note_ops.trim_overlaps(items)
    assert [(x.start, x.end) for x in out] == [(0.0, 1.0), (1.0, 2.0), (2.5, 3.0)]
    assert items[0].end == 1.5                      # copies, the input is untouched
    assert contracts.check_timed(out, "syllables", allow_overlap=False) == []


def test_calibration_shifts_extends_and_never_overlaps():
    notes = [seg("C4", 1.0, 1.5, "a "), seg("D4", 1.6, 2.0, "b "), seg("E4", 2.004, 2.5, "c ")]
    note_ops.apply_timing_calibration(notes, offset=-0.1, end_extension=0.2, min_len=0.01)
    assert notes[0].start == pytest.approx(0.9) and notes[0].end == pytest.approx(1.5)   # capped by next onset
    assert notes[1].end == pytest.approx(1.904)                                          # next onset wins over min_len
    assert notes[2].end == pytest.approx(2.6)
    assert contracts.check_timed(notes, "notes", allow_overlap=False) == []


def test_calibration_min_len():
    notes = [seg("C4", 1.0, 1.0, "a ")]
    note_ops.apply_timing_calibration(notes, offset=0.0, end_extension=0.0, min_len=0.05)
    assert notes[0].end == pytest.approx(1.05)


def test_merge_same_pitch_pieces_of_a_syllable():
    notes = [seg("C4", 0.0, 0.2, "la"), seg("C4", 0.2, 0.4, "~ "), seg("E4", 0.5, 0.7, "lo ")]
    syls = [syl("la", 0.0, 0.2), syl("~ ", 0.2, 0.4), syl("lo ", 0.5, 0.7)]
    out, _ = note_ops.merge_syllable_segments(notes, syls, 120.0, slides=False, same_pitch_syllables=True,
                                              slide_max_semitones=2, slide_max_sixteenths=2.0,
                                              breath_pause_sixteenths=1.0)
    assert [(n.note, n.start, n.end) for n in out] == [("C4", 0.0, 0.4), ("E4", 0.5, 0.7)]


# ----------------------------------------------------------------------------- line breaks

def test_pause_threshold():
    assert lb.pause_threshold([0.1, 0.2], min_gaps=5) == 0.0
    assert lb.pause_threshold([0.1, 0.1, 0.1, 0.1, 2.0], min_gaps=5) == pytest.approx(0.1)


def test_pause_breaks_only_after_word_ends():
    notes = [seg("C4", 0.0, 0.5, "one "), seg("C4", 0.6, 1.0, "two "), seg("C4", 3.0, 3.4, "three "),
             seg("C4", 3.5, 3.9, "fo"), seg("C4", 6.0, 6.3, "ur "), seg("C4", 6.4, 6.8, "five ")]
    lb.plan_pause_breaks(notes, min_gaps=5)
    starts = [n.word.strip() for n in notes if n.line_start]
    assert starts == ["three", "five"]   # gap inside "four" breaks after its word end


def test_planned_breaks_follow_pauses_and_punctuation(monkeypatch):
    words = ["hello ", "world ", "again ", "and ", "again "]
    notes = [seg("C4", i * 0.5, i * 0.5 + 0.4, w) for i, w in enumerate(words)]
    notes[3].start, notes[3].end, notes[4].start, notes[4].end = 4.0, 4.4, 4.5, 4.9   # long pause before "and"
    monkeypatch.setattr(lb, "punctuation_probabilities", lambda w, *a: __import__("numpy").array(
        [0.01, 0.01, 0.9, 0.01, 0.01]))
    m = lb.LineBreakModel(intercept=0.0, w_pause=1.0, w_punct=1.0, chars_mean=15.0, chars_sd=10.0, chars_max=45,
                          dur_median=1.5, dur_log_sd=1.0, prior_weight=0.1, max_words=16, pause_floor=0.02,
                          prob_floor=0.001, dur_floor=0.05, punct_chunk=150, punct_context=30)
    lb.plan_line_breaks(notes, "fake", "cpu", m)
    assert [n.word.strip() for n in notes if n.line_start] == ["and"]


# ----------------------------------------------------------------------------- writer grid, conventions, key

def test_bpm_multiplier_stays_below_max_and_is_at_least_one():
    from modules.Ultrastar.writer import get_multiplier
    assert get_multiplier(100.0, 400.0) == 3
    assert get_multiplier(150.0, 400.0) == 2
    assert get_multiplier(500.0, 400.0) == 1     # the old code returned 0 here


def test_line_break_rules_come_from_parameters():
    from modules.Ultrastar.conventions import fix_line_break_timing
    rules = {"long_gap_s": 4.0, "long_offset_s": 2.0, "mid_gap_s": 2.0, "mid_offset_s": 1.0, "touch_beats": 1,
             "beat_rules": [[8, 2], [12, 3], [16, 4]], "far_offset_beats": 10}
    body = [["note", [":", 0, 4, 0, "a "]], ["break", 4], ["note", [":", 10, 4, 0, "b "]]]
    fix_line_break_timing(body, 300.0, rules)
    assert body[1][1] == 8                       # gap 6 beats -> 2 before the next line
    body[1][1] = 4
    fix_line_break_timing(body, 300.0, {**rules, "beat_rules": [[8, 5]]})
    assert body[1][1] == 5


def test_key_quantization_respects_min_offset():
    from modules.Midi.key_quantize import get_allowed_notes_for_key, quantize_note_to_key
    allowed = get_allowed_notes_for_key("C", "major")
    assert quantize_note_to_key("C#4", allowed, 61.0, 0.0) in ("C4", "D4")
    # a confidently sung C# (pitch hint right on it) stays when the offset rule demands more evidence
    assert quantize_note_to_key("C#4", allowed, 61.0, 0.4) == "C#4"


# ----------------------------------------------------------------------------- infrastructure

def test_contracts_flag_unsorted_and_overlapping_notes():
    notes = [seg("C4", 1.0, 2.0, "a"), seg("C4", 0.5, 0.8, "b"), seg("C4", 0.7, 0.9, "c")]
    problems = contracts.check_timed(notes, "notes", allow_overlap=False)
    assert any("not sorted" in p for p in problems) and any("overlaps" in p for p in problems)


def test_runtime_fit_is_fixed_plus_proportional():
    from pipeline import runtimes
    a, b = runtimes._fit([(30, 4.0), (60, 7.0), (120, 13.0)])
    assert a == pytest.approx(1.0) and b == pytest.approx(0.1)
    a, b = runtimes._fit([(60, 3.0)])                # one length only: proportional
    assert a == 0.0 and b == pytest.approx(0.05)
    a, b = runtimes._fit([(30, 1.0), (60, 4.0)])     # negative intercept: refit through the origin
    assert a == 0.0


def test_repeated_contract_warnings_show_once_then_count():
    from pipeline import log as plog
    f = plog._ConsoleFilter()
    lg = logging.getLogger("ultrasinger.stage.notes.merge")
    recs = [lg.makeRecord(lg.name, logging.WARNING, __file__, 1, f"notes.merge -> notes: notes[{i}]: overlaps "
                          f"previous ({i}.1 < {i}.2)", None, None, extra={"event": "contract"}) for i in range(4)]
    shown = [f.filter(r) for r in recs]
    assert shown == [True, False, False, False]
    assert plog.repeats_summary()[0].startswith("3x more")


def test_untimed_words_stay_none_until_the_fallback_step_places_them(caplog):
    from modules.Speech_Recognition.Whisper import convert_to_transcribed_data
    from pipeline.core import Context
    from pipeline.stages import PlaceUntimedWords
    words = convert_to_transcribed_data({"segments": [{"words": [
        {"word": "1", "score": 0.0}, {"word": "a", "start": 1.0, "end": 1.4, "score": 0.9},
        {"word": "2", "score": 0.0}, {"word": "b", "start": 2.0, "end": 2.4, "score": 0.9}]}]})
    assert words[0].start is None and words[2].end is None
    assert contracts.check("words", words) == []          # untimed words are allowed in words
    step = PlaceUntimedWords({"gap": 0.1, "length": 0.1})
    with caplog.at_level(logging.WARNING):
        out = step.run(Context(cache_dir=tempfile.gettempdir(), output_dir=tempfile.gettempdir()), words)["words"]
    assert [x for w in out for x in (w.start, w.end)] == pytest.approx([0.0, 0.1, 1.0, 1.4, 1.5, 1.6, 2.0, 2.4])
    assert any(r.getMessage().startswith("FALLBACK") for r in caplog.records)
    assert words[0].start is None                           # input untouched
    quiet = step.run(Context(cache_dir=tempfile.gettempdir(), output_dir=tempfile.gettempdir()), out)["words"]
    assert [(w.start, w.end) for w in quiet] == [(w.start, w.end) for w in out]


# ----------------------------------------------------------------------------- eval metrics

def test_boundary_score_quadratic_and_one_to_one():
    import sys
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
    from metrics import boundary_score
    g = np.array([1.0, 2.0, 3.0])
    assert boundary_score(g, g) == 1.0
    assert boundary_score(g, g + 0.05, tau=0.1) == pytest.approx(0.75)          # (0.5)^2 per pair
    assert boundary_score(g, g + 0.2, tau=0.1) == 0.0                           # beyond tau: unpaired
    assert boundary_score(g, np.array([1.0, 2.0]), tau=0.1) == pytest.approx(2 / 3)   # one missed
    assert boundary_score(g, np.array([1.0, 1.01, 2.0, 3.0]), tau=0.1) == pytest.approx(3 / 4)  # one extra


def test_hyphenate_pyphen_backend_offline():
    from types import SimpleNamespace
    from modules.Speech_Recognition.hyphenation import hyphenate_each_word
    words = [SimpleNamespace(word=w) for w in ["Cendrillon,", "la"]]
    out = hyphenate_each_word("fr", words, "pyphen")
    assert out[0][0] == "Cen" and out[0][-1].endswith(",") and out[1] is None
    import pytest
    with pytest.raises(ValueError):
        hyphenate_each_word("fr", words, "nope")
