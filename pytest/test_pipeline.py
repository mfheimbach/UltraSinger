"""Pipeline tests: core mechanics (validation, caching, lineage), contracts, and a golden test of
the deterministic tail (notes -> chart) on a fixed fixture. No models, no network: runs in seconds.

Update the golden chart after a deliberate change:  UPDATE_GOLDEN=1 pytest pytest/test_pipeline.py
"""

import dataclasses
import os
import pickle
import tempfile

import pytest

# test runs must not end up in the real runtime log
os.environ["ULTRASINGER_RUNTIME_LOG"] = os.path.join(tempfile.gettempdir(), "ultrasinger_test_runtimes.jsonl")

from pipeline import stages  # noqa: F401  (registers stages)
from pipeline import contracts
from pipeline.core import Context, Pipeline, Stage, StepSpec, load_preset, resolve_preset, stage
from pipeline.run import INITIAL, preset_path

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "fixtures", "golden_tail.pkl")
GOLDEN = os.path.join(HERE, "fixtures", "golden_tail.txt")
PRESETS = ["best", "bench", "cpu_fullsong", "repitch", "oracle_lyrics"]


# ----------------------------------------------------------------------------- toy stages

CALLS = []


@stage
class _Upper(Stage):
    name = "test.upper"
    inputs = {"text": "text"}
    outputs = {"text": "text"}

    @dataclasses.dataclass
    class Params:
        times: int = 1

    def run(self, ctx, text):
        CALLS.append("upper")
        return {"text": text.upper() * self.params.times}


@stage
class _Strip(Stage):
    name = "test.strip"
    inputs = {"text": "text"}
    outputs = {"text": "text"}

    def run(self, ctx, text):
        CALLS.append("strip")
        return {"text": text.strip()}


def _ctx(tmp, **kw):
    return Context(cache_dir=tmp, output_dir=tmp, verbose=False, **kw)


def test_presets_validate():
    for name in PRESETS:
        assert load_preset(preset_path(name)).validate(INITIAL) == [], name


def test_validation_reports_missing_inputs():
    p = Pipeline([StepSpec(use="notes.calibrate")], params=resolve_preset(preset_path("best"))["params"])
    assert p.validate({"audio.mix"})


def test_every_stage_value_lives_in_best():
    import dataclasses
    from pipeline.core import REGISTRY
    table = resolve_preset(preset_path("best"))["params"]
    for name, cls in REGISTRY.items():
        if name.startswith("test."):
            continue
        fields = {f.name for f in dataclasses.fields(cls.Params)}
        assert set(table.get(name, {})) == fields, name
        assert all(f.default is dataclasses.MISSING for f in dataclasses.fields(cls.Params)), name
    with pytest.raises(ValueError):
        Pipeline([StepSpec(use="notes.calibrate")])  # no values -> loud failure


def test_inheritance_and_sources():
    p = load_preset(preset_path("cpu_fullsong"), ["notes.calibrate.offset=-0.05"])
    cfg = {row["step"]: row for row in p.effective_config()}
    assert cfg["separate"]["params"]["model"] == "htdemucs" and cfg["separate"]["sources"]["model"] == "cpu_fullsong"
    assert cfg["separate"]["sources"]["seed"] == "best"
    assert cfg["notes.calibrate"]["params"]["offset"] == -0.05
    assert cfg["notes.calibrate"]["sources"]["offset"] == "override"


def test_text_stages_reorder_and_cache():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        a = Pipeline([StepSpec(use="test.strip"), StepSpec(use="test.upper")]).run(_ctx(tmp), {"text": " ab "})
        b = Pipeline([StepSpec(use="test.upper"), StepSpec(use="test.strip")]).run(_ctx(tmp), {"text": " ab "})
        assert a["text"] == b["text"] == "AB"
        assert [s["step"] for s in a["_lineage"]["text"]] == ["test.strip", "test.upper"]
        assert [s["step"] for s in b["_lineage"]["text"]] == ["test.upper", "test.strip"]
        CALLS.clear()
        Pipeline([StepSpec(use="test.strip"), StepSpec(use="test.upper")]).run(_ctx(tmp), {"text": " ab "})
        assert CALLS == []  # everything from the cache


def test_cache_by_lineage_vs_content():
    for mode, expected in (("lineage", 2), ("content", 1)):
        with tempfile.TemporaryDirectory() as tmp:
            CALLS.clear()
            Pipeline([StepSpec(use="test.upper")]).run(_ctx(tmp, cache_by=mode), {"text": "ab"})
            # an upstream step that changes nothing: lineage keys rerun, content keys reuse
            Pipeline([StepSpec(use="test.strip"), StepSpec(use="test.upper")]).run(_ctx(tmp, cache_by=mode),
                                                                                  {"text": "ab"})
            assert CALLS.count("upper") == expected, mode


def test_params_change_cache_key():
    with tempfile.TemporaryDirectory() as tmp:
        CALLS.clear()
        Pipeline([StepSpec(use="test.upper")]).run(_ctx(tmp), {"text": "ab"})
        out = Pipeline([StepSpec(use="test.upper", params={"times": 2})]).run(_ctx(tmp), {"text": "ab"})
        assert out["text"] == "ABAB" and CALLS.count("upper") == 2


def test_overrides():
    p = load_preset(preset_path("bench"), ["notes.calibrate.offset=-0.05", "-notes.snap_edges"])
    ids = [s.id for s in p.stages]
    assert "notes.snap_edges" not in ids
    assert next(s for s in p.stages if s.id == "notes.calibrate").params.offset == -0.05
    with pytest.raises(ValueError):
        load_preset(preset_path("bench"), ["-no.such.step"])
    with pytest.raises(ValueError):
        load_preset(preset_path("bench"), ["notes.calibrate.nonsense=1"])


# ----------------------------------------------------------------------------- contracts

class _T:
    def __init__(self, start, end, note="C4"):
        self.start, self.end, self.note = start, end, note


def test_contracts():
    assert contracts.check("notes", [_T(0, 1), _T(1, 2)]) == []
    assert contracts.check("notes", [_T(0, 1.5), _T(1, 2)])            # overlap
    assert contracts.check("notes", [_T(1, 2), _T(0, 0.5)])            # unsorted
    assert contracts.check("notes", [_T(1, 0.5)])                      # negative length
    assert contracts.check("syllables", [_T(0, 1.2), _T(1, 2)]) == []  # small overlaps allowed
    assert contracts.check("tempo", 0) and contracts.check("tempo", 120.0) == []


# ----------------------------------------------------------------------------- golden tail

TAIL = ["notes.from_pitch", "notes.game_pitch", "notes.merge", "notes.snap_edges", "notes.calibrate", "notes.pause_breaks",
        "write.ultrastar", "write.conventions"]


def test_golden_tail(monkeypatch):
    from modules.media_info import MediaInfo
    monkeypatch.setenv("ULTRASINGER_STRICT", "1")
    with open(FIXTURE, "rb") as f:
        fx = pickle.load(f)
    steps = [StepSpec(use=u, params={"stems": False} if u == "write.ultrastar" else {}) for u in TAIL]
    store = {"syllables": fx["syllables"], "aux.game": fx["game"], "pitch": fx["pitch"], "key": fx["key"],
             "tempo": fx["tempo"], "language": fx["language"],
             "media": MediaInfo(artist="Golden", title="Fixture", audio_extension="wav")}
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(cache_dir=tmp, output_dir=tmp, basename="Golden - Fixture", verbose=False)
        out = Pipeline(steps, params=resolve_preset(preset_path("best"))["params"]).run(ctx, store)
        with open(out["file.txt"], encoding="utf-8") as f:
            body = [line for line in f.read().splitlines() if not line.startswith(("#VERSION", "#CREATOR", "#COMMENT"))]
    if os.environ.get("UPDATE_GOLDEN"):
        with open(GOLDEN, "w", encoding="utf-8") as f:
            f.write("\n".join(body) + "\n")
    with open(GOLDEN, encoding="utf-8") as f:
        assert body == f.read().splitlines()


# ----------------------------------------------------------------------------- order flexibility

INIT = {"audio.mix", "media"}


def test_move_and_insert_overrides():
    p = load_preset(preset_path("bench"), ["words.strip_punctuation.after=syllables.silence_split",
                                           "+write.midi@after=write.ultrastar"])
    ids = [s.id for s in p.stages]
    assert ids.index("words.strip_punctuation") == ids.index("syllables.silence_split") + 1
    assert ids.index("write.midi") == ids.index("write.ultrastar") + 1
    assert p.validate(INIT) == []
    with pytest.raises(ValueError):
        load_preset(preset_path("bench"), ["+write.midi@after=no.such.step"])


def test_text_stage_binds_to_current_text_level():
    before = load_preset(preset_path("bench"))
    after = load_preset(preset_path("bench"), ["words.strip_punctuation.after=syllables.hyphenate"])
    before.validate(INIT)
    after.validate(INIT)
    get = lambda p: next(s for s in p.stages if s.id == "words.strip_punctuation")
    assert get(before).input_names["words"] == "words" and get(before).output_names["words"] == "words"
    assert get(after).input_names["words"] == "syllables" and get(after).output_names["words"] == "syllables"


def test_where_lists_valid_positions():
    p = load_preset(preset_path("bench"))
    positions = p.where("notes.calibrate", INIT)
    ids = [s.id for s in p.stages if s.id != "notes.calibrate"]
    valid_after = {ids[i - 1] for i in positions if i}
    assert "notes.from_pitch" in valid_after and "transcribe.whisper" not in valid_after
