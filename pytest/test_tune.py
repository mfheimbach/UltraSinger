"""Tuner logic (eval/tune.py) on synthetic objectives: no audio, no models, runs in seconds.

The fake evaluator scores a configuration with a known function, so the tests can check that the
search finds the optimum, that nested dimensions only exist under their parent value, that the
adoption and fold rules use the noise threshold, and that a resumed run appends.
"""

import csv
import logging
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "eval"))
os.environ["ULTRASINGER_RUNTIME_LOG"] = os.path.join(HERE, ".pytest_runtimes.jsonl")

import tune  # noqa: E402
from pipeline.core import load_preset  # noqa: E402
from pipeline.run import preset_path  # noqa: E402

SPEC = {"name": "t", "base": "bench", "data": {"train": "x", "holdout": "y", "folds": 3, "gt_language": True},
        "objective": {"sum": {"usdx_medium": 1.0}}, "min_gain": 0.001, "noise_sigma": 1.0, "step_sigma": 0.25, "fold_gate": True, "effect_probe": False,
        "sampler": "auto", "trials": 40, "stop": {}, "coordinate_points": 5,
        "cost": {"cheap_seconds": 3.0, "unknown_seconds": 1.0}, "search": {"notes.calibrate.offset": [0.0]}}

NESTED = {
    "notes.merge.slides": {False: {}, True: {"notes.merge.slide_max_semitones": [1, 2, 3]}},
    "notes.calibrate.offset": [-0.1, -0.05, 0.0],
}


class FakeEvaluator:
    """score(song, exp_ov, cheap_ov, tail_start) = f(config) + per-song offset."""

    def __init__(self, f, spec=SPEC):
        self.spec, self.f, self.base_overrides = spec, f, []
        self.calls = []

    def pipeline(self, overrides, name):
        return load_preset(preset_path("bench"), self.base_overrides + list(overrides))

    def score(self, song, exp_ov, cheap_ov, tail_start):
        cfg = dict(o.split("=", 1) for o in list(exp_ov) + list(cheap_ov))
        self.calls.append(cfg)
        self.last_chart = getattr(self, "last_chart", {})
        self.last_chart[song] = repr(round(self.f(cfg), 9))
        return {"v": self.f(cfg) + 0.01 * (hash(song) % 7)}

    def objective(self, per_song):
        return float(np.mean([r["v"] for r in per_song]))


# ----------------------------------------------------------------------------- search space

def test_nested_dims_parse_and_activate():
    dims = tune.parse_search(NESTED)
    keys = [d.key for d in dims]
    assert keys == ["notes.merge.slides", "notes.merge.slide_max_semitones", "notes.calibrate.offset"]
    child = dims[1]
    assert child.parent == ("notes.merge.slides", True)
    on = {d.key for d in tune.active(dims, {"notes.merge.slides": False})}
    assert "notes.merge.slide_max_semitones" not in on
    on = {d.key for d in tune.active(dims, {"notes.merge.slides": True})}
    assert "notes.merge.slide_max_semitones" in on
    # 1 (slides off) + 3 (slides on x 3 ranges) = 4 combos, times 3 offsets
    assert len(tune.valid_configs(dims)) == 12


def test_normalize_drops_inactive_and_fills_new():
    dims = tune.parse_search(NESTED)
    default = {"notes.merge.slides": False, "notes.merge.slide_max_semitones": 2, "notes.calibrate.offset": 0.0}
    cfg = tune.normalize(dims, {"notes.merge.slides": True, "notes.calibrate.offset": -0.1}, default)
    assert cfg == {"notes.merge.slides": True, "notes.merge.slide_max_semitones": 2, "notes.calibrate.offset": -0.1}
    cfg = tune.normalize(dims, {**cfg, "notes.merge.slides": False}, default)
    assert "notes.merge.slide_max_semitones" not in cfg
    assert tune.to_overrides(dims, cfg) == ["notes.merge.slides=false", "notes.calibrate.offset=-0.1"]


def test_range_spec_is_not_mistaken_for_nesting():
    d = tune.Dim("notes.calibrate.offset", {"low": -0.1, "high": 0.0})
    assert d.kind == "float" and d.choices is None
    d = tune.Dim("notes.merge.slides", {False: {}, True: {}})
    assert d.kind == "choice" and d.choices == [False, True]


# ----------------------------------------------------------------------------- search

def _target(cfg):
    """Best: slides on with 3 semitones, offset -0.05."""
    v = -abs(float(cfg.get("notes.calibrate.offset", 0)) + 0.05)
    if cfg.get("notes.merge.slides") == "true":
        v += 0.01 * int(cfg.get("notes.merge.slide_max_semitones", 0))
    return v


def _search(sampler, trials=40, stop=None):
    ev = FakeEvaluator(_target)
    dims = tune.parse_search(NESTED)
    return ev, tune.tune(ev, ["a", "b"], [], dims, 0, trials, 0, log=lambda *_: None,
                         sampler_name=sampler, stop=stop or {})


@pytest.mark.parametrize("sampler", ["auto", "coordinate"])
def test_search_finds_optimum(sampler):
    _, (_, best, _) = _search(sampler)
    assert sorted(best) == ["notes.calibrate.offset=-0.05", "notes.merge.slide_max_semitones=3",
                            "notes.merge.slides=true"]


def test_inactive_child_never_reaches_the_pipeline():
    ev, _ = _search("tpe", trials=25)
    for cfg in ev.calls:
        if cfg.get("notes.merge.slides") == "false":
            assert "notes.merge.slide_max_semitones" not in cfg


def test_patience_stops_early():
    ev = FakeEvaluator(lambda cfg: 0.0)      # flat: nothing ever improves
    dims = tune.parse_search({"notes.calibrate.offset": {"low": -0.1, "high": 0.0}})
    _, _, rows = tune.tune(ev, ["a"], [], dims, 0, 100, 0, log=lambda *_: None, sampler_name="tpe",
                           stop={"tol": 0.001, "patience": 5})
    assert len(rows) <= 7


def test_coordinate_ignores_moves_below_tol():
    ev = FakeEvaluator(lambda cfg: -0.0001 * abs(float(cfg["notes.calibrate.offset"]) + 0.05))
    dims = tune.parse_search({"notes.calibrate.offset": [-0.1, -0.05, 0.0]})
    _, best, _ = tune.tune(ev, ["a"], [], dims, 0, 20, 0, log=lambda *_: None, sampler_name="coordinate",
                           stop={"tol": 0.001})
    # start: the preset's -0.07 is not a grid value, so the first value; the better -0.05 gains < tol
    assert best == ["notes.calibrate.offset=-0.1"]


# ----------------------------------------------------------------------------- spec, noise, gain

def test_spec_without_holdout_or_keys_fails_loudly():
    tune.validate_spec(SPEC)
    for drop in ("min_gain", "sampler", "stop", "objective"):
        bad = {k: v for k, v in SPEC.items() if k != drop}
        with pytest.raises(SystemExit, match=drop):
            tune.validate_spec(bad)
    with pytest.raises(SystemExit, match="holdout"):
        tune.validate_spec({**SPEC, "data": {**SPEC["data"], "holdout": ""}})


def test_group_noise_uses_first_chaotic_module_downstream(caplog):
    p = load_preset(preset_path("bench"))
    obj = {"sum": {"usdx_medium": 1.0}}
    table = {"separate[htdemucs_ft]": {"sd": 0.03, "objective": obj},
             "transcribe.whisper[large-v3-turbo]": {"sd": 0.014, "objective": obj},
             "segment.game[GAME-1.0.3-small-onnx]": {"sd": 0.004, "objective": obj}}
    sd = lambda key: tune.group_noise(p, [tune.Dim(key, [0])], table, obj)[0]  # noqa: E731
    assert sd("separate.model") == 0.03
    assert sd("prepare.denoise") == 0.014           # re-rolls Whisper
    assert sd("syllables.silence_split.min_piece") == 0.004   # re-rolls GAME
    assert sd("notes.calibrate.offset") == 0.0      # deterministic downstream
    with caplog.at_level(logging.WARNING):
        tune.group_noise(p, [tune.Dim("separate.model", [0])], {"separate[htdemucs]": {"sd": 0.02}}, obj)
        tune.group_noise(p, [tune.Dim("separate.model", [0])], table, {"sum": {"wer": -1.0}})
    assert sum("FALLBACK" in r.getMessage() for r in caplog.records) == 2


def test_min_gain_is_the_larger_of_floor_and_noise():
    assert tune.min_gain(SPEC, 0.0) == 0.001
    assert tune.min_gain(SPEC, 0.02) == 0.02
    assert tune.min_gain({**SPEC, "noise_sigma": 0.5}, 0.02) == 0.01


def test_fold_gate_keeps_base_for_gains_below_threshold(monkeypatch):
    monkeypatch.setattr(tune, "step_cost", lambda name: 0.01)      # everything cheap
    monkeypatch.setattr(tune, "load_noise", lambda song_set: {})
    tiny = lambda cfg: 0.0002 * (float(cfg.get("notes.calibrate.offset", -0.07)) + 0.07)  # noqa: E731
    names = [f"Artist{i} - Song" for i in range(6)]
    for gate, expect_zero in ((True, True), (False, False)):
        ev = FakeEvaluator(tiny, {**SPEC, "fold_gate": gate})
        res = tune.run_group(ev, names, {"notes.calibrate.offset": [-0.07, 0.0]}, 10, 3, 0, [], "g",
                             "grid", {}, ev.spec)
        assert (res["cv_gain"] == 0.0) is expect_zero


# ----------------------------------------------------------------------------- resume

def test_resume_appends_report_and_trials(tmp_path):
    import yaml
    groups = [{"name": "one"}, {"name": "two"}, {"name": "three"}]
    (tmp_path / "report.md").write_text("# Tuning report: t\n\n## 1. one\n\nok\n\n## 2. two\n\nold\n\n"
                                        "## Result\n\nx\n", encoding="utf-8")
    with open(tmp_path / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["group", "trial", "objective"])
        w.writeheader()
        w.writerows([{"group": "one", "trial": 0, "objective": 1}, {"group": "two", "trial": 0, "objective": 2}])
    yaml.safe_dump({"overrides": ["a=1"]}, open(tmp_path / "preset.yaml", "w"))
    report, rows, overrides = tune.read_previous(str(tmp_path), groups, 2)
    text = "\n".join(report)
    assert "## 1. one" in text and "## 2. two" not in text and "## Result" not in text
    assert [r["group"] for r in rows] == ["one"]
    assert overrides == ["a=1"]


def test_edge_hits_flag_values_at_the_range_bounds():
    dims = tune.parse_search({"notes.calibrate.end_extension": {"low": 0.0, "high": 0.2},
                              "notes.calibrate.offset": {"low": -0.15, "high": 0.0}})
    hits = tune.edge_hits(dims, ["notes.calibrate.end_extension=0.1919", "notes.calibrate.offset=-0.07"])
    assert len(hits) == 1 and hits[0].startswith("notes.calibrate.end_extension")


def test_objective_sum_times_factors():
    ev = tune.Evaluator.__new__(tune.Evaluator)
    ev.terms, ev.factors = {"a": 1.0, "b": 0.5}, {"c": 0.0}
    assert ev.song_objective({"a": 0.6, "b": 0.4, "c": 0.5}) == pytest.approx(0.4)
    ev.factors = {"c": 0.5}
    assert ev.song_objective({"a": 0.6, "b": 0.4, "c": 0.0}) == pytest.approx(0.4)   # floored, not zeroed
    assert ev.song_objective({"a": 0.6, "b": 0.4, "c": 1.0}) == pytest.approx(0.8)
    assert ev.song_objective({"a": 0.6, "b": 0.4}) == pytest.approx(0.8)            # missing factor = 1
    assert ev.song_objective({"a": 0.6, "b": float("nan"), "c": 1.0}) == pytest.approx(0.6)


def test_effect_probe_drops_dimensions_that_change_nothing(monkeypatch):
    monkeypatch.setattr(tune, "step_cost", lambda name: 0.01)
    f = lambda cfg: float(cfg.get("notes.calibrate.offset", -0.07))   # noqa: E731  (end_extension is dead)
    ev = FakeEvaluator(f)
    dims = tune.parse_search({"notes.calibrate.offset": [-0.1, 0.0], "notes.calibrate.end_extension": [0.0, 0.1],
                              "notes.merge.slides": {False: {}, True: {"notes.merge.slide_max_semitones": [1, 3]}}})
    dead = tune.probe_effect(ev, ["a"], dims, [], dims, 0)
    assert "notes.calibrate.offset" not in dead
    assert {"notes.calibrate.end_extension", "notes.merge.slides", "notes.merge.slide_max_semitones"} <= set(dead)
