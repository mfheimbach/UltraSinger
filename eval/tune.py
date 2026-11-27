"""Tune the whole inference machine on a dataset: module choices, step order and parameters.

    python eval/tune.py tune/notes.yaml [--trials 100] [--work eval_work_ng]

A tuning spec (YAML) names a base preset, the data, the objective and the search space:

    name: notes_v1
    base: bench                         # preset to start from
    data:
      train: eval_singstar/bench        # song folders (charts + audio)
      holdout: eval_singstar/selection  # mandatory, disjoint: scored once, at the very end
      folds: 4                          # cross-validation folds over train, grouped by artist
      gt_language: true                 # language from the chart (short windows)
    objective:                          # per song: weighted sum x factors; mean over songs
      sum: {usdx_medium_f1: 1.0, usdx_hard: 0.5, start_score: 0.5, end_score: 0.25, line_f1: 0.25}
      multiply: {lyrics_acc: 0.5}       # factor floor: song x (0.5 + 0.5 x lyrics_acc)
    min_gain: 0.001                     # adopt only gains above max(min_gain, noise_sigma x noise SD)
    noise_sigma: 1.0                    # noise SD per engine: tune/noise.yaml
    fold_gate: true
    sampler: auto
    stop: {patience: 30}
    coordinate_points: 7
    cost: {cheap_seconds: 3.0, unknown_seconds: 1.0}
    # every key is required (validate_spec): no silent defaults
    search:
      notes.calibrate.offset: {low: -0.15, high: 0.0}       # continuous
      notes.snap_edges.max_shift: {low: 0.0, high: 0.8}
      notes.from_pitch.method: [chroma_mode, median]         # discrete choices
      segment.game.use: [segment.game, segment.contour]      # module alternatives for a step
      notes.merge.slides:                                    # nested: only searched while slides=true
        false: {}
        true:
          notes.merge.slide_max_semitones: {low: 1, high: 3}
      words.strip_punctuation.after: [transcribe.whisper, syllables.hyphenate]   # step order
      separate.model: [htdemucs_ft, htdemucs]                # expensive: evaluated on demand
    trials: 100

How it runs: every search dimension maps to an override (step.param=value, step.use=stage,
step.after=other). Steps before the first step any cheap dimension touches form the prefix: it is
computed once per song and per combination of expensive choices (disk cache across runs, memory
within a run). Each trial then runs only the tail in memory (no cache writes) and scores the chart
against the ground truth. Optuna's TPE searches all dimensions jointly.

Honesty: K-fold cross-validation over artists (tune on K-1 folds, score the held fold) gives the
expected gain on unseen songs; the final configuration is tuned on all train songs and scored once
on the holdout. Gains come with bootstrap CIs over songs; the report lists which dimensions mattered.
Staged runs (`stages:`) tune groups early to late, each from the last best; `--from-group N`
resumes a run in place (report and trials appended).
Output: tune/out/<name>/ with report.md, trials.csv, and preset.yaml (base + chosen overrides).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import metrics  # noqa: E402
import usdx  # noqa: E402
from run_eval import discover, slug  # noqa: E402
from run_song import LANGUAGE_STEPS, language_code  # noqa: E402

from pipeline import log as plog  # noqa: E402
from pipeline import stages  # noqa: E402,F401  (registers the stages)
from pipeline.core import Context, Pipeline, load_preset  # noqa: E402
from pipeline.run import media_from_filename, preset_path  # noqa: E402

LOWER_IS_BETTER = {"wer", "cer", "pitch_mae"}
def print(*args, **kwargs):  # noqa: A001  (every tuner line gets a timestamp, like the pipeline log)
    import builtins
    builtins.print(time.strftime("%H:%M:%S"), *args, **{"flush": True, **kwargs})


BOOT_SAMPLES = 2000    # bootstrap resamples for CIs
PROGRESS_MIN_SECONDS = 5.0   # a song scored slower than this was computed (not cached): progress line
GROUP_LABEL = ""             # "group i/n name", shown on progress lines


def bar(i: int, n: int, width: int = 20) -> str:
    """Text progress bar for log lines (a log file has no terminal to redraw)."""
    k = round(width * i / max(n, 1))
    return f"[{'#' * k}{'-' * (width - k)}] {i}/{n}"

# Every tuning setting lives in the spec; a missing one stops the run (no silent defaults).
REQUIRED = ["name", "base", "data.train", "data.holdout", "data.folds", "data.gt_language", "objective.sum",
            "min_gain", "noise_sigma", "step_sigma", "fold_gate", "workers", "effect_probe", "sampler", "trials", "stop", "coordinate_points",
            "cost.cheap_seconds", "cost.unknown_seconds"]


def validate_spec(spec: dict) -> None:
    """Fail loudly on missing settings. A holdout is mandatory: the final config is scored on songs
    the search never saw."""
    missing = []
    for key in REQUIRED:
        node = spec
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(key)
                break
            node = node[part]
    if not (spec.get("stages") or spec.get("search")):
        missing.append("stages (or search)")
    if missing:
        raise SystemExit(f"tuning spec is missing: {', '.join(missing)} (see tune/staged_singstar.yaml)")
    if not spec["data"]["holdout"]:
        raise SystemExit("tuning spec: data.holdout is empty; tuning without a holdout is not allowed")


# ----------------------------------------------------------------------------- search space

RANGE_KEYS = {"low", "high", "log", "float"}


class Dim:
    """One search dimension -> override string for a value. `parent` = (key, value): the dimension
    only exists while that parent choice has that value (nested in the spec, see parse_search)."""

    def __init__(self, key: str, spec, parent: tuple[str, object] | None = None):
        self.key, self.parent = key, parent
        self.step, self.attr = key.rsplit(".", 1)
        if isinstance(spec, dict) and set(spec) <= RANGE_KEYS and "low" in spec:
            self.kind = "int" if isinstance(spec.get("low"), int) and isinstance(spec.get("high"), int) \
                and not spec.get("float") else "float"
            self.low, self.high, self.log = spec["low"], spec["high"], bool(spec.get("log"))
            self.choices = None
        else:
            self.kind = "choice"
            self.choices = list(spec) if not isinstance(spec, dict) else list(spec.keys())

    def suggest(self, trial):
        if self.kind == "choice":
            return trial.suggest_categorical(self.key, self.choices)
        if self.kind == "int":
            return trial.suggest_int(self.key, self.low, self.high, log=self.log)
        return trial.suggest_float(self.key, self.low, self.high, log=self.log)

    def override(self, value) -> str:
        return f"{self.step}.{self.attr}={json.dumps(value) if not isinstance(value, str) else value}"


def parse_search(search: dict, parent: tuple[str, object] | None = None) -> list[Dim]:
    """Search spec -> dimensions, parents before children. A choice given as a mapping nests the
    dimensions that only matter for one of its values (indent them under that value):

        notes.merge.slides:
          false: {}
          true:
            notes.merge.slide_max_semitones: {low: 1, high: 3}
        segment.game.use:
          segment.game: {}
          segment.contour:
            notes.game_pitch.enabled: [false]    # contour has no GAME pitches
    """
    dims = []
    for key, spec in search.items():
        d = Dim(key, spec, parent)
        dims.append(d)
        if d.kind == "choice" and isinstance(spec, dict):
            for value, sub in spec.items():
                dims += parse_search(sub or {}, (key, value))
    return dims


def active(dims: list[Dim], cfg: dict) -> list[Dim]:
    """The dimensions that exist under this configuration (parents decide)."""
    on = set()
    for d in dims:
        if d.parent is None or (d.parent[0] in on and cfg.get(d.parent[0]) == d.parent[1]):
            on.add(d.key)
    return [d for d in dims if d.key in on]


def normalize(dims: list[Dim], cfg: dict, default: dict) -> dict:
    """Keep the values of active dimensions, fill newly active ones from `default`, drop the rest."""
    out = {}
    for d in dims:
        if d.parent is None or (d.parent[0] in out and out[d.parent[0]] == d.parent[1]):
            out[d.key] = cfg[d.key] if d.key in cfg else default[d.key]
    return out


def valid_configs(dims: list[Dim]) -> list[dict] | None:
    """Every configuration of an all-discrete (possibly nested) space; None if any range."""
    if any(d.kind != "choice" for d in dims):
        return None
    configs = [{}]
    for d in dims:
        nxt = []
        for c in configs:
            if d.parent is None or (d.parent[0] in c and c[d.parent[0]] == d.parent[1]):
                nxt += [{**c, d.key: v} for v in d.choices]
            else:
                nxt.append(c)
        configs = nxt
    return configs


def base_config(ev: "Evaluator", song: str, dims: list[Dim]) -> dict:
    """Current values of the dimensions in the base preset (missing / out-of-range ones left out)."""
    p0 = ev.pipeline([], song)
    ids = [st.id for st in p0.stages]
    vals = {f"{st.id}.{k}": v for st in p0.stages for k, v in vars(st.params).items()}
    base = {}
    for d in dims:
        if d.attr == "use":
            v = p0.stages[ids.index(d.step)].name if d.step in ids else None
        elif d.attr == "enabled":
            v = d.step in ids
        elif d.attr in ("after", "before"):
            i = ids.index(d.step)
            v = ids[i - 1] if d.attr == "after" and i > 0 else (ids[i + 1] if i + 1 < len(ids) else None)
        else:
            v = vals.get(d.key)
        if v is None:
            continue
        if d.kind == "choice" and v not in d.choices:
            continue
        if d.kind != "choice" and not (d.low <= v <= d.high):
            continue
        base[d.key] = v
    return base


DEVICE = "cpu"        # set by main (--device)
POOL = None           # SongPool when spec["workers"] > 1
STORAGE = None        # optuna storage URL (SQLite in the run's out dir): searches survive a crash


class SongPool:
    """Worker processes that run song tails in parallel (cheap groups: notes, line breaks, writer).
    Each song always goes to the same worker, so its prefix is loaded from the disk cache once per
    worker and then reused. Workers run on the CPU (the GPU stays with the main process); they only
    read prefixes the main process has already written."""

    def __init__(self, n: int, spec: dict, work: str, songs: list[tuple[str, str]]):
        import concurrent.futures as cf
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.order = {name: i for i, (name, _) in enumerate(songs)}
        self.execs = [cf.ProcessPoolExecutor(1, mp_context=ctx, initializer=_worker_init,
                                             initargs=(spec, work, songs)) for _ in range(n)]

    def score(self, names, base_overrides, cheap_ov, tail_start):
        futs = [self.execs[self.order[n] % len(self.execs)].submit(_worker_score, n, list(base_overrides),
                                                                   list(cheap_ov), tail_start) for n in names]
        return [f.result() for f in futs]

    def close(self):
        for e in self.execs:
            e.shutdown(cancel_futures=True)


_W = None


def _worker_init(spec, work, songs):
    global _W, DEVICE
    DEVICE = "cpu"
    plog.setup(verbose=False)
    _W = Evaluator(spec, work, songs)


def _worker_score(name, base_overrides, cheap_ov, tail_start):
    _W.base_overrides = base_overrides
    res = _W.score(name, (), cheap_ov, tail_start)
    return res, _W.last_chart[name]
_RECORDS: list | None = None


def step_cost(stage_name: str) -> float | None:
    """Seconds per 30 s of audio from this device's runtime records (None: unknown)."""
    global _RECORDS
    if _RECORDS is None:
        from pipeline import runtimes
        _RECORDS = runtimes.load()
    recs = [r for r in _RECORDS if r.get("stage") == stage_name and r.get("audio_seconds")
            and r.get("device", "cpu") == DEVICE]
    if not recs:
        return None
    return float(np.median([r["seconds"] / r["audio_seconds"] * 30.0 for r in recs[-50:]]))


def print_eta(ev: "Evaluator", names: list[str], groups: list[dict], spec: dict, folds: int, workers: int) -> None:
    """Rough run time from this machine's runtime records: expensive configs x rerun cost, plus cheap
    trials x (folds + 1) searches x tail cost / workers. Printed once, before the first group."""
    import soundfile as sf
    audio = 0.0
    for n in names:
        try:
            audio += sf.info(ev.gt[n].audio_path()).duration
        except Exception:  # noqa: BLE001  (formats soundfile can't read: assume 3.5 min)
            audio += 210.0
    p = ev.pipeline([], names[0])
    ids = [st.id for st in p.stages]
    cost = [step_cost(st.name) or 0.0 for st in p.stages]
    per30 = audio / 30.0
    total = 0.0
    lines = []
    for g in groups:
        before = total
        dims = parse_search(g["search"])
        exp, cheap, tail = split_dims(p, dims, float(spec["cost"]["cheap_seconds"]),
                                      float(spec["cost"]["unknown_seconds"]))
        trials = int(g.get("trials") or spec["trials"])
        if exp:
            first = min(ids.index(d.step) for d in exp if d.step in ids)
            n_cfg = len(valid_configs(exp) or range(trials))
            total += n_cfg * sum(cost[first:]) * per30
        if cheap:
            tail_cost = sum(cost[tail:]) + 0.3          # + metrics per 30 s audio (rough)
            total += trials * (folds + 1) * 0.8 * tail_cost * per30 / max(1, workers)
        lines.append(f"    {(total - before) / 3600:5.1f} h  {g['name']}")
    print(f"ETA (rough, from runtime records): {total / 3600:.1f} h for {len(names)} songs, "
          f"{audio / 60:.0f} min of audio; per group:\n" + "\n".join(lines), flush=True)


def warm_up(ev: "Evaluator", names: list[str]) -> None:
    """Without runtime records for this device the cheap/expensive split would guess; run the base
    pipeline once on every song (fills the cache and records the runtimes)."""
    global _RECORDS
    p = ev.pipeline([], names[0])
    missing = [st.name for st in p.stages if step_cost(st.name) is None]
    if not missing:
        return
    print(f"warm-up: no runtime records on {DEVICE} for {missing}; running the base pipeline on {len(names)} songs")
    for n in names:
        ev.score(n, (), [], _write_index(ev, n))
    _RECORDS = None


def split_dims(base_pipeline: Pipeline, dims: list[Dim], cheap_seconds: float,
               unknown_seconds: float) -> tuple[list[Dim], list[Dim], int]:
    """Cheap dims: rerunning from their step to the end costs at most cheap_seconds per 30 s of
    audio (runtime records; unknown stages count unknown_seconds). Everything else is expensive:
    discrete values only, each computed once (disk cache). Returns the tail start: the earliest step
    a cheap dim can touch, never after the first writer (charts are written per trial)."""
    ids = [st.id for st in base_pipeline.stages]
    names = [st.name for st in base_pipeline.stages]
    cost = [step_cost(n) for n in names]
    cost = [unknown_seconds if c is None else c for c in cost]

    def tail_cost(i: int, alt_stage: str | None = None) -> float:
        extra = 0.0
        if alt_stage:
            c = step_cost(alt_stage)
            extra = (unknown_seconds if c is None else c) - cost[i]
        return sum(cost[i:]) + max(extra, 0.0)

    expensive, cheap = [], []
    for d in dims:
        if d.step not in ids:
            raise ValueError(f"search dimension {d.key}: no step '{d.step}' in the base preset")
        i = ids.index(d.step)
        if d.attr in ("after", "before"):
            i = min([i] + [ids.index(c) + (1 if d.attr == "after" else 0) for c in d.choices if c in ids])
        c = max(tail_cost(i, alt) for alt in d.choices) if d.attr == "use" else tail_cost(i)
        if c <= cheap_seconds:
            cheap.append((d, i))
        else:
            if d.kind != "choice":
                raise ValueError(f"{d.key}: rerunning from {d.step} costs ~{c:.0f}s per window; give a list of "
                                 f"values instead of a range")
            expensive.append(d)
    first_writer = next((i for i, n in enumerate(names) if n.startswith("write.")), len(ids))
    tail_start = min([i for _, i in cheap] + [first_writer])
    return expensive, [d for d, _ in cheap], tail_start


# ----------------------------------------------------------------------------- evaluation

class Evaluator:
    """Prefix per (song, expensive choices) computed once; tails run in memory per trial."""

    def __init__(self, spec: dict, work: str, songs: list[tuple[str, str]]):
        self.spec, self.work = spec, work
        self.base = spec["base"]
        self.base_overrides = list(spec.get("overrides") or [])   # grows during staged tuning
        self.gt = {name: usdx.read(txt) for name, txt in songs}
        self.songs = [name for name, _ in songs]
        self.prefix_cache: dict = {}
        self.tmp = tempfile.mkdtemp(prefix="tune_")
        self.terms = dict(spec["objective"]["sum"])
        # multiply: {metric: floor}; factor = floor + (1 - floor) * metric, so a song whose factor metric
        # collapses (failed transcription) still counts, at floor weight
        self.factors = {k: float(v) for k, v in (spec["objective"].get("multiply") or {}).items()}
        for k, f in self.factors.items():
            if not 0.0 <= f < 1.0:
                raise SystemExit(f"objective.multiply.{k}: floor {f} must be in [0, 1)")
        self.metric_keys = list(self.terms) + [k for k in self.factors if k not in self.terms]
        self.last_chart: dict[str, str] = {}   # song -> chart text of the last score() (no-effect probe)

    def lang_overrides(self, name: str, step_ids: set[str]) -> list[str]:
        if not self.spec.get("data", {}).get("gt_language"):
            return []
        code = language_code(self.gt[name].language)
        return [f"{s}.language={code}" for s in LANGUAGE_STEPS if s in step_ids]

    def pipeline(self, overrides: list[str], name: str) -> Pipeline:
        ov = self.base_overrides + list(overrides)
        p = load_preset(preset_path(self.base), ov)
        return load_preset(preset_path(self.base), self.lang_overrides(name, {st.id for st in p.stages}) + ov)

    def _ctx(self, name: str, no_cache: bool) -> Context:
        gt = self.gt[name]
        out = os.path.join(self.tmp, slug(name))
        os.makedirs(out, exist_ok=True)
        return Context(cache_dir=os.path.abspath(os.path.join(self.work, "cache", slug(name))), output_dir=out,
                       device=DEVICE, basename=f"{gt.artist} - {gt.title}", verbose=False,
                       extra={"no_cache": no_cache})

    def prefix(self, name: str, expensive_ov: tuple, tail_start: int) -> dict:
        key = (name, tuple(self.base_overrides), expensive_ov, tail_start)
        if key not in self.prefix_cache:
            p = self.pipeline(list(expensive_ov), name)
            gt = self.gt[name]
            store = {"audio.mix": os.path.abspath(gt.audio_path()),
                     "media": media_from_filename(gt.audio_path(), gt.artist, gt.title)}
            head = _sub(p, 0, tail_start)
            self.prefix_cache[key] = head.run(self._ctx(name, no_cache=False), store)
        return self.prefix_cache[key]

    def score(self, name: str, expensive_ov: tuple, cheap_ov: list[str], tail_start: int) -> dict:
        p = self.pipeline(list(expensive_ov) + cheap_ov, name)
        store = copy.deepcopy(self.prefix(name, expensive_ov, tail_start))
        tail = _sub(p, tail_start, len(p.stages))
        store = tail.run(self._ctx(name, no_cache=True), store)
        with open(store["file.txt"], encoding="utf-8") as f:
            self.last_chart[name] = f.read()
        pred = usdx.read(store["file.txt"])
        m = metrics.evaluate(self.gt[name], pred)
        return {k: m.get(k) for k in self.metric_keys}

    def song_objective(self, d: dict) -> float:
        """One song: weighted sum of the `sum` terms times the `multiply` factors. A metric a song
        doesn't have (WER for non-Latin lyrics, line F1 without breaks) is left out / counts as 1."""
        def ok(k):
            return d.get(k) is not None and d.get(k) == d.get(k)
        total = sum(w * d[k] for k, w in self.terms.items() if ok(k))
        for k, floor in self.factors.items():
            if ok(k):
                total *= floor + (1.0 - floor) * d[k]
        return float(total)

    def score_many(self, names: list[str], exp_ov: tuple, cheap_ov: list, tail_start: int) -> list[dict]:
        """score() for several songs: in parallel worker processes when a pool is running and the call
        needs no expensive overrides (the base prefixes are on disk then), else here, in order."""
        if POOL is not None and not exp_ov:
            out = POOL.score(names, self.base_overrides, cheap_ov, tail_start)
            for n, (_, chart) in zip(names, out):
                self.last_chart[n] = chart
            return [res for res, _ in out]
        out, t0 = [], time.time()
        for i, n in enumerate(names, 1):
            t = time.time()
            out.append(self.score(n, exp_ov, cheap_ov, tail_start))
            if time.time() - t > PROGRESS_MIN_SECONDS:   # computed, not a cache hit
                left = (time.time() - t0) / i * (len(names) - i)
                print(f"    {bar(i, len(names))} songs  ~{left / 60:.0f} min left  {GROUP_LABEL}  "
                      f"{n[:40]} {time.time() - t:.0f}s", flush=True)
        return out

    def report_floored(self, names: list[str], tail_start_of) -> list[str]:
        """Songs whose factor metric is near zero on the base config (they count at floor weight):
        logged as a warning and returned for the report."""
        import logging
        out = []
        for n in names:
            d = self.score(n, (), [], tail_start_of(n))
            low = [f"{k} {d[k]:.2f}" for k, floor in self.factors.items()
                   if d.get(k) is not None and d.get(k) == d.get(k) and d[k] <= 0.1]
            if low:
                out.append(f"{n} ({', '.join(low)})")
        if out:
            logging.getLogger("ultrasinger.tune").warning(
                f"FLOORED: {len(out)} song(s) count at floor weight (factor metric <= 0.1, e.g. failed "
                f"transcription): {out}")
        return out

    def objective(self, per_song: list[dict]) -> float:
        """Mean over songs of the per-song objective (the same aggregation everywhere)."""
        return float(np.mean([self.song_objective(d) for d in per_song])) if per_song else 0.0


def _sub(p: Pipeline, a: int, b: int) -> Pipeline:
    q = Pipeline.__new__(Pipeline)
    q.name, q.params, q.param_sources = p.name, p.params, p.param_sources
    q.specs, q.stages = p.specs[a:b], p.stages[a:b]
    return q


# ----------------------------------------------------------------------------- tuning

def coordinate_sampler_trials(dims: list[Dim], points: int):
    """Value grid per dimension for coordinate descent (continuous: `points` evenly spaced)."""
    def values(d):
        if d.kind == "choice":
            return list(d.choices)
        if d.kind == "int":
            return sorted({int(round(v)) for v in np.linspace(d.low, d.high, points)})
        return [float(v) for v in np.linspace(d.low, d.high, points)]
    return {d.key: values(d) for d in dims}


def _evaluate(ev, songs, dims, expensive, cheap, tail_start, cfg) -> float:
    """Objective of one configuration (only its active dimensions become overrides)."""
    keys = {d.key for d in active(dims, cfg)}
    exp_ov = tuple(d.override(cfg[d.key]) for d in expensive if d.key in keys)
    cheap_ov = [d.override(cfg[d.key]) for d in cheap if d.key in keys]
    return ev.objective(ev.score_many(songs, exp_ov, cheap_ov, tail_start))


def to_overrides(dims: list[Dim], cfg: dict) -> list[str]:
    return [d.override(cfg[d.key]) for d in active(dims, cfg)]


def tune(ev: Evaluator, songs: list[str], expensive: list[Dim], cheap: list[Dim], tail_start: int,
         trials: int, seed: int, log=print, sampler_name: str = "auto", stop: dict | None = None,
         study_key: str | None = None):
    """Search the dimensions on these songs. sampler_name: auto (grid when everything is discrete and
    fits the budget, else TPE) | grid | tpe | random | cmaes | coordinate. The search ends at whichever
    comes first: `trials` evaluations; no improvement > stop["tol"] for stop["patience"] trials;
    stop["max_seconds"] wall time. Nested dimensions are only sampled while their parent value holds.
    With STORAGE set and a study_key, trials are stored as they finish (SQLite) and a rerun with the
    same key continues the search instead of starting over (resume after a crash).
    Returns (study or None, overrides of the best configuration, rows)."""
    import optuna
    import warnings
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # multivariate/group TPE are flagged experimental; we use them on purpose (conditional spaces)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    dims = order_dims(expensive + cheap)
    if sampler_name == "coordinate":
        return _coordinate(ev, songs, dims, expensive, cheap, tail_start, trials, stop)
    rows = []
    grid = valid_configs(dims)
    use_grid = sampler_name in ("auto", "grid") and grid is not None and len(grid) <= trials

    def objective(trial):
        if use_grid:
            cfg = grid[trial.suggest_categorical("_config", list(range(len(grid))))]
        else:
            cfg = {}
            for d in dims:
                if d.parent is None or (d.parent[0] in cfg and cfg[d.parent[0]] == d.parent[1]):
                    cfg[d.key] = d.suggest(trial)
        trial.set_user_attr("cfg", cfg)
        val = _evaluate(ev, songs, dims, expensive, cheap, tail_start, cfg)
        rows.append({"trial": trial.number, "objective": val, **cfg})
        return val

    if use_grid:
        sampler = optuna.samplers.GridSampler({"_config": list(range(len(grid)))}, seed=seed)
        trials = len(grid)
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False,
                                               independent_sampler=optuna.samplers.TPESampler(seed=seed))
    else:
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True,
                                             warn_independent_sampling=False)
    if STORAGE and study_key:
        study = optuna.create_study(direction="maximize", sampler=sampler, storage=STORAGE,
                                    study_name=study_key, load_if_exists=True)
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    TS = optuna.trial.TrialState
    for t in study.get_trials(deepcopy=False, states=(TS.RUNNING, TS.WAITING)):
        if t.state == TS.RUNNING:   # interrupted by a crash: doesn't count, its config may be retried
            study._storage.set_trial_state_values(t._trial_id, TS.FAIL)
    done = [t for t in study.get_trials(deepcopy=False, states=(TS.COMPLETE,))]
    for t in done:
        rows.append({"trial": t.number, "objective": t.value, **t.user_attrs["cfg"]})
    if study.user_attrs.get("finished"):
        return study, to_overrides(dims, study.best_trial.user_attrs["cfg"]), rows
    # trial 0 = the base preset (so "tuned" can never be worse than base on the tuning songs)
    base = base_config(ev, songs[0], dims)
    if not done and not use_grid and {d.key for d in active(dims, base)} <= set(base):
        study.enqueue_trial({k: v for k, v in base.items() if k in {d.key for d in active(dims, base)}})
    t0 = time.time()
    callbacks = [lambda s, t: log(f"  {bar(t.number + 1, trials)} trials  {t.value:.4f}  best {s.best_value:.4f}"
                                  f"  ({time.time() - t0:.0f}s)  {GROUP_LABEL}")]
    if stop:
        hist = []
        best = -np.inf
        for t in sorted(done, key=lambda t: t.number):    # resumed: patience counts the stored trials
            best = max(best, t.value)
            hist.append(best)

        def early_stop(st, t):
            hist.append(st.best_value)
            k = stop.get("patience")
            if k and len(hist) > k and hist[-1] - hist[-1 - k] <= float(stop["tol"]):
                st.stop()
            if stop.get("max_seconds") and time.time() - t0 > float(stop["max_seconds"]):
                st.stop()
        callbacks.append(early_stop)
    k = stop.get("patience") if stop else None
    stalled = bool(k) and len(hist) > k and hist[-1] - hist[-1 - k] <= float(stop["tol"])
    if len(done) < trials and not stalled:
        study.optimize(objective, n_trials=trials - len(done), callbacks=callbacks)
    if STORAGE and study_key:
        study.set_user_attr("finished", True)
    return study, to_overrides(dims, study.best_trial.user_attrs["cfg"]), rows


def order_dims(dims: list[Dim]) -> list[Dim]:
    """Parents before their children (split_dims may reorder)."""
    out, keys = [], set()
    pending = list(dims)
    while pending:
        nxt = [d for d in pending if d.parent is None or d.parent[0] in keys]
        if not nxt:
            raise ValueError(f"nested dimensions without their parent: {[d.key for d in pending]}")
        for d in nxt:
            out.append(d)
            keys.add(d.key)
            pending.remove(d)
    return out


def _coordinate(ev, songs, dims, expensive, cheap, tail_start, budget, stop=None):
    """Coordinate descent: one dimension at a time over its values (continuous: 7 points), keeping a
    move only if it improves by more than stop["tol"]; repeats passes until nothing improves, the budget
    or stop["max_seconds"] is used up. Nested dimensions take part while their parent value holds.
    Returns (None, best overrides, rows)."""
    stop = stop or {}
    tol = float(stop.get("tol", 0.0))
    t0 = time.time()
    grid = coordinate_sampler_trials(dims, int(ev.spec["coordinate_points"]))
    base = base_config(ev, songs[0], dims)
    default = {d.key: base.get(d.key, grid[d.key][0]) for d in dims}
    cur = normalize(dims, {}, default)
    seen, rows = {}, []

    def evaluate(cfg):
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen[key] = _evaluate(ev, songs, dims, expensive, cheap, tail_start, cfg)
            rows.append({"trial": len(rows), "objective": seen[key], **cfg})
        return seen[key]

    best = evaluate(cur)
    improved = True
    while improved and len(rows) < budget:
        improved = False
        for d in active(dims, cur):
            for v in grid[d.key]:
                if len(rows) >= budget or (stop.get("max_seconds") and time.time() - t0 > stop["max_seconds"]):
                    break
                cand = normalize(dims, {**cur, d.key: v}, default)
                val = evaluate(cand)
                if val > best + max(tol, 1e-12):
                    best, cur, improved = val, cand, True
    return None, to_overrides(dims, cur), rows


NOISE_FILE = os.path.join(ROOT, "tune", "noise.yaml")


def engine_key(stage) -> str:
    """Stage name plus its model, e.g. separate[htdemucs_ft]: noise is a property of the engine."""
    prm = vars(getattr(stage, "params", None) or object())
    model = next((prm[k] for k in ("separator_model", "model", "model_dir") if prm.get(k)), None)
    return f"{stage.name}[{os.path.basename(str(model).rstrip('/'))}]" if model else stage.name


def load_noise(song_set: str) -> dict:
    """Saved noise levels {engine key: {sd, n, how}} for a song set (tune/noise.yaml)."""
    import yaml
    if not os.path.exists(NOISE_FILE):
        return {}
    with open(NOISE_FILE, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get(song_set) or {}


def group_noise(pipeline, dims: list[Dim], table: dict, objective: dict,
                n_songs: int | None = None) -> tuple[float, str]:
    """Noise SD for a group: a change re-rolls the first chaotic module at or after the group's earliest
    step, so use that module's saved noise. Exact engine first; the same module with another engine, or
    a value measured for another objective, is a fallback (logged as one); no chaotic module downstream
    -> 0. With n_songs, the SD (of a mean over the entry's `songs`) is scaled to that many songs.
    Returns (sd, source)."""
    import logging
    warn = logging.getLogger("ultrasinger.tune").warning
    ids = [st.id for st in pipeline.stages]
    first = min((ids.index(d.step) for d in dims if d.step in ids), default=0)
    for st in pipeline.stages[first:]:
        key = engine_key(st)
        same = [k for k in table if k.split("[")[0] == st.name]
        if key in table or same:
            use = key if key in table else same[0]
            src = use
            if use != key:
                warn(f"FALLBACK: no noise measured for {key}; borrowing {use} (measure it)")
                src = f"{use}, borrowed for {key}"
            if table[use].get("objective") not in (None, objective):
                warn(f"FALLBACK: noise of {use} was measured for objective {table[use]['objective']}, not "
                     f"{objective} (measure it)")
                src += ", other objective"
            sd = float(table[use]["sd"])
            if n_songs is not None:
                if "songs" not in table[use]:
                    raise ValueError(f"tune/noise.yaml: {use} has no 'songs' (count the SD is for)")
                ref = int(table[use]["songs"])
                if ref != n_songs:
                    sd *= (ref / n_songs) ** 0.5
                    src += f", scaled from {ref} to {n_songs} songs"
            return sd, src
    return 0.0, "deterministic downstream"


def min_gain(spec: dict, noise_sd: float) -> float:
    """Smallest gain that counts: the larger of the absolute floor and noise_sigma x the step's noise SD."""
    return max(float(spec["min_gain"]), float(spec["noise_sigma"]) * noise_sd)


def per_song_objective(ev, songs, overrides, expensive, cheap, tail_start) -> np.ndarray:
    exp_keys = {d.step + "." + d.attr for d in expensive}
    exp_ov = tuple(o for o in overrides if o.split("=", 1)[0] in exp_keys)
    cheap_ov = [o for o in overrides if o.split("=", 1)[0] not in exp_keys]
    return np.array([ev.objective([d]) for d in ev.score_many(songs, exp_ov, cheap_ov, tail_start)])


def artist_groups(songs: list[str]) -> dict[str, str]:
    """Song (or window 'Song @12s') -> artist group from the SingStar split, else the artist."""
    path = os.path.join(HERE, "splits", "singstar_split.csv")
    groups = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            groups = {r["song"]: r["group"] for r in csv.DictReader(f)}
    return {s: groups.get(s.split(" @")[0], s.split(" - ")[0]) for s in songs}


def boot_ci(x: np.ndarray, n: int = BOOT_SAMPLES) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def probe_effect(ev: Evaluator, songs: list[str], dims: list[Dim], expensive: list[Dim], cheap: list[Dim],
                 tail_start: int) -> list[str]:
    """Dimensions that change no chart: each one is set to a few values (choices, or low/mid/high),
    everything else at the base (parents set so it is active). Identical charts on every song =
    no effect in this configuration (e.g. masked by a later step). Returns their keys."""
    base = base_config(ev, songs[0], dims)
    grid = coordinate_sampler_trials(dims, 3)
    default = {d.key: base.get(d.key, grid[d.key][0]) for d in dims}
    by_key = {d.key: d for d in dims}
    dead = []
    for d in dims:
        need, cur = {}, d
        while cur.parent is not None:          # activate the chain of parents
            need[cur.parent[0]] = cur.parent[1]
            cur = by_key[cur.parent[0]]
        charts = set()
        for v in grid[d.key]:
            cfg = normalize(dims, {**default, **need, d.key: v}, default)
            _evaluate(ev, songs, dims, expensive, cheap, tail_start, cfg)
            charts.add(tuple(ev.last_chart[s] for s in songs))
        if len(charts) == 1:
            dead.append(d.key)
    # children of a dead parent go too
    return [d.key for d in dims if d.key in dead or (d.parent and d.parent[0] in dead)]


def edge_hits(dims: list[Dim], overrides: list[str], share: float = 0.05) -> list[str]:
    """Range dimensions whose chosen value lies within `share` of the range from a bound."""
    chosen = dict(o.split("=", 1) for o in overrides)
    out = []
    for d in dims:
        if d.kind == "choice" or d.key not in chosen:
            continue
        v, span = float(chosen[d.key]), d.high - d.low
        if v <= d.low + share * span or v >= d.high - share * span:
            out.append(f"{d.key}={v:g} (range {d.low:g}..{d.high:g})")
    return out


def run_group(ev: Evaluator, names: list[str], search: dict, trials: int, folds: int, seed: int,
              report: list[str], title: str, sampler: str, stop: dict, spec: dict) -> dict:
    """Tune one group of dimensions (CV estimate + final in-sample config). Returns the result.
    With spec: the group's smallest relevant gain (min_gain) sets the search tolerance (unless the
    group's stop gives its own tol) and is returned for the adoption rule."""
    dims = parse_search(search)
    import hashlib
    # stable identity of this group's searches (resume): group, songs, start point, search space
    gkey = hashlib.sha1(json.dumps([title, sorted(names), ev.base_overrides, repr(search), sampler, trials, seed],
                                   default=str).encode()).hexdigest()[:12]
    base_p = ev.pipeline([], names[0])
    expensive, cheap, tail_start = split_dims(base_p, dims, float(spec["cost"]["cheap_seconds"]),
                                              float(spec["cost"]["unknown_seconds"]))
    table = load_noise(spec["data"]["train"])
    if not table:
        import logging
        logging.getLogger("ultrasinger.tune").warning(
            f"FALLBACK: no noise levels for song set {spec['data']['train']!r} in tune/noise.yaml; every group "
            f"uses min_gain only (measure them, or check the path matches the key there)")
    noise_sd, noise_src = group_noise(base_p, dims, table, spec["objective"], len(names))
    threshold = min_gain(spec, noise_sd)
    # search steps may be smaller than the adoption threshold: several sub-noise steps can add up to
    # a gain that then has to clear ~1 SD out of sample
    step = max(float(spec["min_gain"]), float(spec["step_sigma"]) * noise_sd)
    stop = {"tol": step, **stop}
    steps = [st.id for st in base_p.stages]
    dead = probe_effect(ev, names, dims, expensive, cheap, tail_start) if spec["effect_probe"] else []
    if dead:
        import logging
        logging.getLogger("ultrasinger.tune").warning(f"NO EFFECT in this configuration (dropped): {dead}")
        dims = [d for d in dims if d.key not in dead]
        if not dims:
            report += [f"## {title}", "", f"no dimension changes any chart: {dead}", ""]
            return {"title": title, "cv_gain": None, "rows": [], "overrides": [], "base": 0.0, "tuned": 0.0,
                    "min_gain": threshold, "noise_sd": noise_sd, "dead": dead}
        expensive, cheap, tail_start = split_dims(base_p, dims, float(spec["cost"]["cheap_seconds"]),
                                                  float(spec["cost"]["unknown_seconds"]))
    print(f"\n=== {title}\n  expensive (computed per value): {[d.key for d in expensive]}\n"
          f"  cheap (tail from {steps[tail_start] if tail_start < len(steps) else '-'}): {[d.key for d in cheap]}")
    report += [f"## {title}", "", f"noise SD {noise_sd:.4f} ({noise_src}) -> adoption needs {threshold:.4f}, search steps "
               f"count from {step:.4f}", ""]
    if dead:
        report += [f"no effect in this configuration (dropped before the search): {dead}", ""]
    print(f"  noise SD {noise_sd:.4f} ({noise_src}) -> adoption {threshold:.4f}, step {step:.4f}")
    res = {"title": title, "cv_gain": None, "noise_sd": noise_sd, "min_gain": threshold}
    if folds and folds > 1:
        g = artist_groups(names)
        uniq = sorted(set(g.values()))
        rng = np.random.default_rng(seed)
        rng.shuffle(uniq)
        fold_of = {grp: i % folds for i, grp in enumerate(uniq)}
        cv_base, cv_tuned = [], []
        for k in range(folds):
            tr = [x for x in names if fold_of[g[x]] != k]
            te = [x for x in names if fold_of[g[x]] == k]
            if not te or not tr:
                continue
            _, ov, _ = tune(ev, tr, expensive, cheap, tail_start, trials, seed + k,
                            log=lambda m, k=k: print(f"  fold {k + 1}/{folds}{m}"),
                            sampler_name=sampler, stop=stop, study_key=f"{gkey}-fold{k}")
            # same decision rule inside the fold: leave the base only if the training gain beats the
            # noise (scaled from the song set to this fold's size), else a noise-picked winner is scored
            if spec["fold_gate"]:
                gain_tr = (per_song_objective(ev, tr, ov, expensive, cheap, tail_start)
                           - per_song_objective(ev, tr, [], expensive, cheap, tail_start)).mean()
                if gain_tr <= max(float(spec["min_gain"]),
                                  float(spec["noise_sigma"]) * noise_sd * np.sqrt(len(names) / len(tr))):
                    ov = []
            cv_base.append(per_song_objective(ev, te, [], expensive, cheap, tail_start))
            cv_tuned.append(per_song_objective(ev, te, ov, expensive, cheap, tail_start))
        b, t = np.concatenate(cv_base), np.concatenate(cv_tuned)
        d = t - b
        lo, hi = boot_ci(d)
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("inf")
        res.update(cv_gain=float(d.mean()), cv_ci=(lo, hi), cv_se=se, cv_base=float(b.mean()), cv_tuned=float(t.mean()))
        report += [f"cross-validated (each song scored by a config tuned without its artist): base {b.mean():.4f}, "
                   f"tuned {t.mean():.4f}, gain {d.mean():+.4f} +- {se:.4f} (1 SE; 95% CI {lo:+.4f} .. {hi:+.4f}; "
                   f"{int((d > 1e-9).sum())} better, {int((d < -1e-9).sum())} worse of {len(d)})"]
        print("  " + report[-1])
    study, best_ov, rows = tune(ev, names, expensive, cheap, tail_start, trials, seed, log=lambda *_: None,
                                sampler_name=sampler, stop=stop, study_key=f"{gkey}-all")
    base_v = per_song_objective(ev, names, [], expensive, cheap, tail_start)
    tuned_v = per_song_objective(ev, names, best_ov, expensive, cheap, tail_start)
    res.update(overrides=best_ov, base=float(base_v.mean()), tuned=float(tuned_v.mean()), rows=rows,
               expensive=[d.key for d in expensive], cheap=[d.key for d in cheap], study=study,
               tail_start=tail_start, dims=(expensive, cheap))
    report += [f"config tuned on all songs (scored on the same songs, optimistic): {base_v.mean():.4f} -> "
               f"{tuned_v.mean():.4f}", "", "```"] + best_ov + ["```"]
    edges = edge_hits(dims, best_ov)
    if edges:
        report += ["", "at the edge of the search range (widen the range, or check that the objective doesn't "
                       "reward this direction for free): " + ", ".join(edges)]
        import logging
        logging.getLogger("ultrasinger.tune").warning(f"search range edge: {', '.join(edges)}")
    if study is not None and len(dims) > 1 and len(rows) > 3:
        try:
            import optuna
            imp = optuna.importance.get_param_importances(study)
            report += ["", "importance: " + ", ".join(f"{k.split('.', 1)[1]} {v:.2f}" for k, v in imp.items())]
        except Exception:
            pass
    reps = plog.repeats_summary()
    if reps:
        report += ["", "repeated warnings (shown once, then counted):"] + [f"- {r}" for r in reps]
        print("  " + "\n  ".join(reps))
    report.append("")
    return res


def write_outputs(out_dir, spec, overrides, rows, report):
    import yaml
    with open(os.path.join(out_dir, "preset.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"name": f"tuned_{spec['name']}", "base": spec["base"], "overrides": overrides}, f,
                       sort_keys=False, allow_unicode=True)
    with open(os.path.join(out_dir, "trials.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")


def read_previous(out_dir: str, groups: list[dict], from_group: int) -> tuple[list[str], list[dict], list[str]]:
    """For a resumed run: the earlier report (without its result sections), the trial rows of the
    groups before from_group, and the overrides reached so far (preset.yaml)."""
    import yaml
    with open(os.path.join(out_dir, "report.md"), encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    cut = next((i for i, line in enumerate(lines) if line.startswith(("## Result", "## Holdout"))), len(lines))
    # drop sections of groups that are rerun
    rerun = {f"## {i + 1}. {g['name']}" for i, g in enumerate(groups) if i + 1 >= from_group}
    keep, skipping = [], False
    for line in lines[:cut]:
        if line.startswith("## "):
            skipping = line in rerun
        if not skipping:
            keep.append(line)
    done = {g["name"] for i, g in enumerate(groups) if i + 1 < from_group}
    with open(os.path.join(out_dir, "trials.csv"), newline="", encoding="utf-8") as f:
        rows = [{k: v for k, v in r.items() if v != ""} for r in csv.DictReader(f) if r.get("group") in done]
    with open(os.path.join(out_dir, "preset.yaml"), encoding="utf-8") as f:
        overrides = list(yaml.safe_load(f).get("overrides") or [])
    return keep, rows, overrides


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--work", default="eval_work_ng")
    ap.add_argument("--trials", type=int, help="override every group's trial budget (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="cpu | cuda | auto (cuda if available)")
    ap.add_argument("--race", help="compare search methods on one group, e.g. tpe,cmaes,random,coordinate")
    ap.add_argument("--group", help="with --race: the group (name from stages) to race on")
    ap.add_argument("--start", help="with --race: overrides to start from (a preset.yaml written by a tuning run)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in tune/out/<name>/ where it stopped (or start it if there is none): "
                         "finished groups are skipped, a group in progress continues its stored trials")
    ap.add_argument("--fresh", action="store_true", help="discard an earlier run of this spec and start over")
    ap.add_argument("--from-group", type=int, default=1,
                    help="resume: continue the run in tune/out/<name>/ at this group number (report and trials "
                         "are appended to, overrides taken from its preset.yaml)")
    args = ap.parse_args()
    plog.setup(verbose=False)   # our warnings + each distinct library warning once; repeats counted
    global DEVICE
    if args.device == "auto":
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        DEVICE = args.device
    print(f"device: {DEVICE}" + (" (detected)" if args.device == "auto" else ""))
    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    validate_spec(spec)
    data = spec["data"]
    folds = int(data["folds"])
    out_dir = os.path.join(ROOT, "tune", "out", spec["name"])
    progress_path = os.path.join(out_dir, "progress.json")
    if args.fresh and os.path.isdir(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not args.race and os.path.exists(progress_path) and not args.resume and args.from_group == 1:
        raise SystemExit(f"{out_dir} holds an earlier run: --resume to continue it, --fresh to start over")
    if args.resume and os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            args.from_group = json.load(f)["groups_done"] + 1
        print(f"resuming at group {args.from_group}")
    global STORAGE
    STORAGE = "sqlite:///" + os.path.join(out_dir, "studies.db")
    train = discover(data["train"], None)
    holdout = discover(data["holdout"], None)
    overlap = {n for n, _ in train} & {n for n, _ in holdout}
    if overlap:
        raise SystemExit(f"holdout overlaps train: {sorted(overlap)[:5]}")
    ev = Evaluator(spec, args.work, train + holdout)
    names = [n for n, _ in train]
    if spec.get("train_songs", "all") != "all":   # a fixed random subset (same for every group: caches reused)
        k = int(spec["train_songs"])
        if not 0 < k <= len(names):
            raise SystemExit(f"train_songs {k}: need 1..{len(names)}")
        names = sorted(np.random.default_rng(int(spec.get("train_songs_seed", 0))).choice(
            sorted(names), k, replace=False).tolist())
        print(f"train_songs: {k} of {len(train)} songs")
    groups = spec.get("stages") or [{"name": "all", "search": spec["search"]}]
    warm_up(ev, names)
    global POOL
    n_workers = spec["workers"] if spec["workers"] != "auto" else max(1, min(8, (os.cpu_count() or 2) - 2))
    if int(n_workers) > 1:
        POOL = SongPool(int(n_workers), spec, args.work, train + holdout)
        print(f"{n_workers} worker processes for song tails")
    print_eta(ev, names, groups, spec, folds, int(n_workers))
    floored = ev.report_floored(names, lambda n: _write_index(ev, n)) if ev.factors else []
    if args.race:
        return race(ev, names, groups, args, spec, folds, out_dir)
    start = list(ev.base_overrides)
    if args.from_group > 1:
        report, all_rows, ev.base_overrides = read_previous(out_dir, groups, args.from_group)
        # overrides added to the spec since (format-only ones like stems): keep unless already set
        have = {o.split("=", 1)[0] for o in ev.base_overrides}
        ev.base_overrides = [o for o in start if o.split("=", 1)[0] not in have] + ev.base_overrides
        report += [f"(resumed at group {args.from_group})", ""]
    else:
        report = [f"# Tuning report: {spec['name']}", "", f"base preset `{spec['base']}` + {ev.base_overrides}, "
                  f"objective `{spec['objective']}`, {len(names)} songs, {folds} CV folds, holdout "
                  f"{len(holdout)} songs", ""]
        all_rows = []
        if floored:
            report += [f"songs counting at floor weight on the base config (factor metric <= 0.1): {floored}", ""]
    # keep a group only if its cross-validated gain is above the smallest gain that counts:
    # max(min_gain, noise_sigma x noise SD of the step), see group_noise / min_gain
    if len(names) < len(train):
        report += [f"(groups from here on: fixed random subset of {len(names)} of {len(train)} train songs)", ""]
    for gi, group in enumerate(groups):
        if gi + 1 < args.from_group:
            continue
        trials = args.trials or group.get("trials") or spec["trials"]
        global GROUP_LABEL
        GROUP_LABEL = f"| group {gi + 1}/{len(groups)} {group['name']}"
        print(f"\n{bar(gi, len(groups))} groups done; starting {gi + 1}. {group['name']}", flush=True)
        res = run_group(ev, names, group["search"], trials, folds, args.seed + 100 * gi, report,
                        f"{gi + 1}. {group['name']}", group.get("sampler", spec["sampler"]),
                        group.get("stop", spec["stop"]), spec)
        all_rows += [{"group": group["name"], **r} for r in res["rows"]]
        gain = res["cv_gain"] if res["cv_gain"] is not None else res["tuned"] - res["base"]
        if gain > res["min_gain"]:
            changed = [o for o in res["overrides"] if o not in _current_values(ev, names[0])]
            ev.base_overrides += changed
            report += [f"-> adopted: {changed or 'no change (base values were best)'}", ""]
        else:
            report += [f"-> not adopted (gain {gain:+.4f} <= {res['min_gain']:.4f})", ""]
        print("  " + report[-2])
        exp_g, _, _ = split_dims(ev.pipeline([], names[0]), parse_search(group["search"]),
                                 float(spec["cost"]["cheap_seconds"]), float(spec["cost"]["unknown_seconds"]))
        if exp_g:   # full songs: variants of expensive groups would fill the disk
            try:
                print(f"  pruned {prune_side_files(ev, names):.1f} GB of audio files of configurations not adopted")
            except Exception as e:  # noqa: BLE001  (cleanup must never stop the run)
                import logging
                logging.getLogger("ultrasinger.tune").warning(f"FALLBACK: pruning failed ({e}); disk use keeps growing")
        write_outputs(out_dir, spec, ev.base_overrides, all_rows, report)
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"groups_done": gi + 1, "of": len(groups)}, f)

    final = ev.base_overrides
    report += ["## Result", "", f"overrides on top of `{spec['base']}`:", "", "```"] + \
              [o for o in final if o not in start] + ["```", ""]
    hn = [n for n, _ in holdout]
    ev.base_overrides = start
    hb = np.array([ev.objective([ev.score(x, (), [], _write_index(ev, x))]) for x in hn])
    ev.base_overrides = final
    ht = np.array([ev.objective([ev.score(x, (), [], _write_index(ev, x))]) for x in hn])
    lo, hi = boot_ci(ht - hb)
    report += ["## Holdout (scored once)", "", f"{len(hn)} songs: start {hb.mean():.4f}, tuned {ht.mean():.4f}, "
               f"gain {np.mean(ht - hb):+.4f} (95% CI {lo:+.4f} .. {hi:+.4f})", ""]
    write_outputs(out_dir, spec, final, all_rows, report)
    print("\n".join(report))
    print(f"\nwrote {out_dir}/report.md, preset.yaml, trials.csv")
    if POOL is not None:
        POOL.close()


def race(ev, names, groups, args, spec, folds, out_dir):
    """Same group, same budget, same CV folds: which search method finds configs that generalise?"""
    import yaml
    group = next(g for g in groups if g["name"] == args.group) if args.group else groups[-1]
    if args.start:
        with open(args.start, encoding="utf-8") as f:
            ev.base_overrides = list(yaml.safe_load(f).get("overrides") or [])
    trials = args.trials or group.get("trials") or spec["trials"]
    report = [f"# Search method race: {group['name']}", "", f"{len(names)} songs, {folds} folds, budget {trials} "
              f"evaluations per search, start {ev.base_overrides}", "",
              "| method | CV gain | 95% CI | in-sample | evaluations | seconds |", "|---|---|---|---|---|---|"]
    for name in args.race.split(","):
        t0 = time.time()
        res = run_group(ev, names, group["search"], trials, folds, args.seed, [], name, name.strip(),
                        group.get("stop", spec["stop"]), spec)
        report.append(f"| {name} | {res['cv_gain']:+.4f} | {res['cv_ci'][0]:+.4f} .. {res['cv_ci'][1]:+.4f} | "
                      f"{res['tuned'] - res['base']:+.4f} | {len(res['rows'])} | {time.time() - t0:.0f} |")
        print(report[-1])
    path = os.path.join(out_dir, f"race_{group['name'].replace(' ', '_')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report) + f"\n\nwrote {path}")


RUN_T0 = time.time()


def prune_side_files(ev: Evaluator, names: list[str]) -> float:
    """After an expensive group: delete audio files this run wrote for configurations that were not
    adopted (separation stems, prepared audio, intermediates). Kept: everything the current base's
    prefixes point to, everything older than this run, stage pickles and sidecars. A cached result
    whose file is gone is recomputed (core._missing_files). Returns GB freed."""
    freed = 0
    for n in names:
        store = ev.prefix(n, (), _write_index(ev, n))
        keep = {os.path.realpath(v) for v in store.values() if isinstance(v, str) and os.path.isabs(v)}
        cache = os.path.abspath(os.path.join(ev.work, "cache", slug(n)))
        for root, dirs, files in os.walk(cache, topdown=False):
            for f in files:
                path = os.path.join(root, f)
                if f.endswith((".pkl", ".json")) or os.path.realpath(path) in keep:
                    continue
                if os.path.getmtime(path) >= RUN_T0:
                    freed += os.path.getsize(path)
                    os.remove(path)
            if root != cache and not os.listdir(root):
                os.rmdir(root)
    return freed / 1e9


def _write_index(ev: Evaluator, name: str) -> int:
    p = ev.pipeline([], name)
    return next((i for i, st in enumerate(p.stages) if st.name.startswith("write.")), len(p.stages))


def _current_values(ev: Evaluator, name: str) -> set[str]:
    """Overrides that would restate the current configuration (to report only real changes)."""
    p = ev.pipeline([], name)
    out = set()
    for st in p.stages:
        out.add(f"{st.id}.use={st.name}")
        out.add(f"{st.id}.enabled=true")
        for k, v in vars(st.params).items():
            out.add(f"{st.id}.{k}={json.dumps(v) if not isinstance(v, str) else v}")
    return out


if __name__ == "__main__":
    main()
