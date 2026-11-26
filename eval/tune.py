"""Tune the whole inference machine on a dataset: module choices, step order and parameters.

    python eval/tune.py tune/notes.yaml [--trials 100] [--work eval_work_ng]

A tuning spec (YAML) names a base preset, the data, the objective and the search space:

    name: notes_v1
    base: bench                         # preset to start from
    data:
      train: eval_singstar/bench        # song folders (charts + audio)
      holdout: eval_singstar/selection  # optional: scored once, at the very end
      folds: 4                          # cross-validation folds over train, grouped by artist
      gt_language: true                 # language from the chart (short windows)
    objective: {usdx_medium: 1.0, wer: -0.25}   # weighted sum of per-song means
    search:
      notes.calibrate.offset: {low: -0.15, high: 0.0}       # continuous
      notes.snap_edges.max_shift: {low: 0.0, high: 0.8}
      notes.from_pitch.method: [chroma_mode, median]         # discrete choices
      segment.game.use: [segment.game, segment.contour]      # module alternatives for a step
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

from pipeline import stages  # noqa: E402,F401  (registers the stages)
from pipeline.core import Context, Pipeline, load_preset  # noqa: E402
from pipeline.run import media_from_filename, preset_path  # noqa: E402

LOWER_IS_BETTER = {"wer", "cer", "pitch_mae"}
CHEAP_SECONDS = 3.0    # rerunning from a step to the end faster than this (per 30 s of audio) = cheap
UNKNOWN_SECONDS = 1.0  # cost assumed for stages without runtime records


# ----------------------------------------------------------------------------- search space

class Dim:
    """One search dimension -> override string for a value."""

    def __init__(self, key: str, spec):
        self.key = key
        self.step, self.attr = key.rsplit(".", 1)
        if isinstance(spec, dict):
            self.kind = "int" if isinstance(spec.get("low"), int) and isinstance(spec.get("high"), int) \
                and not spec.get("float") else "float"
            self.low, self.high, self.log = spec["low"], spec["high"], bool(spec.get("log"))
            self.choices = None
        else:
            self.kind, self.choices = "choice", list(spec)

    def suggest(self, trial):
        if self.kind == "choice":
            return trial.suggest_categorical(self.key, self.choices)
        if self.kind == "int":
            return trial.suggest_int(self.key, self.low, self.high, log=self.log)
        return trial.suggest_float(self.key, self.low, self.high, log=self.log)

    def override(self, value) -> str:
        return f"{self.step}.{self.attr}={json.dumps(value) if not isinstance(value, str) else value}"


def step_cost(stage_name: str) -> float | None:
    """Seconds per 30 s of audio from the runtime records (None: unknown)."""
    from pipeline import runtimes
    recs = [r for r in runtimes.load() if r.get("stage") == stage_name and r.get("audio_seconds")]
    if not recs:
        return None
    return float(np.median([r["seconds"] / r["audio_seconds"] * 30.0 for r in recs[-50:]]))


def split_dims(base_pipeline: Pipeline, dims: list[Dim]) -> tuple[list[Dim], list[Dim], int]:
    """Cheap dims: rerunning from their step to the end costs at most CHEAP_SECONDS per 30 s of
    audio (runtime records; unknown stages count UNKNOWN_SECONDS). Everything else is expensive:
    discrete values only, each computed once (disk cache). Returns the tail start: the earliest step
    a cheap dim can touch, never after the first writer (charts are written per trial)."""
    ids = [st.id for st in base_pipeline.stages]
    names = [st.name for st in base_pipeline.stages]
    cost = [step_cost(n) for n in names]
    cost = [UNKNOWN_SECONDS if c is None else c for c in cost]

    def tail_cost(i: int, alt_stage: str | None = None) -> float:
        extra = 0.0
        if alt_stage:
            c = step_cost(alt_stage)
            extra = (UNKNOWN_SECONDS if c is None else c) - cost[i]
        return sum(cost[i:]) + max(extra, 0.0)

    expensive, cheap = [], []
    for d in dims:
        if d.step not in ids:
            raise ValueError(f"search dimension {d.key}: no step '{d.step}' in the base preset")
        i = ids.index(d.step)
        if d.attr in ("after", "before"):
            i = min([i] + [ids.index(c) + (1 if d.attr == "after" else 0) for c in d.choices if c in ids])
        c = max(tail_cost(i, alt) for alt in d.choices) if d.attr == "use" else tail_cost(i)
        if c <= CHEAP_SECONDS:
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
        self.weights = spec["objective"]

    def lang_overrides(self, name: str, step_ids: set[str]) -> list[str]:
        if not self.spec.get("data", {}).get("gt_language"):
            return []
        code = language_code(self.gt[name].language)
        return [f"{s}.language={code}" for s in LANGUAGE_STEPS if s in step_ids]

    def pipeline(self, overrides: list[str], name: str) -> Pipeline:
        ov = self.base_overrides + overrides
        p = load_preset(preset_path(self.base), ov)
        return load_preset(preset_path(self.base), self.lang_overrides(name, {st.id for st in p.stages}) + ov)

    def _ctx(self, name: str, no_cache: bool) -> Context:
        gt = self.gt[name]
        out = os.path.join(self.tmp, slug(name))
        os.makedirs(out, exist_ok=True)
        return Context(cache_dir=os.path.abspath(os.path.join(self.work, "cache", slug(name))), output_dir=out,
                       device="cpu", basename=f"{gt.artist} - {gt.title}", verbose=False,
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
        pred = usdx.read(store["file.txt"])
        m = metrics.evaluate(self.gt[name], pred)
        return {k: m.get(k) for k in self.weights}

    def song_objective(self, d: dict) -> float:
        """Weighted sum of one song's metrics (a metric a song doesn't have, e.g. WER for
        non-Latin lyrics or line F1 without breaks, is left out for that song)."""
        return float(sum(w * d[k] for k, w in self.weights.items() if d.get(k) is not None and d.get(k) == d.get(k)))

    def objective(self, per_song: list[dict]) -> float:
        """Mean over songs of the per-song objective (the same aggregation everywhere)."""
        return float(np.mean([self.song_objective(d) for d in per_song])) if per_song else 0.0


def _sub(p: Pipeline, a: int, b: int) -> Pipeline:
    q = Pipeline.__new__(Pipeline)
    q.name, q.params, q.param_sources = p.name, p.params, p.param_sources
    q.specs, q.stages = p.specs[a:b], p.stages[a:b]
    return q


# ----------------------------------------------------------------------------- tuning

def coordinate_sampler_trials(dims: list[Dim], base: dict, points: int = 7):
    """Coordinate descent as a generator of parameter dicts: one dimension at a time over its values
    (continuous: `points` evenly spaced), keeping the best so far; cycles until a full pass brings
    nothing. Driven by tune() through optuna's enqueue mechanism."""
    def values(d):
        if d.kind == "choice":
            return list(d.choices)
        if d.kind == "int":
            return sorted({int(round(v)) for v in np.linspace(d.low, d.high, points)})
        return [float(v) for v in np.linspace(d.low, d.high, points)]
    return {d.key: values(d) for d in dims}


def tune(ev: Evaluator, songs: list[str], expensive: list[Dim], cheap: list[Dim], tail_start: int,
         trials: int, seed: int, log=print, sampler_name: str = "auto", stop: dict | None = None):
    """Search the dimensions on these songs. sampler_name: auto (grid when everything is discrete and
    fits the budget, else TPE) | grid | tpe | random | cmaes | coordinate. The search ends at whichever
    comes first: `trials` evaluations; no improvement > stop["tol"] for stop["patience"] trials;
    stop["max_seconds"] wall time."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sampler_name == "coordinate":
        return _coordinate(ev, songs, expensive, cheap, tail_start, trials, stop)
    rows = []

    def objective(trial):
        exp_ov = tuple(d.override(d.suggest(trial)) for d in expensive)
        cheap_ov = [d.override(d.suggest(trial)) for d in cheap]
        per_song = [ev.score(s, exp_ov, cheap_ov, tail_start) for s in songs]
        val = ev.objective(per_song)
        rows.append({"trial": trial.number, "objective": val, **trial.params})
        return val

    dims = expensive + cheap
    combos = int(np.prod([len(d.choices) for d in dims])) if all(d.kind == "choice" for d in dims) else None
    if (sampler_name in ("auto", "grid")) and combos is not None and combos <= trials:
        sampler = optuna.samplers.GridSampler({d.key: d.choices for d in dims}, seed=seed)
        trials = combos
    elif sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "cmaes":
        sampler = optuna.samplers.CmaEsSampler(seed=seed, warn_independent_sampling=False,
                                               independent_sampler=optuna.samplers.TPESampler(seed=seed))
    else:
        combos = None
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    # trial 0 = the base preset (so "tuned" can never be worse than base on the tuning songs)
    base = {}
    p0 = ev.pipeline([], songs[0])
    vals = {f"{st.id}.{k}": v for st in p0.stages for k, v in vars(st.params).items()}
    ids = [st.id for st in p0.stages]
    for d in expensive + cheap:
        if d.attr == "use":
            base[d.key] = p0.stages[ids.index(d.step)].name
        elif d.attr in ("after", "before"):
            i = ids.index(d.step)
            anchor = ids[i - 1] if d.attr == "after" and i > 0 else (ids[i + 1] if i + 1 < len(ids) else None)
            if anchor in (d.choices or []):
                base[d.key] = anchor
        elif d.key in vals:
            v = vals[d.key]
            if d.kind == "choice" and v not in d.choices:
                continue
            if d.kind != "choice" and not (d.low <= v <= d.high):
                continue
            base[d.key] = v
    if len(base) == len(expensive) + len(cheap) and combos is None:
        study.enqueue_trial(base)
    t0 = time.time()
    callbacks = [lambda s, t: log(f"  trial {t.number:3d}  {t.value:.4f}  best {s.best_value:.4f}"
                                  f"  ({time.time() - t0:.0f}s)") if t.number % 10 == 0 else None]
    if stop:
        hist = []

        def early_stop(st, t):
            hist.append(st.best_value)
            k = stop.get("patience")
            if k and len(hist) > k and hist[-1] - hist[-1 - k] <= float(stop.get("tol", 0.0)):
                st.stop()
            if stop.get("max_seconds") and time.time() - t0 > float(stop["max_seconds"]):
                st.stop()
        callbacks.append(early_stop)
    study.optimize(objective, n_trials=trials, callbacks=callbacks)
    best_ov_exp = tuple(d.override(study.best_params[d.key]) for d in expensive)
    best_ov_cheap = [d.override(study.best_params[d.key]) for d in cheap]
    return study, list(best_ov_exp) + best_ov_cheap, rows


def _coordinate(ev, songs, expensive, cheap, tail_start, budget, stop=None):
    """Coordinate descent (see coordinate_sampler_trials); returns (None, best overrides, rows).
    A move counts only if it improves by more than stop["tol"]; stops at budget or stop["max_seconds"]."""
    stop = stop or {}
    tol = float(stop.get("tol", 0.0))
    t0 = time.time()
    dims = expensive + cheap
    grid = coordinate_sampler_trials(dims, {})
    p0 = ev.pipeline([], songs[0])
    cur = {}
    for d in dims:
        st = next(x for x in p0.stages if x.id == d.step)
        v = st.name if d.attr == "use" else getattr(st.params, d.attr, None)
        cur[d.key] = v if v in grid[d.key] else grid[d.key][0]
    seen, rows = {}, []

    def evaluate(cfg):
        key = tuple(sorted(cfg.items(), key=lambda kv: kv[0]))
        if key not in seen:
            exp_ov = tuple(d.override(cfg[d.key]) for d in expensive)
            cheap_ov = [d.override(cfg[d.key]) for d in cheap]
            seen[key] = ev.objective([ev.score(x, exp_ov, cheap_ov, tail_start) for x in songs])
            rows.append({"trial": len(rows), "objective": seen[key], **cfg})
        return seen[key]

    best = evaluate(cur)
    improved = True
    while improved and len(rows) < budget:
        improved = False
        for d in dims:
            for v in grid[d.key]:
                if len(rows) >= budget or (stop.get("max_seconds") and time.time() - t0 > stop["max_seconds"]):
                    break
                cand = {**cur, d.key: v}
                val = evaluate(cand)
                if val > best + max(tol, 1e-12):
                    best, cur, improved = val, cand, True
    overrides = [d.override(cur[d.key]) for d in dims]
    return None, overrides, rows


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


def group_noise(pipeline, dims: list[Dim], table: dict) -> tuple[float, str]:
    """Noise SD for a group: a change re-rolls the first chaotic module at or after the group's earliest
    step, so use that module's saved noise. Exact engine first; same module with another engine is
    borrowed (flagged); no chaotic module downstream -> 0. Returns (sd, source)."""
    ids = [st.id for st in pipeline.stages]
    first = min((ids.index(d.step) for d in dims if d.step in ids), default=0)
    for st in pipeline.stages[first:]:
        key = engine_key(st)
        if key in table:
            return float(table[key]["sd"]), key
        same = [k for k in table if k.split("[")[0] == st.name]
        if same:
            return float(table[same[0]]["sd"]), f"{same[0]} (borrowed for {key}: measure it)"
    return 0.0, "deterministic downstream"


def min_gain(spec: dict, noise_sd: float) -> float:
    """Smallest gain that counts: the larger of the absolute floor and noise_sigma x the step's noise SD."""
    return max(float(spec.get("min_gain", 0.001)), float(spec.get("noise_sigma", 1.0)) * noise_sd)


def per_song_objective(ev, songs, overrides, expensive, cheap, tail_start) -> np.ndarray:
    exp_keys = {d.step + "." + d.attr for d in expensive}
    exp_ov = tuple(o for o in overrides if o.split("=", 1)[0] in exp_keys)
    cheap_ov = [o for o in overrides if o.split("=", 1)[0] not in exp_keys]
    return np.array([ev.objective([ev.score(s, exp_ov, cheap_ov, tail_start)]) for s in songs])


def artist_groups(songs: list[str]) -> dict[str, str]:
    """Song (or window 'Song @12s') -> artist group from the SingStar split, else the artist."""
    path = os.path.join(HERE, "splits", "singstar_split.csv")
    groups = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            groups = {r["song"]: r["group"] for r in csv.DictReader(f)}
    return {s: groups.get(s.split(" @")[0], s.split(" - ")[0]) for s in songs}


def boot_ci(x: np.ndarray, n: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run_group(ev: Evaluator, names: list[str], search: dict, trials: int, folds: int, seed: int,
              report: list[str], title: str, sampler: str = "auto", stop: dict | None = None,
              spec: dict | None = None) -> dict:
    """Tune one group of dimensions (CV estimate + final in-sample config). Returns the result.
    With spec: the group's smallest relevant gain (min_gain) sets the search tolerance (unless the
    group's stop gives its own tol) and is returned for the adoption rule."""
    dims = [Dim(k, v) for k, v in search.items()]
    base_p = ev.pipeline([], names[0])
    expensive, cheap, tail_start = split_dims(base_p, dims)
    spec = spec or {}
    table = spec.get("noise") or load_noise(spec["data"]["train"])
    noise_sd, noise_src = group_noise(base_p, dims, table)
    threshold = min_gain(spec, noise_sd)
    stop = {"tol": threshold, **(stop or {})}
    steps = [st.id for st in base_p.stages]
    print(f"\n=== {title}\n  expensive (computed per value): {[d.key for d in expensive]}\n"
          f"  cheap (tail from {steps[tail_start] if tail_start < len(steps) else '-'}): {[d.key for d in cheap]}")
    report += [f"## {title}", "", f"noise SD {noise_sd:.4f} ({noise_src}) -> smallest gain that counts {threshold:.4f}", ""]
    print(f"  noise SD {noise_sd:.4f} ({noise_src}) -> min gain {threshold:.4f}")
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
            _, ov, _ = tune(ev, tr, expensive, cheap, tail_start, trials, seed + k, log=lambda *_: None,
                            sampler_name=sampler, stop=stop)
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
                                sampler_name=sampler, stop=stop)
    base_v = per_song_objective(ev, names, [], expensive, cheap, tail_start)
    tuned_v = per_song_objective(ev, names, best_ov, expensive, cheap, tail_start)
    res.update(overrides=best_ov, base=float(base_v.mean()), tuned=float(tuned_v.mean()), rows=rows,
               expensive=[d.key for d in expensive], cheap=[d.key for d in cheap], study=study,
               tail_start=tail_start, dims=(expensive, cheap))
    report += [f"config tuned on all songs (scored on the same songs, optimistic): {base_v.mean():.4f} -> "
               f"{tuned_v.mean():.4f}", "", "```"] + best_ov + ["```"]
    if study is not None and len(dims) > 1 and len(rows) > 3:
        try:
            import optuna
            imp = optuna.importance.get_param_importances(study)
            report += ["", "importance: " + ", ".join(f"{k.split('.', 1)[1]} {v:.2f}" for k, v in imp.items())]
        except Exception:
            pass
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


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--work", default="eval_work_ng")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--folds", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--race", help="compare search methods on one group, e.g. tpe,cmaes,random,coordinate")
    ap.add_argument("--group", help="with --race: the group (name from stages) to race on")
    ap.add_argument("--start", help="overrides to start from (a preset.yaml written by a tuning run)")
    ap.add_argument("--from-group", type=int, default=1, help="resume at this group number (use with --start)")
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    data = spec["data"]
    folds = args.folds if args.folds is not None else data.get("folds", 4)
    out_dir = os.path.join(ROOT, "tune", "out", spec["name"])
    os.makedirs(out_dir, exist_ok=True)
    train = discover(data["train"], None)
    holdout = discover(data["holdout"], None) if data.get("holdout") else []
    ev = Evaluator(spec, args.work, train + holdout)
    names = [n for n, _ in train]
    report = [f"# Tuning report: {spec['name']}", "", f"base preset `{spec['base']}` + {ev.base_overrides}, "
              f"objective `{spec['objective']}`, {len(names)} songs, {folds} CV folds", ""]
    groups = spec.get("stages") or [{"name": "all", "search": spec["search"], "trials": spec.get("trials")}]
    if args.race:
        return race(ev, names, groups, args, spec, folds, out_dir)
    all_rows, start = [], list(ev.base_overrides)
    if args.start:
        with open(args.start, encoding="utf-8") as f:
            ev.base_overrides = list(yaml.safe_load(f).get("overrides") or [])
        report += [f"resumed at group {args.from_group} from {args.start}: {ev.base_overrides}", ""]
    # keep a group only if its cross-validated gain is above the smallest gain that counts:
    # max(min_gain, noise_sigma x noise SD of the step), see group_noise / min_gain
    for gi, group in enumerate(groups):
        if gi + 1 < args.from_group:
            continue
        trials = args.trials or group.get("trials") or spec.get("trials", 50)
        res = run_group(ev, names, group["search"], trials, folds, args.seed + 100 * gi, report,
                        f"{gi + 1}. {group['name']}", group.get("sampler", spec.get("sampler", "auto")),
                        group.get("stop", spec.get("stop")), spec)
        all_rows += [{"group": group["name"], **r} for r in res["rows"]]
        gain = res["cv_gain"] if res["cv_gain"] is not None else res["tuned"] - res["base"]
        if gain > res["min_gain"]:
            changed = [o for o in res["overrides"] if o not in _current_values(ev, names[0])]
            ev.base_overrides += changed
            report += [f"-> adopted: {changed or 'no change (base values were best)'}", ""]
        else:
            report += [f"-> not adopted (gain {gain:+.4f} <= {res['min_gain']:.4f})", ""]
        print("  " + report[-2])
        write_outputs(out_dir, spec, ev.base_overrides, all_rows, report)

    final = ev.base_overrides
    report += ["## Result", "", f"overrides on top of `{spec['base']}`:", "", "```"] + \
              [o for o in final if o not in start] + ["```", ""]
    if holdout:
        hn = [n for n, _ in holdout]
        ev_start = list(ev.base_overrides)
        ev.base_overrides = start
        hb = np.array([ev.objective([ev.score(x, (), [], _write_index(ev, x))]) for x in hn])
        ev.base_overrides = ev_start
        ht = np.array([ev.objective([ev.score(x, (), [], _write_index(ev, x))]) for x in hn])
        lo, hi = boot_ci(ht - hb)
        report += ["## Holdout (scored once)", "", f"{len(hn)} songs: start {hb.mean():.4f}, tuned {ht.mean():.4f}, "
                   f"gain {np.mean(ht - hb):+.4f} (95% CI {lo:+.4f} .. {hi:+.4f})", ""]
    write_outputs(out_dir, spec, final, all_rows, report)
    print("\n".join(report))
    print(f"\nwrote {out_dir}/report.md, preset.yaml, trials.csv")


def race(ev, names, groups, args, spec, folds, out_dir):
    """Same group, same budget, same CV folds: which search method finds configs that generalise?"""
    import yaml
    group = next(g for g in groups if g["name"] == args.group) if args.group else groups[-1]
    if args.start:
        with open(args.start, encoding="utf-8") as f:
            ev.base_overrides = list(yaml.safe_load(f).get("overrides") or [])
    trials = args.trials or group.get("trials") or spec.get("trials", 50)
    report = [f"# Search method race: {group['name']}", "", f"{len(names)} songs, {folds} folds, budget {trials} "
              f"evaluations per search, start {ev.base_overrides}", "",
              "| method | CV gain | 95% CI | in-sample | evaluations | seconds |", "|---|---|---|---|---|---|"]
    for name in args.race.split(","):
        t0 = time.time()
        res = run_group(ev, names, group["search"], trials, folds, args.seed, [], name, sampler=name.strip())
        report.append(f"| {name} | {res['cv_gain']:+.4f} | {res['cv_ci'][0]:+.4f} .. {res['cv_ci'][1]:+.4f} | "
                      f"{res['tuned'] - res['base']:+.4f} | {len(res['rows'])} | {time.time() - t0:.0f} |")
        print(report[-1])
    path = os.path.join(out_dir, f"race_{group['name'].replace(' ', '_')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report) + f"\n\nwrote {path}")


def _write_index(ev: Evaluator, name: str) -> int:
    p = ev.pipeline([], name)
    return next((i for i, st in enumerate(p.stages) if st.name.startswith("write.")), len(p.stages))


def _current_values(ev: Evaluator, name: str) -> set[str]:
    """Overrides that would restate the current configuration (to report only real changes)."""
    p = ev.pipeline([], name)
    out = set()
    for st in p.stages:
        out.add(f"{st.id}.use={st.name}")
        for k, v in vars(st.params).items():
            out.add(f"{st.id}.{k}={json.dumps(v) if not isinstance(v, str) else v}")
    return out


if __name__ == "__main__":
    main()
