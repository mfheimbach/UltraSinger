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

from pipeline import stages  # noqa: E402,F401
from pipeline.core import Context, Pipeline, load_preset  # noqa: E402
from pipeline.run import media_from_filename, preset_path  # noqa: E402

LOWER_IS_BETTER = {"wer", "cer", "pitch_mae"}
CHEAP_SECONDS = 2.0   # stages faster than this per 30 s of audio (runtime records) count as cheap


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
    """Expensive dims (their step is costly or unknown-but-early) vs cheap dims, and the tail start:
    the earliest step index any cheap dim can touch."""
    ids = [st.id for st in base_pipeline.stages]
    expensive, cheap = [], []
    for d in dims:
        if d.step not in ids:
            raise ValueError(f"search dimension {d.key}: no step '{d.step}' in the base preset")
        stage_name = base_pipeline.stages[ids.index(d.step)].name
        cost = step_cost(stage_name)
        if d.attr == "use":
            costs = [step_cost(c) for c in d.choices]
            cost = max([c for c in costs if c is not None] + [cost or 0.0])
        if cost is not None and cost <= CHEAP_SECONDS:
            cheap.append(d)
        else:
            if d.kind != "choice":
                raise ValueError(f"{d.key}: step {stage_name} is expensive (or has no runtime records yet); "
                                 f"give a list of values instead of a range")
            expensive.append(d)
    positions = []
    for d in cheap:
        positions.append(ids.index(d.step))
        if d.attr in ("after", "before"):
            positions += [ids.index(c) + (1 if d.attr == "after" else 0) for c in d.choices if c in ids]
    tail_start = min(positions) if positions else len(ids)
    return expensive, cheap, tail_start


# ----------------------------------------------------------------------------- evaluation

class Evaluator:
    """Prefix per (song, expensive choices) computed once; tails run in memory per trial."""

    def __init__(self, spec: dict, work: str, songs: list[tuple[str, str]]):
        self.spec, self.work = spec, work
        self.base = spec["base"]
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
        p = load_preset(preset_path(self.base), overrides)
        return load_preset(preset_path(self.base), self.lang_overrides(name, {st.id for st in p.stages}) + overrides)

    def _ctx(self, name: str, no_cache: bool) -> Context:
        gt = self.gt[name]
        out = os.path.join(self.tmp, slug(name))
        os.makedirs(out, exist_ok=True)
        return Context(cache_dir=os.path.abspath(os.path.join(self.work, "cache", slug(name))), output_dir=out,
                       device="cpu", basename=f"{gt.artist} - {gt.title}", verbose=False,
                       extra={"no_cache": no_cache})

    def prefix(self, name: str, expensive_ov: tuple, tail_start: int) -> dict:
        key = (name, expensive_ov)
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

    def objective(self, per_song: list[dict]) -> float:
        total = 0.0
        for k, w in self.weights.items():
            vals = [d[k] for d in per_song if d.get(k) is not None and d.get(k) == d.get(k)]
            total += w * (float(np.mean(vals)) if vals else 0.0)
        return total


def _sub(p: Pipeline, a: int, b: int) -> Pipeline:
    q = Pipeline.__new__(Pipeline)
    q.name, q.params, q.param_sources = p.name, p.params, p.param_sources
    q.specs, q.stages = p.specs[a:b], p.stages[a:b]
    return q


# ----------------------------------------------------------------------------- tuning

def tune(ev: Evaluator, songs: list[str], expensive: list[Dim], cheap: list[Dim], tail_start: int,
         trials: int, seed: int, log=print):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    rows = []

    def objective(trial):
        exp_ov = tuple(d.override(d.suggest(trial)) for d in expensive)
        cheap_ov = [d.override(d.suggest(trial)) for d in cheap]
        per_song = [ev.score(s, exp_ov, cheap_ov, tail_start) for s in songs]
        val = ev.objective(per_song)
        rows.append({"trial": trial.number, "objective": val, **trial.params})
        return val

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
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
    if len(base) == len(expensive) + len(cheap):
        study.enqueue_trial(base)
    t0 = time.time()
    study.optimize(objective, n_trials=trials,
                   callbacks=[lambda s, t: log(f"  trial {t.number:3d}  {t.value:.4f}  best {s.best_value:.4f}"
                                               f"  ({time.time() - t0:.0f}s)") if t.number % 10 == 0 else None])
    best_ov_exp = tuple(d.override(study.best_params[d.key]) for d in expensive)
    best_ov_cheap = [d.override(study.best_params[d.key]) for d in cheap]
    return study, list(best_ov_exp) + best_ov_cheap, rows


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


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--work", default="eval_work_ng")
    ap.add_argument("--trials", type=int)
    ap.add_argument("--folds", type=int)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    with open(args.spec, encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    trials = args.trials or spec.get("trials", 50)
    data = spec["data"]
    folds = args.folds if args.folds is not None else data.get("folds", 4)
    out_dir = os.path.join(ROOT, "tune", "out", spec["name"])
    os.makedirs(out_dir, exist_ok=True)
    report = [f"# Tuning report: {spec['name']}", "", f"base preset `{spec['base']}`, objective "
              f"`{spec['objective']}`, {trials} trials", ""]

    train = discover(data["train"], None)
    holdout = discover(data["holdout"], None) if data.get("holdout") else []
    ev = Evaluator(spec, args.work, train + holdout)
    dims = [Dim(k, v) for k, v in spec["search"].items()]
    expensive, cheap, tail_start = split_dims(ev.pipeline([], train[0][0]), dims)
    steps = [st.id for st in ev.pipeline([], train[0][0]).stages]
    print(f"expensive (outer, on demand): {[d.key for d in expensive]}")
    print(f"cheap (tail from step {tail_start}: {steps[tail_start] if tail_start < len(steps) else '-'}): "
          f"{[d.key for d in cheap]}")
    names = [n for n, _ in train]

    # cross-validation over artist groups
    if folds and folds > 1:
        g = artist_groups(names)
        uniq = sorted(set(g.values()))
        rng = np.random.default_rng(args.seed)
        rng.shuffle(uniq)
        fold_of = {grp: i % folds for i, grp in enumerate(uniq)}
        cv_base, cv_tuned = [], []
        for k in range(folds):
            tr = [s for s in names if fold_of[g[s]] != k]
            te = [s for s in names if fold_of[g[s]] == k]
            if not te or not tr:
                continue
            print(f"fold {k + 1}/{folds}: tune on {len(tr)}, score {len(te)}")
            _, ov, _ = tune(ev, tr, expensive, cheap, tail_start, trials, args.seed + k)
            cv_base.append(per_song_objective(ev, te, [], expensive, cheap, tail_start))
            cv_tuned.append(per_song_objective(ev, te, ov, expensive, cheap, tail_start))
        b, t = np.concatenate(cv_base), np.concatenate(cv_tuned)
        lo, hi = boot_ci(t - b)
        report += ["## Cross-validation (tuned on other artists, scored on held-out artists)", "",
                   f"base {b.mean():.4f}, tuned {t.mean():.4f}, gain {np.mean(t - b):+.4f} "
                   f"(95% CI {lo:+.4f} .. {hi:+.4f}), {int((t > b + 1e-9).sum())} songs better, "
                   f"{int((t < b - 1e-9).sum())} worse", ""]
        print(report[-2])

    # final: tune on all train songs
    print(f"final: tune on all {len(names)} train songs")
    study, best_ov, rows = tune(ev, names, expensive, cheap, tail_start, trials, args.seed)
    base_train = per_song_objective(ev, names, [], expensive, cheap, tail_start)
    tuned_train = per_song_objective(ev, names, best_ov, expensive, cheap, tail_start)
    report += ["## Final configuration (tuned on all train songs; in-sample)", "",
               f"base {base_train.mean():.4f}, tuned {tuned_train.mean():.4f} (in-sample gain "
               f"{np.mean(tuned_train - base_train):+.4f}; see CV for the honest number)", "", "```"]
    report += best_ov + ["```", ""]
    try:
        import optuna
        imp = optuna.importance.get_param_importances(study)
        report += ["Which dimensions mattered (fANOVA importance):", ""]
        report += [f"- {k}: {v:.2f}" for k, v in imp.items()] + [""]
    except Exception as err:
        report += [f"(importance not available: {err})", ""]

    if holdout:
        hn = [n for n, _ in holdout]
        hb = per_song_objective(ev, hn, [], expensive, cheap, tail_start)
        ht = per_song_objective(ev, hn, best_ov, expensive, cheap, tail_start)
        lo, hi = boot_ci(ht - hb)
        report += ["## Holdout (scored once)", "", f"{len(hn)} songs: base {hb.mean():.4f}, tuned {ht.mean():.4f}, "
                   f"gain {np.mean(ht - hb):+.4f} (95% CI {lo:+.4f} .. {hi:+.4f})", ""]
        print(report[-2])

    with open(os.path.join(out_dir, "preset.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"name": f"tuned_{spec['name']}", "base": spec["base"], "overrides": best_ov}, f,
                       sort_keys=False, allow_unicode=True)
    with open(os.path.join(out_dir, "trials.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nwrote {out_dir}/report.md, preset.yaml, trials.csv")


if __name__ == "__main__":
    main()
