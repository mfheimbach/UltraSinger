"""Modular pipeline core: typed artifacts, registered stages, cached runs, presets.

A run keeps a store of named artifacts (``audio.vocals``, ``words``, ``notes`` ...). Each stage
declares the artifacts it reads and writes; a pipeline is an ordered list of stages. Any order
works as long as every input exists before the stage that reads it (checked up front), so stages
with the same signature (e.g. text in / text out) can be swapped or moved.

Slots can accept several artifacts: ``"syllables|words"`` binds to whichever of them was produced
most recently, and an output ``"@slot"`` writes back to the artifact that slot was bound to. So a
text stage (punctuation cleanup, silence split, a segmenter) works before or after hyphenation
without changes. Overrides move and insert steps: ``step.after=other``, ``step.before=other``,
``+stage@after=step``. ``Pipeline.where(step)`` lists every position a step could take.

Caching: a stage's result is stored under a key made from its name, version, parameters, the
content of its inputs, the upstream step chain and the stage's code. Changing anything upstream
invalidates exactly what depends on it.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import pickle
import re
import time
from dataclasses import dataclass, field
from typing import Any

# ----------------------------------------------------------------------------- artifacts

# Artifact kinds; an artifact's kind is the part of its name before the first dot
# (audio.vocals -> audio, pitch.rmvpe -> pitch). Bare names are their own kind.


_FILE_FP: dict[tuple, str] = {}


def _file_fingerprint(path: str) -> str:
    """Full-content hash of a file, memoized per (path, size, mtime)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _FILE_FP:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        _FILE_FP[key] = h.hexdigest()[:16]
    return _FILE_FP[key]


def _dir_fingerprint(path: str) -> str:
    """Model folders: hash of every file's content (memoized like single files)."""
    h = hashlib.sha1()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            h.update(os.path.relpath(full, path).encode())
            h.update(_file_fingerprint(full).encode())
    return h.hexdigest()[:16]


def fingerprint(value: Any) -> str:
    """Stable content hash of an artifact (files: full content)."""
    if isinstance(value, str) and os.path.isfile(value):
        return _file_fingerprint(value)
    try:
        blob = json.dumps(value, sort_keys=True, default=_jsonable).encode()
    except TypeError:
        blob = pickle.dumps(value)
    return hashlib.sha1(blob).hexdigest()[:16]


_CODE_FP: dict[tuple, str] = {}
_IMPORT = re.compile(r"^\s*(?:from\s+(modules(?:\.\w+)+)\s+import\s+([\w, ()]+)|import\s+(modules(?:\.\w+)+))", re.M)
_CLOSURE: dict[str, dict[str, str]] = {}


def _module_file(name: str) -> str | None:
    import importlib.util
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec and spec.origin and os.path.isfile(spec.origin) else None


def _imports(src: str) -> set[str]:
    """modules.* names a source imports (incl. 'from modules.X import submodule')."""
    out = set()
    for frm, names, plain in _IMPORT.findall(src):
        if plain:
            out.add(plain)
            continue
        out.add(frm)
        for n in re.split(r"[,\s()]+", names):
            if n and _module_file(f"{frm}.{n}"):
                out.add(f"{frm}.{n}")
    return out


def code_dependencies(cls: type) -> dict[str, str]:
    """Every modules.* file a stage depends on, transitively, with its content hash."""
    import inspect
    key = cls.__module__ + "." + cls.__qualname__
    if key not in _CLOSURE:
        try:
            todo = list(_imports(inspect.getsource(cls)))
        except (OSError, TypeError):
            todo = []
        files, seen = {}, set()
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            path = _module_file(name)
            if not path:
                continue
            files[os.path.relpath(path)] = _file_fingerprint(path)
            with open(path, encoding="utf-8", errors="replace") as f:
                todo.extend(_imports(f.read()) - seen)
        _CLOSURE[key] = dict(sorted(files.items()))
    return _CLOSURE[key]


def code_fingerprint(cls: type, with_modules: bool = True) -> str:
    """Hash of a stage's own source and (unless opted out) of every modules.* file it depends on,
    transitively, so editing the code behind a stage invalidates its cache without a version bump."""
    key = (cls, with_modules)
    if key not in _CODE_FP:
        import inspect
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            src = cls.__qualname__
        h = hashlib.sha1(src.encode())
        if with_modules:
            h.update(json.dumps(code_dependencies(cls)).encode())
        _CODE_FP[key] = h.hexdigest()[:16]
    return _CODE_FP[key]


def _jsonable(o):
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


# ----------------------------------------------------------------------------- stages

@dataclass
class Context:
    """Run-wide information every stage may use (not an artifact: not part of cache keys)."""
    cache_dir: str
    output_dir: str
    device: str = "cpu"
    basename: str = "song"
    verbose: bool = True
    # "lineage": cache key includes the ordered chain of upstream steps (default: order matters)
    # "content": key = this step + input contents only, so different orders that yield the same
    #            inputs share cache entries (opt-in for test routines)
    cache_by: str = "lineage"
    extra: dict = field(default_factory=dict)


class Stage:
    """Base class. Subclasses set ``name``, ``version``, ``inputs``/``outputs`` (slot -> default
    artifact name), ``Params`` (a dataclass) and implement ``run(ctx, **inputs) -> dict``.
    ``cache`` False for stages that are cheap or write final files."""

    name: str = ""
    version: str = "1"
    # runs a heavy model on ctx.device (GPU): the tuner never puts such a step into the cheap tail,
    # which runs in CPU worker processes (CPU and GPU results differ; CPU copies need lots of RAM)
    gpu_model: bool = False
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    optional_inputs: tuple[str, ...] = ()
    cache: bool = True
    # Code in the cache key: the stage's own source always; every modules.* file it depends on
    # (transitively) too, unless hash_modules is False. Expensive model stages (separation, ASR,
    # pitch trackers) opt out so an unrelated edit doesn't cost an hour of recomputation; when their
    # code changed since a cache entry was written, the run warns and names the files instead
    # (recompute with the step's `fresh` override, or bump `version` for a real change).
    hash_modules: bool = True

    @dataclass
    class Params:
        pass

    def __init__(self, params: dict | None = None, inputs: dict | None = None, outputs: dict | None = None,
                 id: str | None = None):
        self.id = id or self.name
        fields = {f.name for f in dataclasses.fields(self.Params)}
        unknown = set(params or {}) - fields
        if unknown:
            raise ValueError(f"{self.id}: unknown parameter(s) {sorted(unknown)}; known: {sorted(fields)}")
        missing = sorted(f.name for f in dataclasses.fields(self.Params)
                         if f.name not in (params or {}) and f.default is dataclasses.MISSING
                         and f.default_factory is dataclasses.MISSING)
        if missing:
            raise ValueError(f"{self.id}: no value for {missing}; every value lives in the preset "
                             f"(presets/best.yaml, params: {self.name}: ...)")
        self.params = self.Params(**(params or {}))
        # declared specs ("a|b" alternatives, "@slot" = same artifact as that input slot) ...
        self.input_specs = {**self.inputs, **(inputs or {})}
        self.output_specs = {**self.outputs, **(outputs or {})}
        # ... and the concrete artifact names, set by Pipeline.bind()
        self.input_names = {k: v.split("|")[0] for k, v in self.input_specs.items()}
        self.output_names = {k: v for k, v in self.output_specs.items() if not v.startswith("@")}

    def bind(self, produced: list[str]) -> None:
        """Pick concrete artifacts: for alternatives, the one produced most recently."""
        for slot, spec in self.input_specs.items():
            options = spec.split("|")
            latest = next((a for a in reversed(produced) if a in options), None)
            self.input_names[slot] = latest or options[0]
        self.output_names = {slot: (self.input_names[spec[1:]] if spec.startswith("@") else spec)
                             for slot, spec in self.output_specs.items()}

    def run(self, ctx: Context, **inputs) -> dict:  # pragma: no cover - interface
        raise NotImplementedError

    def model_fingerprints(self) -> dict[str, str]:
        """Parameters that point at existing files/folders (model weights) are hashed by content:
        swapping a checkpoint under the same name invalidates the cache."""
        out = {}
        for k, v in dataclasses.asdict(self.params).items():
            if isinstance(v, str) and os.path.isabs(os.path.expanduser(v)):
                path = os.path.expanduser(v)
                if os.path.isfile(path):
                    out[k] = fingerprint(path)
                elif os.path.isdir(path):
                    out[k] = _dir_fingerprint(path)
        return out

    def cache_key(self, store: dict, chain: list | None = None) -> str:
        payload = [self.name, self.version, dataclasses.asdict(self.params),
                   {slot: fingerprint(store[art]) for slot, art in sorted(self.input_names.items()) if art in store},
                   code_fingerprint(type(self), self.hash_modules), self.model_fingerprints()]
        if chain is not None:  # step ids are labels only; stage, version and params define the history
            payload.append([[c["stage"], c["version"], c["params"]] for c in chain])
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


REGISTRY: dict[str, type[Stage]] = {}


def stage(cls: type[Stage]) -> type[Stage]:
    """Class decorator: register a stage under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} has no name")
    REGISTRY[cls.name] = cls
    return cls


# ----------------------------------------------------------------------------- pipeline

@dataclass
class StepSpec:
    use: str
    id: str | None = None
    params: dict = field(default_factory=dict)     # per-step values (above the preset's params table)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    enabled: bool = True
    sources: dict = field(default_factory=dict)    # param -> where a per-step value came from


class Pipeline:
    def __init__(self, steps: list[StepSpec], name: str = "pipeline", params: dict | None = None,
                 param_sources: dict | None = None):
        """params: the preset's table {stage name: {param: value}}; per-step values override it."""
        self.name = name
        self.params = params or {}
        self.param_sources = param_sources or {}
        self.specs = [s for s in steps if s.enabled]
        self.stages = []
        for s in self.specs:
            if s.use not in REGISTRY:
                raise KeyError(f"unknown stage '{s.use}'; registered: {sorted(REGISTRY)}")
            merged = {**self.params.get(s.use, {}), **s.params}
            self.stages.append(REGISTRY[s.use](merged, s.inputs, s.outputs, id=s.id or s.use))

    def effective_config(self) -> list[dict]:
        """Every step with every parameter value and where it came from."""
        out = []
        for spec, st in zip(self.specs, self.stages):
            vals = dataclasses.asdict(st.params)
            src = {k: spec.sources.get(k) or self.param_sources.get(spec.use, {}).get(k, "?") for k in vals}
            out.append({"step": st.id, "stage": st.name, "params": vals, "sources": src})
        return out

    def show_config(self) -> str:
        lines = [f"preset: {self.name}"]
        for row in self.effective_config():
            lines.append(f"{row['step']}" + (f"  ({row['stage']})" if row["stage"] != row["step"] else ""))
            for k, v in row["params"].items():
                lines.append(f"    {k} = {v!r}    [{row['sources'][k]}]")
        return "\n".join(lines)

    def bind(self, initial) -> None:
        produced = list(initial)
        for st in self.stages:
            st.bind(produced)
            for art in st.output_names.values():
                if art in produced:
                    produced.remove(art)
                produced.append(art)

    def validate(self, initial) -> list[str]:
        """Return problems (empty list = runnable): inputs that no earlier stage provides."""
        self.bind(initial)
        have, problems = set(initial), []
        for st in self.stages:
            for slot, art in st.input_names.items():
                if art not in have and slot not in st.optional_inputs:
                    alts = st.input_specs[slot].replace("|", " or ")
                    problems.append(f"{st.id}: input '{slot}' needs {alts}, not provided before this step")
            have |= set(st.output_names.values())
        return problems

    def unused_outputs(self, initial) -> list[str]:
        """Steps whose output nothing reads afterwards (type-checks, but has no effect)."""
        self.bind(initial)
        notes = []
        for i, st in enumerate(self.stages):
            for art in st.output_names.values():
                if art.startswith("file.") or art.startswith("aux."):
                    continue
                later = self.stages[i + 1:]
                read = any(art in s.input_names.values() for s in later)
                overwritten_first = next((s for s in later if art in s.output_names.values()
                                          and art not in s.input_names.values()), None)
                if not read or (overwritten_first is not None and not any(
                        art in s.input_names.values() for s in later[:later.index(overwritten_first)])):
                    notes.append(f"{st.id}: nothing reads '{art}' afterwards")
        return notes

    def where(self, step_id: str, initial) -> list[int]:
        """Positions (index in the step list) where step_id could run with every input available."""
        idx = next((i for i, st in enumerate(self.stages) if st.id == step_id), None)
        if idx is None:
            raise ValueError(f"no step '{step_id}'; steps: {[st.id for st in self.stages]}")
        others = self.stages[:idx] + self.stages[idx + 1:]
        ok = []
        for pos in range(len(others) + 1):
            trial = Pipeline.__new__(Pipeline)
            trial.name, trial.stages = self.name, others[:pos] + [self.stages[idx]] + others[pos:]
            if not trial.validate(initial):
                ok.append(pos)
        self.bind(initial)
        return ok

    def describe_positions(self, step_id: str, initial) -> str:
        others = [st.id for st in self.stages if st.id != step_id]
        valid = set(self.where(step_id, initial))
        idx = next(i for i, st in enumerate(self.stages) if st.id == step_id)
        rest = self.stages[:idx] + self.stages[idx + 1:]
        lines = []
        for pos in range(len(others) + 1):
            mark = ""
            if pos in valid:
                trial = Pipeline.__new__(Pipeline)
                trial.name, trial.stages = self.name, rest[:pos] + [self.stages[idx]] + rest[pos:]
                dead = [n for n in trial.unused_outputs(initial) if n.startswith(step_id + ":")]
                mark = "  <- ok" + ("  (but nothing reads its output afterwards)" if dead else "")
            after = others[pos - 1] if pos else "(start)"
            lines.append(f"after {after}{mark}")
        self.bind(initial)
        return "\n".join(lines)

    def run(self, ctx: Context, store: dict) -> dict:
        # production order matters for "a|b" slots (most recent wins); it travels with the store,
        # so a pipeline continued on an earlier store (a tail run) binds like the full one
        order = store.setdefault("_produced", [k for k in store if not k.startswith("_")])
        problems = self.validate(list(order))
        if problems:
            raise ValueError("pipeline does not type-check:\n  " + "\n  ".join(problems))
        os.makedirs(ctx.cache_dir, exist_ok=True)
        if "audio_seconds" not in ctx.extra and isinstance(store.get("audio.mix"), str):
            from pipeline.runtimes import audio_seconds
            ctx.extra["audio_seconds"] = audio_seconds(store["audio.mix"])
        lineage: dict[str, list] = store.setdefault("_lineage", {})
        for st in self.stages:
            args = {slot: store[art] for slot, art in st.input_names.items() if art in store}
            # provenance: ordered, de-duplicated chain of steps behind every input, then this step
            chain, seen = [], set()
            for art in st.input_names.values():
                for step in lineage.get(art, []):
                    tag = json.dumps(step, sort_keys=True, default=str)
                    if tag not in seen:
                        seen.add(tag)
                        chain.append(step)
            chain.append({"step": st.id, "stage": st.name, "version": st.version,
                          "params": dataclasses.asdict(st.params)})
            use_cache = st.cache and not ctx.extra.get("no_cache")
            path = os.path.join(ctx.cache_dir, f"stage_{st.name}_{st.cache_key(store, chain if ctx.cache_by == 'lineage' else None)}.pkl") if use_cache else None
            if path:
                _record_lineage(path, chain, self.name)
            from pipeline.log import get as get_logger, stage_scope
            key = os.path.basename(path)[:-4] if path else None
            fresh = st.id in ctx.extra.get("fresh", ())
            result = None
            if path and os.path.exists(path) and not fresh:
                with open(path, "rb") as f:
                    result = pickle.load(f)
                missing = _missing_files(result)
                if missing:   # a file the cached result points to was cleaned up: recompute
                    get_logger(f"stage.{st.id}").info(f"{st.id}: cached result points to missing {missing[0]}; "
                                                      f"recomputing", extra={"stage": st.id, "event": "cache_stale"})
                    result = None
            if result is not None:
                get_logger(f"stage.{st.id}").info(f"{st.id}: cached",
                                                  extra={"stage": st.id, "event": "cached", "cache_key": key})
                if not st.hash_modules:
                    changed = _code_changed(path, code_dependencies(type(st)))
                    if changed:
                        get_logger(f"stage.{st.id}").warning(
                            f"{st.id}: cached result was made with different code in {', '.join(changed)}; "
                            f"reusing it. Recompute with --fresh {st.id} (or bump the stage's version).",
                            extra={"stage": st.id, "event": "stale_code"})
            else:
                from pipeline import runtimes
                from pipeline.log import peak_rss_mb
                t0 = time.time()
                with stage_scope(st.id, key, ctx.verbose):
                    result = st.run(ctx, **copy.deepcopy(args)) or {}
                meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 2),
                        "peak_rss_mb": peak_rss_mb(), "device": ctx.device, "cpus": os.cpu_count(),
                        "audio_seconds": ctx.extra.get("audio_seconds")}
                runtimes.record({"stage": st.name, "step": st.id, "params": dataclasses.asdict(st.params),
                                 "code": ctx.extra.get("code_version"), **meta})
                if path:
                    _atomic_pickle(path, result)
                    _atomic_json(path[:-4] + ".meta.json", {**meta, "code": code_dependencies(type(st))})
            missing = set(st.outputs) - set(result)
            if missing:
                raise RuntimeError(f"{st.id} did not return {sorted(missing)}")
            from pipeline import contracts
            for slot, value in result.items():
                if slot in st.output_names:
                    problems = contracts.check(st.output_names[slot], value)
                    if problems:
                        msg = f"{st.id} -> {st.output_names[slot]}: " + "; ".join(problems[:5])
                        if contracts.strict_mode(ctx):
                            raise contracts.ContractError(msg)
                        get_logger(f"stage.{st.id}").warning(msg, extra={"stage": st.id, "event": "contract"})
            for slot, value in result.items():
                if slot in st.output_names:
                    art = st.output_names[slot]
                    store[art] = value
                    lineage[art] = chain
                    if art in order:
                        order.remove(art)
                    order.append(art)
        return store


def _missing_files(result) -> list[str]:
    """Absolute file paths in a stage result (top-level string values) that no longer exist."""
    if not isinstance(result, dict):
        return []
    return [v for v in result.values() if isinstance(v, str) and os.path.isabs(v)
            and os.path.splitext(v)[1] and not os.path.exists(v)]


def _atomic_pickle(path: str, obj) -> None:
    """Write to a temp file and rename: a killed or parallel run never leaves a half-written cache."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _atomic_json(path: str, obj) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _code_changed(pkl_path: str, current: dict[str, str]) -> list[str]:
    """Files whose content differs from when the cache entry was written (unknown = no warning)."""
    side = pkl_path[:-4] + ".meta.json"
    try:
        with open(side, encoding="utf-8") as f:
            then = json.load(f).get("code", {})
    except (OSError, ValueError, AttributeError):
        return []
    return sorted(k for k in set(then) | set(current) if then.get(k) != current.get(k))


def _record_lineage(pkl_path: str, chain: list, preset: str) -> None:
    """Sidecar <cache>.lineage.json: every step order + settings that produced this cache entry.
    The cache key stays content-based (same inputs + params = reuse, whatever the order);
    the sidecar keeps all distinct histories that hit it, for debugging and stale-cache hunts."""
    side = pkl_path[:-4] + ".lineage.json"
    entries = []
    if os.path.exists(side):
        try:
            with open(side, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, ValueError):
            entries = []
    order = [s["step"] for s in chain]
    if any(e.get("chain") == chain for e in entries):
        return
    entries.append({"preset": preset, "order": order, "chain": chain,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S")})
    tmp = f"{side}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=1, default=str)
    os.replace(tmp, side)


# ----------------------------------------------------------------------------- presets

def resolve_preset(path_or_dict, _seen: tuple = ()) -> dict:
    """Flatten preset inheritance into {name, params, sources, steps, overrides}.

    A preset may name a ``base`` (another preset file, relative to its own folder); it inherits the
    base's params table, its steps (unless it lists its own) and the base's overrides (dropped when
    it lists its own steps). Its own ``params`` are merged per stage and its ``overrides`` appended."""
    import yaml
    if isinstance(path_or_dict, dict):
        data, here, label = copy.deepcopy(path_or_dict), os.getcwd(), path_or_dict.get("name", "preset")
    else:
        with open(path_or_dict, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        here, label = os.path.dirname(os.path.abspath(path_or_dict)), data.get("name") or \
            os.path.splitext(os.path.basename(path_or_dict))[0]
    if data.get("base"):
        base_path = data["base"] if os.path.isabs(data["base"]) else os.path.join(here, data["base"])
        if not base_path.endswith(".yaml"):
            base_path += ".yaml"
        if base_path in _seen:
            raise ValueError(f"preset inheritance loops through {base_path}")
        r = resolve_preset(base_path, _seen + (base_path,))
    else:
        r = {"params": {}, "sources": {}, "steps": [], "overrides": []}
    for stage_name, vals in (data.get("params") or {}).items():
        r["params"].setdefault(stage_name, {}).update(vals or {})
        r["sources"].setdefault(stage_name, {}).update({k: label for k in (vals or {})})
    if data.get("steps"):
        r["steps"], r["overrides"] = data["steps"], []
    r["overrides"] = r["overrides"] + [(o, label) for o in data.get("overrides") or []]
    r["name"] = label
    return r


def load_preset(path_or_dict, overrides: list[str] | None = None) -> Pipeline:
    """Preset = {name, base?, params: {stage: {param: value}}, steps: [{use, id?, params?, inputs?,
    outputs?, enabled?}], overrides?: [...]}. Every value of every stage lives in a params table
    (presets/best.yaml); stage classes only declare names and types.
    Overrides (preset ``overrides`` first, then the call's), applied in order:
      step.param=value        change a parameter of one step (YAML-parsed value)
      step.use=other_stage    swap the stage (its values come from the params table)
      step.after=other        move a step (also step.before=other)
      -step                   drop a step
      +stage[#id][@after=step|@before=step]   insert a step (appended without a position)"""
    import yaml

    r = resolve_preset(path_or_dict)
    steps = [StepSpec(**{k: v for k, v in s.items()}) for s in copy.deepcopy(r["steps"])]
    for s in steps:
        s.sources = {k: r["name"] + " (step)" for k in s.params}
    by_id = {(s.id or s.use): s for s in steps}

    def place(spec, where, ref, ov):
        if ref not in by_id:
            raise ValueError(f"override '{ov}': no step '{ref}' in preset; steps: {sorted(by_id)}")
        if spec in steps:
            steps.remove(spec)
        i = steps.index(by_id[ref])
        steps.insert(i + 1 if where == "after" else i, spec)

    for ov, origin in r["overrides"] + [(o, "override") for o in overrides or []]:
        if ov.startswith("+"):  # insert: +stage[#id][@after=step|@before=step]
            m = re.match(r"\+([\w.]+)(?:#([\w.]+))?(?:@(after|before)=([\w.]+))?$", ov)
            if not m:
                raise ValueError(f"override '{ov}': expected +stage[#id][@after=step|@before=step]")
            spec = StepSpec(use=m.group(1), id=m.group(2))
            sid = spec.id or spec.use
            if sid in by_id:
                raise ValueError(f"override '{ov}': step '{sid}' exists; give the new one an id with #id")
            by_id[sid] = spec
            if m.group(3):
                place(spec, m.group(3), m.group(4), ov)
            else:
                steps.append(spec)
            continue
        if ov.startswith("-"):
            if ov[1:] not in by_id:
                raise ValueError(f"override '{ov}': no step '{ov[1:]}' in preset; steps: {sorted(by_id)}")
            by_id[ov[1:]].enabled = False
            continue
        if "=" not in ov or "." not in ov.split("=", 1)[0]:
            raise ValueError(f"override '{ov}': expected step.param=value or -step")
        target, value = ov.split("=", 1)
        step_id, key = target.rsplit(".", 1)
        if step_id not in by_id:
            raise ValueError(f"override '{ov}': no step '{step_id}' in preset; steps: {sorted(by_id)}")
        parsed = yaml.safe_load(value)
        if key in ("after", "before"):  # move the step
            place(by_id[step_id], key, str(parsed), ov)
        elif key == "enabled":
            by_id[step_id].enabled = bool(parsed)
        elif key == "use":  # swap the stage; its values come from the params table
            by_id[step_id].use = parsed
            by_id[step_id].params, by_id[step_id].sources = {}, {}
        else:
            stage_cls = REGISTRY.get(by_id[step_id].use)
            if stage_cls and key not in {f.name for f in dataclasses.fields(stage_cls.Params)}:
                raise ValueError(f"override '{ov}': stage {by_id[step_id].use} has no parameter '{key}'")
            by_id[step_id].params[key] = parsed
            by_id[step_id].sources[key] = origin
    return Pipeline(steps, r["name"], r["params"], r["sources"])


def describe(params: dict | None = None) -> str:
    """Registry as text: stage name, inputs -> outputs, parameters (values from a params table)."""
    lines = []
    for name in sorted(REGISTRY):
        if name.startswith("test."):
            continue
        cls = REGISTRY[name]
        vals = (params or {}).get(name, {})
        ps = ", ".join(f"{f.name}={vals[f.name]!r}" if f.name in vals else f.name
                       for f in dataclasses.fields(cls.Params))
        lines.append(f"{name}: {dict(cls.inputs)} -> {dict(cls.outputs)}  [{ps}]")
    return "\n".join(lines)
