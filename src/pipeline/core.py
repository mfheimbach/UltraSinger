"""Modular pipeline core: typed artifacts, registered stages, cached runs, presets.

A run keeps a store of named artifacts (``audio.vocals``, ``words``, ``notes`` ...). Each stage
declares the artifacts it reads and writes; a pipeline is an ordered list of stages. Any order
works as long as every input exists before the stage that reads it (checked up front), so stages
with the same signature (e.g. text in / text out) can be swapped or moved.

Caching: a stage's result is stored under a key made from its name, version, parameters and the
fingerprints of its inputs. Changing anything upstream invalidates exactly what depends on it.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from typing import Any

# ----------------------------------------------------------------------------- artifacts

# Artifact kinds; an artifact's kind is the part of its name before the first dot
# (audio.vocals -> audio, pitch.rmvpe -> pitch). Bare names are their own kind.
KINDS = {
    "audio": "path to a wav file",
    "words": "list[TranscribedData] (word level, with timing)",
    "syllables": "list[TranscribedData] (syllable level, with timing)",
    "pitch": "PitchedData (frame-level f0 + confidence)",
    "notes": "list[MidiSegment] (line_start flags mark lines)",
    "tempo": "float bpm",
    "key": "tuple[str, str] (key, mode)",
    "language": "str language code",
    "lyrics": "list[dict] candidate lyrics (plain / synced)",
    "media": "MediaInfo",
    "file": "path to an output file",
    "aux": "stage-specific extra data",
}


def kind_of(name: str) -> str:
    return name.split(".", 1)[0]


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


_CODE_FP: dict[type, str] = {}
_IMPORT = re.compile(r"^\s*(?:from\s+(modules(?:\.\w+)+)\s+import|import\s+(modules(?:\.\w+)+))", re.M)


def code_fingerprint(cls: type, with_modules: bool = True) -> str:
    """Hash of a stage's own source and (unless opted out) of the modules.* files it imports, so
    editing the code behind a stage invalidates its cache without a manual version bump."""
    key = (cls, with_modules)
    if key not in _CODE_FP:
        import importlib.util
        import inspect
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            src = cls.__qualname__
        h = hashlib.sha1(src.encode())
        if with_modules:
            for a, b in sorted(set(_IMPORT.findall(src))):
                spec = importlib.util.find_spec(a or b)
                if spec and spec.origin and os.path.isfile(spec.origin):
                    h.update(_file_fingerprint(spec.origin).encode())
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
    inputs: dict[str, str] = {}
    outputs: dict[str, str] = {}
    optional_inputs: tuple[str, ...] = ()
    cache: bool = True
    # Code in the cache key: the stage's own source always; the modules.* files it imports too,
    # unless hash_modules is False (expensive model stages: separation, ASR, pitch trackers keep
    # their caches across unrelated edits and rely on `version` bumps instead).
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
        self.params = self.Params(**(params or {}))
        self.input_names = {**self.inputs, **(inputs or {})}
        self.output_names = {**self.outputs, **(outputs or {})}

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
    params: dict = field(default_factory=dict)
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    enabled: bool = True


class Pipeline:
    def __init__(self, steps: list[StepSpec], name: str = "pipeline"):
        self.name = name
        self.specs = [s for s in steps if s.enabled]
        self.stages = []
        for s in self.specs:
            if s.use not in REGISTRY:
                raise KeyError(f"unknown stage '{s.use}'; registered: {sorted(REGISTRY)}")
            self.stages.append(REGISTRY[s.use](s.params, s.inputs, s.outputs, id=s.id or s.use))

    def validate(self, initial: set[str]) -> list[str]:
        """Return problems (empty list = runnable): inputs that no earlier stage provides."""
        have, problems = set(initial), []
        for st in self.stages:
            for slot, art in st.input_names.items():
                if art not in have and slot not in st.optional_inputs:
                    problems.append(f"{st.id}: input '{slot}' needs '{art}', not provided before this step")
            have |= set(st.output_names.values())
        return problems

    def run(self, ctx: Context, store: dict) -> dict:
        problems = self.validate(set(store))
        if problems:
            raise ValueError("pipeline does not type-check:\n  " + "\n  ".join(problems))
        os.makedirs(ctx.cache_dir, exist_ok=True)
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
            path = os.path.join(ctx.cache_dir, f"stage_{st.name}_{st.cache_key(store, chain if ctx.cache_by == 'lineage' else None)}.pkl") if st.cache else None
            if path:
                _record_lineage(path, chain, self.name)
            from pipeline.log import get as get_logger, stage_scope
            key = os.path.basename(path)[:-4] if path else None
            if path and os.path.exists(path):
                with open(path, "rb") as f:
                    result = pickle.load(f)
                get_logger(f"stage.{st.id}").info(f"{st.id}: cached",
                                                  extra={"stage": st.id, "event": "cached", "cache_key": key})
            else:
                with stage_scope(st.id, key, ctx.verbose):
                    result = st.run(ctx, **copy.deepcopy(args)) or {}
                if path:
                    _atomic_pickle(path, result)
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
                    store[st.output_names[slot]] = value
                    lineage[st.output_names[slot]] = chain
        return store


def _atomic_pickle(path: str, obj) -> None:
    """Write to a temp file and rename: a killed or parallel run never leaves a half-written cache."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


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
                    "created": __import__("time").strftime("%Y-%m-%d %H:%M:%S")})
    tmp = f"{side}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=1, default=str)
    os.replace(tmp, side)


# ----------------------------------------------------------------------------- presets

def load_preset(path_or_dict, overrides: list[str] | None = None) -> Pipeline:
    """Preset = {name, steps: [{use, id?, params?, inputs?, outputs?, enabled?}]}.
    Overrides: 'step_id.param=value' (YAML-parsed value), 'step_id.enabled=false',
    or '-step_id' to drop a step."""
    import yaml

    if isinstance(path_or_dict, dict):
        preset = copy.deepcopy(path_or_dict)
    else:
        with open(path_or_dict, encoding="utf-8") as f:
            preset = yaml.safe_load(f)
    steps = [StepSpec(**s) for s in preset["steps"]]
    by_id = {(s.id or s.use): s for s in steps}
    for ov in overrides or []:
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
        if key == "enabled":
            by_id[step_id].enabled = bool(parsed)
        elif key == "use":  # swap the stage; its parameters start from the new stage's defaults
            by_id[step_id].use = parsed
            by_id[step_id].params = {}
        else:
            by_id[step_id].params[key] = parsed
    return Pipeline(steps, preset.get("name", "pipeline"))


def describe() -> str:
    """Registry as text: stage name, inputs -> outputs, parameters."""
    lines = []
    for name in sorted(REGISTRY):
        cls = REGISTRY[name]
        params = ", ".join(f"{f.name}={f.default!r}" for f in dataclasses.fields(cls.Params)
                           if f.default is not dataclasses.MISSING)
        lines.append(f"{name}: {dict(cls.inputs)} -> {dict(cls.outputs)}  [{params}]")
    return "\n".join(lines)
