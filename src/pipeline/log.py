"""One logging setup for the whole program, third-party output included.

Sources and where they go:
- our modules: ``logging.getLogger("ultrasinger.<area>")``
- Python warnings (``warnings.warn`` in libraries): captured as ``py.warnings`` records
- other libraries' loggers (lightning, whisperx, pyannote, ...): the root logger
- anything printed while a stage runs, by Python or C code or subprocesses (progress bars,
  ffmpeg, onnxruntime): captured at the file-descriptor level and logged per line (DEBUG)

Console (stdout): our INFO messages (WARNING with --quiet), each distinct third-party warning
once per run in one line, and errors. ``debug`` mode shows everything live instead.
Per run: ``run_log.jsonl`` with every record (third-party included), plus structured stage
events (``stage``, ``event`` = start|cached|done|failed|contract, ``seconds``, ``peak_rss_mb``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import tempfile
import time

ROOT = "ultrasinger"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_state = {"configured": False, "verbose": True, "debug": False, "console": None}


def get(name: str) -> logging.Logger:
    return logging.getLogger(name if name.startswith(ROOT) else f"{ROOT}.{name}")


def _ours(record) -> bool:
    return record.name == ROOT or record.name.startswith(ROOT + ".")


_WARN = re.compile(r"^(?P<path>.+?):\d+: (?P<cat>\w+): ?(?P<msg>.*)", re.S)


def _short(msg: str) -> str:
    """One line for the console: 'UserWarning (torchaudio): first line of the message' for Python
    warnings, the first meaningful line for anything else."""
    msg = _ANSI.sub("", msg)
    m = _WARN.match(msg)
    origin = ""
    if m:
        path = m.group("path").replace("\\", "/")
        pkg = path.split("site-packages/", 1)[1].split("/", 1)[0] if "site-packages/" in path \
            else os.path.splitext(os.path.basename(path))[0]
        origin, msg = f"{m.group('cat')} ({pkg}): ", m.group("msg")
    lines = [ln.strip() for ln in msg.splitlines() if ln.strip()]
    first = next((ln for ln in lines if not ln.startswith(("warnings.warn", "File ", "s = ", "Traceback"))),
                 lines[0] if lines else "")
    return (origin + first)[:180]


class _ConsoleFilter(logging.Filter):
    """Our records by level; third-party records: errors always, warnings once per distinct text.
    Captured stage output (ultrasinger.stage.*.output, DEBUG) only shows in debug mode."""

    def __init__(self):
        super().__init__()
        self.seen = set()

    def filter(self, record):
        if _state["debug"]:
            return True
        if _ours(record):
            return record.levelno >= (logging.INFO if _state["verbose"] else logging.WARNING)
        if record.levelno >= logging.ERROR:
            return True
        if record.levelno >= logging.WARNING:
            key = _short(record.getMessage())
            if key in self.seen:
                return False
            self.seen.add(key)
            return True
        return False


class _ConsoleFormatter(logging.Formatter):
    def format(self, record):
        from modules.console_colors import ULTRASINGER_HEAD
        msg = record.getMessage() if (_ours(record) or _state["debug"]) else _short(record.getMessage())
        if not _ours(record) and record.name != "py.warnings":
            msg = f"[{record.name}] {msg}"
        elif record.levelno >= logging.WARNING:
            msg = f"{record.levelname.lower()}: {msg}"
        return f"{ULTRASINGER_HEAD} {msg}"


class _JsonFormatter(logging.Formatter):
    FIELDS = ("stage", "event", "seconds", "peak_rss_mb", "cache_key")

    def format(self, record):
        out = {"t": round(record.created, 3), "level": record.levelname, "logger": record.name,
               "msg": _ANSI.sub("", record.getMessage())}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        for k in self.FIELDS:
            if hasattr(record, k):
                out[k] = getattr(record, k)
        return json.dumps(out, ensure_ascii=False)


def setup(verbose: bool = True, debug: bool = False) -> None:
    """Console handler on the root logger (idempotent). It writes to a duplicate of the original
    stdout, so it stays visible while stages capture file descriptors 1 and 2."""
    _state["verbose"], _state["debug"] = verbose, debug
    if _state["configured"]:
        return
    root = logging.getLogger()
    stream = os.fdopen(os.dup(sys.__stdout__.fileno()), "w", buffering=1, encoding="utf-8", errors="replace")
    h = logging.StreamHandler(stream)
    h.set_name("ultrasinger-console")
    h.addFilter(_ConsoleFilter())
    h.setFormatter(_ConsoleFormatter())
    root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger(ROOT).setLevel(logging.DEBUG)
    logging.getLogger(ROOT).propagate = True
    logging.captureWarnings(True)  # Python's default filters stay (ResourceWarning etc. ignored)
    _state["configured"], _state["console"] = True, stream


@contextlib.contextmanager
def run_logging(path: str, verbose: bool = True, debug: bool = False):
    """Write every record of this run (ours and third-party) to a JSONL file."""
    setup(verbose, debug)
    for h in logging.getLogger().handlers:
        for f in h.filters:
            if isinstance(f, _ConsoleFilter):
                f.seen.clear()
    open(path, "w").close()
    # mode "a": some libraries call logging.config.dictConfig, which closes every handler;
    # an append-mode FileHandler reopens itself on the next record, a "w" one stays silent
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
    root = logging.getLogger()
    root.addHandler(fh)
    try:
        yield
    finally:
        root.removeHandler(fh)
        fh.close()


def peak_rss_mb() -> float | None:
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024 if sys.platform != "darwin" else kb / 1024 / 1024, 1)
    except Exception:
        return None


@contextlib.contextmanager
def _capture_fds(logger: logging.Logger):
    """Redirect fd 1 and 2 (Python, C libraries, subprocesses) into a temp file; log its lines."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 1)
    os.dup2(tmp.fileno(), 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        tmp.seek(0)
        text = tmp.read().decode("utf-8", errors="replace")
        tmp.close()
        for line in text.split("\n"):
            line = _ANSI.sub("", line.split("\r")[-1]).strip()  # progress bars: last state only
            if line:
                logger.debug(line)


@contextlib.contextmanager
def stage_scope(stage_id: str, cache_key: str | None, verbose: bool):
    """Log start/done/failed with duration and memory; route everything printed into the log."""
    logger = get(f"stage.{stage_id}")
    t0 = time.time()
    logger.info(f"{stage_id} ...", extra={"stage": stage_id, "event": "start", "cache_key": cache_key})
    capture = contextlib.nullcontext() if _state["debug"] else _capture_fds(get(f"stage.{stage_id}.output"))
    try:
        with capture:
            yield logger
    except Exception:
        logger.exception(f"{stage_id} failed", extra={"stage": stage_id, "event": "failed",
                                                      "seconds": round(time.time() - t0, 2)})
        raise
    logger.info(f"{stage_id}: {time.time() - t0:.1f}s",
                extra={"stage": stage_id, "event": "done", "seconds": round(time.time() - t0, 2),
                       "peak_rss_mb": peak_rss_mb(), "cache_key": cache_key})
