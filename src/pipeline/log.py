"""One logging setup for the whole program.

- Every module logs through ``logging.getLogger("ultrasinger.<area>")``.
- Console: ``[UltraSinger] message`` (INFO when verbose, WARNING otherwise).
- Per run: ``run_log.jsonl`` next to the chart, one JSON object per record, with structured
  fields for stage events (``stage``, ``event`` = start|cached|done|failed, ``seconds``,
  ``peak_rss_mb``). Third-party output printed while a stage runs (progress bars, library
  prints) is captured into the same log at DEBUG level, so a run's log is complete.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import sys
import time

ROOT = "ultrasinger"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_configured = False


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"{ROOT}.{name}" if not name.startswith(ROOT) else name)


class _ConsoleFormatter(logging.Formatter):
    def format(self, record):
        from modules.console_colors import ULTRASINGER_HEAD
        msg = record.getMessage()
        if record.levelno >= logging.WARNING:
            msg = f"{record.levelname.lower()}: {msg}"
        return f"{ULTRASINGER_HEAD} {msg}"


class _JsonFormatter(logging.Formatter):
    FIELDS = ("stage", "event", "seconds", "peak_rss_mb", "cache_key")

    def format(self, record):
        out = {"t": round(record.created, 3), "level": record.levelname, "logger": record.name,
               "msg": _ANSI.sub("", record.getMessage())}
        for k in self.FIELDS:
            if hasattr(record, k):
                out[k] = getattr(record, k)
        return json.dumps(out, ensure_ascii=False)


def setup(verbose: bool = True) -> None:
    """Console handler on the 'ultrasinger' logger (idempotent)."""
    global _configured
    root = logging.getLogger(ROOT)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    if not _configured:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_ConsoleFormatter())
        h.set_name("ultrasinger-console")
        root.addHandler(h)
        _configured = True
    for h in root.handlers:
        if h.get_name() == "ultrasinger-console":
            h.setLevel(logging.INFO if verbose else logging.WARNING)


@contextlib.contextmanager
def run_logging(path: str, verbose: bool = True):
    """Write every record of this run to a JSONL file."""
    setup(verbose)
    root = logging.getLogger(ROOT)
    open(path, "w").close()
    # mode "a": some libraries call logging.config.dictConfig, which closes every handler;
    # an append-mode FileHandler reopens itself on the next record, a "w" one stays silent
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_JsonFormatter())
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


class _LineWriter(io.TextIOBase):
    """File-like object that turns printed lines into log records."""

    def __init__(self, logger: logging.Logger, level: int, echo):
        self.logger, self.level, self.echo, self.buf = logger, level, echo, ""

    def write(self, s):
        if self.echo is not None:
            self.echo.write(s)
        self.buf += s
        while "\n" in self.buf or "\r" in self.buf:
            cut = min(i for i in (self.buf.find("\n"), self.buf.find("\r")) if i >= 0)
            line, self.buf = self.buf[:cut], self.buf[cut + 1:]
            line = _ANSI.sub("", line).strip()
            if line:
                self.logger.log(self.level, line)
        return len(s)

    def flush(self):
        if self.echo is not None:
            self.echo.flush()


@contextlib.contextmanager
def stage_scope(stage_id: str, cache_key: str | None, verbose: bool):
    """Log start/done/failed with duration and memory; capture stray prints of the stage."""
    logger = get(f"stage.{stage_id}")
    t0 = time.time()
    logger.debug("start", extra={"stage": stage_id, "event": "start", "cache_key": cache_key})
    out = _LineWriter(logger, logging.DEBUG, sys.stdout if verbose else None)
    try:
        with contextlib.redirect_stdout(out):
            yield logger
    except Exception:
        logger.exception("failed", extra={"stage": stage_id, "event": "failed",
                                          "seconds": round(time.time() - t0, 2)})
        raise
    logger.info(f"{stage_id}: {time.time() - t0:.1f}s",
                extra={"stage": stage_id, "event": "done", "seconds": round(time.time() - t0, 2),
                       "peak_rss_mb": peak_rss_mb(), "cache_key": cache_key})
