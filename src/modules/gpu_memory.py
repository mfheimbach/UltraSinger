"""Shared GPU memory housekeeping. Modules that keep models loaded register a release function;
a module that runs out of GPU memory calls release_all() and retries (no module knows the others)."""

import gc

_RELEASERS: list = []


def register(release) -> None:
    if release not in _RELEASERS:
        _RELEASERS.append(release)


def free_torch_cache() -> None:
    """Hand PyTorch's cached (unused) GPU blocks back, e.g. before onnxruntime allocates."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def release_all() -> None:
    for release in _RELEASERS:
        release()
    gc.collect()
    free_torch_cache()
