"""ONNX Runtime sessions on the requested device, loudly when CUDA silently falls back to CPU.

onnxruntime-gpu needs the CUDA major version it was built for (<= 1.26: CUDA 12, >= 1.27: CUDA 13).
PyTorch (cu12x) ships the CUDA 12 libraries as pip packages; preload_dlls() loads them so the CUDA
provider finds them. If CUDA was asked for but the session runs on CPU, GAME/RMVPE get ~5-10x
slower without any error from onnxruntime, hence the warning.
"""

import logging

log = logging.getLogger("ultrasinger.onnx")
_PRELOADED = False


def session(model, device: str, options=None):
    """model: path or serialized bytes."""
    global _PRELOADED
    import onnxruntime as ort
    ort.set_default_logger_severity(3)   # onnxruntime's own C++ warnings (Memcpy nodes): errors only
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        # grow the GPU arena only by what is requested (default doubles it); with Whisper resident
        # on an 8 GB card the doubling ran GAME out of memory
        providers = [("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                     "CPUExecutionProvider"]
        if not _PRELOADED and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as e:  # noqa: BLE001  (reported below via the provider check)
                log.debug(f"onnxruntime preload_dlls: {e}")
            _PRELOADED = True
    s = ort.InferenceSession(model, options, providers=providers) if options else \
        ort.InferenceSession(model, providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in s.get_providers():
        log.warning(f"FALLBACK: onnxruntime {ort.__version__} could not use CUDA, running on CPU (much slower). "
                    "onnxruntime-gpu <= 1.26 is built for CUDA 12 (matches PyTorch cu12x), >= 1.27 for CUDA 13: "
                    "uv pip install 'onnxruntime-gpu==1.26.*'")
    return s
