"""Transcription with Qwen3-ASR + Qwen3-ForcedAligner (word timestamps).

Optional backend: needs `pip install qwen-asr`. Selected with
stage transcribe.qwen3 (e.g. --set in a preset).
"""


import logging
import langcodes
import torch

from modules.console_colors import blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult

log = logging.getLogger("ultrasinger.qwen3asr")


def _language_name(code: str | None) -> str | None:
    """'de' -> 'German' (Qwen expects English language names)."""
    if not code:
        return None
    return langcodes.Language.get(code).display_name("en")


def _language_code(name: str) -> str:
    """'German' -> 'de'; for mixed results like 'German,English' take the first."""
    first = (name or "").split(",")[0].strip()
    if not first:
        log.warning("FALLBACK: Qwen3 detected no language; assuming 'en'")
        return "en"
    try:
        return langcodes.find(first).language
    except LookupError:
        log.warning(f"FALLBACK: unknown language name {name!r} from Qwen3; assuming 'en'")
        return "en"


MAX_NEW_TOKENS = 4096  # generation cap of the ASR model (resource limit, not a tuning knob)
SR = 16000             # model input rate


def transcribe_with_qwen3(
    audio_path: str,
    model_name: str,
    aligner_name: str,
    device: str,
    language: str | None,
    batch_size: int,
    max_align_seconds: float,
) -> TranscriptionResult:
    import gc
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    log.info(f"Loading {blue_highlighted('Qwen3-ASR')} {blue_highlighted(model_name)} "
          f"with aligner {blue_highlighted(aligner_name)} and {red_highlighted(device)} as worker")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"

    # ASR and aligner are loaded one after the other to halve peak memory
    model = Qwen3ASRModel.from_pretrained(model_name, dtype=dtype, device_map=device_map,
                                          max_inference_batch_size=batch_size, max_new_tokens=MAX_NEW_TOKENS)
    import librosa
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    n_chunks = int((len(y) / sr) // max_align_seconds) + 1
    bounds = [(k * len(y) // n_chunks, (k + 1) * len(y) // n_chunks) for k in range(n_chunks)]
    log.info(f"Transcribing {audio_path} in {n_chunks} chunk(s)")
    chunk_results = [model.transcribe(audio=(y[a:b], sr), language=_language_name(language))[0] for a, b in bounds]
    languages = [r.language for r in chunk_results if r.language]
    result = type("R", (), {"language": max(set(languages), key=languages.count) if languages else ""})
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    detected = _language_code(result.language)
    log.info(f"Qwen3 detected language: {blue_highlighted(result.language or '?')}")

    aligner = Qwen3ForcedAligner.from_pretrained(aligner_name, dtype=dtype, device_map=device_map)
    aligned = []
    for (a, b), r in zip(bounds, chunk_results):
        if not r.text.strip():
            continue
        offset = a / sr
        for it in aligner.align(audio=(y[a:b], sr), text=r.text, language=_language_name(detected) or "English")[0]:
            aligned.append(type("Item", (), {"text": it.text, "start_time": it.start_time + offset,
                                             "end_time": it.end_time + offset}))

    transcribed = []
    for item in aligned:
        word = str(item.text).strip()
        if not word:
            continue
        transcribed.append(TranscribedData(
            confidence=1.0,  # the aligner gives no per-word score
            word=word + " ",
            start=float(item.start_time),
            end=float(item.end_time),
        ))

    del aligner
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return TranscriptionResult(transcribed, detected)
