"""Transcription with Qwen3-ASR + Qwen3-ForcedAligner (word timestamps).

Optional backend: needs `pip install qwen-asr`. Selected with
`--transcriber qwen3` / Settings.transcriber = "qwen3".
"""

import langcodes
import torch

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult


def _language_name(code: str | None) -> str | None:
    """'de' -> 'German' (Qwen expects English language names)."""
    if not code:
        return None
    return langcodes.Language.get(code).display_name("en")


def _language_code(name: str) -> str:
    """'German' -> 'de'; for mixed results like 'German,English' take the first."""
    first = (name or "").split(",")[0].strip()
    if not first:
        return "en"
    try:
        return langcodes.find(first).language
    except LookupError:
        return "en"


def is_degenerate_alignment(words, max_zero_share: float = 0.3) -> bool:
    """Qwen3-ForcedAligner sometimes collapses (all or many words get zero length)."""
    if not words:
        return True
    zero = sum(1 for w in words if (w.end - w.start) < 0.02)
    return zero / len(words) > max_zero_share


MAX_ALIGN_SECONDS = 170.0  # Qwen3-ForcedAligner accepts at most 180 s


def _align_chunked(aligner, audio_path: str, text: str, language: str, max_seconds: float = MAX_ALIGN_SECONDS):
    """Qwen3-ForcedAligner accepts at most 180 s. Longer audio is split into equal
    chunks and the words are split proportionally to chunk duration (rough: assumes
    even word density; chunking at silences is a planned improvement)."""
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr
    words = text.split()
    if duration <= max_seconds or len(words) < 2:
        return list(aligner.align(audio=(y, sr), text=text, language=language)[0])
    n = int(duration // max_seconds) + 1
    items = []
    for k in range(n):
        a, b = k * duration / n, (k + 1) * duration / n
        w = words[int(len(words) * k / n): int(len(words) * (k + 1) / n)]
        if not w:
            continue
        seg = y[int(a * sr): int(b * sr)]
        for it in aligner.align(audio=(seg, sr), text=" ".join(w), language=language)[0]:
            items.append(type("Item", (), {"text": it.text, "start_time": it.start_time + a, "end_time": it.end_time + a}))
    return items


def transcribe_with_qwen3(
    audio_path: str,
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    device: str = "cpu",
    language: str = None,
    batch_size: int = 8,
) -> TranscriptionResult:
    import gc
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

    print(f"{ULTRASINGER_HEAD} Loading {blue_highlighted('Qwen3-ASR')} {blue_highlighted(model_name)} "
          f"with aligner {blue_highlighted(aligner_name)} and {red_highlighted(device)} as worker")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"

    # ASR and aligner are loaded one after the other to halve peak memory
    model = Qwen3ASRModel.from_pretrained(model_name, dtype=dtype, device_map=device_map,
                                          max_inference_batch_size=batch_size, max_new_tokens=4096)
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    n_chunks = int((len(y) / sr) // MAX_ALIGN_SECONDS) + 1
    bounds = [(k * len(y) // n_chunks, (k + 1) * len(y) // n_chunks) for k in range(n_chunks)]
    print(f"{ULTRASINGER_HEAD} Transcribing {audio_path} in {n_chunks} chunk(s)")
    chunk_results = [model.transcribe(audio=(y[a:b], sr), language=_language_name(language))[0] for a, b in bounds]
    languages = [r.language for r in chunk_results if r.language]
    result = type("R", (), {"language": max(set(languages), key=languages.count) if languages else ""})
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    detected = _language_code(result.language)
    print(f"{ULTRASINGER_HEAD} Qwen3 detected language: {blue_highlighted(result.language or '?')}")

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


def align_lyrics_with_qwen3(
    audio_path: str,
    lyrics: str,
    language: str,
    aligner_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    device: str = "cpu",
) -> TranscriptionResult:
    """Skip transcription: align known lyrics (plain text) with Qwen3-ForcedAligner.
    Note: the aligner takes at most 180 s of audio per call."""
    from qwen_asr import Qwen3ForcedAligner

    print(f"{ULTRASINGER_HEAD} Aligning known lyrics with {blue_highlighted(aligner_name)} ({language})")
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    aligner = Qwen3ForcedAligner.from_pretrained(aligner_name, dtype=dtype,
                                                 device_map="cuda:0" if device == "cuda" else "cpu")
    result = _align_chunked(aligner, audio_path, " ".join(lyrics.split()), _language_name(language))
    transcribed = [
        TranscribedData(confidence=1.0, word=str(i.text).strip() + " ", start=float(i.start_time), end=float(i.end_time))
        for i in result if str(i.text).strip()
    ]
    return TranscriptionResult(transcribed, language)
