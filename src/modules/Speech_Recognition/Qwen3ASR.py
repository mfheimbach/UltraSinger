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


def transcribe_with_qwen3(
    audio_path: str,
    model_name: str = "Qwen/Qwen3-ASR-1.7B",
    aligner_name: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    device: str = "cpu",
    language: str = None,
    batch_size: int = 8,
) -> TranscriptionResult:
    from qwen_asr import Qwen3ASRModel

    print(f"{ULTRASINGER_HEAD} Loading {blue_highlighted('Qwen3-ASR')} {blue_highlighted(model_name)} "
          f"with aligner {blue_highlighted(aligner_name)} and {red_highlighted(device)} as worker")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    device_map = "cuda:0" if device == "cuda" else "cpu"
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        max_inference_batch_size=batch_size,
        max_new_tokens=4096,
        forced_aligner=aligner_name,
        forced_aligner_kwargs=dict(dtype=dtype, device_map=device_map),
    )

    print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")
    result = model.transcribe(audio=audio_path, language=_language_name(language), return_time_stamps=True)[0]

    detected = _language_code(result.language)
    print(f"{ULTRASINGER_HEAD} Qwen3 detected language: {blue_highlighted(result.language or '?')}")

    transcribed = []
    for item in (result.time_stamps or []):
        word = str(item.text).strip()
        if not word:
            continue
        transcribed.append(TranscribedData(
            confidence=1.0,  # the aligner gives no per-word score
            word=word + " ",
            start=float(item.start_time),
            end=float(item.end_time),
        ))

    del model
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
    result = aligner.align(audio=audio_path, text=" ".join(lyrics.split()), language=_language_name(language))[0]
    transcribed = [
        TranscribedData(confidence=1.0, word=str(i.text).strip() + " ", start=float(i.start_time), end=float(i.end_time))
        for i in result if str(i.text).strip()
    ]
    return TranscriptionResult(transcribed, language)
