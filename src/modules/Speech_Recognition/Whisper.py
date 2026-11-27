"""Whisper Speech Recognition Module"""

import logging

# Set environment variable to handle cuDNN loading issues
# CUDA_MODULE_LOADING=LAZY allows PyTorch to continue even if some cuDNN modules are not found
#os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import torch

# Fix for PyTorch 2.6+ compatibility with omegaconf and old models
# MUST be done before importing whisperx or any other modules that use torch.load
# Monkey-patch torch.load to use weights_only=False for compatibility with older models
# This is necessary because PyTorch 2.6+ changed the default to weights_only=True
# but WhisperX/Pyannote models were saved with pickle and require weights_only=False
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False regardless of what was passed
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
from enum import Enum
from torch.cuda import OutOfMemoryError

from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import from_whisper

#Addition for numbers to words
import re
import ast
from num2words import num2words

log = logging.getLogger("ultrasinger.whisper")

#Addition for numbers to words
re_split_preserve_space = re.compile(r'(\d+|\W+|\w+)')


MEMORY_ERROR_MESSAGE = "whisper ran out of GPU memory: reduce transcribe.whisper.batch_size or run with --cpu"

class WhisperModel(Enum):
    """Whisper model"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"
    LARGE_V3_TURBO = "large-v3-turbo"

#Addition for numbers to words (Using previous code from louispan in PR#135)
def number_to_words(line,language='en'):
    # https://github.com/m-bain/whisperX
    # Transcript words which do not contain characters in the alignment models dictionary e.g. "2014." or "£13.60" cannot be aligned and therefore are not given a timing.
    # Therefore, convert numbers to words
    out_tokens = []
    in_tokens = re_split_preserve_space.findall(line)
    for token in in_tokens:
        try:
            num = ast.literal_eval(token)
            try:
                out_tokens.append(num2words(num, lang=language))
            except NotImplementedError:
                log.warning(f"{red_highlighted('Error:')} Unknown language for number transcription. Keeping number as numeric characters for line: {line}, token: {token}"
                )
        except Exception:
            out_tokens.append(token)
    return ''.join(out_tokens) 


_MODELS: dict = {}   # kind -> (key, model): one loaded model per kind, kept across songs


def _cached(kind: str, key: tuple, load):
    hit = _MODELS.get(kind)
    if hit is not None and hit[0] == key:
        return hit[1]
    _MODELS.pop(kind, None)          # a different model of this kind: release the old one first
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    model = load()
    _MODELS[kind] = (key, model)
    return model


def free_models() -> None:
    """Release the cached Whisper and alignment models (and their GPU memory)."""
    _MODELS.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()


from modules import gpu_memory  # noqa: E402
gpu_memory.register(free_models)


def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device: str,
    alignment_model: str | None,
    batch_size: int,
    compute_type: str | None,
    language: str | None,
    keep_numbers: bool,
) -> TranscriptionResult:
    """Transcribe with whisper"""
    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.
    log.info(f"Loading {blue_highlighted('whisper')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker"
    )
    if alignment_model is not None:
        log.info(f"using alignment model {blue_highlighted(alignment_model)}")

    # null in the preset = the device's usual precision (float16 on CUDA, int8 on CPU)
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        torch.cuda.empty_cache()
        loaded_whisper_model = _cached("asr", (model.value, language, device, compute_type),
                                       lambda: whisperx.load_model(model.value, language=language, device=device,
                                                                   compute_type=compute_type))

        audio = whisperx.load_audio(audio_path)

        log.info(f"Transcribing {audio_path}")

        result = loaded_whisper_model.transcribe(
            audio, batch_size=batch_size, language=language
        )

        detected_language = result["language"]
        if language is None:
            language = detected_language

        # load alignment model and metadata
        try:
            model_a, metadata = _cached("align", (language, device, alignment_model),
                                        lambda: whisperx.load_align_model(language_code=language, device=device,
                                                                          model_name=alignment_model))
        except ValueError as ve:
            log.error(f"{ve}: unknown language for alignment; set transcribe.whisper.align_model "
                      "to a wav2vec2 model from huggingface")
            raise ve

        #Addition for numbers to words (Using previous code from louispan in PR#135)
        if keep_numbers == False: 
            for obj in result["segments"]:
                obj["text"] = number_to_words(obj["text"],language)

        # align whisper output
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        transcribed_data = convert_to_transcribed_data(result_aligned)

        # models stay loaded (one ASR model + one aligner, ~2 GB VRAM on CUDA); a run over many songs
        # skips ~6 s of loading per song. free_models() releases them (out of memory below).
        del loaded_whisper_model, model_a

        return TranscriptionResult(transcribed_data, detected_language)
    except ValueError as value_error:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error.args[0])
        ):
            log.error(str(value_error))
            log.info("Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )

        raise value_error
    except OutOfMemoryError as oom_exception:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        free_models()
        log.error(str(oom_exception))
        log.error(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        # Restore original torch.load in case of error
        torch.load = _original_torch_load

        if "CUDA failed with error out of memory" in str(exception.args[0]):
            log.error(str(exception))
            log.error(MEMORY_ERROR_MESSAGE)
        raise exception
    finally:
        # Restore original torch.load after models are loaded
        # This ensures other modules (like pitch detection) are not affected by the monkey-patch
        torch.load = _original_torch_load


def convert_to_transcribed_data(result_aligned):
    """Whisper/wav2vec2 words -> TranscribedData. Words the aligner left without timestamps (numbers,
    symbols) get start = end = None; the words.place_untimed step places them (and says so)."""
    transcribed_data = []
    untimed = 0
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = from_whisper(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if "start" not in obj or "end" not in obj:
                vtd.start = vtd.end = None
                untimed += 1
            transcribed_data.append(vtd)
    if untimed:
        log.info(f"{untimed} word(s) without timestamps from the aligner")
    return transcribed_data


def align_lyrics_with_whisperx(
    audio_path: str,
    lyrics: str,
    language: str,
    device: str,
    alignment_model: str | None,
) -> TranscriptionResult:
    """Skip transcription: align known lyrics (plain text) to the audio with wav2vec2."""
    log.info(f"Aligning known lyrics with {blue_highlighted('wav2vec2')} ({language})")
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / whisperx.audio.SAMPLE_RATE
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device, model_name=alignment_model)
    text = number_to_words(" ".join(lyrics.split()), language)
    segments = [{"text": text, "start": 0.0, "end": duration}]
    result_aligned = whisperx.align(segments, model_a, metadata, audio, device, return_char_alignments=False)
    return TranscriptionResult(convert_to_transcribed_data(result_aligned), language)
