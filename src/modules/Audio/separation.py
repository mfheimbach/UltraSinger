"""Separate vocals from audio"""

import logging
import os
from enum import Enum

import demucs.separate

from modules.console_colors import blue_highlighted, red_highlighted, green_highlighted
from modules.os_helper import check_file_exists

log = logging.getLogger("ultrasinger.separation")

class DemucsModel(Enum):
    HTDEMUCS = "htdemucs"           # first version of Hybrid Transformer Demucs. Trained on MusDB + 800 songs. Default model.
    HTDEMUCS_FT = "htdemucs_ft"     # fine-tuned version of htdemucs, separation will take 4 times more time but might be a bit better. Same training set as htdemucs.
    HTDEMUCS_6S = "htdemucs_6s"     # 6 sources version of htdemucs, with piano and guitar being added as sources. Note that the piano source is not working great at the moment.
    HDEMUCS_MMI = "hdemucs_mmi"     # Hybrid Demucs v3, retrained on MusDB + 800 songs.
    MDX = "mdx"                     # trained only on MusDB HQ, winning model on track A at the MDX challenge.
    MDX_EXTRA = "mdx_extra"         # trained with extra training data (including MusDB test set), ranked 2nd on the track B of the MDX challenge.
    MDX_Q = "mdx_q"                 # quantized version of the previous models. Smaller download and storage but quality can be slightly worse.
    MDX_EXTRA_Q = "mdx_extra_q"     # quantized version of mdx_extra. Smaller download and storage but quality can be slightly worse.
    SIG = "SIG"                     # Placeholder for a single model from the model zoo.

def separate_audio(input_file_path: str, output_folder: str, model: DemucsModel, device: str, seed: int) -> None:
    """Separate vocals from audio with demucs (its random time shift seeded: reproducible).
    Averaging several shifted passes (--shifts 3) was tested on the SingStar bench: no effect beyond
    noise, 3x slower, so it is not offered."""

    log.info(f"Separating vocals from audio with {blue_highlighted('demucs')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker."
    )

    # demucs shifts the input by a random offset (shifts=1 by default): seed it, or every run differs
    import random
    random.seed(seed)
    demucs.separate.main(
        [
            "--two-stems", "vocals",
            "-d", f"{device}",
            "--float32",
            "-n",
            model.value,
            "--out", f"{os.path.join(output_folder, 'separated')}",
            f"{input_file_path}",
        ]
    )

def separate_audio_with_audio_separator(input_file_path: str, output_path: str, model_filename: str,
                                        lead_model_filename: str | None, segment_size: int, overlap: int,
                                        reuse_from: str | None = None) -> None:
    """Separate with python-audio-separator (RoFormer/MDX/UVR models). Optional second pass:
    a karaoke model on the vocals keeps only the lead vocal. Writes vocals.wav / no_vocals.wav."""
    import shutil
    import tempfile
    from audio_separator.separator import Separator

    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "audio-separator")
    os.makedirs(output_path, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="sep_")

    def run(model: str, src: str) -> tuple[str, str]:
        log.info(f"Separating with {blue_highlighted('audio-separator')} model {blue_highlighted(model)}")
        sep = Separator(output_dir=tmp, model_file_dir=model_dir, output_format="WAV",
                        mdxc_params={"segment_size": segment_size, "override_model_segment_size": False,
                                     "batch_size": 1, "overlap": overlap, "pitch_shift": 0})
        sep.load_model(model_filename=model)
        files = [os.path.join(tmp, f) if not os.path.isabs(f) else f for f in sep.separate(src)]
        # vocal stem: "(Vocals)" for separators, "(noreverb)" for de-reverb models
        vocals = next(f for f in files if "(vocals)" in f.lower() or "(noreverb)" in f.lower())
        other = next(f for f in files if f != vocals)
        return vocals, other

    if reuse_from and os.path.exists(os.path.join(reuse_from, "vocals.wav")):
        # first pass already done (e.g. by a run without the lead-vocal pass)
        log.info(f"{green_highlighted('cache')} reusing first-pass separation from {reuse_from}")
        vocals = os.path.join(tmp, "first_vocals.wav")
        instrumental = os.path.join(tmp, "first_no_vocals.wav")
        shutil.copy(os.path.join(reuse_from, "vocals.wav"), vocals)
        shutil.copy(os.path.join(reuse_from, "no_vocals.wav"), instrumental)
    else:
        vocals, instrumental = run(model_filename, input_file_path)
    if lead_model_filename:
        lead, backing = run(lead_model_filename, vocals)
        vocals = lead
    import gc
    gc.collect()  # separator models are large; free them before transcription loads its own
    shutil.move(vocals, os.path.join(output_path, "vocals.wav"))
    shutil.move(instrumental, os.path.join(output_path, "no_vocals.wav"))
    shutil.rmtree(tmp, ignore_errors=True)


def separate_vocal_from_audio(cache_folder_path: str,
                              audio_output_file_path: str,
                              use_separated_vocal: bool,
                              create_karaoke: bool,
                              pytorch_device: str,
                              model: DemucsModel,
                              skip_cache: bool,
                              separator_model: str | None,
                              lead_vocal_model: str | None,
                              seed: int,
                              separator_segment_size: int,
                              separator_overlap: int) -> str:
    """Separate vocal from audio"""
    demucs_output_folder = os.path.splitext(os.path.basename(audio_output_file_path))[0]
    model_folder = model.value + (f"+{os.path.splitext(lead_vocal_model)[0]}" if lead_vocal_model else "")
    if separator_model:
        model_folder = os.path.splitext(separator_model)[0] + (
            f"+{os.path.splitext(lead_vocal_model)[0]}" if lead_vocal_model else "")
    audio_separation_path = os.path.join(cache_folder_path, "separated", model_folder, demucs_output_folder)

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")
    if use_separated_vocal or create_karaoke:
        cache_available = check_file_exists(vocals_path) and check_file_exists(instrumental_path)
        if skip_cache or not cache_available:
            if separator_model:
                reuse = None
                if lead_vocal_model:
                    # a sibling cache (same song, no lead pass) may already hold the first pass
                    stem = os.path.splitext(separator_model)[0]
                    sibling = os.path.join(os.path.dirname(os.path.normpath(cache_folder_path)), f"sep_{stem}")
                    reuse = os.path.join(sibling, "separated", stem, demucs_output_folder)
                separate_audio_with_audio_separator(audio_output_file_path, audio_separation_path,
                                                    separator_model, lead_vocal_model, separator_segment_size,
                                                    separator_overlap, reuse_from=reuse)
            elif lead_vocal_model:
                # second pass (lead vocal / de-reverb) on top of demucs; reuse a sibling demucs cache
                name = os.path.join("separated", model.value, demucs_output_folder)
                base = os.path.dirname(os.path.normpath(cache_folder_path))
                reuse = next((os.path.join(c, name) for c in (os.path.join(base, f"demucs_{model.value}"), base,
                                                               cache_folder_path)
                              if check_file_exists(os.path.join(c, name, "vocals.wav"))), None)
                if reuse is None:
                    separate_audio(audio_output_file_path, cache_folder_path, model, pytorch_device, seed)
                    reuse = os.path.join(cache_folder_path, name)
                separate_audio_with_audio_separator(audio_output_file_path, audio_separation_path,
                                                    None, lead_vocal_model, separator_segment_size,
                                                    separator_overlap, reuse_from=reuse)
            else:
                separate_audio(audio_output_file_path, cache_folder_path, model, pytorch_device, seed)
        else:
            log.info(f"{green_highlighted('cache')} reusing cached separated vocals")

    return audio_separation_path

