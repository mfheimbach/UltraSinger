"""Separate vocals from audio"""
import os
from enum import Enum

import demucs.separate

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    red_highlighted, green_highlighted,
)
from modules.os_helper import check_file_exists

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

def separate_audio(input_file_path: str, output_folder: str, model: DemucsModel, device="cpu") -> None:
    """Separate vocals from audio with demucs."""

    print(
        f"{ULTRASINGER_HEAD} Separating vocals from audio with {blue_highlighted('demucs')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker."
    )

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

def separate_vocal_from_audio(cache_folder_path: str,
                              audio_output_file_path: str,
                              use_separated_vocal: bool,
                              create_karaoke: bool,
                              pytorch_device: str,
                              model: DemucsModel,
                              skip_cache: bool = False) -> str:
    """Separate vocal from audio"""
    demucs_output_folder = os.path.splitext(os.path.basename(audio_output_file_path))[0]
    audio_separation_path = os.path.join(cache_folder_path, "separated", model.value, demucs_output_folder)

    vocals_path = os.path.join(audio_separation_path, "vocals.wav")
    instrumental_path = os.path.join(audio_separation_path, "no_vocals.wav")
    if use_separated_vocal or create_karaoke:
        cache_available = check_file_exists(vocals_path) and check_file_exists(instrumental_path)
        if skip_cache or not cache_available:
            separate_audio(audio_output_file_path, cache_folder_path, model, pytorch_device)
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached separated vocals")

    return audio_separation_path


def extract_multi_track_vocals(
    cache_folder_path: str,
    audio_output_file_path: str,
    use_separated_vocal: bool,
    create_karaoke: bool,
    pytorch_device: str,
    models=None, 
    enabled=True,
    skip_cache: bool = False
) -> dict:
    """
    Extract multiple vocal tracks using different Demucs models.
    
    Args:
        cache_folder_path: Path to the cache folder
        audio_output_file_path: Path to the audio file
        use_separated_vocal: Whether to use separated vocals
        create_karaoke: Whether to create karaoke files
        pytorch_device: Device to use for processing
        models: List of model names to use
        enabled: Whether multi-track processing is enabled
        skip_cache: Whether to skip using cached files
        
    Returns:
        Dictionary with model names as keys and paths to the vocal track folders as values
    """
    if not enabled or not (use_separated_vocal or create_karaoke):
        # Use default model if multi-track is disabled
        model = DemucsModel.HTDEMUCS
        path = separate_vocal_from_audio(
            cache_folder_path, 
            audio_output_file_path, 
            use_separated_vocal, 
            create_karaoke, 
            pytorch_device, 
            model, 
            skip_cache
        )
        return {model.value: path}
    
    # Use default models if none specified
    if models is None:
        models = ["htdemucs", "htdemucs_6s", "htdemucs_ft"]
    
    print(f"{ULTRASINGER_HEAD} Extracting multiple vocal tracks with models: {blue_highlighted(', '.join(models))}")
    
    # Dictionary to store the path to each model's vocal track folder
    vocal_track_folders = {}
    
    # Process each model
    for model_name in models:
        try:
            model = DemucsModel(model_name)
            path = separate_vocal_from_audio(
                cache_folder_path, 
                audio_output_file_path, 
                use_separated_vocal, 
                create_karaoke, 
                pytorch_device, 
                model, 
                skip_cache
            )
            vocal_track_folders[model_name] = path
            print(f"{ULTRASINGER_HEAD} {green_highlighted('Success')} Model {blue_highlighted(model_name)} extracted vocals → {path}")
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} with model {blue_highlighted(model_name)}: {e}")
    
    # If no models succeeded, fall back to default
    if not vocal_track_folders:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning:')} No models succeeded, falling back to default model")
        model = DemucsModel.HTDEMUCS
        path = separate_vocal_from_audio(
            cache_folder_path, 
            audio_output_file_path, 
            use_separated_vocal, 
            create_karaoke, 
            pytorch_device, 
            model, 
            skip_cache
        )
        vocal_track_folders[model.value] = path
        
    return vocal_track_folders