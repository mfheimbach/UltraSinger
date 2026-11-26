"""Reduce noise from audio"""

import logging

import ffmpeg

from modules.console_colors import blue_highlighted, green_highlighted
from modules.os_helper import check_file_exists

log = logging.getLogger("ultrasinger.denoise")


def __ffmpeg_reduce_noise(input_file_path: str, output_file: str, reduction_db: float, floor_db: float) -> None:
    """Reduce noise from vocal audio with ffmpeg."""

    # Denoise audio samples with FFT.
    # A description of the accepted parameters follows.

    # noise_reduction, nr
    #    Set the noise reduction in dB, allowed range is 0.01 to 97. Default value is 12 dB.
    # noise_floor, nf
    #    Set the noise floor in dB, allowed range is -80 to -20. Default value is -50 dB.
    # track_noise, tn
    #    Enable noise floor tracking. By default is disabled.
    #    With this enabled, noise floor is automatically adjusted.

    log.info(f"Reduce noise from vocal audio with {blue_highlighted('ffmpeg')}."
    )
    try:
        (
            ffmpeg.input(input_file_path)
            .output(output_file, af=f"afftdn=nr={reduction_db:g}:nf={floor_db:g}:tn=1")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as ffmpeg_exception:
        log.error("ffmpeg stdout: %s", ffmpeg_exception.stdout.decode("utf8"))
        log.error("ffmpeg stderr: %s", ffmpeg_exception.stderr.decode("utf8"))
        raise ffmpeg_exception


def denoise_vocal_audio(input_path: str, output_path: str, reduction_db: float, floor_db: float,
                        skip_cache: bool = False) -> None:
    """Denoise vocal audio (ffmpeg afftdn). The output path must identify input and settings."""
    cache_available = check_file_exists(output_path)
    if skip_cache or not cache_available:
        __ffmpeg_reduce_noise(input_path, output_path, reduction_db, floor_db)
    else:
        log.info(f"{green_highlighted('cache')} reusing cached denoised audio")
