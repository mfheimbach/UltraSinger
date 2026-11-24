from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Audio.separation import DemucsModel
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_txt import FormatVersion


@dataclass_json
@dataclass
class Settings:

    APP_VERSION = "0.0.13.dev16"
    CONFIDENCE_THRESHOLD = 0.6
    CONFIDENCE_PROMPT_TIMEOUT = 4

    create_midi = True
    create_plot = False
    create_audio_chunks = False
    hyphenation = True
    use_separated_vocal = True
    create_karaoke = True
    ignore_audio = False
    input_file_is_ultrastar_txt = False # todo: to process_data
    keep_cache = False
    interactive_mode = False
    user_ffmpeg_path = ""
    quantize_to_key = True  # Quantize notes to the detected key of the song

    # Process data Paths
    input_file_path = ""
    output_folder_path = ""
    
    language = None
    format_version = FormatVersion.V1_2_0

    # Demucs
    demucs_model = DemucsModel.HTDEMUCS  # htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q|SIG

    # Whisper
    transcriber = "whisper"  # whisper|qwen3
    whisper_model = WhisperModel.LARGE_V2  # Multilingual model tiny|base|small|medium|large-v1|large-v2|large-v3
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 16   # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)
    keep_numbers = False
    # Which audio the transcriber hears: processed (separated+denoised+muted, original
    # behaviour) | vocals (separated, untouched) | mix (original audio, no separation)
    transcription_audio = "processed"

    # Known lyrics: plain-text file to align instead of transcribing (language must be set)
    lyrics_file = None
    lyrics_aligner = "wav2vec2"  # wav2vec2|qwen3
    # Word timing for Whisper transcripts: wav2vec2 (whisperX, original) | qwen3 (re-align
    # Whisper's text with Qwen3-ForcedAligner, falls back to wav2vec2 if it collapses)
    word_aligner = "wav2vec2"

    # Qwen3-ASR (transcriber = "qwen3", needs `pip install qwen-asr`)
    qwen3_model = "Qwen/Qwen3-ASR-1.7B"  # or Qwen/Qwen3-ASR-0.6B
    qwen3_aligner = "Qwen/Qwen3-ForcedAligner-0.6B"

    # Note pitch: how pitch frames inside a note become one note (defaults = original behaviour)
    note_pitch_method = "mode"  # mode|median|chroma_mode
    note_pitch_conf_threshold = 0.4
    note_pitch_edge_trim = 0.0  # fraction of frames ignored at each end of a note
    note_pitch_tuning_correction = False  # compensate recordings not tuned to A440
    key_quantize_min_offset = 0.0  # limited key quantization: only snap out-of-key notes measured >= this far (semitones) off-centre

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    force_cpu = False
    force_whisper_cpu = False

    # MuseScore
    musescore_path = None

    # yt-dlp
    cookiefile = None

    # UltraSinger Evaluation Configuration
    test_songs_input_folder = None
    cache_override_path = None #"C:\\UltraSinger\\test_output"
    skip_cache_vocal_separation = False
    skip_cache_denoise_vocal_audio = False
    skip_cache_transcription = False
    skip_cache_pitch_detection = False
    calculate_score = True