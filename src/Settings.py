from dataclasses import dataclass

from dataclasses_json import dataclass_json

from modules.Audio.separation import DemucsModel
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_txt import FormatVersion


@dataclass_json
@dataclass
class Settings:

    APP_VERSION = "0.0.13-dev8"
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

    # Process data Paths
    input_file_path = ""
    output_folder_path = ""
    
    language = None
    format_version = FormatVersion.V1_2_0

    # Demucs
    demucs_model = DemucsModel.HTDEMUCS  # htdemucs|htdemucs_ft|htdemucs_6s|hdemucs_mmi|mdx|mdx_extra|mdx_q|mdx_extra_q|SIG

    # Whisper
    transcriber = "whisper"  # whisper
    whisper_model = WhisperModel.LARGE_V2  # Multilingual model tiny|base|small|medium|large-v1|large-v2|large-v3
    # English-only model tiny.en|base.en|small.en|medium.en
    whisper_align_model = None   # Model for other languages from huggingface.co e.g -> "gigant/romanian-wav2vec2"
    whisper_batch_size = 16   # reduce if low on GPU mem
    whisper_compute_type = None   # change to "int8" if low on GPU mem (may reduce accuracy)
    keep_numbers = False

    # Pitch
    crepe_model_capacity = "full"  # tiny|small|medium|large|full
    crepe_step_size = 10 # in miliseconds

    # Device
    pytorch_device = 'cpu'  # cpu|cuda
    tensorflow_device = 'cpu'  # cpu|cuda
    force_cpu = False
    force_whisper_cpu = False
    force_crepe_cpu = False

    # MuseScore
    musescore_path = None

    # yt-dl
    cookiefile = None

    # UltraSinger Evaluation Configuration
    test_songs_input_folder = None
    cache_override_path = None
    skip_cache_vocal_separation = False
    skip_cache_denoise_vocal_audio = False
    skip_cache_transcription = False
    skip_cache_pitch_detection = False
    calculate_score = True
    
    # Linebreak optimization parameters
    OPTIMAL_SYLLABLES_PER_LINE = 10    # Target number of syllables per line
    MAX_LINE_DURATION = 5.0            # Maximum line duration in seconds
    MIN_LINE_DURATION = 2.0            # Minimum line duration in seconds
    NORMAL_BPM = 120.0                 # Reference BPM for adaptation
    SYLLABLE_TOLERANCE = 3.0           # Controls bell curve width for syllables
    TIME_TOLERANCE = 1.0               # Controls bell curve width for time
    USE_OPTIMIZED_LINEBREAKS = True    # Enable optimized linebreak algorithm
    
    # Note sanity check parameters
    ENABLE_NOTE_OPTIMIZATIONS = True   # Enable note length and overlap optimizations
    
    # Multi-track vocal processing
    USE_MULTI_TRACK_PROCESSING = True  # Master toggle
    USE_MULTI_TRACK_LYRICS = True      # For transcription
    USE_MULTI_TRACK_PITCH = True       # For pitch detection
    MULTI_TRACK_MODELS = ["htdemucs", "htdemucs_6s", "htdemucs_ft"]  # Models to use
    DEBUG_LEVEL = 1  # 0=minimal, 1=basic stats, 2=visualizations, 3=interactive

    # Multi-track transcription settings
    USE_MULTI_TRACK_TRANSCRIPTION = True  # Master toggle for multi-track transcription
    TRANSCRIPTION_MODEL_WEIGHTS = {
        "htdemucs_ft": 0.6,  # Fine-tuned model (highest quality)
        "htdemucs": 0.3,     # Default model
        "htdemucs_6s": 0.4   # 6-source model (often better for vocals)
    }
    TRANSCRIPTION_DOMINANT_MODEL = "htdemucs_ft"  # Model to prioritize when in doubt
    TRANSCRIPTION_DEBUG_LEVEL = 1  # 0=minimal, 1=basic stats, 2=visualizations

    # Combination parameters
    PITCH_CONFIDENCE_THRESHOLDS = {
        "htdemucs": 0.4,     # Primary/baseline
        "htdemucs_6s": 0.3,  # More permissive
        "htdemucs_ft": 0.5   # High quality
    }
    MULTI_TRACK_AGREEMENT_BONUS = 0.2  # Confidence boost when tracks agree
    MIN_NOTE_DURATION = 0.35  # Minimum note duration in seconds
    
    # Voice Activity Detection settings
    USE_VAD = True                     # Enable Voice Activity Detection
    VAD_THRESHOLD = 0.15              # Threshold for considering a frame as vocal
    VAD_ENERGY_THRESHOLD = 0.05       # Threshold for energy-based detection
    VAD_CONFIDENCE_BOOST = 0.2        # Amount to boost confidence for vocal frames
    VAD_MODEL_WEIGHTS = {             # Model-specific reliability weights
        "htdemucs_ft": 0.6,           # Most reliable
        "htdemucs": 0.3,              # Standard
        "htdemucs_6s": 0.4            # More sources, better separation
    }

    # Note Regularization Settings
    ENABLE_NOTE_REGULARIZATION = True  # Master toggle for the new regularization
    NOTE_LOW_SIGNIFICANCE_THRESHOLD = 0.3  # Threshold for considering a note "low significance"
    NOTE_HIGH_SIGNIFICANCE_THRESHOLD = 0.6  # Threshold for considering a note "high significance"
    MAX_PITCH_DIFF_FOR_MERGE = 2.0  # Maximum semitone difference to allow merging notes
    NOTE_SIGNIFICANCE_WEIGHTS = {
        'duration': 0.35,       # Weight for duration component
        'pitch_change': 0.25,   # Weight for pitch change component
        'beat_alignment': 0.15, # Weight for beat alignment component
        'word_boundary': 0.15,  # Weight for word boundary component
        'vad': 0.10             # Weight for VAD confidence component
    }
    
    # Multi-track transcription settings
    USE_MULTI_TRACK_TRANSCRIPTION = True  # Master toggle for multi-track transcription
    TRANSCRIPTION_MODEL_WEIGHTS = {
        "htdemucs_ft": 0.6,  # Fine-tuned model (highest quality)
        "htdemucs": 0.3,     # Default model
        "htdemucs_6s": 0.4   # 6-source model (often better for vocals)
    }
    TRANSCRIPTION_DOMINANT_MODEL = "htdemucs_ft"  # Model to prioritize when in doubt
    TRANSCRIPTION_DEBUG_LEVEL = 1  # 0=minimal, 1=basic stats, 2=visualizations

    # Word alignment settings
    WORD_SIMILARITY_THRESHOLD = 0.7  # Minimum similarity to consider words matching
    WORD_OVERLAP_THRESHOLD = 0.3    # Minimum time overlap to consider matching