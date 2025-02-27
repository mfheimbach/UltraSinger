"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import sys
import Levenshtein

from packaging import version

from modules import os_helper
from modules.init_interactive_mode import init_settings_interactive
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import (
    separate_vocal_from_audio,
    extract_multi_track_vocals,
)
from modules.Audio.vocal_chunks import (
    create_audio_chunks_from_transcribed_data,
    create_audio_chunks_from_ultrastar_data,
)
from modules.Audio.silence_processing import remove_silence_from_transcription_data, mute_no_singing_parts
from modules.Audio.separation import DemucsModel
from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_wav_to_mp3
from modules.Audio.youtube import (
    download_from_youtube,
)
from modules.Audio.bpm import get_bpm_from_file

from modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    red_highlighted,
    green_highlighted,
    cyan_highlighted,
    bright_green_highlighted,
)
from modules.Midi.midi_creator import (
    create_midi_segments_from_transcribed_data,
    create_midi_segments_with_probability,
    create_repitched_midi_segments_from_ultrastar_txt,
    create_midi_file,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_length_calculator import get_thirtytwo_note_second, get_sixteenth_note_second
from modules.Pitcher.pitcher import (
    get_pitch_with_crepe_file,
    get_multi_track_pitch,
)
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.hyphenation import (
    hyphenate_each_word,
)
from modules.Speech_Recognition.Whisper import transcribe_with_whisper
from modules.Ultrastar import (
    ultrastar_writer,
)
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.Whisper import WhisperModel
from modules.Ultrastar.ultrastar_score_calculator import Score, calculate_score_points
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING, FormatVersion
from modules.Ultrastar.coverter.ultrastar_txt_converter import from_ultrastar_txt, \
    create_ultrastar_txt_from_midi_segments, create_ultrastar_txt_from_automation
from modules.Ultrastar.ultrastar_parser import parse_ultrastar_txt
from modules.common_print import print_support, print_help, print_version
from modules.os_helper import check_file_exists, get_unused_song_output_dir
from modules.plot import create_plots
from modules.musicbrainz_client import search_musicbrainz
from modules.sheet import create_sheet
from modules.ProcessData import ProcessData, ProcessDataPaths, MediaInfo
from modules.DeviceDetection.device_detection import check_gpu_support
from modules.Image.image_helper import save_image
from modules.ffmpeg_helper import is_ffmpeg_available, get_ffmpeg_and_ffprobe_paths

from Settings import Settings

settings = Settings()


def add_hyphen_to_data(
        transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]
):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            new_data.append(data)
        else:
            chunk_duration = data.end - data.start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = data.start
            for j in enumerate(hyphen_words[i]):
                hyphenated_word_index = j[0]
                dup = copy.copy(data)
                dup.start = next_start
                next_start = data.end - chunk_duration * (
                        len(hyphen_words[i]) - 1 - hyphenated_word_index
                )
                dup.end = next_start
                dup.word = hyphen_words[i][hyphenated_word_index]
                dup.is_hyphen = True
                if hyphenated_word_index == len(hyphen_words[i]) - 1:
                    dup.is_word_end = True
                else:
                    dup.is_word_end = False
                new_data.append(dup)

    return new_data


# Todo: Unused
def correct_words(recognized_words, word_list_file):
    """Docstring"""
    with open(word_list_file, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue

        closest_word = min(
            word_list, key=lambda x: Levenshtein.distance(rec_word.word, x)
        )
        print(recognized_words[i].word + " - " + closest_word)
        recognized_words[i].word = closest_word
    return recognized_words


def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for i, data in enumerate(transcribed_data):
        data.word = data.word.translate({ord(i): None for i in punctuation})


def run() -> tuple[str, Score, Score]:
    """The processing function of this program"""
    #List selected options (can add more later)
    if settings.keep_numbers:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Numbers will be transcribed as numerics (i.e. 1, 2, 3, etc.)')}")
    if settings.create_plot:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Plot will be created')}")
    if settings.keep_cache:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Cache folder will not be deleted')}")
    if settings.create_audio_chunks:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Audio chunks will be created')}")
    if not settings.create_karaoke:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Karaoke txt will not be created')}")
    if not settings.use_separated_vocal:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Vocals will not be separated')}")
    if not settings.hyphenation:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Hyphenation will not be applied')}")
    if hasattr(settings, 'USE_MULTI_TRACK_PROCESSING') and settings.USE_MULTI_TRACK_PROCESSING:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Multi-track vocal processing enabled')}")
        if hasattr(settings, 'USE_MULTI_TRACK_PITCH') and settings.USE_MULTI_TRACK_PITCH:
            print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Multi-track pitch detection enabled')}")
    if hasattr(settings, 'USE_VAD') and settings.USE_VAD:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Voice Activity Detection (VAD) enabled')}")

    process_data = InitProcessData()

    process_data.process_data_paths.cache_folder_path = (
        os.path.join(settings.output_folder_path, "cache")
        if settings.cache_override_path is None
        else settings.cache_override_path
    )

    # Create process audio
    process_data.process_data_paths.processing_audio_path = CreateProcessAudio(process_data)

    # Audio transcription
    process_data.media_info.language = settings.language
    if not settings.ignore_audio:
        TranscribeAudio(process_data)

    # Split syllables into segments
    if not settings.ignore_audio:
        process_data.transcribed_data = split_syllables_into_segments(process_data.transcribed_data,
                                                                  process_data.media_info.bpm)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(process_data)

    # Pitch audio - pass process_data for multi-track processing
    process_data.pitched_data = pitch_audio(process_data.process_data_paths, process_data)

    # Create Midi_Segments
    if not settings.ignore_audio:
        process_data.midi_segments = create_midi_segments_from_transcribed_data(process_data.transcribed_data,
                                                                                process_data.pitched_data)
    else:
        process_data.midi_segments = create_repitched_midi_segments_from_ultrastar_txt(process_data.pitched_data,
                                                                                       process_data.parsed_file)

    # Merge syllable segments
    if not settings.ignore_audio:
        process_data.midi_segments, process_data.transcribed_data = merge_syllable_segments(process_data.midi_segments,
                                                                                        process_data.transcribed_data,
                                                                                        process_data.media_info.bpm)

    # Create plot
    if settings.create_plot:
        create_plots(process_data, settings.output_folder_path)

    # Create Ultrastar txt
    accurate_score, simple_score, ultrastar_file_output = CreateUltraStarTxt(process_data)

    # Create Midi
    if settings.create_midi:
        create_midi_file(process_data.media_info.bpm, settings.output_folder_path, process_data.midi_segments,
                         process_data.basename)

    # Sheet music
    create_sheet(process_data.midi_segments, settings.output_folder_path,
                 process_data.process_data_paths.cache_folder_path, settings.musescore_path, process_data.basename,
                 process_data.media_info)

    # Cleanup
    if not settings.keep_cache:
        remove_cache_folder(process_data.process_data_paths.cache_folder_path)

    # Print Support
    print_support()
    return ultrastar_file_output, simple_score, accurate_score


def split_syllables_into_segments(
        transcribed_data: list[TranscribedData],
        real_bpm: float) -> list[TranscribedData]:
    """Split every syllable into sub-segments"""
    syllable_segment_size = get_sixteenth_note_second(real_bpm)

    segment_size_decimal_points = len(str(syllable_segment_size).split(".")[1])
    new_data = []

    for i, data in enumerate(transcribed_data):
        duration = data.end - data.start
        if duration <= syllable_segment_size:
            new_data.append(data)
            continue

        has_space = str(data.word).endswith(" ")
        first_segment = copy.deepcopy(data)
        filler_words_start = data.start + syllable_segment_size
        remainder = data.end - filler_words_start
        first_segment.end = filler_words_start
        if has_space:
            first_segment.word = first_segment.word[:-1]

        first_segment.is_word_end = False
        new_data.append(first_segment)

        full_segments, partial_segment = divmod(remainder, syllable_segment_size)

        if full_segments >= 1:
            first_segment.is_hyphen = True
            for i in range(int(full_segments)):
                segment = TranscribedData()
                segment.word = "~"
                segment.start = filler_words_start + round(
                    i * syllable_segment_size, segment_size_decimal_points
                )
                segment.end = segment.start + syllable_segment_size
                segment.is_hyphen = True
                segment.is_word_end = False
                new_data.append(segment)

        if partial_segment >= 0.01:
            first_segment.is_hyphen = True
            segment = TranscribedData()
            segment.word = "~"
            segment.start = filler_words_start + round(
                full_segments * syllable_segment_size, segment_size_decimal_points
            )
            segment.end = segment.start + partial_segment
            segment.is_hyphen = True
            segment.is_word_end = False
            new_data.append(segment)

        if has_space:
            new_data[-1].word += " "
            new_data[-1].is_word_end = True
    return new_data


def merge_syllable_segments(midi_segments: list[MidiSegment],
                            transcribed_data: list[TranscribedData],
                            real_bpm: float) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Merge sub-segments of a syllable where the pitch is the same"""

    # Check if there are any segments to process
    if not midi_segments or len(midi_segments) < 2:
        return midi_segments, transcribed_data

    thirtytwo_note = get_thirtytwo_note_second(real_bpm)
    sixteenth_note = get_sixteenth_note_second(real_bpm)

    new_data = []
    new_midi_notes = []

    previous_data = None

    for i, data in enumerate(transcribed_data):
        # Skip this process if we don't have enough MIDI segments
        if i >= len(midi_segments):
            new_data.append(data)
            continue

        is_note_short = (data.end - data.start) < thirtytwo_note
        # Only check is_same_note if i > 0
        is_same_note = i > 0 and midi_segments[i].note == midi_segments[i - 1].note
        has_breath_pause = False

        if previous_data is not None:
            has_breath_pause = (data.start - previous_data.end) > sixteenth_note

        if (str(data.word).startswith("~")
                and previous_data is not None
                and (is_note_short or is_same_note)
                and not has_breath_pause):
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            if str(data.word).endswith(" "):
                new_data[-1].word += " "
                new_midi_notes[-1].word += " "
                new_data[-1].is_word_end = True

        else:
            new_data.append(data)
            new_midi_notes.append(midi_segments[i])

        previous_data = data

    return new_midi_notes, new_data


def create_audio_chunks(process_data):
    if not settings.ignore_audio:
        create_audio_chunks_from_transcribed_data(
            process_data.process_data_paths,
            process_data.transcribed_data)
    else:
        create_audio_chunks_from_ultrastar_data(
            process_data.process_data_paths,
            process_data.parsed_file
        )


def InitProcessData():
    settings.input_file_is_ultrastar_txt = settings.input_file_path.endswith(".txt")
    if settings.input_file_is_ultrastar_txt:
        # Parse Ultrastar txt
        (
            basename,
            settings.output_folder_path,
            audio_file_path,
            ultrastar_class,
        ) = parse_ultrastar_txt(settings.input_file_path, settings.output_folder_path)
        process_data = from_ultrastar_txt(ultrastar_class)
        process_data.basename = basename
        process_data.process_data_paths.audio_output_file_path = audio_file_path
        # todo: ignore transcribe
        settings.ignore_audio = True

    elif settings.input_file_path.startswith("https:"):
        # Youtube
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info
        ) = download_from_youtube(settings.input_file_path, settings.output_folder_path, settings.cookiefile)
    else:
        # Audio File
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = infos_from_audio_input_file()
    return process_data


def TranscribeAudio(process_data):
    """Transcribe audio with multi-track support"""
    
    # Check if multi-track transcription is enabled
    use_multi_track = (hasattr(settings, 'USE_MULTI_TRACK_TRANSCRIPTION') and 
                       settings.USE_MULTI_TRACK_TRANSCRIPTION and
                       hasattr(process_data, 'vocal_tracks') and
                       len(process_data.vocal_tracks) > 1)
    
    if use_multi_track:
        # Use multi-track transcription
        from modules.Speech_Recognition.multi_track_whisper import transcribe_multiple_tracks, analyze_transcription_quality
        
        # Get model weights and dominant model from settings
        model_weights = getattr(settings, 'TRANSCRIPTION_MODEL_WEIGHTS', {
            "htdemucs_ft": 0.6,
            "htdemucs": 0.3,
            "htdemucs_6s": 0.4
        })
        
        dominant_model = getattr(settings, 'TRANSCRIPTION_DOMINANT_MODEL', "htdemucs_ft")
        
        # Run multi-track transcription
        transcription_result, results_by_model = transcribe_multiple_tracks(
            process_data.vocal_tracks,
            settings.whisper_model,
            settings.pytorch_device,
            settings.whisper_align_model,
            settings.whisper_batch_size,
            settings.whisper_compute_type,
            settings.language,
            settings.keep_numbers,
            process_data.process_data_paths.cache_folder_path,
            settings.skip_cache_transcription,
            model_weights,
            dominant_model
        )
        
        # Analyze transcription quality if debug level is high enough
        debug_level = getattr(settings, 'TRANSCRIPTION_DEBUG_LEVEL', 1)
        if debug_level >= 1:
            analyze_transcription_quality(results_by_model, transcription_result)
        
        # Store results in ProcessData
        if hasattr(process_data, 'transcription_results'):
            process_data.transcription_results = results_by_model
    else:
        # Use original single-track transcription
        transcription_result = transcribe_audio(
            process_data.process_data_paths.cache_folder_path,
            process_data.process_data_paths.processing_audio_path
        )

    # Update process_data with transcription result
    if process_data.media_info.language is None:
        process_data.media_info.language = transcription_result.detected_language

    process_data.transcribed_data = transcription_result.transcribed_data

    # Hyphen
    # Todo: Is it really unnecessary?
    remove_unecessary_punctuations(process_data.transcribed_data)
    if settings.hyphenation:
        hyphen_words = hyphenate_each_word(process_data.media_info.language, process_data.transcribed_data)

        if hyphen_words is not None:
            process_data.transcribed_data = add_hyphen_to_data(process_data.transcribed_data, hyphen_words)

    process_data.transcribed_data = remove_silence_from_transcription_data(
        process_data.process_data_paths.processing_audio_path, process_data.transcribed_data
    )


def CreateUltraStarTxt(process_data: ProcessData):
    # Move instrumental and vocals
    if settings.create_karaoke and version.parse(settings.format_version.value) < version.parse(
            FormatVersion.V1_1_0.value):
        karaoke_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Karaoke].mp3")
        convert_wav_to_mp3(process_data.process_data_paths.instrumental_audio_file_path, karaoke_output_path)

    if version.parse(settings.format_version.value) >= version.parse(FormatVersion.V1_1_0.value):
        instrumental_output_path = os.path.join(settings.output_folder_path,
                                                process_data.basename + " [Instrumental].mp3")
        convert_wav_to_mp3(process_data.process_data_paths.instrumental_audio_file_path, instrumental_output_path)
        vocals_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Vocals].mp3")
        convert_wav_to_mp3(process_data.process_data_paths.vocals_audio_file_path, vocals_output_path)

    # Apply note regularization if enabled (ADD THIS SECTION)
    if hasattr(settings, 'ENABLE_NOTE_REGULARIZATION') and settings.ENABLE_NOTE_REGULARIZATION:
        from modules.Pitcher.note_regularization import regularize_notes
        process_data.midi_segments = regularize_notes(
            process_data.midi_segments,
            process_data.media_info.bpm,
            process_data.transcribed_data,
            process_data.pitched_data,
            getattr(process_data, 'vad_results', None),
            settings
        )
    # Apply existing note optimizations if regularization is disabled but optimizations are enabled
    elif hasattr(settings, 'ENABLE_NOTE_OPTIMIZATIONS') and settings.ENABLE_NOTE_OPTIMIZATIONS:
        from modules.Ultrastar.note_processor import optimize_midi_segments
        process_data.midi_segments = optimize_midi_segments(
            process_data.midi_segments, 
            process_data.media_info.bpm
        )
    
    # Use VAD segmentation if available and enabled
    if (hasattr(settings, 'USE_VAD') and settings.USE_VAD and 
        hasattr(process_data, 'vad_results') and process_data.vad_results):
        from modules.Pitcher.vad_pitch_combiner import segment_notes_with_vad
        
        print(f"{ULTRASINGER_HEAD} Using {blue_highlighted('VAD-based note segmentation')} for improved timing")
        
        # Get the first VAD result to use for segmentation
        first_model = next(iter(process_data.vad_results.keys()))
        timestamps, vad_scores = process_data.vad_results[first_model]
        
        # Use VAD to create better note segments
        vad_midi_segments = segment_notes_with_vad(
            process_data.pitched_data,
            timestamps, 
            vad_scores,
            settings
        )
        
        # Merge with transcribed data if available
        if process_data.transcribed_data and len(process_data.transcribed_data) > 0:
            from modules.Midi.midi_creator import attach_lyrics_to_notes
            vad_midi_segments = attach_lyrics_to_notes(vad_midi_segments, process_data.transcribed_data)
            process_data.midi_segments = vad_midi_segments
    
    # Create Ultrastar txt
    if not settings.ignore_audio:
        ultrastar_file_output = create_ultrastar_txt_from_automation(
            process_data.basename,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.media_info,
            settings.format_version,
            settings.create_karaoke,
            settings.APP_VERSION,
            settings
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_midi_segments(
            settings.output_folder_path, settings.input_file_path, process_data.media_info.title,
            process_data.midi_segments
        )

    # Calc Points
    simple_score = None
    accurate_score = None
    if settings.calculate_score:
        simple_score, accurate_score = calculate_score_points(process_data, ultrastar_file_output)

    # Add calculated score to Ultrastar txt
    ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)
    return accurate_score, simple_score, ultrastar_file_output


def CreateProcessAudio(process_data) -> str:
    """Create process audio with optional multi-track extraction"""
    # Set processing audio to cache file
    process_data.process_data_paths.processing_audio_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + ".wav"
    )
    os_helper.create_folder(process_data.process_data_paths.cache_folder_path)

    # Extract vocals - using multi-track if enabled
    if hasattr(settings, 'USE_MULTI_TRACK_PROCESSING') and settings.USE_MULTI_TRACK_PROCESSING:
        # Multi-track processing
        vocal_track_folders = extract_multi_track_vocals(
            process_data.process_data_paths.cache_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            settings.use_separated_vocal,
            settings.create_karaoke,
            settings.pytorch_device,
            models=settings.MULTI_TRACK_MODELS if hasattr(settings, 'MULTI_TRACK_MODELS') else None,
            enabled=True,
            skip_cache=settings.skip_cache_vocal_separation
        )
        
        # Store vocal track paths for later use in pitch detection & transcription
        process_data.vocal_tracks = {}
        for model_name, folder_path in vocal_track_folders.items():
            process_data.vocal_tracks[model_name] = os.path.join(folder_path, "vocals.wav")
        
        # Use the default model's separation for the main processing path
        default_model = settings.demucs_model.value
        if default_model in vocal_track_folders:
            audio_separation_folder_path = vocal_track_folders[default_model]
        else:
            # Fall back to first available model
            first_model = next(iter(vocal_track_folders.keys()))
            audio_separation_folder_path = vocal_track_folders[first_model]
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning:')} Default model not available, using {blue_highlighted(first_model)} instead")
    else:
        # Original single-track approach
        audio_separation_folder_path = separate_vocal_from_audio(
            process_data.process_data_paths.cache_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            settings.use_separated_vocal,
            settings.create_karaoke,
            settings.pytorch_device,
            settings.demucs_model,
            settings.skip_cache_vocal_separation
        )
        # Store the single vocal track for consistency
        process_data.vocal_tracks = {settings.demucs_model.value: os.path.join(audio_separation_folder_path, "vocals.wav")}

    # Set paths for main processing
    process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_separation_folder_path, "vocals.wav")
    process_data.process_data_paths.instrumental_audio_file_path = os.path.join(audio_separation_folder_path, "no_vocals.wav")

    # Determine which audio to use for further processing
    if settings.use_separated_vocal:
        input_path = process_data.process_data_paths.vocals_audio_file_path
    else:
        input_path = process_data.process_data_paths.audio_output_file_path

    # Denoise vocal audio
    denoised_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_denoised.wav"
    )
    denoise_vocal_audio(input_path, denoised_output_path, settings.skip_cache_denoise_vocal_audio)

    # Convert to mono audio
    mono_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mono.wav"
    )
    convert_audio_to_mono_wav(denoised_output_path, mono_output_path)

    # Mute silence sections
    mute_output_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + "_mute.wav"
    )
    mute_no_singing_parts(mono_output_path, mute_output_path)

    # Define the audio file to process
    return mute_output_path


def transcribe_audio(cache_folder_path: str, processing_audio_path: str) -> TranscriptionResult:
    """Transcribe audio with AI"""
    transcription_result = None
    whisper_align_model_string = None
    if settings.transcriber == "whisper":
        if not settings.whisper_align_model is None: whisper_align_model_string = settings.whisper_align_model.replace("/", "_")
        transcription_config = f"{settings.transcriber}_{settings.whisper_model.value}_{settings.pytorch_device}_{whisper_align_model_string}_{settings.whisper_batch_size}_{settings.whisper_compute_type}_{settings.language}"
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        cached_transcription_available = check_file_exists(transcription_path)
        if settings.skip_cache_transcription or not cached_transcription_available:
            transcription_result = transcribe_with_whisper(
                processing_audio_path,
                settings.whisper_model,
                settings.pytorch_device,
                settings.whisper_align_model,
                settings.whisper_batch_size,
                settings.whisper_compute_type,
                settings.language,
                settings.keep_numbers,
            )
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path) as file:
                json = file.read()
                transcription_result = TranscriptionResult.from_json(json)
    else:
        raise NotImplementedError
    return transcription_result


def infos_from_audio_input_file() -> tuple[str, str, str, MediaInfo]:
    """Infos from audio input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]

    artist, title = None, None
    if " - " in basename_without_ext:
        artist, title = basename_without_ext.split(" - ", 1)
    else:
        title = basename_without_ext

    song_info = search_musicbrainz(title, artist)
    basename_without_ext = f"{song_info.artist} - {song_info.title}"
    extension = os.path.splitext(basename)[1]
    basename = f"{basename_without_ext}{extension}"

    song_folder_output_path = os.path.join(settings.output_folder_path, basename_without_ext)
    song_folder_output_path = get_unused_song_output_dir(song_folder_output_path)
    os_helper.create_folder(song_folder_output_path)
    os_helper.copy(settings.input_file_path, song_folder_output_path)
    os_helper.rename(
        os.path.join(song_folder_output_path, os.path.basename(settings.input_file_path)),
        os.path.join(song_folder_output_path, basename),
    )
    # Todo: Read ID3 tags
    if song_info.cover_image_data is not None:
        save_image(song_info.cover_image_data, basename_without_ext, song_folder_output_path)
    ultrastar_audio_input_path = os.path.join(song_folder_output_path, basename)
    real_bpm = get_bpm_from_file(settings.input_file_path)
    return (
        basename_without_ext,
        song_folder_output_path,
        ultrastar_audio_input_path,
        MediaInfo(artist=song_info.artist, title=song_info.title, year=song_info.year, genre=song_info.genres, bpm=real_bpm, cover_url=song_info.cover_url),
    )


def pitch_audio(
        process_data_paths: ProcessDataPaths, process_data=None) -> PitchedData:
    """Pitch audio with optional multi-track processing"""

    # Check if multi-track processing is enabled
    use_multi_track = (hasattr(settings, 'USE_MULTI_TRACK_PROCESSING') and 
                       hasattr(settings, 'USE_MULTI_TRACK_PITCH') and
                       settings.USE_MULTI_TRACK_PROCESSING and 
                       settings.USE_MULTI_TRACK_PITCH and
                       process_data is not None and
                       hasattr(process_data, 'vocal_tracks') and
                       len(process_data.vocal_tracks) > 1)
    
    # Check if VAD is enabled
    use_vad = (hasattr(settings, 'USE_VAD') and
               settings.USE_VAD and
               process_data is not None and
               hasattr(process_data, 'vocal_tracks'))
    
    if use_multi_track:
        # Use multi-track approach
        # Create a cache key based on all tracks and settings
        track_names = sorted(process_data.vocal_tracks.keys())
        track_str = '_'.join(track_names)
        vad_str = 'vad' if use_vad else 'novad'
        cache_key = f"multitrack_{track_str}_{settings.crepe_model_capacity}_{settings.crepe_step_size}_{settings.tensorflow_device}_{vad_str}"
        
        pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{cache_key}.json")
        cache_available = check_file_exists(pitched_data_path)
        
        if settings.skip_cache_pitch_detection or not cache_available:
            if use_vad:
                print(f"{ULTRASINGER_HEAD} Using {blue_highlighted('multi-track pitch detection with VAD')} with {len(process_data.vocal_tracks)} tracks")
            else:
                print(f"{ULTRASINGER_HEAD} Using {blue_highlighted('multi-track pitch detection')} with {len(process_data.vocal_tracks)} tracks")
            
            # Get debug settings
            debug_level = settings.DEBUG_LEVEL if hasattr(settings, 'DEBUG_LEVEL') else 1
            debug_dir = process_data_paths.cache_folder_path if debug_level >= 2 else None
            
            # Get confidence thresholds and agreement bonus
            confidence_thresholds = (settings.PITCH_CONFIDENCE_THRESHOLDS 
                                    if hasattr(settings, 'PITCH_CONFIDENCE_THRESHOLDS') else None)
            agreement_bonus = (settings.MULTI_TRACK_AGREEMENT_BONUS 
                              if hasattr(settings, 'MULTI_TRACK_AGREEMENT_BONUS') else 0.2)
            
            # Process all vocal tracks with optional VAD
            pitched_data = get_multi_track_pitch(
                vocal_tracks=process_data.vocal_tracks,
                model_capacity=settings.crepe_model_capacity,
                step_size=settings.crepe_step_size,
                device=settings.tensorflow_device,
                confidence_thresholds=confidence_thresholds,
                agreement_bonus=agreement_bonus,
                debug_level=debug_level,
                output_dir=debug_dir,
                vad_enabled=use_vad,
                settings=settings
            )
            
            # Store VAD results in process_data if available
            if use_vad and hasattr(pitched_data, 'vad_results'):
                process_data.vad_results = pitched_data.vad_results
            
            # Cache the result
            pitched_data_json = pitched_data.to_json()
            with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
                file.write(pitched_data_json)
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached multi-track pitch data")
            with open(pitched_data_path) as file:
                json = file.read()
                pitched_data = PitchedData.from_json(json)
    else:
        # Use single-track approach (original code)
        pitching_config = f"crepe_{settings.ignore_audio}_{settings.crepe_model_capacity}_{settings.crepe_step_size}_{settings.tensorflow_device}"
        pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{pitching_config}.json")
        cache_available = check_file_exists(pitched_data_path)

        if settings.skip_cache_pitch_detection or not cache_available:
            pitched_data = get_pitch_with_crepe_file(
                process_data_paths.processing_audio_path,
                settings.crepe_model_capacity,
                settings.crepe_step_size,
                settings.tensorflow_device,
            )

            # Apply VAD if enabled
            if use_vad and len(process_data.vocal_tracks) > 0:
                print(f"{ULTRASINGER_HEAD} Enhancing pitch data with {blue_highlighted('Voice Activity Detection')}")
                from modules.Pitcher.pitcher import process_with_vad
                pitched_data, vad_results = process_with_vad(
                    process_data.vocal_tracks, 
                    pitched_data, 
                    settings
                )
                
                # Store VAD results in process_data if available
                if vad_results:
                    process_data.vad_results = vad_results

            pitched_data_json = pitched_data.to_json()
            with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
                file.write(pitched_data_json)
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached pitch data")
            with open(pitched_data_path) as file:
                json = file.read()
                pitched_data = PitchedData.from_json(json)

    return pitched_data


def main(argv: list[str]) -> None:
    """Main function"""
    print_version(settings.APP_VERSION)
    init_settings(argv)
    check_requirements()
    if settings.interactive_mode:
        init_settings_interactive(settings)
    run()
    sys.exit()


def check_requirements() -> None:
    if not settings.force_cpu:
        settings.tensorflow_device, settings.pytorch_device = check_gpu_support()
    print(f"{ULTRASINGER_HEAD} ----------------------")

    if not is_ffmpeg_available(settings.user_ffmpeg_path):
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} {blue_highlighted('FFmpeg')} {red_highlighted('is not available. Provide --ffmpeg ‘path’ or install FFmpeg with PATH')}")
        sys.exit(1)
    else:
        ffmpeg_path, ffprobe_path = get_ffmpeg_and_ffprobe_paths()
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFmpeg')} - using {red_highlighted(ffmpeg_path)}")
        print(f"{ULTRASINGER_HEAD} {blue_highlighted('FFprobe')} - using {red_highlighted(ffprobe_path)}")

    print(f"{ULTRASINGER_HEAD} ----------------------")

def remove_cache_folder(cache_folder_path: str) -> None:
    """Remove cache folder"""
    os_helper.remove_folder(cache_folder_path)


def init_settings(argv: list[str]) -> Settings:
    """Init settings"""
    long, short = arg_options()
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_folder_path = arg
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"

            #Addition of whisper model choice. Added error handling for unknown models.
            try:
                settings.whisper_model = WhisperModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid whisper model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in WhisperModel]))}")
                sys.exit()
        elif opt in ("--whisper_align_model"):
            settings.whisper_align_model = arg
        elif opt in ("--whisper_batch_size"):
            settings.whisper_batch_size = int(arg)
        elif opt in ("--whisper_compute_type"):
            settings.whisper_compute_type = arg
        elif opt in ("--keep_numbers"):
            settings.keep_numbers = True
        elif opt in ("--language"):
            settings.language = arg
        elif opt in ("--crepe"):
            settings.crepe_model_capacity = arg
        elif opt in ("--crepe_step_size"):
            settings.crepe_step_size = int(arg)
        elif opt in ("--plot"):
            settings.create_plot = True
        elif opt in ("--midi"):
            settings.create_midi = True
        elif opt in ("--disable_hyphenation"):
            settings.hyphenation = False
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = False
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = False
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
        elif opt in ("--ignore_audio"):
            settings.ignore_audio = True
        elif opt in ("--force_cpu"):
            settings.force_cpu = True
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = True
        elif opt in ("--force_crepe_cpu"):
            settings.force_crepe_cpu = True
        elif opt in ("--format_version"):
            if arg == FormatVersion.V0_3_0.value:
                settings.format_version = FormatVersion.V0_3_0
            elif arg == FormatVersion.V1_0_0.value:
                settings.format_version = FormatVersion.V1_0_0
            elif arg == FormatVersion.V1_1_0.value:
                settings.format_version = FormatVersion.V1_1_0
            elif arg == FormatVersion.V1_2_0.value:
                settings.format_version = FormatVersion.V1_2_0
            else:
                print(
                    f"{ULTRASINGER_HEAD} {red_highlighted('Error: Format version')} {blue_highlighted(arg)} {red_highlighted('is not supported.')}"
                )
                sys.exit(1)
        elif opt in ("--keep_cache"):
            settings.keep_cache = True
        elif opt in ("--musescore_path"):
            settings.musescore_path = arg
        #Addition of demucs model choice. Work seems to be needed to make sure syntax is same for models. Added error handling for unknown models
        elif opt in ("--demucs"):
            try:
                settings.demucs_model = DemucsModel(arg)
            except ValueError as ve:
                print(f"{ULTRASINGER_HEAD} The model {arg} is not a valid demucs model selection. Please use one of the following models: {blue_highlighted(', '.join([m.value for m in DemucsModel]))}")
                sys.exit()
        elif opt in ("--cookiefile"):
            settings.cookiefile = arg
        elif opt in ("--interactive"):
            settings.interactive_mode = True
        elif opt in ("--ffmpeg"):
            settings.user_ffmpeg_path = arg
        elif opt in ("--disable_vad"):
            settings.USE_VAD = False
        elif opt in ("--vad_threshold"):
            settings.VAD_THRESHOLD = float(arg)
    if settings.output_folder_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_folder_path = os.path.join(dirname, "output")

    return settings


#For convenience, made True/False options into noargs
def arg_options():
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "crepe=",
        "crepe_step_size=",
        "demucs=",
        "whisper=",
        "whisper_align_model=",
        "whisper_batch_size=",
        "whisper_compute_type=",
        "language=",
        "plot",
        "midi",
        "disable_hyphenation",
        "disable_separation",
        "disable_karaoke",
        "create_audio_chunks",
        "ignore_audio",
        "force_cpu",
        "force_whisper_cpu",
        "force_crepe_cpu",
        "format_version=",
        "keep_cache",
        "musescore_path=",
        "keep_numbers",
        "interactive",
        "cookiefile=",
        "disable_vad",
        "vad_threshold=",
        "ffmpeg="
    ]
    return long, short

if __name__ == "__main__":
    main(sys.argv[1:])
