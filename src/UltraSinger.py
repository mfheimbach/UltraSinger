"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import json
import getopt
import os
import re
import sys
import Levenshtein
import librosa
import numpy as np

from packaging import version

from modules import os_helper
from modules.init_interactive_mode import init_settings_interactive
from modules.Audio.denoise import denoise_vocal_audio
from modules.Audio.separation import separate_vocal_from_audio
from modules.Audio.vocal_chunks import (
    create_audio_chunks_from_transcribed_data,
    create_audio_chunks_from_ultrastar_data,
)
from modules.Audio.key_detector import detect_key_from_audio, get_allowed_notes_for_key
from modules.Audio.silence_processing import remove_silence_from_transcription_data, mute_no_singing_parts
from modules.Audio.separation import DemucsModel
from modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_audio_format
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
    create_repitched_midi_segments_from_ultrastar_txt,
    NotePitchOptions,
    create_midi_file,
)
from modules.Midi.MidiSegment import MidiSegment
from modules.Midi.note_length_calculator import get_thirtytwo_note_second, get_sixteenth_note_second
from modules.Pitcher.pitcher import (
    get_pitch_with_file,
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
from modules.ffmpeg_helper import (
    is_ffmpeg_available,
    get_ffmpeg_and_ffprobe_paths,
    is_video_file,
    separate_audio_video,
)

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
    if settings.quantize_to_key:
        print(f"{ULTRASINGER_HEAD} {bright_green_highlighted('Option:')} {cyan_highlighted('Notes will be quantized to the detected musical key')}")

    process_data = InitProcessData()

    process_data.process_data_paths.cache_folder_path = (
        os.path.join(settings.output_folder_path, "cache")
        if settings.cache_override_path is None
        else settings.cache_override_path
    )

    # Create process audio
    process_data.process_data_paths.processing_audio_path = CreateProcessAudio(process_data)

    # Get BPM from wav file
    if not settings.input_file_is_ultrastar_txt:
        paths = process_data.process_data_paths
        bpm_input = {"mix": paths.audio_output_file_path,
                     "instrumental": paths.instrumental_audio_file_path}.get(settings.bpm_audio,
                                                                             paths.processing_audio_path)
        if settings.bpm_detector == "beat_this":
            from modules.Audio.bpm import get_bpm_with_beat_this
            process_data.media_info.bpm = get_bpm_with_beat_this(bpm_input, settings.pytorch_device)
        else:
            process_data.media_info.bpm = get_bpm_from_file(bpm_input)
        process_data.media_info.bpm *= settings.bpm_scale

    # Detect key
    detected_key, detected_mode = detect_key_from_audio(process_data.process_data_paths.processing_audio_path)
    if process_data.media_info.music_key is None:
        process_data.media_info.music_key = f"{detected_key} {detected_mode}"

    # Audio transcription
    process_data.media_info.language = settings.language
    if not settings.ignore_audio:
        TranscribeAudio(process_data)

    # Pitch audio (before splitting: contour segmentation needs it)
    process_data.pitched_data = pitch_audio(process_data.process_data_paths)

    # Split syllables into segments
    if not settings.ignore_audio:
        game_pitches = None
        if settings.note_segmentation == "game":
            process_data.transcribed_data, game_pitches = segment_with_game(process_data)
        elif settings.note_segmentation == "contour":
            process_data.transcribed_data = split_syllables_by_contour(
                process_data.transcribed_data, process_data.pitched_data,
                settings.contour_min_duration, settings.contour_smooth_frames,
                settings.note_pitch_conf_threshold, settings.note_pitch_tuning_correction)
        else:
            process_data.transcribed_data = split_syllables_into_segments(process_data.transcribed_data,
                                                                          process_data.media_info.bpm)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(process_data)

    note_pitch_options = NotePitchOptions(
        method=settings.note_pitch_method,
        conf_threshold=settings.note_pitch_conf_threshold,
        edge_trim=settings.note_pitch_edge_trim,
        tuning_correction=settings.note_pitch_tuning_correction,
        key_min_offset=settings.key_quantize_min_offset,
    )

    # Allowed keys for quantization
    allowed_notes_for_key = None
    if settings.quantize_to_key and not settings.ignore_audio:
        allowed_notes_for_key = get_allowed_notes_for_key(detected_key, detected_mode)

    # Create Midi_Segments
    if not settings.ignore_audio:
        process_data.midi_segments = create_midi_segments_from_transcribed_data(
            process_data.transcribed_data,
            process_data.pitched_data,
            allowed_notes_for_key,
            note_pitch_options
        )
    else:
        process_data.midi_segments = create_repitched_midi_segments_from_ultrastar_txt(process_data.pitched_data,
                                                                                       process_data.parsed_file,
                                                                                       note_pitch_options)

    # GAME's own note pitches (optional)
    if not settings.ignore_audio and settings.note_segmentation == "game" and settings.game_pitch and game_pitches:
        apply_game_pitches(process_data.midi_segments, game_pitches, allowed_notes_for_key)

    # Merge syllable segments
    if not settings.ignore_audio:
        process_data.midi_segments, process_data.transcribed_data = merge_syllable_segments(process_data.midi_segments,
                                                                                        process_data.transcribed_data,
                                                                                        process_data.media_info.bpm)

    # Note edges at pauses follow the voicing
    if not settings.ignore_audio and settings.note_edge_snap > 0:
        snap_note_edges(process_data.midi_segments, process_data.pitched_data,
                        settings.note_edge_snap, settings.note_edge_conf)

    # Timing calibration
    if settings.note_timing_offset or settings.note_end_extension:
        apply_timing_calibration(process_data.midi_segments, settings.note_timing_offset,
                                 settings.note_end_extension)

    # Line breaks planned from pauses and a punctuation model (after all timing changes)
    if not settings.ignore_audio and settings.line_breaks == "planned":
        from modules.Ultrastar.line_breaks import plan_line_breaks
        plan_line_breaks(process_data.midi_segments, settings.line_break_model,
                         "cpu" if settings.force_whisper_cpu else settings.pytorch_device)

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
    """Split every syllable into sub-segments

    This splits long syllables (including hyphenated ones) into smaller segments
    to allow detect for pitch changes within a syllable (e.g., singing a scale).
    """
    syllable_segment_size = get_sixteenth_note_second(real_bpm)
    thirtytwo_note = get_thirtytwo_note_second(real_bpm)

    segment_size_decimal_points = len(str(syllable_segment_size).split(".")[1])
    new_data = []

    for i, data in enumerate(transcribed_data):
        duration = data.end - data.start

        # If duration is less than or equal to a 16th note, don't split
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

        # Only add a partial_segment if it's at least as long as a 32nd note
        # Otherwise add it to the last note
        if partial_segment >= thirtytwo_note:
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
        elif full_segments >= 1 or len(new_data) > 0:
            # Add remaining time to the last note
            new_data[-1].end += partial_segment

        if has_space:
            new_data[-1].word += " "
            new_data[-1].is_word_end = True
    return new_data


def split_syllables_by_contour(
        transcribed_data: list[TranscribedData],
        pitched_data: PitchedData,
        min_duration: float = 0.08,
        smooth_frames: int = 5,
        conf_threshold: float = 0.4,
        tuning_correction: bool = True) -> list[TranscribedData]:
    """Split syllables where the sung pitch actually changes (alternative to the fixed 16th grid).

    Per syllable: smoothed semitone contour of the confident frames, run-length encoded;
    runs shorter than min_duration are merged into the neighbour with the closer pitch.
    Each remaining run becomes a sub-segment ("~" continuation like the grid split)."""
    import numpy as np
    from scipy.signal import medfilt
    from modules.Midi.midi_creator import estimate_tuning_offset

    times = np.asarray(pitched_data.times, dtype=float)
    freqs = np.asarray(pitched_data.frequencies, dtype=float)
    confs = np.asarray(pitched_data.confidence, dtype=float)
    tuning = estimate_tuning_offset(pitched_data, conf_threshold) if tuning_correction else 0.0
    kernel = smooth_frames if smooth_frames % 2 == 1 else smooth_frames + 1

    new_data = []
    for data in transcribed_data:
        a, b = np.searchsorted(times, [data.start, data.end])
        f, c = freqs[a:b], confs[a:b]
        voiced = (c > conf_threshold) & (f > 0)
        if data.end - data.start < 2 * min_duration or voiced.sum() < 3:
            new_data.append(data)
            continue
        m = np.full(f.shape, np.nan)
        m[voiced] = librosa.hz_to_midi(f[voiced]) - tuning
        idx = np.arange(len(m))
        m = np.interp(idx, idx[voiced], m[voiced])  # fill unvoiced gaps from neighbours
        if len(m) >= kernel:
            m = medfilt(m, kernel)
        q = np.round(m).astype(int)

        # run-length encode: [pitch, start_idx, end_idx)
        runs = []
        for i, v in enumerate(q):
            if runs and runs[-1][0] == v:
                runs[-1][2] = i + 1
            else:
                runs.append([v, i, i + 1])
        frame_dt = (data.end - data.start) / max(len(q), 1)
        while len(runs) > 1:
            durs = [(r[2] - r[1]) * frame_dt for r in runs]
            k = int(np.argmin(durs))
            if durs[k] >= min_duration:
                break
            if k == 0:
                j = 1
            elif k == len(runs) - 1:
                j = k - 1
            else:
                j = k - 1 if abs(runs[k - 1][0] - runs[k][0]) <= abs(runs[k + 1][0] - runs[k][0]) else k + 1
            lo, hi = min(k, j), max(k, j)
            runs[lo] = [runs[j][0], runs[lo][1], runs[hi][2]]
            del runs[hi]

        if len(runs) == 1:
            new_data.append(data)
            continue
        has_space = str(data.word).endswith(" ")
        for n, (_v, i0, i1) in enumerate(runs):
            seg = copy.deepcopy(data) if n == 0 else TranscribedData()
            seg.start = data.start + i0 * frame_dt
            seg.end = data.start + i1 * frame_dt if n < len(runs) - 1 else data.end
            seg.word = (data.word[:-1] if has_space else data.word) if n == 0 else "~"
            seg.is_hyphen = True
            seg.is_word_end = False
            new_data.append(seg)
        if has_space:
            new_data[-1].word += " "
            new_data[-1].is_word_end = True
    return new_data


def segment_with_game(process_data):
    """Split syllables at GAME's note boundaries (cached per syllable layout and settings)."""
    import hashlib
    from modules.Midi.game_segmenter import game_notes_per_syllable, split_syllables_by_game
    paths = process_data.process_data_paths
    audio = layer_audio(paths, settings.game_audio)
    syllables = process_data.transcribed_data
    key = json.dumps([[round(t.start, 3), round(t.end, 3)] for t in syllables]
                     + [settings.game_model_dir, settings.game_audio + (f"_{settings.dry_vocals_model}" if settings.game_audio == "dry" else ""),
                        settings.game_boundary_threshold,
                        settings.game_steps, process_data.media_info.language, "ms-rounded", "seed0"]
                     + ([settings.game_chunk_seconds] if settings.game_chunk_seconds != 30.0 else []))
    cache_path = os.path.join(paths.cache_folder_path, f"game_{hashlib.sha1(key.encode()).hexdigest()[:12]}.json")
    if check_file_exists(cache_path) and not settings.skip_cache_pitch_detection:
        with open(cache_path, encoding=FILE_ENCODING) as file:
            per_index = json.load(file)
    else:
        device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        notes = game_notes_per_syllable(audio, syllables, process_data.media_info.language,
                                        settings.game_model_dir, device,
                                        settings.game_boundary_threshold, settings.game_steps,
                                        settings.game_chunk_seconds)
        per_index = [notes.get(id(t), []) for t in syllables]
        with open(cache_path, "w", encoding=FILE_ENCODING) as file:
            json.dump(per_index, file)
    notes_by_id = {id(t): [tuple(n) for n in per_index[i]] for i, t in enumerate(syllables)}
    return split_syllables_by_game(syllables, notes_by_id)


def apply_game_pitches(midi_segments, game_pitches, allowed_notes):
    from modules.Audio.key_detector import quantize_note_to_key
    if len(midi_segments) != len(game_pitches):
        print(f"{ULTRASINGER_HEAD} {red_highlighted('GAME pitch count mismatch, keeping pitch-track notes')}")
        return
    for seg, midi in zip(midi_segments, game_pitches):
        if midi is None:
            continue
        note = librosa.midi_to_note(int(round(midi)), unicode=False)
        if allowed_notes is not None:
            note = quantize_note_to_key(note, allowed_notes, midi, settings.key_quantize_min_offset)
        seg.note = note


def snap_note_edges(midi_segments: list[MidiSegment], pitched_data: PitchedData, max_shift: float,
                    conf_threshold: float, min_gap: float = 0.03, min_len: float = 0.05) -> None:
    """Move note starts/ends that border a pause onto the voicing on/offset (within max_shift).
    Edges between touching notes are pitch changes and stay where they are."""
    times = np.asarray(pitched_data.times)
    voiced = (np.asarray(pitched_data.confidence) >= conf_threshold) & (np.asarray(pitched_data.frequencies) > 0)
    if len(times) < 2:
        return
    hop = float(np.median(np.diff(times)))

    def is_voiced(t):
        i = int(np.searchsorted(times, t))
        return 0 <= i < len(voiced) and voiced[i]

    for k, seg in enumerate(midi_segments):
        prev_end = midi_segments[k - 1].end if k > 0 else 0.0
        next_start = midi_segments[k + 1].start if k + 1 < len(midi_segments) else float("inf")
        if seg.start - prev_end > min_gap:
            t = seg.start
            if is_voiced(t):  # singing already started: move the start back to the onset
                while t - hop > max(prev_end + min_gap, seg.start - max_shift) and is_voiced(t - hop):
                    t -= hop
            else:  # nothing sung yet: move the start forward to the onset
                limit = min(seg.start + max_shift, seg.end - min_len)
                while t + hop < limit and not is_voiced(t):
                    t += hop
                if not is_voiced(t):
                    t = seg.start
            seg.start = t
        if next_start - seg.end > min_gap:
            t = seg.end
            if is_voiced(t):
                while t + hop < min(next_start - min_gap, seg.end + max_shift) and is_voiced(t + hop):
                    t += hop
            else:
                limit = max(seg.end - max_shift, seg.start + min_len)
                while t - hop > limit and not is_voiced(t):
                    t -= hop
                if not is_voiced(t):
                    t = seg.end
            seg.end = t


def apply_timing_calibration(midi_segments: list[MidiSegment], offset: float, end_extension: float) -> None:
    """Shift all notes by offset and lengthen them by end_extension, without overlaps."""
    for i, seg in enumerate(midi_segments):
        seg.start = max(0.0, seg.start + offset)
        end = seg.end + offset + end_extension
        if i + 1 < len(midi_segments):
            end = min(end, midi_segments[i + 1].start + offset)
        seg.end = max(end, seg.start + 0.01)


def merge_syllable_segments(midi_segments: list[MidiSegment],
                            transcribed_data: list[TranscribedData],
                            real_bpm: float) -> tuple[list[MidiSegment], list[TranscribedData]]:
    """Merge sub-segments of a syllable where the pitch is the same

    This function handles three cases:
    1. Merge consecutive ~ segments with the SAME pitch (same note held)
    2. Detect and merge SLIDES: short ~ segments with ±1-2 semitone jumps between syllables
    3. Merge ANY consecutive segments (including regular syllables) with the SAME pitch into one word
    """

    sixteenth_note = get_sixteenth_note_second(real_bpm)
    # Slides are typically very short (1-2 16th notes)
    max_slide_duration = sixteenth_note * 2

    new_data = []
    new_midi_notes = []

    previous_data = None

    for i, data in enumerate(transcribed_data):
        # Check if previous element exists
        is_same_note = i > 0 and midi_segments[i].note == midi_segments[i - 1].note
        has_breath_pause = False

        if previous_data is not None:
            has_breath_pause = (data.start - previous_data.end) > sixteenth_note

        # Check if current word is a ~ segment (continuation marker)
        current_word_stripped = str(data.word).strip()
        is_tilde_segment = current_word_stripped == "~"

        # Slide detection: Detect short ~ segments with small pitch jumps
        is_potential_slide = False
        if i > 0 and is_tilde_segment:
            duration = data.end - data.start

            # Calculate pitch jump in semitones
            try:
                prev_midi = librosa.note_to_midi(midi_segments[i - 1].note)
                curr_midi = librosa.note_to_midi(midi_segments[i].note)
                semitone_diff = abs(curr_midi - prev_midi)

                # Slide: Short duration AND small pitch jump (1-2 semitones)
                is_potential_slide = (settings.merge_slides and
                                     duration <= max_slide_duration and
                                     semitone_diff <= 2 and
                                     semitone_diff > 0)
            except:
                # Ignore slide detection on error
                pass

        # Check if current segment should be merged with previous due to same pitch
        should_merge_same_pitch = False
        if (settings.merge_same_pitch_syllables and
            i > 0 and
            previous_data is not None and
            not is_tilde_segment and  # Not a ~ segment
            is_same_note and
            not has_breath_pause and
            not previous_data.is_word_end):  # Don't merge across word boundaries
            should_merge_same_pitch = True

        should_merge_tilde = (is_tilde_segment and
                             previous_data is not None and
                             (is_same_note or is_potential_slide) and
                             not has_breath_pause)

        if should_merge_tilde:
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # For slides: Keep the original note (not the transition note)
            if is_potential_slide and not is_same_note:
                new_midi_notes[-1].note = midi_segments[i - 1].note

            # Take over space and word_end flag from current segment
            # "~ " means end of word - add space to previous segment
            if str(data.word).endswith(" "):
                if not new_data[-1].word.endswith(" "):
                    new_data[-1].word += " "
                if not new_midi_notes[-1].word.endswith(" "):
                    new_midi_notes[-1].word += " "
                new_data[-1].is_word_end = True

        elif should_merge_same_pitch:
            # Merge regular syllable with previous syllable (same pitch)
            new_data[-1].end = data.end
            new_midi_notes[-1].end = data.end

            # Check if current word has space at end before stripping
            has_space = str(data.word).endswith(" ")

            # Concatenate the words (remove trailing space from previous, add current word)
            prev_word = new_data[-1].word.rstrip()
            curr_word = data.word.rstrip()

            # Remove ~ from BOTH previous and current word if present
            if prev_word == "~":
                prev_word = ""
            if curr_word.startswith("~"):
                curr_word = curr_word[1:]

            # Concatenate
            new_data[-1].word = prev_word + curr_word
            new_midi_notes[-1].word = prev_word + curr_word

            # Preserve space at end if current segment had it
            if has_space:
                new_data[-1].word += " "
                new_midi_notes[-1].word += " "

            if data.is_word_end:
                new_data[-1].is_word_end = True
                new_data[-1].is_hyphen = False
            else:
                # Keep hyphen status if either segment was hyphenated
                new_data[-1].is_hyphen = new_data[-1].is_hyphen or data.is_hyphen

        else:
            # Add as new segment
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
            audio_extension,
        ) = parse_ultrastar_txt(settings.input_file_path, settings.output_folder_path)
        process_data = from_ultrastar_txt(ultrastar_class)
        process_data.basename = basename
        process_data.process_data_paths.audio_output_file_path = audio_file_path
        process_data.media_info.audio_extension = audio_extension
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
        # Audio/Video File
        print(f"{ULTRASINGER_HEAD} {gold_highlighted('Full Automatic Mode')}")
        process_data = ProcessData()
        (
            process_data.basename,
            settings.output_folder_path,
            process_data.process_data_paths.audio_output_file_path,
            process_data.media_info,
        ) = infos_from_audio_video_input_file()
    return process_data


def transcription_input_path(process_data) -> str:
    paths = process_data.process_data_paths
    if settings.transcription_audio == "mix":
        return paths.audio_output_file_path
    if settings.transcription_audio == "vocals":
        return paths.vocals_audio_file_path
    return paths.processing_audio_path


def realign_with_qwen3(cache_folder_path: str, audio_path: str, result: TranscriptionResult) -> TranscriptionResult:
    """Keep the transcript's words, take the timing from Qwen3-ForcedAligner."""
    import hashlib
    from modules.Speech_Recognition.Qwen3ASR import align_lyrics_with_qwen3, is_degenerate_alignment
    text = " ".join(t.word.strip() for t in result.transcribed_data if t.word.strip())
    if not text:
        return result
    language = settings.language or result.detected_language
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    cache_path = os.path.join(cache_folder_path, f"realign_qwen3_{language}_{digest}_{settings.transcription_audio}.json")
    if not settings.skip_cache_transcription and check_file_exists(cache_path):
        with open(cache_path, encoding=FILE_ENCODING) as file:
            realigned = TranscriptionResult.from_json(file.read())
    else:
        device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        realigned = align_lyrics_with_qwen3(audio_path, text, language, settings.qwen3_aligner, device)
        with open(cache_path, "w", encoding=FILE_ENCODING) as file:
            file.write(realigned.to_json())
    def span(words):
        return (words[-1].end - words[0].start) if words else 0.0

    old_span, new_span = span(result.transcribed_data), span(realigned.transcribed_data)
    squeezed = old_span > 0 and new_span < 0.7 * old_span
    if is_degenerate_alignment(realigned.transcribed_data) or squeezed:
        reason = "squeezed" if squeezed else "collapsed"
        print(f"{ULTRASINGER_HEAD} {red_highlighted(f'Qwen3 alignment {reason}')}, keeping wav2vec2 timing")
        return result
    return TranscriptionResult(realigned.transcribed_data, result.detected_language)


def lrclib_candidates(process_data) -> list[dict]:
    """LRCLIB lyrics for this song (looked up once)."""
    if getattr(process_data, "lrclib_candidates", None) is None:
        from modules.Speech_Recognition import lyrics_lookup as ll
        total = librosa.get_duration(path=transcription_input_path(process_data))
        process_data.lrclib_candidates = ll.fetch_lyrics(process_data.media_info.artist or "",
                                                         process_data.media_info.title or "", total)
    return process_data.lrclib_candidates


def correct_lyrics_with_lrclib(process_data, transcription_result):
    """Correct transcribed words with lyrics from LRCLIB, keeping the transcribed timing."""
    from modules.Speech_Recognition import lyrics_lookup as ll
    best, best_share = None, 0.0
    for candidate in lrclib_candidates(process_data):
        corrected, share = ll.correct_words(transcription_result.transcribed_data, candidate["plain"])
        if corrected is not None and share > best_share:
            best, best_share = corrected, share
    if best is None:
        print(f"{ULTRASINGER_HEAD} No matching lyrics on LRCLIB, keeping the transcription")
        return transcription_result
    print(f"{ULTRASINGER_HEAD} Lyrics corrected with {blue_highlighted('LRCLIB')} "
          f"({best_share:.0%} of transcribed words matched)")
    return TranscriptionResult(best, transcription_result.detected_language)


def TranscribeAudio(process_data):
    transcription_result = transcribe_audio(process_data.process_data_paths.cache_folder_path,
                                            transcription_input_path(process_data))
    if settings.word_aligner == "qwen3" and not settings.lyrics_file and settings.transcriber == "whisper":
        transcription_result = realign_with_qwen3(process_data.process_data_paths.cache_folder_path,
                                                  transcription_input_path(process_data), transcription_result)
    if settings.lyrics_source == "lrclib" and not settings.lyrics_file:
        transcription_result = correct_lyrics_with_lrclib(process_data, transcription_result)

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
        karaoke_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Karaoke]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, karaoke_output_path)

    if version.parse(settings.format_version.value) >= version.parse(FormatVersion.V1_1_0.value):
        instrumental_output_path = os.path.join(settings.output_folder_path,
                                                process_data.basename + " [Instrumental]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.instrumental_audio_file_path, instrumental_output_path)
        vocals_output_path = os.path.join(settings.output_folder_path, process_data.basename + " [Vocals]." + process_data.media_info.audio_extension)
        convert_audio_format(process_data.process_data_paths.vocals_audio_file_path, vocals_output_path)

    # Create Ultrastar txt
    if not settings.ignore_audio:
        ultrastar_file_output = create_ultrastar_txt_from_automation(
            process_data.basename,
            settings.output_folder_path,
            process_data.midi_segments,
            process_data.media_info,
            settings.format_version,
            settings.create_karaoke,
            settings.APP_VERSION
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
    #Todo: Missing Karaoke
    if simple_score is not None:
        ultrastar_writer.add_score_to_ultrastar_txt(ultrastar_file_output, simple_score)
    return accurate_score, simple_score, ultrastar_file_output


def CreateProcessAudio(process_data) -> str:
    # Set processing audio to cache file
    process_data.process_data_paths.processing_audio_path = os.path.join(
        process_data.process_data_paths.cache_folder_path, process_data.basename + ".wav"
    )
    os_helper.create_folder(process_data.process_data_paths.cache_folder_path)

    # Separate vocal from audio
    audio_separation_folder_path = separate_vocal_from_audio(
        process_data.process_data_paths.cache_folder_path,
        process_data.process_data_paths.audio_output_file_path,
        settings.use_separated_vocal,
        settings.create_karaoke,
        settings.pytorch_device,
        settings.demucs_model,
        settings.skip_cache_vocal_separation,
        settings.separator_model,
        settings.lead_vocal_model,
    )
    process_data.process_data_paths.vocals_audio_file_path = os.path.join(audio_separation_folder_path, "vocals.wav")
    process_data.process_data_paths.instrumental_audio_file_path = os.path.join(audio_separation_folder_path,
                                                                                "no_vocals.wav")

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
    if settings.lyrics_file:
        return align_known_lyrics(cache_folder_path, processing_audio_path)
    if settings.transcriber == "whisper":
        if not settings.whisper_align_model is None:
            whisper_align_model_string = settings.whisper_align_model.replace("/", "_")
        whisper_device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        transcription_config = f"{settings.transcriber}_{settings.whisper_model.value}_{whisper_device}_{whisper_align_model_string}_{settings.whisper_batch_size}_{settings.whisper_compute_type}_{settings.language}"
        if settings.transcription_audio != "processed":
            transcription_config += f"_{settings.transcription_audio}"
        if settings.whisper_vad != "default":
            transcription_config += f"_vad{settings.whisper_vad}"
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        cached_transcription_available = check_file_exists(transcription_path)
        if settings.skip_cache_transcription or not cached_transcription_available:
            transcription_result = transcribe_with_whisper(
                processing_audio_path,
                settings.whisper_model,
                whisper_device,
                settings.whisper_align_model,
                settings.whisper_batch_size,
                settings.whisper_compute_type,
                settings.language,
                settings.keep_numbers,
                settings.whisper_vad,
            )
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path) as file:
                json = file.read()
                transcription_result = TranscriptionResult.from_json(json)
    elif settings.transcriber == "qwen3":
        from modules.Speech_Recognition.Qwen3ASR import transcribe_with_qwen3
        device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
        transcription_config = (f"qwen3_{settings.qwen3_model.replace('/', '_')}_"
                                f"{settings.qwen3_aligner.replace('/', '_')}_{device}_{settings.language}"
                                f"_{settings.transcription_audio}")
        transcription_path = os.path.join(cache_folder_path, f"{transcription_config}.json")
        if settings.skip_cache_transcription or not check_file_exists(transcription_path):
            transcription_result = transcribe_with_qwen3(
                processing_audio_path,
                settings.qwen3_model,
                settings.qwen3_aligner,
                device,
                settings.language,
            )
            from modules.Speech_Recognition.Qwen3ASR import is_degenerate_alignment
            if is_degenerate_alignment(transcription_result.transcribed_data):
                # keep Qwen's words, redo the timing with wav2vec2
                from modules.Speech_Recognition.Whisper import align_lyrics_with_whisperx
                print(f"{ULTRASINGER_HEAD} {red_highlighted('Qwen3 alignment collapsed')}, re-aligning with wav2vec2")
                text = " ".join(t.word.strip() for t in transcription_result.transcribed_data)
                transcription_result = align_lyrics_with_whisperx(
                    processing_audio_path, text, settings.language or transcription_result.detected_language,
                    device, settings.whisper_align_model)
            with open(transcription_path, "w", encoding=FILE_ENCODING) as file:
                file.write(transcription_result.to_json())
        else:
            print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcribed data")
            with open(transcription_path, encoding=FILE_ENCODING) as file:
                transcription_result = TranscriptionResult.from_json(file.read())
    else:
        raise NotImplementedError
    return transcription_result.rounded()


def align_known_lyrics(cache_folder_path: str, processing_audio_path: str) -> TranscriptionResult:
    """Align a provided plain-text lyrics file instead of transcribing."""
    import hashlib
    if not settings.language:
        raise ValueError("lyrics_file needs language to be set (e.g. --language en)")
    with open(settings.lyrics_file, encoding=FILE_ENCODING) as file:
        lyrics = file.read()
    device = "cpu" if settings.force_whisper_cpu else settings.pytorch_device
    digest = hashlib.sha1(lyrics.encode("utf-8")).hexdigest()[:10]
    cache_path = os.path.join(cache_folder_path,
                              f"lyrics_{settings.lyrics_aligner}_{settings.language}_{digest}_{settings.transcription_audio}.json")
    if not settings.skip_cache_transcription and check_file_exists(cache_path):
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached lyrics alignment")
        with open(cache_path, encoding=FILE_ENCODING) as file:
            return TranscriptionResult.from_json(file.read())
    result = None
    if settings.lyrics_aligner == "qwen3":
        from modules.Speech_Recognition.Qwen3ASR import align_lyrics_with_qwen3, is_degenerate_alignment
        result = align_lyrics_with_qwen3(processing_audio_path, lyrics, settings.language, settings.qwen3_aligner, device)
        if is_degenerate_alignment(result.transcribed_data):
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Qwen3 alignment collapsed')}, falling back to wav2vec2")
            result = None
    if result is None:
        from modules.Speech_Recognition.Whisper import align_lyrics_with_whisperx
        result = align_lyrics_with_whisperx(processing_audio_path, lyrics, settings.language, device,
                                            settings.whisper_align_model)
    with open(cache_path, "w", encoding=FILE_ENCODING) as file:
        file.write(result.to_json())
    return result


def infos_from_audio_video_input_file() -> tuple[str, str, str, MediaInfo]:
    """Infos from audio/video input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]

    artist, title = None, None
    if " - " in basename_without_ext:
        artist, title = basename_without_ext.split(" - ", 1)
    else:
        title = basename_without_ext

    song_info = search_musicbrainz(title, artist)
    basename_without_ext = f"{song_info.artist} - {song_info.title}"

    song_folder_output_path = os.path.join(settings.output_folder_path, basename_without_ext)
    song_folder_output_path = get_unused_song_output_dir(song_folder_output_path)
    os_helper.create_folder(song_folder_output_path)

    extension = os.path.splitext(basename)[1]
    if is_video_file(settings.input_file_path):
        print(f"{ULTRASINGER_HEAD} Video file detected - separating audio and video")

        video_with_audio_basename = f"{basename_without_ext}{extension}"
        video_with_audio_path = os.path.join(song_folder_output_path, video_with_audio_basename)
        os_helper.copy(settings.input_file_path, video_with_audio_path)

        # Separate audio and video
        ultrastar_audio_input_path, final_video_path, audio_ext, video_ext = separate_audio_video(
            video_with_audio_path, basename_without_ext, song_folder_output_path
        )
    else:
        # Audio file
        basename_with_ext = f"{basename_without_ext}{extension}"
        audio_ext = extension.lstrip('.')
        video_ext = None
        os_helper.copy(settings.input_file_path, song_folder_output_path)
        os_helper.rename(
            os.path.join(song_folder_output_path, os.path.basename(settings.input_file_path)),
            os.path.join(song_folder_output_path, basename_with_ext),
        )
        ultrastar_audio_input_path = os.path.join(song_folder_output_path, basename_with_ext)

    # Todo: Read ID3 tags
    if song_info.cover_image_data is not None:
        save_image(song_info.cover_image_data, basename_without_ext, song_folder_output_path)

    return (
        basename_without_ext,
        song_folder_output_path,
        ultrastar_audio_input_path,
        MediaInfo(
            artist=song_info.artist,
            title=song_info.title,
            year=song_info.year,
            genre=song_info.genres,
            cover_url=song_info.cover_url,
            audio_extension=audio_ext,
            video_extension=video_ext
        ),
    )


def pitch_shift_passes() -> tuple[float, ...]:
    return tuple(float(x) for x in re.split(r"[,/;]", str(settings.pitch_shift_passes)) if x.strip()) or (0.0,)


def layer_audio(paths: ProcessDataPaths, which: str) -> str:
    """Audio layer for pitch/notes: mix | vocals | dry (de-reverberated vocals) | processed."""
    if which == "dry":
        from modules.Audio.separation import dry_vocals
        return dry_vocals(paths.vocals_audio_file_path, paths.cache_folder_path, settings.dry_vocals_model)
    return {"mix": paths.audio_output_file_path,
            "vocals": paths.vocals_audio_file_path}.get(which, paths.processing_audio_path)


def pitch_audio(
        process_data_paths: ProcessDataPaths) -> PitchedData:
    """Pitch audio"""

    pitching_config = f"swiftf0_{settings.ignore_audio}_sep{settings.use_separated_vocal}"
    if settings.pitch_detector != "swiftf0" or settings.pitch_audio != "processed":
        pitching_config = f"{settings.pitch_detector}_{settings.pitch_audio}_{settings.ignore_audio}_sep{settings.use_separated_vocal}"
        if settings.pitch_audio == "dry":
            pitching_config += f"_{os.path.splitext(settings.dry_vocals_model)[0]}"
        if pitch_shift_passes() != (0,):
            pitching_config += "_shifts" + "_".join(f"{x:g}" for x in pitch_shift_passes())
    pitched_data_path = os.path.join(process_data_paths.cache_folder_path, f"{pitching_config}.json")
    cache_available = check_file_exists(pitched_data_path)

    if settings.skip_cache_pitch_detection or not cache_available:
        audio_path = layer_audio(process_data_paths, settings.pitch_audio)
        if settings.pitch_detector == "rmvpe":
            from modules.Pitcher.alt_pitchers import get_pitch_with_rmvpe
            pitched_data = get_pitch_with_rmvpe(audio_path, settings.pytorch_device, pitch_shift_passes())
        elif settings.pitch_detector == "fcpe":
            from modules.Pitcher.alt_pitchers import get_pitch_with_fcpe
            pitched_data = get_pitch_with_fcpe(audio_path, settings.pytorch_device)
        else:
            pitched_data = get_pitch_with_file(audio_path)

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
        settings.pytorch_device = check_gpu_support()
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
        elif opt in ("--quantize_to_key"):
            settings.quantize_to_key = arg
        elif opt in ("--ffmpeg"):
            settings.user_ffmpeg_path = arg
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
        "quantize_to_key",
        "interactive",
        "cookiefile=",
        "ffmpeg="
    ]
    return long, short

if __name__ == "__main__":
    main(sys.argv[1:])
