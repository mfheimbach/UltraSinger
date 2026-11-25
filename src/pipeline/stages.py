"""Stages wrapping UltraSinger's processing steps.

Artifact names used by the default presets:
  audio.mix, audio.vocals, audio.instrumental, audio.processed, audio.dry
  words, syllables, pitch, aux.game, notes, tempo, key, language, lyrics, media, file.txt

Where an older cache file of the pre-pipeline code exists (same folder, same naming), stages
reuse it, so existing caches keep working.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass

from pipeline.core import Context, Stage, stage


def _legacy():
    """The old monolith, for helpers not yet moved into modules (imported lazily: heavy)."""
    import UltraSinger
    return UltraSinger


def _read_json(path, cls):
    with open(path, encoding="utf-8") as f:
        return cls.from_json(f.read())


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        f.write(obj.to_json())


# ----------------------------------------------------------------------------- audio

@stage
class SeparateVocals(Stage):
    """mix -> vocals + instrumental (Demucs, or python-audio-separator models; optional 2nd pass)."""
    name = "separate"
    inputs = {"audio": "audio.mix"}
    outputs = {"vocals": "audio.vocals", "instrumental": "audio.instrumental"}

    @dataclass
    class Params:
        model: str = "htdemucs_ft"            # demucs model
        separator_model: str | None = None    # audio-separator model file (replaces demucs)
        second_pass: str | None = None        # lead-vocal / de-reverb model on the vocals

    def run(self, ctx, audio):
        from modules.Audio.separation import DemucsModel, separate_vocal_from_audio
        folder = separate_vocal_from_audio(ctx.cache_dir, audio, True, True, ctx.device,
                                           DemucsModel(self.params.model), False,
                                           self.params.separator_model, self.params.second_pass)
        return {"vocals": os.path.join(folder, "vocals.wav"),
                "instrumental": os.path.join(folder, "no_vocals.wav")}


@stage
class DeReverb(Stage):
    """audio -> de-reverberated audio (audio-separator de-reverb model)."""
    name = "dereverb"
    inputs = {"audio": "audio.vocals"}
    outputs = {"audio": "audio.dry"}

    @dataclass
    class Params:
        model: str = "deverb_bs_roformer_8_384dim_10depth.ckpt"

    def run(self, ctx, audio):
        from modules.Audio.separation import dry_vocals
        return {"audio": dry_vocals(audio, ctx.cache_dir, self.params.model)}


@stage
class PrepareVocals(Stage):
    """vocals -> denoised, mono, silent parts muted (the original 'processing audio')."""
    name = "prepare"
    inputs = {"audio": "audio.vocals"}
    outputs = {"audio": "audio.processed"}

    @dataclass
    class Params:
        denoise: bool = True
        mute_silence: bool = True

    def run(self, ctx, audio):
        from modules.Audio.convert_audio import convert_audio_to_mono_wav
        from modules.Audio.denoise import denoise_vocal_audio
        from modules.Audio.silence_processing import mute_no_singing_parts
        base = os.path.join(ctx.cache_dir, ctx.basename)
        src = audio
        if self.params.denoise:
            denoise_vocal_audio(src, base + "_denoised.wav")
            src = base + "_denoised.wav"
        convert_audio_to_mono_wav(src, base + "_mono.wav")
        src = base + "_mono.wav"
        if self.params.mute_silence:
            mute_no_singing_parts(src, base + "_mute.wav")
            src = base + "_mute.wav"
        return {"audio": src}


# ----------------------------------------------------------------------------- tempo / key

@stage
class Tempo(Stage):
    name = "tempo"
    inputs = {"audio": "audio.processed"}
    outputs = {"tempo": "tempo"}

    @dataclass
    class Params:
        detector: str = "librosa"   # librosa | beat_this
        scale: float = 1.0          # multiply (BPM jitter for evaluation)

    def run(self, ctx, audio):
        if self.params.detector == "beat_this":
            from modules.Audio.bpm import get_bpm_with_beat_this
            bpm = get_bpm_with_beat_this(audio, ctx.device)
        else:
            from modules.Audio.bpm import get_bpm_from_file
            bpm = get_bpm_from_file(audio)
        return {"tempo": bpm * self.params.scale}


@stage
class Key(Stage):
    name = "key"
    inputs = {"audio": "audio.processed"}
    outputs = {"key": "key"}

    def run(self, ctx, audio):
        from modules.Audio.key_detector import detect_key_from_audio
        return {"key": tuple(detect_key_from_audio(audio))}


# ----------------------------------------------------------------------------- words

@stage
class TranscribeWhisper(Stage):
    """audio -> words (+ detected language). Whisper via whisperX, wav2vec2 word timing."""
    name = "transcribe.whisper"
    version = "2"  # 2: timings rounded to 1 ms (fresh vs cached runs were not identical)
    inputs = {"audio": "audio.processed"}
    outputs = {"words": "words", "language": "language"}

    @dataclass
    class Params:
        model: str = "large-v3-turbo"
        batch_size: int = 4
        compute_type: str | None = None
        align_model: str | None = None
        language: str | None = None
        vad: str = "default"          # default | rms
        keep_numbers: bool = False

    def run(self, ctx, audio):
        from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
        from modules.Speech_Recognition.Whisper import WhisperModel, transcribe_with_whisper
        p = self.params
        # same file name as the pre-pipeline cache (reuse existing transcriptions)
        align = p.align_model.replace("/", "_") if p.align_model else None
        legacy = (f"whisper_{p.model}_{ctx.device}_{align}_{p.batch_size}_{p.compute_type}_{p.language}"
                  + (f"_vad{p.vad}" if p.vad != "default" else ""))
        path = os.path.join(os.path.dirname(audio), f"{legacy}.json")
        if os.path.exists(path) and os.path.dirname(audio) == ctx.cache_dir:
            result = _read_json(path, TranscriptionResult)
        else:
            result = transcribe_with_whisper(audio, WhisperModel(p.model), ctx.device, p.align_model, p.batch_size,
                                             p.compute_type, p.language, p.keep_numbers, p.vad)
            if os.path.dirname(audio) == ctx.cache_dir:
                _write_json(path, result)
        result.rounded()  # 1 ms: fresh (float32) and cached (float64) timings must be identical
        return {"words": result.transcribed_data, "language": p.language or result.detected_language}


@stage
class LyricsLookupLRCLIB(Stage):
    """media + audio duration -> candidate lyrics from lrclib.net."""
    name = "lyrics.lrclib"
    inputs = {"media": "media", "audio": "audio.processed"}
    outputs = {"lyrics": "lyrics"}

    def run(self, ctx, media, audio):
        import librosa
        from modules.Speech_Recognition import lyrics_lookup as ll
        return {"lyrics": ll.fetch_lyrics(media.artist or "", media.title or "", librosa.get_duration(path=audio))}


@stage
class CorrectWithLyrics(Stage):
    """words + lyrics -> words with mis-heard words replaced (timing kept). Text in, text out."""
    name = "words.correct_lyrics"
    inputs = {"words": "words", "lyrics": "lyrics"}
    outputs = {"words": "words"}

    @dataclass
    class Params:
        min_match: float = 0.4

    def run(self, ctx, words, lyrics):
        from modules.Speech_Recognition import lyrics_lookup as ll
        best, best_share = None, 0.0
        for cand in lyrics or []:
            corrected, share = ll.correct_words(words, cand["plain"], self.params.min_match)
            if corrected is not None and share > best_share:
                best, best_share = corrected, share
        if ctx.verbose:
            print(f"[pipeline] lyrics correction: {'%.0f%% matched' % (100 * best_share) if best else 'no match'}")
        return {"words": best if best is not None else words}


@stage
class StripPunctuation(Stage):
    """words -> words without '.' and ','. Text in, text out."""
    name = "words.strip_punctuation"
    inputs = {"words": "words"}
    outputs = {"words": "words"}
    cache = False

    def run(self, ctx, words):
        _legacy().remove_unecessary_punctuations(words)
        return {"words": words}


# ----------------------------------------------------------------------------- syllables

@stage
class Hyphenate(Stage):
    """words -> syllables (pyphen), timing split evenly over a word's syllables."""
    name = "syllables.hyphenate"
    inputs = {"words": "words", "language": "language"}
    outputs = {"syllables": "syllables"}

    @dataclass
    class Params:
        enabled: bool = True

    def run(self, ctx, words, language):
        from modules.Speech_Recognition.hyphenation import hyphenate_each_word
        if not self.params.enabled:
            return {"syllables": words}
        hyphen = hyphenate_each_word(language, words)
        return {"syllables": _legacy().add_hyphen_to_data(words, hyphen) if hyphen is not None else words}


@stage
class SilenceSplit(Stage):
    """syllables -> syllables split where the processed audio is muted."""
    name = "syllables.silence_split"
    inputs = {"syllables": "syllables", "audio": "audio.processed"}
    outputs = {"syllables": "syllables"}

    def run(self, ctx, syllables, audio):
        from modules.Audio.silence_processing import remove_silence_from_transcription_data
        return {"syllables": remove_silence_from_transcription_data(audio, syllables)}


# ----------------------------------------------------------------------------- pitch

@stage
class PitchRMVPE(Stage):
    name = "pitch.rmvpe"
    inputs = {"audio": "audio.vocals"}
    outputs = {"pitch": "pitch"}

    @dataclass
    class Params:
        shifts: str = "0"   # semitone passes, '/'-separated, median of the passes

    def run(self, ctx, audio):
        from modules.Pitcher.alt_pitchers import get_pitch_with_rmvpe
        shifts = tuple(float(x) for x in str(self.params.shifts).replace(",", "/").split("/") if x.strip()) or (0.0,)
        return {"pitch": get_pitch_with_rmvpe(audio, ctx.device, shifts)}


@stage
class PitchSwiftF0(Stage):
    name = "pitch.swiftf0"
    inputs = {"audio": "audio.processed"}
    outputs = {"pitch": "pitch"}

    def run(self, ctx, audio):
        from modules.Pitcher.pitcher import get_pitch_with_file
        return {"pitch": get_pitch_with_file(audio)}


# ----------------------------------------------------------------------------- segmentation

@stage
class SegmentGAME(Stage):
    """syllables -> syllables split at GAME note boundaries (+ GAME's note pitches in aux.game)."""
    name = "segment.game"
    version = "3"  # 2: timings rounded to 1 ms; 3: GAME's internal noise seeded (was random per run)
    inputs = {"syllables": "syllables", "audio": "audio.vocals", "language": "language"}
    outputs = {"syllables": "syllables", "game": "aux.game"}

    @dataclass
    class Params:
        model_dir: str = os.path.join(os.path.expanduser("~"), ".cache", "game", "GAME-1.0.3-small-onnx")
        boundary_threshold: float = 0.2
        steps: int = 8
        chunk_seconds: float = 30.0

    def run(self, ctx, syllables, audio, language):
        from modules.Midi.game_segmenter import game_notes_per_syllable, split_syllables_by_game
        p = self.params
        notes = game_notes_per_syllable(audio, syllables, language, p.model_dir, ctx.device,
                                        p.boundary_threshold, p.steps, p.chunk_seconds)
        notes_by_id = {id(t): [tuple(n) for n in notes.get(id(t), [])] for t in syllables}
        new_syllables, pitches = split_syllables_by_game(syllables, notes_by_id)
        return {"syllables": new_syllables, "game": pitches}


@stage
class SegmentContour(Stage):
    name = "segment.contour"
    inputs = {"syllables": "syllables", "pitch": "pitch"}
    outputs = {"syllables": "syllables"}

    @dataclass
    class Params:
        min_duration: float = 0.08
        smooth_frames: int = 5
        conf_threshold: float = 0.4
        tuning_correction: bool = True

    def run(self, ctx, syllables, pitch):
        p = self.params
        return {"syllables": _legacy().split_syllables_by_contour(syllables, pitch, p.min_duration, p.smooth_frames,
                                                                  p.conf_threshold, p.tuning_correction)}


@stage
class SegmentGrid(Stage):
    """The original 16th-note grid split."""
    name = "segment.grid"
    inputs = {"syllables": "syllables", "tempo": "tempo"}
    outputs = {"syllables": "syllables"}

    def run(self, ctx, syllables, tempo):
        return {"syllables": _legacy().split_syllables_into_segments(syllables, tempo)}


# ----------------------------------------------------------------------------- notes

@stage
class NotesFromPitch(Stage):
    """syllables + pitch track (+ key) -> notes, one per syllable piece."""
    name = "notes.from_pitch"
    inputs = {"syllables": "syllables", "pitch": "pitch", "key": "key"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        method: str = "chroma_mode"
        conf_threshold: float = 0.4
        edge_trim: float = 0.0
        tuning_correction: bool = True
        quantize_to_key: bool = True
        key_min_offset: float = 0.1

    def run(self, ctx, syllables, pitch, key):
        from modules.Audio.key_detector import get_allowed_notes_for_key
        from modules.Midi.midi_creator import NotePitchOptions, create_midi_segments_from_transcribed_data
        p = self.params
        options = NotePitchOptions(method=p.method, conf_threshold=p.conf_threshold, edge_trim=p.edge_trim,
                                   tuning_correction=p.tuning_correction, key_min_offset=p.key_min_offset)
        allowed = get_allowed_notes_for_key(*key) if p.quantize_to_key else None
        return {"notes": create_midi_segments_from_transcribed_data(syllables, pitch, allowed, options)}


@stage
class NotesGamePitch(Stage):
    """notes + GAME pitches -> notes with GAME's pitch (key-quantized like the others)."""
    name = "notes.game_pitch"
    inputs = {"notes": "notes", "game": "aux.game", "key": "key"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        quantize_to_key: bool = True
        key_min_offset: float = 0.1

    def run(self, ctx, notes, game, key):
        from modules.Audio.key_detector import get_allowed_notes_for_key
        us = _legacy()
        us.settings.key_quantize_min_offset = self.params.key_min_offset
        allowed = get_allowed_notes_for_key(*key) if self.params.quantize_to_key else None
        us.apply_game_pitches(notes, game, allowed)
        return {"notes": notes}


@stage
class MergeNotes(Stage):
    """notes + syllables -> notes with slides / same-pitch syllables merged (original logic)."""
    name = "notes.merge"
    inputs = {"notes": "notes", "syllables": "syllables", "tempo": "tempo"}
    outputs = {"notes": "notes", "syllables": "syllables"}

    @dataclass
    class Params:
        slides: bool = False
        same_pitch_syllables: bool = True

    def run(self, ctx, notes, syllables, tempo):
        us = _legacy()
        us.settings.merge_slides = self.params.slides
        us.settings.merge_same_pitch_syllables = self.params.same_pitch_syllables
        notes, syllables = us.merge_syllable_segments(notes, syllables, tempo)
        return {"notes": notes, "syllables": syllables}


@stage
class SnapEdges(Stage):
    """notes next to a pause snap to the pitch track's voicing on/offset."""
    name = "notes.snap_edges"
    inputs = {"notes": "notes", "pitch": "pitch"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        max_shift: float = 0.5
        conf: float = 0.1

    def run(self, ctx, notes, pitch):
        _legacy().snap_note_edges(notes, pitch, self.params.max_shift, self.params.conf)
        return {"notes": notes}


@stage
class Calibrate(Stage):
    """Shift notes earlier and extend their ends (charts sit ahead of the audible onset)."""
    name = "notes.calibrate"
    inputs = {"notes": "notes"}
    outputs = {"notes": "notes"}
    cache = False

    @dataclass
    class Params:
        offset: float = -0.07
        end_extension: float = 0.06

    def run(self, ctx, notes):
        _legacy().apply_timing_calibration(notes, self.params.offset, self.params.end_extension)
        return {"notes": notes}


@stage
class LineBreaks(Stage):
    """notes -> notes with planned line starts (pauses + punctuation model + length prior)."""
    name = "notes.line_breaks"
    inputs = {"notes": "notes"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        model: str = "oliverguhr/fullstop-punctuation-multilingual-base"

    def run(self, ctx, notes):
        from modules.Ultrastar.line_breaks import plan_line_breaks
        plan_line_breaks(notes, self.params.model, ctx.device)
        return {"notes": notes}


# ----------------------------------------------------------------------------- output

@stage
class WriteUltraStar(Stage):
    """notes + tempo + media -> UltraStar txt (and stems next to it)."""
    name = "write.ultrastar"
    inputs = {"notes": "notes", "tempo": "tempo", "media": "media", "language": "language", "key": "key",
              "vocals": "audio.vocals", "instrumental": "audio.instrumental"}
    outputs = {"txt": "file.txt"}
    cache = False

    @dataclass
    class Params:
        format_version: str = "1.2.0"
        stems: bool = True

    def run(self, ctx, notes, tempo, media, language, key, vocals, instrumental):
        from modules.Audio.convert_audio import convert_audio_format
        from modules.Ultrastar.coverter.ultrastar_txt_converter import create_ultrastar_txt_from_automation
        from modules.Ultrastar.ultrastar_txt import FormatVersion
        from Settings import Settings
        media = copy.copy(media)
        media.bpm, media.language, media.music_key = tempo, language, f"{key[0]} {key[1]}"
        os.makedirs(ctx.output_dir, exist_ok=True)
        if self.params.stems:
            ext = media.audio_extension
            convert_audio_format(instrumental, os.path.join(ctx.output_dir, f"{ctx.basename} [Instrumental].{ext}"))
            convert_audio_format(vocals, os.path.join(ctx.output_dir, f"{ctx.basename} [Vocals].{ext}"))
        txt = create_ultrastar_txt_from_automation(ctx.basename, ctx.output_dir, notes, media,
                                                   FormatVersion(self.params.format_version), True,
                                                   Settings.APP_VERSION)
        return {"txt": txt}
