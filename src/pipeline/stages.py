"""Stages wrapping UltraSinger's processing steps.

Artifact names used by the default presets:
  audio.mix, audio.vocals, audio.instrumental, audio.processed
  words, syllables, pitch, aux.game, notes, tempo, key, language, lyrics, media, file.txt

Logging: stages log through logging.getLogger("ultrasinger.<area>"); see pipeline/log.py.
"""

from __future__ import annotations

import copy
import dataclasses
import os
from dataclasses import dataclass

from pipeline.core import Stage, stage
from pipeline.log import get

log = get("stages")


# ----------------------------------------------------------------------------- audio

@stage
class SeparateVocals(Stage):
    """mix -> vocals + instrumental (Demucs, or python-audio-separator models; optional 2nd pass)."""
    name = "separate"
    version = "2"  # 2: demucs' random shift seeded (separations were not reproducible)
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    inputs = {"audio": "audio.mix"}
    outputs = {"vocals": "audio.vocals", "instrumental": "audio.instrumental"}

    @dataclass
    class Params:
        model: str            # demucs model
        separator_model: str | None    # audio-separator model file instead of demucs (e.g. new models)
        seed: int                         # demucs' random time shift; other seeds = alternative candidates

    def run(self, ctx, audio):
        from modules.Audio.separation import DemucsModel, separate_vocal_from_audio
        from pipeline.core import fingerprint
        p = self.params
        # the separation module caches by file name; a folder per input content (and seed) keeps it honest
        work = os.path.join(ctx.cache_dir, f"separation_{fingerprint(audio)[:8]}" + (f"_seed{p.seed}" if p.seed else ""))
        folder = separate_vocal_from_audio(work, audio, True, True, ctx.device, DemucsModel(p.model), False,
                                           p.separator_model, seed=p.seed)
        return {"vocals": os.path.join(folder, "vocals.wav"),
                "instrumental": os.path.join(folder, "no_vocals.wav")}


@stage
class PrepareVocals(Stage):
    """vocals -> denoised, mono, silent parts muted (the original 'processing audio')."""
    name = "prepare"
    inputs = {"audio": "audio.vocals"}
    outputs = {"audio": "audio.processed"}

    @dataclass
    class Params:
        denoise: bool
        denoise_reduction_db: float   # ffmpeg afftdn noise reduction
        denoise_floor_db: float       # ffmpeg afftdn noise floor
        mute_silence: bool
        silence_threshold_db: float   # quieter than this counts as silence (muted)
        min_silence_ms: int           # shortest silence that is muted

    def run(self, ctx, audio):
        from modules.Audio.convert_audio import convert_audio_to_mono_wav
        from modules.Audio.denoise import denoise_vocal_audio
        from modules.Audio.silence_processing import mute_no_singing_parts
        from pipeline.core import fingerprint
        # file names carry the hash of input and settings: runs with different separations or
        # settings share a cache folder, and the denoise module reuses files by name
        base = os.path.join(ctx.cache_dir, f"{ctx.basename}_{fingerprint([fingerprint(audio), dataclasses.asdict(self.params)])[:8]}")
        src = audio
        if self.params.denoise:
            denoise_vocal_audio(src, base + "_denoised.wav", self.params.denoise_reduction_db,
                                self.params.denoise_floor_db)
            src = base + "_denoised.wav"
        convert_audio_to_mono_wav(src, base + "_mono.wav")
        src = base + "_mono.wav"
        if self.params.mute_silence:
            mute_no_singing_parts(src, base + "_mute.wav", self.params.min_silence_ms,
                                  self.params.silence_threshold_db)
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
        scale: float          # multiply (BPM jitter for evaluation)

    def run(self, ctx, audio):
        from modules.Audio.bpm import get_bpm_from_file
        return {"tempo": get_bpm_from_file(audio) * self.params.scale}


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
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    version = "2"  # 2: timings rounded to 1 ms (fresh vs cached runs were not identical)
    inputs = {"audio": "audio.processed"}
    outputs = {"words": "words", "language": "language"}

    @dataclass
    class Params:
        model: str
        batch_size: int
        compute_type: str | None
        align_model: str | None
        language: str | None
        keep_numbers: bool

    def run(self, ctx, audio):
        from modules.Speech_Recognition.Whisper import WhisperModel, transcribe_with_whisper
        p = self.params
        result = transcribe_with_whisper(audio, WhisperModel(p.model), ctx.device, p.align_model, p.batch_size,
                                         p.compute_type, p.language, p.keep_numbers)
        result.rounded()  # 1 ms: fresh (float32) and cached (float64) timings must be identical
        return {"words": result.transcribed_data, "language": p.language or result.detected_language}


@stage
class LyricsLookupLRCLIB(Stage):
    """media + audio duration -> candidate lyrics from lrclib.net."""
    name = "lyrics.lrclib"
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    inputs = {"media": "media", "audio": "audio.processed"}
    outputs = {"lyrics": "lyrics"}

    @dataclass
    class Params:
        max_duration_diff: float   # seconds; LRCLIB entries of other versions are skipped

    def run(self, ctx, media, audio):
        import librosa
        from modules.Speech_Recognition import lyrics_lookup as ll
        return {"lyrics": ll.fetch_lyrics(media.artist or "", media.title or "", librosa.get_duration(path=audio),
                                          self.params.max_duration_diff)}


@stage
class CorrectWithLyrics(Stage):
    """words + lyrics -> words with mis-heard words replaced (timing kept). Text in, text out."""
    name = "words.correct_lyrics"
    inputs = {"words": "words", "lyrics": "lyrics"}
    outputs = {"words": "words"}

    @dataclass
    class Params:
        min_match: float          # lyrics with a smaller share of matching words are ignored
        max_length_ratio: float   # replacements longer/shorter than ratio x + 1 words keep the transcript

    def run(self, ctx, words, lyrics):
        from modules.Speech_Recognition import lyrics_lookup as ll
        best, best_share = None, 0.0
        for cand in lyrics or []:
            corrected, share = ll.correct_words(words, cand["plain"], self.params.min_match,
                                                self.params.max_length_ratio)
            if corrected is not None and share > best_share:
                best, best_share = corrected, share
        log.info("lyrics correction: " + (f"{best_share:.0%} of words matched" if best else "no matching lyrics"))
        return {"words": best if best is not None else words}


@stage
class StripPunctuation(Stage):
    """words -> words without '.' and ','. Text in, text out."""
    name = "words.strip_punctuation"
    inputs = {"words": "words|syllables"}   # works on whichever text level is current
    outputs = {"words": "@words"}
    cache = False

    def run(self, ctx, words):
        from modules.Speech_Recognition.text_ops import strip_punctuation
        strip_punctuation(words)
        return {"words": words}


@stage
class AlignText(Stage):
    """Known lyrics text (no timing) -> words aligned to the audio with wav2vec2.
    Oracle runs (chart lyrics) or user-supplied lyrics instead of transcription."""
    name = "words.align_text"
    inputs = {"audio": "audio.processed", "text": "lyrics.text"}
    outputs = {"words": "words", "language": "language"}
    hash_modules = False

    @dataclass
    class Params:
        language: str | None
        align_model: str | None

    def run(self, ctx, audio, text):
        from modules.Speech_Recognition.Whisper import align_lyrics_with_whisperx
        result = align_lyrics_with_whisperx(audio, text, self.params.language, ctx.device, self.params.align_model)
        result.rounded()
        return {"words": result.transcribed_data, "language": self.params.language}


@stage
class TranscribeQwen3(Stage):
    """audio -> words with Qwen3-ASR + Qwen3-ForcedAligner (needs `pip install qwen-asr`; GPU list)."""
    name = "transcribe.qwen3"
    inputs = {"audio": "audio.processed"}
    outputs = {"words": "words", "language": "language"}
    hash_modules = False

    @dataclass
    class Params:
        model: str
        aligner: str
        language: str | None

    def run(self, ctx, audio):
        from modules.Speech_Recognition.Qwen3ASR import transcribe_with_qwen3
        p = self.params
        result = transcribe_with_qwen3(audio, p.model, p.aligner, ctx.device, p.language)
        result.rounded()
        return {"words": result.transcribed_data, "language": p.language or result.detected_language}


# ----------------------------------------------------------------------------- syllables

@stage
class Hyphenate(Stage):
    """words -> syllables (pyphen), timing split evenly over a word's syllables."""
    name = "syllables.hyphenate"
    inputs = {"words": "words", "language": "language"}
    outputs = {"syllables": "syllables"}

    @dataclass
    class Params:
        enabled: bool

    def run(self, ctx, words, language):
        from modules.Speech_Recognition.hyphenation import hyphenate_each_word
        from modules.Speech_Recognition.text_ops import add_hyphen_to_data
        if not self.params.enabled:
            return {"syllables": words}
        hyphen = hyphenate_each_word(language, words)
        return {"syllables": add_hyphen_to_data(words, hyphen) if hyphen is not None else words}


@stage
class SilenceSplit(Stage):
    """syllables -> syllables split where the processed audio is muted."""
    name = "syllables.silence_split"
    inputs = {"syllables": "syllables|words", "audio": "audio.processed"}
    outputs = {"syllables": "@syllables"}

    @dataclass
    class Params:
        silence_threshold_db: float   # quieter than this counts as silence
        min_silence_ms: int           # shortest silence that splits a syllable

    def run(self, ctx, syllables, audio):
        from modules.Audio.silence_processing import remove_silence_from_transcription_data
        p = self.params
        return {"syllables": remove_silence_from_transcription_data(audio, syllables, p.min_silence_ms,
                                                                    p.silence_threshold_db)}


@stage
class SyllablesFromChart(Stage):
    """UltraStar txt -> its syllables with the chart's timing (+ tempo, language).
    Re-pitch runs: everything except the pitch comes from the chart."""
    name = "syllables.from_chart"
    inputs = {"chart": "chart"}
    outputs = {"syllables": "syllables", "tempo": "tempo", "language": "language"}

    @dataclass
    class Params:
        language: str | None   # default: from the chart's #LANGUAGE

    def run(self, ctx, chart):
        import re
        from modules.media_input import read_chart_headers
        from modules.Speech_Recognition.TranscribedData import TranscribedData
        from modules.Ultrastar.ultrastar_txt import FILE_ENCODING
        h = read_chart_headers(chart)
        bpm = float(h["BPM"].replace(",", "."))
        gap = float(h.get("GAP", "0").replace(",", ".")) / 1000.0
        beat = 60.0 / (bpm * 4)
        out = []
        with open(chart, encoding=FILE_ENCODING, errors="replace") as f:
            for line in f:
                if line.startswith(("P2", "P 2")):
                    break  # duets: first voice only
                m = re.match(r"([:*FRG]) +(-?\d+) +(\d+) +(-?\d+) ?(.*)$", line.rstrip("\r\n"))
                if not m:
                    continue
                start, dur, text = int(m.group(2)), int(m.group(3)), m.group(5)
                t = TranscribedData()
                t.word, t.start, t.end = text, round(gap + start * beat, 3), round(gap + (start + dur) * beat, 3)
                t.is_word_end, t.is_hyphen = text.endswith(" "), False
                out.append(t)
        language = self.params.language or (h.get("LANGUAGE") or "en")[:2].lower()
        # tempo only sets the writer's beat grid; the chart's own BPM keeps every note on a beat
        return {"syllables": out, "tempo": bpm, "language": language}


# ----------------------------------------------------------------------------- pitch

@stage
class PitchRMVPE(Stage):
    name = "pitch.rmvpe"
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    inputs = {"audio": "audio.vocals"}
    outputs = {"pitch": "pitch"}

    def run(self, ctx, audio):
        from modules.Pitcher.alt_pitchers import get_pitch_with_rmvpe
        return {"pitch": get_pitch_with_rmvpe(audio, ctx.device)}


@stage
class PitchSwiftF0(Stage):
    name = "pitch.swiftf0"
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
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
    inputs = {"syllables": "syllables|words", "audio": "audio.vocals", "language": "language"}
    outputs = {"syllables": "@syllables", "game": "aux.game"}

    @dataclass
    class Params:
        model_dir: str
        boundary_threshold: float
        steps: int
        chunk_seconds: float
        radius: int                  # boundary decoding radius (frames)
        presence_threshold: float    # note presence probability
        seed: int                    # GAME samples noise; other seeds = alternative candidates

    def run(self, ctx, syllables, audio, language):
        from modules.Midi.game_segmenter import game_notes_per_syllable, split_syllables_by_game
        p = self.params
        notes = game_notes_per_syllable(audio, syllables, language, os.path.expanduser(p.model_dir), ctx.device,
                                        p.boundary_threshold, p.steps, p.chunk_seconds, p.seed, p.radius,
                                        p.presence_threshold)
        notes_by_id = {id(t): [tuple(n) for n in notes.get(id(t), [])] for t in syllables}
        new_syllables, pitches = split_syllables_by_game(syllables, notes_by_id)
        return {"syllables": new_syllables, "game": pitches}


@stage
class SegmentContour(Stage):
    name = "segment.contour"
    inputs = {"syllables": "syllables|words", "pitch": "pitch"}
    outputs = {"syllables": "@syllables"}

    @dataclass
    class Params:
        min_duration: float
        smooth_frames: int
        conf_threshold: float
        tuning_correction: bool

    def run(self, ctx, syllables, pitch):
        from modules.Midi.segmentation import split_syllables_by_contour
        p = self.params
        return {"syllables": split_syllables_by_contour(syllables, pitch, p.min_duration, p.smooth_frames,
                                                        p.conf_threshold, p.tuning_correction)}


@stage
class SegmentGrid(Stage):
    """The original 16th-note grid split."""
    name = "segment.grid"
    inputs = {"syllables": "syllables|words", "tempo": "tempo"}
    outputs = {"syllables": "@syllables"}

    def run(self, ctx, syllables, tempo):
        from modules.Midi.segmentation import split_syllables_into_segments
        return {"syllables": split_syllables_into_segments(syllables, tempo)}


# ----------------------------------------------------------------------------- notes

@stage
class NotesFromPitch(Stage):
    """syllables + pitch track (+ key) -> notes, one per syllable piece."""
    name = "notes.from_pitch"
    inputs = {"syllables": "syllables|words", "pitch": "pitch", "key": "key"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        method: str
        conf_threshold: float
        edge_trim: float
        tuning_correction: bool
        quantize_to_key: bool
        key_min_offset: float

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
        quantize_to_key: bool
        key_min_offset: float

    def run(self, ctx, notes, game, key):
        from modules.Audio.key_detector import get_allowed_notes_for_key
        from modules.Midi.note_ops import apply_game_pitches
        allowed = get_allowed_notes_for_key(*key) if self.params.quantize_to_key else None
        apply_game_pitches(notes, game, allowed, self.params.key_min_offset)
        return {"notes": notes}


@stage
class MergeNotes(Stage):
    """notes + syllables -> notes with slides / same-pitch syllables merged (original logic)."""
    name = "notes.merge"
    inputs = {"notes": "notes", "syllables": "syllables|words", "tempo": "tempo"}
    outputs = {"notes": "notes", "syllables": "@syllables"}

    @dataclass
    class Params:
        slides: bool
        slide_max_semitones: int   # only with slides: neighbours this close merge into a slide
        same_pitch_syllables: bool

    def run(self, ctx, notes, syllables, tempo):
        from modules.Midi.note_ops import merge_syllable_segments
        p = self.params
        notes, syllables = merge_syllable_segments(notes, syllables, tempo, p.slides, p.same_pitch_syllables,
                                                   p.slide_max_semitones)
        return {"notes": notes, "syllables": syllables}


@stage
class SnapEdges(Stage):
    """notes next to a pause snap to the pitch track's voicing on/offset."""
    name = "notes.snap_edges"
    inputs = {"notes": "notes", "pitch": "pitch"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        max_shift: float   # seconds an edge may move
        conf: float        # pitch confidence that counts as voiced
        min_gap: float     # seconds kept free between notes
        min_len: float     # shortest note after snapping

    def run(self, ctx, notes, pitch):
        from modules.Midi.note_ops import snap_note_edges
        p = self.params
        snap_note_edges(notes, pitch, p.max_shift, p.conf, p.min_gap, p.min_len)
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
        offset: float
        end_extension: float

    def run(self, ctx, notes):
        from modules.Midi.note_ops import apply_timing_calibration
        apply_timing_calibration(notes, self.params.offset, self.params.end_extension)
        return {"notes": notes}


@stage
class LineBreaks(Stage):
    """notes -> notes with planned line starts (pauses + punctuation model + length prior)."""
    name = "notes.line_breaks"
    inputs = {"notes": "notes"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        model: str               # punctuation-restoration model
        intercept: float         # break score = intercept + w_pause * log(pause + 0.02) + w_punct * logit(p)
        w_pause: float
        w_punct: float
        chars_mean: float        # line length prior (characters), from the charts
        chars_sd: float
        chars_max: int
        dur_median: float        # line duration prior (seconds, log-normal)
        dur_log_sd: float
        prior_weight: float
        max_words: int

    def run(self, ctx, notes):
        from modules.Ultrastar.line_breaks import LineBreakModel, plan_line_breaks
        p = self.params
        model = LineBreakModel(p.intercept, p.w_pause, p.w_punct, p.chars_mean, p.chars_sd, p.chars_max,
                               p.dur_median, p.dur_log_sd, p.prior_weight, p.max_words)
        plan_line_breaks(notes, p.model, ctx.device, model)
        return {"notes": notes}


# ----------------------------------------------------------------------------- output

@stage
class WriteUltraStar(Stage):
    """notes + tempo + media -> UltraStar txt (and stems next to it)."""
    name = "write.ultrastar"
    inputs = {"notes": "notes", "tempo": "tempo", "media": "media", "language": "language", "key": "key",
              "vocals": "audio.vocals", "instrumental": "audio.instrumental"}
    outputs = {"txt": "file.txt"}
    optional_inputs = ("vocals", "instrumental")
    cache = False

    @dataclass
    class Params:
        format_version: str
        stems: bool

    def run(self, ctx, notes, tempo, media, language, key, vocals=None, instrumental=None):
        from modules.Audio.convert_audio import convert_audio_format
        from modules.Ultrastar.coverter.ultrastar_txt_converter import create_ultrastar_txt_from_automation
        from modules.Ultrastar.ultrastar_txt import FormatVersion
        from version import APP_VERSION
        media = copy.copy(media)
        media.bpm, media.language, media.music_key = tempo, language, f"{key[0]} {key[1]}"
        os.makedirs(ctx.output_dir, exist_ok=True)
        if self.params.stems and vocals and instrumental:
            ext = media.audio_extension
            convert_audio_format(instrumental, os.path.join(ctx.output_dir, f"{ctx.basename} [Instrumental].{ext}"))
            convert_audio_format(vocals, os.path.join(ctx.output_dir, f"{ctx.basename} [Vocals].{ext}"))
        txt = create_ultrastar_txt_from_automation(ctx.basename, ctx.output_dir, notes, media,
                                                   FormatVersion(self.params.format_version), True,
                                                   APP_VERSION)
        return {"txt": txt}


@stage
class WriteMidi(Stage):
    """notes + tempo -> MIDI file next to the chart."""
    name = "write.midi"
    inputs = {"notes": "notes", "tempo": "tempo"}
    outputs = {"midi": "file.midi"}
    cache = False

    def run(self, ctx, notes, tempo):
        from modules.Midi.midi_creator import create_midi_file
        os.makedirs(ctx.output_dir, exist_ok=True)
        create_midi_file(tempo, ctx.output_dir, notes, ctx.basename)
        return {"midi": os.path.join(ctx.output_dir, f"{ctx.basename}.mid")}


@stage
class WriteConventions(Stage):
    """file.txt -> the same chart with the community conventions players get from USDB Syncer:
    YASS line-break timing, typographic apostrophes, capitalised line starts, and optionally one
    free beat between touching notes (modules/Ultrastar/conventions.py).

    touching_notes is off by default: our note boundaries don't sit exactly on the charts', so
    cutting a beat off every touching note loses sung frames (SingStar bench usdx_medium
    0.609 -> 0.559, voicing recall 0.87 -> 0.79). The other rules don't change any score."""
    name = "write.conventions"
    inputs = {"txt": "file.txt"}
    outputs = {"txt": "file.txt"}
    cache = False

    @dataclass
    class Params:
        touching_notes: bool
        line_breaks: bool
        apostrophes: bool
        capitalise: bool

    def run(self, ctx, txt):
        from modules.Ultrastar.conventions import apply_conventions
        p = self.params
        apply_conventions(txt, p.touching_notes, p.line_breaks, p.apostrophes, p.capitalise)
        return {"txt": txt}
