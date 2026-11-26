"""Stages wrapping UltraSinger's processing steps.

Artifact names used by the default presets:
  audio.mix, audio.vocals, audio.instrumental, audio.processed
  words, syllables, pitch, aux.game, notes, tempo, key, language, lyrics, media, file.txt

Logging: stages log through logging.getLogger("ultrasinger.<area>"); see pipeline/log.py.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
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
        separator_segment_size: int       # audio-separator (MDXC) segment size
        separator_overlap: int            # audio-separator (MDXC) overlap

    def run(self, ctx, audio):
        from modules.Audio.separation import DemucsModel, separate_vocal_from_audio
        from pipeline.core import fingerprint
        p = self.params
        # the separation module caches by file name; a folder per input content (and seed) keeps it honest
        work = os.path.join(ctx.cache_dir, f"separation_{fingerprint(audio)[:8]}" + (f"_seed{p.seed}" if p.seed else ""))
        folder = separate_vocal_from_audio(work, audio, True, True, ctx.device, DemucsModel(p.model), False,
                                           p.separator_model, None, p.seed, p.separator_segment_size,
                                           p.separator_overlap)
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

    @dataclass
    class Params:
        analysis_seconds: float    # audio from the start used for the key

    def run(self, ctx, audio):
        from modules.Audio.key_detector import detect_key_from_audio
        return {"key": tuple(detect_key_from_audio(audio, self.params.analysis_seconds))}


# ----------------------------------------------------------------------------- words

@stage
class TranscribeWhisper(Stage):
    """audio -> words (+ detected language). Whisper via whisperX, wav2vec2 word timing."""
    name = "transcribe.whisper"
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    version = "3"  # 2: timings rounded to 1 ms; 3: untimed words left None (words.place_untimed)
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
class PlaceUntimedWords(Stage):
    """Fallback: words the aligner left without timestamps (numbers, symbols) are placed right after
    the previous word. Does nothing when every word has timestamps; warns when it acts."""
    name = "words.place_untimed"
    inputs = {"words": "words"}
    outputs = {"words": "words"}

    @dataclass
    class Params:
        gap: float      # seconds after the previous word (or from 0 for the first word)
        length: float   # seconds

    def run(self, ctx, words):
        out, placed = [], []
        for w in words:
            if w.start is None or w.end is None:
                w = copy.copy(w)
                prev_end = out[-1].end if out else None
                w.start = 0.0 if prev_end is None else prev_end + self.params.gap
                w.end = self.params.length if prev_end is None else prev_end + (self.params.gap + self.params.length)
                placed.append(w.word.strip())
            out.append(w)
        if placed:
            logging.getLogger("ultrasinger.stages").warning(
                f"FALLBACK: placed {len(placed)} untimed word(s) after their predecessor: {placed[:5]}")
        return {"words": out}


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
        max_candidates: int        # lyrics versions kept for the correction step

    def run(self, ctx, media, audio):
        import librosa
        from modules.Speech_Recognition import lyrics_lookup as ll
        p = self.params
        return {"lyrics": ll.fetch_lyrics(media.artist or "", media.title or "", librosa.get_duration(path=audio),
                                          p.max_duration_diff, p.max_candidates)}


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
        p = self.params
        result = align_lyrics_with_whisperx(audio, text, p.language, ctx.device, p.align_model)
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
        batch_size: int
        max_align_seconds: float   # audio per alignment call (the aligner takes at most 180 s)

    def run(self, ctx, audio):
        from modules.Speech_Recognition.Qwen3ASR import transcribe_with_qwen3
        p = self.params
        result = transcribe_with_qwen3(audio, p.model, p.aligner, ctx.device, p.language, p.batch_size,
                                       p.max_align_seconds)
        result.rounded()
        return {"words": result.transcribed_data, "language": p.language or result.detected_language}


# ----------------------------------------------------------------------------- syllables

@stage
class Hyphenate(Stage):
    """words -> syllables (pyhyphen or pyphen), timing split evenly over a word's syllables."""
    name = "syllables.hyphenate"
    inputs = {"words": "words", "language": "language"}
    outputs = {"syllables": "syllables"}

    @dataclass
    class Params:
        enabled: bool
        backend: str      # pyhyphen (downloads LibreOffice dicts; fr broken upstream) | pyphen (bundled)

    def run(self, ctx, words, language):
        from modules.Speech_Recognition.hyphenation import hyphenate_each_word
        from modules.Speech_Recognition.text_ops import add_hyphen_to_data
        if not self.params.enabled:
            return {"syllables": words}
        hyphen = hyphenate_each_word(language, words, self.params.backend)
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
        min_piece: float              # pieces up to this long (s) are not kept on their own

    def run(self, ctx, syllables, audio):
        from modules.Audio.silence_processing import remove_silence_from_transcription_data
        p = self.params
        return {"syllables": remove_silence_from_transcription_data(audio, syllables, p.min_silence_ms,
                                                                    p.silence_threshold_db, p.min_piece)}


@stage
class TrimOverlaps(Stage):
    """Syllables whose end runs past the next start are cut back to it: onsets are the reliable edge
    (bench: median onset error 46 ms, ends 115 ms and biased late). Does nothing without overlaps."""
    name = "syllables.trim_overlaps"
    inputs = {"syllables": "syllables|words"}
    outputs = {"syllables": "@syllables"}

    def run(self, ctx, syllables):
        from modules.Midi.note_ops import trim_overlaps
        out = trim_overlaps(syllables)
        cut = sum(1 for a, b in zip(syllables, out) if a.end != b.end)
        if cut:
            logging.getLogger("ultrasinger.stages").info(f"trimmed {cut} overlapping syllable end(s)")
        return {"syllables": out}


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
        language = self.params.language or (h.get("LANGUAGE") or "")[:2].lower()
        if not language:
            logging.getLogger("ultrasinger.stages").warning(
                "FALLBACK: chart has no #LANGUAGE and syllables.from_chart.language is not set; assuming 'en'")
            language = "en"
        # tempo only sets the writer's beat grid; the chart's own BPM keeps every note on a beat
        return {"syllables": out, "tempo": bpm, "language": language}


# ----------------------------------------------------------------------------- pitch

@stage
class PitchRMVPE(Stage):
    name = "pitch.rmvpe"
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    inputs = {"audio": "audio.vocals"}
    outputs = {"pitch": "pitch"}

    @dataclass
    class Params:
        decode_threshold: float    # frames with peak salience at or below this are unvoiced

    def run(self, ctx, audio):
        from modules.Pitcher.alt_pitchers import get_pitch_with_rmvpe
        return {"pitch": get_pitch_with_rmvpe(audio, ctx.device, self.params.decode_threshold)}


@stage
class PitchSwiftF0(Stage):
    name = "pitch.swiftf0"
    hash_modules = False  # expensive: keep the cache across unrelated edits, bump version instead
    inputs = {"audio": "audio.processed"}
    outputs = {"pitch": "pitch"}

    @dataclass
    class Params:
        fmin: float
        fmax: float
        confidence_threshold: float

    def run(self, ctx, audio):
        from modules.Pitcher.pitcher import get_pitch_with_file
        p = self.params
        return {"pitch": get_pitch_with_file(audio, p.fmin, p.fmax, p.confidence_threshold)}


# ----------------------------------------------------------------------------- segmentation

@stage
class SegmentGAME(Stage):
    """syllables -> syllables split at GAME note boundaries (+ GAME's note pitches in aux.game)."""
    name = "segment.game"
    version = "3"  # 2: timings rounded to 1 ms; 3: GAME's internal noise seeded (overlaps: syllables.trim_overlaps)
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
        tuning_min_frames: int     # confident frames needed to estimate the tuning offset
        min_voiced_frames: int     # syllables with fewer voiced frames are not split

    def run(self, ctx, syllables, pitch):
        from modules.Midi.segmentation import split_syllables_by_contour
        p = self.params
        return {"syllables": split_syllables_by_contour(syllables, pitch, p.min_duration, p.smooth_frames,
                                                        p.conf_threshold, p.tuning_correction, p.tuning_min_frames,
                                                        p.min_voiced_frames)}


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
    inputs = {"syllables": "syllables|words", "pitch": "pitch"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        method: str
        conf_threshold: float
        edge_trim: float
        tuning_correction: bool
        tuning_min_frames: int     # confident frames needed to estimate the tuning offset
        fmin: float                # frames outside fmin..fmax (Hz) are ignored
        fmax: float

    def run(self, ctx, syllables, pitch):
        from modules.Midi.note_pitch import NotePitchOptions, notes_from_pitch
        p = self.params
        options = NotePitchOptions(method=p.method, conf_threshold=p.conf_threshold, edge_trim=p.edge_trim,
                                   tuning_correction=p.tuning_correction, tuning_min_frames=p.tuning_min_frames,
                                   fmin=p.fmin, fmax=p.fmax)
        return {"notes": notes_from_pitch(syllables, pitch, options)}


@stage
class NotesGamePitch(Stage):
    """notes + GAME pitches -> notes with GAME's pitch (key-quantized like the others)."""
    name = "notes.game_pitch"
    inputs = {"notes": "notes", "game": "aux.game"}
    outputs = {"notes": "notes"}

    def run(self, ctx, notes, game):
        from modules.Midi.note_ops import apply_game_pitches
        apply_game_pitches(notes, game)
        return {"notes": notes}


@stage
class QuantizeKey(Stage):
    """notes + key -> out-of-key notes moved into the key. One step after the last step that sets
    pitches, so the setting acts on the final pitch (it used to live, twice, in from_pitch and
    game_pitch, where game_pitch masked from_pitch's copy)."""
    name = "notes.quantize_key"
    inputs = {"notes": "notes", "key": "key"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        min_offset: float   # move a note only if its unrounded pitch is at least this far from the
                            # note's centre (0 = always): a clean chromatic note stays

    def run(self, ctx, notes, key):
        from modules.Midi.key_quantize import get_allowed_notes_for_key
        from modules.Midi.note_ops import quantize_notes
        n = quantize_notes(notes, get_allowed_notes_for_key(*key), self.params.min_offset)
        logging.getLogger("ultrasinger.stages").info(f"key quantization moved {n} of {len(notes)} notes")
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
        slide_max_sixteenths: float    # only with slides: pieces up to this long count as slides
        breath_pause_sixteenths: float  # a gap longer than this keeps same-pitch syllables apart
        same_pitch_syllables: bool

    def run(self, ctx, notes, syllables, tempo):
        from modules.Midi.note_ops import merge_syllable_segments
        p = self.params
        notes, syllables = merge_syllable_segments(notes, syllables, tempo, p.slides, p.same_pitch_syllables,
                                                   p.slide_max_semitones, p.slide_max_sixteenths,
                                                   p.breath_pause_sixteenths)
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
        min_len: float     # shortest note (s); the next onset still wins

    def run(self, ctx, notes):
        from modules.Midi.note_ops import apply_timing_calibration
        p = self.params
        apply_timing_calibration(notes, p.offset, p.end_extension, p.min_len)
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
        intercept: float         # break score = intercept + w_pause * log(pause + pause_floor) + w_punct * logit(p)
        w_pause: float
        w_punct: float
        chars_mean: float        # line length prior (characters), from the charts
        chars_sd: float
        chars_max: int
        dur_median: float        # line duration prior (seconds, log-normal)
        dur_log_sd: float
        prior_weight: float
        max_words: int
        pause_floor: float       # s added to pauses before the log
        prob_floor: float        # punctuation probabilities clipped to [prob_floor, 1 - prob_floor]
        dur_floor: float         # shortest line duration (s) for the prior
        punct_chunk: int         # words per punctuation-model call
        punct_context: int       # words of context on each side of a chunk

    def run(self, ctx, notes):
        from modules.Ultrastar.line_breaks import LineBreakModel, plan_line_breaks
        p = self.params
        model = LineBreakModel(**{k: v for k, v in vars(p).items() if k != "model"})
        plan_line_breaks(notes, p.model, ctx.device, model)
        return {"notes": notes}


@stage
class PauseBreaks(Stage):
    """notes -> notes with line starts from the original pause rule (no model; alternative to
    notes.line_breaks)."""
    name = "notes.pause_breaks"
    inputs = {"notes": "notes"}
    outputs = {"notes": "notes"}

    @dataclass
    class Params:
        min_gaps: int     # with fewer gaps between notes every gap breaks

    def run(self, ctx, notes):
        from modules.Ultrastar.line_breaks import plan_pause_breaks
        plan_pause_breaks(notes, self.params.min_gaps)
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
        max_bpm: float    # the chart BPM is multiplied up to just below this (finer beat grid)

    def run(self, ctx, notes, tempo, media, language, key, vocals=None, instrumental=None):
        from modules.Audio.convert_audio import convert_audio_format
        from modules.Ultrastar.writer import write_chart
        from modules.Ultrastar.ultrastar_txt import FormatVersion
        from version import APP_VERSION
        media = copy.copy(media)
        media.bpm, media.language, media.music_key = tempo, language, f"{key[0]} {key[1]}"
        os.makedirs(ctx.output_dir, exist_ok=True)
        if self.params.stems and vocals and instrumental:
            ext = media.audio_extension
            convert_audio_format(instrumental, os.path.join(ctx.output_dir, f"{ctx.basename} [Instrumental].{ext}"))
            convert_audio_format(vocals, os.path.join(ctx.output_dir, f"{ctx.basename} [Vocals].{ext}"))
        txt = write_chart(ctx.basename, ctx.output_dir, notes, media,
                                                   FormatVersion(self.params.format_version), True,
                                                   APP_VERSION, self.params.max_bpm)
        return {"txt": txt}


@stage
class WriteMidi(Stage):
    """notes + tempo -> MIDI file next to the chart."""
    name = "write.midi"
    inputs = {"notes": "notes", "tempo": "tempo"}
    outputs = {"midi": "file.midi"}
    cache = False

    @dataclass
    class Params:
        velocity: int

    def run(self, ctx, notes, tempo):
        from modules.Midi.midi_writer import write_midi_file
        os.makedirs(ctx.output_dir, exist_ok=True)
        return {"midi": write_midi_file(tempo, ctx.output_dir, notes, ctx.basename, self.params.velocity)}


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
        break_long_gap_s: float      # line-break position rules (see conventions.fix_line_break_timing)
        break_long_offset_s: float
        break_mid_gap_s: float
        break_mid_offset_s: float
        break_touch_beats: int
        break_beat_rules: list       # [[max_gap_beats, beats_before_next_line], ...]
        break_far_offset_beats: int

    def run(self, ctx, txt):
        from modules.Ultrastar.conventions import apply_conventions
        p = self.params
        rules = {"long_gap_s": p.break_long_gap_s, "long_offset_s": p.break_long_offset_s,
                 "mid_gap_s": p.break_mid_gap_s, "mid_offset_s": p.break_mid_offset_s,
                 "touch_beats": p.break_touch_beats, "beat_rules": p.break_beat_rules,
                 "far_offset_beats": p.break_far_offset_beats}
        apply_conventions(txt, p.touching_notes, p.line_breaks, p.apostrophes, p.capitalise, rules)
        return {"txt": txt}
