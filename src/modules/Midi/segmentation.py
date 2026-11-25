"""Syllable segmentation into note pieces: 16th-note grid (original) and pitch contour."""

import copy

import librosa

from modules.Midi.note_length_calculator import get_sixteenth_note_second, get_thirtytwo_note_second
from modules.Pitcher.pitched_data import PitchedData
from modules.Speech_Recognition.TranscribedData import TranscribedData


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

