"""UltraStar chart writer: notes (MidiSegments with line_start flags) + media info -> .txt.

Line breaks are written where the notes say (a line-break step decides them); the beat grid is the
chart BPM multiplied up to just below max_bpm.
"""

import logging
import os
import re

import langcodes
import librosa
from packaging import version

from modules.Midi.MidiSegment import MidiSegment
from modules.media_info import MediaInfo
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING, FormatVersion, UltrastarTxtNoteTypeTag, UltrastarTxtTag, \
    UltrastarTxtValue
from modules.console_colors import blue_highlighted

log = logging.getLogger("ultrasinger.ultrastar_writer")

BEATS_PER_QUARTER = 4     # format: the UltraStar BPM counts quarter beats of the real tempo
ULTRASTAR_C4_MIDI = 48    # format: UltraStar pitch 0 = MIDI 48


def second_to_beat(seconds: float, real_bpm: float) -> float:
    return seconds * real_bpm / 60


def get_multiplier(ultrastar_bpm: float, max_bpm: float) -> int:
    """Largest integer multiplier keeping the chart BPM below max_bpm (finer beat grid), at least 1."""
    if ultrastar_bpm == 0:
        raise ValueError("BPM is 0")
    multiplier = 1
    while ultrastar_bpm * (multiplier + 1) < max_bpm:
        multiplier += 1
    return multiplier


def get_language_name(language: str) -> str:
    return langcodes.Language.make(language=language).display_name()


def write_chart(
        basename: str,
        song_folder_output_path: str,
        midi_segments: list[MidiSegment],
        media_info: MediaInfo,
        format_version: FormatVersion,
        create_karaoke: bool,
        app_version: str,
        max_bpm: float,
):
    """Notes + media info -> <basename>.txt in song_folder_output_path (plus a [Karaoke] copy for
    format versions before 1.1.0). Returns the chart's path."""
    log.info(f"Using UltraStar {blue_highlighted(f'Format Version {format_version.value}')}")

    ultrastar_txt = UltrastarTxtValue()
    ultrastar_txt.version = format_version.value
    if media_info.audio_extension is None:
        raise Exception("Missing Audio extension. It is required to create Ultrastar txt")
    ultrastar_txt.mp3 = basename + "." + media_info.audio_extension
    ultrastar_txt.audio = basename + "." + media_info.audio_extension
    ultrastar_txt.vocals = basename + " [Vocals]." + media_info.audio_extension
    ultrastar_txt.instrumental = basename + " [Instrumental]." + media_info.audio_extension
    if media_info.video_extension is not None:
        ultrastar_txt.video = basename + "." + media_info.video_extension
    ultrastar_txt.language = media_info.language
    cover = basename + " [CO].jpg"
    ultrastar_txt.cover = (
        cover if os.path.isfile(os.path.join(song_folder_output_path, cover)) else None
    )
    ultrastar_txt.creator = f"{ultrastar_txt.creator} {app_version}"
    ultrastar_txt.comment = f"{ultrastar_txt.comment} {app_version}"

    # Additional data
    ultrastar_txt.title = media_info.title
    ultrastar_txt.artist = media_info.artist
    if media_info.year is not None:
        ultrastar_txt.year = extract_year(media_info.year)
    if media_info.genre is not None:
        ultrastar_txt.genre = format_separated_string(media_info.genre)
    if media_info.video_url is not None:
        ultrastar_txt.videoUrl = media_info.video_url
    if media_info.cover_url is not None:
        ultrastar_txt.coverUrl = media_info.cover_url
    if media_info.music_key is not None:
        # todo: as list add here?
        ultrastar_txt.tags = f"key: {media_info.music_key}"

    ultrastar_file_output_path = os.path.join(song_folder_output_path, basename + ".txt")
    create_ultrastar_txt(
        midi_segments,
        ultrastar_file_output_path,
        ultrastar_txt,
        media_info.bpm,
        max_bpm,
    )
    if create_karaoke and version.parse(format_version.value) < version.parse(FormatVersion.V1_1_0.value):
        title = basename + " [Karaoke]"
        ultrastar_txt.title = title
        ultrastar_txt.mp3 = title + "." + media_info.audio_extension
        karaoke_output_path = os.path.join(song_folder_output_path, title)
        karaoke_txt_output_path = karaoke_output_path + ".txt"
        create_ultrastar_txt(
            midi_segments,
            karaoke_txt_output_path,
            ultrastar_txt,
            media_info.bpm,
            max_bpm,
        )
    if version.parse(format_version.value) < version.parse(FormatVersion.V1_2_0.value):
        ultrastar_txt.videoUrl = media_info.video_url

    return ultrastar_file_output_path


def create_ultrastar_txt(
        midi_segments: list[MidiSegment],
        ultrastar_file_output: str,
        ultrastar_class: UltrastarTxtValue,
        real_bpm: float,
        max_bpm: float) -> None:
    """Creates an Ultrastar txt file from the automation data. Line breaks come from the notes'
    line_start flags (a line-break step decides them); without flags the chart has one line."""

    log.info(f"Creating UltraStar file {ultrastar_file_output}")

    ultrastar_bpm = real_bpm / BEATS_PER_QUARTER
    multiplication = get_multiplier(ultrastar_bpm, max_bpm)
    ultrastar_bpm = ultrastar_bpm * multiplication

    with open(ultrastar_file_output, "w", encoding=FILE_ENCODING) as file:
        gap = midi_segments[0].start

        if version.parse(ultrastar_class.version) >= version.parse("1.0.0"):
            file.write(f"#{UltrastarTxtTag.VERSION.value}:{ultrastar_class.version}\n")
        file.write(f"#{UltrastarTxtTag.ARTIST.value}:{ultrastar_class.artist}\n")
        file.write(f"#{UltrastarTxtTag.TITLE.value}:{ultrastar_class.title}\n")
        if ultrastar_class.year is not None:
            file.write(f"#{UltrastarTxtTag.YEAR.value}:{ultrastar_class.year}\n")
        if ultrastar_class.language is not None:
            file.write(f"#{UltrastarTxtTag.LANGUAGE.value}:{get_language_name(ultrastar_class.language)}\n")
        if ultrastar_class.genre:
            file.write(f"#{UltrastarTxtTag.GENRE.value}:{ultrastar_class.genre}\n")
        if ultrastar_class.cover is not None:
            file.write(f"#{UltrastarTxtTag.COVER.value}:{ultrastar_class.cover}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.2.0"):
            if ultrastar_class.coverUrl is not None:
                file.write(f"#{UltrastarTxtTag.COVERURL.value}:{ultrastar_class.coverUrl}\n")
        if ultrastar_class.background is not None:
            file.write(f"#{UltrastarTxtTag.BACKGROUND.value}:{ultrastar_class.background}\n")
        file.write(f"#{UltrastarTxtTag.MP3.value}:{ultrastar_class.mp3}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            file.write(f"#{UltrastarTxtTag.AUDIO.value}:{ultrastar_class.audio}\n")
            if ultrastar_class.vocals is not None:
                file.write(f"#{UltrastarTxtTag.VOCALS.value}:{ultrastar_class.vocals}\n")
            if ultrastar_class.instrumental is not None:
                file.write(f"#{UltrastarTxtTag.INSTRUMENTAL.value}:{ultrastar_class.instrumental}\n")
        if ultrastar_class.video is not None:
            file.write(f"#{UltrastarTxtTag.VIDEO.value}:{ultrastar_class.video}\n")
        if ultrastar_class.videoGap is not None:
            file.write(f"#{UltrastarTxtTag.VIDEOGAP.value}:{ultrastar_class.videoGap}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.2.0"):
            if ultrastar_class.videoUrl is not None:
                file.write(f"#{UltrastarTxtTag.VIDEOURL.value}:{ultrastar_class.videoUrl}\n")
        file.write(f"#{UltrastarTxtTag.BPM.value}:{round(ultrastar_bpm, 2)}\n")  # not the real BPM!
        file.write(f"#{UltrastarTxtTag.GAP.value}:{int(gap * 1000)}\n")
        if version.parse(ultrastar_class.version) >= version.parse("1.1.0"):
            if ultrastar_class.tags is not None:
                file.write(f"#{UltrastarTxtTag.TAGS.value}:{ultrastar_class.tags}\n")
        file.write(f"#{UltrastarTxtTag.CREATOR.value}:{ultrastar_class.creator}\n")
        file.write(f"#{UltrastarTxtTag.COMMENT.value}:{ultrastar_class.comment}\n")

        # Write the singing part
        previous_end_beat = 0

        for i, midi_segment in enumerate(midi_segments):
            # Round start and end from absolute times. Rounding the duration separately and
            # pushing the next note behind it let rounding errors accumulate: with many
            # touching notes the chart drifted seconds late by the end of a song.
            start_beat = round(second_to_beat((midi_segment.start - gap) * multiplication, real_bpm))
            end_beat = round(second_to_beat((midi_segment.end - gap) * multiplication, real_bpm))

            # Don't overlap the previous note; keep at least one beat
            start_beat = max(start_beat, previous_end_beat)
            duration = max(1, end_beat - start_beat)
            previous_end_beat = start_beat + duration

            # : 10 10 10 w
            # ':'   start midi part
            # 'n1'  start at real beat
            # 'n2'  duration at real beat
            # 'n3'  pitch where 0 == C4
            # 'w'   lyric
            line = f"{UltrastarTxtNoteTypeTag.NORMAL.value} " \
                   f"{str(start_beat)} " \
                   f"{str(duration)} " \
                   f"{str(librosa.note_to_midi(midi_segment.note) - ULTRASTAR_C4_MIDI)} " \
                   f"{midi_segment.word}\n"

            file.write(line)

            # breaks only after word ends, where the next note starts a line
            if not midi_segment.word.endswith(" "):
                continue
            if i + 1 < len(midi_segments) and midi_segments[i + 1].line_start:
                # - 10
                # '-' end of current sing part
                # 'n1' show next at time in real beat
                show_next = (
                        second_to_beat(midi_segment.end - gap, real_bpm)
                        * multiplication
                )
                linebreak = f"{UltrastarTxtTag.LINEBREAK.value} " \
                            f"{str(round(show_next))}\n"
                file.write(linebreak)
        file.write(f"{UltrastarTxtTag.FILE_END.value}")


def format_separated_string(data: str) -> str:
    temp = re.sub(r'[;/]', ',', data)
    words = temp.split(',')
    words = [s for s in words if s.strip()]

    for i, word in enumerate(words):
        if "-" not in word:
            words[i] = word.strip().capitalize() + ', '
        else:
            dash_words = word.split('-')
            capitalized_dash_words = [dash_word.strip().capitalize() for dash_word in dash_words]
            formatted_dash_word = '-'.join(capitalized_dash_words) + ', '
            words[i] = formatted_dash_word

    formatted_string = ''.join(words)

    if formatted_string.endswith(', '):
        formatted_string = formatted_string[:-2]

    return formatted_string


def extract_year(date: str) -> str:
    """First four-digit year in a date string (MusicBrainz dates), else the string."""
    match = re.search(r'\b\d{4}\b', date)
    if match:
        return match.group(0)
    else:
        return date
