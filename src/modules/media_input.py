"""Turn a user input (audio file, video file, YouTube URL or UltraStar txt) into a song folder.

Returns a SongInput: where the audio is, the song's output folder and its metadata. Everything
here happens before the pipeline (it creates the folder the pipeline writes into).
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from modules import os_helper
from modules.ProcessData import MediaInfo

log = logging.getLogger("ultrasinger.input")


@dataclass
class SongInput:
    basename: str           # "Artist - Title", used for every output file name
    song_dir: str           # output folder of this song
    audio: str              # audio file the pipeline starts from
    media: MediaInfo
    chart: str | None = None  # UltraStar txt given as input (re-pitch / oracle runs)


def _artist_title(stem: str) -> tuple[str | None, str]:
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip(), title.strip()
    return None, stem.strip()


def _lookup(artist, title, enabled: bool):
    """MusicBrainz metadata, or just what the file name says."""
    if enabled:
        from modules.musicbrainz_client import search_musicbrainz
        try:
            return search_musicbrainz(title, artist)
        except Exception as err:  # offline, rate limit, ...: carry on with the file name
            log.warning("MusicBrainz lookup failed (%s), using the file name", err)
    from modules.musicbrainz_client import SongInfo
    return SongInfo(title=title, artist=artist or "Unknown")


def _new_song_dir(output_root: str, basename: str, reuse: bool) -> str:
    path = os.path.join(output_root, basename)
    if not reuse:
        path = os_helper.get_unused_song_output_dir(path)
    os.makedirs(path, exist_ok=True)
    return path


def from_file(path: str, output_root: str, lookup: bool = True, reuse_dir: bool = False) -> SongInput:
    """Audio or video file. Video is split into audio + silent video next to the output."""
    from modules.ffmpeg_helper import is_video_file, separate_audio_video
    stem, ext = os.path.splitext(os.path.basename(path))
    info = _lookup(*_artist_title(stem), lookup)
    basename = os_helper.sanitize_filename(f"{info.artist} - {info.title}")
    song_dir = _new_song_dir(output_root, basename, reuse_dir)
    video_ext = None
    if is_video_file(path):
        log.info("video file: separating audio and video")
        copy = os.path.join(song_dir, basename + ext)
        os_helper.copy(path, copy)
        audio, _video, audio_ext, video_ext = separate_audio_video(copy, basename, song_dir)
    else:
        audio = os.path.join(song_dir, basename + ext)
        if os.path.abspath(audio) != os.path.abspath(path):
            os_helper.copy(path, audio)
        audio_ext = ext.lstrip(".")
    if getattr(info, "cover_image_data", None) is not None:
        from modules.Image.image_helper import save_image
        save_image(info.cover_image_data, basename, song_dir)
    media = MediaInfo(artist=info.artist, title=info.title, year=getattr(info, "year", None),
                      genre=getattr(info, "genres", None), cover_url=getattr(info, "cover_url", None),
                      audio_extension=audio_ext, video_extension=video_ext)
    return SongInput(basename, song_dir, audio, media)


def from_youtube(url: str, output_root: str, cookiefile: str | None = None) -> SongInput:
    from modules.Audio.youtube import download_from_youtube
    basename, song_dir, audio, media = download_from_youtube(url, output_root, cookiefile)
    return SongInput(basename, song_dir, audio, media)


def read_chart_headers(path: str) -> dict[str, str]:
    from modules.Ultrastar.ultrastar_txt import FILE_ENCODING
    headers = {}
    with open(path, encoding=FILE_ENCODING, errors="replace") as f:
        for line in f:
            m = re.match(r"#([A-Za-z0-9_-]+):(.*)", line.strip("﻿\r\n"))
            if m:
                headers[m.group(1).upper()] = m.group(2).strip()
    return headers


def from_chart(path: str, output_root: str, reuse_dir: bool = False) -> SongInput:
    """UltraStar txt: the audio it references, metadata from its headers (no lookup)."""
    h = read_chart_headers(path)
    name = h.get("AUDIO") or h.get("MP3")
    if not name:
        raise ValueError(f"{path} does not reference an audio file (#AUDIO/#MP3)")
    audio = os.path.join(os.path.dirname(os.path.abspath(path)), name)
    artist, title = h.get("ARTIST", "Unknown"), h.get("TITLE", os.path.splitext(name)[0])
    basename = os_helper.sanitize_filename(f"{artist} - {title}")
    song_dir = _new_song_dir(output_root, basename, reuse_dir)
    media = MediaInfo(artist=artist, title=title, year=h.get("YEAR"), genre=h.get("GENRE"),
                      audio_extension=os.path.splitext(name)[1].lstrip("."))
    return SongInput(basename, song_dir, audio, media, chart=os.path.abspath(path))


def prepare(input_path: str, output_root: str, lookup: bool = True, cookiefile: str | None = None,
            reuse_dir: bool = False) -> SongInput:
    if input_path.startswith(("http://", "https://")):
        return from_youtube(input_path, output_root, cookiefile)
    if input_path.lower().endswith(".txt"):
        return from_chart(input_path, output_root, reuse_dir)
    return from_file(input_path, output_root, lookup, reuse_dir)
