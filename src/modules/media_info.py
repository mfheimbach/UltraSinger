from dataclasses import dataclass
from typing import Optional


@dataclass
class MediaInfo:
    """Media Info"""
    title: str
    artist: str
    bpm: Optional[float] = None
    year: Optional[str] = None
    genre: Optional[str] = None
    language: Optional[str] = None
    cover_url: Optional[str] = None
    video_url: Optional[str] = None
    music_key: Optional[str] = None  # (e.g., "C major", "A minor")
    video_extension: Optional[str] = None  # (e.g., "mp4", "mkv")
    audio_extension: Optional[str] = None  # (e.g., "mp3", "m4a")


