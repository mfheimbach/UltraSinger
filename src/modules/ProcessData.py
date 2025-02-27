﻿from dataclasses import dataclass, field
from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Pitcher.pitched_data import PitchedData
from modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from modules.Midi.MidiSegment import MidiSegment
from typing import Optional, List, Dict 

@dataclass
class ProcessDataPaths:
    # Process data Paths
    processing_audio_path: Optional[str] = ""
    cache_folder_path: Optional[str] = ""
    audio_output_file_path: Optional[str] = "" # Output audio file path
    vocals_audio_file_path: Optional[str] = "" # Separated vocals audio file path
    instrumental_audio_file_path: Optional[str] = "" # Separated instrumental audio file path

@dataclass
class MediaInfo:
    """Media Info"""
    title: str
    artist: str
    bpm: float
    year: Optional[str] = None
    genre: Optional[str] = None
    language: Optional[str] = None
    cover_url: Optional[str] = None
    video_url: Optional[str] = None

@dataclass
class ProcessData:
    """Data for processing"""
    process_data_paths: ProcessDataPaths = ProcessDataPaths()
    basename: Optional[str] = None
    media_info: Optional[MediaInfo] = None
    transcribed_data: Optional[List[TranscribedData]] = field(default_factory=list)
    pitched_data: Optional[PitchedData] = None
    midi_segments: Optional[List[MidiSegment]] = field(default_factory=list)
    parsed_file: Optional[UltrastarTxtValue] = None
    vocal_tracks: Optional[Dict[str, str]] = field(default_factory=dict)  # Dict of model_name -> vocals.wav path
    transcription_results: Optional[Dict[str, TranscriptionResult]] = field(default_factory=dict)