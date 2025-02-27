"""Multi-track Whisper speech recognition module.

This module extends the Whisper transcription to support multiple vocal tracks.
"""

from __future__ import annotations  # For forward reference support

import os
import json
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

# Use conditional imports to prevent circular dependencies
if TYPE_CHECKING:
    from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult

from modules.Speech_Recognition.Whisper import transcribe_with_whisper, WhisperModel
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.Speech_Recognition.transcription_combiner import combine_transcription_results
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted, green_highlighted
from modules.os_helper import check_file_exists


def transcribe_multiple_tracks(
    vocal_tracks: Dict[str, str],
    whisper_model: WhisperModel,
    device: str = "cpu",
    align_model: Optional[str] = None,
    batch_size: int = 16,
    compute_type: Optional[str] = None,
    language: Optional[str] = None,
    keep_numbers: bool = False,
    cache_folder_path: Optional[str] = None,
    skip_cache: bool = False,
    model_weights: Optional[Dict[str, float]] = None,
    dominant_model: Optional[str] = None
) -> Tuple[TranscriptionResult, Dict[str, TranscriptionResult]]:
    """
    Transcribe multiple vocal tracks and combine the results.
    
    Args:
        vocal_tracks: Dictionary mapping model names to vocal track paths
        whisper_model: Whisper model to use
        device: Device to use (cpu/cuda)
        align_model: Optional alignment model
        batch_size: Whisper batch size
        compute_type: Whisper compute type
        language: Language to use (or None for auto-detection)
        keep_numbers: Whether to keep numbers as digits
        cache_folder_path: Optional folder for caching results
        skip_cache: Whether to skip using cached results
        model_weights: Optional weights for each model
        dominant_model: Optional model to give priority in case of doubt
        
    Returns:
        Tuple of (combined_result, individual_results_by_model)
    """
    if not vocal_tracks:
        raise ValueError("No vocal tracks provided for transcription")
    
    print(f"{ULTRASINGER_HEAD} Transcribing {len(vocal_tracks)} vocal tracks with {blue_highlighted(whisper_model.value)}")
    
    # Use default weights if not provided
    if model_weights is None:
        model_weights = {
            "htdemucs_ft": 0.6,  # Fine-tuned model (highest quality)
            "htdemucs": 0.3,     # Default model
            "htdemucs_6s": 0.4   # 6-source model (often better for vocals)
        }
    
    # Process each vocal track
    results_by_model = {}
    
    for model_name, track_path in vocal_tracks.items():
        print(f"{ULTRASINGER_HEAD} Processing transcription for {blue_highlighted(model_name)}")
        
        # Create cache path for this model
        cache_path = None
        if cache_folder_path:
            whisper_align_model_string = None
            if align_model:
                whisper_align_model_string = align_model.replace("/", "_")
                
            cache_config = f"whisper_{model_name}_{whisper_model.value}_{device}_{whisper_align_model_string}_{batch_size}_{compute_type}_{language}"
            cache_path = os.path.join(cache_folder_path, f"{cache_config}.json")
            
            # Check if cached result exists
            if not skip_cache and check_file_exists(cache_path):
                print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached transcription for {blue_highlighted(model_name)}")
                try:
                    with open(cache_path, 'r', encoding='utf-8-sig') as file:
                        json_data = file.read()
                        result = TranscriptionResult.from_json(json_data)
                        results_by_model[model_name] = result
                    continue
                except Exception as e:
                    print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} loading cached transcription: {str(e)}")
                    # Continue to transcribe instead of using cache
        
        # Transcribe this track
        try:
            result = transcribe_with_whisper(
                track_path,
                whisper_model,
                device,
                align_model,
                batch_size,
                compute_type,
                language,
                keep_numbers
            )
            
            # Add to results
            results_by_model[model_name] = result
            
            # Save to cache if enabled
            if cache_path:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    with open(cache_path, 'w', encoding='utf-8-sig') as file:
                        file.write(result.to_json())
                except Exception as e:
                    print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning')} Failed to save cache: {str(e)}")
        
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} transcribing {blue_highlighted(model_name)}: {str(e)}")
            # Log stack trace for debugging if needed
            import traceback
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Debug')} Error details: {traceback.format_exc().splitlines()[-1]}")

    # After processing all tracks:
    if not results_by_model:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} Failed to transcribe any vocal tracks")
        # If we're supposed to return a result but have none, create an empty one
        empty_result = TranscriptionResult(
            transcribed_data=[],
            detected_language=language or "en"  # Use provided language or default to English
        )
        return empty_result, {}
        
    # If only one track was processed, just return its result
    if len(results_by_model) == 1:
        model_name = next(iter(results_by_model.keys()))
        result = results_by_model[model_name]
        return result, results_by_model
    
    # Combine results from all tracks
    print(f"{ULTRASINGER_HEAD} Combining transcriptions from {len(results_by_model)} models")
    combined_result = combine_transcription_results(
        results_by_model, model_weights, dominant_model
    )
    
    # Create visualization if cache folder is provided
    if cache_folder_path:
        from modules.Speech_Recognition.transcription_combiner import visualize_combined_transcription
        
        # Convert TranscriptionResult to list of TranscribedData
        original_transcriptions = {
            model: result.transcribed_data
            for model, result in results_by_model.items()
        }
        
        visualization_path = os.path.join(cache_folder_path, "transcription_comparison.png")
        try:
            visualize_combined_transcription(
                original_transcriptions,
                combined_result.transcribed_data,
                visualization_path
            )
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} creating visualization: {str(e)}")
    
    return combined_result, results_by_model


def analyze_transcription_quality(
    results_by_model: Dict[str, TranscriptionResult],
    combined_result: TranscriptionResult
) -> None:
    """
    Analyze and print quality metrics for multi-track transcription.
    
    Args:
        results_by_model: Dictionary mapping model names to TranscriptionResult
        combined_result: The combined TranscriptionResult
    """
    # Handle empty results case
    if not results_by_model:
        print(f"{ULTRASINGER_HEAD} No valid transcription results to analyze")
        return
        
    # If only one model succeeded, just show basic info
    if len(results_by_model) == 1:
        model = next(iter(results_by_model.keys()))
        result = results_by_model[model]
        word_count = len(result.transcribed_data)
        avg_conf = sum(word.confidence for word in result.transcribed_data) / max(1, word_count)
        
        print(f"{ULTRASINGER_HEAD} Single-track transcription stats:")
        print(f"{ULTRASINGER_HEAD} - Model: {blue_highlighted(model)}")
        print(f"{ULTRASINGER_HEAD} - Word count: {word_count}")
        print(f"{ULTRASINGER_HEAD} - Average confidence: {avg_conf:.3f}")
        return
    
    # Convert TranscriptionResult to list of TranscribedData
    transcriptions = {
        model: result.transcribed_data
        for model, result in results_by_model.items()
    }
    
    # Calculate basic statistics directly instead of using analyze_transcription_differences
    print(f"{ULTRASINGER_HEAD} Multi-track transcription quality analysis:")
    
    # Word counts by model
    print(f"{ULTRASINGER_HEAD} Word counts by model:")
    for model, data in transcriptions.items():
        print(f"{ULTRASINGER_HEAD} - {blue_highlighted(model)}: {len(data)} words")
    
    # Average confidence by model
    print(f"{ULTRASINGER_HEAD} Average confidence by model:")
    for model, data in transcriptions.items():
        if data:
            avg_conf = sum(word.confidence for word in data) / len(data)
            print(f"{ULTRASINGER_HEAD} - {blue_highlighted(model)}: {avg_conf:.3f}")
        else:
            print(f"{ULTRASINGER_HEAD} - {blue_highlighted(model)}: No data")
    
    # Combined result stats
    print(f"{ULTRASINGER_HEAD} Combined result: {len(combined_result.transcribed_data)} words")