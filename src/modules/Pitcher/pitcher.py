"""Pitcher module"""
import os

import crepe
from scipy.io import wavfile
import numpy as np

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Midi.midi_creator import convert_frequencies_to_notes, most_frequent
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Pitcher.pitch_combiner import combine_pitch_data
from modules.Pitcher.vad_pitch_combiner import (
    process_vad_multi_track,
    joint_vad_pitch_processing,
    segment_notes_with_vad
)


def get_pitch_with_crepe_file(
    filename: str, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with crepe"""

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('crepe')} and model {blue_highlighted(model_capacity)} and {red_highlighted(device)} as worker"
    )
    sample_rate, audio = wavfile.read(filename)

    return get_pitch_with_crepe(audio, sample_rate, model_capacity, step_size, device)


def get_pitch_with_crepe(
    audio, sample_rate: int, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with crepe"""

    # Info: The model is trained on 16 kHz audio, so if the input audio has a different sample rate, it will be first resampled to 16 kHz using resampy inside crepe.

    # Set TensorFlow device if using CUDA
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    times, frequencies, confidence, activation = crepe.predict(
        audio, sample_rate, model_capacity, step_size=step_size, viterbi=True
    )

    # convert to native float for serialization
    confidence = [float(x) for x in confidence]

    return PitchedData(times, frequencies, confidence)


def get_multi_track_pitch(
    vocal_tracks: dict,
    model_capacity: str, 
    step_size: int = 10,
    device: str = "cpu",
    confidence_thresholds: dict = None,
    agreement_bonus: float = 0.2,
    debug_level: int = 1,
    output_dir: str = None,
    vad_enabled: bool = False,
    settings = None
) -> PitchedData:
    """
    Process multiple vocal tracks with CREPE and combine the results.
    
    Args:
        vocal_tracks: Dictionary of model_name -> vocal track path
        model_capacity: CREPE model capacity to use
        step_size: Step size in milliseconds for CREPE
        device: Device to use (cpu/cuda)
        confidence_thresholds: Model-specific confidence thresholds
        agreement_bonus: Confidence boost when tracks agree
        debug_level: Debug level (0-3)
        output_dir: Directory to save debug outputs
        vad_enabled: Whether to use voice activity detection
        settings: Settings object
        
    Returns:
        Combined PitchedData
    """
    if not vocal_tracks:
        raise ValueError("No vocal tracks provided")
    
    # Process each track with CREPE
    pitch_data_dict = {}
    for model_name, track_path in vocal_tracks.items():
        print(f"{ULTRASINGER_HEAD} Processing vocal track from {blue_highlighted(model_name)}")
        try:
            pitched_data = get_pitch_with_crepe_file(track_path, model_capacity, step_size, device)
            pitch_data_dict[model_name] = pitched_data
            print(f"{ULTRASINGER_HEAD} {blue_highlighted(model_name)} track processed: {len(pitched_data.times)} time points")
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} processing track from {blue_highlighted(model_name)}: {e}")
    
    # If VAD is enabled, perform joint VAD and pitch processing
    if vad_enabled and settings and len(pitch_data_dict) > 0:
        try:
            print(f"{ULTRASINGER_HEAD} Running {blue_highlighted('voice activity detection')} to enhance pitch data")
            
            # Import here to avoid circular imports
            from modules.Pitcher.vad_pitch_combiner import process_vad_multi_track, joint_vad_pitch_processing
            
            # Process VAD on all tracks
            vad_results = process_vad_multi_track(vocal_tracks, settings)
            
            # Perform joint processing with robust error handling
            enhanced_pitch_data = joint_vad_pitch_processing(vad_results, pitch_data_dict, settings)
            
            print(f"{ULTRASINGER_HEAD} Successfully enhanced pitch data with VAD")
            return enhanced_pitch_data
            
        except Exception as e:
            import traceback
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} during VAD processing: {str(e)}")
            print(traceback.format_exc())
            print(f"{ULTRASINGER_HEAD} Falling back to standard pitch combination")
    
    # Standard pitch combination (without VAD)
    if len(pitch_data_dict) > 0:
        return combine_pitch_data(
            pitch_data_dict,
            confidence_thresholds=confidence_thresholds,
            agreement_bonus=agreement_bonus,
            debug_level=debug_level,
            output_dir=output_dir
        )
    else:
        raise ValueError("No pitch data could be extracted from any vocal track")


def get_pitched_data_with_high_confidence(
    pitched_data: PitchedData, threshold=0.4
) -> PitchedData:
    """Get frequency with high confidence"""
    new_pitched_data = PitchedData([], [], [])
    for i, conf in enumerate(pitched_data.confidence):
        if conf > threshold:
            new_pitched_data.times.append(pitched_data.times[i])
            new_pitched_data.frequencies.append(pitched_data.frequencies[i])
            new_pitched_data.confidence.append(pitched_data.confidence[i])

    return new_pitched_data


def process_with_vad(vocal_tracks: dict, pitched_data: PitchedData, settings) -> tuple:
    """
    Process audio tracks with Voice Activity Detection for improved results.
    
    Args:
        vocal_tracks: Dictionary of model_name -> vocal track path
        pitched_data: Combined pitch data
        settings: Settings object
        
    Returns:
        Tuple of (enhanced pitched data, vad results)
    """
    try:
        # Process VAD on all tracks
        vad_results = process_vad_multi_track(vocal_tracks, settings)
        
        # Get timestamps and combined VAD from the first track as reference
        first_model = next(iter(vad_results.keys()))
        timestamps, vad_scores = vad_results[first_model]
        
        # Get model weights
        model_weights = getattr(settings, 'VAD_MODEL_WEIGHTS', {
            "htdemucs_ft": 0.6,
            "htdemucs": 0.3,
            "htdemucs_6s": 0.4
        })
        
        # Create placeholder pitch data dict for joint processing
        pitch_data_dict = {first_model: pitched_data}
        
        # Perform joint processing
        enhanced_pitch_data = joint_vad_pitch_processing(vad_results, pitch_data_dict, settings)
        
        return enhanced_pitch_data, vad_results
        
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} during VAD processing: {e}")
        print(f"{ULTRASINGER_HEAD} Returning original pitch data without VAD enhancement")
        return pitched_data, None


# Todo: Unused
def pitch_each_chunk_with_crepe(directory: str,
                                crepe_model_capacity: str,
                                crepe_step_size: int,
                                tensorflow_device: str) -> list[str]:
    """Pitch each chunk with crepe and return midi notes"""
    print(f"{ULTRASINGER_HEAD} Pitching each chunk with {blue_highlighted('crepe')}")

    midi_notes = []
    for filename in sorted(
            [f for f in os.listdir(directory) if f.endswith(".wav")],
            key=lambda x: int(x.split("_")[1]),
    ):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_crepe_file(
            filepath,
            crepe_model_capacity,
            crepe_step_size,
            tensorflow_device,
        )
        conf_f = get_frequencies_with_high_confidence(
            pitched_data.frequencies, pitched_data.confidence
        )

        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    return midi_notes

class Pitcher:
    """Docstring"""