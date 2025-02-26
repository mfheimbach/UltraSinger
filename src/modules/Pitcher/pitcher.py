"""Pitcher module"""
import os

import crepe
from scipy.io import wavfile

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Midi.midi_creator import convert_frequencies_to_notes, most_frequent
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence
from modules.Audio.convert_audio import convert_audio_to_mono_wav
from modules.os_helper import check_file_exists
from modules.Ultrastar.ultrastar_txt import FILE_ENCODING

def get_pitch_with_crepe_file(
    filename: str, model_capacity: str, step_size: int = 10, device: str = "cpu"
) -> PitchedData:
    """Pitch with crepe"""

    print(
        f"{ULTRASINGER_HEAD} Pitching with {blue_highlighted('crepe')} and model {blue_highlighted(model_capacity)} and {red_highlighted(device)} as worker"
    )
    sample_rate, audio = wavfile.read(filename)

    return get_pitch_with_crepe(audio, sample_rate, model_capacity, step_size)


def get_pitch_with_crepe(
    audio, sample_rate: int, model_capacity: str, step_size: int = 10
) -> PitchedData:
    """Pitch with crepe"""

    # Info: The model is trained on 16 kHz audio, so if the input audio has a different sample rate, it will be first resampled to 16 kHz using resampy inside crepe.

    times, frequencies, confidence, activation = crepe.predict(
        audio, sample_rate, model_capacity, step_size=step_size, viterbi=True
    )

    # convert to native float for serialization
    confidence = [float(x) for x in confidence]

    return PitchedData(times, frequencies, confidence)


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

def pitch_original_audio(
        audio_file_path: str,
        crepe_model_capacity: str = "full",
        crepe_step_size: int = 10,
        tensorflow_device: str = None  # Set default to None
) -> PitchedData:
    """Pitch detection on original (unseparated) audio."""
    
    # Use system settings for device if not specified
    if tensorflow_device is None:
        tensorflow_device = settings.tensorflow_device
    
    print(f"{ULTRASINGER_HEAD} Running pitch detection on original audio with {blue_highlighted('crepe')} on {red_highlighted(tensorflow_device)}")
    
    # Rest of the function remains the same...
    
    # Create a cache name specific to original audio
    cache_folder = os.path.dirname(audio_file_path)
    pitching_config = f"crepe_original_{crepe_model_capacity}_{crepe_step_size}_{tensorflow_device}"
    pitched_data_path = os.path.join(cache_folder, f"{pitching_config}.json")
    cache_available = check_file_exists(pitched_data_path)
    
    if not cache_available:
        # Convert to WAV if not already
        temp_wav_path = os.path.join(cache_folder, "original_temp.wav")
        convert_audio_to_mono_wav(audio_file_path, temp_wav_path)
        
        # Run Crepe on converted audio file
        pitched_data = get_pitch_with_crepe_file(
            temp_wav_path,
            crepe_model_capacity,
            crepe_step_size,
            tensorflow_device,
        )
        
        # Cache the results
        pitched_data_json = pitched_data.to_json()
        with open(pitched_data_path, "w", encoding=FILE_ENCODING) as file:
            file.write(pitched_data_json)
    else:
        print(f"{ULTRASINGER_HEAD} {green_highlighted('cache')} reusing cached original audio pitch data")
        with open(pitched_data_path) as file:
            json = file.read()
            pitched_data = PitchedData.from_json(json)
    
    return pitched_data