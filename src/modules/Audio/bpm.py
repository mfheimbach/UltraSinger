import librosa
import soundfile as sf

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted


def get_bpm_from_data(data, sampling_rate):
    """Get real bpm from audio data"""
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sampling_rate)

    print(f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(wav_tempo[0], 2)))}")
    return wav_tempo[0]


def get_bpm_from_file(wav_file: str) -> float:
    """Get real bpm from audio file"""
    data, sampling_rate = sf.read(wav_file, dtype='float32')
    # Transpose if stereo to match librosa's expected format
    if len(data.shape) > 1:
        data = data.T
    # Convert to mono if stereo
    if data.ndim > 1:
        data = librosa.to_mono(data)
    return get_bpm_from_data(data, sampling_rate)


def get_bpm_with_beat_this(audio_file: str, device: str = "cpu") -> float:
    """BPM from Beat This! beat tracking (median inter-beat interval)."""
    import numpy as np
    from beat_this.inference import File2Beats
    from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted
    tracker = File2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, _downbeats = tracker(audio_file)
    if len(beats) < 4:
        return get_bpm_from_file(audio_file)
    bpm = float(60.0 / np.median(np.diff(beats)))
    print(f"{ULTRASINGER_HEAD} BPM with {blue_highlighted('Beat This!')}: {bpm:.2f}")
    return bpm
