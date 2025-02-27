"""Voice Activity Detection and Pitch Combiner for multi-track vocal processing.

This module provides functions for detecting vocal activity across multiple
tracks and combining them with pitch data for better note detection.
"""

import numpy as np
import librosa
import soundfile as sf
import os
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, cyan_highlighted, green_highlighted, red_highlighted
from modules.Pitcher.pitched_data import PitchedData
from modules.Midi.MidiSegment import MidiSegment


class VocalActivityDetection:
    """
    Detects vocal activity in audio using multiple methods.
    """
    
    def __init__(self, 
                 energy_threshold: float = 0.05,
                 min_duration: float = 0.1,
                 smooth_window: int = 5):
        """
        Initialize the VAD detector.
        
        Args:
            energy_threshold: Threshold for energy-based VAD (0.0-1.0)
            min_duration: Minimum duration of a vocal segment in seconds
            smooth_window: Window size for smoothing the VAD signal
        """
        self.energy_threshold = energy_threshold
        self.min_duration = min_duration
        self.smooth_window = smooth_window
    
    def detect_from_file(self, audio_file: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect vocal activity from an audio file.
        
        Args:
            audio_file: Path to the audio file
            
        Returns:
            Tuple of (timestamps, activity_scores, sample_rate)
        """
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None)
        return self.detect(audio, sr)
    
    def detect(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Detect vocal activity in an audio signal.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (timestamps, activity_scores, sample_rate)
        """
        # Normalization to ensure consistent energy levels
        audio = audio / (np.max(np.abs(audio)) + 1e-10)
        
        # We'll use a combined approach with multiple features for better accuracy
        
        # 1. Energy-based VAD
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop
        
        # Calculate frame-wise RMS energy
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        energy = np.asarray(energy)  # Ensure numpy array
        energy_norm = energy / (np.max(energy) + 1e-10)  # Normalize
        
        # 2. Spectral features for improved detection
        # Get mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, 
                                                 n_fft=frame_length*2, 
                                                 hop_length=hop_length, 
                                                 n_mels=40)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Calculate spectral flatness (higher for noise, lower for voice)
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=frame_length*2, hop_length=hop_length)
        flatness = np.asarray(flatness)  # Ensure numpy array
        flatness_norm = 1.0 - (flatness / (np.max(flatness) + 1e-10))  # Invert and normalize
        
        # 3. Calculate harmonic-percussive source separation to help identify vocals
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_energy = librosa.feature.rms(y=harmonic, frame_length=frame_length, hop_length=hop_length)[0]
        harmonic_energy = np.asarray(harmonic_energy)  # Ensure numpy array
        harmonic_energy_norm = harmonic_energy / (np.max(harmonic_energy) + 1e-10)
        
        # Make sure all arrays are the same length for combination
        min_length = min(len(energy_norm), len(flatness_norm.flatten()), len(harmonic_energy_norm))
        energy_norm = energy_norm[:min_length]
        harmonic_energy_norm = harmonic_energy_norm[:min_length]
        flatness_norm_flat = flatness_norm.flatten()[:min_length]
        
        # Combine features with different weights
        # Energy and harmonic content are strong indicators of vocal activity
        vad_score = (0.5 * energy_norm + 
                     0.3 * harmonic_energy_norm + 
                     0.2 * flatness_norm_flat)
        
        # Smooth the VAD scores
        vad_score_smoothed = self._smooth_signal(vad_score, self.smooth_window)
        vad_score_smoothed = np.asarray(vad_score_smoothed)  # Ensure numpy array
        
        # Apply threshold - explicitly convert comparison result to numpy array
        threshold_array = np.full_like(vad_score_smoothed, self.energy_threshold)
        binary_vad = np.asarray(vad_score_smoothed > threshold_array).astype(float)
        
        # Apply minimum duration constraint
        binary_vad = self._apply_min_duration(binary_vad, hop_length, sample_rate)
        
        # Generate timestamps - ensure numpy array
        timestamps = np.asarray(librosa.frames_to_time(np.arange(len(binary_vad)), 
                                                    sr=sample_rate, 
                                                    hop_length=hop_length))
        
        return timestamps, vad_score_smoothed, sample_rate
    
    def _smooth_signal(self, signal: np.ndarray, window_size: int) -> np.ndarray:
        """Apply smoothing to a signal using a moving average."""
        signal = np.asarray(signal)  # Ensure numpy array
        
        if window_size <= 1:
            return signal
            
        # Pad signal for smoothing
        padded = np.pad(signal, (window_size//2, window_size//2), mode='edge')
        
        # Apply moving average
        window = np.ones(window_size) / window_size
        smoothed = np.convolve(padded, window, mode='valid')
        
        return np.asarray(smoothed)  # Ensure numpy array
    
    def _apply_min_duration(self, 
                           binary_vad: np.ndarray, 
                           hop_length: int, 
                           sample_rate: int) -> np.ndarray:
        """
        Apply minimum duration constraint to VAD signal.
        
        Args:
            binary_vad: Binary VAD signal (0 or 1)
            hop_length: Hop length in samples
            sample_rate: Sample rate in Hz
            
        Returns:
            Processed binary VAD signal
        """
        binary_vad = np.asarray(binary_vad)  # Ensure numpy array
        
        # Convert min_duration to frames
        min_frames = int(self.min_duration * sample_rate / hop_length)
        
        if min_frames <= 1:
            return binary_vad
        
        # Find boundaries
        changes = np.diff(np.pad(binary_vad, (1, 1), 'constant'))
        onsets = np.where(changes > 0)[0]
        offsets = np.where(changes < 0)[0]
        
        # Ensure we have matching pairs
        if len(onsets) == 0 or len(offsets) == 0:
            return binary_vad
            
        if offsets[0] <= onsets[0]:
            offsets = offsets[1:]
        
        if len(onsets) > len(offsets):
            onsets = onsets[:len(offsets)]
            
        # Process each segment
        result = np.zeros_like(binary_vad)
        
        for onset, offset in zip(onsets, offsets):
            duration = offset - onset
            
            if duration >= min_frames:
                # Keep segments that are long enough
                result[onset:offset] = 1
                
        return result


def process_vad_multi_track(vocal_tracks: Dict[str, str], 
                           settings: object) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Run voice activity detection on multiple tracks.
    
    Args:
        vocal_tracks: Dictionary mapping model names to vocal track paths
        settings: Settings object
        
    Returns:
        Dictionary mapping model names to (timestamps, vad_scores) tuples
    """
    if not vocal_tracks:
        raise ValueError("No vocal tracks provided for VAD processing")
    
    print(f"{ULTRASINGER_HEAD} Running voice activity detection on {len(vocal_tracks)} tracks")
    
    # Get debug level
    debug_level = getattr(settings, 'DEBUG_LEVEL', 1)
    
    # Initialize VAD detector with settings
    vad_threshold = getattr(settings, 'VAD_ENERGY_THRESHOLD', 0.05)
    min_duration = getattr(settings, 'MIN_NOTE_DURATION', 0.15)  # Increased from 0.12
    
    vad_detector = VocalActivityDetection(
        energy_threshold=vad_threshold,
        min_duration=min_duration
    )
    
    # Process each track
    vad_results = {}
    for model_name, track_path in vocal_tracks.items():
        try:
            print(f"{ULTRASINGER_HEAD} Processing VAD for {blue_highlighted(model_name)}")
            timestamps, vad_scores, sr = vad_detector.detect_from_file(track_path)
            
            # Explicitly ensure numpy arrays
            timestamps = np.asarray(timestamps)
            vad_scores = np.asarray(vad_scores)
            
            vad_results[model_name] = (timestamps, vad_scores)
            
            if debug_level >= 2:
                # Safe comparison using numpy
                vad_threshold_array = np.full_like(vad_scores, vad_threshold)
                vocal_percentage = np.sum(vad_scores > vad_threshold_array) / len(vad_scores) * 100
                print(f"{ULTRASINGER_HEAD} {blue_highlighted(model_name)} - "
                      f"detected {vocal_percentage:.1f}% as vocal activity")
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} processing VAD for "
                  f"{blue_highlighted(model_name)}: {e}")
    
    if not vad_results:
        raise ValueError("No VAD results could be generated from any vocal track")
    
    return vad_results


def combine_vad_results(vad_results: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                       model_weights: Dict[str, float] = None,
                       settings: object = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combine VAD results with weighted voting.
    
    Args:
        vad_results: Dictionary mapping model names to (timestamps, vad_scores) tuples
        model_weights: Dictionary mapping model names to weights
        settings: Settings object
        
    Returns:
        Tuple of (timestamps, combined_vad_scores)
    """
    if not vad_results:
        raise ValueError("No VAD results to combine")
    
    # Default weights if not provided
    if model_weights is None:
        model_weights = {
            "htdemucs_ft": 0.6,  # Most reliable
            "htdemucs": 0.3,     # Standard
            "htdemucs_6s": 0.4,  # More sources, better separation but can miss some vocals
            "mdx": 0.3,
            "mdx_extra": 0.4
        }
    
    # Fill in missing weights with default value (0.3)
    for model in vad_results.keys():
        if model not in model_weights:
            model_weights[model] = 0.3
    
    # Make sure all models in vad_results have weights
    available_models = list(vad_results.keys())
    available_weights = {model: model_weights.get(model, 0.3) for model in available_models}
    
    # Log the weights being used
    print(f"{ULTRASINGER_HEAD} Combining VAD results with weights:")
    for model, weight in available_weights.items():
        print(f"{ULTRASINGER_HEAD} - {blue_highlighted(model)}: {cyan_highlighted(str(weight))}")
    
    # Check if we need to resample the timestamps to a common timeline
    # For simplicity, we'll use the first model's timestamps as reference
    reference_model = available_models[0]
    reference_timestamps, _ = vad_results[reference_model]
    reference_timestamps = np.asarray(reference_timestamps)  # Ensure numpy array
    
    # Combine the VAD scores using weighted average
    combined_scores = np.zeros_like(vad_results[reference_model][1], dtype=float)
    total_weight = 0.0
    
    for model in available_models:
        timestamps, vad_scores = vad_results[model]
        timestamps = np.asarray(timestamps)  # Ensure numpy array
        vad_scores = np.asarray(vad_scores)  # Ensure numpy array
        weight = available_weights[model]
        
        # If timestamps don't match the reference, resample
        if len(timestamps) != len(reference_timestamps) or not np.allclose(timestamps, reference_timestamps):
            vad_scores = np.interp(reference_timestamps, timestamps, vad_scores)
        
        combined_scores += vad_scores * weight
        total_weight += weight
    
    # Normalize by total weight
    if total_weight > 0:
        combined_scores /= total_weight
    
    return reference_timestamps, combined_scores


def apply_vad_to_pitch_data(timestamps: np.ndarray,
                           vad_scores: np.ndarray,
                           pitched_data: PitchedData,
                           vad_threshold: float = 0.1,
                           confidence_boost: float = 0.2) -> PitchedData:
    """
    Apply VAD scores to enhance pitch data confidence.
    
    Args:
        timestamps: VAD timestamps
        vad_scores: VAD scores (0.0-1.0)
        pitched_data: Original pitch data
        vad_threshold: Threshold for considering a frame as vocal
        confidence_boost: Amount to boost confidence when vocal is detected
        
    Returns:
        Enhanced PitchedData with updated confidence values
    """
    # Ensure numpy arrays
    timestamps = np.asarray(timestamps)
    vad_scores = np.asarray(vad_scores)
    pitch_times = np.asarray(pitched_data.times)
    
    # Create a copy of the original data
    enhanced_data = PitchedData(
        times=pitched_data.times.copy(),
        frequencies=pitched_data.frequencies.copy(),
        confidence=pitched_data.confidence.copy()
    )
    
    # Interpolate VAD scores to match pitch data timestamps
    interpolated_vad = np.interp(pitch_times, timestamps, vad_scores)
    
    # Apply confidence adjustments based on VAD
    for i, time in enumerate(pitch_times):
        vad_score = interpolated_vad[i]
        
        if vad_score > vad_threshold:
            # Boost confidence for frames with vocal activity
            # The boost is proportional to the VAD score
            boost_factor = confidence_boost * (vad_score - vad_threshold) / (1.0 - vad_threshold)
            enhanced_data.confidence[i] = min(1.0, enhanced_data.confidence[i] + boost_factor)
        else:
            # Reduce confidence for frames without vocal activity
            reduction_factor = 0.5 * (vad_threshold - vad_score) / vad_threshold
            enhanced_data.confidence[i] = max(0.0, enhanced_data.confidence[i] - reduction_factor)
    
    return enhanced_data


def joint_vad_pitch_processing(vad_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
                              pitch_data_dict: Dict[str, PitchedData],
                              settings: object) -> PitchedData:
    """
    Process VAD and pitch data together for better note detection.
    
    Args:
        vad_results: Dictionary mapping model names to (timestamps, vad_scores)
        pitch_data_dict: Dictionary mapping model names to PitchedData
        settings: Settings object
        
    Returns:
        Enhanced PitchedData
    """
    # Get settings
    vad_threshold = getattr(settings, 'VAD_THRESHOLD', 0.15)
    confidence_boost = getattr(settings, 'VAD_CONFIDENCE_BOOST', 0.2)
    debug_level = getattr(settings, 'DEBUG_LEVEL', 1)
    
    # Get model weights based on settings
    model_weights = getattr(settings, 'VAD_MODEL_WEIGHTS', {
        "htdemucs_ft": 0.6,
        "htdemucs": 0.3,
        "htdemucs_6s": 0.4
    })
    
    # Combine VAD results
    try:
        timestamps, combined_vad = combine_vad_results(vad_results, model_weights, settings)
        
        # Ensure numpy arrays
        timestamps = np.asarray(timestamps)
        combined_vad = np.asarray(combined_vad)
        
        # Get a combined pitch data as baseline (using existing pitch combiner)
        # We're assuming this has been done earlier and is passed in pitch_data_dict
        if len(pitch_data_dict) == 1:
            combined_pitch = next(iter(pitch_data_dict.values()))
        else:
            # Get the most complete pitch data as reference
            reference_model = max(pitch_data_dict.items(), key=lambda x: len(x[1].times))[0]
            combined_pitch = pitch_data_dict[reference_model]
        
        # Apply VAD to enhance pitch confidence
        enhanced_pitch = apply_vad_to_pitch_data(
            timestamps, 
            combined_vad, 
            combined_pitch,
            vad_threshold=vad_threshold,
            confidence_boost=confidence_boost
        )
        
        # Print statistics
        if debug_level >= 1:
            # Safe comparison using numpy
            threshold_array = np.full_like(combined_pitch.confidence, 0.5)
            original_high_conf = np.sum(combined_pitch.confidence > threshold_array) / len(combined_pitch.confidence)
            
            threshold_array = np.full_like(enhanced_pitch.confidence, 0.5)
            enhanced_high_conf = np.sum(enhanced_pitch.confidence > threshold_array) / len(enhanced_pitch.confidence)
            
            print(f"{ULTRASINGER_HEAD} VAD enhancement statistics:")
            print(f"{ULTRASINGER_HEAD} - Original high confidence frames: {original_high_conf*100:.1f}%")
            print(f"{ULTRASINGER_HEAD} - Enhanced high confidence frames: {enhanced_high_conf*100:.1f}%")
            print(f"{ULTRASINGER_HEAD} - Change: {(enhanced_high_conf-original_high_conf)*100:+.1f}%")
        
        # Visualize VAD and pitch if debug level is high enough
        if debug_level >= 2 and hasattr(settings, 'output_folder_path'):
            output_dir = os.path.join(settings.output_folder_path, "cache")
            visualize_vad_pitch(timestamps, combined_vad, enhanced_pitch, output_dir)
        
        return enhanced_pitch
    
    except Exception as e:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} in joint VAD processing: {e}")
        # Return the original pitch data as fallback
        if len(pitch_data_dict) == 1:
            return next(iter(pitch_data_dict.values()))
        else:
            reference_model = max(pitch_data_dict.items(), key=lambda x: len(x[1].times))[0]
            return pitch_data_dict[reference_model]


def visualize_vad_pitch(timestamps: np.ndarray, 
                       vad_scores: np.ndarray, 
                       pitch_data: PitchedData,
                       output_dir: str) -> None:
    """
    Create visualization of VAD and pitch data.
    
    Args:
        timestamps: VAD timestamps
        vad_scores: VAD scores
        pitch_data: Pitch data
        output_dir: Directory to save the visualization
    """
    # Ensure numpy arrays
    timestamps = np.asarray(timestamps)
    vad_scores = np.asarray(vad_scores)
    pitch_times = np.asarray(pitch_data.times)
    pitch_freqs = np.asarray(pitch_data.frequencies)
    pitch_confs = np.asarray(pitch_data.confidence)
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: VAD scores
    plt.subplot(3, 1, 1)
    plt.plot(timestamps, vad_scores, 'g-', label='VAD Score')
    plt.ylabel('Voice Activity')
    plt.title('Voice Activity Detection')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    # Plot 2: Pitch frequency
    plt.subplot(3, 1, 2)
    plt.scatter(pitch_times, pitch_freqs, 
               c=pitch_confs, cmap='viridis', 
               s=5, alpha=0.6)
    plt.colorbar(label='Confidence')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch with Confidence')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Confidence with VAD overlay
    plt.subplot(3, 1, 3)
    plt.plot(pitch_times, pitch_confs, 'b-', label='Confidence')
    
    # Interpolate VAD to match pitch times
    interp_vad = np.interp(pitch_times, timestamps, vad_scores)
    plt.plot(pitch_times, interp_vad, 'g-', alpha=0.5, label='VAD Score')
    
    plt.ylabel('Value')
    plt.xlabel('Time (s)')
    plt.title('Confidence vs. VAD Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vad_pitch_analysis.png'), dpi=300)
    plt.close()
    
    print(f"{ULTRASINGER_HEAD} Saved VAD/pitch visualization to {blue_highlighted(output_dir)}")


def segment_notes_with_vad(pitched_data: PitchedData,
                          timestamps: np.ndarray,
                          vad_scores: np.ndarray,
                          settings: object) -> List[MidiSegment]:
    """
    Segment notes using VAD information for better note boundaries.
    
    Args:
        pitched_data: Pitch data
        timestamps: VAD timestamps
        vad_scores: VAD scores
        settings: Settings object
        
    Returns:
        List of MidiSegment objects
    """
    # Ensure numpy arrays
    timestamps = np.asarray(timestamps)
    vad_scores = np.asarray(vad_scores)
    pitch_times = np.asarray(pitched_data.times)
    
    # Get settings
    vad_threshold = getattr(settings, 'VAD_THRESHOLD', 0.15)
    min_note_duration = getattr(settings, 'MIN_NOTE_DURATION', 0.15)  # Increased from 0.12
    debug_level = getattr(settings, 'DEBUG_LEVEL', 1)
    
    # Interpolate VAD scores to match pitch data timestamps
    vad_for_pitch = np.interp(pitch_times, timestamps, vad_scores)
    
    # Create binary mask of vocal activity using safe numpy comparison
    threshold_array = np.full_like(vad_for_pitch, vad_threshold)
    vocal_mask = vad_for_pitch > threshold_array
    
    # Find contiguous segments of vocal activity
    segment_boundaries = []
    in_segment = False
    segment_start = 0
    
    for i, is_vocal in enumerate(vocal_mask):
        if is_vocal and not in_segment:
            # Start of a new segment
            segment_start = i
            in_segment = True
        elif not is_vocal and in_segment:
            # End of a segment
            segment_end = i
            segment_duration = pitch_times[segment_end] - pitch_times[segment_start]
            if segment_duration >= min_note_duration:
                segment_boundaries.append((segment_start, segment_end))
            in_segment = False
    
    # Handle the case where the last segment extends to the end
    if in_segment:
        segment_end = len(vocal_mask)
        if segment_end > 0 and segment_start < len(pitch_times):
            segment_duration = pitch_times[segment_end-1] - pitch_times[segment_start]
            if segment_duration >= min_note_duration:
                segment_boundaries.append((segment_start, segment_end-1))
    
    # For each segment, determine the dominant pitch
    midi_segments = []
    
    for start_idx, end_idx in segment_boundaries:
        if start_idx >= len(pitch_times) or end_idx > len(pitch_times) or start_idx >= end_idx:
            continue  # Skip invalid indices
            
        segment_times = pitch_times[start_idx:end_idx]
        segment_freqs = pitched_data.frequencies[start_idx:end_idx]
        segment_confs = pitched_data.confidence[start_idx:end_idx]
        
        if len(segment_times) == 0:
            continue
            
        # Get weighted average frequency based on confidence
        try:
            weights = np.asarray(segment_confs)
            if np.sum(weights) > 0:
                avg_freq = np.average(segment_freqs, weights=weights)
            else:
                avg_freq = np.mean(segment_freqs)
            
            # Convert to note
            note = librosa.hz_to_note(avg_freq)
            
            # Create MIDI segment
            segment = MidiSegment(
                note=note,
                start=segment_times[0],
                end=segment_times[-1],
                word="♪"  # Placeholder
            )
            
            midi_segments.append(segment)
        except Exception as e:
            print(f"{ULTRASINGER_HEAD} {red_highlighted('Error')} processing segment {start_idx}-{end_idx}: {e}")
    
    if debug_level >= 1:
        print(f"{ULTRASINGER_HEAD} Generated {len(midi_segments)} note segments using VAD boundaries")
    
    return midi_segments