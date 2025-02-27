"""Pitch data combiner for multi-track pitch processing."""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from typing import Dict, List, Tuple, Optional

from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, cyan_highlighted, green_highlighted, red_highlighted
from modules.Pitcher.pitched_data import PitchedData
from modules.Pitcher.pitched_data_helper import get_frequencies_with_high_confidence


def combine_pitch_data(
    pitch_data_dict: Dict[str, PitchedData], 
    confidence_thresholds: Dict[str, float] = None,
    agreement_bonus: float = 0.2,
    debug_level: int = 1,
    output_dir: Optional[str] = None
) -> PitchedData:
    """
    Combine pitch data from multiple tracks with intelligent weighting.
    
    Args:
        pitch_data_dict: Dictionary mapping model names to their PitchedData
        confidence_thresholds: Model-specific confidence thresholds, defaults to 0.4 for all models
        agreement_bonus: Confidence boost when tracks agree on pitch (within semitone)
        debug_level: 0=minimal, 1=basic stats, 2=visualizations
        output_dir: Directory to save visualizations if debug_level >= 2
        
    Returns:
        Combined PitchedData object
    """
    if not pitch_data_dict:
        raise ValueError("No pitch data provided for combination")
    
    # If only one track, return it directly
    if len(pitch_data_dict) == 1:
        model_name = next(iter(pitch_data_dict.keys()))
        print(f"{ULTRASINGER_HEAD} Using single pitch track from {blue_highlighted(model_name)}")
        return pitch_data_dict[model_name]
    
    # Use default thresholds if none provided
    if confidence_thresholds is None:
        confidence_thresholds = {model: 0.4 for model in pitch_data_dict.keys()}
    
    # Fill in missing thresholds with default
    for model in pitch_data_dict.keys():
        if model not in confidence_thresholds:
            confidence_thresholds[model] = 0.4
    
    # Log the start of the combination process
    print(f"{ULTRASINGER_HEAD} Combining pitch data from {len(pitch_data_dict)} tracks")
    for model, threshold in confidence_thresholds.items():
        if model in pitch_data_dict:
            print(f"{ULTRASINGER_HEAD} - {blue_highlighted(model)}: threshold {cyan_highlighted(str(threshold))}, "
                  f"{len(pitch_data_dict[model].times)} time points")
    
    # Find the reference track with most time points (usually they should be the same)
    reference_model = max(pitch_data_dict.items(), key=lambda x: len(x[1].times))[0]
    reference_data = pitch_data_dict[reference_model]
    
    # Initialize the combined data with the reference model's time points
    combined_times = reference_data.times
    combined_frequencies = []
    combined_confidences = []
    
    # Track selection metrics for debugging
    model_selections = {model: 0 for model in pitch_data_dict.keys()}
    agreement_count = 0
    total_points = len(combined_times)
    
    # For plotting if debug_level >= 2
    if debug_level >= 2 and output_dir:
        plot_data = {
            'times': combined_times,
            'models': {model: [] for model in pitch_data_dict.keys()},
            'confidences': {model: [] for model in pitch_data_dict.keys()},
            'combined': [],
            'combined_conf': [],
            'selected_model': []
        }
    
    # Process each time point
    for i, time in enumerate(combined_times):
        # Collect data from all models at this time point
        time_data = {}
        for model, pitched_data in pitch_data_dict.items():
            # Find the closest time point in this model's data
            # (Should be exact match in most cases since CREPE uses fixed step sizes)
            idx = find_closest_time_index(pitched_data.times, time)
            if idx < len(pitched_data.times):
                time_data[model] = {
                    'frequency': pitched_data.frequencies[idx],
                    'confidence': pitched_data.confidence[idx],
                    'adjusted_confidence': pitched_data.confidence[idx] * (confidence_thresholds[model] / 0.4)
                }
        
        # Check for pitch agreement and apply bonus
        frequencies = [data['frequency'] for data in time_data.values()]
        has_agreement = check_pitch_agreement(frequencies)
        
        if has_agreement:
            agreement_count += 1
            # Apply confidence bonus to all models that agree
            for model in time_data:
                time_data[model]['adjusted_confidence'] += agreement_bonus
        
        # Select the best model for this time point based on adjusted confidence
        if time_data:
            best_model = max(time_data.items(), key=lambda x: x[1]['adjusted_confidence'])[0]
            best_data = time_data[best_model]
            
            combined_frequencies.append(best_data['frequency'])
            combined_confidences.append(best_data['confidence'])  # Use original confidence
            
            # Track which model was selected
            model_selections[best_model] += 1
            
            # Add data for plotting
            if debug_level >= 2 and output_dir:
                for model in pitch_data_dict.keys():
                    if model in time_data:
                        plot_data['models'][model].append(time_data[model]['frequency'])
                        plot_data['confidences'][model].append(time_data[model]['confidence'])
                    else:
                        plot_data['models'][model].append(np.nan)
                        plot_data['confidences'][model].append(np.nan)
                plot_data['combined'].append(best_data['frequency'])
                plot_data['combined_conf'].append(best_data['confidence'])
                plot_data['selected_model'].append(best_model)
        else:
            # No data available for this time point, use reference model's data
            combined_frequencies.append(reference_data.frequencies[i])
            combined_confidences.append(reference_data.confidence[i])
            
            if debug_level >= 2 and output_dir:
                for model in pitch_data_dict.keys():
                    plot_data['models'][model].append(np.nan)
                    plot_data['confidences'][model].append(np.nan)
                plot_data['combined'].append(reference_data.frequencies[i])
                plot_data['combined_conf'].append(reference_data.confidence[i])
                plot_data['selected_model'].append(reference_model)
    
    # Create the combined PitchedData
    combined_data = PitchedData(
        times=combined_times,
        frequencies=combined_frequencies,
        confidence=combined_confidences
    )
    
    # Print stats
    if debug_level >= 1:
        print(f"{ULTRASINGER_HEAD} Pitch track combination statistics:")
        print(f"{ULTRASINGER_HEAD} - Agreement between tracks: {green_highlighted(f'{agreement_count/total_points*100:.1f}%')}")
        print(f"{ULTRASINGER_HEAD} - Model contributions:")
        for model, count in model_selections.items():
            print(f"{ULTRASINGER_HEAD}   - {blue_highlighted(model)}: {cyan_highlighted(f'{count/total_points*100:.1f}%')}")
    
    # Generate visualization if requested
    if debug_level >= 2 and output_dir:
        visualize_pitch_combination(plot_data, output_dir)
    
    return combined_data


def find_closest_time_index(times: List[float], target_time: float) -> int:
    """Find the index of the closest time point to the target time."""
    return min(range(len(times)), key=lambda i: abs(times[i] - target_time))


def check_pitch_agreement(frequencies: List[float], semitone_threshold: float = 1.0) -> bool:
    """
    Check if the frequencies agree within a semitone threshold.
    
    Args:
        frequencies: List of frequencies in Hz
        semitone_threshold: Maximum semitone difference to be considered in agreement
        
    Returns:
        True if frequencies agree, False otherwise
    """
    if len(frequencies) < 2:
        return False
    
    # Convert to MIDI note numbers (log scale) for semitone comparison
    midi_notes = [librosa.hz_to_midi(freq) for freq in frequencies]
    
    # Check if max difference is within threshold
    max_diff = max(midi_notes) - min(midi_notes)
    return max_diff <= semitone_threshold


def visualize_pitch_combination(plot_data: dict, output_dir: str) -> None:
    """
    Create visualization of the pitch combination process.
    
    Args:
        plot_data: Dictionary with plotting data
        output_dir: Directory to save the plots
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Convert frequencies to MIDI note numbers for better visualization
    midi_combined = [librosa.hz_to_midi(freq) if not np.isnan(freq) else np.nan for freq in plot_data['combined']]
    midi_models = {}
    for model, freqs in plot_data['models'].items():
        midi_models[model] = [librosa.hz_to_midi(freq) if not np.isnan(freq) else np.nan for freq in freqs]
    
    # Plot the combined pitch track
    plt.plot(plot_data['times'], midi_combined, 'k-', linewidth=2, label='Combined')
    
    # Plot each model's pitch track with transparency
    for model, midi_notes in midi_models.items():
        plt.plot(plot_data['times'], midi_notes, alpha=0.5, label=model)
    
    # Color-code points by selected model
    unique_models = set(plot_data['selected_model'])
    for model in unique_models:
        indices = [i for i, m in enumerate(plot_data['selected_model']) if m == model]
        plt.scatter(
            [plot_data['times'][i] for i in indices],
            [midi_combined[i] for i in indices],
            label=f"{model} selected",
            alpha=0.7,
            s=20
        )
    
    plt.xlabel('Time (s)')
    plt.ylabel('MIDI Note Number')
    plt.title('Multi-Track Pitch Combination')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pitch_combination.png'), dpi=300)
    plt.close()
    
    # Create a confidence comparison plot
    plt.figure(figsize=(12, 6))
    
    # Plot each model's confidence
    for model, confs in plot_data['confidences'].items():
        plt.plot(plot_data['times'], confs, alpha=0.5, label=f"{model} confidence")
    
    # Plot the combined confidence
    plt.plot(plot_data['times'], plot_data['combined_conf'], 'k-', linewidth=2, label='Combined confidence')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Confidence')
    plt.title('Multi-Track Confidence Comparison')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_comparison.png'), dpi=300)
    plt.close()
    
    print(f"{ULTRASINGER_HEAD} Saved pitch combination visualizations to {blue_highlighted(output_dir)}")