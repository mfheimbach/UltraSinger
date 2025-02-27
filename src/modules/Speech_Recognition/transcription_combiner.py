"""Transcription combiner for multi-track vocal processing.

This module provides functions for combining transcriptions from multiple
separated vocal tracks to achieve more accurate lyrics and timing.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import Levenshtein

from modules.Speech_Recognition.TranscribedData import TranscribedData
from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted


@dataclass
class TranscriptionVariant:
    """A variant of a transcribed word from a specific model."""
    word: str
    start: float
    end: float
    confidence: float
    model: str


@dataclass
class AlignedTranscriptionWord:
    """A word aligned across multiple transcription variants."""
    variants: List[TranscriptionVariant]
    selected_word: str = ""
    final_start: float = 0.0
    final_end: float = 0.0
    final_confidence: float = 0.0


def time_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """
    Calculate the overlap between two time ranges.
    
    Args:
        start1: Start time of first range
        end1: End time of first range
        start2: Start time of second range
        end2: End time of second range
        
    Returns:
        Overlap duration in seconds
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap = max(0, overlap_end - overlap_start)
    
    # Calculate overlap as a percentage of the shorter segment
    duration1 = end1 - start1
    duration2 = end2 - start2
    shorter_duration = min(duration1, duration2)
    
    if shorter_duration > 0:
        return overlap / shorter_duration
    return 0.0


def word_similarity(word1: str, word2: str) -> float:
    """
    Calculate similarity between two words using Levenshtein distance.
    
    Args:
        word1: First word
        word2: Second word
        
    Returns:
        Similarity score (0.0-1.0)
    """
    if not word1 or not word2:
        return 0.0
        
    # Remove trailing spaces for comparison
    word1 = word1.strip()
    word2 = word2.strip()
    
    if not word1 or not word2:
        return 0.0
    
    # Calculate Levenshtein ratio
    return Levenshtein.ratio(word1.lower(), word2.lower())


def align_transcriptions(
    transcriptions_by_model: Dict[str, List[TranscribedData]],
    similarity_threshold: float = 0.7,
    overlap_threshold: float = 0.3
) -> List[AlignedTranscriptionWord]:
    """
    Align words across different transcriptions based on timing and text similarity.
    
    Args:
        transcriptions_by_model: Dictionary mapping model names to transcription data
        similarity_threshold: Minimum word similarity to consider a match
        overlap_threshold: Minimum time overlap to consider a match
        
    Returns:
        List of aligned words
    """
    print(f"{ULTRASINGER_HEAD} Aligning transcriptions from {len(transcriptions_by_model)} models")
    
    if not transcriptions_by_model:
        return []
    
    # Convert to TranscriptionVariant objects
    all_variants = []
    for model, transcription in transcriptions_by_model.items():
        for word_data in transcription:
            variant = TranscriptionVariant(
                word=word_data.word,
                start=word_data.start,
                end=word_data.end,
                confidence=word_data.confidence,
                model=model
            )
            all_variants.append(variant)
    
    # Sort all variants by start time
    all_variants.sort(key=lambda v: v.start)
    
    # Initialize aligned words
    aligned_words = []
    processed = set()
    
    # For each variant, find matching variants from other models
    for i, variant in enumerate(all_variants):
        if i in processed:
            continue
            
        aligned_word = AlignedTranscriptionWord(variants=[variant])
        processed.add(i)
        
        # Check for matching variants from other models
        for j, other_variant in enumerate(all_variants):
            if j in processed or other_variant.model == variant.model:
                continue
                
            # Calculate time overlap and word similarity
            overlap = time_overlap(
                variant.start, variant.end,
                other_variant.start, other_variant.end
            )
            
            similarity = word_similarity(variant.word, other_variant.word)
            
            # Check if this is a match
            if (overlap >= overlap_threshold and similarity >= similarity_threshold):
                aligned_word.variants.append(other_variant)
                processed.add(j)
        
        aligned_words.append(aligned_word)
    
    # Sort the final aligned words by the average start time
    for word in aligned_words:
        word.variants.sort(key=lambda v: v.start)
    
    aligned_words.sort(key=lambda w: sum(v.start for v in w.variants) / len(w.variants))
    
    print(f"{ULTRASINGER_HEAD} Aligned {len(aligned_words)} words across models")
    
    return aligned_words


def select_best_word(
    aligned_word: AlignedTranscriptionWord,
    model_weights: Dict[str, float]
) -> str:
    """
    Select the best word variant based on model weights and confidence.
    
    Args:
        aligned_word: Aligned word with variants
        model_weights: Dictionary mapping model names to weights
        
    Returns:
        Selected word text
    """
    if not aligned_word.variants:
        return ""
    
    # If only one variant, use it
    if len(aligned_word.variants) == 1:
        return aligned_word.variants[0].word
    
    # Calculate weighted scores for each variant
    weighted_scores = []
    for variant in aligned_word.variants:
        model_weight = model_weights.get(variant.model, 1.0)
        score = variant.confidence * model_weight
        weighted_scores.append((variant, score))
    
    # Select the variant with the highest score
    best_variant, _ = max(weighted_scores, key=lambda x: x[1])
    
    return best_variant.word


def calculate_weighted_timing(
    aligned_word: AlignedTranscriptionWord,
    model_weights: Dict[str, float],
    dominant_model: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Calculate weighted start/end timing for a word.
    
    Args:
        aligned_word: Aligned word with variants
        model_weights: Dictionary mapping model names to weights
        dominant_model: Model to use when there's doubt
        
    Returns:
        Tuple of (start_time, end_time, confidence)
    """
    if not aligned_word.variants:
        return 0.0, 0.0, 0.0
    
    # If only one variant, use its timing
    if len(aligned_word.variants) == 1:
        variant = aligned_word.variants[0]
        return variant.start, variant.end, variant.confidence
    
    # Check if the dominant model is present
    if dominant_model:
        dominant_variants = [v for v in aligned_word.variants if v.model == dominant_model]
        if dominant_variants:
            # Use the dominant model's timing
            variant = dominant_variants[0]
            return variant.start, variant.end, variant.confidence
    
    # Calculate weighted average timing
    weighted_start_sum = 0.0
    weighted_end_sum = 0.0
    weighted_conf_sum = 0.0
    total_weight = 0.0
    
    for variant in aligned_word.variants:
        model_weight = model_weights.get(variant.model, 1.0)
        combined_weight = model_weight * variant.confidence
        
        weighted_start_sum += variant.start * combined_weight
        weighted_end_sum += variant.end * combined_weight
        weighted_conf_sum += variant.confidence * model_weight
        total_weight += combined_weight
    
    # Calculate weighted averages
    if total_weight > 0:
        weighted_start = weighted_start_sum / total_weight
        weighted_end = weighted_end_sum / total_weight
        weighted_conf = weighted_conf_sum / sum(model_weights.get(v.model, 1.0) for v in aligned_word.variants)
    else:
        # Fallback if no weights
        weighted_start = sum(v.start for v in aligned_word.variants) / len(aligned_word.variants)
        weighted_end = sum(v.end for v in aligned_word.variants) / len(aligned_word.variants)
        weighted_conf = sum(v.confidence for v in aligned_word.variants) / len(aligned_word.variants)
    
    return weighted_start, weighted_end, weighted_conf


def combine_transcriptions(
    transcriptions_by_model: Dict[str, List[TranscribedData]],
    model_weights: Dict[str, float],
    dominant_model: Optional[str] = None
) -> List[TranscribedData]:
    """
    Combine transcriptions from multiple models into a single optimized transcription.
    
    Args:
        transcriptions_by_model: Dictionary mapping model names to transcription data
        model_weights: Dictionary mapping model names to weights
        dominant_model: Model to use when there's doubt
        
    Returns:
        Combined transcription data
    """
    # Step 1: Align words across transcriptions
    aligned_words = align_transcriptions(transcriptions_by_model)
    
    # Step 2: Process each aligned word
    combined_data = []
    
    for aligned_word in aligned_words:
        # Select the best word
        selected_word = select_best_word(aligned_word, model_weights)
        
        # Calculate weighted timing
        start_time, end_time, confidence = calculate_weighted_timing(
            aligned_word, model_weights, dominant_model
        )
        
        # Create combined transcribed data
        combined_word = TranscribedData(
            word=selected_word,
            start=start_time,
            end=end_time,
            confidence=confidence
        )
        
        combined_data.append(combined_word)
    
    print(f"{ULTRASINGER_HEAD} Created combined transcription with {len(combined_data)} words")
    
    # Clean up timing issues (ensure non-overlapping word boundaries)
    combined_data = _clean_word_boundaries(combined_data)
    
    return combined_data


def _clean_word_boundaries(transcribed_data: List[TranscribedData]) -> List[TranscribedData]:
    """
    Clean up word boundaries to ensure no overlaps and proper spacing.
    
    Args:
        transcribed_data: List of transcribed words
        
    Returns:
        Cleaned transcribed data
    """
    if not transcribed_data:
        return []
    
    # Sort by start time
    sorted_data = sorted(transcribed_data, key=lambda x: x.start)
    
    # Fix overlaps
    for i in range(1, len(sorted_data)):
        prev_word = sorted_data[i-1]
        curr_word = sorted_data[i]
        
        # Check for overlap
        if prev_word.end > curr_word.start:
            # Set the boundary at the midpoint
            midpoint = (prev_word.end + curr_word.start) / 2
            
            # Ensure minimum durations
            min_duration = 0.05  # 50ms minimum
            
            if midpoint - prev_word.start < min_duration:
                midpoint = prev_word.start + min_duration
                
            if curr_word.end - midpoint < min_duration:
                midpoint = curr_word.end - min_duration
                
            # Update boundaries
            prev_word.end = midpoint
            curr_word.start = midpoint
    
    return sorted_data


def combine_transcription_results(
    results_by_model: Dict[str, TranscriptionResult],
    model_weights: Dict[str, float],
    dominant_model: Optional[str] = None
) -> TranscriptionResult:
    """
    Combine multiple TranscriptionResults into a single optimized result.
    
    Args:
        results_by_model: Dictionary mapping model names to TranscriptionResult
        model_weights: Dictionary mapping model names to weights
        dominant_model: Model to use when there's doubt
        
    Returns:
        Combined TranscriptionResult
    """
    if not results_by_model:
        raise ValueError("No transcription results to combine")
    
    # Extract transcribed data by model
    transcriptions_by_model = {
        model: result.transcribed_data
        for model, result in results_by_model.items()
    }
    
    # Combine transcriptions
    combined_data = combine_transcriptions(
        transcriptions_by_model, model_weights, dominant_model
    )
    
    # For language detection, use weighted voting
    language_votes = {}
    
    for model, result in results_by_model.items():
        lang = result.detected_language
        weight = model_weights.get(model, 1.0)
        
        if lang in language_votes:
            language_votes[lang] += weight
        else:
            language_votes[lang] = weight
    
    # Select language with highest weighted votes
    detected_language = max(language_votes.items(), key=lambda x: x[1])[0]
    
    # Create combined result
    return TranscriptionResult(
        transcribed_data=combined_data,
        detected_language=detected_language
    )


def analyze_transcription_differences(
    transcriptions_by_model: Dict[str, List[TranscribedData]]
) -> Dict[str, Any]:
    """
    Analyze differences between transcriptions to identify model strengths/weaknesses.
    
    Args:
        transcriptions_by_model: Dictionary mapping model names to transcription data
        
    Returns:
        Dictionary of analysis metrics
    """
    if not transcriptions_by_model or len(transcriptions_by_model) < 2:
        return {"error": "Need at least two transcriptions to compare"}
    
    # Count words per model
    word_counts = {model: len(trans) for model, trans in transcriptions_by_model.items()}
    
    # Calculate average word confidence per model
    avg_confidence = {
        model: sum(word.confidence for word in trans) / len(trans) if trans else 0
        for model, trans in transcriptions_by_model.items()
    }
    
    # Align transcriptions to check agreement
    aligned_words = align_transcriptions(transcriptions_by_model)
    
    # Calculate agreement rate
    model_pairs = []
    for model1 in transcriptions_by_model:
        for model2 in transcriptions_by_model:
            if model1 < model2:  # Avoid duplicates
                model_pairs.append((model1, model2))
    
    agreement_rates = {}
    for model1, model2 in model_pairs:
        matching_words = 0
        total_comparisons = 0
        
        for aligned_word in aligned_words:
            variants_by_model = {v.model: v for v in aligned_word.variants}
            
            if model1 in variants_by_model and model2 in variants_by_model:
                word1 = variants_by_model[model1].word.strip().lower()
                word2 = variants_by_model[model2].word.strip().lower()
                
                if word1 == word2:
                    matching_words += 1
                
                total_comparisons += 1
        
        if total_comparisons > 0:
            agreement_rates[f"{model1}-{model2}"] = matching_words / total_comparisons
    
    # Timing variation analysis
    timing_variations = {}
    for aligned_word in aligned_words:
        if len(aligned_word.variants) > 1:
            start_times = [v.start for v in aligned_word.variants]
            end_times = [v.end for v in aligned_word.variants]
            
            start_var = np.var(start_times) if len(start_times) > 1 else 0
            end_var = np.var(end_times) if len(end_times) > 1 else 0
            
            word = aligned_word.variants[0].word.strip()
            if word:
                timing_variations[word] = (start_var, end_var)
    
    # Calculate average timing variance
    avg_start_var = np.mean([v[0] for v in timing_variations.values()]) if timing_variations else 0
    avg_end_var = np.mean([v[1] for v in timing_variations.values()]) if timing_variations else 0
    
    # Return analysis
    return {
        "word_counts": word_counts,
        "avg_confidence": avg_confidence,
        "agreement_rates": agreement_rates,
        "avg_timing_variance": {
            "start": avg_start_var,
            "end": avg_end_var
        }
    }


def visualize_combined_transcription(
    original_transcriptions: Dict[str, List[TranscribedData]],
    combined_transcription: List[TranscribedData],
    output_path: str
) -> None:
    """
    Create visualization comparing original and combined transcriptions.
    
    Args:
        original_transcriptions: Dictionary mapping model names to transcription data
        combined_transcription: Combined transcription data
        output_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import os
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Plot original transcriptions
    colors = ['lightblue', 'lightgreen', 'lightsalmon', 'lightpink']
    y_positions = {}
    
    for i, (model, transcription) in enumerate(original_transcriptions.items()):
        y_pos = i + 1
        y_positions[model] = y_pos
        
        for word in transcription:
            plt.barh(
                y_pos,
                word.end - word.start,
                left=word.start,
                height=0.6,
                color=colors[i % len(colors)],
                alpha=0.7,
                label=model if word == transcription[0] else None
            )
            
            # Add word text
            plt.text(
                word.start + (word.end - word.start) / 2,
                y_pos,
                word.word.strip(),
                ha='center',
                va='center',
                fontsize=8
            )
    
    # Plot combined transcription
    y_pos = len(original_transcriptions) + 2
    
    for word in combined_transcription:
        plt.barh(
            y_pos,
            word.end - word.start,
            left=word.start,
            height=0.8,
            color='darkblue',
            alpha=0.8,
            label='Combined' if word == combined_transcription[0] else None
        )
        
        # Add word text
        plt.text(
            word.start + (word.end - word.start) / 2,
            y_pos,
            word.word.strip(),
            ha='center',
            va='center',
            color='white',
            fontsize=8
        )
    
    # Configure plot
    plt.yticks(
        list(y_positions.values()) + [y_pos],
        list(y_positions.keys()) + ['Combined']
    )
    
    plt.xlabel('Time (seconds)')
    plt.title('Multi-Track Transcription Comparison')
    plt.legend()
    
    # Set reasonable x-axis limits
    all_words = []
    for trans in original_transcriptions.values():
        all_words.extend(trans)
    all_words.extend(combined_transcription)
    
    if all_words:
        min_start = min(word.start for word in all_words)
        max_end = max(word.end for word in all_words)
        
        # Add some padding
        padding = (max_end - min_start) * 0.05
        plt.xlim(min_start - padding, max_end + padding)
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"{ULTRASINGER_HEAD} Saved transcription visualization to {output_path}")