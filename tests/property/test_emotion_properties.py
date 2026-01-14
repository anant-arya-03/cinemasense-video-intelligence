"""
Property-based tests for Emotion Rhythm Score Analysis.

Feature: cinemasense-stabilization
Property 10: Emotion Analysis Output Completeness
Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6

Tests that for any video analysis, the EmotionRhythmResult SHALL contain:
- A non-empty timeline with valid timestamps
- emotion_distribution with values summing to 1.0
- A valid rhythm_pattern from the defined set
- heatmap_data with correct dimensions
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hypothesis import given, strategies as st, settings, assume
import math

from cinemasense.pipeline.emotion_rhythm import (
    EmotionRhythmResult,
    EmotionFrame,
    EMOTION_CATEGORIES,
    RHYTHM_PATTERNS,
    analyze_frame_emotion,
    classify_emotion,
    _calculate_emotion_distribution,
    _create_heatmap,
    determine_rhythm_pattern,
    extract_emotion_timeline,
)


# Strategy for generating valid frame features
@st.composite
def valid_features_strategy(draw):
    """Generate valid frame features dictionary."""
    return {
        "brightness": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "saturation": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "color_temperature": draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
        "contrast": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        "motion_intensity": draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    }


# Strategy for generating valid BGR frames
@st.composite
def valid_bgr_frame_strategy(draw):
    """Generate a valid BGR frame with random pixel values."""
    height = draw(st.integers(min_value=10, max_value=50))
    width = draw(st.integers(min_value=10, max_value=50))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# Strategy for generating emotion counts
@st.composite
def emotion_counts_strategy(draw):
    """Generate valid emotion counts dictionary."""
    counts = {}
    for emotion in EMOTION_CATEGORIES:
        counts[emotion] = draw(st.integers(min_value=0, max_value=100))
    # Ensure at least one emotion has a count > 0
    if sum(counts.values()) == 0:
        counts[draw(st.sampled_from(EMOTION_CATEGORIES))] = draw(st.integers(min_value=1, max_value=100))
    return counts


# Strategy for generating emotion timeline
@st.composite
def emotion_timeline_strategy(draw):
    """Generate a valid emotion timeline."""
    num_frames = draw(st.integers(min_value=1, max_value=50))
    timeline = []
    
    for i in range(num_frames):
        timestamp = i * 0.1  # 100ms intervals
        emotion = draw(st.sampled_from(EMOTION_CATEGORIES))
        
        frame = EmotionFrame(
            timestamp=timestamp,
            frame_index=i,
            brightness=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            saturation=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            motion_intensity=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            color_temperature=draw(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)),
            contrast=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
            emotion_score=draw(st.floats(min_value=0.0, max_value=100.0, allow_nan=False)),
            dominant_emotion=emotion,
        )
        timeline.append(frame)
    
    return timeline


class TestEmotionAnalysisOutputCompleteness:
    """
    Property tests for Emotion Analysis Output Completeness.
    
    Feature: cinemasense-stabilization, Property 10: Emotion Analysis Output Completeness
    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6
    """
    
    @given(features=valid_features_strategy())
    @settings(max_examples=100)
    def test_emotion_classification_returns_valid_category(self, features):
        """
        Property 10: Emotion Classification Validity (part of completeness)
        
        For any valid frame features, emotion classification SHALL return
        one of the valid emotion categories with confidence in [0, 1].
        
        Validates: Requirements 5.2
        """
        emotion, confidence = classify_emotion(features)
        
        # Verify emotion is a valid category
        assert emotion in EMOTION_CATEGORIES, (
            f"Emotion '{emotion}' not in valid categories: {EMOTION_CATEGORIES}"
        )
        
        # Verify confidence is in [0, 1]
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} should be in range [0, 1]"
        )
    
    @given(emotion_counts=emotion_counts_strategy())
    @settings(max_examples=100)
    def test_emotion_distribution_sums_to_one(self, emotion_counts):
        """
        Property 10: Emotion Distribution Normalization
        
        For any emotion counts, the calculated distribution SHALL have
        values that sum to 1.0 (within floating point tolerance).
        
        Validates: Requirements 5.3
        """
        distribution = _calculate_emotion_distribution(emotion_counts)
        
        # Verify all emotions are present
        assert set(distribution.keys()) == set(EMOTION_CATEGORIES), (
            f"Distribution should contain all emotion categories.\n"
            f"Expected: {EMOTION_CATEGORIES}\n"
            f"Got: {list(distribution.keys())}"
        )
        
        # Verify values sum to 1.0
        total = sum(distribution.values())
        assert math.isclose(total, 1.0, rel_tol=1e-9), (
            f"Distribution values should sum to 1.0, got {total}\n"
            f"Distribution: {distribution}"
        )
        
        # Verify all values are non-negative
        for emotion, value in distribution.items():
            assert value >= 0.0, (
                f"Distribution value for '{emotion}' should be >= 0, got {value}"
            )
    
    @given(timeline=emotion_timeline_strategy())
    @settings(max_examples=100)
    def test_rhythm_pattern_is_valid(self, timeline):
        """
        Property 10: Rhythm Pattern Validity
        
        For any emotion timeline, the determined rhythm pattern SHALL be
        one of the valid rhythm patterns from the defined set.
        
        Validates: Requirements 5.6
        """
        pattern = determine_rhythm_pattern(timeline)
        
        assert pattern in RHYTHM_PATTERNS, (
            f"Rhythm pattern '{pattern}' not in valid patterns: {RHYTHM_PATTERNS}"
        )
    
    @given(
        timeline=emotion_timeline_strategy(),
        duration=st.floats(min_value=0.1, max_value=100.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_heatmap_has_correct_dimensions(self, timeline, duration):
        """
        Property 10: Heatmap Dimensions Correctness
        
        For any emotion timeline and duration, the heatmap SHALL have:
        - Exactly 6 rows (one per emotion category)
        - Width between 1 and 100 columns based on duration
        
        Validates: Requirements 5.4
        """
        heatmap = _create_heatmap(timeline, duration)
        
        # Verify heatmap is a numpy array
        assert isinstance(heatmap, np.ndarray), (
            f"Heatmap should be a numpy array, got {type(heatmap)}"
        )
        
        # Verify number of rows equals number of emotion categories
        expected_rows = len(EMOTION_CATEGORIES)
        assert heatmap.shape[0] == expected_rows, (
            f"Heatmap should have {expected_rows} rows (one per emotion), "
            f"got {heatmap.shape[0]}"
        )
        
        # Verify width is in valid range [1, 100]
        expected_width = max(1, min(100, int(duration)))
        assert heatmap.shape[1] == expected_width, (
            f"Heatmap width should be {expected_width} for duration {duration}, "
            f"got {heatmap.shape[1]}"
        )
        
        # Verify all values are in [0, 1]
        assert heatmap.min() >= 0.0, (
            f"Heatmap values should be >= 0, got min: {heatmap.min()}"
        )
        assert heatmap.max() <= 1.0, (
            f"Heatmap values should be <= 1, got max: {heatmap.max()}"
        )
    
    @given(timeline=emotion_timeline_strategy())
    @settings(max_examples=100)
    def test_timeline_has_valid_timestamps(self, timeline):
        """
        Property 10: Timeline Timestamp Validity
        
        For any emotion timeline, all timestamps SHALL be non-negative
        and in ascending order.
        
        Validates: Requirements 5.1
        """
        if not timeline:
            return  # Empty timeline is valid edge case
        
        # Verify all timestamps are non-negative
        for frame in timeline:
            assert frame.timestamp >= 0.0, (
                f"Timestamp should be non-negative, got {frame.timestamp}"
            )
        
        # Verify timestamps are in ascending order
        timestamps = [f.timestamp for f in timeline]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], (
                f"Timestamps should be in ascending order. "
                f"Found {timestamps[i]} after {timestamps[i-1]}"
            )
    
    @given(frame=valid_bgr_frame_strategy())
    @settings(max_examples=100)
    def test_frame_analysis_returns_complete_features(self, frame):
        """
        Property 10: Frame Analysis Feature Completeness
        
        For any valid BGR frame, analyze_frame_emotion SHALL return
        a dictionary with all required feature keys.
        
        Validates: Requirements 5.1
        """
        features = analyze_frame_emotion(frame)
        
        required_keys = ["brightness", "saturation", "color_temperature", "contrast", "motion_intensity"]
        
        for key in required_keys:
            assert key in features, (
                f"Feature '{key}' missing from analysis result.\n"
                f"Got keys: {list(features.keys())}"
            )
            
            # Verify each feature is a valid float
            assert isinstance(features[key], float), (
                f"Feature '{key}' should be a float, got {type(features[key])}"
            )
            
            # Verify feature values are in expected ranges
            if key in ["brightness", "saturation", "contrast", "motion_intensity"]:
                assert 0.0 <= features[key] <= 1.0, (
                    f"Feature '{key}' should be in [0, 1], got {features[key]}"
                )
            elif key == "color_temperature":
                assert -1.0 <= features[key] <= 1.0, (
                    f"Feature '{key}' should be in [-1, 1], got {features[key]}"
                )


class TestEmotionAnalysisWithRealVideo:
    """
    Property tests for emotion analysis with real video file.
    
    Feature: cinemasense-stabilization, Property 10: Emotion Analysis Output Completeness
    Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6
    """
    
    @settings(max_examples=10, deadline=None)
    @given(sample_rate=st.integers(min_value=1, max_value=30))
    def test_full_analysis_output_completeness(self, sample_rate):
        """
        Property 10: Full Analysis Output Completeness
        
        For any video analysis with valid sample rate, the EmotionRhythmResult
        SHALL contain all required fields with valid values.
        
        Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6
        """
        test_video_path = ROOT / "data" / "test" / "test_video.mp4"
        
        # Skip if test video doesn't exist
        assume(test_video_path.exists())
        
        result = extract_emotion_timeline(str(test_video_path), sample_rate=sample_rate)
        
        # Verify result is EmotionRhythmResult
        assert isinstance(result, EmotionRhythmResult), (
            f"Result should be EmotionRhythmResult, got {type(result)}"
        )
        
        # Verify timeline is non-empty
        assert len(result.timeline) > 0, "Timeline should not be empty"
        
        # Verify all timeline entries have valid timestamps
        for frame in result.timeline:
            assert frame.timestamp >= 0.0, (
                f"Timestamp should be non-negative, got {frame.timestamp}"
            )
            assert frame.dominant_emotion in EMOTION_CATEGORIES, (
                f"Emotion '{frame.dominant_emotion}' not in valid categories"
            )
        
        # Verify emotion_distribution sums to 1.0
        dist_sum = sum(result.emotion_distribution.values())
        assert math.isclose(dist_sum, 1.0, rel_tol=1e-9), (
            f"Emotion distribution should sum to 1.0, got {dist_sum}"
        )
        
        # Verify all emotion categories are present in distribution
        assert set(result.emotion_distribution.keys()) == set(EMOTION_CATEGORIES), (
            f"Distribution should contain all emotion categories"
        )
        
        # Verify rhythm_pattern is valid
        assert result.rhythm_pattern in RHYTHM_PATTERNS, (
            f"Rhythm pattern '{result.rhythm_pattern}' not in valid patterns"
        )
        
        # Verify heatmap_data has correct dimensions
        assert isinstance(result.heatmap_data, np.ndarray), (
            "Heatmap should be a numpy array"
        )
        assert result.heatmap_data.shape[0] == len(EMOTION_CATEGORIES), (
            f"Heatmap should have {len(EMOTION_CATEGORIES)} rows"
        )
        
        # Verify overall_score is in valid range
        assert 0.0 <= result.overall_score <= 100.0, (
            f"Overall score should be in [0, 100], got {result.overall_score}"
        )
        
        # Verify confidence is in [0, 1]
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence should be in [0, 1], got {result.confidence}"
        )


class TestEmotionClassificationValidity:
    """
    Property tests for Emotion Classification Validity.
    
    Feature: cinemasense-stabilization, Property 11: Emotion Classification Validity
    Validates: Requirements 5.2
    
    For any frame features, emotion classification SHALL return one of the valid
    emotion categories (Joy, Tension, Calm, Melancholy, Energy, Mystery) with
    confidence in range [0, 1].
    """
    
    @given(features=valid_features_strategy())
    @settings(max_examples=100)
    def test_emotion_classification_returns_valid_category(self, features):
        """
        Property 11: Emotion Classification Validity
        
        For any valid frame features, emotion classification SHALL return
        one of the valid emotion categories.
        
        Validates: Requirements 5.2
        """
        emotion, confidence = classify_emotion(features)
        
        assert emotion in EMOTION_CATEGORIES, (
            f"Emotion '{emotion}' not in valid categories: {EMOTION_CATEGORIES}\n"
            f"Input features: {features}"
        )
    
    @given(features=valid_features_strategy())
    @settings(max_examples=100)
    def test_emotion_classification_confidence_in_valid_range(self, features):
        """
        Property 11: Emotion Classification Confidence Range
        
        For any valid frame features, emotion classification SHALL return
        confidence in range [0, 1].
        
        Validates: Requirements 5.2
        """
        emotion, confidence = classify_emotion(features)
        
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} should be in range [0, 1]\n"
            f"Emotion: {emotion}\n"
            f"Input features: {features}"
        )
    
    @given(
        brightness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        saturation=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        color_temperature=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        contrast=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        motion_intensity=st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
    )
    @settings(max_examples=100)
    def test_emotion_classification_with_individual_features(
        self, brightness, saturation, color_temperature, contrast, motion_intensity
    ):
        """
        Property 11: Emotion Classification with Individual Features
        
        For any combination of individual feature values within valid ranges,
        emotion classification SHALL return a valid emotion category with
        confidence in [0, 1].
        
        Validates: Requirements 5.2
        """
        features = {
            "brightness": brightness,
            "saturation": saturation,
            "color_temperature": color_temperature,
            "contrast": contrast,
            "motion_intensity": motion_intensity,
        }
        
        emotion, confidence = classify_emotion(features)
        
        # Verify emotion is valid
        assert emotion in EMOTION_CATEGORIES, (
            f"Emotion '{emotion}' not in valid categories: {EMOTION_CATEGORIES}\n"
            f"Input features: {features}"
        )
        
        # Verify confidence is in valid range
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} should be in range [0, 1]\n"
            f"Emotion: {emotion}\n"
            f"Input features: {features}"
        )
    
    @given(features=valid_features_strategy())
    @settings(max_examples=100)
    def test_emotion_classification_deterministic(self, features):
        """
        Property 11: Emotion Classification Determinism
        
        For any valid frame features, calling classify_emotion multiple times
        with the same input SHALL return the same result.
        
        Validates: Requirements 5.2
        """
        emotion1, confidence1 = classify_emotion(features)
        emotion2, confidence2 = classify_emotion(features)
        
        assert emotion1 == emotion2, (
            f"Emotion classification should be deterministic.\n"
            f"First call: {emotion1}, Second call: {emotion2}\n"
            f"Input features: {features}"
        )
        
        assert confidence1 == confidence2, (
            f"Confidence should be deterministic.\n"
            f"First call: {confidence1}, Second call: {confidence2}\n"
            f"Input features: {features}"
        )
    
    @given(
        features=valid_features_strategy(),
        missing_key=st.sampled_from(["brightness", "saturation", "color_temperature", "contrast", "motion_intensity"])
    )
    @settings(max_examples=100)
    def test_emotion_classification_handles_missing_features(self, features, missing_key):
        """
        Property 11: Emotion Classification with Missing Features
        
        For any features dictionary with a missing key, emotion classification
        SHALL still return a valid emotion category with confidence in [0, 1]
        (using default values for missing features).
        
        Validates: Requirements 5.2
        """
        # Remove one key from features
        partial_features = {k: v for k, v in features.items() if k != missing_key}
        
        emotion, confidence = classify_emotion(partial_features)
        
        # Verify emotion is valid
        assert emotion in EMOTION_CATEGORIES, (
            f"Emotion '{emotion}' not in valid categories: {EMOTION_CATEGORIES}\n"
            f"Missing key: {missing_key}\n"
            f"Input features: {partial_features}"
        )
        
        # Verify confidence is in valid range
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} should be in range [0, 1]\n"
            f"Emotion: {emotion}\n"
            f"Missing key: {missing_key}\n"
            f"Input features: {partial_features}"
        )
    
    @settings(max_examples=100)
    @given(st.data())
    def test_emotion_classification_empty_features(self, data):
        """
        Property 11: Emotion Classification with Empty Features
        
        For an empty features dictionary, emotion classification SHALL still
        return a valid emotion category with confidence in [0, 1].
        
        Validates: Requirements 5.2
        """
        emotion, confidence = classify_emotion({})
        
        # Verify emotion is valid
        assert emotion in EMOTION_CATEGORIES, (
            f"Emotion '{emotion}' not in valid categories: {EMOTION_CATEGORIES}\n"
            f"Input features: {{}}"
        )
        
        # Verify confidence is in valid range
        assert 0.0 <= confidence <= 1.0, (
            f"Confidence {confidence} should be in range [0, 1]\n"
            f"Emotion: {emotion}\n"
            f"Input features: {{}}"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
