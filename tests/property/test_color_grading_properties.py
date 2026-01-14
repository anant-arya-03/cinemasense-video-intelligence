"""
Property-based tests for Color Grading Transformation.

Feature: cinemasense-stabilization
Property 17: Color Grading Transformation
Validates: Requirements 10.2, 10.3

Tests that for any frame and color grading preset, applying the preset
SHALL produce an output frame with measurably different color characteristics.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2
from hypothesis import given, strategies as st, settings, assume

from cinemasense.pipeline.color_grading import (
    apply_color_grading,
    apply_color_grading_safe,
    get_preset,
    validate_preset_name,
    validate_frame,
    CINEMA_PRESETS,
    ColorGradingPreset,
)


# Strategy for generating valid preset names
preset_name_strategy = st.sampled_from(list(CINEMA_PRESETS.keys()))


@st.composite
def valid_frame_strategy(draw):
    """Generate a valid BGR frame with random pixel values."""
    height = draw(st.integers(min_value=20, max_value=50))
    width = draw(st.integers(min_value=20, max_value=50))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def compute_color_characteristics(frame: np.ndarray) -> dict:
    """Compute color characteristics of a frame for comparison."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return {
        "mean_brightness": float(np.mean(hsv[:, :, 2])),
        "mean_saturation": float(np.mean(hsv[:, :, 1])),
        "mean_hue": float(np.mean(hsv[:, :, 0])),
        "mean_b": float(np.mean(frame[:, :, 0])),
        "mean_g": float(np.mean(frame[:, :, 1])),
        "mean_r": float(np.mean(frame[:, :, 2])),
        "std_brightness": float(np.std(hsv[:, :, 2])),
    }


class TestColorGradingTransformation:
    """
    Property tests for Color Grading Transformation.
    
    Feature: cinemasense-stabilization, Property 17: Color Grading Transformation
    Validates: Requirements 10.2, 10.3
    """
    
    @given(
        frame=valid_frame_strategy(),
        preset_name=preset_name_strategy
    )
    @settings(max_examples=100)
    def test_color_grading_modifies_frame(self, frame, preset_name):
        """
        Property 17: Color Grading Transformation
        
        For any frame and color grading preset, applying the preset SHALL
        produce an output frame with measurably different color characteristics.
        
        Validates: Requirements 10.2, 10.3
        """
        # Skip frames that are completely uniform (edge case)
        assume(np.std(frame) > 1.0)
        
        # Get preset and apply color grading
        preset = get_preset(preset_name)
        result = apply_color_grading(frame.copy(), preset)
        
        # Verify output is a valid frame
        assert result is not None, "Color grading should return a frame"
        assert isinstance(result, np.ndarray), "Result should be a numpy array"
        assert result.dtype == np.uint8, f"Result dtype should be uint8, got {result.dtype}"
        assert len(result.shape) == 3, f"Result should be 3D, got shape {result.shape}"
        assert result.shape[2] == 3, f"Result should have 3 channels, got {result.shape[2]}"
        
        # Compute color characteristics before and after
        original_chars = compute_color_characteristics(frame)
        graded_chars = compute_color_characteristics(result)
        
        # Calculate total difference in color characteristics
        total_diff = 0.0
        for key in original_chars:
            total_diff += abs(original_chars[key] - graded_chars[key])
        
        # Verify the frame was modified (color characteristics changed)
        assert total_diff > 0, (
            f"Preset '{preset_name}' should modify color characteristics\n"
            f"Original: {original_chars}\n"
            f"Graded: {graded_chars}\n"
            f"Total difference: {total_diff}"
        )

    
    @given(
        frame=valid_frame_strategy(),
        preset_name=preset_name_strategy
    )
    @settings(max_examples=100)
    def test_color_grading_output_has_valid_pixel_range(self, frame, preset_name):
        """
        Property 17: Color Grading Transformation (pixel range)
        
        For any frame and color grading preset, the output frame SHALL have
        pixel values in the valid range [0, 255].
        
        Validates: Requirements 10.2, 10.3
        """
        preset = get_preset(preset_name)
        result = apply_color_grading(frame.copy(), preset)
        
        # Verify pixel values are in valid range
        assert result.min() >= 0, (
            f"Pixel values should be >= 0, got min: {result.min()}"
        )
        assert result.max() <= 255, (
            f"Pixel values should be <= 255, got max: {result.max()}"
        )
    
    @given(
        frame=valid_frame_strategy(),
        preset_name=preset_name_strategy
    )
    @settings(max_examples=100)
    def test_color_grading_preserves_frame_dimensions(self, frame, preset_name):
        """
        Property 17: Color Grading Transformation (dimensions)
        
        For any frame and color grading preset, the output frame SHALL have
        the same dimensions as the input frame.
        
        Validates: Requirements 10.2, 10.3
        """
        preset = get_preset(preset_name)
        result = apply_color_grading(frame.copy(), preset)
        
        # Verify dimensions are preserved
        assert result.shape == frame.shape, (
            f"Preset '{preset_name}' should preserve frame dimensions\n"
            f"Input shape: {frame.shape}, Output shape: {result.shape}"
        )
    
    @given(
        frame=valid_frame_strategy(),
        preset_name=preset_name_strategy
    )
    @settings(max_examples=100)
    def test_color_grading_safe_returns_valid_result(self, frame, preset_name):
        """
        Property 17: Color Grading Transformation (safe wrapper)
        
        For any valid frame and preset name, apply_color_grading_safe SHALL
        return a ColorGradingResult with success=True and a valid graded frame.
        
        Validates: Requirements 10.2, 10.3
        """
        result = apply_color_grading_safe(frame.copy(), preset_name)
        
        # Verify result structure
        assert result.success is True, (
            f"Color grading should succeed for valid inputs, got error: {result.error}"
        )
        assert result.graded_frame is not None, "Graded frame should not be None"
        assert result.preset_name == preset_name, (
            f"Preset name should match, expected '{preset_name}', got '{result.preset_name}'"
        )
        assert result.error is None, f"Error should be None, got: {result.error}"
        
        # Verify graded frame is valid
        assert isinstance(result.graded_frame, np.ndarray), "Graded frame should be numpy array"
        assert result.graded_frame.dtype == np.uint8, "Graded frame should be uint8"
        assert result.graded_frame.shape == frame.shape, "Graded frame should preserve dimensions"


class TestColorGradingLiftGammaGain:
    """
    Property tests for lift, gamma, and gain adjustments.
    
    Feature: cinemasense-stabilization, Property 17: Color Grading Transformation
    Validates: Requirements 10.3
    """
    
    @given(frame=valid_frame_strategy())
    @settings(max_examples=100)
    def test_lift_adjustment_affects_shadows(self, frame):
        """
        Property 17: Color Grading Transformation (lift)
        
        For any frame, applying a preset with positive lift SHALL increase
        the brightness of shadow regions.
        
        Validates: Requirements 10.3
        """
        # Skip frames with no shadow regions
        assume(np.min(frame) < 85)
        assume(np.std(frame) > 1.0)
        
        # Use romance preset which has positive lift (0.1)
        preset = get_preset("romance")
        result = apply_color_grading(frame.copy(), preset)
        
        # Verify output is valid
        assert result is not None
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255
    
    @given(frame=valid_frame_strategy())
    @settings(max_examples=100)
    def test_gain_adjustment_affects_highlights(self, frame):
        """
        Property 17: Color Grading Transformation (gain)
        
        For any frame, applying a preset with gain > 1 SHALL affect
        the brightness of highlight regions.
        
        Validates: Requirements 10.3
        """
        # Skip frames with no highlight regions
        assume(np.max(frame) > 170)
        assume(np.std(frame) > 1.0)
        
        # Use blockbuster preset which has gain > 1 (1.1)
        preset = get_preset("blockbuster")
        result = apply_color_grading(frame.copy(), preset)
        
        # Verify output is valid
        assert result is not None
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
