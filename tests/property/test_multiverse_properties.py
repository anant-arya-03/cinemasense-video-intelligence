"""
Property-based tests for Multiverse Style Application.

Feature: cinemasense-stabilization
Property 8: Multiverse Style Application
Validates: Requirements 4.1, 4.4

Tests that for any valid frame and style name, applying the style
transformation SHALL produce an output frame with different pixel values
than the input.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from cinemasense.pipeline.multiverse import (
    apply_style_to_frame,
    VALID_STYLE_NAMES,
    MULTIVERSE_STYLES,
    InvalidStyleError,
)


# Strategy for generating valid style names
style_name_strategy = st.sampled_from(list(VALID_STYLE_NAMES))


@st.composite
def valid_frame_strategy(draw):
    """Generate a valid BGR frame with random pixel values using numpy arrays."""
    height = draw(st.integers(min_value=20, max_value=50))
    width = draw(st.integers(min_value=20, max_value=50))
    # Use numpy arrays strategy for efficient generation
    frame = draw(
        st.from_type(np.ndarray).filter(lambda x: False)  # Never used
        | st.just(None)  # Placeholder
    )
    # Generate frame using seed for reproducibility
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


class TestMultiverseStyleApplication:
    """
    Property tests for Multiverse style application.
    
    Feature: cinemasense-stabilization, Property 8: Multiverse Style Application
    Validates: Requirements 4.1, 4.4
    """
    
    @given(
        frame=valid_frame_strategy(),
        style_name=style_name_strategy
    )
    @settings(max_examples=100)
    def test_style_application_modifies_frame(self, frame, style_name):
        """
        Property 8: Multiverse Style Application
        
        For any valid frame and style name, applying the style transformation
        SHALL produce an output frame with different pixel values than the input.
        
        Validates: Requirements 4.1, 4.4
        """
        # Skip frames that are completely uniform (edge case where some styles
        # might not produce visible changes)
        assume(np.std(frame) > 1.0)
        
        # Apply style transformation
        result = apply_style_to_frame(frame.copy(), style_name)
        
        # Verify output is a valid frame
        assert result is not None, "Style application should return a frame"
        assert isinstance(result, np.ndarray), "Result should be a numpy array"
        assert result.dtype == np.uint8, f"Result dtype should be uint8, got {result.dtype}"
        assert len(result.shape) == 3, f"Result should be 3D, got shape {result.shape}"
        assert result.shape[2] == 3, f"Result should have 3 channels, got {result.shape[2]}"
        
        # Verify the frame was modified (pixels changed)
        # We compare the mean absolute difference to ensure transformation occurred
        diff = np.abs(result.astype(np.float32) - frame.astype(np.float32))
        mean_diff = np.mean(diff)
        
        assert mean_diff > 0, (
            f"Style '{style_name}' should modify the frame\n"
            f"Input shape: {frame.shape}, Output shape: {result.shape}\n"
            f"Mean pixel difference: {mean_diff}"
        )
    
    @given(
        frame=valid_frame_strategy(),
        style_name=style_name_strategy
    )
    @settings(max_examples=100)
    def test_style_output_has_valid_pixel_range(self, frame, style_name):
        """
        Property 8: Multiverse Style Application (pixel range)
        
        For any valid frame and style name, the output frame SHALL have
        pixel values in the valid range [0, 255].
        
        Validates: Requirements 4.1, 4.4
        """
        result = apply_style_to_frame(frame.copy(), style_name)
        
        # Verify pixel values are in valid range
        assert result.min() >= 0, (
            f"Pixel values should be >= 0, got min: {result.min()}"
        )
        assert result.max() <= 255, (
            f"Pixel values should be <= 255, got max: {result.max()}"
        )
    
    @given(
        frame=valid_frame_strategy(),
        style_name=style_name_strategy
    )
    @settings(max_examples=100)
    def test_style_preserves_frame_dimensions(self, frame, style_name):
        """
        Property 8: Multiverse Style Application (dimensions)
        
        For any valid frame and style name, the output frame SHALL have
        the same dimensions as the input frame (except for letterbox effect).
        
        Validates: Requirements 4.1, 4.4
        """
        result = apply_style_to_frame(frame.copy(), style_name)
        
        # Verify dimensions are preserved
        assert result.shape == frame.shape, (
            f"Style '{style_name}' should preserve frame dimensions\n"
            f"Input shape: {frame.shape}, Output shape: {result.shape}"
        )


class TestMultiversePreviewPositions:
    """
    Property tests for Multiverse preview positions.
    
    Feature: cinemasense-stabilization, Property 9: Multiverse Preview Positions
    Validates: Requirements 4.2
    """
    
    @given(style_name=style_name_strategy)
    @settings(max_examples=100, deadline=None)
    def test_preview_positions_are_correct(self, style_name):
        """
        Property 9: Multiverse Preview Positions
        
        For any video, multiverse preview generation SHALL produce exactly 3
        preview frames at positions 0.25, 0.50, and 0.75 of the video duration.
        
        Validates: Requirements 4.2
        """
        import tempfile
        import shutil
        from cinemasense.pipeline.multiverse import (
            generate_multiverse_preview,
            PREVIEW_POSITIONS,
        )
        
        # Use the test video
        test_video_path = Path(__file__).resolve().parent.parent.parent / "data" / "test" / "test_video.mp4"
        
        # Skip if test video doesn't exist
        assume(test_video_path.exists())
        
        # Use tempfile context manager instead of pytest fixture
        tmp_dir = tempfile.mkdtemp(prefix="multiverse_test_")
        try:
            tmp_path = Path(tmp_dir)
            
            # Generate preview
            result = generate_multiverse_preview(
                str(test_video_path),
                style_name,
                tmp_path
            )
            
            # Verify exactly 3 previews are generated
            assert len(result.previews) == 3, (
                f"Expected exactly 3 previews, got {len(result.previews)}"
            )
            
            # Verify positions are at 0.25, 0.50, 0.75
            expected_positions = PREVIEW_POSITIONS  # [0.25, 0.50, 0.75]
            actual_positions = sorted([p["position"] for p in result.previews])
            
            assert actual_positions == expected_positions, (
                f"Expected positions {expected_positions}, got {actual_positions}"
            )
            
            # Verify each preview has required fields
            for preview in result.previews:
                assert "position" in preview, "Preview should have 'position' field"
                assert "timestamp" in preview, "Preview should have 'timestamp' field"
                assert "path" in preview, "Preview should have 'path' field"
                assert "frame_index" in preview, "Preview should have 'frame_index' field"
                
                # Verify position is one of the expected values
                assert preview["position"] in expected_positions, (
                    f"Position {preview['position']} not in expected {expected_positions}"
                )
                
                # Verify timestamp is non-negative
                assert preview["timestamp"] >= 0, (
                    f"Timestamp should be non-negative, got {preview['timestamp']}"
                )
                
                # Verify frame_index is non-negative
                assert preview["frame_index"] >= 0, (
                    f"Frame index should be non-negative, got {preview['frame_index']}"
                )
                
                # Verify preview file was created
                preview_path = Path(preview["path"])
                assert preview_path.exists(), (
                    f"Preview file should exist at {preview_path}"
                )
        finally:
            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
