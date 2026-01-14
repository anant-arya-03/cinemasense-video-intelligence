"""
Property-based tests for SafeVideoCapture context manager.

Feature: cinemasense-stabilization
Property 4: Video Capture Resource Cleanup
Validates: Requirements 2.2

Tests that video capture resources are properly released after the
context manager exits, whether normally or via exception.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings, assume

from cinemasense.core.video_capture import SafeVideoCapture, VideoOpenError

# Path to test video
TEST_VIDEO = ROOT / "data" / "test" / "test_video.mp4"


class TestVideoCaptureResourceCleanup:
    """
    Property tests for SafeVideoCapture resource cleanup.
    
    Feature: cinemasense-stabilization, Property 4: Video Capture Resource Cleanup
    Validates: Requirements 2.2
    """
    
    @given(st.integers(min_value=0, max_value=10))
    @settings(max_examples=100)
    def test_video_capture_released_after_normal_exit(self, frames_to_read: int):
        """
        Property 4: Video Capture Resource Cleanup (normal exit)
        
        For any video processing operation using SafeVideoCapture,
        after the context manager exits normally, the video capture
        resource SHALL be released.
        
        Validates: Requirements 2.2
        """
        assume(TEST_VIDEO.exists())
        
        capture_instance = SafeVideoCapture(TEST_VIDEO)
        
        # Use context manager and read some frames
        with capture_instance as cap:
            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    break
        
        # After exiting context, resource should be released
        assert not capture_instance.is_open, (
            f"Video capture still open after normal context exit\n"
            f"Frames read: {frames_to_read}"
        )
        assert capture_instance._cap is not None, (
            "Internal capture object should still exist but be released"
        )
        assert not capture_instance._cap.isOpened(), (
            "Internal cv2.VideoCapture should report not opened after release"
        )
    
    @given(st.sampled_from([ValueError, RuntimeError, KeyError, TypeError]))
    @settings(max_examples=100)
    def test_video_capture_released_after_exception(self, exception_type):
        """
        Property 4: Video Capture Resource Cleanup (exception exit)
        
        For any video processing operation using SafeVideoCapture,
        after the context manager exits via exception, the video capture
        resource SHALL be released.
        
        Validates: Requirements 2.2
        """
        assume(TEST_VIDEO.exists())
        
        capture_instance = SafeVideoCapture(TEST_VIDEO)
        
        # Use context manager and raise an exception
        try:
            with capture_instance as cap:
                # Read one frame to ensure capture is active
                ret, frame = cap.read()
                # Raise exception to test cleanup
                raise exception_type("Test exception for cleanup verification")
        except exception_type:
            pass  # Expected exception
        
        # After exiting context via exception, resource should be released
        assert not capture_instance.is_open, (
            f"Video capture still open after exception exit\n"
            f"Exception type: {exception_type.__name__}"
        )
        assert capture_instance._cap is not None, (
            "Internal capture object should still exist but be released"
        )
        assert not capture_instance._cap.isOpened(), (
            "Internal cv2.VideoCapture should report not opened after release"
        )
    
    @given(st.booleans())
    @settings(max_examples=100)
    def test_video_capture_released_regardless_of_read_success(self, read_frames: bool):
        """
        Property 4: Video Capture Resource Cleanup (read success variation)
        
        For any video processing operation using SafeVideoCapture,
        regardless of whether frames were successfully read,
        the video capture resource SHALL be released after exit.
        
        Validates: Requirements 2.2
        """
        assume(TEST_VIDEO.exists())
        
        capture_instance = SafeVideoCapture(TEST_VIDEO)
        
        with capture_instance as cap:
            if read_frames:
                # Read until end of video
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
            # else: don't read any frames
        
        # After exiting context, resource should be released
        assert not capture_instance.is_open, (
            f"Video capture still open after context exit\n"
            f"Read frames: {read_frames}"
        )
        assert not capture_instance._cap.isOpened(), (
            "Internal cv2.VideoCapture should report not opened after release"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
