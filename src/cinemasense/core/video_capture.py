"""
Safe video capture context manager for proper resource cleanup.

Ensures video capture resources are properly released on Windows,
even when exceptions occur during processing.

Requirements: 2.2, 2.5
"""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2

logger = logging.getLogger("cinemasense.core.video_capture")


class VideoOpenError(Exception):
    """Raised when video file cannot be opened."""
    pass


class SafeVideoCapture:
    """
    Context manager for OpenCV VideoCapture with guaranteed resource cleanup.
    
    Ensures video capture resources are properly released after processing
    completes or fails, preventing resource leaks on Windows.
    
    Usage:
        with SafeVideoCapture("video.mp4") as cap:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # process frame
    
    Requirements: 2.2, 2.5
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize SafeVideoCapture.
        
        Args:
            path: Path to video file
            
        Raises:
            VideoOpenError: If video file cannot be opened
        """
        self._path = str(path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
    
    def __enter__(self) -> cv2.VideoCapture:
        """
        Open video capture and return the capture object.
        
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            VideoOpenError: If video cannot be opened or is empty/corrupted
        """
        try:
            self._cap = cv2.VideoCapture(self._path)
            
            if not self._cap.isOpened():
                raise VideoOpenError(
                    f"Cannot open video file: {self._path}. "
                    "File may be corrupted, missing, or in an unsupported format."
                )
            
            # Verify video has frames
            frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                # Try reading a frame to double-check
                ret, _ = self._cap.read()
                if not ret:
                    self._cap.release()
                    raise VideoOpenError(
                        f"Video file appears to be empty or corrupted: {self._path}"
                    )
                # Reset to beginning
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            self._is_open = True
            logger.debug(f"Opened video: {self._path}")
            return self._cap
            
        except VideoOpenError:
            raise
        except Exception as e:
            if self._cap is not None:
                self._cap.release()
            raise VideoOpenError(f"Error opening video {self._path}: {e}")
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Release video capture resources.
        
        Always releases resources, even if an exception occurred.
        """
        if self._cap is not None:
            self._cap.release()
            self._is_open = False
            logger.debug(f"Released video capture: {self._path}")
        
        # Don't suppress exceptions
        return None
    
    @property
    def fps(self) -> float:
        """Get video frames per second."""
        if self._cap is None or not self._is_open:
            return 0.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0  # Default to 30 if unknown
    
    @property
    def frame_count(self) -> int:
        """Get total number of frames in video."""
        if self._cap is None or not self._is_open:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def width(self) -> int:
        """Get video frame width in pixels."""
        if self._cap is None or not self._is_open:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def height(self) -> int:
        """Get video frame height in pixels."""
        if self._cap is None or not self._is_open:
            return 0
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def duration(self) -> float:
        """Get video duration in seconds."""
        if self._cap is None or not self._is_open:
            return 0.0
        fps = self.fps
        if fps <= 0:
            return 0.0
        return self.frame_count / fps
    
    @property
    def is_open(self) -> bool:
        """Check if video capture is currently open."""
        return self._is_open and self._cap is not None and self._cap.isOpened()
