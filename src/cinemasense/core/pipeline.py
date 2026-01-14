"""
Pipeline result dataclass and runner for video processing.

Provides structured results with success/failure status, error handling,
and progress tracking for video analysis pipelines.

Requirements: 2.1, 2.4, 2.6
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import cv2

from .video_capture import SafeVideoCapture, VideoOpenError

logger = logging.getLogger("cinemasense.core.pipeline")

# Supported video formats
SUPPORTED_FORMATS = {'.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v'}


@dataclass
class PipelineResult:
    """
    Structured result from a pipeline operation.
    
    Attributes:
        success: Whether the operation completed successfully
        data: Result data if successful, None otherwise
        error: Error message if failed, None otherwise
        duration_ms: Time taken for the operation in milliseconds
        
    Requirements: 2.4
    """
    success: bool
    data: Optional[Any]
    error: Optional[str]
    duration_ms: float
    
    def __post_init__(self):
        """Validate result structure consistency."""
        # Ensure consistent state: success=True means data exists, error is None
        # success=False means error exists
        if self.success and self.error is not None:
            logger.warning("PipelineResult has success=True but error is set")
        if not self.success and self.error is None:
            self.error = "Unknown error occurred"


class PipelineRunner:
    """
    Orchestrates video processing with progress tracking and error handling.
    
    Requirements: 2.1, 2.4, 2.6
    """
    
    @staticmethod
    def validate_video(path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Validate that a video file can be processed.
        
        Args:
            path: Path to video file
            
        Returns:
            Tuple of (is_valid, message)
            
        Requirements: 2.1, 2.5
        """
        path = Path(path)
        
        # Check file exists
        if not path.exists():
            return False, f"Video file not found: {path}"
        
        # Check file extension
        if path.suffix.lower() not in SUPPORTED_FORMATS:
            return False, (
                f"Unsupported video format: {path.suffix}. "
                f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        if path.stat().st_size == 0:
            return False, f"Video file is empty: {path}"
        
        # Try to open and read video
        try:
            with SafeVideoCapture(str(path)) as cap:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if frame_count <= 0:
                    return False, "Video has no frames"
                
                if width <= 0 or height <= 0:
                    return False, "Video has invalid dimensions"
                
                # Try reading first frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    return False, "Cannot read video frames"
                
                return True, f"Valid video: {frame_count} frames, {width}x{height}, {fps:.1f} fps"
                
        except VideoOpenError as e:
            return False, str(e)
        except Exception as e:
            return False, f"Error validating video: {e}"
    
    @staticmethod
    def run_with_progress(
        task: Callable[..., Any],
        video_path: str,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        **kwargs
    ) -> PipelineResult:
        """
        Run a pipeline task with progress tracking and error handling.
        
        Args:
            task: Callable that performs the processing
            video_path: Path to video file
            progress_callback: Optional callback(progress: float, message: str)
            **kwargs: Additional arguments to pass to task
            
        Returns:
            PipelineResult with success status, data or error, and duration
            
        Requirements: 2.4, 2.6
        """
        start_time = time.time()
        
        def report_progress(progress: float, message: str = ""):
            """Report progress if callback provided."""
            if progress_callback:
                try:
                    progress_callback(min(max(progress, 0.0), 1.0), message)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
        
        try:
            # Validate video first
            report_progress(0.0, "Validating video...")
            is_valid, validation_msg = PipelineRunner.validate_video(video_path)
            
            if not is_valid:
                duration_ms = (time.time() - start_time) * 1000
                return PipelineResult(
                    success=False,
                    data=None,
                    error=validation_msg,
                    duration_ms=duration_ms
                )
            
            report_progress(0.1, "Starting processing...")
            
            # Run the task
            result = task(video_path, **kwargs)
            
            report_progress(1.0, "Complete")
            
            duration_ms = (time.time() - start_time) * 1000
            return PipelineResult(
                success=True,
                data=result,
                error=None,
                duration_ms=duration_ms
            )
            
        except VideoOpenError as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Video open error: {e}")
            return PipelineResult(
                success=False,
                data=None,
                error=str(e),
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return PipelineResult(
                success=False,
                data=None,
                error=f"Processing error: {e}",
                duration_ms=duration_ms
            )
    
    @staticmethod
    def ensure_output_dir(video_name: str, base_dir: Optional[Path] = None) -> Path:
        """
        Ensure output directory exists for a video.
        
        Args:
            video_name: Name of the video (used for directory name)
            base_dir: Base output directory (defaults to data/output)
            
        Returns:
            Path to the output directory
        """
        from .file_ops import FileOps
        
        if base_dir is None:
            # Default to project data/output directory
            base_dir = Path(__file__).parent.parent.parent.parent / "data" / "output"
        
        # Sanitize video name for directory
        safe_name = FileOps.sanitize_filename(video_name)
        output_dir = base_dir / safe_name
        
        return FileOps.ensure_directory(output_dir)
