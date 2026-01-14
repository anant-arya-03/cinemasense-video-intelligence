"""
System dependency verification and health checks
"""

import sys
import subprocess
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger("cinemasense.core.system_check")


@dataclass
class DependencyStatus:
    name: str
    available: bool
    version: str = ""
    error: str = ""


def check_python_version() -> DependencyStatus:
    """Check Python version compatibility"""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    is_compatible = sys.version_info >= (3, 10)
    return DependencyStatus(
        name="Python",
        available=is_compatible,
        version=version,
        error="" if is_compatible else "Python 3.10+ required"
    )


def check_ffmpeg() -> DependencyStatus:
    """Check FFmpeg availability"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            return DependencyStatus(name="FFmpeg", available=True, version=version_line)
        return DependencyStatus(name="FFmpeg", available=False, error="FFmpeg not responding")
    except FileNotFoundError:
        # Try imageio-ffmpeg fallback
        try:
            import imageio_ffmpeg
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            return DependencyStatus(name="FFmpeg (imageio)", available=True, version=ffmpeg_path)
        except Exception as e:
            return DependencyStatus(name="FFmpeg", available=False, error=str(e))
    except Exception as e:
        return DependencyStatus(name="FFmpeg", available=False, error=str(e))


def check_opencv() -> DependencyStatus:
    """Check OpenCV availability"""
    try:
        import cv2
        return DependencyStatus(name="OpenCV", available=True, version=cv2.__version__)
    except ImportError as e:
        return DependencyStatus(name="OpenCV", available=False, error=str(e))


def check_mediapipe() -> DependencyStatus:
    """Check MediaPipe availability"""
    try:
        import mediapipe as mp
        return DependencyStatus(name="MediaPipe", available=True, version=mp.__version__)
    except ImportError as e:
        return DependencyStatus(name="MediaPipe", available=False, error=str(e))


def check_moviepy() -> DependencyStatus:
    """Check MoviePy availability with correct import"""
    try:
        # Try new import style first (MoviePy 2.x)
        try:
            from moviepy import VideoFileClip
            import moviepy
            return DependencyStatus(name="MoviePy", available=True, version=getattr(moviepy, '__version__', '2.x'))
        except ImportError:
            # Fall back to old import style (MoviePy 1.x)
            from moviepy.editor import VideoFileClip
            import moviepy
            return DependencyStatus(name="MoviePy", available=True, version=getattr(moviepy, '__version__', '1.x'))
    except ImportError as e:
        return DependencyStatus(name="MoviePy", available=False, error=str(e))


def check_librosa() -> DependencyStatus:
    """Check Librosa availability"""
    try:
        import librosa
        return DependencyStatus(name="Librosa", available=True, version=librosa.__version__)
    except ImportError as e:
        return DependencyStatus(name="Librosa", available=False, error=str(e))


def check_sklearn() -> DependencyStatus:
    """Check Scikit-learn availability"""
    try:
        import sklearn
        return DependencyStatus(name="Scikit-learn", available=True, version=sklearn.__version__)
    except ImportError as e:
        return DependencyStatus(name="Scikit-learn", available=False, error=str(e))


def run_all_checks() -> Tuple[bool, List[DependencyStatus]]:
    """Run all dependency checks"""
    checks = [
        check_python_version(),
        check_ffmpeg(),
        check_opencv(),
        check_mediapipe(),
        check_moviepy(),
        check_librosa(),
        check_sklearn()
    ]
    
    all_critical_ok = all(c.available for c in checks[:5])  # First 5 are critical
    
    for check in checks:
        if check.available:
            logger.info(f"[OK] {check.name}: {check.version}")
        else:
            logger.warning(f"[MISSING] {check.name}: {check.error}")
    
    return all_critical_ok, checks


def get_system_info() -> Dict:
    """Get system information"""
    import platform
    import os
    
    return {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "cpu_count": os.cpu_count()
    }