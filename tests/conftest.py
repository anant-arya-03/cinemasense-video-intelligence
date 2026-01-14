"""
CinemaSense AI Studio - Pytest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
It also suppresses known third-party deprecation warnings for clean test output.
"""

import sys
import warnings
from pathlib import Path

# Suppress known third-party deprecation warnings before any imports
# This ensures clean test output without noise from dependencies
warnings.filterwarnings("ignore", category=UserWarning, module="imageio_ffmpeg")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")

# Add src to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest


@pytest.fixture(scope="session")
def test_video_path():
    """Provide path to test video file."""
    video_path = ROOT / "data" / "test" / "test_video.mp4"
    if video_path.exists():
        return str(video_path)
    return None


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide path to test data directory."""
    return ROOT / "data" / "test"


@pytest.fixture(scope="session")
def output_dir():
    """Provide path to output directory."""
    output = ROOT / "data" / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture(autouse=True)
def suppress_warnings():
    """
    Automatically suppress known third-party warnings for all tests.
    
    This fixture runs before every test to ensure clean output.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="imageio_ffmpeg")
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
        yield


@pytest.fixture
def sample_frame():
    """Provide a sample frame for testing image processing functions."""
    import numpy as np
    # Create a 480x640 RGB frame with random values
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_metadata():
    """Provide sample video metadata for testing."""
    return {
        "name": "test_video.mp4",
        "path": str(ROOT / "data" / "test" / "test_video.mp4"),
        "fps": 30.0,
        "frame_count": 300,
        "width": 1920,
        "height": 1080,
        "duration_s": 10.0,
        "format": "mp4",
        "file_size_bytes": 1024000
    }
