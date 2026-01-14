"""
Configuration settings for CinemaSense
"""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_INPUT_DIR = ROOT_DIR / "data" / "input"
DATA_OUTPUT_DIR = ROOT_DIR / "data" / "output"

# Default analysis settings
DEFAULT_SAMPLE_EVERY_N_FRAMES = 2
DEFAULT_CUT_THRESHOLD = 0.55
DEFAULT_AUDIO_SAMPLE_RATE = 22050
DEFAULT_AUDIO_HOP_LENGTH = 512

# Video settings
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "mkv"]