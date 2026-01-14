"""
Application constants and configuration values
"""

# Video processing
DEFAULT_SAMPLE_EVERY_N_FRAMES = 2
DEFAULT_CUT_THRESHOLD = 0.55
DEFAULT_KEYFRAME_INTERVAL = 5.0  # seconds
DEFAULT_SCENE_GAP_THRESHOLD = 3.0  # seconds between cuts to group into scenes

# Audio processing
DEFAULT_AUDIO_SAMPLE_RATE = 22050
DEFAULT_AUDIO_HOP_LENGTH = 512

# Supported formats
SUPPORTED_VIDEO_FORMATS = ["mp4", "mov", "mkv", "avi", "webm"]
SUPPORTED_IMAGE_FORMATS = ["jpg", "jpeg", "png"]

# UI settings
MAX_STORYBOARD_THUMBNAILS = 50
THUMBNAIL_SIZE = (160, 90)  # 16:9 aspect ratio

# Mood classification thresholds
WARM_HUE_RANGE = (0, 30, 150, 180)  # Red-orange and yellow ranges in HSV
BRIGHTNESS_THRESHOLD = 0.5

# Anomaly detection
ANOMALY_CONTAMINATION = 0.1  # 10% of segments flagged as anomalies
MIN_SEGMENTS_FOR_ANOMALY = 5