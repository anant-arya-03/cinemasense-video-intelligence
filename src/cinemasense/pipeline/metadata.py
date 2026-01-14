"""
Video metadata extraction
"""

import cv2
from typing import Tuple


def get_video_metadata(video_path: str) -> Tuple[float, int, int, int, float]:
    """Extract comprehensive video metadata"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        
        duration_s = (frame_count / fps) if fps > 0 else 0.0
        
        return float(fps), frame_count, width, height, float(duration_s)
    finally:
        cap.release()