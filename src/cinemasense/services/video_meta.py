"""
Video metadata extraction
"""

import cv2


def get_video_meta(video_path: str):
    """Extract video metadata using OpenCV"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Unsupported codec or invalid file.")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    
    duration_s = (frame_count / fps) if fps > 0 else 0.0
    return float(fps), frame_count, width, height, float(duration_s)