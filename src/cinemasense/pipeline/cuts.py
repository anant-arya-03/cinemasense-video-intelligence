"""
Video cut detection algorithms
"""

import cv2
import numpy as np
from typing import List, Tuple


def detect_cuts_histogram(video_path: str, sample_every_n_frames: int = 2, 
                         threshold: float = 0.55) -> Tuple[List[int], List[float], List[float]]:
    """Detect cuts using histogram correlation analysis"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for cut detection: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cut_frames, cut_times, diff_series = [], [], []

        prev_hist = None
        frame_idx = -1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1

            if frame_idx % sample_every_n_frames != 0:
                continue

            # Convert to HSV and calculate histogram
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                # Calculate correlation and difference
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                diff = float(1.0 - corr)
                diff_series.append(diff)
                
                if diff >= threshold:
                    cut_frames.append(frame_idx)
                    cut_times.append(frame_idx / fps)

            prev_hist = hist

        return cut_frames, cut_times, diff_series
    
    finally:
        cap.release()


def calculate_rhythm_score(cut_times: List[float], duration_s: float) -> Tuple[float, str]:
    """Calculate rhythm score and pace classification"""
    if duration_s <= 0:
        return 0.0, "Unknown"
    
    cuts_per_min = len(cut_times) / (duration_s / 60.0)
    
    if cuts_per_min < 15:
        pace = "Slow 🐢"
    elif cuts_per_min < 30:
        pace = "Medium 🐇"
    else:
        pace = "Fast ⚡"
    
    return cuts_per_min, pace