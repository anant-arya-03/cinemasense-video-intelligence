"""
Video quality and motion analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict


def analyze_motion_magnitude(video_path: str, sample_every_n_frames: int = 5) -> List[float]:
    """Analyze motion magnitude using optical flow"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for motion analysis: {video_path}")
    
    try:
        motion_magnitudes = []
        prev_gray = None
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx % sample_every_n_frames != 0:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # Calculate optical flow
                flow = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, 
                    np.array([[x, y] for x in range(0, gray.shape[1], 20) 
                             for y in range(0, gray.shape[0], 20)], dtype=np.float32).reshape(-1, 1, 2),
                    None
                )[0]
                
                if flow is not None and len(flow) > 0:
                    # Calculate magnitude of motion vectors
                    magnitudes = np.sqrt(flow[:, 0, 0]**2 + flow[:, 0, 1]**2)
                    avg_magnitude = np.mean(magnitudes[~np.isnan(magnitudes)])
                    motion_magnitudes.append(float(avg_magnitude) if not np.isnan(avg_magnitude) else 0.0)
                else:
                    motion_magnitudes.append(0.0)
            
            prev_gray = gray
        
        return motion_magnitudes
    
    finally:
        cap.release()


def analyze_brightness_variance(video_path: str, sample_every_n_frames: int = 5) -> List[float]:
    """Analyze brightness variance across frames"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for brightness analysis: {video_path}")
    
    try:
        brightness_variances = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            if frame_idx % sample_every_n_frames != 0:
                continue
            
            # Convert to grayscale and calculate variance
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = float(np.var(gray))
            brightness_variances.append(variance)
        
        return brightness_variances
    
    finally:
        cap.release()


def calculate_quality_metrics(motion_magnitudes: List[float], 
                            brightness_variances: List[float]) -> Dict:
    """Calculate overall quality metrics"""
    if not motion_magnitudes or not brightness_variances:
        return {
            "avg_motion": 0.0,
            "motion_stability": 0.0,
            "avg_brightness_var": 0.0,
            "visual_complexity": 0.0
        }
    
    motion_array = np.array(motion_magnitudes)
    brightness_array = np.array(brightness_variances)
    
    return {
        "avg_motion": float(np.mean(motion_array)),
        "motion_stability": float(1.0 / (1.0 + np.std(motion_array))),  # Higher = more stable
        "avg_brightness_var": float(np.mean(brightness_array)),
        "visual_complexity": float(np.mean(brightness_array) * np.mean(motion_array))
    }