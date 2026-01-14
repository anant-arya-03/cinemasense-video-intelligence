"""
Keyframe extraction for storyboard generation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from cinemasense.constants import THUMBNAIL_SIZE


def extract_keyframes_from_cuts(video_path: str, cut_times: List[float], 
                               output_dir: Path, max_frames: int = 50) -> List[dict]:
    """Extract keyframes based on cut detection results"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for keyframe extraction: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        keyframes = []
        
        # If no cuts detected, extract frames at regular intervals
        if not cut_times:
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            interval = max(duration / max_frames, 1.0)
            cut_times = [i * interval for i in range(min(max_frames, int(duration)))]
        
        # Limit number of keyframes
        if len(cut_times) > max_frames:
            step = len(cut_times) // max_frames
            cut_times = cut_times[::step]
        
        for i, time_s in enumerate(cut_times):
            frame_number = int(time_s * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Resize frame for thumbnail
            thumbnail = cv2.resize(frame, THUMBNAIL_SIZE)
            
            # Save thumbnail
            thumbnail_path = output_dir / f"keyframe_{i:03d}_{time_s:.2f}s.jpg"
            cv2.imwrite(str(thumbnail_path), thumbnail)
            
            keyframes.append({
                "index": i,
                "time_s": time_s,
                "frame_number": frame_number,
                "thumbnail_path": str(thumbnail_path.relative_to(output_dir.parent.parent))
            })
        
        return keyframes
    
    finally:
        cap.release()


def extract_keyframes_interval(video_path: str, interval_s: float, 
                              output_dir: Path, max_frames: int = 50) -> List[dict]:
    """Extract keyframes at regular intervals"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for keyframe extraction: {video_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        keyframes = []
        current_time = 0.0
        frame_index = 0
        
        while current_time < duration and frame_index < max_frames:
            frame_number = int(current_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for thumbnail
            thumbnail = cv2.resize(frame, THUMBNAIL_SIZE)
            
            # Save thumbnail
            thumbnail_path = output_dir / f"keyframe_{frame_index:03d}_{current_time:.2f}s.jpg"
            cv2.imwrite(str(thumbnail_path), thumbnail)
            
            keyframes.append({
                "index": frame_index,
                "time_s": current_time,
                "frame_number": frame_number,
                "thumbnail_path": str(thumbnail_path.relative_to(output_dir.parent.parent))
            })
            
            current_time += interval_s
            frame_index += 1
        
        return keyframes
    
    finally:
        cap.release()