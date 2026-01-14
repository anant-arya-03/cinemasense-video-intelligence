"""
Mood and color analysis from keyframes
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from cinemasense.constants import WARM_HUE_RANGE, BRIGHTNESS_THRESHOLD


def analyze_keyframe_colors(keyframes: List[Dict]) -> List[Dict]:
    """Analyze color characteristics of keyframes"""
    color_analysis = []
    
    for keyframe in keyframes:
        thumbnail_path = keyframe.get("thumbnail_path")
        if not thumbnail_path:
            continue
        
        # Load thumbnail
        img = cv2.imread(str(thumbnail_path))
        if img is None:
            continue
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate average HSV values
        avg_hue = np.mean(hsv[:, :, 0])
        avg_saturation = np.mean(hsv[:, :, 1]) / 255.0
        avg_brightness = np.mean(hsv[:, :, 2]) / 255.0
        
        # Classify warmth
        is_warm = (
            (0 <= avg_hue <= WARM_HUE_RANGE[1]) or 
            (WARM_HUE_RANGE[2] <= avg_hue <= WARM_HUE_RANGE[3])
        )
        
        # Calculate RGB averages for additional analysis
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        avg_red = np.mean(rgb[:, :, 0]) / 255.0
        avg_green = np.mean(rgb[:, :, 1]) / 255.0
        avg_blue = np.mean(rgb[:, :, 2]) / 255.0
        
        color_analysis.append({
            "keyframe_index": keyframe["index"],
            "time_s": keyframe["time_s"],
            "avg_hue": float(avg_hue),
            "avg_saturation": float(avg_saturation),
            "avg_brightness": float(avg_brightness),
            "is_warm": is_warm,
            "avg_red": float(avg_red),
            "avg_green": float(avg_green),
            "avg_blue": float(avg_blue)
        })
    
    return color_analysis


def analyze_mood_progression(color_analysis: List[Dict]) -> Dict:
    """Analyze mood progression throughout the video"""
    if not color_analysis:
        return {
            "overall_warmth": 0.5,
            "overall_brightness": 0.5,
            "overall_saturation": 0.5,
            "mood_stability": 0.0,
            "dominant_mood": "Neutral"
        }
    
    # Calculate overall characteristics
    warmth_scores = [1.0 if frame["is_warm"] else 0.0 for frame in color_analysis]
    brightness_scores = [frame["avg_brightness"] for frame in color_analysis]
    saturation_scores = [frame["avg_saturation"] for frame in color_analysis]
    
    overall_warmth = np.mean(warmth_scores)
    overall_brightness = np.mean(brightness_scores)
    overall_saturation = np.mean(saturation_scores)
    
    # Calculate mood stability (lower variance = more stable)
    brightness_stability = 1.0 / (1.0 + np.std(brightness_scores))
    saturation_stability = 1.0 / (1.0 + np.std(saturation_scores))
    mood_stability = (brightness_stability + saturation_stability) / 2.0
    
    # Classify dominant mood
    dominant_mood = classify_dominant_mood(overall_warmth, overall_brightness, overall_saturation)
    
    return {
        "overall_warmth": float(overall_warmth),
        "overall_brightness": float(overall_brightness),
        "overall_saturation": float(overall_saturation),
        "mood_stability": float(mood_stability),
        "dominant_mood": dominant_mood
    }


def classify_dominant_mood(warmth: float, brightness: float, saturation: float) -> str:
    """Classify the dominant mood based on color characteristics"""
    if brightness > 0.7 and saturation > 0.5:
        return "Energetic" if warmth > 0.6 else "Cool & Bright"
    elif brightness < 0.3:
        return "Dark & Moody" if saturation > 0.4 else "Somber"
    elif saturation < 0.3:
        return "Muted & Calm"
    elif warmth > 0.6:
        return "Warm & Inviting"
    else:
        return "Cool & Balanced"


def create_mood_timeline(color_analysis: List[Dict]) -> List[Dict]:
    """Create a timeline of mood changes"""
    timeline = []
    
    for frame in color_analysis:
        mood = classify_dominant_mood(
            1.0 if frame["is_warm"] else 0.0,
            frame["avg_brightness"],
            frame["avg_saturation"]
        )
        
        timeline.append({
            "time_s": frame["time_s"],
            "mood": mood,
            "warmth": frame["is_warm"],
            "brightness": frame["avg_brightness"],
            "saturation": frame["avg_saturation"]
        })
    
    return timeline