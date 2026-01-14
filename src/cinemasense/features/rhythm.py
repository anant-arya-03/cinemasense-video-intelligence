"""
Advanced rhythm and pacing analysis
"""

import numpy as np
from typing import List, Dict


def analyze_rhythm_patterns(cut_times: List[float], duration_s: float) -> Dict:
    """Analyze rhythm patterns in cut timing"""
    if len(cut_times) < 2:
        return {
            "rhythm_regularity": 0.0,
            "acceleration_trend": 0.0,
            "rhythm_complexity": 0.0
        }
    
    # Calculate intervals between cuts
    intervals = np.diff(sorted(cut_times))
    
    # Rhythm regularity (lower variance = more regular)
    regularity = 1.0 / (1.0 + np.std(intervals)) if len(intervals) > 1 else 0.0
    
    # Acceleration trend (are cuts getting faster or slower?)
    if len(intervals) > 2:
        # Linear regression on intervals
        x = np.arange(len(intervals))
        slope = np.polyfit(x, intervals, 1)[0]
        acceleration = -slope  # Negative slope = acceleration (shorter intervals)
    else:
        acceleration = 0.0
    
    # Rhythm complexity (entropy of interval distribution)
    if len(intervals) > 1:
        hist, _ = np.histogram(intervals, bins=min(10, len(intervals)))
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log(hist + 1e-10))  # Add small value to avoid log(0)
        complexity = entropy / np.log(len(hist))  # Normalize by max entropy
    else:
        complexity = 0.0
    
    return {
        "rhythm_regularity": float(regularity),
        "acceleration_trend": float(acceleration),
        "rhythm_complexity": float(complexity)
    }


def calculate_pacing_score(cuts_per_min: float, rhythm_regularity: float, 
                          acceleration_trend: float) -> Dict:
    """Calculate comprehensive pacing score"""
    # Base score from cuts per minute
    base_score = min(cuts_per_min / 60.0, 1.0)  # Normalize to 0-1
    
    # Adjust for rhythm characteristics
    rhythm_bonus = rhythm_regularity * 0.2  # Regular rhythm is good
    acceleration_bonus = abs(acceleration_trend) * 0.1  # Dynamic pacing is interesting
    
    final_score = base_score + rhythm_bonus + acceleration_bonus
    final_score = min(final_score, 1.0)  # Cap at 1.0
    
    # Classify pacing style
    if cuts_per_min < 10:
        style = "Contemplative"
    elif cuts_per_min < 20:
        style = "Steady"
    elif cuts_per_min < 35:
        style = "Dynamic"
    else:
        style = "Frenetic"
    
    return {
        "pacing_score": float(final_score),
        "pacing_style": style,
        "cuts_per_minute": float(cuts_per_min)
    }