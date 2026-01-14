"""
Audio analysis pipeline
"""

import numpy as np
import librosa
from typing import Tuple, List, Optional


def extract_audio_features(video_path: str, target_sr: int = 22050, 
                          hop_length: int = 512) -> Optional[Tuple[int, int, List[float], List[float]]]:
    """Extract RMS energy and other audio features"""
    try:
        # Load audio from video
        y, sr = librosa.load(video_path, sr=target_sr, mono=True)
        
        # Extract RMS energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        return int(sr), int(hop_length), times.tolist(), rms.tolist()
    
    except Exception as e:
        # Return None if audio extraction fails (no audio track, etc.)
        return None


def analyze_audio_energy(rms_values: List[float]) -> dict:
    """Analyze audio energy characteristics"""
    if not rms_values:
        return {"mean": 0, "std": 0, "max": 0, "energy_variance": 0}
    
    rms_array = np.array(rms_values)
    
    return {
        "mean": float(np.mean(rms_array)),
        "std": float(np.std(rms_array)),
        "max": float(np.max(rms_array)),
        "energy_variance": float(np.var(rms_array))
    }