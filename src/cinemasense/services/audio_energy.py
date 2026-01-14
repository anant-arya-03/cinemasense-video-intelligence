"""
Audio energy analysis using librosa
"""

import numpy as np
import librosa


def extract_rms_energy(video_path: str, target_sr: int = 22050, hop_length: int = 512):
    """Extract RMS energy from video audio track"""
    y, sr = librosa.load(video_path, sr=target_sr, mono=True)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return int(sr), int(hop_length), times.tolist(), rms.tolist()