"""
Emotion Rhythm Score (ERS) - Advanced emotional analysis pipeline

Provides emotion analysis for video content by extracting visual features
and classifying emotional characteristics over time.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from ..core.video_capture import SafeVideoCapture, VideoOpenError

logger = logging.getLogger("cinemasense.pipeline.emotion_rhythm")

# Valid emotion categories
EMOTION_CATEGORIES = ["Joy", "Tension", "Calm", "Melancholy", "Energy", "Mystery"]

# Valid rhythm patterns
RHYTHM_PATTERNS = [
    "Sustained High Energy",
    "Calm & Steady", 
    "Building Crescendo",
    "Descending Arc",
    "Dynamic Rollercoaster",
    "Gradual Rise",
    "Gentle Decline",
    "Balanced Flow",
    "Unknown"
]

# Minimum video duration in seconds for analysis
MIN_VIDEO_DURATION = 0.1


@dataclass
class EmotionFrame:
    """Single frame emotion data"""
    timestamp: float
    frame_index: int
    brightness: float
    saturation: float
    motion_intensity: float
    color_temperature: float  # warm vs cool
    contrast: float
    emotion_score: float
    dominant_emotion: str


@dataclass
class EmotionRhythmResult:
    """Complete ERS analysis result"""
    timeline: List[EmotionFrame]
    overall_score: float
    emotion_distribution: Dict[str, float]
    peak_moments: List[Dict]
    rhythm_pattern: str
    heatmap_data: np.ndarray
    confidence: float


class EmotionAnalysisError(Exception):
    """Raised when emotion analysis fails."""
    pass


def analyze_frame_emotion(frame: np.ndarray, prev_frame: Optional[np.ndarray] = None) -> Dict:
    """
    Analyze emotional characteristics of a single frame.
    
    Args:
        frame: BGR image frame
        prev_frame: Previous frame for motion analysis (optional)
        
    Returns:
        Dictionary with brightness, saturation, color_temperature, contrast, motion_intensity
        
    Raises:
        EmotionAnalysisError: If frame analysis fails
        
    Requirements: 5.1
    """
    if frame is None or frame.size == 0:
        raise EmotionAnalysisError("Invalid frame: frame is None or empty")
    
    try:
        # Convert to different color spaces
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Extract features
        brightness = np.mean(hsv[:, :, 2]) / 255.0
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # Color temperature (warm vs cool based on LAB b channel)
        b_channel = lab[:, :, 2]
        color_temp = (np.mean(b_channel) - 128) / 128.0  # -1 (cool) to 1 (warm)
        
        # Contrast
        contrast = np.std(gray) / 128.0
        
        # Motion intensity
        motion_intensity = 0.0
        if prev_frame is not None and prev_frame.size > 0:
            try:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                motion_intensity = np.mean(diff) / 255.0
            except Exception as e:
                logger.warning(f"Motion analysis failed: {e}")
                motion_intensity = 0.0
        
        return {
            "brightness": float(brightness),
            "saturation": float(saturation),
            "color_temperature": float(color_temp),
            "contrast": float(contrast),
            "motion_intensity": float(motion_intensity)
        }
    except cv2.error as e:
        raise EmotionAnalysisError(f"OpenCV error during frame analysis: {e}")
    except Exception as e:
        raise EmotionAnalysisError(f"Frame analysis failed: {e}")


def classify_emotion(features: Dict) -> Tuple[str, float]:
    """
    Classify dominant emotion from frame features.
    
    Args:
        features: Dictionary with brightness, saturation, color_temperature, contrast, motion_intensity
        
    Returns:
        Tuple of (dominant_emotion, confidence) where confidence is in [0, 1]
        
    Requirements: 5.2
    """
    brightness = features.get("brightness", 0.5)
    saturation = features.get("saturation", 0.5)
    color_temp = features.get("color_temperature", 0.0)
    contrast = features.get("contrast", 0.5)
    motion = features.get("motion_intensity", 0.0)
    
    # Emotion scoring based on visual characteristics
    emotions = {
        "Joy": (brightness * 0.4 + saturation * 0.3 + max(0, color_temp) * 0.2 + (1 - contrast) * 0.1),
        "Tension": (contrast * 0.4 + motion * 0.3 + (1 - brightness) * 0.2 + saturation * 0.1),
        "Calm": ((1 - motion) * 0.4 + (1 - contrast) * 0.3 + brightness * 0.2 + (1 - saturation) * 0.1),
        "Melancholy": ((1 - saturation) * 0.4 + (1 - brightness) * 0.3 + min(0, color_temp) * -0.2 + (1 - motion) * 0.1),
        "Energy": (motion * 0.4 + saturation * 0.3 + contrast * 0.2 + brightness * 0.1),
        "Mystery": ((1 - brightness) * 0.4 + contrast * 0.3 + (1 - saturation) * 0.2 + motion * 0.1)
    }
    
    # Normalize scores
    total = sum(emotions.values())
    if total > 0:
        emotions = {k: v / total for k, v in emotions.items()}
    else:
        # Default to equal distribution
        emotions = {k: 1.0 / len(emotions) for k in emotions}
    
    dominant = max(emotions, key=emotions.get)
    confidence = emotions[dominant]
    
    # Ensure confidence is in [0, 1]
    confidence = max(0.0, min(1.0, confidence))
    
    return dominant, confidence


def calculate_emotion_score(features: Dict, emotion: str) -> float:
    """
    Calculate overall emotion intensity score (0-100).
    
    Args:
        features: Frame features dictionary
        emotion: Dominant emotion string
        
    Returns:
        Score between 0 and 100
    """
    base_score = 50.0
    
    # Adjust based on features
    base_score += features.get("saturation", 0.5) * 20
    base_score += features.get("contrast", 0.5) * 15
    base_score += features.get("motion_intensity", 0.0) * 15
    
    # Emotion-specific adjustments
    if emotion in ["Tension", "Energy"]:
        base_score += 10
    elif emotion in ["Calm", "Melancholy"]:
        base_score -= 5
    
    return max(0, min(100, base_score))


def _validate_video_for_analysis(duration: float, frame_count: int) -> None:
    """
    Validate video is suitable for emotion analysis.
    
    Args:
        duration: Video duration in seconds
        frame_count: Total frame count
        
    Raises:
        EmotionAnalysisError: If video is too short or empty
        
    Requirements: 5.5
    """
    if frame_count <= 0:
        raise EmotionAnalysisError("Video has no frames")
    
    if duration < MIN_VIDEO_DURATION:
        raise EmotionAnalysisError(
            f"Video too short for analysis. Duration: {duration:.2f}s, "
            f"minimum required: {MIN_VIDEO_DURATION}s"
        )


def _create_heatmap(
    timeline: List[EmotionFrame],
    duration: float,
    num_emotions: int = 6
) -> np.ndarray:
    """
    Create heatmap data from emotion timeline.
    
    Args:
        timeline: List of EmotionFrame objects
        duration: Video duration in seconds
        num_emotions: Number of emotion categories
        
    Returns:
        numpy array of shape (num_emotions, heatmap_width)
        
    Requirements: 5.4
    """
    # Calculate heatmap width based on duration (1-100 columns)
    heatmap_width = max(1, min(100, int(duration)))
    heatmap_data = np.zeros((num_emotions, heatmap_width), dtype=np.float64)
    
    if not timeline or duration <= 0:
        return heatmap_data
    
    emotion_names = EMOTION_CATEGORIES
    
    for frame in timeline:
        # Calculate heatmap column index
        heatmap_idx = int(frame.timestamp / duration * heatmap_width)
        heatmap_idx = max(0, min(heatmap_idx, heatmap_width - 1))
        
        # Update heatmap for the dominant emotion
        for i, emo in enumerate(emotion_names):
            if emo == frame.dominant_emotion:
                heatmap_data[i, heatmap_idx] = max(
                    heatmap_data[i, heatmap_idx], 
                    frame.emotion_score / 100.0
                )
    
    return heatmap_data


def _calculate_emotion_distribution(emotion_counts: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate normalized emotion distribution.
    
    Args:
        emotion_counts: Dictionary of emotion -> count
        
    Returns:
        Dictionary of emotion -> proportion (sums to 1.0)
        
    Requirements: 5.3
    """
    total = sum(emotion_counts.values())
    
    if total <= 0:
        # Return uniform distribution if no data
        return {emo: 1.0 / len(EMOTION_CATEGORIES) for emo in EMOTION_CATEGORIES}
    
    distribution = {}
    for emotion in EMOTION_CATEGORIES:
        count = emotion_counts.get(emotion, 0)
        distribution[emotion] = count / total
    
    return distribution


def extract_emotion_timeline(video_path: str, sample_rate: int = 5) -> EmotionRhythmResult:
    """
    Extract complete emotion timeline from video.
    
    Uses SafeVideoCapture for proper resource management and includes
    validation for empty/short videos.
    
    Args:
        video_path: Path to video file
        sample_rate: Analyze every Nth frame (default: 5)
        
    Returns:
        EmotionRhythmResult with timeline, scores, and heatmap
        
    Raises:
        EmotionAnalysisError: If video cannot be analyzed
        VideoOpenError: If video cannot be opened
        
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    logger.info(f"Starting emotion analysis for: {video_path}")
    
    # Target size for faster processing
    TARGET_WIDTH = 320
    TARGET_HEIGHT = 180
    
    try:
        with SafeVideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # Validate video is suitable for analysis
            _validate_video_for_analysis(duration, total_frames)
            
            logger.debug(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.2f}s")
            
            # Adaptive sample rate based on video length for faster processing
            if duration > 120:  # > 2 minutes
                sample_rate = max(sample_rate, 15)
            elif duration > 60:  # > 1 minute
                sample_rate = max(sample_rate, 10)
            elif duration > 30:  # > 30 seconds
                sample_rate = max(sample_rate, 7)
            
            timeline: List[EmotionFrame] = []
            emotion_counts: Dict[str, int] = {}
            prev_frame: Optional[np.ndarray] = None
            frame_idx = 0
            frames_analyzed = 0
            frames_skipped = 0
            
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                
                # Skip empty frames
                if frame.size == 0:
                    frame_idx += 1
                    frames_skipped += 1
                    continue
                
                if frame_idx % sample_rate == 0:
                    timestamp = frame_idx / fps
                    
                    try:
                        # Resize frame for faster processing
                        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), 
                                                  interpolation=cv2.INTER_AREA)
                        
                        # Analyze resized frame
                        features = analyze_frame_emotion(small_frame, prev_frame)
                        emotion, confidence = classify_emotion(features)
                        score = calculate_emotion_score(features, emotion)
                        
                        # Create emotion frame
                        emotion_frame = EmotionFrame(
                            timestamp=timestamp,
                            frame_index=frame_idx,
                            brightness=features["brightness"],
                            saturation=features["saturation"],
                            motion_intensity=features["motion_intensity"],
                            color_temperature=features["color_temperature"],
                            contrast=features["contrast"],
                            emotion_score=score,
                            dominant_emotion=emotion
                        )
                        timeline.append(emotion_frame)
                        
                        # Update emotion counts
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                        frames_analyzed += 1
                        
                        prev_frame = small_frame.copy()
                        
                    except EmotionAnalysisError as e:
                        logger.warning(f"Skipping frame {frame_idx}: {e}")
                        frames_skipped += 1
                    except Exception as e:
                        logger.warning(f"Error processing frame {frame_idx}: {e}")
                        frames_skipped += 1
                
                frame_idx += 1
            
            logger.info(f"Analyzed {frames_analyzed} frames, skipped {frames_skipped}")
            
            # Handle case where no frames were successfully analyzed
            if not timeline:
                raise EmotionAnalysisError(
                    "No frames could be analyzed. Video may be corrupted or in an unsupported format."
                )
            
            # Calculate emotion distribution (normalized to sum to 1.0)
            emotion_distribution = _calculate_emotion_distribution(emotion_counts)
            
            # Create heatmap with correct dimensions
            heatmap_data = _create_heatmap(timeline, duration)
            
            # Find peak moments
            peak_moments = find_peak_moments(timeline)
            
            # Determine rhythm pattern
            rhythm_pattern = determine_rhythm_pattern(timeline)
            
            # Calculate overall score
            overall_score = np.mean([ef.emotion_score for ef in timeline])
            
            # Calculate confidence based on analysis coverage
            coverage = frames_analyzed / max(1, total_frames // sample_rate)
            confidence = min(0.95, max(0.1, 0.8 * coverage))
            
            result = EmotionRhythmResult(
                timeline=timeline,
                overall_score=float(overall_score),
                emotion_distribution=emotion_distribution,
                peak_moments=peak_moments,
                rhythm_pattern=rhythm_pattern,
                heatmap_data=heatmap_data,
                confidence=float(confidence)
            )
            
            logger.info(f"Emotion analysis complete. Pattern: {rhythm_pattern}, Score: {overall_score:.1f}")
            return result
            
    except VideoOpenError:
        raise
    except EmotionAnalysisError:
        raise
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}", exc_info=True)
        raise EmotionAnalysisError(f"Failed to analyze video: {e}")


def find_peak_moments(timeline: List[EmotionFrame], threshold: float = 75) -> List[Dict]:
    """
    Find emotional peak moments in timeline.
    
    Args:
        timeline: List of EmotionFrame objects
        threshold: Minimum score to consider as peak (default: 75)
        
    Returns:
        List of peak moment dictionaries with timestamp, score, emotion, reason
        
    Requirements: 5.5
    """
    if not timeline:
        return []
    
    peaks = []
    
    for i, frame in enumerate(timeline):
        if frame.emotion_score >= threshold:
            # Check if it's a local maximum
            is_peak = True
            window = 3
            
            for j in range(max(0, i - window), min(len(timeline), i + window + 1)):
                if j != i and timeline[j].emotion_score > frame.emotion_score:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append({
                    "timestamp": frame.timestamp,
                    "score": frame.emotion_score,
                    "emotion": frame.dominant_emotion,
                    "reason": f"High {frame.dominant_emotion.lower()} intensity detected"
                })
    
    return peaks[:10]  # Return top 10 peaks


def determine_rhythm_pattern(timeline: List[EmotionFrame]) -> str:
    """
    Determine the overall emotional rhythm pattern.
    
    Args:
        timeline: List of EmotionFrame objects
        
    Returns:
        One of the valid rhythm patterns from RHYTHM_PATTERNS
        
    Requirements: 5.6
    """
    if not timeline:
        return "Unknown"
    
    scores = [ef.emotion_score for ef in timeline]
    
    # Calculate variance and trend
    variance = np.var(scores)
    
    # Calculate trend (rising, falling, stable)
    if len(scores) > 10:
        first_half = np.mean(scores[:len(scores)//2])
        second_half = np.mean(scores[len(scores)//2:])
        trend = second_half - first_half
    else:
        trend = 0
    
    # Classify pattern
    if variance < 100:
        if np.mean(scores) > 60:
            return "Sustained High Energy"
        else:
            return "Calm & Steady"
    elif variance > 400:
        if trend > 10:
            return "Building Crescendo"
        elif trend < -10:
            return "Descending Arc"
        else:
            return "Dynamic Rollercoaster"
    else:
        if trend > 5:
            return "Gradual Rise"
        elif trend < -5:
            return "Gentle Decline"
        else:
            return "Balanced Flow"
