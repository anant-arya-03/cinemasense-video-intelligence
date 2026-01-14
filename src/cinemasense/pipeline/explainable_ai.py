"""
Explainable AI - Cut detection with reasoning and confidence scores

Provides detailed explanations for detected cuts including primary reasons,
secondary factors, confidence scores, and cut type classification.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import logging

from ..core.video_capture import SafeVideoCapture, VideoOpenError

logger = logging.getLogger("cinemasense.pipeline.explainable_ai")

# Valid cut types as per design document
VALID_CUT_TYPES = {"hard_cut", "dissolve", "fade_to_black", "fade_to_white"}


@dataclass
class CutExplanation:
    """
    Detailed explanation for a detected cut.
    
    Requirements: 6.1, 6.2, 6.3, 6.4
    """
    frame_index: int
    timestamp: float
    confidence: float
    primary_reason: str
    secondary_reasons: List[str]
    visual_change_score: float
    color_change_score: float
    motion_discontinuity: float
    cut_type: str  # hard_cut, dissolve, fade_to_black, fade_to_white
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure confidence is in [0, 1]
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Ensure primary_reason is non-empty
        if not self.primary_reason or not self.primary_reason.strip():
            self.primary_reason = "Visual transition detected"
        
        # Ensure secondary_reasons is a list
        if self.secondary_reasons is None:
            self.secondary_reasons = []
        
        # Ensure cut_type is valid
        if self.cut_type not in VALID_CUT_TYPES:
            self.cut_type = "hard_cut"
        
        # Ensure scores are non-negative
        self.visual_change_score = max(0.0, self.visual_change_score)
        self.color_change_score = max(0.0, self.color_change_score)
        self.motion_discontinuity = max(0.0, self.motion_discontinuity)


@dataclass
class ExplainableAnalysis:
    """
    Complete explainable analysis result.
    
    Requirements: 6.5
    """
    cuts: List[CutExplanation]
    total_cuts: int
    avg_confidence: float
    cut_type_distribution: Dict[str, int]
    explanation_summary: str
    
    def __post_init__(self):
        """Validate and normalize fields after initialization."""
        # Ensure avg_confidence is in [0, 1]
        self.avg_confidence = max(0.0, min(1.0, self.avg_confidence))
        
        # Ensure total_cuts matches cuts list
        self.total_cuts = len(self.cuts)
        
        # Ensure explanation_summary is non-empty
        if not self.explanation_summary or not self.explanation_summary.strip():
            self.explanation_summary = "No significant cuts detected in the video."


def analyze_cut_reason(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    hist_diff: float,
    motion_diff: float
) -> Tuple[str, List[str], str]:
    """
    Analyze and explain why a cut was detected.
    
    Args:
        frame_before: Frame before the cut
        frame_after: Frame after the cut
        hist_diff: Histogram difference score
        motion_diff: Motion difference score
        
    Returns:
        Tuple of (primary_reason, secondary_reasons, cut_type)
        
    Requirements: 6.1, 6.4
    """
    reasons = []
    cut_type = "hard_cut"
    
    try:
        # Color histogram analysis
        hsv_before = cv2.cvtColor(frame_before, cv2.COLOR_BGR2HSV)
        hsv_after = cv2.cvtColor(frame_after, cv2.COLOR_BGR2HSV)
        
        # Hue change
        hue_diff = np.abs(np.mean(hsv_before[:, :, 0]) - np.mean(hsv_after[:, :, 0]))
        if hue_diff > 30:
            reasons.append(f"Significant color palette shift (Δhue={hue_diff:.1f}°)")
        
        # Brightness change
        brightness_before = np.mean(hsv_before[:, :, 2])
        brightness_after = np.mean(hsv_after[:, :, 2])
        brightness_diff = abs(brightness_after - brightness_before)
        
        if brightness_diff > 50:
            if brightness_after < brightness_before:
                reasons.append(f"Scene darkening detected (Δbrightness={brightness_diff:.1f})")
                if brightness_after < 30:
                    cut_type = "fade_to_black"
            else:
                reasons.append(f"Scene brightening detected (Δbrightness={brightness_diff:.1f})")
                if brightness_after > 225:
                    cut_type = "fade_to_white"
        
        # Edge detection for scene structure change
        edges_before = cv2.Canny(cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY), 50, 150)
        edges_after = cv2.Canny(cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_diff = np.mean(np.abs(edges_before.astype(float) - edges_after.astype(float)))
        
        if edge_diff > 30:
            reasons.append(f"Scene structure change detected (edge_diff={edge_diff:.1f})")
        
        # Motion discontinuity
        if motion_diff > 0.5:
            reasons.append(f"Motion discontinuity detected (motion_diff={motion_diff:.2f})")
        
        # Histogram correlation
        if hist_diff > 0.6:
            reasons.append(f"Visual content change (histogram_diff={hist_diff:.2f})")
        
        # Detect dissolve (gradual transition)
        if 0.3 < hist_diff < 0.6 and brightness_diff < 30:
            cut_type = "dissolve"
            
    except Exception as e:
        logger.warning(f"Error analyzing cut reason: {e}")
        reasons.append("Visual transition detected")
    
    # Determine primary reason - ensure it's never empty
    if not reasons:
        primary_reason = "Subtle visual transition detected"
        secondary_reasons = []
    else:
        primary_reason = reasons[0]
        secondary_reasons = reasons[1:] if len(reasons) > 1 else []
    
    # Validate cut_type
    if cut_type not in VALID_CUT_TYPES:
        cut_type = "hard_cut"
    
    return primary_reason, secondary_reasons, cut_type


def calculate_confidence(
    hist_diff: float,
    motion_diff: float,
    edge_diff: float,
    brightness_diff: float
) -> float:
    """
    Calculate confidence score for cut detection.
    
    Args:
        hist_diff: Histogram difference (0-1+)
        motion_diff: Motion difference (0-1)
        edge_diff: Edge difference score
        brightness_diff: Brightness difference
        
    Returns:
        Confidence score clamped to [0, 1]
        
    Requirements: 6.2
    """
    # Weighted combination of factors
    confidence = 0.0
    
    # Histogram difference is primary indicator
    confidence += min(hist_diff / 0.7, 1.0) * 0.4
    
    # Motion discontinuity
    confidence += min(motion_diff / 0.8, 1.0) * 0.25
    
    # Edge structure change
    confidence += min(edge_diff / 50, 1.0) * 0.2
    
    # Brightness change
    confidence += min(brightness_diff / 100, 1.0) * 0.15
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, confidence))


def generate_analysis_summary(cuts: List[CutExplanation], cut_types: Dict[str, int]) -> str:
    """
    Generate human-readable analysis summary.
    
    Args:
        cuts: List of detected cuts with explanations
        cut_types: Distribution of cut types
        
    Returns:
        Human-readable summary string
        
    Requirements: 6.5
    """
    if not cuts:
        return "No significant cuts detected in the video."
    
    total = len(cuts)
    avg_conf = np.mean([c.confidence for c in cuts])
    
    # Most common cut type
    most_common = max(cut_types, key=cut_types.get) if cut_types else "hard_cut"
    most_common_count = cut_types.get(most_common, 0)
    
    # High confidence cuts
    high_conf_cuts = [c for c in cuts if c.confidence > 0.8]
    
    summary_parts = [
        f"Detected {total} cut{'s' if total != 1 else ''} with {avg_conf:.0%} average confidence.",
        f"Most common cut type: {most_common.replace('_', ' ').title()} ({most_common_count} occurrence{'s' if most_common_count != 1 else ''}).",
    ]
    
    if high_conf_cuts:
        summary_parts.append(f"{len(high_conf_cuts)} high-confidence cut{'s' if len(high_conf_cuts) != 1 else ''} identified.")
    
    # Notable patterns
    if cut_types.get("fade_to_black", 0) > 2:
        summary_parts.append("Multiple fade-to-black transitions suggest scene breaks or time jumps.")
    
    if cut_types.get("fade_to_white", 0) > 2:
        summary_parts.append("Multiple fade-to-white transitions detected, often used for flashbacks or dream sequences.")
    
    dissolve_count = cut_types.get("dissolve", 0)
    if dissolve_count > total * 0.3:
        summary_parts.append("Frequent dissolves indicate a contemplative or artistic editing style.")
    
    return " ".join(summary_parts)


def detect_cuts_with_explanation(
    video_path: str,
    sample_every_n_frames: int = 2,
    threshold: float = 0.55
) -> ExplainableAnalysis:
    """
    Detect cuts with detailed explanations.
    
    Uses SafeVideoCapture for proper resource management and provides
    complete explanations for each detected cut.
    
    Args:
        video_path: Path to video file
        sample_every_n_frames: Sample rate for frame analysis
        threshold: Histogram difference threshold for cut detection
        
    Returns:
        ExplainableAnalysis with all detected cuts and summary
        
    Raises:
        VideoOpenError: If video cannot be opened
        RuntimeError: If video processing fails
        
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    logger.info(f"Starting cut detection for: {video_path}")
    
    cuts: List[CutExplanation] = []
    cut_types: Dict[str, int] = {}
    
    # Target size for faster processing
    TARGET_WIDTH = 320
    TARGET_HEIGHT = 180
    
    try:
        with SafeVideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.debug(f"Video info: {total_frames} frames at {fps} fps")
            
            # Adaptive sample rate based on video length
            # For longer videos, sample less frequently
            duration = total_frames / fps if fps > 0 else 0
            if duration > 120:  # > 2 minutes
                sample_every_n_frames = max(sample_every_n_frames, 5)
            elif duration > 60:  # > 1 minute
                sample_every_n_frames = max(sample_every_n_frames, 3)
            
            prev_frame = None
            prev_hist = None
            frame_idx = 0
            frames_processed = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_every_n_frames == 0:
                    try:
                        # Resize frame for faster processing
                        small_frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), 
                                                  interpolation=cv2.INTER_AREA)
                        
                        # Calculate histogram on small frame
                        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                        hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                        cv2.normalize(hist, hist)
                        
                        if prev_hist is not None and prev_frame is not None:
                            # Calculate histogram difference
                            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                            hist_diff = float(1.0 - corr)
                            
                            if hist_diff >= threshold:
                                # Only do detailed analysis when cut is detected
                                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                                motion_diff = np.mean(np.abs(gray.astype(float) - prev_gray.astype(float))) / 255.0
                                
                                # Calculate edge difference
                                edges = cv2.Canny(gray, 50, 150)
                                prev_edges = cv2.Canny(prev_gray, 50, 150)
                                edge_diff = np.mean(np.abs(edges.astype(float) - prev_edges.astype(float)))
                                
                                # Calculate brightness difference
                                brightness_diff = abs(np.mean(small_frame) - np.mean(prev_frame))
                                
                                # Analyze cut reason
                                primary_reason, secondary_reasons, cut_type = analyze_cut_reason(
                                    prev_frame, small_frame, hist_diff, motion_diff
                                )
                                
                                # Calculate confidence
                                confidence = calculate_confidence(hist_diff, motion_diff, edge_diff, brightness_diff)
                                
                                # Calculate color change score
                                prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
                                color_change_score = float(np.mean(np.abs(hsv.astype(float) - prev_hsv.astype(float))))
                                
                                # Create cut explanation
                                cut_explanation = CutExplanation(
                                    frame_index=frame_idx,
                                    timestamp=frame_idx / fps,
                                    confidence=confidence,
                                    primary_reason=primary_reason,
                                    secondary_reasons=secondary_reasons,
                                    visual_change_score=hist_diff,
                                    color_change_score=color_change_score,
                                    motion_discontinuity=motion_diff,
                                    cut_type=cut_type
                                )
                                cuts.append(cut_explanation)
                                
                                # Track cut types
                                cut_types[cut_type] = cut_types.get(cut_type, 0) + 1
                                
                                logger.debug(
                                    f"Cut detected at frame {frame_idx} "
                                    f"(t={frame_idx/fps:.2f}s): {cut_type}, "
                                    f"confidence={confidence:.2f}"
                                )
                        
                        prev_hist = hist
                        prev_frame = small_frame.copy()
                        frames_processed += 1
                        
                    except Exception as e:
                        # Skip frame on error but continue processing
                        logger.warning(f"Error processing frame {frame_idx}: {e}")
                
                frame_idx += 1
            
            logger.info(f"Processed {frames_processed} frames, detected {len(cuts)} cuts")
        
        # Generate summary
        avg_confidence = float(np.mean([c.confidence for c in cuts])) if cuts else 0.0
        summary = generate_analysis_summary(cuts, cut_types)
        
        # Create result
        result = ExplainableAnalysis(
            cuts=cuts,
            total_cuts=len(cuts),
            avg_confidence=avg_confidence,
            cut_type_distribution=cut_types,
            explanation_summary=summary
        )
        
        logger.info(f"Cut detection complete: {result.total_cuts} cuts, avg confidence {result.avg_confidence:.2%}")
        return result
        
    except VideoOpenError:
        logger.error(f"Failed to open video: {video_path}")
        raise
    except Exception as e:
        logger.error(f"Error during cut detection: {e}", exc_info=True)
        raise RuntimeError(f"Cut detection failed: {e}")


# Alias for backward compatibility
detect = detect_cuts_with_explanation
