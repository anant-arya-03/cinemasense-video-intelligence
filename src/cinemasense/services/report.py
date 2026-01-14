"""
Report generation service for CinemaSense AI Studio.

Generates comprehensive JSON reports containing all analysis data,
video metadata, timestamps, and provides download functionality.

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("cinemasense.services.report")


@dataclass
class VideoMetadataReport:
    """Video metadata section of the report."""
    path: str
    name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    format: str
    file_size_bytes: int


@dataclass
class CutAnalysisReport:
    """Cut analysis section of the report."""
    total_cuts: int
    avg_confidence: float
    cut_type_distribution: Dict[str, int]
    cuts: List[Dict[str, Any]]


@dataclass
class EmotionAnalysisReport:
    """Emotion analysis section of the report."""
    overall_score: float
    rhythm_pattern: str
    emotion_distribution: Dict[str, float]
    peak_moments: List[Dict[str, Any]]
    confidence: float


@dataclass
class SocialPackReport:
    """Social pack section of the report."""
    thumbnail_path: Optional[str]
    title_suggestions: List[str]
    hashtags: List[str]
    caption: str
    platforms: List[str]


@dataclass
class Report:
    """
    Complete analysis report structure.
    
    Requirements: 14.1, 14.2, 14.5
    """
    generated_at: str
    video_name: str
    metadata: Dict[str, Any]
    cuts: Optional[Dict[str, Any]]
    emotion: Optional[Dict[str, Any]]
    keyframes_count: int
    social: Optional[Dict[str, Any]]
    multiverse: Optional[Dict[str, Any]]


class ReportValidationError(Exception):
    """Raised when report validation fails."""
    pass


class ReportGenerator:
    """
    Generates comprehensive JSON reports for video analysis.
    
    Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
    """
    
    # Required fields for a valid report
    REQUIRED_FIELDS = {"generated_at", "video_name", "metadata"}
    
    # Required metadata fields
    REQUIRED_METADATA_FIELDS = {"name", "fps", "frame_count", "width", "height", "duration_s"}
    
    @classmethod
    def generate(
        cls,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]] = None,
        emotion: Optional[Dict[str, Any]] = None,
        keyframes: Optional[List[Dict[str, Any]]] = None,
        social: Optional[Dict[str, Any]] = None,
        multiverse: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report dictionary.
        
        Args:
            video_name: Name of the analyzed video
            metadata: Video metadata dictionary
            cuts: Cut analysis results (optional)
            emotion: Emotion analysis results (optional)
            keyframes: Keyframe extraction results (optional)
            social: Social pack generation results (optional)
            multiverse: Multiverse style results (optional)
            
        Returns:
            Complete report dictionary
            
        Requirements: 14.1, 14.2
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "video_name": video_name,
            "metadata": cls._sanitize_metadata(metadata),
            "cuts": cls._sanitize_cuts(cuts) if cuts else None,
            "emotion": cls._sanitize_emotion(emotion) if emotion else None,
            "keyframes_count": len(keyframes) if keyframes else 0,
            "keyframes": cls._sanitize_keyframes(keyframes) if keyframes else None,
            "social": cls._sanitize_social(social) if social else None,
            "multiverse": cls._sanitize_multiverse(multiverse) if multiverse else None
        }
        
        logger.info(f"Generated report for video: {video_name}")
        return report
    
    @classmethod
    def _sanitize_metadata(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and normalize metadata for JSON serialization."""
        if not metadata:
            return {}
        
        sanitized = {}
        for key, value in metadata.items():
            sanitized[key] = cls._make_json_serializable(value)
        
        return sanitized
    
    @classmethod
    def _sanitize_cuts(cls, cuts: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize cut analysis data for JSON serialization."""
        if not cuts:
            return {}
        
        sanitized = {}
        for key, value in cuts.items():
            sanitized[key] = cls._make_json_serializable(value)
        
        return sanitized
    
    @classmethod
    def _sanitize_emotion(cls, emotion: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize emotion analysis data for JSON serialization."""
        if not emotion:
            return {}
        
        sanitized = {}
        for key, value in emotion.items():
            # Skip numpy arrays that can't be serialized directly
            if key == "heatmap_data":
                continue
            sanitized[key] = cls._make_json_serializable(value)
        
        return sanitized
    
    @classmethod
    def _sanitize_keyframes(cls, keyframes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sanitize keyframe data for JSON serialization."""
        if not keyframes:
            return []
        
        sanitized = []
        for kf in keyframes:
            sanitized_kf = {}
            for key, value in kf.items():
                # Skip frame data (numpy arrays)
                if key in ("frame", "image"):
                    continue
                sanitized_kf[key] = cls._make_json_serializable(value)
            sanitized.append(sanitized_kf)
        
        return sanitized
    
    @classmethod
    def _sanitize_social(cls, social: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize social pack data for JSON serialization."""
        if not social:
            return {}
        
        sanitized = {}
        for key, value in social.items():
            sanitized[key] = cls._make_json_serializable(value)
        
        return sanitized
    
    @classmethod
    def _sanitize_multiverse(cls, multiverse: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize multiverse data for JSON serialization."""
        if not multiverse:
            return {}
        
        sanitized = {}
        for key, value in multiverse.items():
            sanitized[key] = cls._make_json_serializable(value)
        
        return sanitized
    
    @classmethod
    def _make_json_serializable(cls, value: Any) -> Any:
        """Convert a value to a JSON-serializable type."""
        import numpy as np
        
        if value is None:
            return None
        elif isinstance(value, (str, int, bool)):
            return value
        elif isinstance(value, float):
            # Handle NaN and Inf
            if np.isnan(value) or np.isinf(value):
                return None
            return value
        elif isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, dict):
            return {k: cls._make_json_serializable(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [cls._make_json_serializable(v) for v in value]
        elif isinstance(value, Path):
            return str(value)
        elif hasattr(value, '__dict__'):
            return cls._make_json_serializable(vars(value))
        else:
            return str(value)
    
    @classmethod
    def validate(cls, report: Dict[str, Any]) -> bool:
        """
        Validate report structure before saving.
        
        Args:
            report: Report dictionary to validate
            
        Returns:
            True if valid
            
        Raises:
            ReportValidationError: If validation fails
            
        Requirements: 14.1, 14.5
        """
        # Check required top-level fields
        missing_fields = cls.REQUIRED_FIELDS - set(report.keys())
        if missing_fields:
            raise ReportValidationError(
                f"Missing required fields: {missing_fields}"
            )
        
        # Validate generated_at is a valid timestamp
        generated_at = report.get("generated_at")
        if not generated_at or not isinstance(generated_at, str):
            raise ReportValidationError("generated_at must be a non-empty string")
        
        try:
            datetime.fromisoformat(generated_at)
        except ValueError:
            raise ReportValidationError(
                f"generated_at is not a valid ISO timestamp: {generated_at}"
            )
        
        # Validate video_name
        video_name = report.get("video_name")
        if not video_name or not isinstance(video_name, str):
            raise ReportValidationError("video_name must be a non-empty string")
        
        # Validate metadata
        metadata = report.get("metadata")
        if not isinstance(metadata, dict):
            raise ReportValidationError("metadata must be a dictionary")
        
        # Check required metadata fields
        missing_meta = cls.REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing_meta:
            raise ReportValidationError(
                f"Missing required metadata fields: {missing_meta}"
            )
        
        # Validate JSON serializability
        try:
            json.dumps(report)
        except (TypeError, ValueError) as e:
            raise ReportValidationError(f"Report is not JSON serializable: {e}")
        
        logger.debug(f"Report validation passed for: {video_name}")
        return True
    
    @classmethod
    def save(
        cls,
        report: Dict[str, Any],
        output_path: Union[str, Path],
        validate: bool = True
    ) -> Path:
        """
        Save report as JSON file.
        
        Args:
            report: Report dictionary to save
            output_path: Path to save the report
            validate: Whether to validate before saving
            
        Returns:
            Path to saved report file
            
        Raises:
            ReportValidationError: If validation fails
            
        Requirements: 14.1, 14.3
        """
        output_path = Path(output_path)
        
        # Validate if requested
        if validate:
            cls.validate(report)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with pretty formatting
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        
        logger.info(f"Report saved to: {output_path}")
        return output_path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a report from JSON file.
        
        Args:
            path: Path to report file
            
        Returns:
            Report dictionary
        """
        path = Path(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        logger.info(f"Report loaded from: {path}")
        return report
    
    @classmethod
    def to_json_string(cls, report: Dict[str, Any], indent: int = 2) -> str:
        """
        Convert report to JSON string for preview or download.
        
        Args:
            report: Report dictionary
            indent: JSON indentation level
            
        Returns:
            JSON string representation
            
        Requirements: 14.4
        """
        return json.dumps(report, indent=indent, ensure_ascii=False)
    
    @classmethod
    def get_download_filename(cls, video_name: str) -> str:
        """
        Generate a filename for report download.
        
        Args:
            video_name: Name of the video
            
        Returns:
            Filename for download
            
        Requirements: 14.3, 14.5
        """
        # Sanitize video name
        safe_name = "".join(
            c if c.isalnum() or c in ("_", "-") else "_"
            for c in video_name
        ).strip("_")[:50] or "video"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cinemasense_report_{safe_name}_{timestamp}.json"
    
    @classmethod
    def generate_summary(cls, report: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the report.
        
        Args:
            report: Report dictionary
            
        Returns:
            Summary string
        """
        lines = []
        lines.append(f"CinemaSense Analysis Report")
        lines.append(f"Generated: {report.get('generated_at', 'Unknown')}")
        lines.append(f"Video: {report.get('video_name', 'Unknown')}")
        lines.append("")
        
        # Metadata summary
        metadata = report.get("metadata", {})
        if metadata:
            duration = metadata.get("duration_s", 0)
            lines.append(f"Duration: {duration:.1f}s")
            lines.append(f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}")
            lines.append(f"FPS: {metadata.get('fps', 0):.1f}")
            lines.append("")
        
        # Analysis summary
        if report.get("cuts"):
            cuts = report["cuts"]
            lines.append(f"Cuts Detected: {cuts.get('total_cuts', 0)}")
            lines.append(f"Avg Confidence: {cuts.get('avg_confidence', 0):.2f}")
            lines.append("")
        
        if report.get("emotion"):
            emotion = report["emotion"]
            lines.append(f"Emotion Score: {emotion.get('overall_score', 0):.2f}")
            lines.append(f"Rhythm Pattern: {emotion.get('rhythm_pattern', 'Unknown')}")
            lines.append("")
        
        if report.get("keyframes_count"):
            lines.append(f"Keyframes: {report['keyframes_count']}")
            lines.append("")
        
        if report.get("social"):
            social = report["social"]
            titles = social.get("title_suggestions", [])
            lines.append(f"Title Suggestions: {len(titles)}")
            lines.append("")
        
        return "\n".join(lines)


# Legacy function for backward compatibility
def create_report_dict(
    fname: str,
    fps: float,
    frame_count: int,
    width: int,
    height: int,
    duration_s: float,
    threshold: float,
    sample_every_n_frames: int,
    cut_times: list,
    cut_frames: list,
    cuts_per_min: float,
    pace: str,
    audio_payload: dict = None
) -> dict:
    """
    Create a structured report dictionary (legacy format).
    
    This function is maintained for backward compatibility.
    """
    return {
        "video_file": fname,
        "meta": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": duration_s,
        },
        "cuts": {
            "threshold": float(threshold),
            "sample_every_n_frames": int(sample_every_n_frames),
            "cut_times_s": cut_times,
            "cut_frames": cut_frames,
            "cuts_per_min": cuts_per_min,
            "pace": pace,
        },
        "audio": audio_payload,
    }


def save_report_json(report: dict, output_path: Path) -> None:
    """Save report as JSON file (legacy function)."""
    ReportGenerator.save(report, output_path, validate=False)
