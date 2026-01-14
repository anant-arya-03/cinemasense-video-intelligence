# Design Document: CinemaSense AI Studio Stabilization

## Overview

This design document describes the architecture and implementation approach for stabilizing and enhancing CinemaSense AI Studio into an industry-ready video intelligence platform. The system follows a clean architecture with modular pipeline stages, robust error handling, and a premium glassmorphic UI.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit App (app.py)                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Session     │  │ Glassmorphic│  │ Page Router             │  │
│  │ Manager     │  │ UI System   │  │ (Home/Analysis/Styles)  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Service Layer                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Video    │ │ Analysis │ │ Export   │ │ Storage  │           │
│  │ Service  │ │ Service  │ │ Service  │ │ Service  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
├─────────────────────────────────────────────────────────────────┤
│                      Pipeline Layer                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ Multiverse │ │ Emotion    │ │ Explainable│ │ Social     │   │
│  │ Generator  │ │ Rhythm     │ │ AI         │ │ Pack       │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │
│  │ Gesture    │ │ Text       │ │ Color      │ │ Keyframes  │   │
│  │ Control    │ │ Effects    │ │ Grading    │ │ Extractor  │   │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      Core Layer                                  │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────────┐ │
│  │ System Check │ │ Logging      │ │ Config & Constants       │ │
│  └──────────────┘ └──────────────┘ └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Session Manager

Manages Streamlit session state with type-safe access and automatic initialization.

```python
@dataclass
class SessionState:
    video_path: Optional[str] = None
    video_name: Optional[str] = None
    metadata: Optional[Dict] = None
    analysis: Optional[Dict] = None
    keyframes: Optional[List[Dict]] = None
    emotion: Optional[Dict] = None
    social: Optional[Dict] = None
    multiverse: Optional[Dict] = None
    gesture_enabled: bool = False

class SessionManager:
    def initialize() -> None
    def get(key: str, default: Any = None) -> Any
    def set(key: str, value: Any) -> None
    def clear_analysis() -> None
    def reset() -> None
```

### 2. Pipeline Runner

Orchestrates video processing with progress tracking and error handling.

```python
@dataclass
class PipelineResult:
    success: bool
    data: Optional[Any]
    error: Optional[str]
    duration_ms: float

class PipelineRunner:
    def run_with_progress(
        task: Callable,
        video_path: str,
        progress_callback: Callable[[float, str], None]
    ) -> PipelineResult
    
    def validate_video(path: str) -> Tuple[bool, str]
    def ensure_output_dir(video_name: str) -> Path
```

### 3. Video Capture Context Manager

Ensures proper resource cleanup on Windows.

```python
class SafeVideoCapture:
    def __init__(self, path: str)
    def __enter__(self) -> cv2.VideoCapture
    def __exit__(self, exc_type, exc_val, exc_tb) -> None
    
    # Properties
    fps: float
    frame_count: int
    width: int
    height: int
    duration: float
```

### 4. File Operations Utility

Windows-safe file operations with retry logic.

```python
class FileOps:
    @staticmethod
    def safe_write(path: Path, data: bytes, max_retries: int = 3) -> bool
    
    @staticmethod
    def sanitize_filename(name: str) -> str
    
    @staticmethod
    def ensure_directory(path: Path) -> Path
    
    @staticmethod
    def get_temp_path(prefix: str = "cinemasense_") -> Path
```

### 5. Multiverse Generator Interface

```python
@dataclass
class StylePreview:
    style_name: str
    description: str
    previews: List[Dict[str, Any]]  # {path, timestamp, position}

class MultiverseGenerator:
    STYLES: Dict[str, MultiverseStyle]
    
    def generate_preview(
        video_path: str,
        style_name: str,
        output_dir: Path
    ) -> StylePreview
    
    def apply_style_to_frame(
        frame: np.ndarray,
        style_name: str
    ) -> np.ndarray
    
    def get_available_styles() -> List[Dict[str, str]]
```

### 6. Emotion Rhythm Analyzer Interface

```python
@dataclass
class EmotionFrame:
    timestamp: float
    frame_index: int
    brightness: float
    saturation: float
    motion_intensity: float
    color_temperature: float
    contrast: float
    emotion_score: float
    dominant_emotion: str

@dataclass
class EmotionRhythmResult:
    timeline: List[EmotionFrame]
    overall_score: float
    emotion_distribution: Dict[str, float]
    peak_moments: List[Dict]
    rhythm_pattern: str
    heatmap_data: np.ndarray
    confidence: float

class EmotionRhythmAnalyzer:
    def analyze(video_path: str, sample_rate: int = 5) -> EmotionRhythmResult
    def analyze_frame(frame: np.ndarray, prev_frame: Optional[np.ndarray]) -> Dict
    def classify_emotion(features: Dict) -> Tuple[str, float]
```

### 7. Explainable AI Cut Detector Interface

```python
@dataclass
class CutExplanation:
    frame_index: int
    timestamp: float
    confidence: float
    primary_reason: str
    secondary_reasons: List[str]
    visual_change_score: float
    color_change_score: float
    motion_discontinuity: float
    cut_type: str  # hard_cut, dissolve, fade_to_black, fade_to_white

@dataclass
class ExplainableAnalysis:
    cuts: List[CutExplanation]
    total_cuts: int
    avg_confidence: float
    cut_type_distribution: Dict[str, int]
    explanation_summary: str

class ExplainableCutDetector:
    def detect(
        video_path: str,
        sample_rate: int = 2,
        threshold: float = 0.55
    ) -> ExplainableAnalysis
    
    def analyze_cut_reason(
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        hist_diff: float,
        motion_diff: float
    ) -> Tuple[str, List[str], str]
```

### 8. Social Pack Generator Interface

```python
@dataclass
class SocialPackResult:
    thumbnail_path: str
    title_suggestions: List[str]
    hashtags: List[str]
    caption: str
    platform_optimized: Dict[str, Dict]

class SocialPackGenerator:
    PLATFORM_SPECS: Dict[str, Dict]
    
    def generate(
        video_path: str,
        output_dir: Path,
        metadata: Dict,
        emotion_analysis: Optional[Dict] = None,
        platforms: List[str] = None
    ) -> SocialPackResult
    
    def extract_best_thumbnail(video_path: str) -> np.ndarray
    def score_thumbnail_frame(frame: np.ndarray) -> float
    def generate_titles(metadata: Dict, emotion: Optional[Dict]) -> List[str]
    def generate_hashtags(metadata: Dict, platform: str) -> List[str]
```

## Data Models

### Video Metadata

```python
@dataclass
class VideoMetadata:
    path: str
    name: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_s: float
    format: str
    file_size_bytes: int
```

### Analysis Result

```python
@dataclass
class AnalysisResult:
    video_name: str
    timestamp: str
    metadata: VideoMetadata
    cuts: Optional[ExplainableAnalysis]
    emotion: Optional[EmotionRhythmResult]
    keyframes: Optional[List[Dict]]
    social: Optional[SocialPackResult]
```

### Report Structure

```python
@dataclass
class Report:
    generated_at: str
    video_name: str
    metadata: Dict
    analysis: Dict
    emotion: Dict
    keyframes_count: int
    social: Dict
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Session State Initialization Completeness

*For any* application startup, after SessionManager.initialize() is called, all required session state keys SHALL exist with valid default values.

**Validates: Requirements 1.1**

### Property 2: Session State Reset on Upload

*For any* video upload operation, the session state SHALL have analysis-related keys reset to None while preserving the new video path and metadata.

**Validates: Requirements 1.2**

### Property 3: Widget Key Uniqueness

*For any* set of Streamlit widget keys in the application, no two widgets SHALL have the same key string.

**Validates: Requirements 1.3**

### Property 4: Video Capture Resource Cleanup

*For any* video processing operation using SafeVideoCapture, after the context manager exits (normally or via exception), the video capture resource SHALL be released.

**Validates: Requirements 2.2**

### Property 5: Pipeline Result Structure

*For any* pipeline operation, the returned PipelineResult SHALL have success=True with non-null data, OR success=False with non-null error message.

**Validates: Requirements 2.4**

### Property 6: Path Sanitization for Windows

*For any* input filename string, the sanitized output SHALL contain only characters valid for Windows filenames (no <>:"/\|?*).

**Validates: Requirements 3.5**

### Property 7: File Write Retry Logic

*For any* file write operation that encounters a locking error, the system SHALL retry up to max_retries times before failing.

**Validates: Requirements 3.2**

### Property 8: Multiverse Style Application

*For any* valid frame and style name, applying the style transformation SHALL produce an output frame with different pixel values than the input.

**Validates: Requirements 4.1, 4.4**

### Property 9: Multiverse Preview Positions

*For any* video, multiverse preview generation SHALL produce exactly 3 preview frames at positions 0.25, 0.50, and 0.75 of the video duration.

**Validates: Requirements 4.2**

### Property 10: Emotion Analysis Output Completeness

*For any* video analysis, the EmotionRhythmResult SHALL contain: a non-empty timeline with valid timestamps, emotion_distribution with values summing to 1.0, a valid rhythm_pattern from the defined set, and heatmap_data with correct dimensions.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.6**

### Property 11: Emotion Classification Validity

*For any* frame features, emotion classification SHALL return one of the valid emotion categories (Joy, Tension, Calm, Melancholy, Energy, Mystery) with confidence in range [0, 1].

**Validates: Requirements 5.2**

### Property 12: Cut Explanation Completeness

*For any* detected cut, the CutExplanation SHALL have: non-empty primary_reason, confidence in range [0, 1], cut_type from valid set, and secondary_reasons as a list.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4**

### Property 13: Social Pack Platform Compliance

*For any* platform in the social pack generation, the generated thumbnail SHALL have the exact dimensions specified for that platform, and hashtag count SHALL not exceed the platform limit.

**Validates: Requirements 7.2, 7.4**

### Property 14: Title Generation Count

*For any* social pack generation, exactly 5 title suggestions SHALL be returned, each being a non-empty string.

**Validates: Requirements 7.3**

### Property 15: Gesture Classification Validity

*For any* valid hand landmarks input, gesture classification SHALL return a GestureType from the defined enum.

**Validates: Requirements 8.2**

### Property 16: Gesture Cooldown Enforcement

*For any* sequence of identical gestures detected within the cooldown period, only the first gesture SHALL trigger an action.

**Validates: Requirements 8.3**

### Property 17: Color Grading Transformation

*For any* frame and color grading preset, applying the preset SHALL produce an output frame with measurably different color characteristics.

**Validates: Requirements 10.2, 10.3**

### Property 18: Dependency Check Completeness

*For any* system check execution, the result SHALL include status for all critical dependencies (Python, FFmpeg, OpenCV, MediaPipe, MoviePy).

**Validates: Requirements 13.1, 13.3**

### Property 19: Report Structure Validity

*For any* generated report, the JSON SHALL be valid and contain: generated_at timestamp, video_name, metadata object, and all analysis sections that were performed.

**Validates: Requirements 14.1, 14.2, 14.5**

### Property 20: Error Logging Completeness

*For any* caught exception in the system, the error SHALL be logged with timestamp, error type, message, and stack trace.

**Validates: Requirements 12.1, 12.3**

## Error Handling

### Error Categories

1. **User Errors**: Invalid input, unsupported formats
   - Display friendly message with guidance
   - Log at INFO level

2. **Processing Errors**: Frame read failures, analysis failures
   - Attempt recovery (skip frame, retry)
   - Display warning if partial results
   - Log at WARNING level

3. **System Errors**: Missing dependencies, file system issues
   - Display error with installation/fix instructions
   - Log at ERROR level with stack trace

4. **Critical Errors**: Unrecoverable failures
   - Display error and suggest restart
   - Log at CRITICAL level

### Error Response Pattern

```python
def safe_operation(func):
    """Decorator for safe operation execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return PipelineResult(
                success=True,
                data=func(*args, **kwargs),
                error=None,
                duration_ms=elapsed
            )
        except UserError as e:
            logger.info(f"User error: {e}")
            return PipelineResult(success=False, data=None, error=str(e), duration_ms=elapsed)
        except ProcessingError as e:
            logger.warning(f"Processing error: {e}", exc_info=True)
            return PipelineResult(success=False, data=None, error=str(e), duration_ms=elapsed)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return PipelineResult(success=False, data=None, error=f"Unexpected error: {e}", duration_ms=elapsed)
    return wrapper
```

## Testing Strategy

### Unit Tests

Unit tests verify specific examples and edge cases:

- Session state initialization with various configurations
- File sanitization with special characters
- Video metadata extraction from sample files
- Individual style transformations
- Emotion classification edge cases
- Cut detection threshold boundaries

### Property-Based Tests

Property-based tests verify universal properties across generated inputs using **Hypothesis** (Python PBT library):

- Session state invariants across operations
- File path sanitization produces valid Windows paths
- Pipeline results always have consistent structure
- Style transformations always modify frames
- Emotion scores always in valid ranges
- Cut confidence always in [0, 1]
- Social pack outputs meet platform constraints

**Configuration:**
- Minimum 100 iterations per property test
- Each test tagged with: **Feature: cinemasense-stabilization, Property {N}: {description}**

### Integration Tests

- Full pipeline execution with sample videos
- Multi-page navigation state preservation
- Report generation with all analysis types
- Gesture control initialization and detection

### Test File Organization

```
tests/
├── unit/
│   ├── test_session_manager.py
│   ├── test_file_ops.py
│   ├── test_video_capture.py
│   └── test_emotion_classifier.py
├── property/
│   ├── test_session_properties.py
│   ├── test_pipeline_properties.py
│   ├── test_multiverse_properties.py
│   ├── test_emotion_properties.py
│   ├── test_social_properties.py
│   └── test_report_properties.py
└── integration/
    ├── test_full_pipeline.py
    └── test_ui_navigation.py
```
