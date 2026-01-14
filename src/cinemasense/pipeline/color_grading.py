"""
Cinema Color Grading Presets - Professional color grading for video

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger("cinemasense.pipeline.color_grading")


class InvalidPresetError(Exception):
    """Raised when an invalid preset name is provided."""
    pass


class InvalidFrameError(Exception):
    """Raised when an invalid frame is provided for processing."""
    pass


@dataclass
class ColorGradingPreset:
    """Color grading preset configuration"""
    name: str
    description: str
    shadows_rgb: Tuple[int, int, int]
    midtones_rgb: Tuple[int, int, int]
    highlights_rgb: Tuple[int, int, int]
    contrast: float
    saturation: float
    temperature: float  # -100 to 100 (cool to warm)
    tint: float  # -100 to 100 (green to magenta)
    lift: float
    gamma: float
    gain: float


@dataclass
class ColorGradingResult:
    """Result of color grading operation"""
    success: bool
    graded_frame: Optional[np.ndarray]
    preset_name: str
    error: Optional[str] = None


@dataclass
class PresetSuggestion:
    """Suggested preset based on frame analysis"""
    preset_name: str
    confidence: float
    reason: str


# Professional cinema presets (Requirement 10.1)
CINEMA_PRESETS: Dict[str, ColorGradingPreset] = {
    "blockbuster": ColorGradingPreset(
        name="Blockbuster",
        description="High-contrast teal and orange look popular in action films",
        shadows_rgb=(0, 30, 50),
        midtones_rgb=(0, 0, 0),
        highlights_rgb=(40, 20, 0),
        contrast=1.2,
        saturation=0.9,
        temperature=10,
        tint=0,
        lift=-0.05,
        gamma=0.95,
        gain=1.1
    ),
    "indie_film": ColorGradingPreset(
        name="Indie Film",
        description="Desaturated, slightly lifted blacks with warm highlights",
        shadows_rgb=(10, 10, 15),
        midtones_rgb=(5, 0, -5),
        highlights_rgb=(20, 15, 5),
        contrast=0.95,
        saturation=0.75,
        temperature=15,
        tint=5,
        lift=0.05,
        gamma=1.05,
        gain=0.95
    ),
    "horror": ColorGradingPreset(
        name="Horror",
        description="Cold, desaturated with crushed blacks and green tint",
        shadows_rgb=(0, 10, 5),
        midtones_rgb=(-10, 5, 0),
        highlights_rgb=(0, -5, 10),
        contrast=1.3,
        saturation=0.6,
        temperature=-20,
        tint=-10,
        lift=-0.1,
        gamma=0.9,
        gain=0.9
    ),
    "romance": ColorGradingPreset(
        name="Romance",
        description="Soft, warm tones with lifted shadows and gentle contrast",
        shadows_rgb=(15, 10, 5),
        midtones_rgb=(10, 5, 0),
        highlights_rgb=(30, 20, 10),
        contrast=0.85,
        saturation=0.85,
        temperature=25,
        tint=5,
        lift=0.1,
        gamma=1.1,
        gain=1.0
    ),
    "sci_fi": ColorGradingPreset(
        name="Sci-Fi",
        description="Cool blue tones with high contrast and cyan highlights",
        shadows_rgb=(0, 5, 20),
        midtones_rgb=(-5, 0, 10),
        highlights_rgb=(0, 20, 40),
        contrast=1.25,
        saturation=0.8,
        temperature=-30,
        tint=0,
        lift=-0.05,
        gamma=0.95,
        gain=1.05
    ),
    "vintage": ColorGradingPreset(
        name="Vintage",
        description="Faded look with warm shadows and reduced contrast",
        shadows_rgb=(20, 15, 10),
        midtones_rgb=(10, 5, 0),
        highlights_rgb=(25, 20, 15),
        contrast=0.8,
        saturation=0.7,
        temperature=20,
        tint=10,
        lift=0.15,
        gamma=1.1,
        gain=0.9
    ),
    "documentary": ColorGradingPreset(
        name="Documentary",
        description="Natural, slightly desaturated with neutral tones",
        shadows_rgb=(5, 5, 5),
        midtones_rgb=(0, 0, 0),
        highlights_rgb=(5, 5, 5),
        contrast=1.05,
        saturation=0.85,
        temperature=0,
        tint=0,
        lift=0,
        gamma=1.0,
        gain=1.0
    ),
    "neon_nights": ColorGradingPreset(
        name="Neon Nights",
        description="Vibrant neon colors with deep blacks and magenta/cyan split",
        shadows_rgb=(10, 0, 20),
        midtones_rgb=(0, 10, 20),
        highlights_rgb=(40, 10, 30),
        contrast=1.35,
        saturation=1.3,
        temperature=-10,
        tint=15,
        lift=-0.1,
        gamma=0.9,
        gain=1.15
    )
}


def validate_preset_name(preset_name: str) -> bool:
    """
    Validate that a preset name exists in CINEMA_PRESETS.
    
    Args:
        preset_name: Name of the preset to validate
        
    Returns:
        True if preset exists, False otherwise
    """
    return preset_name.lower() in CINEMA_PRESETS


def get_preset(preset_name: str) -> ColorGradingPreset:
    """
    Get a preset by name with validation.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        ColorGradingPreset object
        
    Raises:
        InvalidPresetError: If preset name is not valid
    """
    normalized_name = preset_name.lower()
    if normalized_name not in CINEMA_PRESETS:
        available = ", ".join(CINEMA_PRESETS.keys())
        raise InvalidPresetError(
            f"Invalid preset name: '{preset_name}'. "
            f"Available presets: {available}"
        )
    return CINEMA_PRESETS[normalized_name]


def validate_frame(frame: np.ndarray) -> bool:
    """
    Validate that a frame is suitable for color grading.
    
    Args:
        frame: Input frame to validate
        
    Returns:
        True if frame is valid, False otherwise
    """
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0:
        return False
    if len(frame.shape) != 3:
        return False
    if frame.shape[2] != 3:
        return False
    return True


def ensure_valid_pixel_range(frame: np.ndarray) -> np.ndarray:
    """
    Ensure all pixel values are within valid range [0, 255].
    
    Args:
        frame: Input frame (can be float or int)
        
    Returns:
        Frame with pixel values clipped to [0, 255] as uint8
    """
    return np.clip(frame, 0, 255).astype(np.uint8)


def apply_color_grading(frame: np.ndarray, preset: ColorGradingPreset) -> np.ndarray:
    """
    Apply color grading preset to a frame.
    
    Applies lift, gamma, and gain adjustments along with color shifts
    for shadows, midtones, and highlights. (Requirements 10.2, 10.3)
    
    Args:
        frame: Input BGR frame
        preset: ColorGradingPreset to apply
        
    Returns:
        Graded frame with valid pixel range [0, 255]
        
    Raises:
        InvalidFrameError: If frame is invalid
    """
    if not validate_frame(frame):
        raise InvalidFrameError("Invalid frame provided for color grading")
    
    result = frame.copy().astype(np.float32)
    
    # Calculate luminance for shadow/midtone/highlight masks
    luminance = 0.299 * result[:, :, 2] + 0.587 * result[:, :, 1] + 0.114 * result[:, :, 0]
    
    # Create masks
    shadow_mask = luminance < 85
    highlight_mask = luminance > 170
    midtone_mask = ~shadow_mask & ~highlight_mask
    
    # Apply color shifts to shadows, midtones, highlights (Requirement 10.2)
    for i, (s, m, h) in enumerate(zip(preset.shadows_rgb, preset.midtones_rgb, preset.highlights_rgb)):
        result[:, :, 2-i][shadow_mask] += s
        result[:, :, 2-i][midtone_mask] += m
        result[:, :, 2-i][highlight_mask] += h
    
    # Apply lift (shadows) - Requirement 10.3
    result = result + preset.lift * 255
    
    # Clip before gamma to avoid negative values in power operation
    result = np.clip(result, 0, 255)
    
    # Apply gamma (midtones) - Requirement 10.3
    # Avoid division by zero and ensure valid gamma
    gamma = max(preset.gamma, 0.01)
    result = 255 * np.power(result / 255, 1 / gamma)
    
    # Apply gain (highlights) - Requirement 10.3
    result = result * preset.gain
    
    # Apply contrast (Requirement 10.2)
    result = (result - 128) * preset.contrast + 128
    
    # Clip before color space conversion
    result = np.clip(result, 0, 255)
    
    # Apply saturation (Requirement 10.2)
    hsv = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * preset.saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    
    # Apply temperature (Requirement 10.2)
    if preset.temperature != 0:
        temp_shift = preset.temperature / 100 * 30
        result[:, :, 2] += temp_shift  # Red
        result[:, :, 0] -= temp_shift  # Blue
    
    # Apply tint (Requirement 10.2)
    if preset.tint != 0:
        tint_shift = preset.tint / 100 * 20
        result[:, :, 1] -= tint_shift  # Green (negative = more magenta)
    
    # Ensure valid pixel range [0, 255]
    return ensure_valid_pixel_range(result)


def apply_color_grading_safe(
    frame: np.ndarray, 
    preset_name: str
) -> ColorGradingResult:
    """
    Safely apply color grading with validation and error handling.
    
    Args:
        frame: Input BGR frame
        preset_name: Name of the preset to apply
        
    Returns:
        ColorGradingResult with success status and graded frame or error
    """
    try:
        # Validate preset name
        if not validate_preset_name(preset_name):
            available = ", ".join(CINEMA_PRESETS.keys())
            return ColorGradingResult(
                success=False,
                graded_frame=None,
                preset_name=preset_name,
                error=f"Invalid preset name: '{preset_name}'. Available: {available}"
            )
        
        # Validate frame
        if not validate_frame(frame):
            return ColorGradingResult(
                success=False,
                graded_frame=None,
                preset_name=preset_name,
                error="Invalid frame: must be a non-empty BGR image (3 channels)"
            )
        
        preset = get_preset(preset_name)
        graded = apply_color_grading(frame, preset)
        
        return ColorGradingResult(
            success=True,
            graded_frame=graded,
            preset_name=preset_name,
            error=None
        )
        
    except Exception as e:
        logger.error(f"Error applying color grading: {e}", exc_info=True)
        return ColorGradingResult(
            success=False,
            graded_frame=None,
            preset_name=preset_name,
            error=str(e)
        )


def create_lut_from_preset(preset: ColorGradingPreset, size: int = 33) -> np.ndarray:
    """Create a 3D LUT from a preset"""
    lut = np.zeros((size, size, size, 3), dtype=np.float32)
    
    for r in range(size):
        for g in range(size):
            for b in range(size):
                # Create a single pixel with this color
                pixel = np.array([[[
                    b * 255 / (size - 1),
                    g * 255 / (size - 1),
                    r * 255 / (size - 1)
                ]]], dtype=np.float32)
                
                # Apply grading
                graded = apply_color_grading(pixel.astype(np.uint8), preset)
                
                lut[r, g, b] = graded[0, 0] / 255.0
    
    return lut


def apply_lut(frame: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 3D LUT to a frame"""
    if not validate_frame(frame):
        raise InvalidFrameError("Invalid frame for LUT application")
    
    size = lut.shape[0]
    
    # Normalize frame to LUT indices
    frame_norm = frame.astype(np.float32) / 255.0 * (size - 1)
    
    # Trilinear interpolation
    r = frame_norm[:, :, 2]
    g = frame_norm[:, :, 1]
    b = frame_norm[:, :, 0]
    
    r0, g0, b0 = np.floor(r).astype(int), np.floor(g).astype(int), np.floor(b).astype(int)
    r1, g1, b1 = np.ceil(r).astype(int), np.ceil(g).astype(int), np.ceil(b).astype(int)
    
    # Clamp indices
    r0, r1 = np.clip(r0, 0, size-1), np.clip(r1, 0, size-1)
    g0, g1 = np.clip(g0, 0, size-1), np.clip(g1, 0, size-1)
    b0, b1 = np.clip(b0, 0, size-1), np.clip(b1, 0, size-1)
    
    # Simple nearest neighbor for performance
    result = lut[r0, g0, b0] * 255
    
    return ensure_valid_pixel_range(result)


def get_preset_preview(preset_name: str, sample_frame: np.ndarray) -> np.ndarray:
    """
    Generate a preview of a preset applied to a sample frame.
    
    Args:
        preset_name: Name of the preset to preview
        sample_frame: Sample frame to apply preset to
        
    Returns:
        Graded frame
        
    Raises:
        InvalidPresetError: If preset name is invalid
        InvalidFrameError: If frame is invalid
    """
    preset = get_preset(preset_name)  # Validates preset name
    return apply_color_grading(sample_frame, preset)


def generate_before_after_preview(
    frame: np.ndarray, 
    preset_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate before/after comparison frames. (Requirement 10.5)
    
    Args:
        frame: Original frame
        preset_name: Name of the preset to apply
        
    Returns:
        Tuple of (original_frame, graded_frame)
        
    Raises:
        InvalidPresetError: If preset name is invalid
        InvalidFrameError: If frame is invalid
    """
    if not validate_frame(frame):
        raise InvalidFrameError("Invalid frame for preview generation")
    
    preset = get_preset(preset_name)
    graded = apply_color_grading(frame, preset)
    
    return frame.copy(), graded


def generate_side_by_side_preview(
    frame: np.ndarray,
    preset_name: str
) -> np.ndarray:
    """
    Generate a side-by-side before/after comparison image. (Requirement 10.5)
    
    Args:
        frame: Original frame
        preset_name: Name of the preset to apply
        
    Returns:
        Combined image with original on left, graded on right
        
    Raises:
        InvalidPresetError: If preset name is invalid
        InvalidFrameError: If frame is invalid
    """
    original, graded = generate_before_after_preview(frame, preset_name)
    
    # Create side-by-side comparison
    combined = np.hstack([original, graded])
    
    return combined


def get_available_presets() -> List[Dict]:
    """
    Get list of available color grading presets. (Requirement 10.1)
    
    Returns:
        List of preset info dictionaries with id, name, and description
    """
    return [
        {
            "id": key,
            "name": preset.name,
            "description": preset.description
        }
        for key, preset in CINEMA_PRESETS.items()
    ]


def analyze_color_palette(frame: np.ndarray) -> Dict:
    """
    Analyze the color palette of a frame. (Requirement 10.4)
    
    Args:
        frame: Input BGR frame
        
    Returns:
        Dictionary with color analysis and suggested preset
        
    Raises:
        InvalidFrameError: If frame is invalid
    """
    if not validate_frame(frame):
        raise InvalidFrameError("Invalid frame for color analysis")
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Extract dominant colors using k-means
    pixels = frame.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    try:
        _, labels, centers = cv2.kmeans(
            pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
    except cv2.error as e:
        logger.warning(f"K-means clustering failed: {e}")
        # Fallback to simple analysis without dominant colors
        centers = np.array([[128, 128, 128]])
        labels = np.zeros(len(pixels), dtype=np.int32)
    
    # Count pixels per cluster
    unique, counts = np.unique(labels, return_counts=True)
    sorted_indices = np.argsort(-counts)
    
    dominant_colors = []
    for idx in sorted_indices:
        if idx < len(centers):
            color = centers[idx].astype(int)
            percentage = counts[idx] / len(labels) * 100
            dominant_colors.append({
                "rgb": (int(color[2]), int(color[1]), int(color[0])),
                "percentage": float(percentage)
            })
    
    # Calculate overall characteristics
    avg_brightness = float(np.mean(hsv[:, :, 2]) / 255)
    avg_saturation = float(np.mean(hsv[:, :, 1]) / 255)
    
    # Temperature estimation from LAB b channel
    temperature = float((np.mean(lab[:, :, 2]) - 128) / 128)  # -1 (cool) to 1 (warm)
    
    # Get preset suggestion with confidence and reason
    suggestion = suggest_preset_with_reason(avg_brightness, avg_saturation, temperature)
    
    return {
        "dominant_colors": dominant_colors,
        "avg_brightness": avg_brightness,
        "avg_saturation": avg_saturation,
        "temperature": temperature,
        "suggested_preset": suggestion.preset_name,
        "suggestion_confidence": suggestion.confidence,
        "suggestion_reason": suggestion.reason
    }


def suggest_preset(brightness: float, saturation: float, temperature: float) -> str:
    """
    Suggest a color grading preset based on frame analysis. (Requirement 10.4)
    
    Args:
        brightness: Average brightness (0-1)
        saturation: Average saturation (0-1)
        temperature: Color temperature (-1 to 1, cool to warm)
        
    Returns:
        Preset name string
    """
    return suggest_preset_with_reason(brightness, saturation, temperature).preset_name


def suggest_preset_with_reason(
    brightness: float, 
    saturation: float, 
    temperature: float
) -> PresetSuggestion:
    """
    Suggest a color grading preset with confidence and reasoning. (Requirement 10.4)
    
    Args:
        brightness: Average brightness (0-1)
        saturation: Average saturation (0-1)
        temperature: Color temperature (-1 to 1, cool to warm)
        
    Returns:
        PresetSuggestion with preset name, confidence, and reason
    """
    # Clamp inputs to valid ranges
    brightness = max(0.0, min(1.0, brightness))
    saturation = max(0.0, min(1.0, saturation))
    temperature = max(-1.0, min(1.0, temperature))
    
    # Decision logic with confidence scoring
    if brightness < 0.3 and saturation < 0.4:
        return PresetSuggestion(
            preset_name="horror",
            confidence=0.85,
            reason="Dark, desaturated footage suits horror aesthetic"
        )
    elif brightness > 0.6 and temperature > 0.2:
        return PresetSuggestion(
            preset_name="romance",
            confidence=0.80,
            reason="Bright, warm footage enhances romantic mood"
        )
    elif saturation > 0.6:
        return PresetSuggestion(
            preset_name="neon_nights",
            confidence=0.75,
            reason="High saturation footage works well with vibrant neon look"
        )
    elif temperature < -0.2:
        return PresetSuggestion(
            preset_name="sci_fi",
            confidence=0.78,
            reason="Cool-toned footage complements sci-fi aesthetic"
        )
    elif brightness > 0.5 and saturation < 0.5:
        return PresetSuggestion(
            preset_name="indie_film",
            confidence=0.70,
            reason="Moderate brightness with low saturation suits indie look"
        )
    elif brightness < 0.4:
        return PresetSuggestion(
            preset_name="vintage",
            confidence=0.65,
            reason="Darker footage benefits from vintage warmth"
        )
    elif saturation < 0.3:
        return PresetSuggestion(
            preset_name="documentary",
            confidence=0.72,
            reason="Low saturation footage suits natural documentary style"
        )
    else:
        return PresetSuggestion(
            preset_name="blockbuster",
            confidence=0.60,
            reason="Balanced footage works well with versatile blockbuster look"
        )
