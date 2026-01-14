"""
Multiverse Video Generation - Create style variants of videos

Provides functionality to generate multiple style variants of videos
including Romantic, Thriller, Viral, Anime, Cinematic, and Noir styles.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

from ..core.video_capture import SafeVideoCapture, VideoOpenError
from ..core.file_ops import FileOps

logger = logging.getLogger("cinemasense.pipeline.multiverse")


class MultiverseError(Exception):
    """Base exception for multiverse generation errors."""
    pass


class InvalidStyleError(MultiverseError):
    """Raised when an invalid style name is provided."""
    pass


class PreviewGenerationError(MultiverseError):
    """Raised when preview generation fails."""
    pass


@dataclass
class MultiverseStyle:
    """Style configuration for multiverse generation"""
    name: str
    description: str
    color_transform: Callable
    brightness_adjust: float
    contrast_adjust: float
    saturation_adjust: float
    vignette_strength: float
    grain_amount: float
    blur_amount: float
    special_effect: Optional[str]


@dataclass
class StylePreview:
    """Result of style preview generation"""
    style_name: str
    description: str
    previews: List[Dict[str, any]]  # {path, timestamp, position}


def apply_warm_filter(img: np.ndarray) -> np.ndarray:
    """Apply warm color filter"""
    result = img.copy().astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)  # Red
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.05, 0, 255)  # Green
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.9, 0, 255)  # Blue
    return result.astype(np.uint8)


def apply_teal_orange(img: np.ndarray) -> np.ndarray:
    """Apply teal-orange color grading"""
    result = img.copy().astype(np.float32)
    
    # Split channels
    b, g, r = cv2.split(result)
    
    # Push shadows toward teal
    shadow_mask = (0.299 * r + 0.587 * g + 0.114 * b) < 128
    b[shadow_mask] = np.clip(b[shadow_mask] * 1.2, 0, 255)
    g[shadow_mask] = np.clip(g[shadow_mask] * 1.1, 0, 255)
    
    # Push highlights toward orange
    highlight_mask = ~shadow_mask
    r[highlight_mask] = np.clip(r[highlight_mask] * 1.15, 0, 255)
    g[highlight_mask] = np.clip(g[highlight_mask] * 0.95, 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)


def apply_vibrant_filter(img: np.ndarray) -> np.ndarray:
    """Apply vibrant social media filter"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Boost saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)  # Boost value
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_anime_filter(img: np.ndarray) -> np.ndarray:
    """Apply anime/cel-shaded effect"""
    # Reduce colors using bilateral filter
    filtered = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Quantize colors
    div = 64
    quantized = (filtered // div) * div + div // 2
    
    # Edge detection for outlines
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine
    result = cv2.bitwise_and(quantized, edges)
    return result


def apply_cinematic_lut(img: np.ndarray) -> np.ndarray:
    """Apply cinematic color grading"""
    result = img.copy().astype(np.float32)
    
    # Lift shadows slightly
    result = np.clip(result + 10, 0, 255)
    
    # Crush blacks
    result = np.where(result < 20, result * 0.5, result)
    
    # Add slight blue to shadows
    shadow_mask = np.mean(result, axis=2) < 80
    result[:, :, 0][shadow_mask] = np.clip(result[:, :, 0][shadow_mask] + 15, 0, 255)
    
    return result.astype(np.uint8)


def apply_noir_filter(img: np.ndarray) -> np.ndarray:
    """Apply noir black and white filter"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast curve
    gray = np.clip((gray.astype(np.float32) - 128) * 1.5 + 128, 0, 255).astype(np.uint8)
    
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_cyberpunk_filter(img: np.ndarray) -> np.ndarray:
    """Apply cyberpunk neon filter"""
    result = img.copy().astype(np.float32)
    
    # Boost blues and magentas
    result[:, :, 0] = np.clip(result[:, :, 0] * 1.3, 0, 255)  # Blue
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.2, 0, 255)  # Red (for magenta)
    result[:, :, 1] = np.clip(result[:, :, 1] * 0.8, 0, 255)  # Reduce green
    
    return result.astype(np.uint8)


def apply_vignette(img: np.ndarray, strength: float = 0.5) -> np.ndarray:
    """Apply vignette effect"""
    rows, cols = img.shape[:2]
    
    # Create gradient mask
    X = np.arange(0, cols)
    Y = np.arange(0, rows)
    X, Y = np.meshgrid(X, Y)
    
    center_x, center_y = cols // 2, rows // 2
    mask = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    mask = mask / mask.max()
    mask = 1 - mask * strength
    
    # Apply mask
    result = img.copy().astype(np.float32)
    for i in range(3):
        result[:, :, i] = result[:, :, i] * mask
    
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_film_grain(img: np.ndarray, amount: float = 0.05) -> np.ndarray:
    """Apply film grain effect"""
    noise = np.random.normal(0, amount * 255, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_soft_glow(img: np.ndarray) -> np.ndarray:
    """Apply soft glow effect"""
    blurred = cv2.GaussianBlur(img, (21, 21), 0)
    result = cv2.addWeighted(img, 0.7, blurred, 0.3, 0)
    return result


def apply_letterbox(img: np.ndarray, ratio: float = 2.35) -> np.ndarray:
    """Apply cinematic letterboxing"""
    h, w = img.shape[:2]
    target_h = int(w / ratio)
    
    if target_h < h:
        bar_height = (h - target_h) // 2
        result = img.copy()
        result[:bar_height, :] = 0
        result[-bar_height:, :] = 0
        return result
    
    return img


# Style definitions
MULTIVERSE_STYLES: Dict[str, MultiverseStyle] = {
    "romantic": MultiverseStyle(
        name="Romantic",
        description="Soft, warm tones with dreamy atmosphere",
        color_transform=apply_warm_filter,
        brightness_adjust=1.1,
        contrast_adjust=0.9,
        saturation_adjust=0.85,
        vignette_strength=0.3,
        grain_amount=0.02,
        blur_amount=0.5,
        special_effect="soft_glow"
    ),
    "thriller": MultiverseStyle(
        name="Thriller",
        description="High contrast, desaturated with teal-orange grading",
        color_transform=apply_teal_orange,
        brightness_adjust=0.9,
        contrast_adjust=1.3,
        saturation_adjust=0.7,
        vignette_strength=0.5,
        grain_amount=0.05,
        blur_amount=0,
        special_effect="sharpen"
    ),
    "viral": MultiverseStyle(
        name="Viral",
        description="Punchy, vibrant colors optimized for social media",
        color_transform=apply_vibrant_filter,
        brightness_adjust=1.15,
        contrast_adjust=1.2,
        saturation_adjust=1.4,
        vignette_strength=0.1,
        grain_amount=0,
        blur_amount=0,
        special_effect="clarity"
    ),
    "anime": MultiverseStyle(
        name="Anime",
        description="Cel-shaded look with bold outlines and flat colors",
        color_transform=apply_anime_filter,
        brightness_adjust=1.05,
        contrast_adjust=1.1,
        saturation_adjust=1.2,
        vignette_strength=0,
        grain_amount=0,
        blur_amount=0,
        special_effect="edge_detect"
    ),
    "cinematic": MultiverseStyle(
        name="Cinematic",
        description="Film-like look with letterboxing and color grading",
        color_transform=apply_cinematic_lut,
        brightness_adjust=0.95,
        contrast_adjust=1.15,
        saturation_adjust=0.9,
        vignette_strength=0.4,
        grain_amount=0.03,
        blur_amount=0,
        special_effect="letterbox"
    ),
    "noir": MultiverseStyle(
        name="Noir",
        description="Classic black and white with high contrast",
        color_transform=apply_noir_filter,
        brightness_adjust=0.9,
        contrast_adjust=1.4,
        saturation_adjust=0,
        vignette_strength=0.6,
        grain_amount=0.08,
        blur_amount=0,
        special_effect="film_grain"
    ),
    "cyberpunk": MultiverseStyle(
        name="Cyberpunk",
        description="Neon-lit futuristic aesthetic with bold colors",
        color_transform=apply_cyberpunk_filter,
        brightness_adjust=1.0,
        contrast_adjust=1.25,
        saturation_adjust=1.3,
        vignette_strength=0.35,
        grain_amount=0.02,
        blur_amount=0,
        special_effect="neon_glow"
    )
}

# Valid style names for validation
VALID_STYLE_NAMES = frozenset(MULTIVERSE_STYLES.keys())

# Preview positions as specified in requirements (25%, 50%, 75%)
PREVIEW_POSITIONS = [0.25, 0.50, 0.75]


def validate_style_name(style_name: str) -> str:
    """
    Validate and normalize style name.
    
    Args:
        style_name: Style name to validate
        
    Returns:
        Normalized style name (lowercase)
        
    Raises:
        InvalidStyleError: If style name is not valid
        
    Requirements: 4.1
    """
    if not style_name:
        raise InvalidStyleError("Style name cannot be empty")
    
    normalized = style_name.lower().strip()
    
    if normalized not in VALID_STYLE_NAMES:
        available = ", ".join(sorted(VALID_STYLE_NAMES))
        raise InvalidStyleError(
            f"Unknown style: '{style_name}'. "
            f"Available styles: {available}"
        )
    
    return normalized


def process_frame_with_style(frame: np.ndarray, style: MultiverseStyle) -> np.ndarray:
    """
    Process a single frame with the given style.
    
    Args:
        frame: Input BGR frame
        style: Style configuration to apply
        
    Returns:
        Processed frame with style applied
        
    Requirements: 4.1, 4.4
    """
    if frame is None or frame.size == 0:
        logger.warning("Received empty frame for style processing")
        return frame
    
    try:
        result = frame.copy()
        
        # Apply color transform
        result = style.color_transform(result)
        
        # Adjust brightness
        if style.brightness_adjust != 1.0:
            result = cv2.convertScaleAbs(result, alpha=style.brightness_adjust, beta=0)
        
        # Adjust contrast
        if style.contrast_adjust != 1.0:
            result = cv2.convertScaleAbs(
                result, 
                alpha=style.contrast_adjust, 
                beta=128 * (1 - style.contrast_adjust)
            )
        
        # Adjust saturation
        if style.saturation_adjust != 1.0:
            hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * style.saturation_adjust, 0, 255)
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Apply vignette
        if style.vignette_strength > 0:
            result = apply_vignette(result, style.vignette_strength)
        
        # Apply grain
        if style.grain_amount > 0:
            result = apply_film_grain(result, style.grain_amount)
        
        # Apply blur
        if style.blur_amount > 0:
            kernel_size = int(style.blur_amount * 10) * 2 + 1
            result = cv2.GaussianBlur(result, (kernel_size, kernel_size), 0)
        
        # Apply special effects
        if style.special_effect == "soft_glow":
            result = apply_soft_glow(result)
        elif style.special_effect == "letterbox":
            result = apply_letterbox(result)
        elif style.special_effect == "sharpen":
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            result = cv2.filter2D(result, -1, kernel)
        elif style.special_effect == "clarity":
            result = cv2.detailEnhance(result, sigma_s=10, sigma_r=0.15)
        elif style.special_effect == "film_grain":
            # Additional film grain for noir style
            result = apply_film_grain(result, 0.03)
        elif style.special_effect == "neon_glow":
            # Neon glow effect for cyberpunk
            blurred = cv2.GaussianBlur(result, (15, 15), 0)
            result = cv2.addWeighted(result, 0.8, blurred, 0.4, 10)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing frame with style '{style.name}': {e}")
        # Return original frame on error
        return frame


def generate_multiverse_preview(
    video_path: str, 
    style_name: str, 
    output_dir: Path
) -> StylePreview:
    """
    Generate preview frames for a multiverse style.
    
    Uses SafeVideoCapture for proper resource cleanup and FileOps
    for Windows-safe file operations.
    
    Args:
        video_path: Path to input video file
        style_name: Name of style to apply (case-insensitive)
        output_dir: Directory to save preview images
        
    Returns:
        StylePreview with generated preview information
        
    Raises:
        InvalidStyleError: If style name is not valid
        PreviewGenerationError: If preview generation fails
        VideoOpenError: If video cannot be opened
        
    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Validate style name
    normalized_style = validate_style_name(style_name)
    style = MULTIVERSE_STYLES[normalized_style]
    
    logger.info(f"Generating {style.name} preview for: {video_path}")
    
    # Ensure output directory exists using FileOps
    output_path = FileOps.ensure_directory(Path(output_dir))
    logger.debug(f"Output directory ensured: {output_path}")
    
    previews = []
    
    try:
        # Use SafeVideoCapture for proper resource cleanup
        with SafeVideoCapture(video_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            if total_frames <= 0:
                raise PreviewGenerationError(
                    f"Video has no frames or frame count unavailable: {video_path}"
                )
            
            logger.debug(
                f"Video info: {total_frames} frames, {fps:.2f} fps, "
                f"duration: {total_frames/fps:.2f}s"
            )
            
            # Extract preview frames at 25%, 50%, 75% positions
            for pos in PREVIEW_POSITIONS:
                frame_idx = int(total_frames * pos)
                
                # Ensure frame index is valid
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning(
                        f"Could not read frame at position {pos:.0%} "
                        f"(frame {frame_idx}), skipping"
                    )
                    continue
                
                # Process frame with style
                processed = process_frame_with_style(frame, style)
                
                # Generate safe filename
                safe_style_name = FileOps.sanitize_filename(normalized_style)
                preview_filename = f"preview_{safe_style_name}_{int(pos * 100)}.jpg"
                preview_path = output_path / preview_filename
                
                # Save preview image
                success = cv2.imwrite(str(preview_path), processed)
                
                if not success:
                    logger.warning(f"Failed to save preview: {preview_path}")
                    continue
                
                timestamp = frame_idx / fps
                previews.append({
                    "position": pos,
                    "timestamp": timestamp,
                    "path": str(preview_path),
                    "frame_index": frame_idx
                })
                
                logger.debug(
                    f"Generated preview at {pos:.0%} "
                    f"(frame {frame_idx}, {timestamp:.2f}s)"
                )
        
        if not previews:
            raise PreviewGenerationError(
                f"Failed to generate any preview frames for style '{style_name}'"
            )
        
        logger.info(
            f"Successfully generated {len(previews)} previews "
            f"for style '{style.name}'"
        )
        
        return StylePreview(
            style_name=normalized_style,
            description=style.description,
            previews=previews
        )
        
    except VideoOpenError:
        # Re-raise video open errors as-is
        raise
    except PreviewGenerationError:
        # Re-raise preview generation errors as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating preview: {e}", exc_info=True)
        raise PreviewGenerationError(
            f"Failed to generate preview for style '{style_name}': {e}"
        )


def apply_style_to_frame(frame: np.ndarray, style_name: str) -> np.ndarray:
    """
    Apply a style transformation to a single frame.
    
    Args:
        frame: Input BGR frame
        style_name: Name of style to apply (case-insensitive)
        
    Returns:
        Processed frame with style applied
        
    Raises:
        InvalidStyleError: If style name is not valid
        
    Requirements: 4.1, 4.4
    """
    normalized_style = validate_style_name(style_name)
    style = MULTIVERSE_STYLES[normalized_style]
    return process_frame_with_style(frame, style)


def get_available_styles() -> List[Dict[str, str]]:
    """
    Get list of available multiverse styles.
    
    Returns:
        List of style dictionaries with id, name, and description
        
    Requirements: 4.1
    """
    return [
        {
            "id": key,
            "name": style.name,
            "description": style.description
        }
        for key, style in MULTIVERSE_STYLES.items()
    ]


def get_style_info(style_name: str) -> Optional[Dict[str, str]]:
    """
    Get information about a specific style.
    
    Args:
        style_name: Name of style to look up (case-insensitive)
        
    Returns:
        Style info dictionary or None if not found
    """
    try:
        normalized = validate_style_name(style_name)
        style = MULTIVERSE_STYLES[normalized]
        return {
            "id": normalized,
            "name": style.name,
            "description": style.description
        }
    except InvalidStyleError:
        return None
