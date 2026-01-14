"""
Premium Text-Behind-Video Effects
Creative text overlays with cinematic effects
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger("cinemasense.pipeline.text_effects")


@dataclass
class TextStyle:
    """Text styling configuration"""
    font_scale: float
    thickness: int
    color: Tuple[int, int, int]
    shadow_color: Tuple[int, int, int]
    shadow_offset: Tuple[int, int]
    glow_color: Optional[Tuple[int, int, int]]
    glow_radius: int
    outline_color: Optional[Tuple[int, int, int]]
    outline_thickness: int


# Premium text styles - expanded with more creative options
TEXT_STYLES = {
    "cinematic": TextStyle(
        font_scale=2.0,
        thickness=3,
        color=(255, 255, 255),
        shadow_color=(0, 0, 0),
        shadow_offset=(4, 4),
        glow_color=None,
        glow_radius=0,
        outline_color=(0, 0, 0),
        outline_thickness=2
    ),
    "neon": TextStyle(
        font_scale=2.0,
        thickness=2,
        color=(255, 0, 255),
        shadow_color=(100, 0, 100),
        shadow_offset=(2, 2),
        glow_color=(255, 100, 255),
        glow_radius=15,
        outline_color=None,
        outline_thickness=0
    ),
    "minimal": TextStyle(
        font_scale=1.5,
        thickness=2,
        color=(255, 255, 255),
        shadow_color=(50, 50, 50),
        shadow_offset=(2, 2),
        glow_color=None,
        glow_radius=0,
        outline_color=None,
        outline_thickness=0
    ),
    "bold": TextStyle(
        font_scale=2.5,
        thickness=4,
        color=(255, 255, 255),
        shadow_color=(0, 0, 0),
        shadow_offset=(6, 6),
        glow_color=None,
        glow_radius=0,
        outline_color=(0, 0, 0),
        outline_thickness=3
    ),
    "elegant": TextStyle(
        font_scale=1.8,
        thickness=2,
        color=(220, 200, 180),
        shadow_color=(50, 40, 30),
        shadow_offset=(3, 3),
        glow_color=(255, 240, 220),
        glow_radius=8,
        outline_color=None,
        outline_thickness=0
    ),
    "cyberpunk": TextStyle(
        font_scale=2.2,
        thickness=2,
        color=(0, 255, 255),
        shadow_color=(255, 0, 100),
        shadow_offset=(3, -3),
        glow_color=(0, 255, 255),
        glow_radius=20,
        outline_color=(255, 0, 100),
        outline_thickness=1
    ),
    "retro": TextStyle(
        font_scale=2.0,
        thickness=3,
        color=(255, 200, 100),
        shadow_color=(100, 50, 0),
        shadow_offset=(5, 5),
        glow_color=(255, 150, 50),
        glow_radius=10,
        outline_color=(80, 40, 0),
        outline_thickness=2
    ),
    "glitch": TextStyle(
        font_scale=2.3,
        thickness=3,
        color=(255, 255, 255),
        shadow_color=(255, 0, 0),
        shadow_offset=(-3, 0),
        glow_color=(0, 255, 255),
        glow_radius=5,
        outline_color=None,
        outline_thickness=0
    ),
    "holographic": TextStyle(
        font_scale=2.0,
        thickness=2,
        color=(200, 255, 255),
        shadow_color=(100, 200, 255),
        shadow_offset=(2, 2),
        glow_color=(150, 255, 255),
        glow_radius=25,
        outline_color=(255, 255, 255),
        outline_thickness=1
    ),
    "fire": TextStyle(
        font_scale=2.2,
        thickness=3,
        color=(255, 200, 50),
        shadow_color=(255, 100, 0),
        shadow_offset=(0, 4),
        glow_color=(255, 150, 0),
        glow_radius=18,
        outline_color=(255, 50, 0),
        outline_thickness=2
    )
}


def create_text_mask(
    frame_shape: Tuple[int, int],
    text: str,
    position: Tuple[int, int],
    style: TextStyle
) -> np.ndarray:
    """Create a binary mask for text"""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    cv2.putText(
        mask, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        style.font_scale,
        255,
        style.thickness + (style.outline_thickness if style.outline_color else 0)
    )
    
    return mask


def apply_text_behind_subject(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    style_name: str = "cinematic",
    subject_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Apply text behind the main subject in the frame"""
    if style_name not in TEXT_STYLES:
        style_name = "cinematic"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    
    # If no subject mask provided, create one using edge detection
    if subject_mask is None:
        subject_mask = create_subject_mask(frame)
    
    # Create text layer
    text_layer = np.zeros_like(frame)
    
    # Draw shadow
    shadow_pos = (position[0] + style.shadow_offset[0], position[1] + style.shadow_offset[1])
    cv2.putText(
        text_layer, text, shadow_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        style.font_scale,
        style.shadow_color,
        style.thickness
    )
    
    # Draw outline if specified
    if style.outline_color and style.outline_thickness > 0:
        cv2.putText(
            text_layer, text, position,
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            style.outline_color,
            style.thickness + style.outline_thickness
        )
    
    # Draw main text
    cv2.putText(
        text_layer, text, position,
        cv2.FONT_HERSHEY_SIMPLEX,
        style.font_scale,
        style.color,
        style.thickness
    )
    
    # Apply glow if specified
    if style.glow_color and style.glow_radius > 0:
        glow_layer = np.zeros_like(frame)
        cv2.putText(
            glow_layer, text, position,
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            style.glow_color,
            style.thickness + 2
        )
        glow_layer = cv2.GaussianBlur(glow_layer, (style.glow_radius * 2 + 1, style.glow_radius * 2 + 1), 0)
        text_layer = cv2.addWeighted(text_layer, 1, glow_layer, 0.5, 0)
    
    # Create text mask
    text_mask = cv2.cvtColor(text_layer, cv2.COLOR_BGR2GRAY)
    text_mask = (text_mask > 10).astype(np.uint8) * 255
    
    # Combine: text behind subject
    # Where subject is, use original frame; where text is and no subject, use text
    subject_mask_3ch = cv2.cvtColor(subject_mask, cv2.COLOR_GRAY2BGR) if len(subject_mask.shape) == 2 else subject_mask
    text_mask_3ch = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
    
    # Text visible only where there's no subject
    text_visible_mask = cv2.bitwise_and(text_mask_3ch, cv2.bitwise_not(subject_mask_3ch))
    
    # Blend
    result = np.where(text_visible_mask > 0, text_layer, result)
    
    return result


def create_subject_mask(frame: np.ndarray) -> np.ndarray:
    """Create a mask for the main subject using edge detection and morphology"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create connected regions
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask from largest contours
    mask = np.zeros(gray.shape, dtype=np.uint8)
    
    if contours:
        # Sort by area and take largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Fill the largest contours
        for contour in contours[:5]:
            if cv2.contourArea(contour) > frame.shape[0] * frame.shape[1] * 0.01:
                cv2.fillPoly(mask, [contour], 255)
    
    # Smooth the mask
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    
    return mask


def create_kinetic_typography(
    frame: np.ndarray,
    words: list,
    animation_progress: float,
    style_name: str = "bold"
) -> np.ndarray:
    """Create kinetic typography effect"""
    if style_name not in TEXT_STYLES:
        style_name = "bold"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    
    h, w = frame.shape[:2]
    
    # Calculate which words to show based on progress
    words_to_show = int(len(words) * animation_progress) + 1
    words_to_show = min(words_to_show, len(words))
    
    # Position words
    y_start = h // 3
    y_spacing = int(style.font_scale * 50)
    
    for i, word in enumerate(words[:words_to_show]):
        # Calculate word position with animation
        word_progress = min(1.0, (animation_progress * len(words) - i) * 2)
        
        if word_progress <= 0:
            continue
        
        # Get text size
        (text_w, text_h), _ = cv2.getTextSize(
            word, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
        )
        
        # Center horizontally with slide-in effect
        x = int((w - text_w) // 2 + (1 - word_progress) * 200)
        y = y_start + i * y_spacing
        
        # Apply fade-in
        alpha = word_progress
        
        # Draw text with alpha
        overlay = result.copy()
        
        # Shadow
        shadow_pos = (x + style.shadow_offset[0], y + style.shadow_offset[1])
        cv2.putText(
            overlay, word, shadow_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            style.shadow_color,
            style.thickness
        )
        
        # Main text
        cv2.putText(
            overlay, word, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            style.color,
            style.thickness
        )
        
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    return result


def create_lower_third(
    frame: np.ndarray,
    title: str,
    subtitle: str = "",
    style_name: str = "minimal"
) -> np.ndarray:
    """Create a professional lower third graphic"""
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Lower third dimensions
    lt_height = int(h * 0.15)
    lt_y = h - lt_height - int(h * 0.05)
    
    # Create semi-transparent background
    overlay = result.copy()
    cv2.rectangle(
        overlay,
        (int(w * 0.05), lt_y),
        (int(w * 0.6), lt_y + lt_height),
        (0, 0, 0),
        -1
    )
    result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)
    
    # Add accent line
    cv2.line(
        result,
        (int(w * 0.05), lt_y),
        (int(w * 0.05), lt_y + lt_height),
        (0, 200, 255),
        3
    )
    
    # Add title
    cv2.putText(
        result, title,
        (int(w * 0.07), lt_y + int(lt_height * 0.45)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2
    )
    
    # Add subtitle
    if subtitle:
        cv2.putText(
            result, subtitle,
            (int(w * 0.07), lt_y + int(lt_height * 0.75)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1
        )
    
    return result


def get_available_text_styles() -> list:
    """Get list of available text styles"""
    return list(TEXT_STYLES.keys())



# ============== CREATIVE TEXT OVERLAY EFFECTS ==============

def apply_parallax_text(
    frame: np.ndarray,
    text: str,
    style_name: str = "cinematic",
    depth_level: float = 0.5,
    position: str = "center"
) -> np.ndarray:
    """
    Apply text with parallax depth effect - text appears to float at different depths.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        depth_level: 0.0 = far background, 1.0 = foreground
        position: "center", "top", "bottom", "left", "right"
    
    Returns:
        Frame with parallax text effect
    """
    if style_name not in TEXT_STYLES:
        style_name = "cinematic"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Adjust scale based on depth
    scale_factor = 0.5 + depth_level * 0.8
    adjusted_scale = style.font_scale * scale_factor
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, adjusted_scale, style.thickness
    )
    
    # Calculate position
    if position == "center":
        x, y = (w - text_w) // 2, (h + text_h) // 2
    elif position == "top":
        x, y = (w - text_w) // 2, int(h * 0.15) + text_h
    elif position == "bottom":
        x, y = (w - text_w) // 2, int(h * 0.85)
    elif position == "left":
        x, y = int(w * 0.05), (h + text_h) // 2
    elif position == "right":
        x, y = w - text_w - int(w * 0.05), (h + text_h) // 2
    else:
        x, y = (w - text_w) // 2, (h + text_h) // 2
    
    # Create depth blur effect for background text
    if depth_level < 0.7:
        blur_amount = int((1 - depth_level) * 15) * 2 + 1
        text_layer = np.zeros_like(frame)
        
        # Draw text on separate layer
        _draw_styled_text(text_layer, text, (x, y), style, adjusted_scale)
        
        # Apply blur for depth
        text_layer = cv2.GaussianBlur(text_layer, (blur_amount, blur_amount), 0)
        
        # Reduce opacity for distant text
        alpha = 0.4 + depth_level * 0.6
        mask = cv2.cvtColor(text_layer, cv2.COLOR_BGR2GRAY)
        mask = (mask > 10).astype(np.float32) * alpha
        mask = np.stack([mask] * 3, axis=-1)
        
        result = (result * (1 - mask) + text_layer * mask).astype(np.uint8)
    else:
        _draw_styled_text(result, text, (x, y), style, adjusted_scale)
    
    return result


def apply_3d_extrusion_text(
    frame: np.ndarray,
    text: str,
    style_name: str = "bold",
    extrusion_depth: int = 10,
    extrusion_angle: float = 45.0
) -> np.ndarray:
    """
    Apply 3D extruded text effect with depth shadows.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        extrusion_depth: Number of shadow layers for 3D effect
        extrusion_angle: Angle of extrusion in degrees
    
    Returns:
        Frame with 3D extruded text
    """
    if style_name not in TEXT_STYLES:
        style_name = "bold"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
    )
    
    # Center position
    x, y = (w - text_w) // 2, (h + text_h) // 2
    
    # Calculate extrusion direction
    angle_rad = math.radians(extrusion_angle)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    
    # Draw extrusion layers (back to front)
    for i in range(extrusion_depth, 0, -1):
        offset_x = int(dx * i)
        offset_y = int(dy * i)
        
        # Gradient color for depth
        depth_factor = i / extrusion_depth
        shadow_color = tuple(int(c * (0.2 + 0.3 * depth_factor)) for c in style.color)
        
        cv2.putText(
            result, text,
            (x + offset_x, y + offset_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            style.font_scale,
            shadow_color,
            style.thickness
        )
    
    # Draw main text on top
    _draw_styled_text(result, text, (x, y), style, style.font_scale)
    
    return result


def apply_wave_text(
    frame: np.ndarray,
    text: str,
    style_name: str = "neon",
    wave_amplitude: float = 20.0,
    wave_frequency: float = 0.1,
    phase: float = 0.0
) -> np.ndarray:
    """
    Apply wavy animated text effect - each character follows a sine wave.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        wave_amplitude: Height of wave in pixels
        wave_frequency: Frequency of wave
        phase: Animation phase (0.0 to 2*pi for full cycle)
    
    Returns:
        Frame with wavy text
    """
    if style_name not in TEXT_STYLES:
        style_name = "neon"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Calculate total text width
    total_width = 0
    char_widths = []
    for char in text:
        (char_w, _), _ = cv2.getTextSize(
            char, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
        )
        char_widths.append(char_w)
        total_width += char_w
    
    # Starting position (centered)
    start_x = (w - total_width) // 2
    base_y = h // 2
    
    # Draw each character with wave offset
    current_x = start_x
    for i, char in enumerate(text):
        # Calculate wave offset
        wave_offset = int(wave_amplitude * math.sin(phase + i * wave_frequency * 10))
        char_y = base_y + wave_offset
        
        # Draw character
        _draw_styled_text(result, char, (current_x, char_y), style, style.font_scale)
        
        current_x += char_widths[i]
    
    return result


def apply_glitch_text(
    frame: np.ndarray,
    text: str,
    style_name: str = "glitch",
    glitch_intensity: float = 0.5,
    rgb_split: int = 5
) -> np.ndarray:
    """
    Apply glitch/distortion effect to text with RGB channel splitting.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        glitch_intensity: 0.0 to 1.0, controls glitch amount
        rgb_split: Pixel offset for RGB channel separation
    
    Returns:
        Frame with glitched text
    """
    if style_name not in TEXT_STYLES:
        style_name = "glitch"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Get text size and position
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
    )
    x, y = (w - text_w) // 2, (h + text_h) // 2
    
    # Create separate layers for RGB channels
    red_layer = np.zeros_like(frame)
    green_layer = np.zeros_like(frame)
    blue_layer = np.zeros_like(frame)
    
    # Draw text on each channel with offset
    offset = int(rgb_split * glitch_intensity)
    
    cv2.putText(red_layer, text, (x - offset, y), cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale, (0, 0, 255), style.thickness)
    cv2.putText(green_layer, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale, (0, 255, 0), style.thickness)
    cv2.putText(blue_layer, text, (x + offset, y), cv2.FONT_HERSHEY_SIMPLEX,
                style.font_scale, (255, 0, 0), style.thickness)
    
    # Combine channels
    text_layer = cv2.addWeighted(red_layer, 0.33, green_layer, 0.33, 0)
    text_layer = cv2.addWeighted(text_layer, 1, blue_layer, 0.33, 0)
    
    # Add random horizontal slice displacement for glitch effect
    if glitch_intensity > 0.3:
        num_slices = int(5 * glitch_intensity)
        for _ in range(num_slices):
            slice_y = np.random.randint(y - text_h, y + 10)
            slice_height = np.random.randint(2, 8)
            slice_offset = np.random.randint(-10, 10)
            
            if 0 <= slice_y < h and slice_y + slice_height < h:
                slice_data = text_layer[slice_y:slice_y+slice_height, :].copy()
                text_layer[slice_y:slice_y+slice_height, :] = np.roll(slice_data, slice_offset, axis=1)
    
    # Blend with original
    mask = cv2.cvtColor(text_layer, cv2.COLOR_BGR2GRAY)
    mask = (mask > 5).astype(np.float32)
    mask = np.stack([mask] * 3, axis=-1)
    
    result = (result * (1 - mask) + text_layer).astype(np.uint8)
    
    return result


def apply_reveal_text(
    frame: np.ndarray,
    text: str,
    style_name: str = "cinematic",
    reveal_progress: float = 1.0,
    reveal_direction: str = "left"
) -> np.ndarray:
    """
    Apply text reveal animation effect.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        reveal_progress: 0.0 to 1.0, how much text is revealed
        reveal_direction: "left", "right", "top", "bottom", "center"
    
    Returns:
        Frame with revealed text
    """
    if style_name not in TEXT_STYLES:
        style_name = "cinematic"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
    )
    x, y = (w - text_w) // 2, (h + text_h) // 2
    
    # Create text layer
    text_layer = np.zeros_like(frame)
    _draw_styled_text(text_layer, text, (x, y), style, style.font_scale)
    
    # Create reveal mask
    mask = np.zeros((h, w), dtype=np.float32)
    
    if reveal_direction == "left":
        reveal_x = int(x + text_w * reveal_progress)
        mask[:, :reveal_x] = 1.0
    elif reveal_direction == "right":
        reveal_x = int(x + text_w * (1 - reveal_progress))
        mask[:, reveal_x:] = 1.0
    elif reveal_direction == "top":
        reveal_y = int((y - text_h) + (text_h + 20) * reveal_progress)
        mask[:reveal_y, :] = 1.0
    elif reveal_direction == "bottom":
        reveal_y = int(y - (text_h + 20) * reveal_progress)
        mask[reveal_y:, :] = 1.0
    elif reveal_direction == "center":
        center_x = w // 2
        half_reveal = int(text_w * reveal_progress / 2)
        mask[:, center_x - half_reveal:center_x + half_reveal] = 1.0
    
    # Apply mask
    mask = np.stack([mask] * 3, axis=-1)
    text_mask = cv2.cvtColor(text_layer, cv2.COLOR_BGR2GRAY)
    text_mask = (text_mask > 5).astype(np.float32)
    text_mask = np.stack([text_mask] * 3, axis=-1)
    
    combined_mask = mask * text_mask
    result = (result * (1 - combined_mask) + text_layer * combined_mask).astype(np.uint8)
    
    return result


def apply_floating_text(
    frame: np.ndarray,
    text: str,
    style_name: str = "holographic",
    float_offset: float = 0.0,
    blur_background: bool = True
) -> np.ndarray:
    """
    Apply floating text effect with optional background blur behind text.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        float_offset: Vertical float animation offset (-1.0 to 1.0)
        blur_background: Whether to blur area behind text
    
    Returns:
        Frame with floating text
    """
    if style_name not in TEXT_STYLES:
        style_name = "holographic"
    
    style = TEXT_STYLES[style_name]
    result = frame.copy()
    h, w = frame.shape[:2]
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale, style.thickness
    )
    
    # Calculate floating position
    base_y = h // 2
    float_amount = int(float_offset * 30)
    x, y = (w - text_w) // 2, base_y + float_amount
    
    # Blur background behind text area
    if blur_background:
        padding = 30
        x1 = max(0, x - padding)
        y1 = max(0, y - text_h - padding)
        x2 = min(w, x + text_w + padding)
        y2 = min(h, y + padding)
        
        roi = result[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (21, 21), 0)
        
        # Create gradient mask for smooth blend
        mask = np.zeros((y2-y1, x2-x1), dtype=np.float32)
        cv2.ellipse(mask, ((x2-x1)//2, (y2-y1)//2), 
                   ((x2-x1)//2, (y2-y1)//2), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        mask = np.stack([mask] * 3, axis=-1)
        
        result[y1:y2, x1:x2] = (roi * (1 - mask * 0.7) + blurred_roi * mask * 0.7).astype(np.uint8)
    
    # Draw text with glow
    _draw_styled_text(result, text, (x, y), style, style.font_scale)
    
    return result


def apply_text_behind_video(
    frame: np.ndarray,
    text: str,
    style_name: str = "cinematic",
    position: str = "center",
    opacity: float = 0.8
) -> np.ndarray:
    """
    Apply text that appears behind the main subject using edge detection.
    Creates the popular "text behind subject" effect.
    
    Args:
        frame: Input frame
        text: Text to overlay
        style_name: Style preset name
        position: Text position
        opacity: Text opacity (0.0 to 1.0)
    
    Returns:
        Frame with text behind subject
    """
    if style_name not in TEXT_STYLES:
        style_name = "cinematic"
    
    style = TEXT_STYLES[style_name]
    h, w = frame.shape[:2]
    
    # Get text size
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, style.font_scale * 1.5, style.thickness
    )
    
    # Calculate position
    if position == "center":
        x, y = (w - text_w) // 2, (h + text_h) // 2
    elif position == "top":
        x, y = (w - text_w) // 2, int(h * 0.2) + text_h
    elif position == "bottom":
        x, y = (w - text_w) // 2, int(h * 0.8)
    else:
        x, y = (w - text_w) // 2, (h + text_h) // 2
    
    # Create subject mask
    subject_mask = create_subject_mask(frame)
    
    # Create text layer
    text_layer = np.zeros_like(frame)
    _draw_styled_text(text_layer, text, (x, y), style, style.font_scale * 1.5)
    
    # Create text mask
    text_mask = cv2.cvtColor(text_layer, cv2.COLOR_BGR2GRAY)
    text_mask = (text_mask > 10).astype(np.uint8) * 255
    
    # Text visible only where there's no subject
    visible_mask = cv2.bitwise_and(text_mask, cv2.bitwise_not(subject_mask))
    visible_mask = visible_mask.astype(np.float32) / 255.0 * opacity
    visible_mask = np.stack([visible_mask] * 3, axis=-1)
    
    # Blend
    result = (frame * (1 - visible_mask) + text_layer * visible_mask).astype(np.uint8)
    
    return result


def _draw_styled_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    style: TextStyle,
    scale: float
) -> None:
    """Helper function to draw text with full styling."""
    x, y = position
    
    # Draw glow first (if enabled)
    if style.glow_color and style.glow_radius > 0:
        glow_layer = np.zeros_like(image)
        cv2.putText(
            glow_layer, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            style.glow_color,
            style.thickness + 4
        )
        glow_layer = cv2.GaussianBlur(
            glow_layer, 
            (style.glow_radius * 2 + 1, style.glow_radius * 2 + 1), 
            0
        )
        mask = cv2.cvtColor(glow_layer, cv2.COLOR_BGR2GRAY)
        mask = (mask > 5).astype(np.float32) * 0.5
        mask = np.stack([mask] * 3, axis=-1)
        image[:] = (image * (1 - mask) + glow_layer).astype(np.uint8)
    
    # Draw shadow
    shadow_pos = (x + style.shadow_offset[0], y + style.shadow_offset[1])
    cv2.putText(
        image, text, shadow_pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        style.shadow_color,
        style.thickness
    )
    
    # Draw outline (if enabled)
    if style.outline_color and style.outline_thickness > 0:
        cv2.putText(
            image, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            style.outline_color,
            style.thickness + style.outline_thickness
        )
    
    # Draw main text
    cv2.putText(
        image, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        style.color,
        style.thickness
    )


def get_style_preview_info() -> List[Dict]:
    """Get information about all available text styles for UI display."""
    style_info = []
    for name, style in TEXT_STYLES.items():
        info = {
            "id": name,
            "name": name.replace("_", " ").title(),
            "description": _get_style_description(name),
            "color": f"rgb{style.color}",
            "has_glow": style.glow_color is not None,
            "has_outline": style.outline_color is not None
        }
        style_info.append(info)
    return style_info


def _get_style_description(style_name: str) -> str:
    """Get description for a text style."""
    descriptions = {
        "cinematic": "Classic movie title look with clean shadows",
        "neon": "Vibrant glowing neon sign effect",
        "minimal": "Clean, modern minimalist style",
        "bold": "Strong, impactful heavy text",
        "elegant": "Sophisticated warm-toned style",
        "cyberpunk": "Futuristic cyan with pink accents",
        "retro": "Vintage warm golden tones",
        "glitch": "Digital distortion effect",
        "holographic": "Ethereal floating hologram look",
        "fire": "Fiery orange glow effect"
    }
    return descriptions.get(style_name, "Custom text style")


def get_effect_types() -> List[Dict]:
    """Get list of available text effect types."""
    return [
        {"id": "standard", "name": "Standard Overlay", "description": "Simple text overlay on video"},
        {"id": "behind", "name": "Text Behind Subject", "description": "Text appears behind the main subject"},
        {"id": "parallax", "name": "Parallax Depth", "description": "Text with depth blur effect"},
        {"id": "3d", "name": "3D Extrusion", "description": "Text with 3D shadow depth"},
        {"id": "wave", "name": "Wave Animation", "description": "Wavy animated text"},
        {"id": "glitch", "name": "Glitch Effect", "description": "Digital glitch distortion"},
        {"id": "reveal", "name": "Reveal Animation", "description": "Text reveal transition"},
        {"id": "floating", "name": "Floating Text", "description": "Text floating with blur background"}
    ]
