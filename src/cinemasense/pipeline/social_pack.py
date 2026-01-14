"""
Social Pack Generator - Thumbnails, titles, hashtags, captions

Generates social media assets including platform-optimized thumbnails,
title suggestions, hashtags, and captions.

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import logging

from ..core.video_capture import SafeVideoCapture, VideoOpenError
from ..core.file_ops import FileOps

logger = logging.getLogger("cinemasense.pipeline.social_pack")


@dataclass
class SocialPackResult:
    """Complete social pack generation result"""
    thumbnail_path: str
    title_suggestions: List[str]
    hashtags: List[str]
    caption: str
    platform_optimized: Dict[str, Dict]


# Platform specifications - Requirements 7.2, 7.4
PLATFORM_SPECS = {
    "youtube": {
        "thumbnail_size": (1280, 720),
        "title_max_length": 100,
        "description_max_length": 5000,
        "hashtag_limit": 15
    },
    "instagram": {
        "thumbnail_size": (1080, 1080),
        "title_max_length": 0,  # No title, just caption
        "description_max_length": 2200,
        "hashtag_limit": 30
    },
    "tiktok": {
        "thumbnail_size": (1080, 1920),
        "title_max_length": 150,
        "description_max_length": 2200,
        "hashtag_limit": 5
    },
    "twitter": {
        "thumbnail_size": (1200, 675),
        "title_max_length": 0,
        "description_max_length": 280,
        "hashtag_limit": 3
    }
}

# Valid platform names
VALID_PLATFORMS: Set[str] = set(PLATFORM_SPECS.keys())


class SocialPackError(Exception):
    """Raised when social pack generation fails."""
    pass


class InvalidPlatformError(SocialPackError):
    """Raised when an invalid platform is specified."""
    pass


def validate_platform(platform: str) -> str:
    """
    Validate platform name and return normalized version.
    
    Args:
        platform: Platform name to validate
        
    Returns:
        Normalized platform name (lowercase)
        
    Raises:
        InvalidPlatformError: If platform is not supported
    """
    normalized = platform.lower().strip()
    if normalized not in VALID_PLATFORMS:
        raise InvalidPlatformError(
            f"Invalid platform '{platform}'. "
            f"Supported platforms: {', '.join(sorted(VALID_PLATFORMS))}"
        )
    return normalized


def get_platform_spec(platform: str) -> Dict:
    """
    Get platform specifications with validation.
    
    Args:
        platform: Platform name
        
    Returns:
        Platform specification dictionary
        
    Raises:
        InvalidPlatformError: If platform is not supported
    """
    normalized = validate_platform(platform)
    return PLATFORM_SPECS[normalized]


def extract_best_thumbnail_frame(
    video_path: str,
    analysis_results: Dict = None
) -> np.ndarray:
    """
    Extract the best frame for thumbnail based on analysis.
    
    Uses SafeVideoCapture for proper resource cleanup.
    
    Args:
        video_path: Path to video file
        analysis_results: Optional analysis results for smarter selection
        
    Returns:
        Best frame as numpy array
        
    Raises:
        SocialPackError: If no valid frame can be extracted
        
    Requirements: 7.1
    """
    try:
        with SafeVideoCapture(video_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                raise SocialPackError(f"Video has no frames: {video_path}")
            
            best_frame = None
            best_score = -1
            
            # Sample frames throughout the video
            sample_positions = [0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9]
            
            for pos in sample_positions:
                frame_idx = int(total_frames * pos)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                # Score the frame
                score = score_thumbnail_frame(frame)
                
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()
            
            if best_frame is None:
                # Fallback to middle frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, best_frame = cap.read()
                
                if not ret or best_frame is None:
                    # Last resort: try first frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, best_frame = cap.read()
            
            if best_frame is None:
                raise SocialPackError(
                    f"Could not extract any valid frame from video: {video_path}"
                )
            
            logger.debug(f"Extracted best thumbnail frame with score {best_score:.3f}")
            return best_frame
            
    except VideoOpenError as e:
        raise SocialPackError(f"Failed to open video for thumbnail extraction: {e}")


def score_thumbnail_frame(frame: np.ndarray) -> float:
    """
    Score a frame for thumbnail quality.
    
    Evaluates brightness, contrast, saturation, and sharpness.
    
    Args:
        frame: Frame to score
        
    Returns:
        Quality score between 0 and 1
        
    Requirements: 7.1
    """
    if frame is None or frame.size == 0:
        return 0.0
    
    score = 0.0
    
    try:
        # Brightness score (not too dark, not too bright)
        brightness = np.mean(frame)
        brightness_score = 1.0 - abs(brightness - 128) / 128
        score += brightness_score * 0.2
        
        # Contrast score
        contrast = np.std(frame)
        contrast_score = min(contrast / 80, 1.0)
        score += contrast_score * 0.2
        
        # Saturation score
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        saturation_score = min(saturation / 100, 1.0)
        score += saturation_score * 0.2
        
        # Sharpness score (using Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500, 1.0)
        score += sharpness_score * 0.2
        
        # Face detection bonus (if faces present)
        try:
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                score += 0.2
        except Exception:
            # Face detection is optional, don't fail if it doesn't work
            pass
        
    except Exception as e:
        logger.warning(f"Error scoring thumbnail frame: {e}")
        return 0.0
    
    return min(max(score, 0.0), 1.0)


def create_thumbnail(
    frame: np.ndarray,
    platform: str = "youtube",
    add_text: str = None,
    style: str = "vibrant"
) -> np.ndarray:
    """
    Create an optimized thumbnail for the specified platform.
    
    Ensures output dimensions exactly match platform specifications.
    
    Args:
        frame: Source frame
        platform: Target platform (youtube, instagram, tiktok, twitter)
        add_text: Optional text overlay
        style: Enhancement style (vibrant, dramatic, clean)
        
    Returns:
        Thumbnail with exact platform dimensions
        
    Raises:
        InvalidPlatformError: If platform is not supported
        
    Requirements: 7.2
    """
    # Validate platform
    normalized_platform = validate_platform(platform)
    specs = PLATFORM_SPECS[normalized_platform]
    target_size = specs["thumbnail_size"]
    target_w, target_h = target_size
    
    if frame is None or frame.size == 0:
        # Return blank frame with correct dimensions
        logger.warning("Empty frame provided, creating blank thumbnail")
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Get current dimensions
    h, w = frame.shape[:2]
    target_ratio = target_w / target_h
    current_ratio = w / h
    
    # Crop to target aspect ratio
    if current_ratio > target_ratio:
        # Crop width
        new_w = int(h * target_ratio)
        start_x = (w - new_w) // 2
        cropped = frame[:, start_x:start_x + new_w]
    else:
        # Crop height
        new_h = int(w / target_ratio)
        start_y = (h - new_h) // 2
        cropped = frame[start_y:start_y + new_h, :]
    
    # Resize to exact target dimensions
    thumbnail = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Verify dimensions match exactly - Requirements 7.2
    assert thumbnail.shape[1] == target_w, f"Width mismatch: {thumbnail.shape[1]} != {target_w}"
    assert thumbnail.shape[0] == target_h, f"Height mismatch: {thumbnail.shape[0]} != {target_h}"
    
    # Apply style enhancements
    if style == "vibrant":
        thumbnail = enhance_vibrant(thumbnail)
    elif style == "dramatic":
        thumbnail = enhance_dramatic(thumbnail)
    elif style == "clean":
        thumbnail = enhance_clean(thumbnail)
    
    # Add text overlay if specified
    if add_text:
        thumbnail = add_thumbnail_text(thumbnail, add_text)
    
    # Final dimension verification
    if thumbnail.shape[1] != target_w or thumbnail.shape[0] != target_h:
        thumbnail = cv2.resize(thumbnail, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    return thumbnail


def enhance_vibrant(frame: np.ndarray) -> np.ndarray:
    """Apply vibrant enhancement for thumbnails."""
    if frame is None or frame.size == 0:
        return frame
    
    try:
        # Increase saturation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        # Increase contrast
        result = cv2.convertScaleAbs(result, alpha=1.2, beta=10)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        result = cv2.filter2D(result, -1, kernel)
        
        return result
    except Exception as e:
        logger.warning(f"Error in vibrant enhancement: {e}")
        return frame


def enhance_dramatic(frame: np.ndarray) -> np.ndarray:
    """Apply dramatic enhancement for thumbnails."""
    if frame is None or frame.size == 0:
        return frame
    
    try:
        # Increase contrast significantly
        result = cv2.convertScaleAbs(frame, alpha=1.4, beta=-20)
        
        # Add vignette
        h, w = result.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        center_x, center_y = w // 2, h // 2
        mask = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        mask = 1 - (mask / mask.max()) * 0.4
        
        for i in range(3):
            result[:, :, i] = (result[:, :, i] * mask).astype(np.uint8)
        
        return result
    except Exception as e:
        logger.warning(f"Error in dramatic enhancement: {e}")
        return frame


def enhance_clean(frame: np.ndarray) -> np.ndarray:
    """Apply clean, minimal enhancement."""
    if frame is None or frame.size == 0:
        return frame
    
    try:
        # Slight brightness boost
        result = cv2.convertScaleAbs(frame, alpha=1.05, beta=5)
        
        # Subtle sharpening
        blurred = cv2.GaussianBlur(result, (0, 0), 3)
        result = cv2.addWeighted(result, 1.2, blurred, -0.2, 0)
        
        return result
    except Exception as e:
        logger.warning(f"Error in clean enhancement: {e}")
        return frame


def add_thumbnail_text(thumbnail: np.ndarray, text: str) -> np.ndarray:
    """Add text overlay to thumbnail."""
    if thumbnail is None or thumbnail.size == 0:
        return thumbnail
    
    try:
        result = thumbnail.copy()
        h, w = result.shape[:2]
        
        # Calculate text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = w / 400  # Scale based on thumbnail width
        thickness = max(2, int(font_scale * 2))
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position at bottom center
        x = (w - text_w) // 2
        y = h - int(h * 0.1)
        
        # Draw background rectangle
        padding = 10
        cv2.rectangle(
            result,
            (x - padding, y - text_h - padding),
            (x + text_w + padding, y + padding),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(result, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        return result
    except Exception as e:
        logger.warning(f"Error adding thumbnail text: {e}")
        return thumbnail



def generate_title_suggestions(
    video_metadata: Dict,
    emotion_analysis: Dict = None,
    scene_analysis: Dict = None
) -> List[str]:
    """
    Generate exactly 5 title suggestions based on video analysis.
    
    Args:
        video_metadata: Video metadata dictionary
        emotion_analysis: Optional emotion analysis results
        scene_analysis: Optional scene analysis results
        
    Returns:
        List of exactly 5 title suggestions
        
    Requirements: 7.3
    """
    duration = video_metadata.get("duration_s", 0)
    
    # Determine mood from analysis
    mood = "Cinematic"
    if emotion_analysis:
        dominant_emotion = emotion_analysis.get("dominant_emotion", "")
        if dominant_emotion:
            mood = dominant_emotion
    
    # Duration string
    duration_str = f"{int(duration)}s" if duration < 60 else f"{int(duration/60)}min"
    
    # Extended template pool to ensure we always have enough
    templates = [
        "Epic {mood} Moments Compilation",
        "{mood} Vibes | {duration}",
        "When {mood} Hits Different",
        "POV: You're Feeling {mood}",
        "{mood} Energy Only 🔥",
        "This {mood} Edit Goes Hard",
        "The Ultimate {mood} Experience",
        "{mood} Mode: Activated",
        "Pure {mood} Content",
        "That {mood} Feeling ✨",
        "{mood} Aesthetic | {duration}",
        "Living for This {mood} Energy"
    ]
    
    suggestions = []
    template_idx = 0
    
    # Generate exactly 5 unique suggestions - Requirements 7.3
    while len(suggestions) < 5 and template_idx < len(templates):
        title = templates[template_idx].format(mood=mood, duration=duration_str)
        if title not in suggestions:
            suggestions.append(title)
        template_idx += 1
    
    # Fallback if we somehow don't have enough
    while len(suggestions) < 5:
        suggestions.append(f"{mood} Video #{len(suggestions) + 1}")
    
    # Ensure exactly 5 suggestions
    result = suggestions[:5]
    
    # Validate all suggestions are non-empty strings
    result = [s if s and isinstance(s, str) else f"Video Title {i+1}" 
              for i, s in enumerate(result)]
    
    assert len(result) == 5, f"Expected 5 title suggestions, got {len(result)}"
    assert all(isinstance(s, str) and len(s) > 0 for s in result), "All titles must be non-empty strings"
    
    logger.debug(f"Generated {len(result)} title suggestions")
    return result


def generate_hashtags(
    video_metadata: Dict,
    emotion_analysis: Dict = None,
    platform: str = "instagram"
) -> List[str]:
    """
    Generate relevant hashtags with platform-specific limits enforced.
    
    Args:
        video_metadata: Video metadata dictionary
        emotion_analysis: Optional emotion analysis results
        platform: Target platform for hashtag limit
        
    Returns:
        List of hashtags (with # prefix) respecting platform limit
        
    Raises:
        InvalidPlatformError: If platform is not supported
        
    Requirements: 7.4
    """
    # Validate platform and get limit
    normalized_platform = validate_platform(platform)
    limit = PLATFORM_SPECS[normalized_platform]["hashtag_limit"]
    
    hashtags = []
    
    # Base hashtags
    base_tags = [
        "videoediting", "cinematography", "filmmaking",
        "contentcreator", "videography", "editing"
    ]
    
    # Mood-based hashtags
    mood_tags = {
        "Joy": ["happy", "goodvibes", "positivity", "happiness"],
        "Tension": ["thriller", "suspense", "intense", "dramatic"],
        "Calm": ["peaceful", "relaxing", "chill", "serene"],
        "Energy": ["energetic", "hype", "motivation", "power"],
        "Melancholy": ["emotional", "deep", "moody", "aesthetic"],
        "Mystery": ["mysterious", "dark", "cinematic", "atmospheric"]
    }
    
    # Platform-specific tags
    platform_tags = {
        "youtube": ["youtube", "youtuber", "subscribe"],
        "instagram": ["reels", "instareels", "viral"],
        "tiktok": ["fyp", "foryou", "tiktok"],
        "twitter": ["video", "content"]
    }
    
    # Add platform tags first (highest priority)
    if normalized_platform in platform_tags:
        hashtags.extend(platform_tags[normalized_platform])
    
    # Add mood tags
    if emotion_analysis:
        mood = emotion_analysis.get("dominant_emotion", "")
        if mood in mood_tags:
            hashtags.extend(mood_tags[mood])
    
    # Add base tags
    hashtags.extend(base_tags)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_hashtags = []
    for tag in hashtags:
        tag_lower = tag.lower()
        if tag_lower not in seen:
            seen.add(tag_lower)
            unique_hashtags.append(tag)
    
    # Enforce platform limit - Requirements 7.4
    limited_hashtags = unique_hashtags[:limit]
    
    # Add # prefix
    result = ["#" + tag for tag in limited_hashtags]
    
    # Verify limit is enforced
    assert len(result) <= limit, f"Hashtag count {len(result)} exceeds limit {limit} for {platform}"
    
    logger.debug(f"Generated {len(result)} hashtags for {platform} (limit: {limit})")
    return result


def generate_caption(
    video_metadata: Dict,
    emotion_analysis: Dict = None,
    platform: str = "instagram"
) -> str:
    """
    Generate an engaging caption for the specified platform.
    
    Args:
        video_metadata: Video metadata dictionary
        emotion_analysis: Optional emotion analysis results
        platform: Target platform
        
    Returns:
        Caption string within platform character limit
        
    Raises:
        InvalidPlatformError: If platform is not supported
        
    Requirements: 7.5
    """
    # Validate platform
    normalized_platform = validate_platform(platform)
    max_length = PLATFORM_SPECS[normalized_platform]["description_max_length"]
    
    mood = "amazing"
    if emotion_analysis:
        mood = emotion_analysis.get("dominant_emotion", "amazing").lower()
    
    # Caption templates by platform
    templates = {
        "youtube": f"Experience the {mood} vibes in this edit! 🎬\n\nDon't forget to like, subscribe, and hit the bell! 🔔\n\nWhat moment was your favorite? Let me know in the comments! 👇",
        "instagram": f"That {mood} feeling hits different ✨\n\n.\n.\n.\n",
        "tiktok": f"When the {mood} hits 🔥 #fyp",
        "twitter": f"New edit just dropped! {mood.title()} vibes only 🎬"
    }
    
    caption = templates.get(normalized_platform, templates["instagram"])
    
    # Truncate if needed to respect platform limit
    if len(caption) > max_length:
        caption = caption[:max_length - 3] + "..."
    
    return caption


def generate_social_pack(
    video_path: str,
    output_dir: Path,
    video_metadata: Dict,
    emotion_analysis: Dict = None,
    platforms: List[str] = None
) -> SocialPackResult:
    """
    Generate complete social media pack with platform-optimized assets.
    
    Uses SafeVideoCapture for video access and enforces all platform
    specifications including thumbnail dimensions and hashtag limits.
    
    Args:
        video_path: Path to source video
        output_dir: Directory for output files
        video_metadata: Video metadata dictionary
        emotion_analysis: Optional emotion analysis results
        platforms: List of target platforms (default: youtube, instagram, tiktok)
        
    Returns:
        SocialPackResult with all generated assets
        
    Raises:
        SocialPackError: If generation fails
        InvalidPlatformError: If invalid platform specified
        
    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    if platforms is None:
        platforms = ["youtube", "instagram", "tiktok"]
    
    # Validate all platforms upfront
    validated_platforms = []
    for platform in platforms:
        try:
            validated_platforms.append(validate_platform(platform))
        except InvalidPlatformError as e:
            logger.warning(f"Skipping invalid platform: {e}")
    
    if not validated_platforms:
        raise SocialPackError("No valid platforms specified")
    
    # Ensure output directory exists
    output_dir = Path(output_dir)
    FileOps.ensure_directory(output_dir)
    
    # Extract best thumbnail frame using SafeVideoCapture
    logger.info(f"Extracting best thumbnail frame from {video_path}")
    best_frame = extract_best_thumbnail_frame(video_path)
    
    # Generate thumbnails for each platform
    platform_optimized = {}
    
    for platform in validated_platforms:
        specs = PLATFORM_SPECS[platform]
        target_size = specs["thumbnail_size"]
        
        # Create thumbnail with exact platform dimensions
        thumbnail = create_thumbnail(best_frame, platform, style="vibrant")
        
        # Verify dimensions match platform spec - Requirements 7.2
        actual_h, actual_w = thumbnail.shape[:2]
        expected_w, expected_h = target_size
        
        if actual_w != expected_w or actual_h != expected_h:
            logger.warning(
                f"Thumbnail dimension mismatch for {platform}: "
                f"got {actual_w}x{actual_h}, expected {expected_w}x{expected_h}. Resizing."
            )
            thumbnail = cv2.resize(thumbnail, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Save thumbnail
        thumbnail_filename = FileOps.sanitize_filename(f"thumbnail_{platform}.jpg")
        thumbnail_path = output_dir / thumbnail_filename
        
        success = cv2.imwrite(str(thumbnail_path), thumbnail)
        if not success:
            logger.warning(f"Failed to save thumbnail for {platform}")
        
        # Generate platform-specific hashtags with limit enforced
        hashtags = generate_hashtags(video_metadata, emotion_analysis, platform)
        
        # Verify hashtag limit - Requirements 7.4
        hashtag_limit = specs["hashtag_limit"]
        assert len(hashtags) <= hashtag_limit, \
            f"Hashtag count {len(hashtags)} exceeds limit {hashtag_limit} for {platform}"
        
        platform_optimized[platform] = {
            "thumbnail_path": str(thumbnail_path),
            "thumbnail_dimensions": target_size,
            "hashtags": hashtags,
            "hashtag_count": len(hashtags),
            "hashtag_limit": hashtag_limit,
            "caption": generate_caption(video_metadata, emotion_analysis, platform)
        }
        
        logger.debug(
            f"Generated {platform} assets: thumbnail {target_size}, "
            f"{len(hashtags)}/{hashtag_limit} hashtags"
        )
    
    # Generate exactly 5 title suggestions - Requirements 7.3
    titles = generate_title_suggestions(video_metadata, emotion_analysis)
    assert len(titles) == 5, f"Expected 5 title suggestions, got {len(titles)}"
    assert all(isinstance(t, str) and len(t) > 0 for t in titles), \
        "All titles must be non-empty strings"
    
    # Generate general hashtags (using instagram as default)
    hashtags = generate_hashtags(video_metadata, emotion_analysis, "instagram")
    
    # Generate general caption
    caption = generate_caption(video_metadata, emotion_analysis, "instagram")
    
    # Save main thumbnail (YouTube format as default)
    main_thumbnail = create_thumbnail(best_frame, "youtube", style="vibrant")
    main_thumbnail_filename = FileOps.sanitize_filename("thumbnail_main.jpg")
    main_thumbnail_path = output_dir / main_thumbnail_filename
    cv2.imwrite(str(main_thumbnail_path), main_thumbnail)
    
    logger.info(
        f"Social pack generated: {len(validated_platforms)} platforms, "
        f"{len(titles)} titles, {len(hashtags)} hashtags"
    )
    
    return SocialPackResult(
        thumbnail_path=str(main_thumbnail_path),
        title_suggestions=titles,
        hashtags=hashtags,
        caption=caption,
        platform_optimized=platform_optimized
    )
