"""
Property-based tests for Social Pack Platform Compliance and Title Generation.

Feature: cinemasense-stabilization
Property 13: Social Pack Platform Compliance
Validates: Requirements 7.2, 7.4

Property 14: Title Generation Count
Validates: Requirements 7.3

Tests that for any platform in the social pack generation, the generated
thumbnail SHALL have the exact dimensions specified for that platform,
and hashtag count SHALL not exceed the platform limit.

Also tests that exactly 5 title suggestions are returned, each being a non-empty string.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from cinemasense.pipeline.social_pack import (
    PLATFORM_SPECS,
    VALID_PLATFORMS,
    create_thumbnail,
    generate_hashtags,
    generate_title_suggestions,
    validate_platform,
    InvalidPlatformError,
)


# Strategy for generating valid platform names
platform_strategy = st.sampled_from(list(VALID_PLATFORMS))


@st.composite
def valid_frame_strategy(draw):
    """Generate a valid BGR frame with random pixel values."""
    height = draw(st.integers(min_value=50, max_value=200))
    width = draw(st.integers(min_value=50, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@st.composite
def video_metadata_strategy(draw):
    """Generate valid video metadata dictionary."""
    return {
        "duration_s": draw(st.floats(min_value=1.0, max_value=3600.0)),
        "fps": draw(st.floats(min_value=15.0, max_value=60.0)),
        "width": draw(st.integers(min_value=320, max_value=3840)),
        "height": draw(st.integers(min_value=240, max_value=2160)),
        "name": draw(st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N')))),
    }


@st.composite
def emotion_analysis_strategy(draw):
    """Generate valid emotion analysis dictionary."""
    emotions = ["Joy", "Tension", "Calm", "Melancholy", "Energy", "Mystery"]
    return {
        "dominant_emotion": draw(st.sampled_from(emotions)),
        "overall_score": draw(st.floats(min_value=0.0, max_value=1.0)),
    }


class TestSocialPackPlatformCompliance:
    """
    Property tests for Social Pack Platform Compliance.
    
    Feature: cinemasense-stabilization, Property 13: Social Pack Platform Compliance
    Validates: Requirements 7.2, 7.4
    """
    
    @given(
        frame=valid_frame_strategy(),
        platform=platform_strategy
    )
    @settings(max_examples=100)
    def test_thumbnail_dimensions_match_platform_spec(self, frame, platform):
        """
        Property 13: Social Pack Platform Compliance (thumbnail dimensions)
        
        For any platform in the social pack generation, the generated thumbnail
        SHALL have the exact dimensions specified for that platform.
        
        Validates: Requirements 7.2
        """
        # Get expected dimensions from platform spec
        specs = PLATFORM_SPECS[platform]
        expected_width, expected_height = specs["thumbnail_size"]
        
        # Create thumbnail for platform
        thumbnail = create_thumbnail(frame.copy(), platform)
        
        # Verify thumbnail is valid
        assert thumbnail is not None, "Thumbnail should not be None"
        assert isinstance(thumbnail, np.ndarray), "Thumbnail should be numpy array"
        assert len(thumbnail.shape) == 3, f"Thumbnail should be 3D, got {thumbnail.shape}"
        assert thumbnail.shape[2] == 3, f"Thumbnail should have 3 channels"
        
        # Verify exact dimensions match platform spec - Requirements 7.2
        actual_height, actual_width = thumbnail.shape[:2]
        
        assert actual_width == expected_width, (
            f"Platform '{platform}' thumbnail width mismatch: "
            f"expected {expected_width}, got {actual_width}"
        )
        assert actual_height == expected_height, (
            f"Platform '{platform}' thumbnail height mismatch: "
            f"expected {expected_height}, got {actual_height}"
        )
    
    @given(
        metadata=video_metadata_strategy(),
        emotion=emotion_analysis_strategy(),
        platform=platform_strategy
    )
    @settings(max_examples=100)
    def test_hashtag_count_within_platform_limit(self, metadata, emotion, platform):
        """
        Property 13: Social Pack Platform Compliance (hashtag limit)
        
        For any platform in the social pack generation, the hashtag count
        SHALL not exceed the platform limit.
        
        Validates: Requirements 7.4
        """
        # Get hashtag limit from platform spec
        specs = PLATFORM_SPECS[platform]
        hashtag_limit = specs["hashtag_limit"]
        
        # Generate hashtags for platform
        hashtags = generate_hashtags(metadata, emotion, platform)
        
        # Verify hashtags is a list
        assert isinstance(hashtags, list), "Hashtags should be a list"
        
        # Verify hashtag count does not exceed limit - Requirements 7.4
        assert len(hashtags) <= hashtag_limit, (
            f"Platform '{platform}' hashtag count {len(hashtags)} "
            f"exceeds limit {hashtag_limit}"
        )
        
        # Verify all hashtags have # prefix
        for tag in hashtags:
            assert isinstance(tag, str), f"Hashtag should be string, got {type(tag)}"
            assert tag.startswith("#"), f"Hashtag should start with #, got '{tag}'"
    
    @given(
        frame=valid_frame_strategy(),
        platform=platform_strategy,
        style=st.sampled_from(["vibrant", "dramatic", "clean"])
    )
    @settings(max_examples=100)
    def test_thumbnail_pixel_values_valid(self, frame, platform, style):
        """
        Property 13: Social Pack Platform Compliance (pixel validity)
        
        For any platform thumbnail, pixel values SHALL be in valid range [0, 255].
        
        Validates: Requirements 7.2
        """
        # Create thumbnail with specified style
        thumbnail = create_thumbnail(frame.copy(), platform, style=style)
        
        # Verify pixel values are in valid range
        assert thumbnail.min() >= 0, (
            f"Pixel values should be >= 0, got min: {thumbnail.min()}"
        )
        assert thumbnail.max() <= 255, (
            f"Pixel values should be <= 255, got max: {thumbnail.max()}"
        )
        assert thumbnail.dtype == np.uint8, (
            f"Thumbnail dtype should be uint8, got {thumbnail.dtype}"
        )
    
    @given(
        metadata=video_metadata_strategy(),
        platform=platform_strategy
    )
    @settings(max_examples=100)
    def test_hashtag_generation_without_emotion(self, metadata, platform):
        """
        Property 13: Social Pack Platform Compliance (no emotion analysis)
        
        For any platform, hashtag generation without emotion analysis
        SHALL still respect platform limits.
        
        Validates: Requirements 7.4
        """
        # Get hashtag limit from platform spec
        specs = PLATFORM_SPECS[platform]
        hashtag_limit = specs["hashtag_limit"]
        
        # Generate hashtags without emotion analysis
        hashtags = generate_hashtags(metadata, None, platform)
        
        # Verify hashtag count does not exceed limit
        assert len(hashtags) <= hashtag_limit, (
            f"Platform '{platform}' hashtag count {len(hashtags)} "
            f"exceeds limit {hashtag_limit} (no emotion analysis)"
        )


class TestTitleGenerationCount:
    """
    Property tests for Title Generation Count.
    
    Feature: cinemasense-stabilization, Property 14: Title Generation Count
    Validates: Requirements 7.3
    """
    
    @given(
        metadata=video_metadata_strategy(),
        emotion=emotion_analysis_strategy()
    )
    @settings(max_examples=100)
    def test_exactly_five_title_suggestions_with_emotion(self, metadata, emotion):
        """
        Property 14: Title Generation Count
        
        For any social pack generation, exactly 5 title suggestions SHALL be
        returned, each being a non-empty string.
        
        Validates: Requirements 7.3
        """
        # Generate title suggestions with emotion analysis
        titles = generate_title_suggestions(metadata, emotion)
        
        # Verify exactly 5 titles are returned - Requirements 7.3
        assert len(titles) == 5, (
            f"Expected exactly 5 title suggestions, got {len(titles)}"
        )
        
        # Verify all titles are non-empty strings
        for i, title in enumerate(titles):
            assert isinstance(title, str), (
                f"Title {i+1} should be a string, got {type(title)}"
            )
            assert len(title) > 0, (
                f"Title {i+1} should be non-empty"
            )
    
    @given(
        metadata=video_metadata_strategy()
    )
    @settings(max_examples=100)
    def test_exactly_five_title_suggestions_without_emotion(self, metadata):
        """
        Property 14: Title Generation Count (no emotion analysis)
        
        For any social pack generation without emotion analysis, exactly 5 title
        suggestions SHALL be returned, each being a non-empty string.
        
        Validates: Requirements 7.3
        """
        # Generate title suggestions without emotion analysis
        titles = generate_title_suggestions(metadata, None)
        
        # Verify exactly 5 titles are returned - Requirements 7.3
        assert len(titles) == 5, (
            f"Expected exactly 5 title suggestions, got {len(titles)}"
        )
        
        # Verify all titles are non-empty strings
        for i, title in enumerate(titles):
            assert isinstance(title, str), (
                f"Title {i+1} should be a string, got {type(title)}"
            )
            assert len(title) > 0, (
                f"Title {i+1} should be non-empty"
            )
    
    @given(
        metadata=video_metadata_strategy(),
        emotion=emotion_analysis_strategy()
    )
    @settings(max_examples=100)
    def test_title_suggestions_are_unique(self, metadata, emotion):
        """
        Property 14: Title Generation Count (uniqueness)
        
        For any social pack generation, the 5 title suggestions SHOULD be unique.
        
        Validates: Requirements 7.3
        """
        # Generate title suggestions
        titles = generate_title_suggestions(metadata, emotion)
        
        # Verify titles are unique
        unique_titles = set(titles)
        assert len(unique_titles) == len(titles), (
            f"Title suggestions should be unique, got {len(titles)} titles "
            f"but only {len(unique_titles)} unique"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
