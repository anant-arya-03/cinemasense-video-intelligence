"""
Property-based tests for Report Structure Validity.

Feature: cinemasense-stabilization
Property 19: Report Structure Validity
Validates: Requirements 14.1, 14.2, 14.5

Tests that for any generated report, the JSON SHALL be valid and contain:
generated_at timestamp, video_name, metadata object, and all analysis
sections that were performed.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings, assume

from cinemasense.services.report import ReportGenerator, ReportValidationError


# Strategy for generating valid video names
video_name_strategy = st.text(
    min_size=1,
    max_size=100,
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S'))
).filter(lambda x: x.strip())

# Strategy for generating valid metadata
metadata_strategy = st.fixed_dictionaries({
    "name": st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    "fps": st.floats(min_value=1.0, max_value=120.0, allow_nan=False, allow_infinity=False),
    "frame_count": st.integers(min_value=1, max_value=1000000),
    "width": st.integers(min_value=1, max_value=8192),
    "height": st.integers(min_value=1, max_value=8192),
    "duration_s": st.floats(min_value=0.1, max_value=36000.0, allow_nan=False, allow_infinity=False),
    "format": st.sampled_from(["mp4", "avi", "mov", "mkv", "webm"]),
    "file_size_bytes": st.integers(min_value=1, max_value=10_000_000_000),
    "path": st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
})

# Strategy for generating cut analysis data
cuts_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        "total_cuts": st.integers(min_value=0, max_value=1000),
        "avg_confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "cut_type_distribution": st.dictionaries(
            keys=st.sampled_from(["hard_cut", "dissolve", "fade_to_black", "fade_to_white"]),
            values=st.integers(min_value=0, max_value=100),
            max_size=4
        ),
        "cuts": st.lists(
            st.fixed_dictionaries({
                "frame_index": st.integers(min_value=0, max_value=100000),
                "timestamp": st.floats(min_value=0.0, max_value=36000.0, allow_nan=False, allow_infinity=False),
                "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            }),
            max_size=50
        )
    })
)

# Strategy for generating emotion analysis data
emotion_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        "overall_score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "rhythm_pattern": st.sampled_from(["Dynamic", "Steady", "Building Crescendo", "Falling Action", "Chaotic"]),
        "emotion_distribution": st.fixed_dictionaries({
            "Joy": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            "Tension": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            "Calm": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        }),
        "confidence": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        "peak_moments": st.lists(
            st.fixed_dictionaries({
                "timestamp": st.floats(min_value=0.0, max_value=36000.0, allow_nan=False, allow_infinity=False),
                "emotion": st.sampled_from(["Joy", "Tension", "Calm", "Melancholy", "Energy", "Mystery"]),
            }),
            max_size=10
        )
    })
)

# Strategy for generating keyframes data
keyframes_strategy = st.one_of(
    st.none(),
    st.lists(
        st.fixed_dictionaries({
            "frame_index": st.integers(min_value=0, max_value=100000),
            "timestamp": st.floats(min_value=0.0, max_value=36000.0, allow_nan=False, allow_infinity=False),
            "score": st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        }),
        max_size=20
    )
)

# Strategy for generating social pack data
social_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        "thumbnail_path": st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
        "title_suggestions": st.lists(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()), min_size=5, max_size=5),
        "hashtags": st.lists(st.text(min_size=1, max_size=30).filter(lambda x: x.strip()), max_size=30),
        "caption": st.text(min_size=1, max_size=500).filter(lambda x: x.strip()),
        "platforms": st.lists(st.sampled_from(["YouTube", "Instagram", "TikTok", "Twitter"]), min_size=1, max_size=4),
    })
)

# Strategy for generating multiverse data
multiverse_strategy = st.one_of(
    st.none(),
    st.fixed_dictionaries({
        "styles_generated": st.lists(
            st.sampled_from(["Romantic", "Thriller", "Viral", "Anime", "Noir", "Cyberpunk"]),
            min_size=1,
            max_size=6
        ),
        "previews": st.lists(
            st.fixed_dictionaries({
                "style": st.sampled_from(["Romantic", "Thriller", "Viral", "Anime", "Noir", "Cyberpunk"]),
                "path": st.text(min_size=1, max_size=200).filter(lambda x: x.strip()),
            }),
            max_size=18
        )
    })
)


class TestReportStructureValidity:
    """
    Property tests for report structure validity.
    
    Feature: cinemasense-stabilization, Property 19: Report Structure Validity
    Validates: Requirements 14.1, 14.2, 14.5
    """
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy,
        cuts=cuts_strategy,
        emotion=emotion_strategy,
        keyframes=keyframes_strategy,
        social=social_strategy,
        multiverse=multiverse_strategy
    )
    @settings(max_examples=100)
    def test_generated_report_is_valid_json(
        self,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]],
        keyframes: Optional[List[Dict[str, Any]]],
        social: Optional[Dict[str, Any]],
        multiverse: Optional[Dict[str, Any]]
    ):
        """
        Property 19: Report Structure Validity (JSON validity)
        
        For any generated report, the output SHALL be valid JSON.
        
        Validates: Requirements 14.1
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        # Verify report can be serialized to JSON and back
        json_str = json.dumps(report)
        parsed = json.loads(json_str)
        
        assert isinstance(parsed, dict), "Parsed JSON should be a dictionary"
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy,
        cuts=cuts_strategy,
        emotion=emotion_strategy,
        keyframes=keyframes_strategy,
        social=social_strategy,
        multiverse=multiverse_strategy
    )
    @settings(max_examples=100)
    def test_generated_report_contains_timestamp(
        self,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]],
        keyframes: Optional[List[Dict[str, Any]]],
        social: Optional[Dict[str, Any]],
        multiverse: Optional[Dict[str, Any]]
    ):
        """
        Property 19: Report Structure Validity (timestamp)
        
        For any generated report, the output SHALL contain a valid
        generated_at timestamp in ISO format.
        
        Validates: Requirements 14.5
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        # Verify generated_at exists and is a valid ISO timestamp
        assert "generated_at" in report, "Report must contain 'generated_at' field"
        assert isinstance(report["generated_at"], str), "generated_at must be a string"
        
        # Verify it's a valid ISO timestamp
        try:
            datetime.fromisoformat(report["generated_at"])
        except ValueError:
            raise AssertionError(
                f"generated_at '{report['generated_at']}' is not a valid ISO timestamp"
            )
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy,
        cuts=cuts_strategy,
        emotion=emotion_strategy,
        keyframes=keyframes_strategy,
        social=social_strategy,
        multiverse=multiverse_strategy
    )
    @settings(max_examples=100)
    def test_generated_report_contains_video_name(
        self,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]],
        keyframes: Optional[List[Dict[str, Any]]],
        social: Optional[Dict[str, Any]],
        multiverse: Optional[Dict[str, Any]]
    ):
        """
        Property 19: Report Structure Validity (video_name)
        
        For any generated report, the output SHALL contain the video_name.
        
        Validates: Requirements 14.1, 14.2
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        assert "video_name" in report, "Report must contain 'video_name' field"
        assert report["video_name"] == video_name, (
            f"Report video_name '{report['video_name']}' should match input '{video_name}'"
        )
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy,
        cuts=cuts_strategy,
        emotion=emotion_strategy,
        keyframes=keyframes_strategy,
        social=social_strategy,
        multiverse=multiverse_strategy
    )
    @settings(max_examples=100)
    def test_generated_report_contains_metadata_object(
        self,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]],
        keyframes: Optional[List[Dict[str, Any]]],
        social: Optional[Dict[str, Any]],
        multiverse: Optional[Dict[str, Any]]
    ):
        """
        Property 19: Report Structure Validity (metadata object)
        
        For any generated report, the output SHALL contain a metadata object.
        
        Validates: Requirements 14.2
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        assert "metadata" in report, "Report must contain 'metadata' field"
        assert isinstance(report["metadata"], dict), "metadata must be a dictionary"
        
        # Verify required metadata fields are present
        required_fields = {"name", "fps", "frame_count", "width", "height", "duration_s"}
        actual_fields = set(report["metadata"].keys())
        missing = required_fields - actual_fields
        
        assert not missing, f"Metadata missing required fields: {missing}"
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy,
        cuts=cuts_strategy,
        emotion=emotion_strategy,
        keyframes=keyframes_strategy,
        social=social_strategy,
        multiverse=multiverse_strategy
    )
    @settings(max_examples=100)
    def test_generated_report_includes_performed_analysis_sections(
        self,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]],
        keyframes: Optional[List[Dict[str, Any]]],
        social: Optional[Dict[str, Any]],
        multiverse: Optional[Dict[str, Any]]
    ):
        """
        Property 19: Report Structure Validity (analysis sections)
        
        For any generated report, the output SHALL contain all analysis
        sections that were performed (non-None inputs).
        
        Validates: Requirements 14.2
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        # Verify cuts section
        if cuts is not None:
            assert "cuts" in report, "Report should contain 'cuts' when cuts analysis was performed"
            assert report["cuts"] is not None, "cuts should not be None when analysis was performed"
        
        # Verify emotion section
        if emotion is not None:
            assert "emotion" in report, "Report should contain 'emotion' when emotion analysis was performed"
            assert report["emotion"] is not None, "emotion should not be None when analysis was performed"
        
        # Verify keyframes section
        if keyframes is not None:
            assert "keyframes_count" in report, "Report should contain 'keyframes_count'"
            assert report["keyframes_count"] == len(keyframes), (
                f"keyframes_count {report['keyframes_count']} should match input length {len(keyframes)}"
            )
        
        # Verify social section
        if social is not None:
            assert "social" in report, "Report should contain 'social' when social pack was generated"
            assert report["social"] is not None, "social should not be None when analysis was performed"
        
        # Verify multiverse section
        if multiverse is not None:
            assert "multiverse" in report, "Report should contain 'multiverse' when multiverse was generated"
            assert report["multiverse"] is not None, "multiverse should not be None when analysis was performed"
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy,
        cuts=cuts_strategy,
        emotion=emotion_strategy,
        keyframes=keyframes_strategy,
        social=social_strategy,
        multiverse=multiverse_strategy
    )
    @settings(max_examples=100)
    def test_generated_report_passes_validation(
        self,
        video_name: str,
        metadata: Dict[str, Any],
        cuts: Optional[Dict[str, Any]],
        emotion: Optional[Dict[str, Any]],
        keyframes: Optional[List[Dict[str, Any]]],
        social: Optional[Dict[str, Any]],
        multiverse: Optional[Dict[str, Any]]
    ):
        """
        Property 19: Report Structure Validity (validation)
        
        For any generated report, the report SHALL pass the validation check.
        
        Validates: Requirements 14.1, 14.5
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        # Validation should pass without raising an exception
        try:
            result = ReportGenerator.validate(report)
            assert result is True, "Validation should return True for valid reports"
        except ReportValidationError as e:
            raise AssertionError(f"Report validation failed: {e}")
    
    @given(
        video_name=video_name_strategy,
        metadata=metadata_strategy
    )
    @settings(max_examples=100)
    def test_report_json_string_conversion(
        self,
        video_name: str,
        metadata: Dict[str, Any]
    ):
        """
        Property 19: Report Structure Validity (JSON string conversion)
        
        For any generated report, converting to JSON string and back
        SHALL preserve the report structure.
        
        Validates: Requirements 14.1
        """
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata
        )
        
        # Convert to JSON string
        json_str = ReportGenerator.to_json_string(report)
        
        # Parse back
        parsed = json.loads(json_str)
        
        # Verify key fields are preserved
        assert parsed["video_name"] == report["video_name"]
        assert parsed["generated_at"] == report["generated_at"]
        assert parsed["metadata"] == report["metadata"]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
