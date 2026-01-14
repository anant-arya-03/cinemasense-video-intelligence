"""
Property-based tests for Explainable AI Cut Detection.

Feature: cinemasense-stabilization
Property 12: Cut Explanation Completeness
Validates: Requirements 6.1, 6.2, 6.3, 6.4

Tests that for any detected cut, the CutExplanation SHALL have:
- Non-empty primary_reason
- Confidence in range [0, 1]
- cut_type from valid set (hard_cut, dissolve, fade_to_black, fade_to_white)
- secondary_reasons as a list
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from hypothesis import given, strategies as st, settings, assume

from cinemasense.pipeline.explainable_ai import (
    CutExplanation,
    ExplainableAnalysis,
    VALID_CUT_TYPES,
    analyze_cut_reason,
    calculate_confidence,
    generate_analysis_summary,
    detect_cuts_with_explanation,
)


# Strategy for generating valid frame indices
frame_index_strategy = st.integers(min_value=0, max_value=100000)

# Strategy for generating valid timestamps
timestamp_strategy = st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating confidence values (including out-of-range to test clamping)
confidence_strategy = st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False)

# Strategy for generating score values
score_strategy = st.floats(min_value=-1.0, max_value=10.0, allow_nan=False, allow_infinity=False)

# Strategy for generating primary reasons (including empty to test validation)
primary_reason_strategy = st.one_of(
    st.text(min_size=0, max_size=200),
    st.just(""),
    st.just("   "),
    st.just(None),
)

# Strategy for generating secondary reasons list
secondary_reasons_strategy = st.one_of(
    st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=5),
    st.just(None),
    st.just([]),
)

# Strategy for generating cut types (including invalid to test validation)
cut_type_strategy = st.one_of(
    st.sampled_from(list(VALID_CUT_TYPES)),
    st.text(min_size=1, max_size=20),
    st.just("invalid_cut"),
)


@st.composite
def cut_explanation_strategy(draw):
    """Generate CutExplanation instances with various inputs."""
    return CutExplanation(
        frame_index=draw(frame_index_strategy),
        timestamp=draw(timestamp_strategy),
        confidence=draw(confidence_strategy),
        primary_reason=draw(primary_reason_strategy) or "",
        secondary_reasons=draw(secondary_reasons_strategy) or [],
        visual_change_score=draw(score_strategy),
        color_change_score=draw(score_strategy),
        motion_discontinuity=draw(score_strategy),
        cut_type=draw(cut_type_strategy),
    )


@st.composite
def valid_bgr_frame_strategy(draw):
    """Generate a valid BGR frame with random pixel values."""
    height = draw(st.integers(min_value=10, max_value=100))
    width = draw(st.integers(min_value=10, max_value=100))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


@st.composite
def frame_pair_strategy(draw):
    """Generate a pair of BGR frames for cut analysis."""
    height = draw(st.integers(min_value=10, max_value=100))
    width = draw(st.integers(min_value=10, max_value=100))
    seed1 = draw(st.integers(min_value=0, max_value=2**31 - 1))
    seed2 = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng1 = np.random.default_rng(seed1)
    rng2 = np.random.default_rng(seed2)
    frame_before = rng1.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    frame_after = rng2.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return frame_before, frame_after


class TestCutExplanationCompleteness:
    """
    Property tests for Cut Explanation Completeness.
    
    Feature: cinemasense-stabilization, Property 12: Cut Explanation Completeness
    Validates: Requirements 6.1, 6.2, 6.3, 6.4
    
    For any detected cut, the CutExplanation SHALL have:
    - Non-empty primary_reason
    - Confidence in range [0, 1]
    - cut_type from valid set
    - secondary_reasons as a list
    """
    
    @given(cut=cut_explanation_strategy())
    @settings(max_examples=100)
    def test_cut_explanation_has_non_empty_primary_reason(self, cut):
        """
        Property 12: Cut Explanation Primary Reason Non-Empty
        
        For any CutExplanation, the primary_reason SHALL be a non-empty string.
        
        Validates: Requirements 6.1
        """
        assert isinstance(cut.primary_reason, str), (
            f"primary_reason should be a string, got {type(cut.primary_reason)}"
        )
        assert cut.primary_reason.strip() != "", (
            f"primary_reason should be non-empty after stripping whitespace, "
            f"got '{cut.primary_reason}'"
        )
    
    @given(cut=cut_explanation_strategy())
    @settings(max_examples=100)
    def test_cut_explanation_confidence_in_valid_range(self, cut):
        """
        Property 12: Cut Explanation Confidence Range
        
        For any CutExplanation, the confidence SHALL be in range [0, 1].
        
        Validates: Requirements 6.2
        """
        assert 0.0 <= cut.confidence <= 1.0, (
            f"confidence should be in range [0, 1], got {cut.confidence}"
        )
    
    @given(cut=cut_explanation_strategy())
    @settings(max_examples=100)
    def test_cut_explanation_cut_type_is_valid(self, cut):
        """
        Property 12: Cut Explanation Cut Type Validity
        
        For any CutExplanation, the cut_type SHALL be from the valid set
        (hard_cut, dissolve, fade_to_black, fade_to_white).
        
        Validates: Requirements 6.3
        """
        assert cut.cut_type in VALID_CUT_TYPES, (
            f"cut_type should be one of {VALID_CUT_TYPES}, got '{cut.cut_type}'"
        )
    
    @given(cut=cut_explanation_strategy())
    @settings(max_examples=100)
    def test_cut_explanation_secondary_reasons_is_list(self, cut):
        """
        Property 12: Cut Explanation Secondary Reasons Is List
        
        For any CutExplanation, the secondary_reasons SHALL be a list.
        
        Validates: Requirements 6.4
        """
        assert isinstance(cut.secondary_reasons, list), (
            f"secondary_reasons should be a list, got {type(cut.secondary_reasons)}"
        )
    
    @given(cut=cut_explanation_strategy())
    @settings(max_examples=100)
    def test_cut_explanation_scores_are_non_negative(self, cut):
        """
        Property 12: Cut Explanation Scores Non-Negative
        
        For any CutExplanation, all score fields SHALL be non-negative.
        
        Validates: Requirements 6.1, 6.2
        """
        assert cut.visual_change_score >= 0.0, (
            f"visual_change_score should be >= 0, got {cut.visual_change_score}"
        )
        assert cut.color_change_score >= 0.0, (
            f"color_change_score should be >= 0, got {cut.color_change_score}"
        )
        assert cut.motion_discontinuity >= 0.0, (
            f"motion_discontinuity should be >= 0, got {cut.motion_discontinuity}"
        )


class TestAnalyzeCutReason:
    """
    Property tests for analyze_cut_reason function.
    
    Feature: cinemasense-stabilization, Property 12: Cut Explanation Completeness
    Validates: Requirements 6.1, 6.4
    """
    
    @given(
        frames=frame_pair_strategy(),
        hist_diff=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        motion_diff=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_analyze_cut_reason_returns_non_empty_primary_reason(self, frames, hist_diff, motion_diff):
        """
        Property 12: Analyze Cut Reason Returns Non-Empty Primary Reason
        
        For any frame pair and difference scores, analyze_cut_reason SHALL
        return a non-empty primary_reason string.
        
        Validates: Requirements 6.1
        """
        frame_before, frame_after = frames
        primary_reason, secondary_reasons, cut_type = analyze_cut_reason(
            frame_before, frame_after, hist_diff, motion_diff
        )
        
        assert isinstance(primary_reason, str), (
            f"primary_reason should be a string, got {type(primary_reason)}"
        )
        assert primary_reason.strip() != "", (
            f"primary_reason should be non-empty, got '{primary_reason}'"
        )
    
    @given(
        frames=frame_pair_strategy(),
        hist_diff=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        motion_diff=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_analyze_cut_reason_returns_valid_cut_type(self, frames, hist_diff, motion_diff):
        """
        Property 12: Analyze Cut Reason Returns Valid Cut Type
        
        For any frame pair and difference scores, analyze_cut_reason SHALL
        return a cut_type from the valid set.
        
        Validates: Requirements 6.3
        """
        frame_before, frame_after = frames
        primary_reason, secondary_reasons, cut_type = analyze_cut_reason(
            frame_before, frame_after, hist_diff, motion_diff
        )
        
        assert cut_type in VALID_CUT_TYPES, (
            f"cut_type should be one of {VALID_CUT_TYPES}, got '{cut_type}'"
        )
    
    @given(
        frames=frame_pair_strategy(),
        hist_diff=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        motion_diff=st.floats(min_value=0.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_analyze_cut_reason_returns_list_of_secondary_reasons(self, frames, hist_diff, motion_diff):
        """
        Property 12: Analyze Cut Reason Returns List of Secondary Reasons
        
        For any frame pair and difference scores, analyze_cut_reason SHALL
        return secondary_reasons as a list.
        
        Validates: Requirements 6.4
        """
        frame_before, frame_after = frames
        primary_reason, secondary_reasons, cut_type = analyze_cut_reason(
            frame_before, frame_after, hist_diff, motion_diff
        )
        
        assert isinstance(secondary_reasons, list), (
            f"secondary_reasons should be a list, got {type(secondary_reasons)}"
        )


class TestCalculateConfidence:
    """
    Property tests for calculate_confidence function.
    
    Feature: cinemasense-stabilization, Property 12: Cut Explanation Completeness
    Validates: Requirements 6.2
    """
    
    @given(
        hist_diff=st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        motion_diff=st.floats(min_value=-1.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        edge_diff=st.floats(min_value=-100.0, max_value=500.0, allow_nan=False, allow_infinity=False),
        brightness_diff=st.floats(min_value=-100.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_calculate_confidence_returns_value_in_valid_range(
        self, hist_diff, motion_diff, edge_diff, brightness_diff
    ):
        """
        Property 12: Calculate Confidence Returns Value in [0, 1]
        
        For any input scores (including out-of-range values), calculate_confidence
        SHALL return a confidence value clamped to [0, 1].
        
        Validates: Requirements 6.2
        """
        confidence = calculate_confidence(hist_diff, motion_diff, edge_diff, brightness_diff)
        
        assert 0.0 <= confidence <= 1.0, (
            f"confidence should be in range [0, 1], got {confidence}\n"
            f"Inputs: hist_diff={hist_diff}, motion_diff={motion_diff}, "
            f"edge_diff={edge_diff}, brightness_diff={brightness_diff}"
        )


class TestExplainableAnalysis:
    """
    Property tests for ExplainableAnalysis dataclass.
    
    Feature: cinemasense-stabilization, Property 12: Cut Explanation Completeness
    Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
    """
    
    @given(
        num_cuts=st.integers(min_value=0, max_value=20),
        avg_confidence=st.floats(min_value=-1.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        summary=st.one_of(st.text(min_size=0, max_size=500), st.just(""), st.just("   ")),
    )
    @settings(max_examples=100)
    def test_explainable_analysis_has_valid_structure(self, num_cuts, avg_confidence, summary):
        """
        Property 12: Explainable Analysis Structure Validity
        
        For any ExplainableAnalysis, the structure SHALL be valid with:
        - total_cuts matching cuts list length
        - avg_confidence in [0, 1]
        - non-empty explanation_summary
        
        Validates: Requirements 6.5
        """
        # Create valid cuts
        cuts = []
        cut_types = {}
        for i in range(num_cuts):
            cut = CutExplanation(
                frame_index=i * 30,
                timestamp=i * 1.0,
                confidence=0.8,
                primary_reason="Test reason",
                secondary_reasons=[],
                visual_change_score=0.5,
                color_change_score=0.3,
                motion_discontinuity=0.2,
                cut_type="hard_cut",
            )
            cuts.append(cut)
            cut_types["hard_cut"] = cut_types.get("hard_cut", 0) + 1
        
        analysis = ExplainableAnalysis(
            cuts=cuts,
            total_cuts=num_cuts,
            avg_confidence=avg_confidence,
            cut_type_distribution=cut_types,
            explanation_summary=summary,
        )
        
        # Verify total_cuts matches cuts list
        assert analysis.total_cuts == len(analysis.cuts), (
            f"total_cuts should match cuts list length, "
            f"got total_cuts={analysis.total_cuts}, len(cuts)={len(analysis.cuts)}"
        )
        
        # Verify avg_confidence is in [0, 1]
        assert 0.0 <= analysis.avg_confidence <= 1.0, (
            f"avg_confidence should be in [0, 1], got {analysis.avg_confidence}"
        )
        
        # Verify explanation_summary is non-empty
        assert analysis.explanation_summary.strip() != "", (
            f"explanation_summary should be non-empty, got '{analysis.explanation_summary}'"
        )


class TestGenerateAnalysisSummary:
    """
    Property tests for generate_analysis_summary function.
    
    Feature: cinemasense-stabilization, Property 12: Cut Explanation Completeness
    Validates: Requirements 6.5
    """
    
    @given(num_cuts=st.integers(min_value=0, max_value=20))
    @settings(max_examples=100)
    def test_generate_analysis_summary_returns_non_empty_string(self, num_cuts):
        """
        Property 12: Generate Analysis Summary Returns Non-Empty String
        
        For any list of cuts, generate_analysis_summary SHALL return
        a non-empty string.
        
        Validates: Requirements 6.5
        """
        cuts = []
        cut_types = {}
        for i in range(num_cuts):
            cut = CutExplanation(
                frame_index=i * 30,
                timestamp=i * 1.0,
                confidence=0.8,
                primary_reason="Test reason",
                secondary_reasons=[],
                visual_change_score=0.5,
                color_change_score=0.3,
                motion_discontinuity=0.2,
                cut_type="hard_cut",
            )
            cuts.append(cut)
            cut_types["hard_cut"] = cut_types.get("hard_cut", 0) + 1
        
        summary = generate_analysis_summary(cuts, cut_types)
        
        assert isinstance(summary, str), (
            f"summary should be a string, got {type(summary)}"
        )
        assert summary.strip() != "", (
            f"summary should be non-empty, got '{summary}'"
        )


class TestCutDetectionWithRealVideo:
    """
    Property tests for cut detection with real video file.
    
    Feature: cinemasense-stabilization, Property 12: Cut Explanation Completeness
    Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
    """
    
    @settings(max_examples=10, deadline=None)
    @given(
        sample_rate=st.integers(min_value=1, max_value=10),
        threshold=st.floats(min_value=0.3, max_value=0.9, allow_nan=False, allow_infinity=False),
    )
    def test_full_cut_detection_output_completeness(self, sample_rate, threshold):
        """
        Property 12: Full Cut Detection Output Completeness
        
        For any video analysis with valid parameters, all detected cuts
        SHALL have complete explanations with valid fields.
        
        Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
        """
        test_video_path = ROOT / "data" / "test" / "test_video.mp4"
        
        # Skip if test video doesn't exist
        assume(test_video_path.exists())
        
        result = detect_cuts_with_explanation(
            str(test_video_path),
            sample_every_n_frames=sample_rate,
            threshold=threshold,
        )
        
        # Verify result is ExplainableAnalysis
        assert isinstance(result, ExplainableAnalysis), (
            f"Result should be ExplainableAnalysis, got {type(result)}"
        )
        
        # Verify total_cuts matches cuts list
        assert result.total_cuts == len(result.cuts), (
            f"total_cuts should match cuts list length"
        )
        
        # Verify avg_confidence is in [0, 1]
        assert 0.0 <= result.avg_confidence <= 1.0, (
            f"avg_confidence should be in [0, 1], got {result.avg_confidence}"
        )
        
        # Verify explanation_summary is non-empty
        assert result.explanation_summary.strip() != "", (
            f"explanation_summary should be non-empty"
        )
        
        # Verify each cut has complete explanation
        for i, cut in enumerate(result.cuts):
            # Non-empty primary_reason
            assert cut.primary_reason.strip() != "", (
                f"Cut {i}: primary_reason should be non-empty"
            )
            
            # Confidence in [0, 1]
            assert 0.0 <= cut.confidence <= 1.0, (
                f"Cut {i}: confidence should be in [0, 1], got {cut.confidence}"
            )
            
            # Valid cut_type
            assert cut.cut_type in VALID_CUT_TYPES, (
                f"Cut {i}: cut_type should be valid, got '{cut.cut_type}'"
            )
            
            # secondary_reasons is a list
            assert isinstance(cut.secondary_reasons, list), (
                f"Cut {i}: secondary_reasons should be a list"
            )
            
            # Non-negative scores
            assert cut.visual_change_score >= 0.0, (
                f"Cut {i}: visual_change_score should be >= 0"
            )
            assert cut.color_change_score >= 0.0, (
                f"Cut {i}: color_change_score should be >= 0"
            )
            assert cut.motion_discontinuity >= 0.0, (
                f"Cut {i}: motion_discontinuity should be >= 0"
            )
            
            # Valid timestamp
            assert cut.timestamp >= 0.0, (
                f"Cut {i}: timestamp should be >= 0"
            )
            
            # Valid frame_index
            assert cut.frame_index >= 0, (
                f"Cut {i}: frame_index should be >= 0"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
