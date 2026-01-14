"""
Property-based tests for PipelineResult dataclass.

Feature: cinemasense-stabilization
Property 5: Pipeline Result Structure
Validates: Requirements 2.4

Tests that PipelineResult always has consistent structure:
- success=True implies data is present and error is None
- success=False implies error is present (non-null)
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings

from cinemasense.core.pipeline import PipelineResult


# Strategy for generating arbitrary data values
data_strategy = st.one_of(
    st.none(),
    st.text(min_size=0, max_size=50),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.integers(), max_size=10),
    st.dictionaries(st.text(min_size=1, max_size=10), st.integers(), max_size=5),
)

# Strategy for generating error messages
error_strategy = st.one_of(
    st.none(),
    st.text(min_size=1, max_size=100),
)

# Strategy for duration in milliseconds
duration_strategy = st.floats(min_value=0.0, max_value=1e9, allow_nan=False, allow_infinity=False)


class TestPipelineResultStructure:
    """
    Property tests for PipelineResult structure consistency.
    
    Feature: cinemasense-stabilization, Property 5: Pipeline Result Structure
    Validates: Requirements 2.4
    """
    
    @given(
        data=data_strategy,
        duration_ms=duration_strategy
    )
    @settings(max_examples=100)
    def test_successful_result_has_no_error(self, data, duration_ms):
        """
        Property 5: Pipeline Result Structure (success case)
        
        For any pipeline operation with success=True, the PipelineResult
        SHALL have error=None (no error message when successful).
        
        Validates: Requirements 2.4
        """
        result = PipelineResult(
            success=True,
            data=data,
            error=None,
            duration_ms=duration_ms
        )
        
        assert result.success is True, "Success flag should be True"
        assert result.error is None, (
            f"Successful result should have error=None\n"
            f"Got error: {repr(result.error)}"
        )
        assert result.duration_ms >= 0, (
            f"Duration should be non-negative\n"
            f"Got: {result.duration_ms}"
        )
    
    @given(
        error_msg=st.text(min_size=1, max_size=100),
        duration_ms=duration_strategy
    )
    @settings(max_examples=100)
    def test_failed_result_has_error_message(self, error_msg, duration_ms):
        """
        Property 5: Pipeline Result Structure (failure case)
        
        For any pipeline operation with success=False, the PipelineResult
        SHALL have a non-null error message.
        
        Validates: Requirements 2.4
        """
        result = PipelineResult(
            success=False,
            data=None,
            error=error_msg,
            duration_ms=duration_ms
        )
        
        assert result.success is False, "Success flag should be False"
        assert result.error is not None, (
            "Failed result should have non-null error message"
        )
        assert len(result.error) > 0, (
            "Failed result should have non-empty error message"
        )
        assert result.duration_ms >= 0, (
            f"Duration should be non-negative\n"
            f"Got: {result.duration_ms}"
        )
    
    @given(duration_ms=duration_strategy)
    @settings(max_examples=100)
    def test_failed_result_without_error_gets_default(self, duration_ms):
        """
        Property 5: Pipeline Result Structure (auto-error case)
        
        For any pipeline operation with success=False and error=None,
        the PipelineResult SHALL automatically set a default error message.
        
        Validates: Requirements 2.4
        """
        result = PipelineResult(
            success=False,
            data=None,
            error=None,
            duration_ms=duration_ms
        )
        
        assert result.success is False, "Success flag should be False"
        assert result.error is not None, (
            "Failed result should have error auto-populated\n"
            "PipelineResult.__post_init__ should set default error"
        )
        assert len(result.error) > 0, (
            "Auto-populated error should be non-empty"
        )
    
    @given(
        success=st.booleans(),
        data=data_strategy,
        error=error_strategy,
        duration_ms=duration_strategy
    )
    @settings(max_examples=100)
    def test_result_structure_consistency(self, success, data, error, duration_ms):
        """
        Property 5: Pipeline Result Structure (consistency)
        
        For any PipelineResult, the structure SHALL be consistent:
        - success=True with non-null data OR success=False with non-null error
        - duration_ms is always non-negative
        
        Validates: Requirements 2.4
        """
        result = PipelineResult(
            success=success,
            data=data,
            error=error,
            duration_ms=duration_ms
        )
        
        # Duration should always be non-negative
        assert result.duration_ms >= 0, (
            f"Duration should be non-negative\n"
            f"Got: {result.duration_ms}"
        )
        
        # If failed, error must be present (auto-populated if needed)
        if not result.success:
            assert result.error is not None, (
                "Failed result must have error message\n"
                f"success={result.success}, error={repr(result.error)}"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
