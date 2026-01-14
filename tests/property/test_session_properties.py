"""
Property-based tests for SessionManager initialization.

Feature: cinemasense-stabilization
Property 1: Session State Initialization Completeness
Validates: Requirements 1.1

Tests that after SessionManager.initialize() is called, all required
session state keys exist with valid default values.
"""

import sys
from pathlib import Path
from dataclasses import fields
from unittest.mock import MagicMock, patch
from typing import Any, Dict

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings

from cinemasense.core.session import SessionManager, SessionState


class MockSessionState(dict):
    """Mock for Streamlit's session_state that behaves like a dict."""
    pass


class TestSessionStateInitializationCompleteness:
    """
    Property tests for SessionManager initialization completeness.
    
    Feature: cinemasense-stabilization, Property 1: Session State Initialization Completeness
    Validates: Requirements 1.1
    """
    
    def _get_expected_fields(self) -> Dict[str, Any]:
        """Get all expected fields from SessionState dataclass with defaults."""
        default_state = SessionState()
        return {f.name: getattr(default_state, f.name) for f in fields(SessionState)}
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=100)
    def test_all_session_keys_exist_after_initialize(self, call_count: int):
        """
        Property 1: Session State Initialization Completeness
        
        For any application startup, after SessionManager.initialize() is called,
        all required session state keys SHALL exist with valid default values.
        
        Validates: Requirements 1.1
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Call initialize multiple times (should be idempotent)
            for _ in range(call_count):
                SessionManager.initialize()
            
            # Get expected fields from SessionState dataclass
            expected_fields = self._get_expected_fields()
            
            # Verify all expected keys exist with prefix
            for field_name in expected_fields.keys():
                prefixed_key = f"{SessionManager.PREFIX}{field_name}"
                assert prefixed_key in mock_session_state, (
                    f"Session key '{prefixed_key}' should exist after initialize()\n"
                    f"Missing from session state after {call_count} initialize() calls"
                )
    
    @given(st.booleans())
    @settings(max_examples=100)
    def test_session_keys_have_valid_default_types(self, _: bool):
        """
        Property 1: Session State Initialization Completeness (type validation)
        
        For any application startup, after SessionManager.initialize() is called,
        all session state values SHALL have the correct default types.
        
        Validates: Requirements 1.1
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            SessionManager.initialize()
            
            # Verify types of key fields
            prefix = SessionManager.PREFIX
            
            # Boolean fields should be bool
            assert isinstance(mock_session_state.get(f"{prefix}gesture_enabled"), bool), (
                "gesture_enabled should be a boolean"
            )
            assert isinstance(mock_session_state.get(f"{prefix}analysis_running"), bool), (
                "analysis_running should be a boolean"
            )
            
            # String fields should be str
            assert isinstance(mock_session_state.get(f"{prefix}current_page"), str), (
                "current_page should be a string"
            )
            assert isinstance(mock_session_state.get(f"{prefix}theme"), str), (
                "theme should be a string"
            )
            
            # Float fields should be float or int (numeric)
            progress = mock_session_state.get(f"{prefix}processing_progress")
            assert isinstance(progress, (int, float)), (
                "processing_progress should be numeric"
            )
            
            # List fields should be list
            assert isinstance(mock_session_state.get(f"{prefix}error_log"), list), (
                "error_log should be a list"
            )
            
            # Timestamp fields should be strings
            assert isinstance(mock_session_state.get(f"{prefix}created_at"), str), (
                "created_at should be a string"
            )
            assert isinstance(mock_session_state.get(f"{prefix}last_updated"), str), (
                "last_updated should be a string"
            )
    
    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=100)
    def test_initialize_is_idempotent(self, call_count: int):
        """
        Property 1: Session State Initialization Completeness (idempotency)
        
        For any number of initialize() calls, the session state SHALL
        contain the same keys and not overwrite existing values.
        
        Validates: Requirements 1.1
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # First initialization
            SessionManager.initialize()
            
            # Capture initial values
            initial_keys = set(mock_session_state.keys())
            initial_created_at = mock_session_state.get(f"{SessionManager.PREFIX}created_at")
            
            # Call initialize multiple more times
            for _ in range(call_count):
                SessionManager.initialize()
            
            # Keys should be the same
            final_keys = set(mock_session_state.keys())
            assert initial_keys == final_keys, (
                f"Keys changed after multiple initialize() calls\n"
                f"Initial: {initial_keys}\n"
                f"Final: {final_keys}"
            )
            
            # created_at should not change (idempotent)
            final_created_at = mock_session_state.get(f"{SessionManager.PREFIX}created_at")
            assert initial_created_at == final_created_at, (
                f"created_at changed after multiple initialize() calls\n"
                f"Initial: {initial_created_at}\n"
                f"Final: {final_created_at}"
            )
    
    @given(st.sampled_from(list(SessionState.__dataclass_fields__.keys())))
    @settings(max_examples=100)
    def test_each_field_initialized_with_default(self, field_name: str):
        """
        Property 1: Session State Initialization Completeness (field defaults)
        
        For any field in SessionState, after initialize() is called,
        the field SHALL exist with its defined default value.
        
        Validates: Requirements 1.1
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            SessionManager.initialize()
            
            # Get expected default from SessionState
            default_state = SessionState()
            expected_default = getattr(default_state, field_name)
            
            # Get actual value from session state
            prefixed_key = f"{SessionManager.PREFIX}{field_name}"
            actual_value = mock_session_state.get(prefixed_key)
            
            # For timestamps, just verify they exist and are strings
            if field_name in ("created_at", "last_updated"):
                assert isinstance(actual_value, str), (
                    f"Field '{field_name}' should be a string timestamp"
                )
                assert len(actual_value) > 0, (
                    f"Field '{field_name}' should have a non-empty timestamp"
                )
            else:
                # For other fields, verify exact default match
                assert actual_value == expected_default, (
                    f"Field '{field_name}' has wrong default value\n"
                    f"Expected: {repr(expected_default)}\n"
                    f"Actual: {repr(actual_value)}"
                )


class TestSessionStateResetOnUpload:
    """
    Property tests for session state reset on video upload.
    
    Feature: cinemasense-stabilization, Property 2: Session State Reset on Upload
    Validates: Requirements 1.2
    """
    
    @given(
        video_path=st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        video_name=st.text(min_size=1, max_size=50).filter(lambda x: x.strip()),
    )
    @settings(max_examples=100)
    def test_analysis_keys_reset_on_upload(self, video_path: str, video_name: str):
        """
        Property 2: Session State Reset on Upload
        
        For any video upload operation, the session state SHALL have 
        analysis-related keys reset to None while preserving the new 
        video path and metadata.
        
        Validates: Requirements 1.2
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Initialize session state
            SessionManager.initialize()
            
            # Simulate existing analysis data before upload
            analysis_keys_to_set = {
                "analysis": {"cuts": [1, 2, 3], "score": 0.85},
                "keyframes": [{"frame": 1}, {"frame": 2}],
                "emotion": {"joy": 0.5, "tension": 0.3},
                "social": {"hashtags": ["#video", "#test"]},
                "multiverse": {"styles": ["romantic", "thriller"]},
            }
            
            for key, value in analysis_keys_to_set.items():
                prefixed_key = f"{SessionManager.PREFIX}{key}"
                mock_session_state[prefixed_key] = value
            
            # Verify analysis data exists before clear
            for key in analysis_keys_to_set:
                prefixed_key = f"{SessionManager.PREFIX}{key}"
                assert mock_session_state.get(prefixed_key) is not None, (
                    f"Analysis key '{key}' should have data before clear_analysis()"
                )
            
            # Simulate video upload: set new video path/name, then clear analysis
            SessionManager.set("video_path", video_path)
            SessionManager.set("video_name", video_name)
            SessionManager.clear_analysis()
            
            # Verify analysis keys are reset to None
            for key in ["analysis", "keyframes", "emotion", "social", "multiverse"]:
                prefixed_key = f"{SessionManager.PREFIX}{key}"
                assert mock_session_state.get(prefixed_key) is None, (
                    f"Analysis key '{key}' should be None after clear_analysis()\n"
                    f"Actual value: {mock_session_state.get(prefixed_key)}"
                )
            
            # Verify video path and name are preserved
            assert mock_session_state.get(f"{SessionManager.PREFIX}video_path") == video_path, (
                f"video_path should be preserved after clear_analysis()\n"
                f"Expected: {video_path}\n"
                f"Actual: {mock_session_state.get(f'{SessionManager.PREFIX}video_path')}"
            )
            assert mock_session_state.get(f"{SessionManager.PREFIX}video_name") == video_name, (
                f"video_name should be preserved after clear_analysis()\n"
                f"Expected: {video_name}\n"
                f"Actual: {mock_session_state.get(f'{SessionManager.PREFIX}video_name')}"
            )
    
    @given(
        metadata=st.fixed_dictionaries({
            "fps": st.floats(min_value=1.0, max_value=120.0, allow_nan=False),
            "width": st.integers(min_value=1, max_value=7680),
            "height": st.integers(min_value=1, max_value=4320),
            "duration": st.floats(min_value=0.1, max_value=36000.0, allow_nan=False),
        })
    )
    @settings(max_examples=100)
    def test_metadata_preserved_on_upload(self, metadata: Dict[str, Any]):
        """
        Property 2: Session State Reset on Upload (metadata preservation)
        
        For any video upload operation, the video metadata SHALL be 
        preserved after clearing analysis results.
        
        Validates: Requirements 1.2
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Initialize session state
            SessionManager.initialize()
            
            # Set metadata before clear
            SessionManager.set("metadata", metadata)
            
            # Set some analysis data
            SessionManager.set("analysis", {"some": "data"})
            SessionManager.set("emotion", {"joy": 0.5})
            
            # Clear analysis (simulating new upload)
            SessionManager.clear_analysis()
            
            # Verify metadata is preserved
            preserved_metadata = mock_session_state.get(f"{SessionManager.PREFIX}metadata")
            assert preserved_metadata == metadata, (
                f"Metadata should be preserved after clear_analysis()\n"
                f"Expected: {metadata}\n"
                f"Actual: {preserved_metadata}"
            )
    
    @given(
        processing_states=st.lists(
            st.tuples(
                st.booleans(),  # analysis_running
                st.floats(min_value=0.0, max_value=1.0, allow_nan=False)  # processing_progress
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100)
    def test_processing_state_reset_on_upload(self, processing_states):
        """
        Property 2: Session State Reset on Upload (processing state)
        
        For any video upload operation, the processing state (analysis_running, 
        processing_progress) SHALL be reset to default values.
        
        Validates: Requirements 1.2
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Initialize session state
            SessionManager.initialize()
            
            # Test with various processing states
            for analysis_running, processing_progress in processing_states:
                # Set processing state
                mock_session_state[f"{SessionManager.PREFIX}analysis_running"] = analysis_running
                mock_session_state[f"{SessionManager.PREFIX}processing_progress"] = processing_progress
                
                # Clear analysis (simulating new upload)
                SessionManager.clear_analysis()
                
                # Verify processing state is reset
                assert mock_session_state.get(f"{SessionManager.PREFIX}analysis_running") == False, (
                    f"analysis_running should be False after clear_analysis()\n"
                    f"Was: {analysis_running}"
                )
                assert mock_session_state.get(f"{SessionManager.PREFIX}processing_progress") == 0.0, (
                    f"processing_progress should be 0.0 after clear_analysis()\n"
                    f"Was: {processing_progress}"
                )
    
    @given(
        current_page=st.sampled_from(["home", "analysis", "multiverse", "emotion", "social"]),
        theme=st.sampled_from(["dark", "light"]),
        gesture_enabled=st.booleans()
    )
    @settings(max_examples=100)
    def test_ui_state_preserved_on_upload(self, current_page: str, theme: str, gesture_enabled: bool):
        """
        Property 2: Session State Reset on Upload (UI state preservation)
        
        For any video upload operation, UI state (current_page, theme, 
        gesture_enabled) SHALL be preserved after clearing analysis.
        
        Validates: Requirements 1.2
        """
        mock_session_state = MockSessionState()
        
        with patch.object(sys.modules['cinemasense.core.session'], 'st') as mock_st:
            mock_st.session_state = mock_session_state
            
            # Initialize session state
            SessionManager.initialize()
            
            # Set UI state
            SessionManager.set("current_page", current_page)
            SessionManager.set("theme", theme)
            SessionManager.set("gesture_enabled", gesture_enabled)
            
            # Set some analysis data
            SessionManager.set("analysis", {"data": "test"})
            
            # Clear analysis
            SessionManager.clear_analysis()
            
            # Verify UI state is preserved
            assert mock_session_state.get(f"{SessionManager.PREFIX}current_page") == current_page, (
                f"current_page should be preserved after clear_analysis()"
            )
            assert mock_session_state.get(f"{SessionManager.PREFIX}theme") == theme, (
                f"theme should be preserved after clear_analysis()"
            )
            assert mock_session_state.get(f"{SessionManager.PREFIX}gesture_enabled") == gesture_enabled, (
                f"gesture_enabled should be preserved after clear_analysis()"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
