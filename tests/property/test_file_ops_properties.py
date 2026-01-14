"""
Property-based tests for FileOps utility class.

Feature: cinemasense-stabilization
Property 6: Path Sanitization for Windows
Validates: Requirements 3.5

Tests that filename sanitization produces valid Windows filenames
for any input string.
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings

from cinemasense.core import FileOps

# Characters invalid on Windows filesystems
WINDOWS_INVALID_CHARS = set('<>:"/\\|?*')
# Control characters (0-31)
CONTROL_CHARS = set(chr(i) for i in range(32))


class TestFileOpsSanitization:
    """
    Property tests for FileOps.sanitize_filename.
    
    Feature: cinemasense-stabilization, Property 6: Path Sanitization for Windows
    Validates: Requirements 3.5
    """
    
    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_sanitized_filename_contains_no_invalid_windows_chars(self, filename: str):
        """
        Property 6: Path Sanitization for Windows
        
        For any input filename string, the sanitized output SHALL contain
        only characters valid for Windows filenames (no <>:"/\\|?*).
        
        Validates: Requirements 3.5
        """
        sanitized = FileOps.sanitize_filename(filename)
        
        # Check no invalid Windows characters
        invalid_found = WINDOWS_INVALID_CHARS.intersection(set(sanitized))
        assert not invalid_found, (
            f"Sanitized filename contains invalid Windows chars: {invalid_found}\n"
            f"Input: {repr(filename)}\n"
            f"Output: {repr(sanitized)}"
        )
    
    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_sanitized_filename_contains_no_control_chars(self, filename: str):
        """
        Property 6: Path Sanitization for Windows (control characters)
        
        For any input filename string, the sanitized output SHALL contain
        no control characters (ASCII 0-31).
        
        Validates: Requirements 3.5
        """
        sanitized = FileOps.sanitize_filename(filename)
        
        # Check no control characters
        control_found = CONTROL_CHARS.intersection(set(sanitized))
        assert not control_found, (
            f"Sanitized filename contains control chars: {[ord(c) for c in control_found]}\n"
            f"Input: {repr(filename)}\n"
            f"Output: {repr(sanitized)}"
        )
    
    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_sanitized_filename_is_non_empty(self, filename: str):
        """
        Property 6: Path Sanitization for Windows (non-empty result)
        
        For any input filename string, the sanitized output SHALL be
        a non-empty string.
        
        Validates: Requirements 3.5
        """
        sanitized = FileOps.sanitize_filename(filename)
        
        assert len(sanitized) > 0, (
            f"Sanitized filename is empty\n"
            f"Input: {repr(filename)}"
        )
    
    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=100)
    def test_sanitized_filename_no_leading_trailing_spaces_or_dots(self, filename: str):
        """
        Property 6: Path Sanitization for Windows (no leading/trailing spaces/dots)
        
        For any input filename string, the sanitized output SHALL not
        have leading or trailing spaces or dots (Windows restriction).
        
        Validates: Requirements 3.5
        """
        sanitized = FileOps.sanitize_filename(filename)
        
        # Check no leading/trailing spaces or dots
        assert not sanitized.startswith(' '), (
            f"Sanitized filename starts with space\n"
            f"Input: {repr(filename)}\n"
            f"Output: {repr(sanitized)}"
        )
        assert not sanitized.startswith('.'), (
            f"Sanitized filename starts with dot\n"
            f"Input: {repr(filename)}\n"
            f"Output: {repr(sanitized)}"
        )
        assert not sanitized.endswith(' '), (
            f"Sanitized filename ends with space\n"
            f"Input: {repr(filename)}\n"
            f"Output: {repr(sanitized)}"
        )
        assert not sanitized.endswith('.'), (
            f"Sanitized filename ends with dot\n"
            f"Input: {repr(filename)}\n"
            f"Output: {repr(sanitized)}"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
