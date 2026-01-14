"""
Property-based tests for system dependency checks.

Feature: cinemasense-stabilization
Property 18: Dependency Check Completeness
Validates: Requirements 13.1, 13.3

Tests that system check execution includes status for all critical dependencies
(Python, FFmpeg, OpenCV, MediaPipe, MoviePy).
"""

import sys
from pathlib import Path
from typing import List, Set

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from hypothesis import given, strategies as st, settings

from cinemasense.core.system_check import (
    DependencyStatus,
    run_all_checks,
    check_python_version,
    check_ffmpeg,
    check_opencv,
    check_mediapipe,
    check_moviepy,
    check_librosa,
    check_sklearn,
    get_system_info,
)


# Critical dependencies that MUST be checked
CRITICAL_DEPENDENCIES = {"Python", "FFmpeg", "FFmpeg (imageio)", "OpenCV", "MediaPipe", "MoviePy"}

# All dependencies that should be checked
ALL_DEPENDENCIES = CRITICAL_DEPENDENCIES | {"Librosa", "Scikit-learn"}


class TestDependencyCheckCompleteness:
    """
    Property tests for dependency check completeness.
    
    Feature: cinemasense-stabilization, Property 18: Dependency Check Completeness
    Validates: Requirements 13.1, 13.3
    """
    
    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=100, deadline=None)
    def test_all_critical_dependencies_checked(self, call_count: int):
        """
        Property 18: Dependency Check Completeness
        
        For any system check execution, the result SHALL include status
        for all critical dependencies (Python, FFmpeg, OpenCV, MediaPipe, MoviePy).
        
        Validates: Requirements 13.1, 13.3
        """
        # Run checks multiple times to verify consistency
        for _ in range(call_count):
            all_ok, checks = run_all_checks()
            
            # Extract dependency names from results
            checked_names = {check.name for check in checks}
            
            # Verify all critical dependencies are present
            # Note: FFmpeg may appear as "FFmpeg" or "FFmpeg (imageio)" depending on system
            ffmpeg_checked = "FFmpeg" in checked_names or "FFmpeg (imageio)" in checked_names
            assert ffmpeg_checked, (
                f"FFmpeg dependency check missing from results\n"
                f"Checked dependencies: {checked_names}"
            )
            
            # Check other critical dependencies
            for dep in ["Python", "OpenCV", "MediaPipe", "MoviePy"]:
                assert dep in checked_names, (
                    f"Critical dependency '{dep}' missing from check results\n"
                    f"Checked dependencies: {checked_names}"
                )
    
    @given(st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_dependency_status_structure_valid(self, _: bool):
        """
        Property 18: Dependency Check Completeness (structure validation)
        
        For any dependency check result, each DependencyStatus SHALL have:
        - A non-empty name
        - A boolean available flag
        - A version string (may be empty if unavailable)
        - An error string (may be empty if available)
        
        Validates: Requirements 13.1, 13.3
        """
        _, checks = run_all_checks()
        
        for check in checks:
            # Verify it's a DependencyStatus instance
            assert isinstance(check, DependencyStatus), (
                f"Check result should be DependencyStatus, got {type(check)}"
            )
            
            # Name must be non-empty
            assert check.name and len(check.name) > 0, (
                "Dependency name must be non-empty"
            )
            
            # Available must be boolean
            assert isinstance(check.available, bool), (
                f"available must be boolean, got {type(check.available)}"
            )
            
            # Version and error must be strings
            assert isinstance(check.version, str), (
                f"version must be string, got {type(check.version)}"
            )
            assert isinstance(check.error, str), (
                f"error must be string, got {type(check.error)}"
            )
            
            # If available, version should typically be non-empty (except edge cases)
            # If not available, error should be non-empty
            if not check.available:
                assert len(check.error) > 0, (
                    f"Unavailable dependency '{check.name}' should have error message"
                )
    
    @given(st.integers(min_value=1, max_value=3))
    @settings(max_examples=100, deadline=None)
    def test_check_results_are_consistent(self, call_count: int):
        """
        Property 18: Dependency Check Completeness (consistency)
        
        For any number of check executions, the set of checked dependencies
        SHALL remain consistent (same dependencies checked each time).
        
        Validates: Requirements 13.1, 13.3
        """
        results = []
        for _ in range(call_count):
            _, checks = run_all_checks()
            checked_names = frozenset(check.name for check in checks)
            results.append(checked_names)
        
        # All results should have the same set of dependency names
        first_result = results[0]
        for i, result in enumerate(results[1:], start=2):
            assert result == first_result, (
                f"Dependency check results inconsistent between calls\n"
                f"Call 1: {first_result}\n"
                f"Call {i}: {result}"
            )
    
    @given(st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_python_version_check_present(self, _: bool):
        """
        Property 18: Dependency Check Completeness (Python version)
        
        For any system check execution, Python version check SHALL be present
        and verify Python 3.10+ compatibility.
        
        Validates: Requirements 13.1
        """
        python_status = check_python_version()
        
        assert python_status.name == "Python", (
            f"Python check should have name 'Python', got '{python_status.name}'"
        )
        
        # Version should be in format X.Y.Z
        version_parts = python_status.version.split(".")
        assert len(version_parts) >= 2, (
            f"Python version should have at least major.minor format\n"
            f"Got: {python_status.version}"
        )
        
        # If available, version should indicate 3.10+
        if python_status.available:
            major = int(version_parts[0])
            minor = int(version_parts[1])
            assert major >= 3 and (major > 3 or minor >= 10), (
                f"Python 3.10+ required, got {python_status.version}"
            )
    
    @given(st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_all_ok_flag_reflects_critical_deps(self, _: bool):
        """
        Property 18: Dependency Check Completeness (all_ok flag)
        
        For any system check execution, the all_ok flag SHALL be True
        only if all critical dependencies are available.
        
        Validates: Requirements 13.1, 13.3
        """
        all_ok, checks = run_all_checks()
        
        # Get status of first 5 dependencies (critical ones per implementation)
        critical_checks = checks[:5]
        critical_all_available = all(c.available for c in critical_checks)
        
        assert all_ok == critical_all_available, (
            f"all_ok flag should reflect critical dependency status\n"
            f"all_ok: {all_ok}\n"
            f"Critical deps available: {critical_all_available}\n"
            f"Critical deps: {[(c.name, c.available) for c in critical_checks]}"
        )
    
    @given(st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_individual_check_functions_return_valid_status(self, _: bool):
        """
        Property 18: Dependency Check Completeness (individual checks)
        
        For any individual dependency check function, it SHALL return
        a valid DependencyStatus with appropriate name.
        
        Validates: Requirements 13.3
        """
        check_functions = [
            ("Python", check_python_version),
            ("FFmpeg", check_ffmpeg),
            ("OpenCV", check_opencv),
            ("MediaPipe", check_mediapipe),
            ("MoviePy", check_moviepy),
            ("Librosa", check_librosa),
            ("Scikit-learn", check_sklearn),
        ]
        
        for expected_name, check_func in check_functions:
            status = check_func()
            
            assert isinstance(status, DependencyStatus), (
                f"{expected_name} check should return DependencyStatus"
            )
            
            # FFmpeg may have alternate name
            if expected_name == "FFmpeg":
                assert status.name in ("FFmpeg", "FFmpeg (imageio)"), (
                    f"FFmpeg check should have name 'FFmpeg' or 'FFmpeg (imageio)'\n"
                    f"Got: {status.name}"
                )
            else:
                assert status.name == expected_name, (
                    f"Check function should return status with name '{expected_name}'\n"
                    f"Got: {status.name}"
                )
    
    @given(st.booleans())
    @settings(max_examples=100, deadline=None)
    def test_system_info_contains_required_fields(self, _: bool):
        """
        Property 18: Dependency Check Completeness (system info)
        
        For any system info request, the result SHALL contain
        os, python_version, and architecture information.
        
        Validates: Requirements 13.1
        """
        info = get_system_info()
        
        required_fields = ["os", "os_version", "python_version", "architecture", "processor", "cpu_count"]
        
        for field in required_fields:
            assert field in info, (
                f"System info missing required field '{field}'\n"
                f"Available fields: {list(info.keys())}"
            )
        
        # Verify types
        assert isinstance(info["os"], str) and len(info["os"]) > 0, (
            "os should be a non-empty string"
        )
        assert isinstance(info["python_version"], str) and len(info["python_version"]) > 0, (
            "python_version should be a non-empty string"
        )
        assert isinstance(info["architecture"], str), (
            "architecture should be a string"
        )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
