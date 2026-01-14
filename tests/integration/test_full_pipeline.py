"""
Integration Tests for CinemaSense Full Pipeline

Tests the complete video analysis workflow including:
- Video upload → analysis → report flow
- Multi-page navigation state preservation
- Error recovery scenarios

Requirements: 18.3
"""

import sys
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import cv2


class TestVideoAnalysisPipeline:
    """Test the complete video analysis pipeline flow."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_video_path(self, test_video_path):
        """Get the test video path or skip if not available."""
        if test_video_path is None:
            pytest.skip("Test video not available")
        return test_video_path
    
    def test_video_metadata_extraction(self, sample_video_path):
        """Test that video metadata can be extracted correctly."""
        from cinemasense.core.video_capture import SafeVideoCapture
        
        with SafeVideoCapture(sample_video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        assert fps > 0, "FPS should be positive"
        assert frame_count > 0, "Frame count should be positive"
        assert width > 0, "Width should be positive"
        assert height > 0, "Height should be positive"
    
    def test_cut_detection_pipeline(self, sample_video_path):
        """Test the explainable AI cut detection pipeline."""
        from cinemasense.pipeline.explainable_ai import detect_cuts_with_explanation
        
        result = detect_cuts_with_explanation(
            sample_video_path,
            sample_every_n_frames=5,
            threshold=0.55
        )
        
        # Verify result structure
        assert hasattr(result, 'cuts'), "Result should have cuts attribute"
        assert hasattr(result, 'total_cuts'), "Result should have total_cuts"
        assert hasattr(result, 'avg_confidence'), "Result should have avg_confidence"
        assert hasattr(result, 'explanation_summary'), "Result should have explanation_summary"
        
        # Verify confidence is in valid range
        assert 0.0 <= result.avg_confidence <= 1.0, "Avg confidence should be in [0, 1]"
        
        # Verify each cut has required fields
        for cut in result.cuts:
            assert hasattr(cut, 'timestamp'), "Cut should have timestamp"
            assert hasattr(cut, 'confidence'), "Cut should have confidence"
            assert hasattr(cut, 'primary_reason'), "Cut should have primary_reason"
            assert hasattr(cut, 'cut_type'), "Cut should have cut_type"
            assert 0.0 <= cut.confidence <= 1.0, "Cut confidence should be in [0, 1]"
    
    def test_emotion_analysis_pipeline(self, sample_video_path):
        """Test the emotion rhythm analysis pipeline."""
        from cinemasense.pipeline.emotion_rhythm import extract_emotion_timeline
        
        result = extract_emotion_timeline(sample_video_path, sample_rate=10)
        
        # Verify result structure
        assert hasattr(result, 'timeline'), "Result should have timeline"
        assert hasattr(result, 'overall_score'), "Result should have overall_score"
        assert hasattr(result, 'emotion_distribution'), "Result should have emotion_distribution"
        assert hasattr(result, 'rhythm_pattern'), "Result should have rhythm_pattern"
        assert hasattr(result, 'heatmap_data'), "Result should have heatmap_data"
        
        # Verify timeline is not empty
        assert len(result.timeline) > 0, "Timeline should not be empty"
        
        # Verify emotion distribution sums to approximately 1.0
        total = sum(result.emotion_distribution.values())
        assert 0.99 <= total <= 1.01, f"Emotion distribution should sum to 1.0, got {total}"
        
        # Verify rhythm pattern is valid
        valid_patterns = [
            "Sustained High Energy", "Calm & Steady", "Building Crescendo",
            "Descending Arc", "Dynamic Rollercoaster", "Gradual Rise",
            "Gentle Decline", "Balanced Flow", "Unknown"
        ]
        assert result.rhythm_pattern in valid_patterns, f"Invalid rhythm pattern: {result.rhythm_pattern}"
    
    def test_multiverse_preview_generation(self, sample_video_path, temp_output_dir):
        """Test multiverse style preview generation."""
        from cinemasense.pipeline.multiverse import generate_multiverse_preview, get_available_styles
        
        styles = get_available_styles()
        assert len(styles) > 0, "Should have available styles"
        
        # Test with first available style
        style = styles[0]
        result = generate_multiverse_preview(
            sample_video_path,
            style['id'],
            temp_output_dir
        )
        
        # Verify result structure
        assert result.style_name == style['id'], "Style name should match"
        assert len(result.previews) == 3, "Should generate 3 previews (25%, 50%, 75%)"
        
        # Verify preview files exist
        for preview in result.previews:
            assert Path(preview['path']).exists(), f"Preview file should exist: {preview['path']}"
    
    def test_social_pack_generation(self, sample_video_path, temp_output_dir):
        """Test social pack generation."""
        from cinemasense.pipeline.social_pack import generate_social_pack
        
        metadata = {
            "duration_s": 10.0,
            "width": 1920,
            "height": 1080
        }
        
        result = generate_social_pack(
            sample_video_path,
            temp_output_dir,
            metadata,
            emotion_analysis=None,
            platforms=["youtube", "instagram"]
        )
        
        # Verify result structure
        assert len(result.title_suggestions) == 5, "Should generate exactly 5 titles"
        assert len(result.hashtags) > 0, "Should generate hashtags"
        assert len(result.caption) > 0, "Should generate caption"
        
        # Verify platform-specific content
        assert "youtube" in result.platform_optimized, "Should have YouTube content"
        assert "instagram" in result.platform_optimized, "Should have Instagram content"
        
        # Verify thumbnail dimensions
        yt_dims = result.platform_optimized["youtube"]["thumbnail_dimensions"]
        assert yt_dims == (1280, 720), f"YouTube thumbnail should be 1280x720, got {yt_dims}"
        
        ig_dims = result.platform_optimized["instagram"]["thumbnail_dimensions"]
        assert ig_dims == (1080, 1080), f"Instagram thumbnail should be 1080x1080, got {ig_dims}"


class TestSessionStatePreservation:
    """Test multi-page navigation state preservation."""
    
    def test_session_manager_initialization(self):
        """Test that SessionManager initializes all required keys."""
        from cinemasense.core.session import SessionManager
        
        # Mock streamlit session state
        with patch('cinemasense.core.session.st') as mock_st:
            mock_st.session_state = {}
            
            SessionManager.initialize()
            
            # Verify required keys exist (with cs_ prefix)
            required_keys = [
                'cs_video_path', 'cs_video_name', 'cs_metadata',
                'cs_analysis', 'cs_keyframes', 'cs_emotion', 'cs_social', 'cs_multiverse'
            ]
            
            for key in required_keys:
                assert key in mock_st.session_state, f"Session should have {key}"
    
    def test_session_state_reset_on_upload(self):
        """Test that analysis results are cleared when new video is uploaded."""
        from cinemasense.core.session import SessionManager
        
        with patch('cinemasense.core.session.st') as mock_st:
            # Initialize with prefixed keys
            mock_st.session_state = {
                'cs_video_path': '/old/path.mp4',
                'cs_video_name': 'old_video',
                'cs_metadata': {'fps': 30},
                'cs_analysis': {'cuts': []},
                'cs_keyframes': [{'time': 1.0}],
                'cs_emotion': {'score': 50},
                'cs_social': {'titles': []},
                'cs_multiverse': {'romantic': {}},
                'cs_initialized': True
            }
            
            SessionManager.clear_analysis()
            
            # Verify analysis results are cleared (with cs_ prefix)
            assert mock_st.session_state.get('cs_analysis') is None
            assert mock_st.session_state.get('cs_keyframes') is None
            assert mock_st.session_state.get('cs_emotion') is None
            assert mock_st.session_state.get('cs_social') is None
            assert mock_st.session_state.get('cs_multiverse') is None
    
    def test_widget_key_uniqueness(self):
        """Test that generated widget keys are unique."""
        from cinemasense.core.session import SessionManager
        
        with patch('cinemasense.core.session.st') as mock_st:
            mock_st.session_state = {}
            
            keys = set()
            for i in range(100):
                key = SessionManager.generate_widget_key("test", str(i))
                assert key not in keys, f"Duplicate key generated: {key}"
                keys.add(key)


class TestErrorRecovery:
    """Test error recovery scenarios."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_invalid_video_path_handling(self):
        """Test graceful handling of invalid video paths."""
        from cinemasense.core.video_capture import SafeVideoCapture, VideoOpenError
        
        with pytest.raises(VideoOpenError):
            with SafeVideoCapture("/nonexistent/video.mp4") as cap:
                pass
    
    def test_corrupted_frame_handling(self, test_video_path):
        """Test that corrupted frames are skipped gracefully."""
        if test_video_path is None:
            pytest.skip("Test video not available")
        
        from cinemasense.pipeline.explainable_ai import detect_cuts_with_explanation
        
        # This should complete without raising exceptions
        # even if some frames are problematic
        result = detect_cuts_with_explanation(
            test_video_path,
            sample_every_n_frames=2,
            threshold=0.55
        )
        
        assert result is not None, "Should return result even with potential frame issues"
    
    def test_empty_analysis_report_generation(self):
        """Test report generation with minimal data."""
        from cinemasense.services.report import ReportGenerator
        
        # Generate report with minimal data
        report = ReportGenerator.generate(
            video_name="test_video",
            metadata={"fps": 30, "frame_count": 100, "width": 1920, "height": 1080, "duration_s": 3.33},
            cuts=None,
            emotion=None,
            keyframes=None,
            social=None,
            multiverse=None
        )
        
        # Verify report structure
        assert "generated_at" in report, "Report should have timestamp"
        assert "video_name" in report, "Report should have video name"
        assert report["video_name"] == "test_video"
    
    def test_invalid_style_handling(self, test_video_path, temp_output_dir):
        """Test handling of invalid multiverse style names."""
        if test_video_path is None:
            pytest.skip("Test video not available")
        
        from cinemasense.pipeline.multiverse import generate_multiverse_preview, InvalidStyleError
        
        with pytest.raises(InvalidStyleError):
            generate_multiverse_preview(
                test_video_path,
                "nonexistent_style",
                temp_output_dir
            )
    
    def test_invalid_platform_handling(self, test_video_path, temp_output_dir):
        """Test handling of invalid social pack platform names."""
        if test_video_path is None:
            pytest.skip("Test video not available")
        
        from cinemasense.pipeline.social_pack import generate_social_pack, InvalidPlatformError
        
        metadata = {"duration_s": 10.0}
        
        # Should skip invalid platforms and continue with valid ones
        result = generate_social_pack(
            test_video_path,
            temp_output_dir,
            metadata,
            platforms=["youtube", "invalid_platform"]
        )
        
        # Should still generate for valid platform
        assert "youtube" in result.platform_optimized


class TestReportGeneration:
    """Test comprehensive report generation."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_report_generation(self, temp_output_dir):
        """Test generating a complete report with all analysis data."""
        from cinemasense.services.report import ReportGenerator
        
        # Create sample data
        metadata = {
            "fps": 30.0,
            "frame_count": 300,
            "width": 1920,
            "height": 1080,
            "duration_s": 10.0
        }
        
        cuts = {
            "total": 5,
            "avg_confidence": 0.85,
            "cut_types": {"hard_cut": 3, "dissolve": 2},
            "summary": "Test summary"
        }
        
        emotion = {
            "overall_score": 65.5,
            "rhythm_pattern": "Dynamic Rollercoaster",
            "confidence": 0.9,
            "emotion_distribution": {"Joy": 0.3, "Tension": 0.2, "Calm": 0.5}
        }
        
        report = ReportGenerator.generate(
            video_name="test_video",
            metadata=metadata,
            cuts=cuts,
            emotion=emotion,
            keyframes=[{"time": 1.0}, {"time": 2.0}],
            social={"titles": ["Title 1", "Title 2"]},
            multiverse={"romantic": {"previews": []}}
        )
        
        # Verify report structure
        assert "generated_at" in report
        assert "video_name" in report
        assert "metadata" in report
        assert "cuts" in report
        assert "emotion" in report
        assert "keyframes_count" in report
        
        # Verify data integrity
        assert report["video_name"] == "test_video"
        assert report["keyframes_count"] == 2
    
    def test_report_save_and_load(self, temp_output_dir):
        """Test saving and loading reports."""
        from cinemasense.services.report import ReportGenerator
        
        report = ReportGenerator.generate(
            video_name="test_video",
            metadata={"fps": 30, "frame_count": 100, "width": 1920, "height": 1080, "duration_s": 3.33}
        )
        
        report_path = temp_output_dir / "test_report.json"
        
        # Save report
        ReportGenerator.save(report, report_path, validate=False)
        
        assert report_path.exists(), "Report file should be created"
        
        # Load report
        loaded = ReportGenerator.load(report_path)
        
        assert loaded["video_name"] == report["video_name"]
        assert loaded["generated_at"] == report["generated_at"]
