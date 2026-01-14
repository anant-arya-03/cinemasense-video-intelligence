"""
Report Page - Comprehensive analysis reports and exports

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import streamlit as st
import json
import logging
from pathlib import Path
from datetime import datetime

from cinemasense.core.session import SessionManager
from cinemasense.storage.paths import ProjectPaths
from cinemasense.services.report import ReportGenerator, ReportValidationError

logger = logging.getLogger("cinemasense.ui.views.report")


def render():
    """Render the report page"""
    st.title("📋 Comprehensive Report")
    st.markdown("*Export all analysis results in a single comprehensive report*")
    
    # Ensure session is initialized
    SessionManager.initialize()
    
    # Check for video
    video_path = SessionManager.get("video_path")
    video_name = SessionManager.get("video_name")
    
    if not video_path:
        st.warning("⚠️ Please upload a video on the Home page first")
        return
    
    st.info(f"📹 Video: {Path(video_path).name}")
    
    # Analysis status
    st.markdown("### 📊 Analysis Status")
    
    analysis = SessionManager.get("analysis")
    keyframes = SessionManager.get("keyframes")
    multiverse = SessionManager.get("multiverse")
    emotion = SessionManager.get("emotion")
    social = SessionManager.get("social")
    
    status_items = [
        ("Core Analysis", bool(analysis)),
        ("Keyframes", bool(keyframes)),
        ("Multiverse", bool(multiverse)),
        ("Emotion Rhythm", bool(emotion)),
        ("Social Pack", bool(social))
    ]
    
    cols = st.columns(5)
    for col, (name, completed) in zip(cols, status_items):
        with col:
            status = "✅" if completed else "❌"
            st.metric(name, status)
    
    # Generate report button
    key = SessionManager.generate_widget_key("report_gen_btn")
    if st.button("📋 Generate Comprehensive Report", type="primary", key=key):
        generate_report()
    
    # Display existing report if available
    paths = ProjectPaths()
    if video_name:
        report_dir = paths.output / video_name
        report_path = report_dir / "comprehensive_report.json"
        
        if report_path.exists():
            display_report(report_path)


def generate_report():
    """Generate comprehensive report using ReportGenerator service."""
    progress = st.progress(0, "Generating report...")
    
    try:
        # Get all session data
        video_path = SessionManager.get("video_path")
        video_name = SessionManager.get("video_name", Path(video_path).stem if video_path else "unknown")
        metadata = SessionManager.get("metadata", {})
        analysis = SessionManager.get("analysis")
        keyframes = SessionManager.get("keyframes")
        emotion = SessionManager.get("emotion")
        social = SessionManager.get("social")
        multiverse = SessionManager.get("multiverse")
        
        progress.progress(20, "Compiling analysis data...")
        
        # Ensure metadata has required fields
        if not metadata:
            metadata = {}
        
        # Add required metadata fields if missing
        if "name" not in metadata:
            metadata["name"] = video_name
        if "fps" not in metadata:
            metadata["fps"] = 0.0
        if "frame_count" not in metadata:
            metadata["frame_count"] = 0
        if "width" not in metadata:
            metadata["width"] = 0
        if "height" not in metadata:
            metadata["height"] = 0
        if "duration_s" not in metadata:
            metadata["duration_s"] = 0.0
        
        progress.progress(40, "Generating report structure...")
        
        # Generate report using ReportGenerator
        report = ReportGenerator.generate(
            video_name=video_name,
            metadata=metadata,
            cuts=analysis,
            emotion=emotion,
            keyframes=keyframes,
            social=social,
            multiverse=multiverse
        )
        
        progress.progress(60, "Validating report...")
        
        # Validate report structure
        try:
            ReportGenerator.validate(report)
        except ReportValidationError as e:
            logger.warning(f"Report validation warning: {e}")
            # Continue anyway - validation is advisory
        
        progress.progress(80, "Saving report...")
        
        # Save report
        paths = ProjectPaths()
        output_dir = paths.output / video_name
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "comprehensive_report.json"
        
        ReportGenerator.save(report, report_path, validate=False)
        
        progress.progress(100, "Complete!")
        st.success("✅ Comprehensive report generated!")
        logger.info(f"Report generated and saved to: {report_path}")
        st.rerun()
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        st.error(f"❌ Report generation failed: {str(e)}")


def display_report(report_path: Path):
    """Display the generated report with preview and download options."""
    st.markdown("### 📄 Report Preview")
    
    try:
        report = ReportGenerator.load(report_path)
    except Exception as e:
        st.error(f"Failed to load report: {e}")
        return
    
    # Metadata section
    with st.expander("📋 Report Metadata", expanded=True):
        generated_at = report.get("generated_at", "Unknown")
        video_name = report.get("video_name", "Unknown")
        
        col1, col2 = st.columns(2)
        col1.write(f"**Generated:** {generated_at[:19] if len(generated_at) > 19 else generated_at}")
        col2.write(f"**Video:** {video_name}")
    
    # Video info section
    with st.expander("📹 Video Information"):
        metadata = report.get("metadata", {})
        if metadata:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration", f"{metadata.get('duration_s', 0):.1f}s")
            col2.metric("Resolution", f"{metadata.get('width', 0)}×{metadata.get('height', 0)}")
            col3.metric("FPS", f"{metadata.get('fps', 0):.1f}")
            col4.metric("Frames", f"{metadata.get('frame_count', 0):,}")
    
    # Analysis summary section
    with st.expander("📊 Cut Analysis"):
        cuts = report.get("cuts")
        if cuts:
            col1, col2, col3 = st.columns(3)
            col1.metric("Cuts Detected", cuts.get("total_cuts", 0))
            col2.metric("Avg Confidence", f"{cuts.get('avg_confidence', 0):.2f}")
            
            cut_types = cuts.get("cut_type_distribution", {})
            if cut_types:
                col3.metric("Cut Types", len(cut_types))
            
            summary = cuts.get("explanation_summary")
            if summary:
                st.write("**AI Summary:**")
                st.info(summary)
        else:
            st.write("No cut analysis available")
    
    # Emotion analysis section
    with st.expander("💫 Emotion Analysis"):
        emotion = report.get("emotion")
        if emotion:
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Score", f"{emotion.get('overall_score', 0):.1f}")
            col2.metric("Pattern", emotion.get("rhythm_pattern", "Unknown"))
            col3.metric("Confidence", f"{emotion.get('confidence', 0):.2f}")
            
            peaks = emotion.get("peak_moments", [])
            if peaks:
                st.write(f"**Peak Moments:** {len(peaks)}")
            
            distribution = emotion.get("emotion_distribution", {})
            if distribution:
                st.write("**Emotion Distribution:**")
                for emo, score in distribution.items():
                    st.progress(score, text=f"{emo}: {score:.1%}")
        else:
            st.write("No emotion analysis available")
    
    # Keyframes section
    with st.expander("🎬 Keyframes"):
        keyframes_count = report.get("keyframes_count", 0)
        st.metric("Keyframes Extracted", keyframes_count)
    
    # Social pack section
    with st.expander("📱 Social Pack"):
        social = report.get("social")
        if social:
            titles = social.get("title_suggestions", [])
            if titles:
                st.write("**Title Suggestions:**")
                for i, title in enumerate(titles, 1):
                    st.write(f"{i}. {title}")
            
            hashtags = social.get("hashtags", [])
            if hashtags:
                st.write("**Hashtags:**")
                st.write(" ".join(f"#{tag}" for tag in hashtags[:10]))
            
            caption = social.get("caption")
            if caption:
                st.write("**Caption:**")
                st.info(caption)
        else:
            st.write("No social pack available")
    
    # Download section
    st.markdown("### ⬇️ Download Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Full JSON download
        json_data = ReportGenerator.to_json_string(report)
        download_filename = ReportGenerator.get_download_filename(
            report.get("video_name", "video")
        )
        
        key1 = SessionManager.generate_widget_key("download_full_json")
        st.download_button(
            "📥 Download Full Report (JSON)",
            data=json_data,
            file_name=download_filename,
            mime="application/json",
            key=key1
        )
    
    with col2:
        # Summary text download
        summary = ReportGenerator.generate_summary(report)
        summary_filename = download_filename.replace(".json", ".txt")
        
        key2 = SessionManager.generate_widget_key("download_summary_txt")
        st.download_button(
            "📄 Download Summary (TXT)",
            data=summary,
            file_name=summary_filename,
            mime="text/plain",
            key=key2
        )
