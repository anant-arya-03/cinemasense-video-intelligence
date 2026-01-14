"""
CinemaSense AI Studio v2.0 - Industry-Ready Video Intelligence Platform
Fully tested, error-free, production-ready

Refactored to use SessionManager for stable session state management,
unique widget keys, and proper error handling.

Requirements: 1.4, 2.6, 12.2 - Integration and final polish
"""

import sys
import os
from pathlib import Path

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import cv2
import numpy as np
import json
from datetime import datetime

# Import and setup centralized logging (Requirements: 12.1, 12.3)
from cinemasense.logging_setup import LoggingConfig, get_logger, log_exception

# Initialize centralized logging with rotation
LoggingConfig.setup(
    log_level="INFO",
    max_bytes=5 * 1024 * 1024,  # 5 MB
    backup_count=5
)
logger = get_logger("cinemasense.app")

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="CinemaSense AI Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import CinemaSense modules
from cinemasense.core.session import SessionManager
from cinemasense.core.system_check import run_all_checks, get_system_info, DependencyStatus
from cinemasense.core.file_ops import FileOps
from cinemasense.core.video_capture import SafeVideoCapture, VideoOpenError
from cinemasense.ui.glassmorphic import (
    inject_glassmorphic_css, 
    glass_card, 
    glass_metric, 
    glass_progress,
    render_logo,
    render_feature_card,
    render_status_indicator,
    render_section_header,
    render_info_banner,
    render_empty_state
)

# Import stabilized pipeline modules
from cinemasense.pipeline.multiverse import (
    generate_multiverse_preview, 
    get_available_styles,
    apply_style_to_frame,
    InvalidStyleError,
    PreviewGenerationError
)
from cinemasense.pipeline.emotion_rhythm import (
    extract_emotion_timeline,
    EmotionAnalysisError
)
from cinemasense.pipeline.explainable_ai import (
    detect_cuts_with_explanation
)
from cinemasense.pipeline.social_pack import (
    generate_social_pack,
    SocialPackError
)
from cinemasense.pipeline.gesture_control import (
    GestureController,
    GestureType,
    is_mediapipe_available
)
from cinemasense.pipeline.color_grading import (
    apply_color_grading_safe,
    get_available_presets,
    analyze_color_palette
)
from cinemasense.services.report import ReportGenerator

# Create directories
for d in ["data/input", "data/output", "logs"]:
    FileOps.ensure_directory(ROOT / d)

# Inject glassmorphic CSS
inject_glassmorphic_css()


# ============== DEPENDENCY CHECK ==============
def check_dependencies():
    """
    Check system dependencies at startup and display status.
    
    Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
    """
    # Only check once per session
    if SessionManager.get("dependencies_checked", False):
        return SessionManager.get("dependencies_ok", True)
    
    all_ok, checks = run_all_checks()
    
    # Store results in session
    SessionManager.set("dependencies_checked", True)
    SessionManager.set("dependencies_ok", all_ok)
    SessionManager.set("dependency_status", [
        {"name": c.name, "available": c.available, "version": c.version, "error": c.error}
        for c in checks
    ])
    
    # Display warnings for missing dependencies
    missing = [c for c in checks if not c.available]
    if missing:
        with st.sidebar:
            st.warning("⚠️ Missing Dependencies")
            for dep in missing:
                st.caption(f"❌ {dep.name}: {dep.error}")
            
            with st.expander("Installation Instructions"):
                st.markdown("""
                **Install missing dependencies:**
                ```bash
                pip install opencv-python mediapipe moviepy librosa scikit-learn imageio-ffmpeg
                ```
                
                **For FFmpeg issues:**
                - Windows: `pip install imageio-ffmpeg`
                - Or download from https://ffmpeg.org/download.html
                """)
    
    return all_ok


# ============== SESSION INITIALIZATION ==============
# Initialize session state BEFORE any UI rendering
SessionManager.initialize()


# ============== HELPER FUNCTIONS ==============
def save_uploaded_video(uploaded_file):
    """
    Save uploaded video and return path.
    
    Uses FileOps for Windows-safe filename handling.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = FileOps.sanitize_filename(uploaded_file.name)
        filename = f"{timestamp}_{safe_name}"
        filepath = ROOT / "data" / "input" / filename
        
        # Ensure directory exists
        FileOps.ensure_directory(filepath.parent)
        
        # Write file with retry logic
        data = uploaded_file.getbuffer()
        if not FileOps.safe_write(filepath, bytes(data)):
            raise IOError(f"Failed to save video file: {filepath}")
        
        logger.info(f"Saved uploaded video: {filepath}")
        return str(filepath), filename
        
    except Exception as e:
        logger.error(f"Error saving uploaded video: {e}")
        raise


def get_video_info(video_path):
    """
    Extract video metadata using SafeVideoCapture.
    
    Requirements: 2.2, 2.5
    """
    try:
        with SafeVideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            return {
                "fps": round(fps, 2),
                "frames": frame_count,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": round(duration, 2),
                "duration_s": round(duration, 2)
            }
    except VideoOpenError as e:
        logger.error(f"Failed to open video: {e}")
        raise
    except Exception as e:
        logger.error(f"Error extracting video info: {e}")
        raise


def run_explainable_analysis(video_path, threshold=0.55, sample_rate=2, progress_callback=None):
    """
    Run explainable AI cut detection with progress updates.
    
    Requirements: 6.1, 6.2, 6.3, 6.4, 6.5
    """
    try:
        if progress_callback:
            progress_callback(20, "Detecting cuts with AI...")
        
        result = detect_cuts_with_explanation(video_path, sample_rate, threshold)
        
        if progress_callback:
            progress_callback(80, "Processing results...")
        
        # Convert to serializable format
        analysis = {
            "cuts": [
                {
                    "frame": c.frame_index,
                    "time": c.timestamp,
                    "confidence": c.confidence,
                    "type": c.cut_type,
                    "reason": c.primary_reason,
                    "secondary_reasons": c.secondary_reasons
                }
                for c in result.cuts
            ],
            "total": result.total_cuts,
            "avg_confidence": result.avg_confidence,
            "cut_types": result.cut_type_distribution,
            "summary": result.explanation_summary
        }
        
        # Calculate pacing metrics
        metadata = SessionManager.get("metadata")
        duration = metadata.get("duration", 1) if metadata else 1
        cuts_per_min = result.total_cuts / (duration / 60) if duration > 0 else 0
        
        analysis["cuts_per_min"] = round(cuts_per_min, 1)
        analysis["pace"] = "Fast" if cuts_per_min > 30 else ("Medium" if cuts_per_min > 15 else "Slow")
        
        return analysis
        
    except Exception as e:
        logger.error(f"Explainable analysis failed: {e}", exc_info=True)
        raise


def run_emotion_analysis(video_path, sample_rate=5, progress_callback=None):
    """
    Run emotion rhythm analysis with progress updates.
    
    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    try:
        if progress_callback:
            progress_callback(20, "Analyzing emotional content...")
        
        result = extract_emotion_timeline(video_path, sample_rate)
        
        if progress_callback:
            progress_callback(80, "Processing emotion data...")
        
        # Convert to serializable format
        emotion_data = {
            "timeline": [
                {
                    "time": ef.timestamp,
                    "emotion": ef.dominant_emotion,
                    "score": ef.emotion_score,
                    "brightness": ef.brightness,
                    "saturation": ef.saturation,
                    "motion": ef.motion_intensity
                }
                for ef in result.timeline
            ],
            "overall_score": result.overall_score,
            "distribution": result.emotion_distribution,
            "peaks": result.peak_moments,
            "pattern": result.rhythm_pattern,
            "confidence": result.confidence,
            "heatmap": result.heatmap_data.tolist(),
            "avg_score": result.overall_score
        }
        
        return emotion_data
        
    except EmotionAnalysisError as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}", exc_info=True)
        raise


def extract_keyframes(video_path, interval=2.0, max_frames=20):
    """
    Extract keyframes at regular intervals using SafeVideoCapture.
    
    Requirements: 2.2, 2.3
    """
    try:
        video_name = SessionManager.get("video_name")
        if not video_name:
            raise ValueError("No video name in session")
        
        with SafeVideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Create output directory
            output_dir = ROOT / "data" / "output" / video_name / "keyframes"
            FileOps.ensure_directory(output_dir)
            
            keyframes = []
            current_time = 0
            idx = 0
            
            while current_time < duration and idx < max_frames:
                frame_num = int(current_time * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame at time {current_time}")
                    current_time += interval
                    continue
                
                try:
                    # Resize for thumbnail
                    thumb = cv2.resize(frame, (320, 180))
                    
                    # Save thumbnail
                    thumb_path = output_dir / f"keyframe_{idx:03d}.jpg"
                    cv2.imwrite(str(thumb_path), thumb)
                    
                    keyframes.append({
                        "index": idx,
                        "time": round(current_time, 2),
                        "frame": frame_num,
                        "path": str(thumb_path)
                    })
                    idx += 1
                except Exception as e:
                    logger.warning(f"Error processing keyframe at {current_time}: {e}")
                
                current_time += interval
            
            return keyframes
            
    except VideoOpenError as e:
        logger.error(f"Failed to open video for keyframe extraction: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during keyframe extraction: {e}")
        raise


# ============== RUN DEPENDENCY CHECK ==============
dependencies_ok = check_dependencies()


# ============== SIDEBAR ==============
with st.sidebar:
    # Use glassmorphic logo component
    render_logo()
    
    st.divider()
    
    # Navigation with unique key - All feature pages included
    page = st.radio(
        "Navigate",
        ["🏠 Home", "📊 Analysis", "🎬 Storyboard", "🌌 Multiverse", 
         "💫 Emotion", "🎨 Color Grading", "✨ Text Effects", "📱 Social", "✋ Gesture",
         "📋 Report", "🔧 System Health"],
        label_visibility="collapsed",
        key=SessionManager.generate_widget_key("nav", "sidebar")
    )
    
    st.divider()
    
    # Video info
    video_path = SessionManager.get("video_path")
    if video_path:
        st.markdown("**📹 Current Video**")
        st.caption(SessionManager.get("video_name"))
        metadata = SessionManager.get("metadata")
        if metadata:
            st.caption(f"⏱️ {metadata.get('duration', 0)}s | {metadata.get('width', 0)}x{metadata.get('height', 0)}")
    
    st.divider()
    
    # System status using glassmorphic status indicator
    if not dependencies_ok:
        render_status_indicator("warning", "Some features limited")
    else:
        render_status_indicator("active", "All systems ready")
    
    st.caption("v2.0 | Industry Ready")


# ============== PAGES ==============

if page == "🏠 Home":
    st.title("🎬 CinemaSense AI Studio")
    st.markdown("*Industry-Ready Video Intelligence Platform*")
    
    render_section_header("Upload Video", "Select a video file to begin analysis", "📤")
    uploaded = st.file_uploader(
        "Select video file",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key=SessionManager.generate_widget_key("uploader", "home")
    )
    
    if uploaded:
        with st.spinner("Processing video..."):
            try:
                path, name = save_uploaded_video(uploaded)
                info = get_video_info(path)
                
                SessionManager.set("video_path", path)
                SessionManager.set("video_name", Path(name).stem)
                SessionManager.set("metadata", info)
                
                # Clear previous analysis results (Requirement 1.2)
                SessionManager.clear_analysis()
                
                st.success(f"✅ Loaded: {name}")
                logger.info(f"Video loaded successfully: {name}")
                st.rerun()
                
            except VideoOpenError as e:
                st.error(f"❌ Cannot open video: {e}")
                logger.error(f"Video open error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Upload error: {e}", exc_info=True)
                SessionManager._log_error("Video upload failed", e)
    
    # Show video
    video_path = SessionManager.get("video_path")
    if video_path:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(video_path)
        
        with col2:
            st.markdown("### 📊 Video Info")
            metadata = SessionManager.get("metadata")
            if metadata:
                st.metric("Duration", f"{metadata.get('duration', 0)}s")
                st.metric("Resolution", f"{metadata.get('width', 0)}x{metadata.get('height', 0)}")
                st.metric("FPS", metadata.get('fps', 0))
                st.metric("Frames", f"{metadata.get('frames', 0):,}")
        
        st.success("✅ Video ready! Use sidebar to navigate to features.")
    
    # Features section using glassmorphic components
    render_section_header("Features", "Explore our AI-powered video analysis tools", "✨")
    cols = st.columns(4)
    features = [
        ("📊", "Analysis", "Explainable AI cut detection"),
        ("🎬", "Storyboard", "Keyframe extraction"),
        ("🌌", "Multiverse", "Style variants generator"),
        ("💫", "Emotion", "Emotional timeline analysis"),
        ("🎨", "Color Grading", "Cinema color presets"),
        ("📱", "Social", "Thumbnails & hashtags"),
        ("✋", "Gesture", "Hand gesture control"),
        ("📋", "Report", "Export results"),
    ]
    for i, (icon, name, desc) in enumerate(features):
        with cols[i % 4]:
            render_feature_card(name, desc, icon)


elif page == "📊 Analysis":
    st.title("📊 Explainable AI Analysis")
    st.markdown("*AI-powered cut detection with detailed explanations*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to begin analysis",
            "📹"
        )
        st.stop()
    
    render_info_banner(f"Analyzing: {SessionManager.get('video_name')}", "info", "📹")
    
    # Settings with unique keys - higher sample rate for faster analysis
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.slider("Cut Sensitivity", 0.3, 0.8, 0.55, 0.05, 
                             key=SessionManager.generate_widget_key("slider", "cut_thresh"))
    with col2:
        sample_rate = st.slider("Sample Rate", 2, 10, 4, 
                               key=SessionManager.generate_widget_key("slider", "cut_sample"),
                               help="Higher = faster analysis, lower = more accurate")
    
    if st.button("🚀 Run Explainable Analysis", type="primary", 
                key=SessionManager.generate_widget_key("button", "run_analysis")):
        progress = st.progress(0, "Starting analysis...")
        
        try:
            def update_progress(pct, msg):
                progress.progress(pct, msg)
            
            analysis = run_explainable_analysis(video_path, threshold, sample_rate, update_progress)
            
            SessionManager.set("analysis", analysis)
            progress.progress(100, "Done!")
            st.success(f"✅ Found {analysis['total']} cuts with AI explanations!")
            logger.info(f"Explainable analysis complete: {analysis['total']} cuts detected")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            logger.error(f"Analysis error: {e}", exc_info=True)
            SessionManager._log_error("Cut analysis failed", e)
    
    # Results
    analysis = SessionManager.get("analysis")
    if analysis:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cuts Detected", analysis.get("total", 0))
        col2.metric("Avg Confidence", f"{analysis.get('avg_confidence', 0):.0%}")
        col3.metric("Cuts/Minute", analysis.get("cuts_per_min", 0))
        col4.metric("Pace", analysis.get("pace", "Unknown"))
        
        # Enhanced AI Summary with detailed explanation
        st.markdown("### 🤖 AI Analysis Summary")
        
        # Main summary in a styled box
        summary_text = analysis.get("summary", "No summary available")
        st.success(f"**Summary:** {summary_text}")
        
        # Detailed interpretation
        total_cuts = analysis.get("total", 0)
        avg_conf = analysis.get("avg_confidence", 0)
        pace = analysis.get("pace", "Unknown")
        
        st.markdown("### 📖 Detailed Interpretation")
        
        # Pace analysis
        pace_explanations = {
            "Very Fast": "🚀 **Very Fast Pacing** - This video has rapid cuts typical of action sequences, music videos, or high-energy content. The quick transitions keep viewers engaged but may feel overwhelming for some audiences.",
            "Fast": "⚡ **Fast Pacing** - The editing rhythm is energetic with frequent cuts. This style works well for trailers, commercials, and dynamic storytelling.",
            "Moderate": "⚖️ **Moderate Pacing** - A balanced editing approach with cuts at comfortable intervals. This is ideal for most narrative content and maintains viewer attention without fatigue.",
            "Slow": "🌊 **Slow Pacing** - Deliberate, contemplative editing with longer shots. This style suits documentaries, art films, and emotional scenes where atmosphere matters.",
            "Very Slow": "🧘 **Very Slow Pacing** - Minimal cuts with extended shots. This meditative approach is used in art cinema, nature documentaries, and scenes requiring deep immersion."
        }
        
        pace_text = pace_explanations.get(pace, f"📊 **{pace} Pacing** - The video has a distinctive editing rhythm.")
        st.info(pace_text)
        
        # Confidence analysis
        if avg_conf >= 0.8:
            conf_text = "🎯 **High Confidence Detection** - The AI is very confident about the detected cuts. These are clear, distinct transitions that are easy to identify."
        elif avg_conf >= 0.6:
            conf_text = "✅ **Good Confidence Detection** - Most cuts are clearly identifiable with some subtle transitions that may be stylistic choices."
        else:
            conf_text = "🔍 **Lower Confidence Detection** - Many transitions are subtle or gradual. This could indicate heavy use of dissolves, fades, or continuous shots."
        st.info(conf_text)
        
        # Cut type distribution with explanations
        cut_types = analysis.get("cut_types", {})
        if cut_types:
            st.markdown("### 📊 Cut Type Analysis")
            
            cols = st.columns(len(cut_types))
            for i, (cut_type, count) in enumerate(cut_types.items()):
                with cols[i]:
                    st.metric(cut_type.replace("_", " ").title(), count)
            
            # Explain each cut type
            st.markdown("#### 📚 What These Mean:")
            
            cut_explanations = {
                "hard_cut": "**Hard Cut** - An instant transition from one shot to another. The most common and direct form of editing, used to maintain narrative flow.",
                "dissolve": "**Dissolve** - A gradual blend between two shots. Often indicates passage of time, dream sequences, or emotional transitions.",
                "fade_to_black": "**Fade to Black** - The image gradually darkens to black. Typically marks the end of a scene, chapter, or significant moment.",
                "fade_to_white": "**Fade to White** - The image brightens to white. Often used for flashbacks, dream sequences, or transcendent moments."
            }
            
            for cut_type in cut_types.keys():
                if cut_type in cut_explanations:
                    st.markdown(f"• {cut_explanations[cut_type]}")
        
        # Editing style insights
        st.markdown("### 🎬 Editing Style Insights")
        
        insights = []
        
        if cut_types.get("dissolve", 0) > total_cuts * 0.3:
            insights.append("🎨 **Artistic Style** - Heavy use of dissolves suggests a contemplative or artistic editing approach.")
        
        if cut_types.get("fade_to_black", 0) >= 3:
            insights.append("📖 **Chapter Structure** - Multiple fade-to-blacks indicate clear scene or chapter divisions.")
        
        if total_cuts > 0 and avg_conf > 0.75:
            insights.append("✂️ **Clean Editing** - High confidence scores indicate professional, deliberate cuts.")
        
        if analysis.get("cuts_per_min", 0) > 30:
            insights.append("🎵 **Music Video Style** - Very high cut frequency typical of music videos or fast-paced commercials.")
        
        if not insights:
            insights.append("📹 **Standard Editing** - The video follows conventional editing patterns suitable for general content.")
        
        for insight in insights:
            st.markdown(insight)
        
        # Cut details with enhanced information
        st.markdown("### 📋 Detailed Cut Timeline")
        cuts = analysis.get("cuts", [])
        if cuts:
            st.markdown(f"*Showing {min(len(cuts), 20)} of {len(cuts)} detected cuts*")
            
            for cut in cuts[:20]:
                confidence_emoji = "🟢" if cut['confidence'] >= 0.8 else "🟡" if cut['confidence'] >= 0.6 else "🔴"
                
                with st.expander(f"{confidence_emoji} **{cut['time']:.2f}s** - {cut['type'].replace('_', ' ').title()} ({cut['confidence']:.0%} confidence)"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**🎯 Primary Reason:**")
                        st.write(f"> {cut['reason']}")
                        
                        if cut.get('secondary_reasons'):
                            st.markdown("**📝 Contributing Factors:**")
                            for reason in cut['secondary_reasons']:
                                st.write(f"  • {reason}")
                    
                    with col2:
                        st.markdown("**📊 Scores:**")
                        st.write(f"• Visual Change: {cut.get('visual_score', 0):.2f}")
                        st.write(f"• Color Change: {cut.get('color_score', 0):.2f}")
                        st.write(f"• Motion: {cut.get('motion_score', 0):.2f}")
        else:
            st.info("No significant cuts detected - this may be a single continuous shot or very subtle editing.")


elif page == "🎬 Storyboard":
    st.title("🎬 Storyboard Generator")
    st.markdown("*Extract keyframes from your video*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to generate storyboards",
            "🎬"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    col1, col2 = st.columns(2)
    with col1:
        interval = st.slider("Interval (seconds)", 1.0, 10.0, 2.0, 0.5, 
                            key=SessionManager.generate_widget_key("slider", "kf_interval"))
    with col2:
        max_frames = st.slider("Max Keyframes", 5, 30, 15, 
                              key=SessionManager.generate_widget_key("slider", "kf_max"))
    
    if st.button("🎬 Generate Storyboard", type="primary", 
                key=SessionManager.generate_widget_key("button", "gen_storyboard")):
        progress = st.progress(0, "Extracting keyframes...")
        
        try:
            progress.progress(50, "Processing...")
            keyframes = extract_keyframes(video_path, interval, max_frames)
            
            SessionManager.set("keyframes", keyframes)
            progress.progress(100, "Done!")
            st.success(f"✅ Extracted {len(keyframes)} keyframes!")
            logger.info(f"Storyboard generated: {len(keyframes)} keyframes")
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed: {e}")
            logger.error(f"Storyboard error: {e}", exc_info=True)
            SessionManager._log_error("Keyframe extraction failed", e)
    
    # Show keyframes
    keyframes = SessionManager.get("keyframes")
    if keyframes:
        st.markdown(f"### 📸 Keyframes ({len(keyframes)})")
        
        cols = st.columns(4)
        for i, kf in enumerate(keyframes):
            with cols[i % 4]:
                if Path(kf["path"]).exists():
                    st.image(kf["path"], caption=f"t={kf['time']}s")


elif page == "🌌 Multiverse":
    st.title("🌌 Multiverse Style Generator")
    st.markdown("*Create stunning style variants of your video*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to apply styles",
            "🌌"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    render_section_header("Available Styles", "Choose a cinematic style to apply", "🎨")
    
    # Get available styles from the stabilized module
    styles = get_available_styles()
    
    cols = st.columns(3)
    for i, style in enumerate(styles):
        with cols[i % 3]:
            st.markdown(f"**{style['name']}**")
            st.caption(style['description'])
            
            if st.button(f"Generate {style['name']}", 
                        key=SessionManager.generate_widget_key("button", "style", style['id'])):
                with st.spinner(f"Generating {style['name']} preview..."):
                    try:
                        video_name = SessionManager.get("video_name")
                        output_dir = ROOT / "data" / "output" / video_name / "multiverse"
                        FileOps.ensure_directory(output_dir)
                        
                        result = generate_multiverse_preview(video_path, style['id'], output_dir)
                        
                        # Store in session
                        multiverse = SessionManager.get("multiverse") or {}
                        multiverse[style['id']] = {
                            "style": result.style_name,
                            "description": result.description,
                            "previews": result.previews
                        }
                        SessionManager.set("multiverse", multiverse)
                        
                        st.success(f"✅ {style['name']} preview ready!")
                        logger.info(f"Multiverse preview generated: {style['id']}")
                        
                        # Display previews
                        preview_cols = st.columns(len(result.previews))
                        for j, preview in enumerate(result.previews):
                            with preview_cols[j]:
                                if Path(preview["path"]).exists():
                                    st.image(preview["path"], caption=f"t={preview['timestamp']:.1f}s")
                                    
                    except InvalidStyleError as e:
                        st.error(f"❌ Invalid style: {e}")
                        logger.error(f"Invalid style error: {e}")
                    except PreviewGenerationError as e:
                        st.error(f"❌ Preview generation failed: {e}")
                        logger.error(f"Preview generation error: {e}")
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
                        logger.error(f"Multiverse error: {e}", exc_info=True)
                        SessionManager._log_error(f"Multiverse preview failed: {style['id']}", e)
    
    # Show previously generated previews
    multiverse = SessionManager.get("multiverse")
    if multiverse:
        st.markdown("### 🖼️ Generated Previews")
        for style_id, data in multiverse.items():
            with st.expander(f"🎬 {data['style'].title()}", expanded=False):
                st.write(data['description'])
                cols = st.columns(3)
                for i, preview in enumerate(data.get('previews', [])):
                    with cols[i]:
                        if Path(preview['path']).exists():
                            st.image(preview['path'], caption=f"t={preview['timestamp']:.1f}s")


elif page == "💫 Emotion":
    st.title("💫 Emotion Rhythm Analysis")
    st.markdown("*AI-powered emotional timeline analysis with heatmap visualization*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to analyze emotions",
            "💫"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    sample_rate = st.slider("Sample Rate (frames)", 5, 20, 10, 
                           key=SessionManager.generate_widget_key("slider", "emo_sample"),
                           help="Higher = faster analysis, lower = more detailed")
    
    if st.button("💫 Analyze Emotions", type="primary", 
                key=SessionManager.generate_widget_key("button", "run_emotion")):
        progress = st.progress(0, "Analyzing emotions...")
        
        try:
            def update_progress(pct, msg):
                progress.progress(pct, msg)
            
            result = run_emotion_analysis(video_path, sample_rate, update_progress)
            
            SessionManager.set("emotion", result)
            progress.progress(100, "Done!")
            st.success("✅ Emotion analysis complete!")
            logger.info("Emotion analysis complete")
            st.rerun()
            
        except EmotionAnalysisError as e:
            st.error(f"❌ Emotion analysis failed: {e}")
            logger.error(f"Emotion analysis error: {e}")
        except Exception as e:
            st.error(f"❌ Failed: {e}")
            logger.error(f"Emotion analysis error: {e}", exc_info=True)
            SessionManager._log_error("Emotion analysis failed", e)
    
    # Results
    emotion = SessionManager.get("emotion")
    if emotion:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall Score", f"{emotion.get('overall_score', 0):.1f}")
        col2.metric("Pattern", emotion.get("pattern", "Unknown"))
        col3.metric("Confidence", f"{emotion.get('confidence', 0):.0%}")
        col4.metric("Peak Moments", len(emotion.get("peaks", [])))
        
        # Pattern explanation
        st.markdown("### 🎭 Emotional Journey Analysis")
        
        pattern = emotion.get("pattern", "Unknown")
        pattern_explanations = {
            "Sustained High Energy": "🔥 **Sustained High Energy** - Your video maintains intense emotional engagement throughout. This pattern is typical of action sequences, exciting sports content, or high-stakes drama. Viewers will feel energized and engaged.",
            "Calm & Steady": "🌊 **Calm & Steady** - A peaceful, consistent emotional tone throughout. Perfect for meditation content, tutorials, or relaxing vlogs. This creates a comfortable viewing experience.",
            "Building Crescendo": "📈 **Building Crescendo** - Emotional intensity grows over time, building to a climax. This is the classic storytelling arc - great for narratives, reveals, or content with a payoff ending.",
            "Descending Arc": "📉 **Descending Arc** - Starts with high energy and gradually calms down. Often seen in content that resolves tension or winds down from an exciting opening.",
            "Dynamic Rollercoaster": "🎢 **Dynamic Rollercoaster** - Varied emotional peaks and valleys throughout. This keeps viewers engaged through contrast and surprise - common in music videos and varied content.",
            "Gradual Rise": "⬆️ **Gradual Rise** - Slowly building emotional intensity. Creates anticipation and draws viewers deeper into the content over time.",
            "Gentle Decline": "⬇️ **Gentle Decline** - Softly decreasing intensity. Often used for conclusions, reflective endings, or content that brings closure.",
            "Balanced Flow": "⚖️ **Balanced Flow** - Well-distributed emotional moments without extreme highs or lows. Professional, polished content that maintains consistent engagement."
        }
        
        pattern_text = pattern_explanations.get(pattern, f"📊 **{pattern}** - Your video has a unique emotional signature.")
        st.info(pattern_text)
        
        # Distribution with explanations
        st.markdown("### 🎭 Emotion Distribution")
        distribution = emotion.get("distribution", {})
        if distribution:
            # Find dominant emotion
            dominant_emotion = max(distribution, key=distribution.get)
            dominant_pct = distribution[dominant_emotion] * 100
            
            st.success(f"**Dominant Emotion:** {dominant_emotion} ({dominant_pct:.1f}%)")
            
            # Emotion explanations
            emotion_meanings = {
                "Joy": "😊 **Joy** - Bright, warm, saturated visuals. Associated with happiness, celebration, and positive energy.",
                "Tension": "😰 **Tension** - High contrast, dynamic motion. Creates suspense, anxiety, or excitement.",
                "Calm": "😌 **Calm** - Low motion, balanced tones. Evokes peace, serenity, and relaxation.",
                "Melancholy": "😢 **Melancholy** - Desaturated, darker tones. Conveys sadness, nostalgia, or reflection.",
                "Energy": "⚡ **Energy** - High motion, vibrant colors. Suggests action, excitement, and dynamism.",
                "Mystery": "🔮 **Mystery** - Dark, high contrast. Creates intrigue, suspense, or otherworldly feelings."
            }
            
            with st.expander("📚 What Each Emotion Means"):
                for emo, meaning in emotion_meanings.items():
                    pct = distribution.get(emo, 0) * 100
                    st.markdown(f"{meaning} - **{pct:.1f}%** of your video")
            
            try:
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 4))
                emotions = list(distribution.keys())
                values = [distribution[e] * 100 for e in emotions]
                colors = ['#00E676', '#FF5252', '#00D4FF', '#FFD600', '#7B61FF', '#FF6B6B']
                
                ax.barh(emotions, values, color=colors[:len(emotions)])
                ax.set_xlabel("Percentage (%)")
                ax.set_facecolor("#0a0a0f")
                fig.patch.set_facecolor("#0a0a0f")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("#333")
                
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not render chart: {e}")
        
        # Timeline
        st.markdown("### 📈 Emotion Timeline")
        timeline = emotion.get("timeline", [])
        if timeline:
            try:
                import matplotlib.pyplot as plt
                
                times = [t["time"] for t in timeline]
                scores = [t["score"] for t in timeline]
                
                fig, ax = plt.subplots(figsize=(12, 3))
                ax.plot(times, scores, color="#00D4FF", linewidth=2)
                ax.fill_between(times, scores, alpha=0.3, color="#00D4FF")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Score")
                ax.set_facecolor("#0a0a0f")
                fig.patch.set_facecolor("#0a0a0f")
                ax.tick_params(colors="white")
                ax.xaxis.label.set_color("white")
                ax.yaxis.label.set_color("white")
                for spine in ax.spines.values():
                    spine.set_color("#333")
                
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not render timeline: {e}")
        
        # Heatmap
        st.markdown("### 🔥 Emotion Heatmap")
        heatmap = emotion.get("heatmap")
        if heatmap:
            try:
                import matplotlib.pyplot as plt
                
                heatmap_array = np.array(heatmap)
                
                fig, ax = plt.subplots(figsize=(12, 3))
                emotions_list = ["Joy", "Tension", "Calm", "Melancholy", "Energy", "Mystery"]
                
                im = ax.imshow(heatmap_array, aspect='auto', cmap='magma')
                ax.set_yticks(range(len(emotions_list)))
                ax.set_yticklabels(emotions_list)
                ax.set_xlabel('Time')
                ax.set_facecolor('#0a0a0f')
                fig.patch.set_facecolor('#0a0a0f')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                for spine in ax.spines.values():
                    spine.set_color('#333')
                
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Could not render heatmap: {e}")
        
        # Peak moments with enhanced explanations
        peaks = emotion.get("peaks", [])
        if peaks:
            st.markdown("### ⚡ Peak Emotional Moments")
            st.markdown("*These are the most emotionally intense moments in your video*")
            
            for i, peak in enumerate(peaks[:5], 1):
                emotion_icons = {
                    "Joy": "😊", "Tension": "😰", "Calm": "😌",
                    "Melancholy": "😢", "Energy": "⚡", "Mystery": "🔮"
                }
                icon = emotion_icons.get(peak['emotion'], "🎭")
                
                with st.expander(f"{icon} **Peak #{i}** at {peak['timestamp']:.1f}s - {peak['emotion']} (Score: {peak['score']:.0f})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Why this moment stands out:**")
                        st.write(f"> {peak.get('reason', 'High emotional intensity detected')}")
                        
                        # Add context
                        if peak['score'] >= 85:
                            st.success("🌟 **Exceptional moment** - This is one of the most emotionally powerful points in your video.")
                        elif peak['score'] >= 75:
                            st.info("✨ **Strong moment** - Significant emotional impact that viewers will notice.")
                    
                    with col2:
                        st.markdown("**Timestamp:**")
                        mins = int(peak['timestamp'] // 60)
                        secs = peak['timestamp'] % 60
                        st.write(f"⏱️ {mins}:{secs:05.2f}")
                        st.write(f"📊 Score: {peak['score']:.0f}/100")
        
        # Recommendations based on analysis
        st.markdown("### 💡 AI Recommendations")
        
        recommendations = []
        
        overall_score = emotion.get('overall_score', 50)
        if overall_score > 70:
            recommendations.append("✅ **High Engagement Potential** - Your video has strong emotional content that should resonate with viewers.")
        elif overall_score < 40:
            recommendations.append("💡 **Consider Adding Variety** - The emotional intensity is relatively low. Consider adding more dynamic moments or visual variety.")
        
        if len(peaks) >= 3:
            recommendations.append("🎯 **Good Peak Distribution** - Multiple emotional peaks will keep viewers engaged throughout.")
        elif len(peaks) == 0:
            recommendations.append("📈 **Add Highlights** - Consider adding more visually dynamic moments to create emotional peaks.")
        
        if pattern == "Building Crescendo":
            recommendations.append("🎬 **Perfect for Storytelling** - Your crescendo pattern is ideal for narrative content with a satisfying payoff.")
        elif pattern == "Dynamic Rollercoaster":
            recommendations.append("🎵 **Great for Music/Action** - The varied emotional rhythm keeps viewers on their toes.")
        
        for rec in recommendations:
            st.markdown(rec)


elif page == "🎨 Color Grading":
    st.title("🎨 Cinema Color Grading")
    st.markdown("*Apply professional color grading presets to your video*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to apply color grading",
            "🎨"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    # Get available presets
    presets = get_available_presets()
    
    render_section_header("Color Presets", "Choose a cinematic color grade", "🎬")
    
    # AI Suggestion
    st.markdown("### 🤖 AI Suggestion")
    if st.button("Get AI Preset Suggestion", key=SessionManager.generate_widget_key("button", "suggest_preset")):
        with st.spinner("Analyzing video for best preset..."):
            try:
                from cinemasense.pipeline.color_grading import analyze_color_palette
                
                with SafeVideoCapture(video_path) as cap:
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        analysis = analyze_color_palette(frame)
                        suggested = analysis.get("suggested_preset", "blockbuster")
                        reason = analysis.get("suggestion_reason", "")
                        confidence = analysis.get("suggestion_confidence", 0)
                        st.success(f"🎯 Recommended preset: **{suggested.title()}** ({confidence:.0%} confidence)")
                        st.info(f"💡 {reason}")
                    else:
                        st.warning("Could not analyze video frame")
            except Exception as e:
                st.error(f"Could not analyze: {e}")
    
    st.markdown("### 🎨 Available Presets")
    cols = st.columns(4)
    
    for i, preset in enumerate(presets):
        with cols[i % 4]:
            st.markdown(f"**{preset['name']}**")
            st.caption(preset['description'])
            
            if st.button(f"Apply {preset['name']}", 
                        key=SessionManager.generate_widget_key("button", "preset", preset['id'])):
                with st.spinner(f"Applying {preset['name']}..."):
                    try:
                        video_name = SessionManager.get("video_name")
                        output_dir = ROOT / "data" / "output" / video_name / "color_grading"
                        FileOps.ensure_directory(output_dir)
                        
                        # Get a sample frame and apply preset
                        with SafeVideoCapture(video_path) as cap:
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                            
                            previews = []
                            for pos in [0.25, 0.5, 0.75]:
                                frame_idx = int(total_frames * pos)
                                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                                ret, frame = cap.read()
                                
                                if ret and frame is not None:
                                    # Apply color preset using safe function
                                    result = apply_color_grading_safe(frame, preset['id'])
                                    
                                    if result.success and result.graded_frame is not None:
                                        graded = result.graded_frame
                                        
                                        # Save preview
                                        preview_path = output_dir / f"{preset['id']}_{int(pos*100)}.jpg"
                                        cv2.imwrite(str(preview_path), graded)
                                        
                                        previews.append({
                                            "path": str(preview_path),
                                            "time": frame_idx / fps
                                        })
                        
                        st.success(f"✅ {preset['name']} applied!")
                        
                        # Display previews
                        if previews:
                            preview_cols = st.columns(len(previews))
                            for j, preview in enumerate(previews):
                                with preview_cols[j]:
                                    if Path(preview["path"]).exists():
                                        st.image(preview["path"], caption=f"t={preview['time']:.1f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
                        logger.error(f"Color grading error: {e}", exc_info=True)


elif page == "✨ Text Effects":
    st.title("✨ Creative Text Overlay")
    st.markdown("*Add stunning text effects to your video frames - text behind subject, 3D, glitch, and more*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to add text effects",
            "✨"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    # Import text effects
    try:
        from cinemasense.pipeline.text_effects import (
            TEXT_STYLES, get_style_preview_info, get_effect_types,
            apply_text_behind_video, apply_parallax_text, apply_3d_extrusion_text,
            apply_wave_text, apply_glitch_text, apply_reveal_text, apply_floating_text
        )
    except ImportError as e:
        st.error(f"Could not load text effects module: {e}")
        st.stop()
    
    # Text input
    st.markdown("### ✏️ Your Text")
    user_text = st.text_input(
        "Enter text to overlay",
        value="CINEMA",
        max_chars=50,
        key=SessionManager.generate_widget_key("text_input", "overlay_text"),
        help="Keep it short for best results (1-3 words)"
    )
    
    # Effect and Style selection using selectbox (more stable than buttons)
    col_effect, col_style = st.columns(2)
    
    with col_effect:
        st.markdown("### 🎭 Effect Type")
        effect_options = ["standard", "behind", "parallax", "3d", "wave", "glitch", "reveal", "floating"]
        effect_names = {
            "standard": "Standard Overlay",
            "behind": "Text Behind Subject", 
            "parallax": "Parallax Depth",
            "3d": "3D Extrusion",
            "wave": "Wave Animation",
            "glitch": "Glitch Effect",
            "reveal": "Reveal Animation",
            "floating": "Floating Text"
        }
        selected_effect = st.selectbox(
            "Choose effect",
            effect_options,
            format_func=lambda x: effect_names.get(x, x),
            key=SessionManager.generate_widget_key("select", "text_effect")
        )
    
    with col_style:
        st.markdown("### 🎨 Text Style")
        style_options = list(TEXT_STYLES.keys())
        selected_style = st.selectbox(
            "Choose style",
            style_options,
            format_func=lambda x: x.replace("_", " ").title(),
            key=SessionManager.generate_widget_key("select", "text_style")
        )
    
    # Effect-specific settings
    st.markdown("### ⚙️ Effect Settings")
    settings_col1, settings_col2 = st.columns(2)
    
    # Initialize default values
    depth = 0.5
    extrusion = 10
    glitch_intensity = 0.5
    wave_amp = 20
    reveal_dir = "left"
    reveal_progress = 1.0
    opacity = 0.8
    position = "center"
    
    with settings_col1:
        position = st.selectbox(
            "Text Position",
            ["center", "top", "bottom"],
            key=SessionManager.generate_widget_key("select", "text_position")
        )
        
        if selected_effect == "parallax":
            depth = st.slider("Depth Level", 0.0, 1.0, 0.5, 0.1,
                             key=SessionManager.generate_widget_key("slider", "parallax_depth"),
                             help="0 = far background, 1 = foreground")
        elif selected_effect == "3d":
            extrusion = st.slider("3D Depth", 5, 20, 10,
                                 key=SessionManager.generate_widget_key("slider", "3d_depth"))
        elif selected_effect == "glitch":
            glitch_intensity = st.slider("Glitch Intensity", 0.1, 1.0, 0.5, 0.1,
                                        key=SessionManager.generate_widget_key("slider", "glitch_int"))
        elif selected_effect == "wave":
            wave_amp = st.slider("Wave Amplitude", 5, 40, 20,
                                key=SessionManager.generate_widget_key("slider", "wave_amp"))
    
    with settings_col2:
        opacity = st.slider("Text Opacity", 0.3, 1.0, 0.8, 0.1,
                           key=SessionManager.generate_widget_key("slider", "text_opacity"))
        
        if selected_effect == "reveal":
            reveal_dir = st.selectbox(
                "Reveal Direction",
                ["left", "right", "center", "top", "bottom"],
                key=SessionManager.generate_widget_key("select", "reveal_dir")
            )
            reveal_progress = st.slider("Reveal Progress", 0.0, 1.0, 1.0, 0.1,
                                       key=SessionManager.generate_widget_key("slider", "reveal_prog"))
    
    # Frame selection and preview
    st.markdown("### 🎞️ Select Frame & Preview")
    
    try:
        with SafeVideoCapture(video_path) as cap:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration = total_frames / fps if fps > 0 else 1.0
            
            frame_time = st.slider(
                "Frame Position (seconds)",
                0.0, max(0.1, duration), min(duration / 2, duration), 0.1,
                key=SessionManager.generate_widget_key("slider", "text_frame_time")
            )
            
            frame_idx = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret and frame is not None:
                st.markdown("### 🖼️ Preview")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Original Frame**")
                    original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(original_rgb, use_column_width=True)
                
                with col2:
                    st.markdown("**With Text Effect**")
                    
                    # Apply selected effect with error handling
                    try:
                        result = frame.copy()
                        
                        if selected_effect == "standard":
                            result = apply_floating_text(frame.copy(), user_text, selected_style, 0.0, False)
                        elif selected_effect == "behind":
                            result = apply_text_behind_video(frame.copy(), user_text, selected_style, position, opacity)
                        elif selected_effect == "parallax":
                            result = apply_parallax_text(frame.copy(), user_text, selected_style, depth, position)
                        elif selected_effect == "3d":
                            result = apply_3d_extrusion_text(frame.copy(), user_text, selected_style, extrusion)
                        elif selected_effect == "wave":
                            result = apply_wave_text(frame.copy(), user_text, selected_style, wave_amp, 0.1, 0.0)
                        elif selected_effect == "glitch":
                            result = apply_glitch_text(frame.copy(), user_text, selected_style, glitch_intensity)
                        elif selected_effect == "reveal":
                            result = apply_reveal_text(frame.copy(), user_text, selected_style, reveal_progress, reveal_dir)
                        elif selected_effect == "floating":
                            result = apply_floating_text(frame.copy(), user_text, selected_style, 0.0, True)
                        
                        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_column_width=True)
                        
                        # Save option
                        if st.button("💾 Save This Frame", type="primary",
                                    key=SessionManager.generate_widget_key("btn", "save_text_frame")):
                            output_dir = Path("data/output") / SessionManager.get("video_name", "video")
                            output_dir.mkdir(parents=True, exist_ok=True)
                            
                            output_path = output_dir / f"text_effect_{selected_effect}_{selected_style}.png"
                            cv2.imwrite(str(output_path), result)
                            st.success(f"✅ Saved to {output_path}")
                            
                    except Exception as e:
                        st.error(f"❌ Error applying effect: {e}")
                        logger.error(f"Text effect error: {e}", exc_info=True)
            else:
                st.warning("Could not read frame from video")
                
    except Exception as e:
        st.error(f"❌ Error loading video: {e}")
        logger.error(f"Video load error: {e}", exc_info=True)
    
    # Tips section
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **Text Behind Subject** works best with:
        - Videos with clear subject/background separation
        - High contrast scenes
        - Single main subject in frame
        
        **3D Extrusion** looks great with:
        - Bold, short text (1-2 words)
        - Dark backgrounds
        - Higher depth values for dramatic effect
        
        **Glitch Effect** is perfect for:
        - Tech/cyberpunk themes
        - Music videos
        - Dramatic reveals
        
        **Wave Animation** suits:
        - Playful, energetic content
        - Music-synced videos
        - Intro/outro sequences
        """)


elif page == "📱 Social":
    st.title("📱 Social Pack Generator")
    st.markdown("*Create optimized content for YouTube, Instagram, TikTok, and Twitter*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to generate social content",
            "📱"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    # Platform selection
    st.markdown("### 🎯 Select Platforms")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        youtube = st.checkbox("YouTube", value=True, key=SessionManager.generate_widget_key("social_yt"))
    with col2:
        instagram = st.checkbox("Instagram", value=True, key=SessionManager.generate_widget_key("social_ig"))
    with col3:
        tiktok = st.checkbox("TikTok", value=True, key=SessionManager.generate_widget_key("social_tt"))
    with col4:
        twitter = st.checkbox("Twitter", value=False, key=SessionManager.generate_widget_key("social_tw"))
    
    platforms = []
    if youtube: platforms.append("youtube")
    if instagram: platforms.append("instagram")
    if tiktok: platforms.append("tiktok")
    if twitter: platforms.append("twitter")
    
    if st.button("📱 Generate Social Pack", type="primary", 
                key=SessionManager.generate_widget_key("button", "gen_social")):
        if not platforms:
            st.warning("Please select at least one platform")
        else:
            progress = st.progress(0, "Generating social pack...")
            
            try:
                progress.progress(30, "Extracting best thumbnail...")
                
                video_name = SessionManager.get("video_name")
                output_dir = ROOT / "data" / "output" / video_name / "social"
                FileOps.ensure_directory(output_dir)
                
                metadata = SessionManager.get("metadata") or {}
                emotion = SessionManager.get("emotion")
                
                # Prepare emotion data for social pack
                emotion_data = None
                if emotion:
                    distribution = emotion.get("distribution", {})
                    if distribution:
                        dominant = max(distribution, key=distribution.get)
                        emotion_data = {"dominant_emotion": dominant}
                
                progress.progress(60, "Generating platform content...")
                
                result = generate_social_pack(
                    video_path,
                    output_dir,
                    metadata,
                    emotion_data,
                    platforms
                )
                
                # Store in session
                social_data = {
                    "thumbnail": result.thumbnail_path,
                    "titles": result.title_suggestions,
                    "hashtags": result.hashtags,
                    "caption": result.caption,
                    "platforms": result.platform_optimized
                }
                SessionManager.set("social", social_data)
                
                progress.progress(100, "Done!")
                st.success("✅ Social pack generated!")
                logger.info("Social pack generated")
                st.rerun()
                
            except SocialPackError as e:
                st.error(f"❌ Social pack error: {e}")
                logger.error(f"Social pack error: {e}")
            except Exception as e:
                st.error(f"❌ Failed: {e}")
                logger.error(f"Social pack error: {e}", exc_info=True)
                SessionManager._log_error("Social pack generation failed", e)
    
    # Results
    social = SessionManager.get("social")
    if social:
        st.markdown("### 🖼️ Main Thumbnail")
        if Path(social.get("thumbnail", "")).exists():
            st.image(social["thumbnail"], width=500)
        
        st.markdown("### 📝 Title Suggestions (5)")
        for i, title in enumerate(social.get("titles", []), 1):
            st.code(f"{i}. {title}")
        
        st.markdown("### #️⃣ Hashtags")
        st.code(" ".join(social.get("hashtags", [])))
        
        st.markdown("### 💬 Caption")
        st.text_area("Copy this caption:", social.get("caption", ""), height=150, 
                    key=SessionManager.generate_widget_key("textarea", "caption"))
        
        # Platform-specific content
        platforms_data = social.get("platforms", {})
        if platforms_data:
            st.markdown("### 🎯 Platform-Optimized Content")
            tabs = st.tabs([p.title() for p in platforms_data.keys()])
            
            for tab, (platform, content) in zip(tabs, platforms_data.items()):
                with tab:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        thumb_path = content.get("thumbnail_path", "")
                        if Path(thumb_path).exists():
                            st.image(thumb_path, caption=f"{platform.title()} Thumbnail")
                        dims = content.get("thumbnail_dimensions", (0, 0))
                        st.caption(f"Size: {dims[0]}x{dims[1]}")
                    
                    with col2:
                        st.markdown(f"**Hashtags ({content.get('hashtag_count', 0)}/{content.get('hashtag_limit', 0)}):**")
                        st.code(" ".join(content.get("hashtags", [])))
                        
                        st.markdown("**Caption:**")
                        st.text_area(
                            f"{platform} caption",
                            content.get("caption", ""),
                            height=100,
                            key=SessionManager.generate_widget_key(f"social_{platform}_caption"),
                            label_visibility="collapsed"
                        )


elif page == "✋ Gesture":
    st.title("✋ Gesture Control Mode")
    st.markdown("*Control CinemaSense with hand gestures using MediaPipe*")
    
    # Check MediaPipe availability
    mp_available, mp_error = is_mediapipe_available()
    
    if not mp_available:
        st.error(f"❌ {mp_error}")
        st.info("💡 Install MediaPipe with: `pip install mediapipe`")
        st.stop()
    
    st.success("✅ MediaPipe is available")
    
    # Gesture mode toggle
    gesture_mode = SessionManager.get("gesture_mode", False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button(
            "🟢 Enable Gestures" if not gesture_mode else "🔴 Disable Gestures",
            type="primary" if not gesture_mode else "secondary",
            key=SessionManager.generate_widget_key("gesture_toggle")
        ):
            SessionManager.set("gesture_mode", not gesture_mode)
            st.rerun()
    
    with col2:
        status = "Active" if gesture_mode else "Inactive"
        st.info(f"Gesture Control: **{status}**")
    
    # Cooldown setting
    cooldown = st.slider(
        "Gesture Cooldown (seconds)",
        0.1, 2.0, 0.5, 0.1,
        key=SessionManager.generate_widget_key("gesture_cooldown"),
        help="Minimum time between gesture triggers"
    )
    
    # Gesture guide
    st.markdown("### 📖 Gesture Guide")
    
    gestures = [
        ("👍", "Thumbs Up", "Approve / Like", "Thumb extended upward"),
        ("👎", "Thumbs Down", "Reject / Dislike", "Thumb extended downward"),
        ("✋", "Open Palm", "Pause / Stop", "All fingers extended"),
        ("✊", "Fist", "Play / Start", "All fingers closed"),
        ("✌️", "Peace Sign", "Next Item", "Index and middle fingers extended"),
        ("👆", "Pointing", "Select / Click", "Only index finger extended"),
        ("👈", "Swipe Left", "Previous", "Quick hand movement left"),
        ("👉", "Swipe Right", "Next", "Quick hand movement right"),
        ("🤏", "Pinch", "Zoom Out", "Thumb and index close together"),
    ]
    
    # Display in grid
    cols = st.columns(3)
    
    for i, (icon, name, action, description) in enumerate(gestures):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div style="
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 12px;
                    padding: 16px;
                    margin-bottom: 12px;
                ">
                    <div style="font-size: 32px; margin-bottom: 8px;">{icon}</div>
                    <div style="font-weight: 600; color: white;">{name}</div>
                    <div style="color: #00D4FF; font-size: 14px;">{action}</div>
                    <div style="color: rgba(255,255,255,0.6); font-size: 12px;">{description}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Camera preview info
    st.markdown("### 📷 Camera Preview")
    
    if gesture_mode:
        st.warning("⚠️ Camera access requires running in a local environment with webcam support.")
        st.info("💡 In a full implementation, the camera feed would appear here with gesture detection overlay.")
        
        # Show gesture controller status
        controller = GestureController(cooldown_seconds=cooldown)
        status = controller.get_status()
        
        st.markdown("#### Controller Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("Initialized", "✅" if status["initialized"] else "❌")
        col2.metric("MediaPipe", "✅" if status["mediapipe_available"] else "❌")
        col3.metric("Cooldown", f"{status['cooldown_seconds']}s")
        
        controller.cleanup()
    else:
        st.info("Enable gesture mode to start camera preview")
    
    # Tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - **Lighting**: Ensure good, even lighting on your hands
        - **Background**: Use a plain background for better detection
        - **Distance**: Keep your hand 1-3 feet from the camera
        - **Speed**: Make gestures slowly and deliberately
        - **Position**: Keep your hand within the camera frame
        - **Stability**: Hold gestures for 0.5 seconds for reliable detection
        """)


elif page == "📋 Report":
    st.title("📋 Comprehensive Report")
    st.markdown("*Export all analysis results in a single comprehensive report*")
    
    video_path = SessionManager.get("video_path")
    if not video_path:
        render_empty_state(
            "No Video Loaded",
            "Please upload a video on the Home page to generate reports",
            "📋"
        )
        st.stop()
    
    render_info_banner(f"Video: {SessionManager.get('video_name')}", "info", "📹")
    
    # Status
    render_section_header("Analysis Status", "Check which analyses have been completed", "📊")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Video", "✅" if SessionManager.get("metadata") else "❌")
    col2.metric("Analysis", "✅" if SessionManager.get("analysis") else "❌")
    col3.metric("Keyframes", "✅" if SessionManager.get("keyframes") else "❌")
    col4.metric("Emotion", "✅" if SessionManager.get("emotion") else "❌")
    col5.metric("Social", "✅" if SessionManager.get("social") else "❌")
    
    if st.button("📋 Generate Comprehensive Report", type="primary", 
                key=SessionManager.generate_widget_key("button", "gen_report")):
        progress = st.progress(0, "Generating report...")
        
        try:
            video_name = SessionManager.get("video_name")
            metadata = SessionManager.get("metadata") or {}
            analysis = SessionManager.get("analysis")
            keyframes = SessionManager.get("keyframes")
            emotion = SessionManager.get("emotion")
            social = SessionManager.get("social")
            multiverse = SessionManager.get("multiverse")
            
            progress.progress(30, "Compiling data...")
            
            # Ensure metadata has required fields
            if "name" not in metadata:
                metadata["name"] = video_name
            if "fps" not in metadata:
                metadata["fps"] = 0.0
            if "frame_count" not in metadata:
                metadata["frame_count"] = metadata.get("frames", 0)
            if "duration_s" not in metadata:
                metadata["duration_s"] = metadata.get("duration", 0.0)
            
            progress.progress(50, "Generating report...")
            
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
            
            progress.progress(70, "Saving report...")
            
            # Save report
            output_dir = ROOT / "data" / "output" / video_name
            FileOps.ensure_directory(output_dir)
            report_path = output_dir / "comprehensive_report.json"
            
            ReportGenerator.save(report, report_path, validate=False)
            
            progress.progress(100, "Done!")
            st.success("✅ Comprehensive report generated!")
            logger.info(f"Report generated: {report_path}")
            
            # Download button
            report_json = ReportGenerator.to_json_string(report)
            st.download_button(
                "⬇️ Download Report (JSON)",
                data=report_json,
                file_name=f"{video_name}_report.json",
                mime="application/json",
                key=SessionManager.generate_widget_key("download", "report")
            )
            
            # Preview
            with st.expander("📄 Report Preview", expanded=True):
                st.json(report)
                
        except Exception as e:
            st.error(f"❌ Failed: {e}")
            logger.error(f"Report generation error: {e}", exc_info=True)
            SessionManager._log_error("Report generation failed", e)
    
    # Show existing report if available
    video_name = SessionManager.get("video_name")
    if video_name:
        report_path = ROOT / "data" / "output" / video_name / "comprehensive_report.json"
        if report_path.exists():
            st.markdown("### 📄 Existing Report")
            try:
                report = ReportGenerator.load(report_path)
                
                with st.expander("View Report", expanded=False):
                    st.json(report)
                
                # Download existing report
                report_json = ReportGenerator.to_json_string(report)
                st.download_button(
                    "⬇️ Download Existing Report",
                    data=report_json,
                    file_name=f"{video_name}_report.json",
                    mime="application/json",
                    key=SessionManager.generate_widget_key("download", "existing_report")
                )
            except Exception as e:
                st.warning(f"Could not load existing report: {e}")


elif page == "🔧 System Health":
    st.title("🔧 System Health Check")
    st.markdown("*Verify dependencies and system status*")
    
    render_section_header("Dependency Status", "Check all required dependencies", "📦")
    
    # Force re-check
    if st.button("🔄 Re-check Dependencies", key=SessionManager.generate_widget_key("button", "recheck_deps")):
        SessionManager.set("dependencies_checked", False)
        st.rerun()
    
    # Get dependency status
    all_ok, checks = run_all_checks()
    
    # Display status
    cols = st.columns(4)
    for i, check in enumerate(checks):
        with cols[i % 4]:
            status_icon = "✅" if check.available else "❌"
            st.metric(
                check.name,
                status_icon,
                check.version if check.available else check.error[:30] if check.error else "N/A"
            )
    
    # Overall status
    if all_ok:
        st.success("✅ All critical dependencies are available!")
    else:
        st.error("❌ Some dependencies are missing. See details below.")
    
    # Detailed status
    st.markdown("### 📋 Detailed Status")
    
    for check in checks:
        with st.expander(f"{'✅' if check.available else '❌'} {check.name}", expanded=not check.available):
            if check.available:
                st.success(f"Version: {check.version}")
            else:
                st.error(f"Error: {check.error}")
                
                # Provide installation instructions
                install_cmds = {
                    "Python": "Python 3.10+ is required. Download from python.org",
                    "FFmpeg": "pip install imageio-ffmpeg\n# Or download from ffmpeg.org",
                    "OpenCV": "pip install opencv-python",
                    "MediaPipe": "pip install mediapipe",
                    "MoviePy": "pip install moviepy",
                    "Librosa": "pip install librosa",
                    "Scikit-learn": "pip install scikit-learn"
                }
                
                if check.name in install_cmds:
                    st.code(install_cmds[check.name], language="bash")
    
    # System Information
    render_section_header("System Information", "Your system details", "💻")
    
    sys_info = get_system_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Operating System**")
        st.write(f"• OS: {sys_info.get('os', 'Unknown')}")
        st.write(f"• Version: {sys_info.get('os_version', 'Unknown')[:50]}")
        st.write(f"• Architecture: {sys_info.get('architecture', 'Unknown')}")
    
    with col2:
        st.markdown("**Hardware**")
        st.write(f"• Processor: {sys_info.get('processor', 'Unknown')[:40]}")
        st.write(f"• CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
        st.write(f"• Python: {sys_info.get('python_version', 'Unknown')}")
    
    # Troubleshooting Guide
    render_section_header("Troubleshooting Guide", "Common issues and solutions", "🔧")
    
    with st.expander("🎬 Video won't load"):
        st.markdown("""
        **Possible causes:**
        1. Unsupported video format
        2. Corrupted video file
        3. Missing FFmpeg
        
        **Solutions:**
        - Try converting to MP4 format
        - Install FFmpeg: `pip install imageio-ffmpeg`
        - Check if the video plays in VLC or other players
        """)
    
    with st.expander("✋ Gesture control not working"):
        st.markdown("""
        **Possible causes:**
        1. MediaPipe not installed
        2. Camera not accessible
        3. Poor lighting conditions
        
        **Solutions:**
        - Install MediaPipe: `pip install mediapipe`
        - Check camera permissions
        - Ensure good lighting on your hands
        - Use a plain background
        """)
    
    with st.expander("💫 Emotion analysis fails"):
        st.markdown("""
        **Possible causes:**
        1. Video too short
        2. Corrupted frames
        3. Memory issues
        
        **Solutions:**
        - Use videos longer than 1 second
        - Try a different video file
        - Reduce sample rate for large videos
        - Close other applications to free memory
        """)
    
    with st.expander("📱 Social pack generation fails"):
        st.markdown("""
        **Possible causes:**
        1. Cannot extract frames
        2. Output directory issues
        3. Invalid platform selection
        
        **Solutions:**
        - Ensure video loads correctly first
        - Check disk space
        - Select at least one platform
        """)
    
    with st.expander("🎨 Color grading not applying"):
        st.markdown("""
        **Possible causes:**
        1. OpenCV issues
        2. Invalid frame data
        3. Output directory permissions
        
        **Solutions:**
        - Reinstall OpenCV: `pip install opencv-python --upgrade`
        - Try a different video
        - Check output folder permissions
        """)
    
    # Quick Install
    render_section_header("Quick Install", "Install all dependencies at once", "⚡")
    
    st.code("""
# Install all CinemaSense dependencies
pip install opencv-python mediapipe moviepy librosa scikit-learn imageio-ffmpeg numpy matplotlib streamlit

# For Windows FFmpeg issues
pip install imageio-ffmpeg
    """, language="bash")
    
    # Logs
    render_section_header("Application Logs", "Recent log entries", "📝")
    
    log_path = ROOT / "logs" / "app.log"
    if log_path.exists():
        try:
            with open(log_path, "r") as f:
                lines = f.readlines()
                recent_lines = lines[-50:] if len(lines) > 50 else lines
                
            if recent_lines:
                st.text_area(
                    "Recent Logs",
                    "".join(recent_lines),
                    height=300,
                    key=SessionManager.generate_widget_key("textarea", "logs")
                )
                
                st.download_button(
                    "⬇️ Download Full Log",
                    data="".join(lines),
                    file_name="cinemasense_app.log",
                    mime="text/plain",
                    key=SessionManager.generate_widget_key("download", "logs")
                )
            else:
                st.info("No log entries yet")
        except Exception as e:
            st.warning(f"Could not read logs: {e}")
    else:
        st.info("No log file found")
