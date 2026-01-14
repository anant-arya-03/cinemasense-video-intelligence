"""
Home Page - Video upload and overview
"""

import streamlit as st
from pathlib import Path
import cv2

from cinemasense.core.session import (
    create_video_session, get_video_session, 
    set_session_value, generate_unique_key
)
from cinemasense.storage.paths import ProjectPaths
from cinemasense.storage.io import unique_filename, save_uploaded_file
from cinemasense.pipeline.metadata import get_video_metadata
from cinemasense.ui.glassmorphic import (
    glass_card, 
    render_feature_card,
    render_section_header,
    render_info_banner,
    render_empty_state
)
from cinemasense.constants import SUPPORTED_VIDEO_FORMATS


def render():
    """Render the home page"""
    st.title("🎬 Welcome to CinemaSense")
    st.markdown("*AI-Powered Video Intelligence Platform*")
    
    # Initialize paths
    paths = ProjectPaths()
    paths.ensure_all_dirs()
    
    # Check for existing session
    session = get_video_session()
    
    # File upload section
    st.markdown("### 📤 Upload Your Video")
    
    uploaded = st.file_uploader(
        "Drop your video file here",
        type=SUPPORTED_VIDEO_FORMATS,
        key=generate_unique_key("home_uploader"),
        help=f"Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
    )
    
    if uploaded:
        with st.spinner("Processing video..."):
            try:
                # Save uploaded file
                fname = unique_filename(uploaded.name)
                video_path = paths.input / fname
                save_uploaded_file(uploaded, video_path)
                
                # Get metadata
                fps, frame_count, width, height, duration_s = get_video_metadata(str(video_path))
                
                metadata = {
                    "fps": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "duration_s": duration_s
                }
                
                # Create session
                create_video_session(str(video_path), Path(fname).stem, metadata)
                
                st.success(f"✅ Video loaded: {fname}")
                
            except Exception as e:
                st.error(f"❌ Failed to load video: {str(e)}")
                return
    
    # Display current video if loaded
    session = get_video_session()
    if session and session.video_path:
        st.markdown("### 📹 Current Video")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.video(session.video_path)
        
        with col2:
            st.markdown("#### Video Info")
            if session.metadata:
                st.metric("Duration", f"{session.metadata.get('duration_s', 0):.1f}s")
                st.metric("Resolution", f"{session.metadata.get('width', 0)}×{session.metadata.get('height', 0)}")
                st.metric("FPS", f"{session.metadata.get('fps', 0):.1f}")
                st.metric("Frames", f"{session.metadata.get('frame_count', 0):,}")
        
        st.info("👈 Use the sidebar to navigate to analysis features")
    
    # Feature overview
    st.markdown("### ✨ Available Features")
    
    cols = st.columns(4)
    
    features = [
        ("📊", "Smart Analysis", "Cut detection, rhythm scoring, quality metrics"),
        ("🎬", "Storyboard", "Automatic keyframe extraction and thumbnails"),
        ("🌌", "Multiverse", "Generate style variants: Romantic, Thriller, Anime"),
        ("💫", "Emotion Rhythm", "AI-powered emotional timeline analysis"),
        ("📱", "Social Pack", "Thumbnails, titles, hashtags for all platforms"),
        ("✋", "Gesture Control", "Control the app with hand gestures"),
        ("🎨", "Color Grading", "Cinema-quality color presets"),
        ("📋", "Reports", "Comprehensive JSON exports")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            render_feature_card(title, desc, icon)