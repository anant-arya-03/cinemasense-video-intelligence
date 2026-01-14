"""
Storyboard Page - Keyframe extraction and visualization
"""

import streamlit as st
from pathlib import Path

from cinemasense.core.session import (
    get_video_session, update_video_session, generate_unique_key
)
from cinemasense.storage.paths import ProjectPaths
from cinemasense.pipeline.keyframes import extract_keyframes_from_cuts, extract_keyframes_interval
from cinemasense.constants import MAX_STORYBOARD_THUMBNAILS


def render():
    """Render the storyboard page"""
    st.title("🎬 Storyboard Generator")
    st.markdown("*Extract keyframes and create visual storyboards*")
    
    session = get_video_session()
    if not session or not session.video_path:
        st.warning("⚠️ Please upload a video on the Home page first")
        return
    
    st.info(f"📹 Video: {Path(session.video_path).name}")
    
    # Settings
    with st.expander("⚙️ Storyboard Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            method = st.radio(
                "Extraction Method",
                ["cuts", "interval"],
                format_func=lambda x: "Based on Cuts" if x == "cuts" else "Time Intervals",
                key=generate_unique_key("storyboard_method")
            )
        
        with col2:
            if method == "interval":
                interval = st.slider(
                    "Interval (seconds)",
                    1.0, 30.0, 5.0,
                    key=generate_unique_key("storyboard_interval")
                )
            else:
                interval = 5.0
            
            max_frames = st.slider(
                "Max Keyframes",
                10, MAX_STORYBOARD_THUMBNAILS, 30,
                key=generate_unique_key("storyboard_max")
            )
    
    # Generate button
    if st.button("🎬 Generate Storyboard", type="primary", key=generate_unique_key("storyboard_gen")):
        generate_storyboard(session, method, interval, max_frames)
    
    # Display storyboard
    if session.storyboard_keyframes:
        display_storyboard(session)


def generate_storyboard(session, method, interval, max_frames):
    """Generate storyboard keyframes"""
    progress = st.progress(0, "Extracting keyframes...")
    
    try:
        paths = ProjectPaths()
        storyboard_dir = paths.get_storyboard_dir(session.video_stem)
        
        # Get cut times if using cuts method
        cut_times = None
        if method == "cuts" and session.analysis_results:
            cuts = session.analysis_results.get("cuts", {})
            cut_times = [c["timestamp"] for c in cuts.get("details", [])]
        
        if method == "cuts" and not cut_times:
            st.warning("⚠️ No cuts detected. Run analysis first or use interval method.")
            return
        
        progress.progress(30, "Processing frames...")
        
        if method == "cuts":
            keyframes = extract_keyframes_from_cuts(
                session.video_path, cut_times, storyboard_dir, max_frames
            )
        else:
            keyframes = extract_keyframes_interval(
                session.video_path, interval, storyboard_dir, max_frames
            )
        
        progress.progress(90, "Saving...")
        
        update_video_session(storyboard_keyframes=keyframes)
        
        progress.progress(100, "Complete!")
        st.success(f"✅ Generated {len(keyframes)} keyframes!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Storyboard generation failed: {str(e)}")


def display_storyboard(session):
    """Display the generated storyboard"""
    keyframes = session.storyboard_keyframes
    
    st.markdown(f"### 📸 Storyboard ({len(keyframes)} keyframes)")
    
    # Grid display
    cols_per_row = 4
    
    for i in range(0, len(keyframes), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(keyframes):
                kf = keyframes[idx]
                
                # Construct full path
                paths = ProjectPaths()
                thumb_path = paths.root / kf.get("thumbnail_path", "")
                
                if thumb_path.exists():
                    with col:
                        st.image(
                            str(thumb_path),
                            caption=f"t={kf['time_s']:.1f}s",
                            use_container_width=True
                        )
    
    # Statistics
    with st.expander("📊 Storyboard Statistics"):
        if keyframes:
            time_span = keyframes[-1]["time_s"] - keyframes[0]["time_s"]
            avg_interval = time_span / max(len(keyframes) - 1, 1)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Keyframes", len(keyframes))
            col2.metric("Time Span", f"{time_span:.1f}s")
            col3.metric("Avg Interval", f"{avg_interval:.1f}s")