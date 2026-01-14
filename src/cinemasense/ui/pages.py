"""
Streamlit UI components and plotting functions
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


def render_video_metadata(fps: float, frame_count: int, width: int, height: int, duration_s: float):
    """Render video metadata in columns"""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FPS", f"{fps:.2f}")
    c2.metric("Frames", f"{frame_count}")
    c3.metric("Resolution", f"{width}×{height}")
    c4.metric("Duration", f"{duration_s:.1f}s")


def render_settings_panel():
    """Render analysis settings panel"""
    with st.expander("⚙️ Settings", expanded=True):
        sample_every = st.slider("Sample every N frames (speed vs accuracy)", 1, 10, 2)
        threshold = st.slider("Cut threshold (higher = fewer cuts)", 0.20, 1.20, 0.55)
        do_audio = st.toggle("Analyze audio energy (RMS)", value=True)
        keyframe_interval = st.slider("Keyframe interval (seconds)", 1.0, 10.0, 5.0)
    return sample_every, threshold, do_audio, keyframe_interval


def render_rhythm_summary(cut_count: int, cuts_per_min: float, pace: str):
    """Render rhythm analysis summary"""
    st.subheader("📌 Rhythm Summary")
    m1, m2, m3 = st.columns(3)
    m1.metric("Detected cuts", f"{cut_count}")
    m2.metric("Cuts per minute", f"{cuts_per_min:.2f}")
    m3.metric("Pace", pace)


def plot_difference_curve(diff_series: list):
    """Plot visual difference curve"""
    st.subheader("📈 Visual Difference Curve")
    fig1 = plt.figure(figsize=(10, 4))
    plt.plot(diff_series)
    plt.xlabel("Sample index")
    plt.ylabel("Difference (1 - histogram correlation)")
    plt.title("Frame-to-frame difference (proxy for cut points)")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True)


def plot_audio_energy(times_s: list, rms: list):
    """Plot audio energy over time"""
    st.subheader("🎧 Audio Energy (RMS)")
    fig2 = plt.figure(figsize=(10, 4))
    plt.plot(times_s, rms)
    plt.xlabel("Time (s)")
    plt.ylabel("RMS")
    plt.title("Audio energy over time")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)


def render_storyboard_grid(keyframes: List[Dict], base_path: Path):
    """Render storyboard as a grid of thumbnails"""
    if not keyframes:
        st.warning("No keyframes available")
        return
    
    st.subheader(f"🎬 Storyboard ({len(keyframes)} keyframes)")
    
    # Display in grid format
    cols_per_row = 4
    for i in range(0, len(keyframes), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx < len(keyframes):
                keyframe = keyframes[idx]
                thumbnail_path = base_path / keyframe["thumbnail_path"]
                
                if thumbnail_path.exists():
                    with col:
                        st.image(str(thumbnail_path), caption=f"t={keyframe['time_s']:.1f}s")


def render_scenes_table(scenes: List[Dict]):
    """Render scenes as a table"""
    if not scenes:
        st.warning("No scenes detected")
        return
    
    st.subheader(f"🎭 Scene Analysis ({len(scenes)} scenes)")
    
    # Create table data
    table_data = []
    for scene in scenes:
        table_data.append({
            "Scene": scene["scene_id"] + 1,
            "Start (s)": f"{scene['start_time']:.1f}",
            "End (s)": f"{scene['end_time']:.1f}",
            "Duration (s)": f"{scene['duration']:.1f}",
            "Cuts": scene["cut_count"]
        })
    
    st.dataframe(table_data, use_container_width=True)


def plot_mood_timeline(mood_timeline: List[Dict]):
    """Plot mood progression over time"""
    if not mood_timeline:
        return
    
    st.subheader("🎨 Mood Timeline")
    
    times = [frame["time_s"] for frame in mood_timeline]
    brightness = [frame["brightness"] for frame in mood_timeline]
    saturation = [frame["saturation"] for frame in mood_timeline]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Brightness over time
    ax1.plot(times, brightness, 'b-', label='Brightness', linewidth=2)
    ax1.set_ylabel('Brightness')
    ax1.set_title('Mood Characteristics Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Saturation over time
    ax2.plot(times, saturation, 'r-', label='Saturation', linewidth=2)
    ax2.set_ylabel('Saturation')
    ax2.set_xlabel('Time (s)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def render_anomalies_section(anomalous_segments: List[Dict]):
    """Render anomaly detection results"""
    if not anomalous_segments:
        st.info("No anomalies detected")
        return
    
    st.subheader(f"⚠️ Anomaly Detection ({len(anomalous_segments)} anomalies)")
    
    for segment in anomalous_segments:
        with st.expander(f"Anomaly at {segment['start_time']:.1f}s - {segment['end_time']:.1f}s"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Type:** {segment.get('anomaly_type', 'Unknown')}")
                st.write(f"**Duration:** {segment['duration']:.1f}s")
                st.write(f"**Anomaly Score:** {segment['anomaly_score']:.3f}")
            
            with col2:
                st.write(f"**Cut Density:** {segment['cut_density']:.2f} cuts/s")
                st.write(f"**Motion:** {segment['motion_magnitude']:.1f}")
                st.write(f"**Brightness Var:** {segment['brightness_variance']:.1f}")


def render_health_check():
    """Render system health check"""
    st.sidebar.subheader("🔧 System Health")
    
    if st.sidebar.button("Run Health Check"):
        with st.sidebar:
            with st.spinner("Checking system..."):
                health_results = []
                
                # Check imports
                try:
                    import cv2
                    health_results.append("✅ OpenCV available")
                except ImportError:
                    health_results.append("❌ OpenCV missing")
                
                try:
                    import librosa
                    health_results.append("✅ Librosa available")
                except ImportError:
                    health_results.append("❌ Librosa missing")
                
                try:
                    import sklearn
                    health_results.append("✅ Scikit-learn available")
                except ImportError:
                    health_results.append("❌ Scikit-learn missing")
                
                # Check directories
                from cinemasense.storage.paths import ProjectPaths
                paths = ProjectPaths()
                
                if paths.input.exists():
                    health_results.append("✅ Input directory exists")
                else:
                    health_results.append("❌ Input directory missing")
                
                if paths.output.exists():
                    health_results.append("✅ Output directory exists")
                else:
                    health_results.append("❌ Output directory missing")
                
                for result in health_results:
                    st.write(result)


def calculate_pace(cuts_per_min: float) -> str:
    """Calculate video pace based on cuts per minute"""
    if cuts_per_min < 15:
        return "Slow 🐢"
    elif cuts_per_min < 30:
        return "Medium 🐇"
    else:
        return "Fast ⚡"