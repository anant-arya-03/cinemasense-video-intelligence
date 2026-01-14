"""
Emotion Rhythm Page - AI-powered emotional timeline analysis
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cinemasense.core.session import (
    get_video_session, update_video_session, generate_unique_key
)
from cinemasense.storage.paths import ProjectPaths
from cinemasense.storage.io import save_json
from cinemasense.pipeline.emotion_rhythm import extract_emotion_timeline


def render():
    """Render the emotion rhythm page"""
    st.title("💫 Emotion Rhythm Score")
    st.markdown("*AI-powered emotional timeline analysis with heatmap visualization*")
    
    session = get_video_session()
    if not session or not session.video_path:
        st.warning("⚠️ Please upload a video on the Home page first")
        return
    
    st.info(f"📹 Video: {Path(session.video_path).name}")
    
    # Settings
    with st.expander("⚙️ Analysis Settings", expanded=True):
        sample_rate = st.slider(
            "Sample Rate (frames)",
            1, 15, 5,
            key=generate_unique_key("emotion_sample"),
            help="Higher = faster but less accurate"
        )
    
    # Analyze button
    if st.button("💫 Analyze Emotions", type="primary", key=generate_unique_key("emotion_run")):
        analyze_emotions(session, sample_rate)
    
    # Display results
    if session.emotion_timeline:
        display_emotion_results(session)


def analyze_emotions(session, sample_rate):
    """Run emotion analysis"""
    progress = st.progress(0, "Analyzing emotional content...")
    
    try:
        progress.progress(30, "Processing frames...")
        
        result = extract_emotion_timeline(session.video_path, sample_rate)
        
        progress.progress(80, "Computing metrics...")
        
        # Convert to serializable format
        timeline_data = [
            {
                "timestamp": ef.timestamp,
                "emotion": ef.dominant_emotion,
                "score": ef.emotion_score,
                "brightness": ef.brightness,
                "saturation": ef.saturation,
                "motion": ef.motion_intensity
            }
            for ef in result.timeline
        ]
        
        emotion_data = {
            "timeline": timeline_data,
            "overall_score": result.overall_score,
            "distribution": result.emotion_distribution,
            "peaks": result.peak_moments,
            "pattern": result.rhythm_pattern,
            "confidence": result.confidence,
            "heatmap": result.heatmap_data.tolist()
        }
        
        update_video_session(emotion_timeline=emotion_data)
        
        # Save to file
        paths = ProjectPaths()
        output_dir = paths.get_session_dir(session.video_stem)
        save_json(emotion_data, output_dir / "emotion_analysis.json")
        
        progress.progress(100, "Complete!")
        st.success("✅ Emotion analysis completed!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Emotion analysis failed: {str(e)}")


def display_emotion_results(session):
    """Display emotion analysis results"""
    data = session.emotion_timeline
    
    # Overall metrics
    st.markdown("### 📊 Emotion Overview")
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Overall Score", f"{data.get('overall_score', 0):.1f}")
    
    with cols[1]:
        st.metric("Rhythm Pattern", data.get('pattern', 'Unknown'))
    
    with cols[2]:
        st.metric("Confidence", f"{data.get('confidence', 0):.0%}")
    
    with cols[3]:
        st.metric("Peak Moments", len(data.get('peaks', [])))
    
    # Emotion distribution
    st.markdown("### 🎭 Emotion Distribution")
    
    distribution = data.get('distribution', {})
    if distribution:
        fig, ax = plt.subplots(figsize=(10, 4))
        
        emotions = list(distribution.keys())
        values = list(distribution.values())
        colors = ['#00E676', '#FF5252', '#00D4FF', '#FFD600', '#7B61FF', '#FF6B6B']
        
        bars = ax.barh(emotions, values, color=colors[:len(emotions)])
        ax.set_xlabel('Proportion')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        st.pyplot(fig)
        plt.close()
    
    # Timeline visualization
    st.markdown("### 📈 Emotion Timeline")
    
    timeline = data.get('timeline', [])
    if timeline:
        times = [t['timestamp'] for t in timeline]
        scores = [t['score'] for t in timeline]
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(times, scores, color='#00D4FF', linewidth=2)
        ax.fill_between(times, scores, alpha=0.3, color='#00D4FF')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Emotion Score')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        st.pyplot(fig)
        plt.close()
    
    # Heatmap
    st.markdown("### 🔥 Emotion Heatmap")
    
    heatmap = data.get('heatmap')
    if heatmap:
        heatmap_array = np.array(heatmap)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        emotions = ["Joy", "Tension", "Calm", "Melancholy", "Energy", "Mystery"]
        
        im = ax.imshow(heatmap_array, aspect='auto', cmap='magma')
        ax.set_yticks(range(len(emotions)))
        ax.set_yticklabels(emotions)
        ax.set_xlabel('Time')
        ax.set_facecolor('#0a0a0f')
        fig.patch.set_facecolor('#0a0a0f')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        st.pyplot(fig)
        plt.close()
    
    # Peak moments
    peaks = data.get('peaks', [])
    if peaks:
        st.markdown("### ⚡ Peak Moments")
        
        for peak in peaks[:5]:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write(f"**{peak['timestamp']:.1f}s**")
            with col2:
                st.write(peak.get('reason', ''))
            with col3:
                st.write(f"{peak['emotion']} ({peak['score']:.0f})")