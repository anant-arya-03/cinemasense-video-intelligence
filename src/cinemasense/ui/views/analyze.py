"""
Analysis Page - Core video analysis with explainable AI
"""

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from cinemasense.core.session import (
    get_video_session, update_video_session, 
    generate_unique_key, log_error
)
from cinemasense.storage.paths import ProjectPaths
from cinemasense.storage.io import save_json
from cinemasense.pipeline.explainable_ai import detect_cuts_with_explanation
from cinemasense.pipeline.audio import extract_audio_features, analyze_audio_energy
from cinemasense.pipeline.quality import analyze_motion_magnitude, analyze_brightness_variance
from cinemasense.features.rhythm import analyze_rhythm_patterns, calculate_pacing_score
from cinemasense.ui.glassmorphic import (
    glass_metric,
    render_section_header,
    render_info_banner,
    render_empty_state
)


def render():
    """Render the analysis page"""
    st.title("📊 Video Analysis")
    st.markdown("*Explainable AI-powered cut detection and rhythm analysis*")
    
    session = get_video_session()
    if not session or not session.video_path:
        st.warning("⚠️ Please upload a video on the Home page first")
        return
    
    st.info(f"📹 Analyzing: {Path(session.video_path).name}")
    
    # Settings
    with st.expander("⚙️ Analysis Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sample_rate = st.slider(
                "Sample Rate",
                1, 10, 2,
                key=generate_unique_key("analyze_sample"),
                help="Sample every N frames"
            )
        
        with col2:
            threshold = st.slider(
                "Cut Threshold",
                0.2, 1.2, 0.55,
                key=generate_unique_key("analyze_threshold"),
                help="Higher = fewer cuts detected"
            )
        
        with col3:
            analyze_audio = st.toggle(
                "Analyze Audio",
                value=True,
                key=generate_unique_key("analyze_audio")
            )
    
    # Run analysis
    if st.button("🚀 Run Analysis", type="primary", key=generate_unique_key("analyze_run")):
        run_analysis(session, sample_rate, threshold, analyze_audio)
    
    # Display results if available
    if session.analysis_results:
        display_results(session)


def run_analysis(session, sample_rate, threshold, analyze_audio):
    """Run the video analysis pipeline"""
    progress = st.progress(0, "Starting analysis...")
    
    try:
        results = {}
        
        # Cut detection with explanations
        progress.progress(20, "Detecting cuts...")
        explainable = detect_cuts_with_explanation(
            session.video_path, sample_rate, threshold
        )
        
        results["cuts"] = {
            "total": explainable.total_cuts,
            "avg_confidence": explainable.avg_confidence,
            "cut_types": explainable.cut_type_distribution,
            "summary": explainable.explanation_summary,
            "details": [
                {
                    "timestamp": c.timestamp,
                    "confidence": c.confidence,
                    "reason": c.primary_reason,
                    "type": c.cut_type
                }
                for c in explainable.cuts
            ]
        }
        
        # Rhythm analysis
        progress.progress(40, "Analyzing rhythm...")
        cut_times = [c.timestamp for c in explainable.cuts]
        duration = session.metadata.get("duration_s", 1)
        
        rhythm = analyze_rhythm_patterns(cut_times, duration)
        pacing = calculate_pacing_score(
            len(cut_times) / (duration / 60),
            rhythm["rhythm_regularity"],
            rhythm["acceleration_trend"]
        )
        
        results["rhythm"] = {**rhythm, **pacing}
        
        # Audio analysis
        if analyze_audio:
            progress.progress(60, "Analyzing audio...")
            audio_result = extract_audio_features(session.video_path)
            
            if audio_result:
                sr, hop, times, rms = audio_result
                audio_stats = analyze_audio_energy(rms)
                results["audio"] = {
                    "sr": sr,
                    "hop_length": hop,
                    "times": times,
                    "rms": rms,
                    "stats": audio_stats
                }
            else:
                results["audio"] = None
        
        # Quality metrics
        progress.progress(80, "Computing quality metrics...")
        try:
            motion = analyze_motion_magnitude(session.video_path)
            brightness = analyze_brightness_variance(session.video_path)
            results["quality"] = {
                "motion_magnitudes": motion,
                "brightness_variances": brightness
            }
        except Exception as e:
            log_error("Quality analysis failed", e)
            results["quality"] = None
        
        # Save results
        progress.progress(95, "Saving results...")
        update_video_session(analysis_results=results)
        
        # Save to file
        paths = ProjectPaths()
        output_dir = paths.get_session_dir(session.video_stem)
        save_json(results, output_dir / "analysis_results.json")
        
        progress.progress(100, "Complete!")
        st.success("✅ Analysis completed successfully!")
        st.rerun()
        
    except Exception as e:
        log_error("Analysis failed", e)
        st.error(f"❌ Analysis failed: {str(e)}")


def display_results(session):
    """Display analysis results"""
    results = session.analysis_results
    
    st.markdown("### 📌 Results Summary")
    
    # Metrics row
    cols = st.columns(4)
    
    cuts = results.get("cuts", {})
    rhythm = results.get("rhythm", {})
    
    with cols[0]:
        st.metric("Cuts Detected", cuts.get("total", 0))
    
    with cols[1]:
        st.metric("Avg Confidence", f"{cuts.get('avg_confidence', 0):.0%}")
    
    with cols[2]:
        st.metric("Pacing Style", rhythm.get("pacing_style", "Unknown"))
    
    with cols[3]:
        st.metric("Cuts/Min", f"{rhythm.get('cuts_per_minute', 0):.1f}")
    
    # Explanation summary
    st.markdown("### 🤖 AI Explanation")
    st.info(cuts.get("summary", "No summary available"))
    
    # Cut details
    with st.expander("📋 Cut Details", expanded=False):
        for i, cut in enumerate(cuts.get("details", [])[:20]):
            col1, col2, col3 = st.columns([1, 2, 1])
            with col1:
                st.write(f"**{cut['timestamp']:.2f}s**")
            with col2:
                st.write(cut["reason"])
            with col3:
                st.write(f"{cut['confidence']:.0%} | {cut['type']}")
    
    # Audio visualization
    audio = results.get("audio")
    if audio and audio.get("rms"):
        st.markdown("### 🎵 Audio Energy")
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(audio["times"], audio["rms"], color="#00D4FF", linewidth=1)
        ax.fill_between(audio["times"], audio["rms"], alpha=0.3, color="#00D4FF")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RMS Energy")
        ax.set_facecolor("#0a0a0f")
        fig.patch.set_facecolor("#0a0a0f")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        st.pyplot(fig)
        plt.close()
    
    # Download button
    paths = ProjectPaths()
    output_path = paths.get_session_dir(session.video_stem) / "analysis_results.json"
    
    if output_path.exists():
        st.download_button(
            "⬇️ Download Analysis Report",
            data=output_path.read_bytes(),
            file_name=f"{session.video_stem}_analysis.json",
            mime="application/json",
            key=generate_unique_key("download_analysis")
        )