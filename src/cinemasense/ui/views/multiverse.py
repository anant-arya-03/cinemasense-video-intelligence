"""
Multiverse Page - Generate style variants of videos
"""

import streamlit as st
from pathlib import Path

from cinemasense.core.session import (
    get_video_session, update_video_session, generate_unique_key
)
from cinemasense.storage.paths import ProjectPaths
from cinemasense.pipeline.multiverse import (
    MULTIVERSE_STYLES, generate_multiverse_preview, get_available_styles
)


def render():
    """Render the multiverse page"""
    st.title("🌌 Multiverse Generator")
    st.markdown("*Create stunning style variants of your video*")
    
    session = get_video_session()
    if not session or not session.video_path:
        st.warning("⚠️ Please upload a video on the Home page first")
        return
    
    st.info(f"📹 Video: {Path(session.video_path).name}")
    
    # Style selection
    st.markdown("### 🎨 Choose Your Style")
    
    styles = get_available_styles()
    
    # Display style cards
    cols = st.columns(3)
    
    for i, style in enumerate(styles):
        with cols[i % 3]:
            selected = st.button(
                f"{style['name']}",
                key=generate_unique_key(f"style_{style['id']}"),
                use_container_width=True,
                help=style['description']
            )
            st.caption(style['description'])
            
            if selected:
                generate_preview(session, style['id'])
    
    # Display previews
    if session.multiverse_outputs:
        display_previews(session)


def generate_preview(session, style_id):
    """Generate preview for selected style"""
    progress = st.progress(0, f"Generating {style_id} preview...")
    
    try:
        paths = ProjectPaths()
        output_dir = paths.get_session_dir(session.video_stem) / "multiverse"
        output_dir.mkdir(exist_ok=True)
        
        progress.progress(50, "Processing frames...")
        
        result = generate_multiverse_preview(
            session.video_path, style_id, output_dir
        )
        
        # Update session
        outputs = session.multiverse_outputs or {}
        outputs[style_id] = result
        update_video_session(multiverse_outputs=outputs)
        
        progress.progress(100, "Complete!")
        st.success(f"✅ {style_id.title()} preview generated!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Preview generation failed: {str(e)}")


def display_previews(session):
    """Display generated multiverse previews"""
    st.markdown("### 🖼️ Generated Previews")
    
    outputs = session.multiverse_outputs
    
    for style_id, result in outputs.items():
        with st.expander(f"🎬 {result['style'].title()}", expanded=True):
            st.write(result['description'])
            
            cols = st.columns(3)
            for i, preview in enumerate(result.get('previews', [])):
                with cols[i]:
                    preview_path = Path(preview['path'])
                    if preview_path.exists():
                        st.image(
                            str(preview_path),
                            caption=f"t={preview['timestamp']:.1f}s"
                        )