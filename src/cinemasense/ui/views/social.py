"""
Social Pack Page - Generate thumbnails, titles, hashtags for social media
"""

import streamlit as st
from pathlib import Path

from cinemasense.core.session import (
    get_video_session, update_video_session, generate_unique_key
)
from cinemasense.storage.paths import ProjectPaths
from cinemasense.pipeline.social_pack import generate_social_pack


def render():
    """Render the social pack page"""
    st.title("📱 Social Pack Generator")
    st.markdown("*Create optimized content for YouTube, Instagram, TikTok, and Twitter*")
    
    session = get_video_session()
    if not session or not session.video_path:
        st.warning("⚠️ Please upload a video on the Home page first")
        return
    
    st.info(f"📹 Video: {Path(session.video_path).name}")
    
    # Platform selection
    st.markdown("### 🎯 Select Platforms")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        youtube = st.checkbox("YouTube", value=True, key=generate_unique_key("social_yt"))
    with col2:
        instagram = st.checkbox("Instagram", value=True, key=generate_unique_key("social_ig"))
    with col3:
        tiktok = st.checkbox("TikTok", value=True, key=generate_unique_key("social_tt"))
    with col4:
        twitter = st.checkbox("Twitter", value=False, key=generate_unique_key("social_tw"))
    
    platforms = []
    if youtube: platforms.append("youtube")
    if instagram: platforms.append("instagram")
    if tiktok: platforms.append("tiktok")
    if twitter: platforms.append("twitter")
    
    # Generate button
    if st.button("📱 Generate Social Pack", type="primary", key=generate_unique_key("social_gen")):
        if not platforms:
            st.warning("Please select at least one platform")
            return
        generate_pack(session, platforms)
    
    # Display results
    if session.social_pack:
        display_social_pack(session)


def generate_pack(session, platforms):
    """Generate social media pack"""
    progress = st.progress(0, "Generating social pack...")
    
    try:
        paths = ProjectPaths()
        output_dir = paths.get_session_dir(session.video_stem) / "social"
        output_dir.mkdir(exist_ok=True)
        
        progress.progress(30, "Extracting best thumbnail...")
        
        # Get emotion data if available
        emotion_data = None
        if session.emotion_timeline:
            emotion_data = {
                "dominant_emotion": max(
                    session.emotion_timeline.get("distribution", {}),
                    key=session.emotion_timeline.get("distribution", {}).get,
                    default="Cinematic"
                )
            }
        
        progress.progress(60, "Generating platform content...")
        
        result = generate_social_pack(
            session.video_path,
            output_dir,
            session.metadata,
            emotion_data,
            platforms
        )
        
        # Convert to dict
        pack_data = {
            "thumbnail_path": result.thumbnail_path,
            "title_suggestions": result.title_suggestions,
            "hashtags": result.hashtags,
            "caption": result.caption,
            "platforms": result.platform_optimized
        }
        
        update_video_session(social_pack=pack_data)
        
        progress.progress(100, "Complete!")
        st.success("✅ Social pack generated!")
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Social pack generation failed: {str(e)}")


def display_social_pack(session):
    """Display generated social pack"""
    pack = session.social_pack
    
    # Main thumbnail
    st.markdown("### 🖼️ Main Thumbnail")
    
    thumb_path = Path(pack.get("thumbnail_path", ""))
    if thumb_path.exists():
        st.image(str(thumb_path), width=400)
    
    # Title suggestions
    st.markdown("### 📝 Title Suggestions")
    
    for i, title in enumerate(pack.get("title_suggestions", []), 1):
        st.code(title, language=None)
    
    # Hashtags
    st.markdown("### #️⃣ Hashtags")
    
    hashtags = pack.get("hashtags", [])
    st.code(" ".join(hashtags), language=None)
    
    # Caption
    st.markdown("### 💬 Caption")
    st.text_area(
        "Copy this caption:",
        pack.get("caption", ""),
        height=150,
        key=generate_unique_key("social_caption")
    )
    
    # Platform-specific content
    st.markdown("### 🎯 Platform-Optimized Content")
    
    platforms = pack.get("platforms", {})
    
    tabs = st.tabs([p.title() for p in platforms.keys()])
    
    for tab, (platform, content) in zip(tabs, platforms.items()):
        with tab:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                thumb = Path(content.get("thumbnail_path", ""))
                if thumb.exists():
                    st.image(str(thumb), caption=f"{platform.title()} Thumbnail")
            
            with col2:
                st.markdown("**Hashtags:**")
                st.code(" ".join(content.get("hashtags", [])), language=None)
                
                st.markdown("**Caption:**")
                st.text_area(
                    f"{platform} caption",
                    content.get("caption", ""),
                    height=100,
                    key=generate_unique_key(f"social_{platform}_caption"),
                    label_visibility="collapsed"
                )