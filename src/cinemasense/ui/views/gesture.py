"""
Gesture Control Page - Hand gesture recognition interface
"""

import streamlit as st
from pathlib import Path

from cinemasense.core.session import (
    get_video_session, get_session_value, set_session_value, generate_unique_key
)


def render():
    """Render the gesture control page"""
    st.title("✋ Gesture Control Mode")
    st.markdown("*Control CinemaSense with hand gestures using MediaPipe*")
    
    # Check MediaPipe availability
    try:
        import mediapipe
        mediapipe_available = True
    except ImportError:
        mediapipe_available = False
    
    if not mediapipe_available:
        st.error("❌ MediaPipe is not installed. Please install it with: `pip install mediapipe`")
        return
    
    # Gesture mode toggle
    gesture_mode = get_session_value("gesture_mode", False)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button(
            "🟢 Enable Gestures" if not gesture_mode else "🔴 Disable Gestures",
            type="primary" if not gesture_mode else "secondary",
            key=generate_unique_key("gesture_toggle")
        ):
            set_session_value("gesture_mode", not gesture_mode)
            st.rerun()
    
    with col2:
        status = "Active" if gesture_mode else "Inactive"
        st.info(f"Gesture Control: **{status}**")
    
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
        ("🖐️", "Spread", "Zoom In", "Fingers spreading apart")
    ]
    
    # Display in grid
    cols = st.columns(2)
    
    for i, (icon, name, action, description) in enumerate(gestures):
        with cols[i % 2]:
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
    
    # Camera preview (placeholder)
    st.markdown("### 📷 Camera Preview")
    
    if gesture_mode:
        st.warning("⚠️ Camera access requires running in a local environment with webcam support.")
        st.info("💡 In a full implementation, the camera feed would appear here with gesture detection overlay.")
        
        # Simulated gesture detection status
        st.markdown("#### Last Detected Gesture")
        
        last_gesture = get_session_value("last_gesture", None)
        if last_gesture:
            st.success(f"Detected: **{last_gesture}**")
        else:
            st.info("No gesture detected yet")
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