"""
Streamlit session state management with type-safe access and automatic initialization.

This module provides a SessionManager class that ensures stable session state
across Streamlit interactions, preventing crashes and data loss during video analysis.
"""

import streamlit as st
from typing import Any, Dict, List, Optional, TypeVar, Generic
from dataclasses import dataclass, field, fields
from datetime import datetime
import logging
import hashlib

logger = logging.getLogger("cinemasense.core.session")

T = TypeVar('T')


@dataclass
class SessionState:
    """
    Video analysis session state with all required fields.
    
    This dataclass defines the complete state structure for CinemaSense,
    ensuring type-safe access and automatic initialization of all fields.
    """
    # Video context
    video_path: Optional[str] = None
    video_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Analysis results
    analysis: Optional[Dict[str, Any]] = None
    keyframes: Optional[List[Dict[str, Any]]] = None
    emotion: Optional[Dict[str, Any]] = None
    social: Optional[Dict[str, Any]] = None
    multiverse: Optional[Dict[str, Any]] = None
    
    # UI state
    gesture_enabled: bool = False
    current_page: str = "home"
    theme: str = "dark"
    
    # Processing state
    analysis_running: bool = False
    processing_progress: float = 0.0
    
    # Error tracking
    error_log: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    created_at: str = ""
    last_updated: str = ""
    
    def __post_init__(self):
        """Initialize timestamps if not set."""
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_updated:
            self.last_updated = now


class SessionManager:
    """
    Manages Streamlit session state with type-safe access and automatic initialization.
    
    This class provides a centralized interface for managing session state,
    ensuring all required keys exist with valid defaults before any UI rendering.
    It also generates unique widget keys to prevent duplicate key errors.
    
    Usage:
        SessionManager.initialize()
        value = SessionManager.get("video_path")
        SessionManager.set("video_path", "/path/to/video.mp4")
        SessionManager.clear_analysis()
        SessionManager.reset()
    """
    
    # Prefix for all session state keys to avoid conflicts
    PREFIX = "cs_"
    
    # Keys that should be preserved during clear_analysis
    _PRESERVED_KEYS = {"video_path", "video_name", "metadata", "current_page", 
                       "theme", "gesture_enabled", "created_at"}
    
    # Keys that represent analysis results and should be cleared on new upload
    _ANALYSIS_KEYS = {"analysis", "keyframes", "emotion", "social", "multiverse",
                      "analysis_running", "processing_progress"}
    
    @classmethod
    def _get_prefixed_key(cls, key: str) -> str:
        """Get the prefixed version of a key."""
        if key.startswith(cls.PREFIX):
            return key
        return f"{cls.PREFIX}{key}"
    
    @classmethod
    def _get_unprefixed_key(cls, key: str) -> str:
        """Get the unprefixed version of a key."""
        if key.startswith(cls.PREFIX):
            return key[len(cls.PREFIX):]
        return key
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize all session state variables with default values.
        
        This method MUST be called before any UI rendering to ensure
        all required session state keys exist with valid defaults.
        
        The initialization is idempotent - calling it multiple times
        will not overwrite existing values.
        """
        # Check if already initialized
        init_key = cls._get_prefixed_key("initialized")
        if st.session_state.get(init_key, False):
            return
        
        # Create default SessionState to get all field defaults
        default_state = SessionState()
        
        # Initialize all fields from SessionState dataclass
        for f in fields(SessionState):
            prefixed_key = cls._get_prefixed_key(f.name)
            if prefixed_key not in st.session_state:
                default_value = getattr(default_state, f.name)
                st.session_state[prefixed_key] = default_value
                logger.debug(f"Initialized session key: {prefixed_key}")
        
        # Mark as initialized
        st.session_state[init_key] = True
        logger.info("Session state initialized with all default values")
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Safely get a session state value.
        
        Args:
            key: The key to retrieve (with or without prefix)
            default: Default value if key doesn't exist
            
        Returns:
            The value associated with the key, or the default
        """
        prefixed_key = cls._get_prefixed_key(key)
        try:
            value = st.session_state.get(prefixed_key, default)
            return value
        except Exception as e:
            logger.error(f"Failed to get session key '{key}': {e}")
            cls._log_error(f"Session state access failed for key: {key}", e)
            return default
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """
        Safely set a session state value.
        
        Args:
            key: The key to set (with or without prefix)
            value: The value to store
        """
        prefixed_key = cls._get_prefixed_key(key)
        try:
            st.session_state[prefixed_key] = value
            # Update last_updated timestamp
            st.session_state[cls._get_prefixed_key("last_updated")] = datetime.now().isoformat()
            logger.debug(f"Set session key: {prefixed_key}")
        except Exception as e:
            logger.error(f"Failed to set session key '{key}': {e}")
            cls._log_error(f"Session state write failed for key: {key}", e)
    
    @classmethod
    def clear_analysis(cls) -> None:
        """
        Clear all analysis-related session state while preserving video context.
        
        This method is called when a new video is uploaded to reset
        previous analysis results while keeping the video path and metadata.
        """
        logger.info("Clearing analysis results from session state")
        
        for key in cls._ANALYSIS_KEYS:
            prefixed_key = cls._get_prefixed_key(key)
            # Reset to default values based on type
            if key in {"analysis_running"}:
                st.session_state[prefixed_key] = False
            elif key in {"processing_progress"}:
                st.session_state[prefixed_key] = 0.0
            else:
                st.session_state[prefixed_key] = None
        
        # Update timestamp
        st.session_state[cls._get_prefixed_key("last_updated")] = datetime.now().isoformat()
        logger.info("Analysis results cleared")
    
    @classmethod
    def reset(cls) -> None:
        """
        Reset all session state to default values.
        
        This performs a complete reset, clearing all values including
        video context and analysis results.
        """
        logger.info("Resetting all session state to defaults")
        
        # Create fresh default state
        default_state = SessionState()
        
        # Reset all fields
        for f in fields(SessionState):
            prefixed_key = cls._get_prefixed_key(f.name)
            default_value = getattr(default_state, f.name)
            st.session_state[prefixed_key] = default_value
        
        # Reset initialized flag to trigger re-initialization
        st.session_state[cls._get_prefixed_key("initialized")] = True
        
        logger.info("Session state reset complete")
    
    @classmethod
    def generate_widget_key(cls, base: str, *args, **kwargs) -> str:
        """
        Generate a unique, deterministic key for Streamlit widgets.
        
        This method creates unique keys by combining a base name with
        additional arguments, ensuring no duplicate key errors occur.
        
        Args:
            base: Base name for the widget key
            *args: Additional components to include in the key
            **kwargs: Named components to include in the key
            
        Returns:
            A unique, deterministic widget key string
            
        Example:
            key = SessionManager.generate_widget_key("button", "analysis", page="home")
            # Returns: "cs_widget_button_analysis_page_home"
        """
        parts = [cls.PREFIX + "widget", base]
        
        # Add positional arguments
        for arg in args:
            parts.append(str(arg))
        
        # Add keyword arguments (sorted for determinism)
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}_{v}")
        
        key = "_".join(parts)
        
        # Ensure key is valid (no special characters that might cause issues)
        key = "".join(c if c.isalnum() or c == "_" else "_" for c in key)
        
        return key
    
    @classmethod
    def generate_unique_key(cls, base: str, context: str = "") -> str:
        """
        Generate a unique key using hash for complex contexts.
        
        This is useful when you need a unique key but the context
        might contain special characters or be very long.
        
        Args:
            base: Base name for the key
            context: Additional context to make the key unique
            
        Returns:
            A unique widget key with a hash suffix
        """
        if context:
            hash_suffix = hashlib.md5(context.encode()).hexdigest()[:8]
            return f"{cls.PREFIX}widget_{base}_{hash_suffix}"
        return f"{cls.PREFIX}widget_{base}"
    
    @classmethod
    def _log_error(cls, message: str, exception: Optional[Exception] = None) -> None:
        """
        Log an error to the session state error log.
        
        Args:
            message: Error message
            exception: Optional exception that caused the error
        """
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "exception": str(exception) if exception else None
        }
        
        error_log_key = cls._get_prefixed_key("error_log")
        if error_log_key not in st.session_state:
            st.session_state[error_log_key] = []
        
        st.session_state[error_log_key].append(error_entry)
        logger.error(f"{message}: {exception}" if exception else message)
    
    @classmethod
    def get_error_log(cls) -> List[Dict[str, Any]]:
        """
        Get the current error log.
        
        Returns:
            List of error entries with timestamp, message, and exception
        """
        return cls.get("error_log", [])
    
    @classmethod
    def clear_errors(cls) -> None:
        """Clear the error log."""
        cls.set("error_log", [])
        logger.info("Error log cleared")
    
    @classmethod
    def is_initialized(cls) -> bool:
        """
        Check if session state has been initialized.
        
        Returns:
            True if initialize() has been called, False otherwise
        """
        return st.session_state.get(cls._get_prefixed_key("initialized"), False)
    
    @classmethod
    def get_all_keys(cls) -> List[str]:
        """
        Get all session state keys managed by SessionManager.
        
        Returns:
            List of all prefixed keys in session state
        """
        return [k for k in st.session_state.keys() if k.startswith(cls.PREFIX)]
    
    @classmethod
    def has_video(cls) -> bool:
        """
        Check if a video is currently loaded.
        
        Returns:
            True if video_path is set, False otherwise
        """
        return cls.get("video_path") is not None
    
    @classmethod
    def has_analysis(cls) -> bool:
        """
        Check if analysis results are available.
        
        Returns:
            True if analysis data exists, False otherwise
        """
        return cls.get("analysis") is not None


# Convenience functions for backward compatibility
def init_session_state():
    """Initialize session state (backward compatible wrapper)."""
    SessionManager.initialize()


def get_session_value(key: str, default: Any = None) -> Any:
    """Get session value (backward compatible wrapper)."""
    return SessionManager.get(key, default)


def set_session_value(key: str, value: Any) -> None:
    """Set session value (backward compatible wrapper)."""
    SessionManager.set(key, value)


def generate_unique_key(base: str, *args) -> str:
    """Generate unique key (backward compatible wrapper)."""
    return SessionManager.generate_widget_key(base, *args)
