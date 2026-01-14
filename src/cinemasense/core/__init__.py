"""
CinemaSense Core - Foundation utilities and system checks
"""

import sys
import logging
from pathlib import Path

from .file_ops import FileOps
from .video_capture import SafeVideoCapture, VideoOpenError
from .pipeline import PipelineResult, PipelineRunner, SUPPORTED_FORMATS
from .session import (
    SessionManager,
    SessionState,
    init_session_state,
    get_session_value,
    set_session_value,
    generate_unique_key,
)

logger = logging.getLogger("cinemasense.core")

__all__ = [
    'FileOps',
    'SafeVideoCapture',
    'VideoOpenError',
    'PipelineResult',
    'PipelineRunner',
    'SUPPORTED_FORMATS',
    'SessionManager',
    'SessionState',
    'init_session_state',
    'get_session_value',
    'set_session_value',
    'generate_unique_key',
]