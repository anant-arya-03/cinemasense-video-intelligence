"""
Centralized logging configuration for CinemaSense AI Studio.

This module provides comprehensive logging setup with:
- File and console handlers
- Log rotation for file handler
- Per-module log level configuration
- Consistent timestamp, level, module, message format

Requirements: 12.1, 12.3
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict


# Default log format with timestamp, level, module, and message
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Default log file settings
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "app.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
DEFAULT_BACKUP_COUNT = 5

# Module-specific log levels
# More verbose for core modules, less verbose for UI/external
MODULE_LOG_LEVELS: Dict[str, int] = {
    "cinemasense": logging.INFO,
    "cinemasense.core": logging.DEBUG,
    "cinemasense.core.session": logging.INFO,
    "cinemasense.core.file_ops": logging.INFO,
    "cinemasense.core.video_capture": logging.INFO,
    "cinemasense.core.pipeline": logging.INFO,
    "cinemasense.core.system_check": logging.INFO,
    "cinemasense.pipeline": logging.INFO,
    "cinemasense.pipeline.multiverse": logging.INFO,
    "cinemasense.pipeline.emotion_rhythm": logging.INFO,
    "cinemasense.pipeline.explainable_ai": logging.INFO,
    "cinemasense.pipeline.social_pack": logging.INFO,
    "cinemasense.pipeline.gesture_control": logging.INFO,
    "cinemasense.pipeline.color_grading": logging.INFO,
    "cinemasense.services": logging.INFO,
    "cinemasense.ui": logging.WARNING,
}


class LoggingConfig:
    """
    Centralized logging configuration manager.
    
    Provides methods to setup and configure logging for the entire application
    with support for file rotation, console output, and per-module log levels.
    """
    
    _initialized: bool = False
    _root_logger: Optional[logging.Logger] = None
    _file_handler: Optional[logging.Handler] = None
    _console_handler: Optional[logging.Handler] = None
    
    @classmethod
    def setup(
        cls,
        log_dir: Optional[Path] = None,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        console_output: bool = True,
        log_format: str = DEFAULT_LOG_FORMAT,
        date_format: str = DEFAULT_DATE_FORMAT
    ) -> logging.Logger:
        """
        Setup centralized logging configuration.
        
        Args:
            log_dir: Directory for log files (default: logs/)
            log_file: Log file name (default: app.log)
            log_level: Root log level (default: INFO)
            max_bytes: Max size per log file before rotation (default: 5MB)
            backup_count: Number of backup files to keep (default: 5)
            console_output: Whether to output to console (default: True)
            log_format: Log message format
            date_format: Timestamp format
            
        Returns:
            The root cinemasense logger
        """
        if cls._initialized:
            return cls._root_logger
        
        # Setup log directory
        log_dir = log_dir or DEFAULT_LOG_DIR
        log_file = log_file or DEFAULT_LOG_FILE
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / log_file
        
        # Create formatter
        formatter = logging.Formatter(log_format, datefmt=date_format)
        
        # Get root cinemasense logger
        cls._root_logger = logging.getLogger("cinemasense")
        cls._root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Clear any existing handlers to avoid duplicates
        cls._root_logger.handlers.clear()
        
        # Setup rotating file handler
        cls._file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        cls._file_handler.setLevel(logging.DEBUG)  # File captures all levels
        cls._file_handler.setFormatter(formatter)
        cls._root_logger.addHandler(cls._file_handler)
        
        # Setup console handler
        if console_output:
            cls._console_handler = logging.StreamHandler(sys.stdout)
            cls._console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
            cls._console_handler.setFormatter(formatter)
            cls._root_logger.addHandler(cls._console_handler)
        
        # Apply module-specific log levels
        cls._apply_module_levels()
        
        # Prevent propagation to root logger
        cls._root_logger.propagate = False
        
        cls._initialized = True
        
        return cls._root_logger
    
    @classmethod
    def _apply_module_levels(cls) -> None:
        """Apply per-module log levels from configuration."""
        for module_name, level in MODULE_LOG_LEVELS.items():
            logger = logging.getLogger(module_name)
            logger.setLevel(level)
    
    @classmethod
    def set_module_level(cls, module_name: str, level: int) -> None:
        """
        Set log level for a specific module.
        
        Args:
            module_name: Full module name (e.g., 'cinemasense.pipeline.multiverse')
            level: Logging level (e.g., logging.DEBUG)
        """
        logger = logging.getLogger(module_name)
        logger.setLevel(level)
        MODULE_LOG_LEVELS[module_name] = level
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Args:
            name: Logger name (typically __name__ of the module)
            
        Returns:
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup()
        return logging.getLogger(name)
    
    @classmethod
    def set_console_level(cls, level: int) -> None:
        """
        Set the console handler log level.
        
        Args:
            level: Logging level (e.g., logging.WARNING)
        """
        if cls._console_handler:
            cls._console_handler.setLevel(level)
    
    @classmethod
    def set_file_level(cls, level: int) -> None:
        """
        Set the file handler log level.
        
        Args:
            level: Logging level (e.g., logging.DEBUG)
        """
        if cls._file_handler:
            cls._file_handler.setLevel(level)
    
    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration (useful for testing)."""
        if cls._root_logger:
            cls._root_logger.handlers.clear()
        cls._initialized = False
        cls._root_logger = None
        cls._file_handler = None
        cls._console_handler = None
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Check if logging has been initialized."""
        return cls._initialized


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Setup application logging (backward-compatible function).
    
    This function provides backward compatibility with the original API
    while using the new centralized configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Root log level
        
    Returns:
        The root cinemasense logger
    """
    return LoggingConfig.setup(
        log_dir=log_dir,
        log_level=log_level
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Convenience function that ensures logging is initialized.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Configured logger instance
    """
    return LoggingConfig.get_logger(name)


def log_exception(
    logger: logging.Logger,
    error: Exception,
    message: str = "An error occurred",
    include_traceback: bool = True
) -> None:
    """
    Log an exception with consistent formatting.
    
    Ensures all exceptions are logged with stack traces as per Requirements 12.1, 12.3.
    
    Args:
        logger: Logger instance to use
        error: The exception to log
        message: Custom message prefix
        include_traceback: Whether to include full stack trace
    """
    if include_traceback:
        logger.error(f"{message}: {error}", exc_info=True)
    else:
        logger.error(f"{message}: {type(error).__name__}: {error}")
