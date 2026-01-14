"""
Windows-safe file operations utility class.

Provides robust file handling with retry logic for file locking,
filename sanitization, and cross-platform path operations.

Requirements: 3.1, 3.2, 3.4, 3.5
"""

import re
import time
import tempfile
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger("cinemasense.core.file_ops")

# Characters invalid on Windows filesystems
WINDOWS_INVALID_CHARS = r'<>:"/\|?*'
# Control characters (0-31)
CONTROL_CHARS = ''.join(chr(i) for i in range(32))
# Reserved Windows filenames
WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
}


class FileOps:
    """Windows-safe file operations utility class."""
    
    @staticmethod
    def safe_write(
        path: Union[str, Path],
        data: bytes,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ) -> bool:
        """
        Write data to file with retry logic for Windows file locking.
        
        Args:
            path: Target file path
            data: Bytes to write
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if write succeeded, False otherwise
            
        Requirements: 3.2
        """
        path = Path(path)
        
        for attempt in range(max_retries):
            try:
                # Ensure parent directory exists
                path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to file
                path.write_bytes(data)
                logger.debug(f"Successfully wrote {len(data)} bytes to {path}")
                return True
                
            except PermissionError as e:
                # File is locked, retry after delay
                if attempt < max_retries - 1:
                    logger.warning(
                        f"File locked, retrying ({attempt + 1}/{max_retries}): {path}"
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"Failed to write after {max_retries} attempts: {path}")
                    logger.error(f"Error: {e}")
                    return False
                    
            except OSError as e:
                logger.error(f"OS error writing to {path}: {e}")
                return False
                
            except Exception as e:
                logger.error(f"Unexpected error writing to {path}: {e}")
                return False
        
        return False
    
    @staticmethod
    def sanitize_filename(name: str, replacement: str = "_") -> str:
        """
        Remove characters invalid on Windows from filename.
        
        Args:
            name: Original filename
            replacement: Character to replace invalid chars with
            
        Returns:
            Sanitized filename safe for Windows
            
        Requirements: 3.5
        """
        if not name:
            return "unnamed"
        
        # Remove control characters
        result = ''.join(c for c in name if c not in CONTROL_CHARS)
        
        # Replace Windows invalid characters
        for char in WINDOWS_INVALID_CHARS:
            result = result.replace(char, replacement)
        
        # Remove leading/trailing spaces and dots (Windows restriction)
        result = result.strip(' .')
        
        # Handle reserved Windows names
        name_upper = result.upper()
        base_name = name_upper.split('.')[0] if '.' in name_upper else name_upper
        if base_name in WINDOWS_RESERVED_NAMES:
            result = f"_{result}"
        
        # Ensure we have a valid filename
        if not result:
            return "unnamed"
        
        # Collapse multiple replacement characters
        while replacement + replacement in result:
            result = result.replace(replacement + replacement, replacement)
        
        return result
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, creating it if necessary.
        
        Args:
            path: Directory path to ensure exists
            
        Returns:
            Path object for the directory
            
        Requirements: 3.1
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {path}")
        return path
    
    @staticmethod
    def get_temp_path(prefix: str = "cinemasense_", suffix: str = "") -> Path:
        """
        Get a platform-appropriate temporary file path.
        
        Args:
            prefix: Prefix for the temp filename
            suffix: Suffix/extension for the temp filename
            
        Returns:
            Path to a temporary file location
            
        Requirements: 3.4
        """
        # Use system temp directory for cross-platform compatibility
        temp_dir = Path(tempfile.gettempdir())
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        filename = f"{prefix}{timestamp}{suffix}"
        
        temp_path = temp_dir / filename
        logger.debug(f"Generated temp path: {temp_path}")
        return temp_path
    
    @staticmethod
    def safe_delete(path: Union[str, Path], max_retries: int = 3) -> bool:
        """
        Safely delete a file with retry logic for Windows file locking.
        
        Args:
            path: File path to delete
            max_retries: Maximum retry attempts
            
        Returns:
            True if deleted or doesn't exist, False on failure
        """
        path = Path(path)
        
        if not path.exists():
            return True
        
        for attempt in range(max_retries):
            try:
                path.unlink()
                logger.debug(f"Deleted file: {path}")
                return True
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    logger.error(f"Failed to delete after {max_retries} attempts: {path}")
                    return False
            except Exception as e:
                logger.error(f"Error deleting {path}: {e}")
                return False
        
        return False
