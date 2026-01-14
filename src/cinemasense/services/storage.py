"""
File storage and management utilities
"""

import time
from pathlib import Path


def ensure_dirs(*paths):
    """Create directories if they don't exist"""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def unique_filename(original_name: str) -> str:
    """Generate a unique filename with timestamp"""
    base = Path(original_name).stem
    ext = Path(original_name).suffix or ".mp4"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_base = "".join(ch for ch in base if ch.isalnum() or ch in ("_", "-", " "))
    safe_base = safe_base.strip().replace(" ", "_")[:60] or "video"
    return f"{safe_base}_{stamp}{ext}"


def save_uploaded_file(uploaded_file, target_path: Path) -> None:
    """Save uploaded file to target path"""
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())