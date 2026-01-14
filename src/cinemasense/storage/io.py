"""
File I/O operations
"""

import time
import json
from pathlib import Path
from typing import Any, Dict


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


def save_json(data: Dict[str, Any], file_path: Path) -> None:
    """Save data as JSON file"""
    file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file"""
    return json.loads(file_path.read_text(encoding="utf-8"))