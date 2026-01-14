"""
Path management and directory structure
"""

from pathlib import Path


class ProjectPaths:
    """Centralized path management"""
    
    def __init__(self, root_dir: Path = None):
        self.root = root_dir or Path(__file__).resolve().parent.parent.parent.parent
        self.data = self.root / "data"
        self.input = self.data / "input"
        self.output = self.data / "output"
        self.logs = self.root / "logs"
        self.src = self.root / "src"
    
    def get_session_dir(self, video_stem: str) -> Path:
        """Get session-specific output directory"""
        session_dir = self.output / video_stem
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir
    
    def get_storyboard_dir(self, video_stem: str) -> Path:
        """Get storyboard directory for a video"""
        storyboard_dir = self.get_session_dir(video_stem) / "storyboard"
        storyboard_dir.mkdir(parents=True, exist_ok=True)
        return storyboard_dir
    
    def ensure_all_dirs(self):
        """Ensure all required directories exist"""
        for path in [self.input, self.output, self.logs]:
            path.mkdir(parents=True, exist_ok=True)