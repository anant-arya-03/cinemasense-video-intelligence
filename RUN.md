# CinemaSense AI Studio v2.0
## Revolutionary Video Intelligence Platform

✅ **ALL TESTS PASSED** - Industry-Ready, Error-Free

---

## 🚀 Quick Start (PowerShell)

```powershell
# Navigate to project
cd C:\Users\Anant\Documents\CinemaSense

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Clear broken configs
Remove-Item -Recurse -Force C:\Users\Anant\.streamlit -ErrorAction SilentlyContinue

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

---

## 🔧 System Requirements

- **Python**: 3.11+
- **OS**: Windows 10/11, macOS, Linux
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional (for faster processing)

---

## ✅ Health Check

```powershell
# Test all imports
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve() / 'src'))

from cinemasense.core.system_check import run_all_checks
ok, checks = run_all_checks()
print('All systems OK!' if ok else 'Some issues detected')
"
```

---

## 🎬 Features

### 📊 Smart Analysis
- Explainable AI cut detection with confidence scores
- Rhythm analysis and pacing classification
- Audio energy visualization
- Quality metrics (motion, brightness)

### 🌌 Multiverse Generator
- **Romantic**: Soft, warm, dreamy
- **Thriller**: High contrast, teal-orange
- **Viral**: Punchy, vibrant colors
- **Anime**: Cel-shaded, bold outlines
- **Cinematic**: Film-like grading
- **Noir**: Classic black and white

### 💫 Emotion Rhythm Score (ERS)
- AI-powered emotional timeline
- Heatmap visualization
- Peak moment detection
- Rhythm pattern classification

### 📱 Social Pack Generator
- Platform-optimized thumbnails
- Title suggestions
- Hashtag generation
- Caption templates
- Supports: YouTube, Instagram, TikTok, Twitter

### ✋ Gesture Control
- Hand gesture recognition via MediaPipe
- 10+ supported gestures
- Real-time detection

### 🎨 Cinema Color Grading
- Blockbuster, Indie, Horror, Romance
- Sci-Fi, Vintage, Documentary, Neon

---

## 📁 Project Structure

```
CinemaSense/
├── app.py                      # Main entry point
├── requirements.txt            # Dependencies
├── RUN.md                      # This file
├── src/cinemasense/
│   ├── core/
│   │   ├── session.py          # Session management
│   │   └── system_check.py     # Dependency checks
│   ├── storage/
│   │   ├── paths.py            # Path management
│   │   └── io.py               # File I/O
│   ├── pipeline/
│   │   ├── metadata.py         # Video metadata
│   │   ├── explainable_ai.py   # Cut detection + reasoning
│   │   ├── emotion_rhythm.py   # ERS analysis
│   │   ├── multiverse.py       # Style variants
│   │   ├── color_grading.py    # Cinema presets
│   │   ├── text_effects.py     # Text-behind-video
│   │   ├── social_pack.py      # Social media content
│   │   └── gesture_control.py  # Hand gestures
│   ├── features/
│   │   ├── rhythm.py           # Rhythm analysis
│   │   ├── mood.py             # Mood detection
│   │   └── anomalies.py        # Anomaly detection
│   └── ui/
│       ├── glassmorphic.py     # Premium UI components
│       └── views/              # Page components
├── data/
│   ├── input/                  # Uploaded videos
│   └── output/                 # Analysis results
└── logs/                       # Application logs
```

---

## 🎨 UI Design

The interface follows **Apple VisionOS** design principles:
- Glassmorphic cards with blur effects
- Gradient accents (cyan to purple)
- Smooth micro-interactions
- Dark theme optimized
- Zero clutter, maximum clarity

---

## 🐛 Troubleshooting

### ModuleNotFoundError
```powershell
# Ensure src is in path
$env:PYTHONPATH = ".\src"
streamlit run app.py
```

### TOML Errors
```powershell
# Remove all Streamlit configs
Remove-Item -Recurse -Force .streamlit -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $env:USERPROFILE\.streamlit -ErrorAction SilentlyContinue
```

### FFmpeg Issues
```powershell
# Install via imageio
pip install imageio-ffmpeg
python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
```

### MediaPipe Issues
```powershell
pip install mediapipe --upgrade
```

---

## 🔄 Adding New Features

1. **Create pipeline module**: `src/cinemasense/pipeline/new_feature.py`
2. **Create view**: `src/cinemasense/ui/views/new_feature.py`
3. **Add to navigation**: Update `app.py` pages dict
4. **Import in view**: Use relative imports from cinemasense

---

## 📝 License

MIT License - Free for academic and commercial use.

---

## 🙏 Credits

Built with:
- Streamlit
- OpenCV
- MediaPipe
- MoviePy
- Librosa
- Scikit-learn

---

**CinemaSense AI Studio v2.0**
*Revolutionary Video Intelligence*