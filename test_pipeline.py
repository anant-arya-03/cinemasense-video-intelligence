"""
CinemaSense Pipeline Test Script
Creates a synthetic test video and validates all components
"""

import sys
from pathlib import Path

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import json
from datetime import datetime

print("=" * 60)
print("CinemaSense AI Studio - Pipeline Test")
print("=" * 60)

# Create test directories
test_dir = ROOT / "data" / "test"
test_dir.mkdir(parents=True, exist_ok=True)
output_dir = ROOT / "data" / "output" / "test_video"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Create synthetic test video
print("\n[1/8] Creating synthetic test video...")

video_path = test_dir / "test_video.mp4"
fps = 30
duration = 3  # seconds (reduced for faster testing)
width, height = 320, 240  # smaller resolution
total_frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

for frame_idx in range(total_frames):
    # Create frame with changing colors to simulate scene changes
    t = frame_idx / total_frames
    
    # Change color every second to create "cuts"
    second = frame_idx // fps
    colors = [
        (255, 100, 100),  # Blue-ish
        (100, 255, 100),  # Green-ish
        (100, 100, 255),  # Red-ish
    ]
    base_color = colors[second % len(colors)]
    
    # Create solid color frame with noise (faster than per-pixel)
    frame = np.full((height, width, 3), base_color, dtype=np.uint8)
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add moving circle
    cx = int(width * (0.3 + 0.4 * np.sin(t * 4 * np.pi)))
    cy = int(height * (0.3 + 0.4 * np.cos(t * 4 * np.pi)))
    cv2.circle(frame, (cx, cy), 30, (255, 255, 255), -1)
    
    out.write(frame)

out.release()
print(f"   Created: {video_path}")
print(f"   Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")


# Step 2: Test metadata extraction
print("\n[2/8] Testing metadata extraction...")
try:
    from cinemasense.pipeline.metadata import get_video_metadata
    
    fps_out, frame_count, w, h, dur = get_video_metadata(str(video_path))
    print(f"   FPS: {fps_out:.1f}")
    print(f"   Frames: {frame_count}")
    print(f"   Resolution: {w}x{h}")
    print(f"   Duration: {dur:.2f}s")
    print("   ✓ Metadata extraction: PASSED")
except Exception as e:
    print(f"   ✗ Metadata extraction: FAILED - {e}")

# Step 3: Test explainable cut detection
print("\n[3/8] Testing explainable AI cut detection...")
try:
    from cinemasense.pipeline.explainable_ai import detect_cuts_with_explanation
    
    result = detect_cuts_with_explanation(str(video_path), sample_every_n_frames=2, threshold=0.4)
    print(f"   Total cuts: {result.total_cuts}")
    print(f"   Avg confidence: {result.avg_confidence:.2%}")
    print(f"   Cut types: {result.cut_type_distribution}")
    print(f"   Summary: {result.explanation_summary[:100]}...")
    
    if result.cuts:
        print(f"   First cut: {result.cuts[0].timestamp:.2f}s - {result.cuts[0].primary_reason}")
    
    print("   ✓ Explainable AI: PASSED")
except Exception as e:
    print(f"   ✗ Explainable AI: FAILED - {e}")

# Step 4: Test emotion rhythm analysis
print("\n[4/8] Testing emotion rhythm analysis...")
try:
    from cinemasense.pipeline.emotion_rhythm import extract_emotion_timeline
    
    emotion_result = extract_emotion_timeline(str(video_path), sample_rate=10)
    print(f"   Timeline points: {len(emotion_result.timeline)}")
    print(f"   Overall score: {emotion_result.overall_score:.1f}")
    print(f"   Rhythm pattern: {emotion_result.rhythm_pattern}")
    print(f"   Emotion distribution: {emotion_result.emotion_distribution}")
    print(f"   Peak moments: {len(emotion_result.peak_moments)}")
    print(f"   Heatmap shape: {emotion_result.heatmap_data.shape}")
    print("   ✓ Emotion rhythm: PASSED")
except Exception as e:
    print(f"   ✗ Emotion rhythm: FAILED - {e}")


# Step 5: Test keyframe extraction
print("\n[5/8] Testing keyframe extraction...")
try:
    from cinemasense.pipeline.keyframes import extract_keyframes_interval
    
    storyboard_dir = output_dir / "storyboard"
    storyboard_dir.mkdir(exist_ok=True)
    
    keyframes = extract_keyframes_interval(str(video_path), interval_s=1.0, 
                                          output_dir=storyboard_dir, max_frames=10)
    print(f"   Keyframes extracted: {len(keyframes)}")
    
    if keyframes:
        print(f"   First keyframe: t={keyframes[0]['time_s']:.1f}s")
        print(f"   Last keyframe: t={keyframes[-1]['time_s']:.1f}s")
    
    # Check if thumbnails were created
    thumbnails = list(storyboard_dir.glob("*.jpg"))
    print(f"   Thumbnails saved: {len(thumbnails)}")
    print("   ✓ Keyframe extraction: PASSED")
except Exception as e:
    print(f"   ✗ Keyframe extraction: FAILED - {e}")

# Step 6: Test multiverse style preview
print("\n[6/8] Testing multiverse style generation...")
try:
    from cinemasense.pipeline.multiverse import generate_multiverse_preview, get_available_styles
    
    styles = get_available_styles()
    print(f"   Available styles: {[s['id'] for s in styles]}")
    
    multiverse_dir = output_dir / "multiverse"
    multiverse_dir.mkdir(exist_ok=True)
    
    # Test one style
    preview = generate_multiverse_preview(str(video_path), "thriller", multiverse_dir)
    print(f"   Generated style: {preview['style']}")
    print(f"   Preview frames: {len(preview['previews'])}")
    print("   ✓ Multiverse generation: PASSED")
except Exception as e:
    print(f"   ✗ Multiverse generation: FAILED - {e}")

# Step 7: Test social pack generation
print("\n[7/8] Testing social pack generation...")
try:
    from cinemasense.pipeline.social_pack import generate_social_pack
    
    social_dir = output_dir / "social"
    social_dir.mkdir(exist_ok=True)
    
    metadata = {"duration_s": dur, "width": w, "height": h, "fps": fps_out}
    
    social_result = generate_social_pack(
        str(video_path), social_dir, metadata,
        platforms=["youtube", "instagram"]
    )
    
    print(f"   Thumbnail: {Path(social_result.thumbnail_path).name}")
    print(f"   Title suggestions: {len(social_result.title_suggestions)}")
    print(f"   Hashtags: {len(social_result.hashtags)}")
    print(f"   Platforms: {list(social_result.platform_optimized.keys())}")
    
    if social_result.title_suggestions:
        print(f"   Sample title: {social_result.title_suggestions[0]}")
    
    print("   ✓ Social pack: PASSED")
except Exception as e:
    print(f"   ✗ Social pack: FAILED - {e}")


# Step 8: Test color grading
print("\n[8/8] Testing color grading presets...")
try:
    from cinemasense.pipeline.color_grading import (
        get_available_presets, apply_color_grading, 
        CINEMA_PRESETS, analyze_color_palette
    )
    
    presets = get_available_presets()
    print(f"   Available presets: {[p['id'] for p in presets]}")
    
    # Load a test frame
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        # Test color grading
        graded = apply_color_grading(frame, CINEMA_PRESETS["blockbuster"])
        print(f"   Graded frame shape: {graded.shape}")
        
        # Test color analysis
        palette = analyze_color_palette(frame)
        print(f"   Dominant colors: {len(palette['dominant_colors'])}")
        print(f"   Suggested preset: {palette['suggested_preset']}")
        print("   ✓ Color grading: PASSED")
    else:
        print("   ✗ Color grading: FAILED - Could not read frame")
except Exception as e:
    print(f"   ✗ Color grading: FAILED - {e}")

# Summary
print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

# Save test results
results = {
    "timestamp": datetime.now().isoformat(),
    "test_video": str(video_path),
    "output_dir": str(output_dir),
    "tests_completed": True
}

results_path = output_dir / "test_results.json"
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nTest results saved to: {results_path}")
print(f"Test video: {video_path}")
print(f"Output directory: {output_dir}")

# List generated files
print("\nGenerated files:")
for f in output_dir.rglob("*"):
    if f.is_file():
        print(f"   - {f.relative_to(output_dir)}")

print("\n" + "=" * 60)
print("All pipeline tests completed!")
print("=" * 60)