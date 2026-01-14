"""
Scene segmentation based on cut detection
"""

from typing import List, Dict
from cinemasense.constants import DEFAULT_SCENE_GAP_THRESHOLD


def create_scenes_from_cuts(cut_times: List[float], duration_s: float, 
                           gap_threshold: float = DEFAULT_SCENE_GAP_THRESHOLD) -> List[Dict]:
    """Group cuts into scenes based on time gaps"""
    if not cut_times:
        # Single scene for entire video
        return [{
            "scene_id": 0,
            "start_time": 0.0,
            "end_time": duration_s,
            "duration": duration_s,
            "cut_count": 0
        }]
    
    scenes = []
    scene_id = 0
    scene_start = 0.0
    cuts_in_scene = 0
    
    # Sort cut times to ensure proper ordering
    sorted_cuts = sorted(cut_times)
    
    for i, cut_time in enumerate(sorted_cuts):
        cuts_in_scene += 1
        
        # Check if this is the last cut or if there's a significant gap to next cut
        is_scene_end = (
            i == len(sorted_cuts) - 1 or  # Last cut
            (i < len(sorted_cuts) - 1 and sorted_cuts[i + 1] - cut_time > gap_threshold)
        )
        
        if is_scene_end:
            # End current scene
            scene_end = cut_time if i < len(sorted_cuts) - 1 else duration_s
            
            scenes.append({
                "scene_id": scene_id,
                "start_time": scene_start,
                "end_time": scene_end,
                "duration": scene_end - scene_start,
                "cut_count": cuts_in_scene
            })
            
            # Start new scene
            scene_id += 1
            scene_start = cut_time
            cuts_in_scene = 0
    
    # Add final scene if needed
    if scene_start < duration_s:
        scenes.append({
            "scene_id": scene_id,
            "start_time": scene_start,
            "end_time": duration_s,
            "duration": duration_s - scene_start,
            "cut_count": 0
        })
    
    return scenes


def analyze_scene_characteristics(scenes: List[Dict]) -> Dict:
    """Analyze overall scene characteristics"""
    if not scenes:
        return {"avg_duration": 0, "total_scenes": 0, "longest_scene": 0, "shortest_scene": 0}
    
    durations = [scene["duration"] for scene in scenes]
    
    return {
        "total_scenes": len(scenes),
        "avg_duration": sum(durations) / len(durations),
        "longest_scene": max(durations),
        "shortest_scene": min(durations),
        "total_cuts": sum(scene["cut_count"] for scene in scenes)
    }