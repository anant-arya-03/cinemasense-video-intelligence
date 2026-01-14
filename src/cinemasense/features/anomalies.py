"""
Anomaly detection in video segments
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Dict, Tuple
from cinemasense.constants import ANOMALY_CONTAMINATION, MIN_SEGMENTS_FOR_ANOMALY


def extract_segment_features(scenes: List[Dict], motion_magnitudes: List[float], 
                           brightness_variances: List[float], cut_times: List[float]) -> List[Dict]:
    """Extract features for each video segment for anomaly detection"""
    if not scenes:
        return []
    
    segment_features = []
    
    for scene in scenes:
        start_time = scene["start_time"]
        end_time = scene["end_time"]
        duration = scene["duration"]
        
        # Calculate cut density for this segment
        cuts_in_segment = [t for t in cut_times if start_time <= t <= end_time]
        cut_density = len(cuts_in_segment) / max(duration, 0.1)  # Avoid division by zero
        
        # Estimate motion and brightness for this segment
        # (This is simplified - in practice you'd map frame indices to time)
        segment_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0.0
        segment_brightness_var = np.mean(brightness_variances) if brightness_variances else 0.0
        
        segment_features.append({
            "scene_id": scene["scene_id"],
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "cut_density": cut_density,
            "motion_magnitude": segment_motion,
            "brightness_variance": segment_brightness_var,
            "feature_vector": [cut_density, segment_motion, segment_brightness_var, duration]
        })
    
    return segment_features


def detect_anomalous_segments(segment_features: List[Dict], 
                            contamination: float = ANOMALY_CONTAMINATION) -> Tuple[List[Dict], List[int]]:
    """Detect anomalous video segments using Isolation Forest"""
    if len(segment_features) < MIN_SEGMENTS_FOR_ANOMALY:
        # Not enough segments for meaningful anomaly detection
        return segment_features, []
    
    # Extract feature vectors
    feature_vectors = np.array([seg["feature_vector"] for seg in segment_features])
    
    # Normalize features
    feature_vectors = (feature_vectors - np.mean(feature_vectors, axis=0)) / (np.std(feature_vectors, axis=0) + 1e-8)
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomaly_labels = iso_forest.fit_predict(feature_vectors)
    anomaly_scores = iso_forest.score_samples(feature_vectors)
    
    # Add anomaly information to segments
    anomalous_indices = []
    for i, (segment, label, score) in enumerate(zip(segment_features, anomaly_labels, anomaly_scores)):
        is_anomaly = label == -1
        segment["is_anomaly"] = is_anomaly
        segment["anomaly_score"] = float(score)
        
        if is_anomaly:
            anomalous_indices.append(i)
    
    return segment_features, anomalous_indices


def analyze_anomaly_patterns(anomalous_segments: List[Dict]) -> Dict:
    """Analyze patterns in detected anomalies"""
    if not anomalous_segments:
        return {
            "total_anomalies": 0,
            "anomaly_duration_total": 0.0,
            "avg_anomaly_duration": 0.0,
            "anomaly_characteristics": {}
        }
    
    total_duration = sum(seg["duration"] for seg in anomalous_segments)
    avg_duration = total_duration / len(anomalous_segments)
    
    # Analyze characteristics of anomalous segments
    cut_densities = [seg["cut_density"] for seg in anomalous_segments]
    motion_magnitudes = [seg["motion_magnitude"] for seg in anomalous_segments]
    brightness_variances = [seg["brightness_variance"] for seg in anomalous_segments]
    
    characteristics = {
        "avg_cut_density": float(np.mean(cut_densities)),
        "avg_motion": float(np.mean(motion_magnitudes)),
        "avg_brightness_var": float(np.mean(brightness_variances))
    }
    
    return {
        "total_anomalies": len(anomalous_segments),
        "anomaly_duration_total": float(total_duration),
        "avg_anomaly_duration": float(avg_duration),
        "anomaly_characteristics": characteristics
    }


def classify_anomaly_types(anomalous_segments: List[Dict]) -> List[Dict]:
    """Classify types of anomalies detected"""
    classified_anomalies = []
    
    for segment in anomalous_segments:
        anomaly_type = "Unknown"
        
        # Simple classification based on feature values
        if segment["cut_density"] > 2.0:  # High cut density
            anomaly_type = "Rapid Cutting"
        elif segment["motion_magnitude"] > 50.0:  # High motion
            anomaly_type = "High Motion"
        elif segment["brightness_variance"] > 1000.0:  # High brightness variance
            anomaly_type = "Visual Instability"
        elif segment["duration"] > 30.0:  # Long segment
            anomaly_type = "Extended Scene"
        else:
            anomaly_type = "Statistical Outlier"
        
        classified_anomalies.append({
            **segment,
            "anomaly_type": anomaly_type
        })
    
    return classified_anomalies