"""
Video cut detection using histogram analysis
"""

import cv2


def detect_cuts_histogram(video_path: str, sample_every_n_frames: int = 2, threshold: float = 0.55):
    """Detect cuts in video using histogram correlation"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video for cut detection.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cut_frames, cut_times, diff_series = [], [], []

    prev_hist = None
    frame_idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % sample_every_n_frames != 0:
            continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        if prev_hist is not None:
            corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            diff = float(1.0 - corr)
            diff_series.append(diff)
            if diff >= threshold:
                cut_frames.append(frame_idx)
                cut_times.append(frame_idx / fps)

        prev_hist = hist

    cap.release()
    return cut_frames, cut_times, diff_series