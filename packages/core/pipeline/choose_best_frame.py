"""
Best-frame selection module for PolicyCapture Local.

Groups candidate frames into time windows and selects the highest-scoring
frame from each window.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Scoring weights
WEIGHT_BLUR = 0.35
WEIGHT_RELEVANCE = 0.45
WEIGHT_STABILITY = 0.20


def score_candidate(frame_meta: dict) -> float:
    """
    Compute a composite score for a candidate frame.

    Expected keys in frame_meta:
        blur_score (float): Sharpness score in [0, 1].
        relevance_score (float): Relevance score in [0, 1].
        stability_score (float): Stability / low-change score in [0, 1].

    Args:
        frame_meta: Dictionary of frame metadata and quality scores.

    Returns:
        Weighted composite score in [0, 1].
    """
    blur = frame_meta.get("blur_score", 0.0)
    relevance = frame_meta.get("relevance_score", 0.0)
    stability = frame_meta.get("stability_score", 0.5)  # default neutral

    score = (
        WEIGHT_BLUR * blur
        + WEIGHT_RELEVANCE * relevance
        + WEIGHT_STABILITY * stability
    )
    return round(min(max(score, 0.0), 1.0), 4)


def choose_best_frames(
    frames: List[dict],
    window_sec: float = 4.0,
) -> List[dict]:
    """
    Group frames into time windows and select the best from each window.

    Each frame dict must contain at least:
        timestamp_ms (int): Timestamp in the source video.
        blur_score (float), relevance_score (float), stability_score (float).

    Args:
        frames: List of frame metadata dicts.
        window_sec: Duration of each time window in seconds.

    Returns:
        List of selected frame dicts (one per window), each augmented with
        a ``composite_score`` key.
    """
    if not frames:
        logger.warning("No frames provided to choose_best_frames.")
        return []

    window_ms = int(window_sec * 1000)

    # Sort by timestamp
    sorted_frames = sorted(frames, key=lambda f: f.get("timestamp_ms", 0))

    # Group into windows
    windows: List[List[dict]] = []
    current_window: List[dict] = []
    window_start = sorted_frames[0].get("timestamp_ms", 0)

    for frame in sorted_frames:
        ts = frame.get("timestamp_ms", 0)
        if ts - window_start >= window_ms and current_window:
            windows.append(current_window)
            current_window = [frame]
            window_start = ts
        else:
            current_window.append(frame)

    if current_window:
        windows.append(current_window)

    # Pick best from each window
    selected: List[dict] = []
    for i, window in enumerate(windows):
        for f in window:
            f["composite_score"] = score_candidate(f)

        best = max(window, key=lambda f: f["composite_score"])
        selected.append(best)

        logger.debug(
            "Window %d (%d-%d ms): %d candidates, best score=%.4f",
            i,
            window[0].get("timestamp_ms", 0),
            window[-1].get("timestamp_ms", 0),
            len(window),
            best["composite_score"],
        )

    logger.info(
        "Selected %d best frames from %d windows (%d total candidates).",
        len(selected),
        len(windows),
        len(frames),
    )
    return selected
