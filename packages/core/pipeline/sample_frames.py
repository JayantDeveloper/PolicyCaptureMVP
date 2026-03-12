"""
Frame sampling module for PolicyCapture Local.

Extracts frames from a video at a configurable interval and saves them as PNG files.
Uses timestamp-based sampling (CAP_PROP_POS_MSEC) instead of FPS-based frame counting,
which is critical for WebM files from MediaRecorder that report unreliable FPS metadata.
"""

import logging
import os
from typing import List

import cv2

logger = logging.getLogger(__name__)


def sample_frames(
    video_path: str,
    output_dir: str,
    interval_sec: float = 2.0,
) -> List[dict]:
    """
    Sample frames from a video at a fixed time interval.

    Uses the actual timestamp reported by OpenCV (CAP_PROP_POS_MSEC) rather than
    relying on FPS metadata, which is unreliable for WebM/MediaRecorder files.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory where extracted frame PNGs will be saved.
        interval_sec: Seconds between each sampled frame.

    Returns:
        List of dicts, each containing:
            frame_index (int): Sequential index of the sampled frame.
            timestamp_ms (int): Timestamp in the video in milliseconds.
            image_path (str): Absolute path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(
        "Sampling frames from %s — reported fps=%.1f, total_frames=%d, interval=%.1fs",
        video_path,
        fps,
        total_frames,
        interval_sec,
    )

    interval_ms = interval_sec * 1000.0
    sampled: List[dict] = []
    sample_index = 0
    next_sample_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        # Always capture the first frame; then capture when we pass the next interval
        if sample_index == 0 or timestamp_ms >= next_sample_ms:
            ts_int = int(timestamp_ms)
            filename = f"frame_{sample_index:06d}_{ts_int}ms.png"
            image_path = os.path.join(output_dir, filename)

            cv2.imwrite(image_path, frame)

            sampled.append(
                {
                    "frame_index": sample_index,
                    "timestamp_ms": ts_int,
                    "image_path": os.path.abspath(image_path),
                }
            )
            logger.debug(
                "Saved frame %d at %d ms -> %s",
                sample_index,
                ts_int,
                image_path,
            )
            sample_index += 1
            next_sample_ms = timestamp_ms + interval_ms

    cap.release()
    logger.info("Sampled %d frames from %s", len(sampled), video_path)
    return sampled
