"""
Video validation module for PolicyCapture Local.

Validates video files for format, size, and readability before pipeline processing.
"""

import logging
import os
from typing import Optional

import cv2

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB


def validate_video(video_path: str) -> dict:
    """
    Validate a video file for pipeline processing.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        dict with keys:
            valid (bool): Whether the video passed all checks.
            duration_ms (int): Video duration in milliseconds.
            width (int): Frame width in pixels.
            height (int): Frame height in pixels.
            fps (float): Frames per second.
            frame_count (int): Total number of frames.
            codec (str): Four-character codec code.
            file_size_mb (float): File size in megabytes.
            error (Optional[str]): Error description if validation failed.
    """
    result: dict = {
        "valid": False,
        "duration_ms": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "codec": "",
        "file_size_mb": 0.0,
        "error": None,
    }

    # --- Check file exists ---
    if not os.path.isfile(video_path):
        result["error"] = f"File not found: {video_path}"
        logger.error(result["error"])
        return result

    # --- Check extension ---
    _, ext = os.path.splitext(video_path)
    if ext.lower() not in SUPPORTED_EXTENSIONS:
        result["error"] = (
            f"Unsupported extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
        logger.error(result["error"])
        return result

    # --- Check file size ---
    file_size = os.path.getsize(video_path)
    result["file_size_mb"] = round(file_size / (1024 * 1024), 2)
    if file_size > MAX_FILE_SIZE_BYTES:
        result["error"] = (
            f"File size {result['file_size_mb']:.1f} MB exceeds 2 GB limit."
        )
        logger.error(result["error"])
        return result

    # --- Read video properties with OpenCV ---
    cap: Optional[cv2.VideoCapture] = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result["error"] = f"OpenCV could not open video: {video_path}"
            logger.error(result["error"])
            return result

        result["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result["fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
        result["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Codec as four-character code
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        result["codec"] = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        )

        # Duration: for WebM/MediaRecorder files, FPS metadata is often wrong
        # (e.g. 1000fps). Use seek-to-end to get actual duration.
        if result["fps"] > 0 and result["frame_count"] > 0:
            result["duration_ms"] = int(
                (result["frame_count"] / result["fps"]) * 1000
            )

        # If duration looks wrong (0 or implausibly short), get it from timestamps
        if result["duration_ms"] <= 0 or result["fps"] >= 500:
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)  # seek to end
            end_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if end_ms > 0:
                result["duration_ms"] = int(end_ms)
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)  # seek back to start

        if result["duration_ms"] <= 0 and result["fps"] <= 0:
            result["error"] = "Could not determine video duration; file may be corrupt."
            logger.error(result["error"])
            return result

        result["valid"] = True
        logger.info(
            "Video validated: %s (%dx%d, %.1f fps, %d frames, %.1f MB)",
            video_path,
            result["width"],
            result["height"],
            result["fps"],
            result["frame_count"],
            result["file_size_mb"],
        )

    except Exception as exc:
        result["error"] = f"Unexpected error reading video: {exc}"
        logger.exception(result["error"])
    finally:
        if cap is not None:
            cap.release()

    return result
