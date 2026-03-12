"""Utility functions for PolicyCapture Local."""
import re
import uuid
from pathlib import Path

from packages.shared.config import JOBS_DIR, SUPPORTED_EXTENSIONS


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid.uuid4())


def ensure_dir(path: Path | str) -> Path:
    """Ensure a directory exists, creating it if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_job_dir(job_id: str) -> Path:
    """Get the root directory for a job."""
    return JOBS_DIR / job_id


def get_job_subdir(job_id: str, subdir: str) -> Path:
    """Get a subdirectory within a job's directory."""
    return get_job_dir(job_id) / subdir


def format_timestamp_ms(ms: int) -> str:
    """Format milliseconds as MM:SS.mmm."""
    total_sec = ms / 1000
    minutes = int(total_sec // 60)
    seconds = total_sec % 60
    return f"{minutes:02d}:{seconds:06.3f}"


def safe_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    return re.sub(r"[^\w\-.]", "_", name)[:200]


def validate_video_path(path: str) -> tuple[bool, str]:
    """Validate that a video path exists and has a supported extension."""
    p = Path(path)
    if not p.exists():
        return False, f"File not found: {path}"
    if not p.is_file():
        return False, f"Not a file: {path}"
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported extension: {p.suffix}. Supported: {SUPPORTED_EXTENSIONS}"
    return True, "OK"
