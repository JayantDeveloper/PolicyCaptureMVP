"""Configuration for PolicyCapture Local."""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"
DB_PATH = DATA_DIR / "policycapture.db"

# Server settings
SERVER_HOST = os.getenv("PC_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("PC_PORT", "8420"))

# Video processing
FRAME_SAMPLE_INTERVAL_SEC = float(os.getenv("PC_FRAME_INTERVAL", "0.5"))
SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = int(os.getenv("PC_MAX_FILE_MB", "2048"))

# Relevance detection
RELEVANCE_THRESHOLD = float(os.getenv("PC_RELEVANCE_THRESHOLD", "0.3"))
SIMILARITY_THRESHOLD = float(os.getenv("PC_SIMILARITY_THRESHOLD", "0.92"))

RELEVANCE_KEYWORDS = [
    "demographics", "income", "household", "members", "eligibility",
    "application", "policy", "address", "amount", "case number",
    "benefits", "enrollment", "coverage", "deductible", "premium",
    "copay", "provider", "plan", "medicaid", "medicare",
    "snap", "tanf", "wic",
]

# Section types
SECTION_TYPES = [
    "demographics", "income", "household", "eligibility",
    "policy_guidance", "application_step", "table", "unknown",
]

# Frame quality thresholds
BLUR_THRESHOLD = 0.3  # Below this = too blurry
MIN_CANDIDATE_SCORE = 0.2  # Minimum score to be a candidate
BEST_FRAME_WINDOW_SEC = 4.0  # Time window for selecting best frame
