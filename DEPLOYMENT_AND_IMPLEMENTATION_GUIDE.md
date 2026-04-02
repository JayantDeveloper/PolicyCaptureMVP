# PolicyCapture Local — Deployment & Implementation Guide

> **Version:** 0.1.0 | **Last Updated:** March 2026 | **Platform:** macOS / Linux

---

## Table of Contents

1. [What Is PolicyCapture Local?](#1-what-is-policycapture-local)
2. [Why This Architecture?](#2-why-this-architecture)
3. [Technology Stack & Rationale](#3-technology-stack--rationale)
4. [System Architecture Overview](#4-system-architecture-overview)
5. [Project Structure](#5-project-structure)
6. [Deployment Guide](#6-deployment-guide)
7. [Configuration Reference](#7-configuration-reference)
8. [Database Design](#8-database-design)
9. [API Reference](#9-api-reference)
10. [Processing Pipeline — Deep Dive](#10-processing-pipeline--deep-dive)
11. [Frontend Architecture](#11-frontend-architecture)
12. [Algorithms & Scoring](#12-algorithms--scoring)
13. [Key Implementation Details](#13-key-implementation-details)
14. [User Workflow — End to End](#14-user-workflow--end-to-end)
15. [Extension Points](#15-extension-points)
16. [Troubleshooting](#16-troubleshooting)
17. [Performance Characteristics](#17-performance-characteristics)
18. [Security Considerations](#18-security-considerations)

---

## 1. What Is PolicyCapture Local?

PolicyCapture Local is a **local-first** screen recording and document extraction platform. It is purpose-built for **policy and benefits system review workflows** — the kind where a reviewer walks through a government benefits portal (Medicaid, SNAP, TANF, etc.), captures their screen session, and needs to produce structured, evidence-style documentation from what they saw.

**The problem it solves:** Manually screenshotting dozens of pages, organizing them, and writing up findings is tedious and error-prone. PolicyCapture automates the mechanical parts:

- Records the screen session as video
- Extracts the most informative frames automatically
- Runs OCR and entity recognition on those frames
- Generates a structured HTML evidence report

**The key design constraint:** Everything runs locally. No cloud APIs, no external services, no data leaves the machine. This is non-negotiable because the content being captured (benefits applications, SSNs, income data) is sensitive PII.

---

## 2. Why This Architecture?

### Local-First

All processing happens on `localhost:8420`. The SQLite database, video files, extracted frames, and reports all live in the `data/` directory on the user's machine. There is no authentication layer because there is no network exposure — the server binds to `0.0.0.0:8420` for local access only.

### Modular Pipeline

Each processing stage (frame extraction, blur detection, OCR, entity recognition, etc.) is a standalone Python module in `packages/core/pipeline/`. This means:

- Stages can be tested independently
- Stages can be replaced (e.g., swap regex NER for a trained model) without touching anything else
- The pipeline can be run partially (e.g., extract frames without running OCR)

### No ML Dependencies in Core

The core pipeline deliberately avoids heavy ML frameworks. OCR uses Tesseract (a system binary), entity extraction uses regex, and classification uses keyword rules. This keeps the install footprint small and startup instant. ML-based alternatives are marked as extension points in the code.

### Why FastAPI + Vanilla JS (Not React/Next.js)?

- **FastAPI** gives us async endpoints, automatic OpenAPI docs, Pydantic validation, and Jinja2 template rendering — all in one lightweight package
- **Vanilla JS** avoids a build step, node_modules, and framework churn. The UI is a review tool, not a SaaS product — it doesn't need React's component model
- **Jinja2 templates** give us server-rendered pages where it matters (recorder, frame review, OCR review) while `app.js` provides a lightweight SPA for the dashboard

---

## 3. Technology Stack & Rationale

| Layer | Technology | Why |
|-------|-----------|-----|
| **Runtime** | Python 3.11+ | Type hints, `match` statements, walrus operator, fast startup |
| **Web Framework** | FastAPI 0.104+ | Async, automatic OpenAPI, Pydantic integration, Jinja2 support |
| **ASGI Server** | Uvicorn | Production-grade, supports `--reload` for development |
| **Computer Vision** | OpenCV (headless) 4.8+ | Industry standard for frame extraction, image processing, SSIM |
| **OCR Engine** | Tesseract (subprocess) | Open-source, offline, no Python ML stack needed |
| **NER** | Regex (custom) | Deterministic, fast, no model files to ship, handles PII patterns |
| **Database** | SQLite (WAL mode) | Zero-config, file-based, concurrent reads, perfect for single-user local apps |
| **Data Validation** | Pydantic 2.5+ | Fast, type-safe request/response schemas |
| **Templating** | Jinja2 3.1+ | Server-side rendering for multi-page UI |
| **Frontend** | Vanilla JavaScript | No build step, no dependencies, instant page loads |
| **Image Processing** | Pillow 10+ | Thumbnail generation, image format conversion |
| **PDF (planned)** | ReportLab 4+ | Programmatic PDF generation (stub exists, not yet wired) |
| **Math** | NumPy 1.25+ | Array operations for image comparison, MSE, histograms |

### System Dependencies (not in pip)

| Dependency | Required For | Install |
|-----------|-------------|---------|
| **Tesseract** | OCR text extraction | `brew install tesseract` (macOS) / `apt install tesseract-ocr` (Ubuntu) |
| **Python 3.11+** | Runtime | `brew install python@3.11` or pyenv |

---

## 4. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Browser (localhost:8420)                    │
│                                                                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │
│  │  Dashboard   │  │  Recorder    │  │Frame Review│  │OCR Review│ │
│  │  (app.js)    │  │(recorder.js) │  │(frame_     │  │(ocr_    │ │
│  │  SPA/Hash    │  │ MediaRecorder│  │ review.js) │  │review.js)│ │
│  │  Router      │  │ API          │  │            │  │          │ │
│  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  └────┬─────┘ │
│         │                 │                │               │       │
└─────────┼─────────────────┼────────────────┼───────────────┼───────┘
          │                 │                │               │
          ▼                 ▼                ▼               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI Server (Uvicorn)                       │
│                    apps/local_api/                                │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                     routes.py (~1200 lines)                   │ │
│  │                                                                │ │
│  │  Job CRUD  │  Video Upload  │  Processing  │  OCR/NER  │ ...  │ │
│  └──────┬─────┴───────┬────────┴──────┬───────┴─────┬────────────┘ │
│         │             │               │             │              │
│  ┌──────▼─────────────▼───────────────▼─────────────▼────────────┐ │
│  │              packages/shared/                                  │ │
│  │  config.py │ database.py │ schemas.py │ utils.py              │ │
│  └──────┬─────┴──────┬──────┴────────────┴───────────────────────┘ │
│         │            │                                             │
│  ┌──────▼────────────▼───────────────────────────────────────────┐ │
│  │              packages/core/pipeline/                           │ │
│  │                                                                │ │
│  │  validate → sample → preprocess → relevance → scene_change    │ │
│  │  → choose_best → dedupe → detect_elements → extract_entities  │ │
│  │  → classify → synthesize → generate_report                    │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                    │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     data/ (filesystem)                            │
│                                                                    │
│  policycapture.db          ← SQLite (WAL mode)                    │
│  jobs/                                                             │
│    {job_id}/                                                       │
│      crop.json             ← Crop region coordinates               │
│      input/                ← Source video files                    │
│      frames/               ← Extracted frame PNGs                  │
│      processed_frames/     ← Thumbnails (320×180)                  │
│      screenshots/          ← Selected/promoted frames              │
│      reports/              ← Generated HTML reports                │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User records screen
    │
    ▼
Browser MediaRecorder API captures video (WebM/MP4)
    │
    ▼
Video blob uploaded to /api/jobs/{id}/upload
    │
    ▼
Stored in data/jobs/{id}/input/recording.webm
    │
    ▼
User clicks "Extract Frames" → POST /api/jobs/{id}/extract-frames
    │
    ▼
Pipeline runs in background thread:
    │
    ├─ Validate video (format, size, readability)
    ├─ Sample frames at 0.5s intervals (adaptive)
    ├─ Score each frame (blur, stability, relevance)
    ├─ Detect scene changes (perceptual hash + SSIM)
    ├─ Select best frame per 4-second window
    └─ Deduplicate near-identical frames
    │
    ▼
User reviews frames in Frame Review UI
    │
    ├─ Accept/reject auto-selected frames
    ├─ Manually extract frames at specific timestamps
    └─ Promote frames to screenshots
    │
    ▼
User clicks "Run OCR" → POST /api/jobs/{id}/run-ocr
    │
    ├─ Tesseract OCR on each screenshot (multi-strategy preprocessing)
    ├─ Entity extraction (20+ regex patterns)
    ├─ Table detection (grid lines → cell boundaries → per-cell OCR)
    └─ Section classification (keyword rules)
    │
    ▼
User reviews OCR results in OCR Review UI
    │
    ├─ View extracted text, entities, tables
    ├─ Add notes
    └─ Export JSON or generate report
    │
    ▼
POST /api/jobs/{id}/report → Self-contained HTML report
```

---

## 5. Project Structure

```
policycapture-local/
│
├── apps/
│   ├── local_api/
│   │   ├── main.py                 # FastAPI app initialization, CORS, route mounting
│   │   └── routes.py               # ALL API endpoints (~1200 lines)
│   │
│   └── review-ui/
│       ├── templates/
│       │   ├── base.html           # Shared layout (nav, head, scripts)
│       │   ├── index.html          # Dashboard SPA shell
│       │   ├── recorder.html       # Screen recording page
│       │   ├── frame_review.html   # Frame selection/review page
│       │   ├── ocr_review.html     # OCR results review page
│       │   └── docs.html           # System documentation page
│       │
│       └── static/
│           ├── css/
│           │   └── styles.css      # All styles (~500 lines)
│           ├── js/
│           │   ├── app.js          # Dashboard SPA + API client (~1000 lines)
│           │   ├── recorder.js     # Screen capture logic (~400 lines)
│           │   ├── frame_review.js # Frame review UI logic (~700 lines)
│           │   └── ocr_review.js   # OCR review UI logic (~600 lines)
│           └── img/
│               └── logo.png        # Application logo
│
├── packages/
│   ├── shared/
│   │   ├── config.py               # All configuration constants + env overrides
│   │   ├── database.py             # SQLite schema, CRUD operations (~340 lines)
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── utils.py                # ID generation, path validation utilities
│   │
│   └── core/
│       └── pipeline/
│           ├── orchestrator.py         # Full pipeline orchestration
│           ├── validate_video.py       # Format, size, readability checks
│           ├── sample_frames.py        # Timestamp-based frame extraction
│           ├── preprocess_frame.py     # Blur & stability scoring
│           ├── detect_relevance.py     # Keyword + visual structure detection
│           ├── scene_change.py         # Two-pass scene change detection
│           ├── choose_best_frame.py    # Windowed best-frame selection
│           ├── dedupe_candidates.py    # Perceptual hash deduplication
│           ├── classify_screenshot.py  # Rule-based section classification
│           ├── detect_elements.py      # Tesseract OCR + table detection
│           ├── extract_entities.py     # Regex NER (20+ entity types)
│           ├── synthesize_section.py   # Summary & key-point generation
│           ├── generate_report.py      # HTML report generation
│           └── medicaid_ner.py         # Experimental Medicaid-specific NER
│
├── data/
│   ├── policycapture.db            # SQLite database
│   └── jobs/
│       └── {job_id}/               # Per-job artifact storage
│           ├── crop.json
│           ├── input/
│           ├── frames/
│           ├── processed_frames/
│           ├── screenshots/
│           └── reports/
│
├── scripts/
│   ├── run.sh                      # Server startup script
│   └── seed_demo.py                # Demo data seeding
│
├── tests/
│   └── test_pipeline.py            # Pipeline integration tests
│
├── pyproject.toml                  # Python project metadata + dependencies
├── requirements.txt                # Pinned dependency list
└── README.md                       # Quick-start guide
```

---

## 6. Deployment Guide

### Prerequisites

| Requirement | Version | Check |
|------------|---------|-------|
| Python | 3.11+ | `python --version` |
| pip | Latest | `pip --version` |
| Tesseract OCR | 4.x+ | `tesseract --version` |
| Disk Space | 2GB+ free | For video/frame storage |

### Step-by-Step Installation

#### 1. Install System Dependencies

**macOS (Homebrew):**
```bash
brew install python@3.11 tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv tesseract-ocr
```

**Windows (WSL recommended):**
```bash
# Inside WSL Ubuntu
sudo apt install python3.11 python3.11-venv tesseract-ocr
```

#### 2. Clone and Set Up the Project

```bash
git clone <repository-url>
cd policycapture-local

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install from pyproject.toml with optional deps
pip install -e ".[dev,ocr]"
```

#### 3. Verify Installation

```bash
# Check Tesseract is accessible
tesseract --version

# Check Python packages
python -c "import cv2; print(cv2.__version__)"
python -c "import fastapi; print(fastapi.__version__)"
```

#### 4. Start the Server

```bash
bash scripts/run.sh
```

This runs:
```bash
python -m uvicorn apps.local_api.main:app --host 0.0.0.0 --port 8420 --reload
```

- `--host 0.0.0.0` — Listens on all interfaces (for local access)
- `--port 8420` — Default port
- `--reload` — Auto-restarts on code changes (development mode)

#### 5. Open the Dashboard

Navigate to **http://localhost:8420** in your browser.

#### 6. (Optional) Seed Demo Data

```bash
curl -X POST http://localhost:8420/api/demo/seed
```

Or click "Seed Demo" in the dashboard UI.

### Production Deployment Notes

For a more hardened deployment (e.g., on a shared workstation):

```bash
# Without --reload, with multiple workers
python -m uvicorn apps.local_api.main:app \
    --host 127.0.0.1 \
    --port 8420 \
    --workers 2

# Or with gunicorn (install separately)
pip install gunicorn
gunicorn apps.local_api.main:app \
    -w 2 \
    -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8420
```

Key changes for production:
- Bind to `127.0.0.1` instead of `0.0.0.0` to prevent network exposure
- Remove `--reload` (saves CPU)
- Use 2+ workers for concurrent requests
- Consider running behind a reverse proxy (nginx) if network access is needed

---

## 7. Configuration Reference

All configuration lives in `packages/shared/config.py`. Every value can be overridden via environment variables.

| Variable | Env Override | Default | Description |
|----------|-------------|---------|-------------|
| `SERVER_HOST` | `PC_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `PC_PORT` | `8420` | Server port |
| `FRAME_SAMPLE_INTERVAL_SEC` | `PC_FRAME_INTERVAL` | `0.5` | Seconds between sampled frames |
| `RELEVANCE_THRESHOLD` | `PC_RELEVANCE_THRESHOLD` | `0.3` | Minimum relevance score to keep a frame |
| `SIMILARITY_THRESHOLD` | `PC_SIMILARITY_THRESHOLD` | `0.92` | Threshold for deduplication (higher = more aggressive) |
| `MAX_FILE_SIZE_MB` | `PC_MAX_FILE_MB` | `2048` | Maximum upload file size (MB) |
| `BLUR_THRESHOLD` | — | `0.3` | Frames below this blur score are rejected |
| `MIN_CANDIDATE_SCORE` | — | `0.2` | Minimum composite score to be a candidate |
| `BEST_FRAME_WINDOW_SEC` | — | `4.0` | Time window for best-frame selection (seconds) |

### Supported Video Formats

`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

### Relevance Keywords

The system looks for these keywords in extracted text to determine if a frame is relevant to policy/benefits review:

```
demographics, income, household, members, eligibility,
application, policy, address, amount, case number,
benefits, enrollment, coverage, deductible, premium,
copay, provider, plan, medicaid, medicare, snap, tanf, wic
```

### Section Types

Frames are classified into these categories:

```
demographics, income, household, eligibility,
policy_guidance, application_step, table, unknown
```

---

## 8. Database Design

### Engine: SQLite with WAL Mode

**Why SQLite?**
- Zero configuration — no server process, no connection strings
- Single file (`data/policycapture.db`) — easy to backup, move, reset
- WAL (Write-Ahead Logging) mode — concurrent reads don't block writes
- Foreign keys enabled — referential integrity enforced
- Perfect for single-user, local-first applications

**Thread Safety:**
- Each thread gets its own connection via `threading.local()`
- Connections are reused within the same thread
- `check_same_thread=False` allows FastAPI's thread pool to share connections safely

### Schema

```sql
-- Core entity: a processing job tied to one video
CREATE TABLE jobs (
    id              TEXT PRIMARY KEY,    -- UUID
    title           TEXT NOT NULL,
    source_video_path TEXT,              -- Path to video file
    status          TEXT NOT NULL DEFAULT 'pending',
                    -- pending → processing → completed | failed
    duration_ms     INTEGER,            -- Video duration
    frame_count     INTEGER,            -- Total extracted frames
    screenshot_count INTEGER,           -- Selected screenshots
    created_at      TEXT NOT NULL,       -- ISO 8601 UTC
    updated_at      TEXT NOT NULL
);

-- Every frame sampled from the video
CREATE TABLE frames (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    frame_index     INTEGER NOT NULL,    -- Sequential index
    timestamp_ms    INTEGER NOT NULL,    -- Position in video
    source_image_path TEXT NOT NULL,     -- Path to PNG file
    blur_score      REAL DEFAULT 0,      -- 0.0 (blurry) → 1.0 (sharp)
    stability_score REAL DEFAULT 0,      -- 0.0 (changing) → 1.0 (static)
    relevance_score REAL DEFAULT 0,      -- 0.0 (irrelevant) → 1.0 (relevant)
    matched_keywords TEXT DEFAULT '[]',  -- JSON array of matched keywords
    extracted_text  TEXT DEFAULT '',     -- OCR text
    ocr_confidence  REAL DEFAULT 0,     -- 0.0 → 1.0
    candidate_score REAL DEFAULT 0      -- Composite score
);

-- Frames selected for the final report
CREATE TABLE screenshots (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    source_frame_id TEXT,                -- FK to frames.id (nullable for manual)
    image_path      TEXT NOT NULL,
    thumbnail_path  TEXT DEFAULT '',
    captured_at_ms  INTEGER DEFAULT 0,
    section_type    TEXT DEFAULT 'unknown',
    confidence      REAL DEFAULT 0,
    rationale       TEXT DEFAULT '',     -- Why this frame was selected
    matched_keywords TEXT DEFAULT '[]',
    extracted_text  TEXT DEFAULT '',
    accepted        INTEGER DEFAULT 1,   -- 0 = rejected, 1 = accepted
    notes           TEXT DEFAULT '',     -- User annotations
    order_index     INTEGER DEFAULT 0   -- Position in report
);

-- Synthesized report sections (one per screenshot)
CREATE TABLE sections (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    screenshot_id   TEXT REFERENCES screenshots(id),
    heading         TEXT DEFAULT '',
    section_type    TEXT DEFAULT 'unknown',
    summary         TEXT DEFAULT '',
    key_points      TEXT DEFAULT '[]',  -- JSON array
    confidence      REAL DEFAULT 0,
    final_order     INTEGER DEFAULT 0
);

-- Generated reports
CREATE TABLE reports (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL REFERENCES jobs(id),
    html_path       TEXT DEFAULT '',
    pdf_path        TEXT DEFAULT '',
    created_at      TEXT NOT NULL
);

-- Performance indexes
CREATE INDEX idx_frames_job ON frames(job_id);
CREATE INDEX idx_screenshots_job ON screenshots(job_id);
CREATE INDEX idx_sections_job ON sections(job_id);
```

### Entity Relationships

```
jobs (1) ──→ (N) frames
jobs (1) ──→ (N) screenshots
jobs (1) ──→ (N) sections
jobs (1) ──→ (N) reports
screenshots (1) ──→ (1) sections
frames (1) ──→ (0..1) screenshots  (via source_frame_id)
```

### How Data Flows Through Tables

1. **Job created** → `jobs` row with status `pending`
2. **Video registered** → `jobs.source_video_path` updated
3. **Frames extracted** → `frames` rows created (hundreds per video)
4. **Frames scored** → `frames.blur_score`, `relevance_score`, etc. updated
5. **Best frames selected** → `screenshots` rows created from top `frames`
6. **User reviews** → `screenshots.accepted`, `screenshots.notes` updated
7. **OCR runs** → `screenshots.extracted_text` populated
8. **Sections synthesized** → `sections` rows created
9. **Report generated** → `reports` row with path to HTML file

---

## 9. API Reference

### Base URL: `http://localhost:8420`

### Job Management

#### `POST /api/jobs`
Create a new processing job.

**Request:**
```json
{ "title": "March Eligibility Review" }
```

**Response:**
```json
{
  "id": "a1b2c3d4",
  "title": "March Eligibility Review",
  "status": "pending",
  "created_at": "2026-03-23T12:00:00Z"
}
```

#### `GET /api/jobs`
List all jobs, newest first.

#### `GET /api/jobs/{id}`
Get full details for a single job.

#### `DELETE /api/jobs/{id}`
Delete a job and all associated data (frames, screenshots, sections, reports, files on disk).

#### `PATCH /api/jobs/{id}/title`
Update job title.

**Request:**
```json
{ "title": "Updated Title" }
```

#### `POST /api/jobs/{id}/auto-title`
Auto-generate a title from OCR text found in the job's screenshots.

---

### Video Registration

#### `POST /api/jobs/{id}/upload`
Upload a video file (multipart form data).

**Request:** `Content-Type: multipart/form-data` with field `file`.

The video is saved to `data/jobs/{id}/input/recording.{ext}`.

#### `POST /api/jobs/{id}/register-video`
Register an existing local video file path.

**Request:**
```json
{ "path": "/path/to/video.mp4" }
```

#### `POST /api/jobs/{id}/crop`
Store a crop region to apply during frame extraction.

**Request:**
```json
{ "x": 100, "y": 50, "w": 1280, "h": 720 }
```

Saved as `data/jobs/{id}/crop.json`. The crop is applied server-side during sampling — the full video is always stored.

#### `GET /api/jobs/{id}/video-info`
Get video metadata (FPS, dimensions, duration, codec).

---

### Processing

#### `POST /api/jobs/{id}/extract-frames`
Run the frame extraction pipeline in a background thread. Returns immediately.

Stages: validate → sample → preprocess → relevance → scene change → best frame → dedupe.

#### `POST /api/jobs/{id}/process`
Run the full pipeline (extraction + classification + synthesis + report). Background thread.

#### `GET /api/jobs/{id}/status`
Poll processing status.

**Response:**
```json
{
  "status": "processing",
  "stage": "sample_frames",
  "progress": 0.45
}
```

---

### Frames & Screenshots

#### `GET /api/jobs/{id}/frames`
Get all extracted frames for a job.

**Query Parameters:**
- `min_relevance` (float, default 0.0) — Filter by minimum relevance score
- `limit` (int, default 500) — Maximum frames to return

#### `GET /api/jobs/{id}/screenshots`
Get selected screenshots.

**Query Parameters:**
- `section_type` (string) — Filter by section type
- `accepted_only` (bool) — Only return accepted screenshots

#### `PATCH /api/screenshots/{id}`
Update a screenshot's metadata.

**Request (any combination):**
```json
{
  "accepted": true,
  "notes": "Shows income verification step",
  "section_type": "income",
  "order_index": 3
}
```

#### `POST /api/frames/{id}/promote`
Promote a frame to a screenshot (copy it to screenshots, create DB record).

#### `POST /api/jobs/{id}/extract-frame-at`
Manually extract a frame at a specific timestamp.

**Request:**
```json
{ "timestamp_ms": 15000 }
```

---

### OCR & Entities

#### `POST /api/jobs/{id}/run-ocr`
Run Tesseract OCR + entity extraction + table detection on all screenshots for this job. This is a batch operation using a thread pool (4 workers).

#### `GET /api/jobs/{id}/ocr-data`
Get all OCR results (text, entities, tables) for the job.

#### `GET /api/jobs/{id}/search?q={query}`
Full-text search across all OCR-extracted text in the job.

---

### Reports

#### `POST /api/jobs/{id}/report`
Generate an HTML evidence report.

#### `GET /api/jobs/{id}/report`
Get report metadata (paths, creation date).

#### `GET /api/jobs/{id}/report/html`
Get the HTML report content directly.

#### `GET /api/jobs/{id}/sections`
Get synthesized sections for the job.

---

### File Serving

#### `GET /api/artifacts/{job_id}/{type}/{filename}`
Serve artifact files (frames, screenshots, thumbnails, reports).

- `type` is one of: `frames`, `processed_frames`, `screenshots`, `reports`, `input`
- Paths are sanitized to prevent directory traversal

---

### Utilities

#### `POST /api/demo/seed`
Seed demo data for testing the UI without a real video recording.

---

## 10. Processing Pipeline — Deep Dive

The pipeline is the heart of the system. Each stage is a Python module in `packages/core/pipeline/`.

### Stage 1: Video Validation (`validate_video.py`)

**What it does:** Checks that the video file is valid before processing begins.

**Checks:**
- File exists on disk
- Extension is in the supported set (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`)
- File size ≤ 2GB (configurable via `PC_MAX_FILE_MB`)
- OpenCV can open and read at least one frame
- Extracts metadata: FPS, frame count, duration

**WebM workaround:** Browser MediaRecorder often reports incorrect FPS (1000fps) for WebM. The validator detects this and falls back to seek-to-end duration measurement using `CAP_PROP_POS_MSEC`.

---

### Stage 2: Frame Sampling (`sample_frames.py`)

**What it does:** Extracts frames from the video at regular intervals.

**How it works:**
1. Opens video with OpenCV (`cv2.VideoCapture`)
2. Seeks to timestamps at `FRAME_SAMPLE_INTERVAL_SEC` intervals (default: 0.5s)
3. Uses `CAP_PROP_POS_MSEC` for seeking (not frame counting — critical for WebM compatibility)
4. Applies crop region from `crop.json` if it exists
5. Saves each frame as a PNG in `data/jobs/{id}/frames/frame_{index}.png`
6. Creates a 320×180 thumbnail in `data/jobs/{id}/processed_frames/`

**Adaptive sampling:** The sampler can dynamically adjust density:
- Static sections (pixel correlation > 0.995) → skip frames
- High-change sections (similarity < 0.97) → increase sampling density
- This reduces frame count by 30-60% on idle-heavy recordings

**Output:** Hundreds of frame PNGs + `frames` database records.

---

### Stage 3: Preprocessing (`preprocess_frame.py`)

**What it does:** Computes quality scores for each frame.

**Blur Score:**
```
blur_score = min(laplacian_variance / 5000, 1.0)
```
- Laplacian operator detects edges; variance measures edge strength
- High variance = sharp image, low variance = blurry
- Divided by 5000 to normalize to 0.0–1.0 range
- Threshold: ≥ 0.15 is considered "sharp enough"

**Stability Score:**
```
stability = 1.0 - (MSE / 255²)
```
- MSE = mean squared error between current frame and previous frame (grayscale)
- Low MSE (frames are similar) → high stability
- High MSE (major change) → low stability
- A perfectly static screen scores 1.0

---

### Stage 4: Relevance Detection (`detect_relevance.py`)

**What it does:** Determines how relevant each frame is to the policy/benefits review task.

**Two signals combined:**

**1. Keyword Score (70% weight):**
- Runs lightweight OCR on the frame (Tesseract PSM 3, downscaled for speed)
- Searches for matches against 23 policy/benefits keywords
- Score = `min(matched_count / (total_keywords × 0.3), 1.0)`

**2. Structure Score (30% weight):**
- Runs Canny edge detection on the frame
- Runs Hough line transform to find horizontal and vertical lines
- Score = `0.6 × horizontal_line_ratio + 0.4 × vertical_line_ratio`
- Tables, forms, and structured UI elements score high

**Combined:**
```
relevance = 0.7 × keyword_score + 0.3 × structure_score
```

Frames below `RELEVANCE_THRESHOLD` (0.3) are dropped.

---

### Stage 5: Scene Change Detection (`scene_change.py`)

**What it does:** Identifies frames where the visual content changes significantly (new page, new screen, scroll).

**Two-Pass Approach (for speed):**

**Pass 1 — Perceptual Hash (fast pre-filter):**
- Compute DCT-based perceptual hash (16×16 grid) for each frame
- Compare adjacent frames by Hamming distance
- Frames with high similarity are quickly ruled out (< 1ms per comparison)

**Pass 2 — SSIM + Histogram (accurate, selective):**
- Only run on "ambiguous" frames from Pass 1
- Compute SSIM (Structural Similarity Index) between adjacent frames
- Compute histogram correlation
- Combined: `similarity = 0.65 × SSIM + 0.35 × histogram_correlation`
- Threshold: similarity < 0.92 = scene change

**Why two passes?** Pure SSIM is accurate but costs 50-100ms per comparison. The perceptual hash pre-filter eliminates obvious non-changes in < 1ms, so SSIM only runs on ~20-40% of frames. Net speedup: 2-3×.

---

### Stage 6: Best Frame Selection (`choose_best_frame.py`)

**What it does:** From all the frames that passed quality and relevance filters, selects the single best frame for each time window.

**Algorithm:**
1. Group frames into 4-second windows (configurable via `BEST_FRAME_WINDOW_SEC`)
2. For each window, compute composite score:
   ```
   candidate_score = 0.35 × blur_score + 0.45 × relevance_score + 0.20 × stability_score
   ```
3. Select the frame with the highest composite score in each window
4. These become the "candidate screenshots"

---

### Stage 7: Deduplication (`dedupe_candidates.py`)

**What it does:** Removes near-duplicate frames that might appear across different time windows.

**Algorithm:**
- Compute perceptual hash for each candidate frame
- Compare all pairs
- If similarity ≥ 0.92 → mark the lower-scored frame as duplicate
- Duplicate frames get `candidate_score = 0.0`

---

### Stage 8: OCR & Element Detection (`detect_elements.py`)

**What it does:** Extracts text and detects tables from screenshots.

**Multi-Strategy OCR Preprocessing:**

The system tries multiple preprocessing strategies and picks the one with the best results:

| Strategy | Technique | Best For |
|----------|-----------|----------|
| Standard | Denoise → CLAHE → sharpen | Clean screen captures |
| Otsu | Global binarization | High-contrast documents |
| Adaptive | Local threshold (block size 11) | Uneven lighting, gradients |
| Heavy | 3× upscale → aggressive denoise → deskew | Low-resolution captures |

**Tesseract Configuration:**
- Page Segmentation Mode 3 (auto, fully automatic page segmentation)
- TSV output for structured data (block, paragraph, line, word positions)
- Confidence filtering: words below 30% confidence are discarded

**Table Detection:**
1. Detect horizontal and vertical grid lines (morphological operations)
2. Find line intersections to determine cell boundaries
3. Run per-cell OCR within detected boundaries
4. Output structured table data (rows × columns × text)

**Parallelism:** Uses `ThreadPoolExecutor` with 4 workers for batch processing.

---

### Stage 9: Entity Extraction (`extract_entities.py`)

**What it does:** Identifies structured entities in OCR text using 20+ regex patterns.

**Entity Types:**

| Category | Entity Types |
|----------|-------------|
| **Dates/Times** | `date` (MM/DD/YYYY, YYYY-MM-DD, Month DD YYYY, etc.), `time_value` |
| **Identifiers** | `ssn`, `ein`, `npi`, `policy_number`, `case_number`, `claim_number`, `group_number`, `account_number`, `id_number` |
| **Financial** | `currency` ($, USD, EUR, GBP), `percentage` |
| **Contact** | `phone`, `email`, `url` |
| **Location** | `address`, `zip_code`, `state` |
| **Identity** | `person_name` (title-case heuristic), `organization` |
| **Medical** | `medical_code` (ICD, CPT, HCPCS patterns) |

**Special Features:**
- **Overlapping deduplication:** If multiple patterns match the same text span, the most specific one wins
- **PII masking:** SSNs displayed as `***-**-XXXX`
- **Form field detection:** Extracts key-value pairs from form-style text
- **List extraction:** Detects bulleted, numbered, and lettered lists
- **Section header detection:** Identifies section headings by formatting patterns

---

### Stage 10: Classification (`classify_screenshot.py`)

**What it does:** Assigns each screenshot to a section type (demographics, income, household, etc.).

**Approach:** Rule-based keyword matching. Each section type has a set of trigger keywords:

```
demographics → name, address, date of birth, ssn, contact, phone, email
income       → income, wages, salary, employment, employer, pay stub
household    → household, members, dependents, family size, spouse
eligibility  → eligible, eligibility, qualify, determination, approved, denied
...
```

**Scoring:** Count keyword matches × weight. Highest-scoring category wins. Structure score (table detection) gives a boost to the "table" category.

---

### Stage 11: Synthesis (`synthesize_section.py`)

**What it does:** Generates a human-readable heading, summary, and key points for each screenshot section.

**Heuristic-based (no LLM):**
- Heading: derived from section type ("Demographics Information", "Income Verification", etc.)
- Summary: first 200 characters of extracted text, cleaned up
- Key points: matched keywords formatted as bullet points
- Ordering weight: 10-99 based on section type priority

---

### Stage 12: Report Generation (`generate_report.py`)

**What it does:** Produces a self-contained HTML evidence report.

**Output features:**
- Self-contained: images embedded as base64 data URIs (no external file references)
- Professional styling: blue/gray theme, card-based layout
- Sections ordered by `final_order`
- Each section includes: heading, screenshot image, summary, key points, extracted text
- PDF generation is stubbed out (extension point for ReportLab/weasyprint)

---

## 11. Frontend Architecture

### Routing Model

The frontend uses a **hybrid routing** approach:

| Route | Type | File |
|-------|------|------|
| `/` | Jinja2 template → SPA | `index.html` + `app.js` |
| `/#/` | Hash route (job list) | `app.js` |
| `/#/jobs/{id}` | Hash route (job detail) | `app.js` |
| `/#/jobs/{id}/report` | Hash route (report view) | `app.js` |
| `/recorder` | Jinja2 template page | `recorder.html` + `recorder.js` |
| `/jobs/{id}/frames` | Jinja2 template page | `frame_review.html` + `frame_review.js` |
| `/jobs/{id}/ocr-review` | Jinja2 template page | `ocr_review.html` + `ocr_review.js` |
| `/docs` | Jinja2 template page | `docs.html` |

**Why hybrid?** The dashboard (job list, detail, report) is simple enough for hash-based SPA routing. The recorder, frame review, and OCR review pages are complex enough to warrant their own server-rendered pages with dedicated JavaScript.

### app.js — Dashboard SPA

**Responsibilities:**
- `API` object: centralized fetch wrapper for all API calls
- `Toast` notifications: success/error feedback
- Hash router: parses `window.location.hash` to determine which view to render
- Job list view: cards with status badges, create/delete actions
- Job detail view: video info, processing controls, screenshot gallery
- Report view: embedded HTML report display
- Modal system: create job, rename job dialogs

### recorder.js — Screen Capture

**Responsibilities:**
- `navigator.mediaDevices.getDisplayMedia()` — requests screen sharing permission
- Live preview of the captured screen
- Interactive crop overlay (drag to select a region)
- Coordinate transformation: screen CSS pixels → video pixel coordinates (handles DPI scaling)
- MediaRecorder API with codec negotiation:
  1. Try `video/mp4` (Safari)
  2. Try `video/webm;codecs=vp9` (Chrome preferred)
  3. Fall back to `video/webm;codecs=vp8`
- Records to a Blob, then uploads via `POST /api/jobs/{id}/upload`
- Stores crop region via `POST /api/jobs/{id}/crop`

### frame_review.js — Frame Selection

**Responsibilities:**
- Timeline visualization: continuous horizontal scrubber showing all frames by timestamp
- Color-coded markers:
  - Green = auto-selected (high candidate score)
  - Blue = manually extracted
  - Orange = scene change detected
  - Gray = redundant/duplicate
- Frame gallery: thumbnail grid with selection state
- Side panel with tabs: Auto-selected, Manual, All Frames
- Confidence score badges on each thumbnail
- Actions: promote frame to screenshot, remove from selection, extract at specific timestamp
- Keyboard shortcuts: arrow keys (navigate), Space (select/deselect), Escape (close)

### ocr_review.js — OCR Results

**Responsibilities:**
- Frame strip: horizontal scrollable strip of screenshots
- Tabbed content panel:
  - **Text** tab: raw OCR output
  - **Entities** tab: extracted entities with color-coded badges by type
  - **Tables** tab: rendered HTML tables from detected grid structures
  - **Notes** tab: free-form text area for user annotations
- Entity filter chips: click to filter by entity type
- Search: full-text search across all OCR results
- Entity highlighting: entities are highlighted inline in the text view
- Export: download all OCR data as structured JSON

### styles.css — Unified Styling

- CSS custom properties for theming
- Primary color: `#0984e3` (blue)
- Dark color: `#2d3436` (charcoal)
- Card-based layout with subtle shadows
- Badge system for status indicators
- Modal overlay pattern
- Timeline and gallery grid styles
- Responsive: mobile-first with breakpoints

---

## 12. Algorithms & Scoring

### Blur Detection
```
laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
blur_score = min(laplacian.var() / 5000.0, 1.0)

Interpretation:
  0.00 – 0.15 = blurry (rejected)
  0.15 – 0.40 = acceptable
  0.40 – 1.00 = sharp
```

### Frame Stability
```
mse = mean((frame_a - frame_b) ** 2)
stability = 1.0 - (mse / 65025.0)    # 255² = 65025

Interpretation:
  1.00 = identical frames (static screen)
  0.95 = minor change (cursor move, animation)
  0.70 = significant change (scroll, navigation)
  < 0.50 = major scene change
```

### Relevance Scoring
```
keyword_score = min(matched_keywords / (23 × 0.3), 1.0)
structure_score = 0.6 × horizontal_lines + 0.4 × vertical_lines
relevance = 0.7 × keyword_score + 0.3 × structure_score
```

### Scene Change Detection
```
# Pass 1: Perceptual hash
phash_similarity = 1 - (hamming_distance / hash_bits)
if phash_similarity > 0.95 → not a scene change (skip)
if phash_similarity < 0.85 → scene change (skip pass 2)

# Pass 2: SSIM + histogram (ambiguous cases only)
combined = 0.65 × SSIM + 0.35 × histogram_correlation
if combined < 0.92 → scene change
```

### Candidate Scoring (Final Frame Selection)
```
candidate_score = 0.35 × blur + 0.45 × relevance + 0.20 × stability
```

### Redundancy Detection
```
if SSIM(frame_i, frame_j) > 0.95 → frame_j is redundant
```

---

## 13. Key Implementation Details

### WebM Compatibility

The browser's `MediaRecorder` API primarily produces WebM video with VP8 or VP9 codecs. OpenCV's WebM support has a known issue: `CAP_PROP_FPS` often returns 1000.0 instead of the actual framerate. PolicyCapture handles this:

1. **Detection:** If reported FPS > 120, it's likely wrong
2. **Duration fallback:** Seek to end of video, read `CAP_PROP_POS_MSEC` for actual duration
3. **Frame extraction:** Uses `CAP_PROP_POS_MSEC` for seeking (not frame counting) — this always works correctly regardless of FPS metadata

### Crop Region Handling

Crop is deliberately NOT applied during recording:
- Full screen is always captured (for smooth playback and potential re-cropping)
- Crop region stored as `{x, y, w, h}` in video pixel coordinates
- Applied server-side during `sample_frames` using NumPy array slicing
- Coordinate transformation in `recorder.js` handles DPI scaling between CSS pixels and video pixels

### Background Processing

Long-running operations (frame extraction, OCR) run in Python background threads (`threading.Thread`). The API returns immediately and clients poll `GET /api/jobs/{id}/status`. This avoids HTTP timeout issues and allows the user to continue interacting with the UI.

### File Organization

Each job gets an isolated directory:
```
data/jobs/{uuid}/
  crop.json              # Crop region (optional)
  input/                 # Source video
    recording.webm
  frames/                # ALL sampled frames
    frame_000.png
    frame_001.png
    ...
  processed_frames/      # Thumbnails (320×180)
    frame_000.png
    ...
  screenshots/           # Selected frames
    screenshot_001.png
    ...
  reports/               # Generated reports
    report.html
```

### ID Generation

All IDs are UUIDs generated by `packages/shared/utils.py`. This ensures uniqueness across jobs, frames, screenshots, and sections without coordination.

### Path Sanitization

The artifact serving endpoint (`/api/artifacts/{id}/{type}/{file}`) sanitizes file paths to prevent directory traversal attacks. Only allowed subdirectories (`frames`, `processed_frames`, `screenshots`, `reports`, `input`) can be accessed.

---

## 14. User Workflow — End to End

### Step 1: Create a Job

Open `http://localhost:8420`. Click **"New Job"**. Enter a title (e.g., "March SNAP Eligibility Review"). This creates a `jobs` row with status `pending`.

### Step 2: Record the Screen

Click **"Record"** on the job card. This opens the Recorder page.

1. Click **"Start Recording"** → browser requests screen sharing permission
2. Select the screen/window to capture
3. (Optional) Drag on the preview to define a crop region
4. Navigate through the policy/benefits system while recording
5. Click **"Stop Recording"**
6. The video blob is uploaded to the server automatically

### Step 3: Extract Frames

After upload, click **"Extract Frames"**. The pipeline runs in the background:

1. Validates the video
2. Samples frames every 0.5 seconds
3. Scores each frame for blur, stability, and relevance
4. Detects scene changes
5. Selects the best frame per 4-second window
6. Removes near-duplicates

You're redirected to the **Frame Review** page.

### Step 4: Review Frames

The Frame Review page shows:
- A **timeline scrubber** at the top with color-coded markers
- A **thumbnail gallery** of all extracted frames
- A **side panel** showing auto-selected and manual frames

Actions available:
- Click a frame to view it full-size
- Click the checkmark to accept/reject a frame
- Use the timeline to navigate to a specific moment
- Click "Extract Frame" to manually grab a frame at any timestamp
- Use arrow keys to navigate, Space to select/deselect

When done, click **"Continue to OCR Review"**.

### Step 5: Run OCR

On the OCR Review page, click **"Run OCR & Extract"**. This processes all selected screenshots:

1. Multi-strategy OCR preprocessing
2. Tesseract text extraction
3. Table detection and per-cell OCR
4. Entity extraction (dates, SSNs, currencies, etc.)
5. Section classification

### Step 6: Review Results

Browse the results:
- **Text tab:** Raw OCR output for each screenshot
- **Entities tab:** Extracted entities with color-coded badges
- **Tables tab:** Detected tables rendered as HTML
- **Notes tab:** Add your own annotations

Use the search bar to find specific text across all screenshots. Filter by entity type using the chips.

### Step 7: Generate Report

Click **"Generate Report"** to produce a self-contained HTML evidence report. The report includes:
- All accepted screenshots (embedded as base64 images)
- Section headings and summaries
- Extracted key points
- OCR text

### Step 8: Export

- Download the HTML report (self-contained, no external dependencies)
- Export OCR data as JSON for integration with other systems

---

## 15. Extension Points

The codebase is designed to be modular. These are the marked extension points:

### Replace OCR Engine
**File:** `packages/core/pipeline/detect_elements.py`
**Current:** Tesseract via subprocess
**Upgrade to:** PaddleOCR, EasyOCR, or a cloud API (Azure Document Intelligence, AWS Textract)

### Replace NER with ML Model
**File:** `packages/core/pipeline/extract_entities.py`
**Current:** 20+ regex patterns
**Upgrade to:** SpaCy, GLiNER, or a fine-tuned transformer model
**Note:** `medicaid_ner.py` contains an experimental Medicaid-specific NER module

### Replace Classification with Trained Model
**File:** `packages/core/pipeline/classify_screenshot.py`
**Current:** Keyword-based rules
**Upgrade to:** Fine-tuned image classifier or multimodal model

### Replace Synthesis with LLM
**File:** `packages/core/pipeline/synthesize_section.py`
**Current:** Heuristic summarization
**Upgrade to:** Claude API for intelligent summaries, key-point extraction, and report narratives

### Add PDF Generation
**File:** `packages/core/pipeline/generate_report.py`
**Current:** HTML only (PDF is stubbed)
**Upgrade to:** ReportLab or weasyprint for styled PDF output

### Add Chrome Extension
**New component** that captures screen via Chrome APIs and feeds video into the same pipeline. The API is already designed to accept uploaded video from any source.

---

## 16. Troubleshooting

### Server won't start

```
ERROR: [Errno 48] Address already in use
```
Port 8420 is occupied. Kill the existing process:
```bash
lsof -i :8420
kill -9 <PID>
```
Or change the port: `PC_PORT=8421 bash scripts/run.sh`

### Tesseract not found

```
FileNotFoundError: tesseract is not installed or not in PATH
```
Install Tesseract:
```bash
brew install tesseract        # macOS
sudo apt install tesseract-ocr  # Ubuntu
```

### OCR returns empty text

- Check that the screenshot has readable text (not just graphics)
- Verify Tesseract language packs: `tesseract --list-langs`
- Try increasing frame resolution (reduce crop area)
- Check blur score — very blurry frames produce poor OCR

### Video won't process

- Ensure the file is one of: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
- File size must be < 2GB
- OpenCV must be able to read it: `python -c "import cv2; print(cv2.VideoCapture('path').isOpened())"`

### Frames are all blurry

- Reduce `PC_FRAME_INTERVAL` to sample more frequently (e.g., `0.25`)
- Lower `BLUR_THRESHOLD` in `config.py` to accept slightly blurry frames
- Ensure the source recording was at reasonable resolution

### Database is locked

SQLite WAL mode handles most concurrency, but if you see "database is locked":
- Kill any zombie server processes
- Delete `data/policycapture.db-wal` and `data/policycapture.db-shm` (they regenerate)
- Restart the server

### WebM FPS issues

If frame extraction produces wrong timestamps, this is the WebM FPS bug. The code handles it automatically, but if issues persist:
- Try recording in MP4 instead (Safari supports this)
- Or re-encode: `ffmpeg -i input.webm -c:v libx264 output.mp4`

---

## 17. Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| Frame sampling (1 min video) | 3-8 seconds | Depends on resolution and interval |
| Blur/stability scoring | < 1ms per frame | Pure NumPy |
| Relevance detection | 10-50ms per frame | Includes lightweight OCR |
| Scene change (Pass 1 — phash) | < 1ms per comparison | DCT-based hash |
| Scene change (Pass 2 — SSIM) | 50-100ms per comparison | Only on ambiguous frames |
| Tesseract OCR | 50-200ms per frame | Depends on content density |
| Entity extraction | < 10ms per frame | Pure regex |
| Table detection | 100-500ms per frame | Only on frames with grid lines |
| Report generation | < 1 second | HTML with base64 images |
| **Full pipeline (1 min video)** | **15-45 seconds** | **End to end** |
| **Full pipeline (5 min video)** | **60-180 seconds** | **Depends on content variety** |

### Storage

| Asset | Size per Frame | Notes |
|-------|---------------|-------|
| Full frame PNG | 200-800 KB | 1920×1080 typical |
| Thumbnail PNG | 15-30 KB | 320×180 |
| Total per minute of video | 50-200 MB | At 0.5s interval |

---

## 18. Security Considerations

### PII Handling

- **All data stays local.** No network calls, no telemetry, no cloud storage.
- SSNs are masked in the entity extraction output (`***-**-XXXX`)
- The database and artifact files should be protected by OS-level file permissions

### Path Traversal Protection

The artifact serving endpoint sanitizes file paths:
- Only allows access to known subdirectories (`frames`, `screenshots`, etc.)
- Rejects paths containing `..` or absolute paths
- Job IDs are validated as UUIDs

### Input Validation

- Video file sizes are capped (default 2GB)
- Video formats are allowlisted
- Pydantic models validate all API request bodies
- File uploads use multipart form data (not raw body)

### No Authentication

This is intentional. The server is designed to run on `localhost` for a single user. If you need multi-user access:
- Add an authentication middleware to FastAPI
- Bind to `127.0.0.1` instead of `0.0.0.0`
- Put a reverse proxy (nginx + basic auth) in front

### CORS

CORS is configured to allow all origins (`*`) for local development. Tighten this if deploying on a network.

---

*This guide covers the complete PolicyCapture Local system as of March 2026. For questions or contributions, see the project repository.*
