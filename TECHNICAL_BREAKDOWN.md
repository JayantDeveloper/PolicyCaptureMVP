# PolicyCapture — Technical Breakdown

> Local-first screen recording, frame extraction, OCR, and entity recognition pipeline for policy documentation.
> Runs entirely at `localhost:8420` with zero cloud dependencies.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | FastAPI + Uvicorn | HTTP API, CORS, background processing |
| **Templates** | Jinja2 | Server-rendered HTML pages |
| **Frontend** | Vanilla JS, HTML5, CSS3 | SPA dashboard, recorder, review UIs |
| **Computer Vision** | OpenCV (cv2) | Frame sampling, SSIM, preprocessing, contours |
| **OCR** | Tesseract (subprocess) | Text extraction from screenshots |
| **NER** | Pure `regex` | 13 entity types, no ML dependencies |
| **Database** | SQLite (WAL mode) | Jobs, frames, screenshots, sections, reports |
| **Recording** | MediaRecorder API | Browser screen capture → WebM/MP4 |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     BROWSER (Frontend)                  │
│                                                         │
│  index.html (SPA)  ──  app.js (hash router)             │
│  recorder.html     ──  recorder.js (MediaRecorder)      │
│  frame_review.html ──  frame_review.js (timeline)       │
│  ocr_review.html   ──  ocr_review.js (entities/tables)  │
│  docs.html         ──  system documentation             │
│                                                         │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP (fetch)
                        ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI (Port 8420)                    │
│                                                         │
│  apps/local_api/main.py     ── App setup, CORS, mounts  │
│  apps/local_api/routes.py   ── All API endpoints        │
│                                                         │
│  Background threads for frame extraction & pipeline     │
│                                                         │
└───────────────────────┬─────────────────────────────────┘
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
┌──────────────┐ ┌───────────┐ ┌────────────────────┐
│   SQLite DB  │ │ File      │ │ Core Pipeline      │
│              │ │ System    │ │                    │
│ jobs         │ │ data/     │ │ validate_video     │
│ frames       │ │  jobs/    │ │ sample_frames      │
│ screenshots  │ │   {id}/   │ │ preprocess_frame   │
│ sections     │ │    input/ │ │ detect_relevance   │
│ reports      │ │    frames/│ │ scene_change       │
│              │ │    etc.   │ │ detect_elements    │
└──────────────┘ └───────────┘ │ extract_entities   │
                               │ choose_best_frame  │
                               │ dedupe_candidates  │
                               │ orchestrator       │
                               └────────────────────┘
```

---

## How the Frontend Works

### Routing Model

The app uses a **hybrid routing** approach:

- **Hash-based SPA** (`app.js`) — Dashboard at `/` handles `#/`, `#/jobs/{id}`, `#/jobs/{id}/report` client-side
- **Separate template pages** — Recorder, Frame Review, OCR Review, and Docs are full HTML pages served by FastAPI

### Page Breakdown

#### 1. Dashboard (`/` → `index.html` + `app.js`)

The main SPA. Hash-based routing renders different views:

- **Job List** (`#/`) — Grid of job cards with status badges, create/delete
- **Job Detail** (`#/jobs/{id}`) — Metadata, action buttons, screenshot gallery
- **Report View** (`#/jobs/{id}/report`) — Embedded HTML report

```
API client object handles all HTTP:
  API.getJobs(), API.createJob(), API.startProcessing(),
  API.uploadVideo(), API.generateReport(), etc.

Polling: startPolling(jobId) checks status every 3s until completed/failed
```

#### 2. Screen Recorder (`/recorder` → `recorder.html` + `recorder.js`)

**Recording flow:**

```
User clicks Record
       │
       ▼
getDisplayMedia() → live preview in video element
       │
       ▼
Canvas crop overlay appears
  ├── Drag rectangle to select crop area
  ├── Convert screen coords → video pixel coords
  └── Choose: Confirm Crop │ Full Screen │ Cancel
       │
       ▼
MediaRecorder starts (full screen always captured)
  ├── Chunks collected every 1000ms
  ├── Timer updates every second
  ├── Stop banner visible at all times (click or Escape)
  └── Waveform animation while recording
       │
       ▼
Stop → combine chunks → Blob
       │
       ▼
POST /api/jobs (create job)
POST /api/jobs/{id}/upload (upload blob)
POST /api/jobs/{id}/crop (if crop selected)
       │
       ▼
User clicks "Extract Frames" pill button
  └── Polls /api/jobs/{id}/status every 2s
  └── On completion → redirect to /jobs/{id}/frames
```

**Key details:**
- Crop is NOT applied during recording — full screen is always captured
- Crop is applied server-side during frame extraction
- Crop stored as `crop.json: {x, y, w, h}` in video pixel coordinates
- Codec selection: MP4 > WebM VP9 > WebM VP8

#### 3. Frame Review (`/jobs/{id}/frames` → `frame_review.html` + `frame_review.js`)

**Layout:**

```
┌──────────────────────────────────────────────────────┐
│ Topbar: Title │ Stats (Total, Selected, Unique) │ Done│
├──────────┬───────────────────────────────────────────┤
│          │                                           │
│  Side    │           Preview Image                   │
│  Panel   │                                           │
│          │    Time │ Score │ BBox toggle │ Select     │
│ Auto     │                                           │
│ Manual   ├───────────────────────────────────────────┤
│ All      │  Video Timeline (continuous scrub bar)    │
│          │  ●──●────●──●───●────●──●────────playhead │
│ +/x btns │  Frame Thumbnail Strip (scrollable)       │
│          │  [thumb][thumb][thumb][thumb][thumb]...    │
└──────────┴───────────────────────────────────────────┘
```

**Data loaded on init:**
```
GET /api/jobs/{id}/frames?limit=2000  → allFrames[]
GET /api/jobs/{id}/screenshots        → screenshots[]
GET /api/jobs/{id}/video-info         → videoDurationMs
```

**Features:**
- **Video timeline** — Continuous bar showing full video duration. Color-coded markers:
  - 🟢 Auto-selected (scene changes)
  - 🔵 Manual (user-extracted)
  - 🟠 Scene change (detected but not selected)
  - ⚪ Redundant (SSIM > 0.95, faded)
- **Click** timeline to scrub, **double-click** to extract frame at that position
- **Side panel tabs:** Auto shows all auto-detected frames with deselect; Manual shows user-extracted; All shows everything selected
- **Keyboard:** Arrow keys navigate, Space/Enter toggle selection, Escape closes lightbox
- **Manual extraction:** `POST /api/jobs/{id}/extract-frame-at` with `{timestamp_ms}`, frame_index >= 100000
- **Done button** → navigates to `/jobs/{id}/ocr-review`

#### 4. OCR Review (`/jobs/{id}/ocr-review` → `ocr_review.html` + `ocr_review.js`)

**Layout:**

```
┌────────────────────────────────────────────────────────┐
│ Topbar: Stats │ Run OCR │ Re-run │ Back │ Export       │
├────────────────────────────────────────────────────────┤
│ Search: [___________________________] │ Entity Chips   │
├─────────┬──────────────────┬──────────────────────────┤
│         │                  │ Tabs: Text│Entities│      │
│  Frame  │   Preview        │       Tables│Notes        │
│  Strip  │   Image          │                          │
│         │                  │  Structured text with    │
│ 00:03 ● │                  │  entity highlighting     │
│ 00:07 ● │                  │  OR entity groups        │
│ 00:12   │    OCR: 87%      │  OR HTML tables          │
│ 00:15 ● │                  │  OR annotation textarea  │
│         │                  │                          │
└─────────┴──────────────────┴──────────────────────────┘
```

**OCR processing:**
```
Click "Run OCR & Extract"
       │
       ▼
POST /api/jobs/{id}/run-ocr
  For each screenshot:
    1. Preprocess image (upscale → denoise → CLAHE → sharpen)
    2. Tesseract OCR with structured output (block/par/line)
    3. Table detection (grid lines → cell boundaries → per-cell OCR)
    4. NER regex extraction (13 entity types)
    5. Store text + entities + tables in DB
       │
       ▼
Reload OCR data → render all tabs
```

**Tabs:**
- **Text** — Full extracted text in monospace, entity spans color-coded, search highlights in yellow
- **Entities** — Grouped by type (Dates, Currency, People, etc.), click to copy
- **Tables** — Detected tables rendered as HTML `<table>` with styled headers
- **Notes** — Free-form textarea, saves to `PUT /api/screenshots/{id}/notes`

**Other features:**
- **Search** (`Cmd+F`) — Full-text search across all frames via `GET /api/jobs/{id}/search?q=...`
- **Entity filter chips** — Toggle visibility of specific entity types
- **Re-run OCR** — `POST /api/jobs/{id}/run-ocr?force=true` reprocesses all frames
- **Export** — Downloads `ocr_export_{id}.json` with all text, entities, tables, notes
- **Confidence badges** — Green (≥70%), Yellow (≥40%), Red (<40%) on each frame card

---

## How the Backend Works

### FastAPI Setup (`main.py`)

```python
app = FastAPI(title="PolicyCapture Local", version="0.1.0")

# CORS for local dev
app.add_middleware(CORSMiddleware, allow_origin_regex=r"localhost|127.0.0.1|chrome-extension://")

# Static files from review-ui
app.mount("/static", StaticFiles(directory="apps/review-ui/static"))

# Jinja2 templates
templates = Jinja2Templates(directory="apps/review-ui/templates")

# API routes prefixed with /api
app.include_router(router, prefix="/api")

# Page routes
GET /              → index.html (dashboard)
GET /recorder      → recorder.html
GET /jobs/{id}/frames     → frame_review.html
GET /jobs/{id}/ocr-review → ocr_review.html
GET /docs          → docs.html
```

### API Endpoints (`routes.py`, ~1200 lines)

#### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/jobs` | Create new job |
| `GET` | `/api/jobs` | List all jobs (newest first) |
| `GET` | `/api/jobs/{id}` | Get job details |
| `DELETE` | `/api/jobs/{id}` | Delete job + all files |
| `POST` | `/api/jobs/{id}/upload` | Upload video (multipart) |
| `POST` | `/api/jobs/{id}/crop` | Store crop region JSON |
| `POST` | `/api/jobs/{id}/register-video` | Register local video path |
| `POST` | `/api/jobs/{id}/process` | Run full pipeline (background) |
| `POST` | `/api/jobs/{id}/extract-frames` | Extract frames only (background) |
| `GET` | `/api/jobs/{id}/status` | Poll processing status |
| `GET` | `/api/jobs/{id}/video-info` | Get video duration/fps/dimensions |
| `POST` | `/api/jobs/{id}/extract-frame-at` | Extract single frame at timestamp |

#### Frames & Screenshots

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/jobs/{id}/frames` | Get frames with quality filters |
| `GET` | `/api/jobs/{id}/screenshots` | Get selected screenshots |
| `PATCH` | `/api/screenshots/{id}` | Update screenshot metadata |
| `POST` | `/api/frames/{id}/promote` | Promote frame → screenshot |
| `DELETE` | `/api/screenshots/{id}` | Remove screenshot |

#### OCR & Search

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/jobs/{id}/run-ocr` | Run OCR + NER (`?force=true` to re-run) |
| `GET` | `/api/jobs/{id}/ocr-data` | Get all OCR text + entities + tables |
| `GET` | `/api/jobs/{id}/search?q=...` | Full-text search with snippets |
| `PUT` | `/api/screenshots/{id}/notes` | Save annotations |

#### Reports & Artifacts

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/jobs/{id}/report` | Generate HTML + PDF report |
| `GET` | `/api/jobs/{id}/report` | Get report metadata |
| `GET` | `/api/jobs/{id}/report/html` | Get HTML report content |
| `GET` | `/api/jobs/{id}/sections` | Get classified sections |
| `GET` | `/api/artifacts/{id}/{type}/{file}` | Serve file (sanitized path) |

### Background Processing

Frame extraction and full pipeline processing run in **daemon threads**:

```python
_running_jobs = {}  # tracks active jobs

def _extract_frames_bg(job_id, video_path):
    try:
        # 1. Load crop, validate, sample, preprocess
        # 2. Detect relevance, scene changes, redundancy
        # 3. OCR on scene-change frames
        # 4. Generate thumbnails, persist to DB
        update_job_status(job_id, "completed")
    except Exception:
        update_job_status(job_id, "failed")

thread = threading.Thread(target=_extract_frames_bg, args=(job_id, path), daemon=True)
thread.start()
_running_jobs[job_id] = thread
```

---

## How the Pipeline Works

### Processing Stages

```
Video Upload
    │
    ▼
┌─────────────────────────────────────────────┐
│ 1. VALIDATE                                 │
│    validate_video.py                        │
│    Check: exists, format, size ≤ 2GB,       │
│    readable by OpenCV. Handle WebM FPS bug. │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 2. SAMPLE FRAMES                            │
│    sample_frames.py                         │
│    Timestamp-based extraction at 0.5s       │
│    intervals. Uses CAP_PROP_POS_MSEC        │
│    (not frame counting) for WebM compat.    │
│    Applies crop from crop.json if exists.   │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 3. QUALITY METRICS                          │
│    preprocess_frame.py                      │
│                                             │
│    blur_score = min(laplacian_var / 5000, 1)│
│    stability  = 1 - (MSE / 255²)           │
│                                             │
│    Sharp threshold: ≥ 0.15                  │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 4. RELEVANCE SCORING                        │
│    detect_relevance.py                      │
│                                             │
│    keyword_score = matched / (total × 0.3)  │
│    structure     = 0.6 × h + 0.4 × v       │
│                                             │
│    With text: 0.7 × keyword + 0.3 × struct  │
│    No text:   structure_score only          │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 5. SCENE CHANGE DETECTION                   │
│    scene_change.py                          │
│                                             │
│    similarity = 0.65×SSIM + 0.35×histogram  │
│    is_change  = similarity < 0.92           │
│                                             │
│    visual_importance =                      │
│      0.40 × scene_change_score              │
│    + 0.20 × text_density                    │
│    + 0.25 × relevance_score                 │
│    + 0.15 × blur_score                      │
│                                             │
│    Compared against LAST KEPT frame,        │
│    not previous frame.                      │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 5b. REDUNDANCY DETECTION                   │
│     (in routes.py)                          │
│                                             │
│     Compare scene-change frames pairwise.   │
│     If SSIM > 0.95 → mark redundant,       │
│     set candidate_score = 0.0               │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 6. CANDIDATE SCORING                        │
│    choose_best_frame.py                     │
│                                             │
│    candidate_score =                        │
│      0.35 × blur_score                      │
│    + 0.45 × relevance_score                 │
│    + 0.20 × stability_score                 │
│                                             │
│    Best frame per 4-second window selected. │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 7. DEDUPLICATION                            │
│    dedupe_candidates.py                     │
│                                             │
│    Perceptual hash: 8×8 grayscale → 64-bit  │
│    Hamming similarity ≥ 0.92 = duplicate    │
│    Higher-scored frame kept.                │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 8. OCR & ENTITY EXTRACTION                  │
│    detect_elements.py + extract_entities.py │
│                                             │
│    Preprocessing: upscale → denoise →       │
│      CLAHE → sharpen                        │
│    Tesseract: PSM 3, structured TSV output  │
│    Tables: grid line detection → cell OCR   │
│    NER: 13 regex entity types               │
│                                             │
│    Element classification:                  │
│      navbar, footer, button, input_field,   │
│      image, table, text_block               │
└─────────────────┬───────────────────────────┘
                  ▼
┌─────────────────────────────────────────────┐
│ 9. PERSIST & SERVE                          │
│    database.py + routes.py                  │
│                                             │
│    Store all frames with scores in SQLite   │
│    Generate 320×180 thumbnails              │
│    Auto-select scene-change frames          │
│    Serve via /api/artifacts/                │
└─────────────────────────────────────────────┘
```

### Scoring Formulas Summary

| Formula | Weights | Module |
|---------|---------|--------|
| **Candidate Score** | 35% blur + 45% relevance + 20% stability | `choose_best_frame.py` |
| **Visual Importance** | 40% scene + 20% text + 25% relevance + 15% blur | `scene_change.py` |
| **Relevance** (with text) | 70% keywords + 30% structure | `detect_relevance.py` |
| **Scene Change** | 65% SSIM + 35% histogram correlation | `scene_change.py` |
| **Structure Score** | 60% horizontal lines + 40% vertical lines | `detect_relevance.py` |
| **Blur Score** | `min(laplacian_variance / 5000, 1.0)` | `preprocess_frame.py` |
| **Stability Score** | `1.0 - (MSE / 255²)` | `preprocess_frame.py` |

### Key Thresholds

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Frame sample interval | 0.5 sec | How often to capture frames |
| Scene change | < 0.92 combined similarity | Detect new content |
| Redundancy | > 0.95 SSIM | Mark near-duplicate frames |
| Hash dedup | ≥ 0.92 similarity | Remove perceptual duplicates |
| Sharp frame | ≥ 0.15 blur_score | Minimum sharpness |
| Min candidate | 0.2 score | Minimum to be considered |
| Best frame window | 4.0 sec | Time window for selection |
| OCR upscale | < 1200px width | Upscale 2× for Tesseract |
| OCR word confidence | ≥ 30 | Include word in output |
| Min table area | 0.5% of image | Minimum table detection size |

---

## Database Schema

**SQLite** with WAL mode, foreign keys, thread-local connections.

```sql
jobs
├── id TEXT (PK, UUID)
├── title TEXT
├── source_video_path TEXT
├── status TEXT (pending → processing → completed/failed)
├── duration_ms INTEGER
├── frame_count INTEGER
├── screenshot_count INTEGER
├── created_at TEXT (ISO 8601)
└── updated_at TEXT

frames
├── id TEXT (PK, UUID)
├── job_id TEXT (FK → jobs)
├── frame_index INTEGER (≥100000 = manual)
├── timestamp_ms INTEGER
├── source_image_path TEXT
├── blur_score REAL [0,1]
├── stability_score REAL [0,1]
├── relevance_score REAL [0,1]
├── candidate_score REAL [0,1]
├── matched_keywords TEXT (JSON)
├── extracted_text TEXT
└── ocr_confidence REAL

screenshots
├── id TEXT (PK, UUID)
├── job_id TEXT (FK → jobs)
├── source_frame_id TEXT ("frame_000042")
├── image_path TEXT
├── thumbnail_path TEXT
├── captured_at_ms INTEGER
├── section_type TEXT
├── confidence REAL
├── rationale TEXT
├── matched_keywords TEXT (JSON: entities + tables)
├── extracted_text TEXT
├── accepted INTEGER (0/1)
├── notes TEXT
└── order_index INTEGER

sections
├── id TEXT (PK, UUID)
├── job_id TEXT (FK → jobs)
├── screenshot_id TEXT (FK → screenshots)
├── heading TEXT
├── section_type TEXT
├── summary TEXT
├── key_points TEXT (JSON)
├── confidence REAL
└── final_order INTEGER

reports
├── id TEXT (PK, UUID)
├── job_id TEXT (FK → jobs)
├── html_path TEXT
├── pdf_path TEXT
└── created_at TEXT
```

---

## Directory Structure

```
BAH-vid-appdev/
├── apps/
│   ├── local_api/
│   │   ├── main.py              # FastAPI app, CORS, template routes
│   │   └── routes.py            # All API endpoints (~1200 lines)
│   └── review-ui/
│       ├── templates/
│       │   ├── base.html        # Base layout with nav
│       │   ├── index.html       # Dashboard SPA shell
│       │   ├── recorder.html    # Screen capture UI
│       │   ├── frame_review.html# Timeline frame viewer
│       │   ├── ocr_review.html  # OCR + entity review
│       │   └── docs.html        # System documentation
│       └── static/
│           ├── js/
│           │   ├── app.js       # Dashboard SPA logic
│           │   ├── recorder.js  # MediaRecorder + crop
│           │   ├── frame_review.js # Timeline, scrubber
│           │   └── ocr_review.js   # OCR viewer, tables
│           └── css/
│               └── styles.css   # Dashboard styles
├── packages/
│   ├── core/pipeline/
│   │   ├── validate_video.py    # Video file validation
│   │   ├── sample_frames.py     # Timestamp-based extraction
│   │   ├── preprocess_frame.py  # Blur + stability scores
│   │   ├── detect_relevance.py  # Keyword + structure scoring
│   │   ├── scene_change.py      # SSIM + histogram detection
│   │   ├── choose_best_frame.py # Composite scoring, windowing
│   │   ├── classify_screenshot.py # Section type classification
│   │   ├── dedupe_candidates.py # Perceptual hash dedup
│   │   ├── detect_elements.py   # UI elements, OCR, tables
│   │   ├── extract_entities.py  # Regex NER (13 types)
│   │   ├── synthesize_section.py# Section summaries
│   │   ├── generate_report.py   # HTML/PDF report gen
│   │   └── orchestrator.py      # Full pipeline orchestration
│   └── shared/
│       ├── database.py          # SQLite layer (WAL, thread-local)
│       └── config.py            # All settings + env overrides
├── data/
│   ├── policycapture.db         # SQLite database
│   └── jobs/{job_id}/
│       ├── crop.json            # Crop region (if set)
│       ├── input/               # Uploaded videos
│       ├── frames/              # Sampled frame PNGs
│       ├── screenshots/         # Selected frames
│       ├── thumbnails/          # 320×180 previews
│       └── reports/             # Generated HTML/PDF
└── run.sh                       # Startup script
```

---

## End-to-End User Flow

```
1. Open http://localhost:8420
2. Click "Record" → /recorder
3. Select screen/window → draw crop region (optional)
4. Record screen actions → click Stop or press Escape
5. Click "Extract Frames" → backend samples + scores + detects scenes
6. Redirected to /jobs/{id}/frames
7. Review auto-selected frames on timeline
8. Add/remove frames, extract at specific timestamps
9. Click "Done" → /jobs/{id}/ocr-review
10. Click "Run OCR & Extract" → Tesseract + NER + table detection
11. Browse text, entities, tables per frame
12. Search across all text, add annotations
13. Export JSON or generate report
```
