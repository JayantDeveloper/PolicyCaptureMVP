# PolicyCapture Local

Local-first extraction workflow for recorded screen sessions. Process recorded screen sessions to extract, classify, and synthesize evidence-style documentation from policy/benefits system reviews.

## Quick Start

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Start the server
bash scripts/run.sh

# 3. Open the dashboard
# http://localhost:8420
```

## Demo Mode

Seed demo data to test the UI without a real video:

```bash
# Via the API
curl -X POST http://localhost:8420/api/demo/seed

# Or click "Seed Demo" in the dashboard
```

## Workflow

1. **Create a job** - Give it a title
2. **Register a video** - Point to a local `.mp4` / `.mov` / `.avi` / `.mkv` / `.webm` file
3. **Process** - The pipeline extracts frames, detects relevance, classifies screenshots
4. **Review** - Accept/reject screenshots, edit notes, reorder sections
5. **Generate report** - Export an HTML evidence report

## Architecture

```
apps/
  local_api/          # FastAPI server (endpoints, job orchestration)
  review-ui/          # Jinja2 + vanilla JS frontend
    static/css/       # Styles
    static/js/        # SPA application
    templates/        # HTML templates

packages/
  shared/             # Shared utilities
    config.py         # Configuration constants
    schemas.py        # Pydantic models
    database.py       # SQLite data layer
    utils.py          # Utility functions

  core/
    pipeline/         # Processing pipeline modules
      validate_video.py       # Video validation
      sample_frames.py        # Frame extraction at intervals
      preprocess_frame.py     # Blur/quality scoring
      detect_relevance.py     # Keyword + visual structure detection
      choose_best_frame.py    # Best frame per time window
      dedupe_candidates.py    # Near-duplicate removal
      classify_screenshot.py  # Section type classification
      synthesize_section.py   # Summary/key-point generation
      generate_report.py      # HTML/PDF report generation
      orchestrator.py         # Full pipeline orchestration

data/
  jobs/{job_id}/      # Per-job artifacts
    input/            # Source video
    frames/           # Sampled frames
    screenshots/      # Selected screenshots
    thumbnails/       # Thumbnail images
    reports/          # Generated reports
  policycapture.db    # SQLite database
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/jobs` | Create a new job |
| GET | `/api/jobs` | List all jobs |
| GET | `/api/jobs/{id}` | Get job details |
| POST | `/api/jobs/{id}/upload` | Upload video file |
| POST | `/api/jobs/{id}/register-video` | Register local video path |
| POST | `/api/jobs/{id}/process` | Start processing pipeline |
| GET | `/api/jobs/{id}/status` | Get processing status |
| GET | `/api/jobs/{id}/frames` | Get extracted frames |
| GET | `/api/jobs/{id}/screenshots` | Get screenshot candidates |
| PATCH | `/api/screenshots/{id}` | Update screenshot (accept/reject/notes) |
| GET | `/api/jobs/{id}/sections` | Get extracted sections |
| POST | `/api/jobs/{id}/report` | Generate report |
| GET | `/api/jobs/{id}/report` | Get report metadata |
| GET | `/api/jobs/{id}/report/html` | Get HTML report content |
| GET | `/api/artifacts/{id}/{type}/{file}` | Serve artifact files |
| POST | `/api/demo/seed` | Seed demo data |

## Pipeline

Each pipeline stage is independently testable and replaceable:

```
video file
  -> validate_video      # Check format, size, duration
  -> sample_frames       # Extract frames at configurable interval
  -> preprocess_frame    # Compute blur/quality scores
  -> detect_relevance    # Keyword matching + visual structure detection
  -> choose_best_frame   # Select best frame per time window
  -> dedupe_candidates   # Remove near-duplicates via image hashing
  -> classify_screenshot # Assign section type (demographics, income, etc.)
  -> synthesize_section  # Generate heading, summary, key points
  -> generate_report     # Produce HTML/PDF evidence report
```

## Future Extension Points

Comments in the code mark where to integrate:

- **OCR**: `detect_relevance.py` → `mock_ocr_extract()` — replace with Tesseract or PaddleOCR
- **ML Classifier**: `classify_screenshot.py` — replace keyword rules with trained model
- **LLM Synthesis**: `synthesize_section.py` — replace heuristics with Claude API
- **PDF Generation**: `generate_report.py` → `generate_pdf_report()` — integrate ReportLab/weasyprint
- **Chrome Extension**: Optional capture mode can feed video files into the same pipeline

## Configuration

Edit `config.yaml` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PC_FRAME_INTERVAL` | `2.0` | Frame sampling interval (seconds) |
| `PC_RELEVANCE_THRESHOLD` | `0.3` | Minimum relevance score |
| `PC_SIMILARITY_THRESHOLD` | `0.92` | Deduplication similarity threshold |
| `PC_MAX_FILE_MB` | `2048` | Maximum video file size (MB) |
| `PC_HOST` | `0.0.0.0` | Server host |
| `PC_PORT` | `8420` | Server port |

## Testing

```bash
pip install pytest
pytest tests/
```

## Tech Stack

- Python 3.11+ / FastAPI / Uvicorn
- OpenCV for video processing
- SQLite for metadata
- Pydantic for schemas
- Vanilla JS + Jinja2 for review UI
