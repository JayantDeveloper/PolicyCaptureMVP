# PolicyCapture — V1 Experimental Memory Dump

**Branch**: `experimental` on `BoozAllenHamilton-backend-Spring2026`
**Snapshot Date**: March 18, 2026
**Status**: Feature-complete for demo/review

---

## 1. What This Thing Is

PolicyCapture is a local-first tool that records your screen while you browse Medicaid policy websites, intelligently extracts the important frames, runs OCR to pull out text, then runs a four-layer NER pipeline to identify Medicaid-specific entities (eligibility groups, waivers, provider types, etc.). Everything runs on localhost:8420 — zero cloud, zero API keys.

---

## 2. Architecture

```
Browser Recording → Video File → Frame Extraction → Scene Change Detection
    → Frame Review UI → Screenshot Promotion → OCR (Tesseract)
    → NER (Dictionary + spaCy + GLiNER + Normalization)
    → Entity Review UI → Report Generation → Export (.docx/.txt)
```

**Stack**:
- Backend: FastAPI + uvicorn (port 8420, --reload)
- Frontend: Vanilla JS SPA, hash-based routing, no React/Vue/anything
- Database: SQLite WAL mode (`data/policycapture.db`)
- OCR: Tesseract with multi-strategy preprocessing
- NER: Four-layer (dictionary → spaCy → GLiNER → normalization)
- CV: OpenCV for all image processing
- Extension: Chrome Manifest V3 for recording trigger

---

## 3. Directory Map

```
BAH-vid-appdev/
├── apps/
│   ├── local_api/
│   │   └── routes.py              # ~1500 lines — ALL API endpoints
│   └── review-ui/
│       ├── templates/
│       │   ├── base.html           # Main layout, navbar with logo
│       │   ├── recorder.html       # Screen recorder page
│       │   ├── frame_review.html   # Frame review (dark theme, timeline, side panel)
│       │   ├── ocr_review.html     # OCR & entity review (dark theme)
│       │   └── docs.html           # System documentation
│       └── static/
│           ├── css/styles.css      # All styles (~710 lines)
│           ├── js/
│           │   ├── app.js          # ~700 lines — SPA router, job CRUD, rename
│           │   ├── recorder.js     # Screen recording logic
│           │   ├── frame_review.js # ~500 lines — frame review page
│           │   └── ocr_review.js   # OCR review page
│           └── img/
│               ├── logo.svg        # Dark logo (light backgrounds)
│               └── logo-light.svg  # Light logo (dark backgrounds)
├── packages/
│   ├── core/pipeline/
│   │   ├── medicaid_ner.py         # ~950 lines — FOUR-LAYER MEDICAID NER
│   │   ├── extract_entities.py     # ~760 lines — Regex NER + form extraction
│   │   ├── scene_change.py         # ~257 lines — Two-pass scene detection
│   │   ├── detect_elements.py      # OCR + visual element detection (Tesseract)
│   │   ├── preprocess_frame.py     # Image preprocessing strategies
│   │   ├── detect_relevance.py     # Relevance scoring
│   │   ├── dedupe_candidates.py    # Deduplication
│   │   ├── choose_best_frame.py    # Best frame selection
│   │   ├── classify_screenshot.py  # Screenshot classification
│   │   ├── sample_frames.py        # Frame sampling from video
│   │   ├── synthesize_section.py   # Section synthesis
│   │   ├── generate_report.py      # Report generation
│   │   ├── orchestrator.py         # Pipeline orchestration
│   │   └── validate_video.py       # Video validation
│   └── shared/
│       ├── database.py             # ~341 lines — SQLite schema + CRUD
│       └── schemas.py              # ~125 lines — Pydantic models
├── extension/
│   ├── manifest.json               # Chrome extension manifest v3
│   ├── icons/                      # 16, 48, 128px PNGs
│   ├── popup/                      # Extension popup
│   └── dashboard/                  # Extension dashboard redirect
├── data/
│   ├── policycapture.db            # SQLite database
│   └── jobs/{job_id}/              # Per-job artifacts
│       ├── input/                  # Source video
│       ├── frames/                 # Extracted frames
│       ├── screenshots/            # Promoted screenshots
│       └── thumbnails/             # Thumbnail images
└── TECHNICAL_BREAKDOWN.md
```

---

## 4. Database Schema

```sql
jobs (id, title, source_video_path, status, duration_ms, frame_count,
      screenshot_count, created_at, updated_at)

frames (id, job_id, frame_index, timestamp_ms, source_image_path,
        blur_score, stability_score, relevance_score,
        matched_keywords TEXT DEFAULT '[]',    -- JSON
        extracted_text TEXT DEFAULT '',
        ocr_confidence REAL DEFAULT 0,
        candidate_score REAL DEFAULT 0)

screenshots (id, job_id, source_frame_id, image_path, thumbnail_path,
             captured_at_ms, section_type, confidence, rationale,
             matched_keywords TEXT DEFAULT '[]',  -- JSON (entities + medicaid_entities + tables + forms)
             extracted_text TEXT DEFAULT '',
             accepted, notes, order_index)

sections (id, job_id, screenshot_id, heading, section_type,
          summary, key_points, confidence, final_order)

reports (id, job_id, html_path, pdf_path, created_at)
```

---

## 5. API Endpoints (Complete)

### Jobs
```
GET    /api/jobs                          # List all jobs
GET    /api/jobs/{id}                     # Job detail
POST   /api/jobs                          # Create job {title, source_video_path?}
DELETE /api/jobs/{id}                     # Delete job + all artifacts
PATCH  /api/jobs/{id}/title              # Rename job {title}
POST   /api/jobs/{id}/auto-title         # Auto-generate title from OCR text
POST   /api/jobs/{id}/upload             # Upload video file (multipart)
POST   /api/jobs/{id}/register-video     # Register local video path
```

### Processing
```
POST   /api/jobs/{id}/extract-frames     # Extract frames from video
POST   /api/jobs/{id}/process            # Full pipeline (extract + process)
POST   /api/jobs/{id}/run-ocr            # OCR + regex NER + Medicaid NER
POST   /api/jobs/{id}/run-medicaid-ner   # Rerun only Medicaid NER (no re-OCR)
POST   /api/jobs/{id}/backfill-confidence # Rerun OCR on frames missing confidence
```

### Frames & Screenshots
```
GET    /api/jobs/{id}/frames             # List frames
GET    /api/jobs/{id}/screenshots        # List screenshots
POST   /api/frames/{id}/promote          # Promote frame → screenshot
PATCH  /api/screenshots/{id}             # Update screenshot (accepted, notes, section_type)
POST   /api/jobs/{id}/select-all         # Promote ALL frames to screenshots
POST   /api/jobs/{id}/unselect-all       # Delete all screenshots for job
```

### NER
```
POST   /api/ner/analyze                  # Analyze text {text} → full entity extraction
POST   /api/ner/normalize                # Normalize term {text} → canonical form
GET    /api/ner/labels                   # List 18 Medicaid entity types
```

### Data & Export
```
GET    /api/jobs/{id}/ocr-data           # All OCR text + entities for job
GET    /api/jobs/{id}/sections           # Extracted sections
POST   /api/jobs/{id}/report             # Generate report
GET    /api/jobs/{id}/report             # Get report metadata
GET    /api/jobs/{id}/report/html        # Get report HTML
GET    /api/jobs/{id}/export-ocr?format=docx|txt  # Export OCR data
GET    /api/artifacts/{id}/{type}/{file}  # Serve frame/screenshot/thumbnail images
```

### Misc
```
POST   /api/demo/seed                    # Seed demo data
```

---

## 6. NER Pipeline Deep Dive

### Regex NER (`extract_entities.py`) — 22 entity types
Structural/format entities extracted via regex patterns:
```
url, email, ssn (masked), ein, npi, medical_code (ICD-10/CPT/HCPCS/DRG/NDC),
claim_number, group_number, phone, currency, percentage, date, time_value,
zip_code, policy_number, case_number, address, account_number, id_number,
person_name (heuristic), organization (heuristic), state
```
Also extracts: key-value pairs (80+ known form fields), bulleted/numbered/lettered lists, section headers.

### Medicaid NER (`medicaid_ner.py`) — 18 entity types, 4 layers

**Entity Types**:
```
PROGRAM_BRAND          — Medi-Cal, MassHealth, SoonerCare, etc.
PROGRAM_TYPE           — Medicaid, CHIP, managed care, FFS, LTSS, HCBS
AGENCY_OR_GOV_BODY     — CMS, MACPAC, state Medicaid agency
ELIGIBILITY_GROUP      — pregnant women, ABD, dual eligible, expansion adult
PERSON_ROLE            — beneficiary, enrollee, caseworker, navigator
FINANCIAL_TERM         — FPL, MAGI, premium, copayment, spenddown, TPL
APPLICATION_PROCESS    — application, renewal, redetermination, appeal, ex parte
BENEFIT_OR_SERVICE     — dental, behavioral health, NEMT, DME, hospice, MAT
CARE_SETTING           — nursing home, assisted living, FQHC, clinic
PROVIDER_TYPE          — hospital, PCP, specialist, pharmacist, FQHC, RHC
PAYMENT_OR_DELIVERY_MODEL — MCO, FFS, capitation, VBC, ACO, D-SNP, MLTSS
WAIVER_OR_AUTHORITY    — 1115 waiver, 1915(c), SPA, ACA, state plan
DOCUMENT_OR_RECORD     — EHR, HIE, HIPAA, fee schedule, handbook
QUALITY_OR_COMPLIANCE  — fraud, grievance, quality measure, program integrity
TECH_OR_SYSTEM         — portal, AVRS, claims system, dashboard
SOCIAL_SUPPORT_OR_COMMUNITY_NEED — housing, food insecurity, SDOH, HRSN
LOCATION               — state, county, service region
ACRONYM                — CHIP, CMS, HCBS, LTSS, MCO, FPL, MAGI, etc.
```

**Layer 1 — Dictionary** (confidence 0.95):
- 500+ seed terms precompiled as regex patterns
- Longest-match-first to avoid partial matches
- Case-sensitive for short acronyms, case-insensitive otherwise

**Layer 2 — spaCy** (confidence 0.70):
- `en_core_web_sm` model
- Maps: ORG→AGENCY, GPE→LOCATION, PERSON→PERSON_ROLE, LAW→WAIVER, MONEY→FINANCIAL
- Truncates text at 100K chars for performance

**Layer 3 — GLiNER** (confidence = model score):
- `urchade/gliner_medium-v2.1` (zero-shot transformer)
- 16 human-readable label prompts mapped to our 18 types
- Chunks text at 1500 chars with 200 overlap for long documents
- Catches plurals, novel phrases, multi-word variants the dictionary misses
- Lazy-loaded (first call downloads ~200MB model from HuggingFace)

**Layer 4 — Normalization**:
- Maps surface forms to canonical labels
- "fee for service" / "FFS" / "fee-for-service" → canonical FFS
- "renewal" / "redetermination" / "recertification" → ELIGIBILITY_REVIEW
- "1115" / "Section 1115" / "1115 demonstration" → WAIVER_1115
- "FQHC" / "Federally Qualified Health Center" → FQHC

**Overlap Resolution**: dictionary > spaCy > GLiNER. Longer spans win ties.

**Output per entity**:
```json
{
  "text": "Home and Community-Based Services",
  "label": "PROGRAM_TYPE",
  "canonical_name": "HCBS",
  "confidence": 0.95,
  "source": "dictionary",
  "normalized_acronym": "HCBS",
  "start": 10,
  "end": 43
}
```

---

## 7. Scene Change Detection (`scene_change.py`)

Two-pass for speed:
1. **Fast pass** — DCT perceptual hash (256-bit), <1ms per frame
   - Hash sim > 0.92 → same frame, skip
   - Hash sim < 0.70 → obvious change, keep
   - In between → send to pass 2
2. **Precise pass** — SSIM (65%) + color histogram (35%), ~50ms per frame
   - Only runs on 20-40% of frames
   - Adaptive threshold adjusts based on content variance

Compares to **last kept frame** (not previous frame) — gradual scrolls still trigger capture.

Output per frame: `scene_change_score`, `is_scene_change`, `text_density`, `visual_importance`

Visual importance = 40% scene change + 20% text density + 25% relevance + 15% blur score

---

## 8. OCR Pipeline (`detect_elements.py`)

- Tesseract with multi-strategy preprocessing
- Strategies: CLAHE, Otsu threshold, adaptive threshold, deskew, denoise
- Picks best result by confidence score
- Batch parallel processing: `detect_elements_batch(paths, max_workers=4)`
- Returns: extracted_text, ocr_confidence (0-100), elements, tables, checkboxes, form_fields, structured_data

---

## 9. Frontend Architecture

**SPA Router** (`app.js`):
- Hash-based: `#/` → job list, `#/jobs/{id}` → detail, `#/jobs/{id}/report` → report
- `route()` function dispatches to `loadJobs()`, `loadJobDetail()`, `loadReport()`

**Standalone Pages** (server-rendered, own CSS):
- `/recorder` → screen recording
- `/jobs/{id}/frames` → frame review (dark theme)
- `/jobs/{id}/ocr` → OCR review (dark theme)
- `/docs` → documentation

**Key UI Patterns**:
- Always-visible "Rename" button (not hover-reveal)
- Native `<input type="range">` for sliders (custom ones broke in WebKit)
- Raw `fetch()` for bulk operations (select-all, unselect-all)
- Toast notifications for feedback
- Inline editing with Enter/Escape/blur save

---

## 10. Dependencies

```
# Core
fastapi, uvicorn, pydantic

# CV & OCR
opencv-python-headless, pytesseract, numpy<2

# NER
spacy (en_core_web_sm), gliner, torch, transformers, safetensors, sentencepiece

# Export
python-docx

# GLiNER model (auto-downloaded)
urchade/gliner_medium-v2.1
```

---

## 11. Known Issues & Tech Debt

1. **Browser caching** — #1 cause of "it doesn't work". Always Cmd+Shift+R after JS/CSS changes.
2. **Disk space** — torch + GLiNER model eat ~500MB. Dev machine had only 4.6GB free.
3. **numpy pinned to <2** — pandas and sklearn break with numpy 2.x due to C ABI mismatch.
4. **Pyright false positives** — `gliner` and `docx` imports show errors but work at runtime.
5. **Auto-title quality** — depends on OCR quality of first frames. Garbage OCR = garbage title.
6. **No auth** — localhost only, no users, no sessions.
7. **No tests** — zero unit/integration tests.
8. **routes.py is 1500 lines** — should be split into router modules.
9. **GLiNER cold start** — first NER call takes ~5s to load the model into memory.

---

## 12. What to Build Next (V2 Ideas)

- [ ] Split routes.py into modular routers (jobs, frames, ner, export)
- [ ] Add unit tests for NER pipeline
- [ ] PaddleOCR evaluation (potentially faster + better than Tesseract)
- [ ] Batch NER across all frames in a job (not just screenshots)
- [ ] Entity relationship extraction (which entities appear together)
- [ ] Policy comparison mode (diff two recordings)
- [ ] User auth if moving beyond localhost
- [ ] Cache-busting for static assets (hash in filename or query param)
- [ ] WebSocket for real-time processing progress
- [ ] GLiNER fine-tuning on Medicaid-specific training data
