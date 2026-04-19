"""FastAPI application for PolicyCapture Local."""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from packages.shared.database import init_db
from packages.shared.config import DATA_DIR, PROJECT_ROOT

from apps.local_api.routes import router

app = FastAPI(
    title="PolicyCapture Local",
    description="Local-first extraction workflow for recorded screen sessions",
    version="0.1.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
    allow_origin_regex=r"(http://(localhost|127\.0\.0\.1)(:\d+)?|chrome-extension://[a-z]+)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
ui_dir = PROJECT_ROOT / "apps" / "review-ui"
app.mount("/static", StaticFiles(directory=str(ui_dir / "static")), name="static")

# Templates
templates = Jinja2Templates(directory=str(ui_dir / "templates"))

# API routes
app.include_router(router, prefix="/api")


@app.on_event("startup")
def startup():
    """Initialize database and eagerly load ML models on startup."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    # Eagerly load classifier models so the first classify request isn't slow.
    # This is a no-op if PC_RUN_CLASSIFICATION is not set, keeping startup fast
    # for deployments that don't use the classifier.
    from packages.shared.config import RUN_CLASSIFICATION
    if RUN_CLASSIFICATION:
        try:
            from packages.core.pipeline.classifier.ocr_service import load_models
            load_models()
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Classifier models failed to load: %s", exc)


# Serve classify React SPA if the build output exists
_classify_dist = PROJECT_ROOT / "classify-ui" / "dist"
if _classify_dist.exists():
    app.mount("/classify", StaticFiles(directory=str(_classify_dist), html=True), name="classify_ui")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(str(ui_dir / "static" / "img" / "logo.svg"), media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the review UI."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/recorder", response_class=HTMLResponse)
async def recorder(request: Request):
    """Serve the screen recorder page."""
    return templates.TemplateResponse(request, "recorder.html")


@app.get("/jobs/{job_id}/frames", response_class=HTMLResponse)
async def frame_review(request: Request, job_id: str):
    """Serve the frame review/selection page."""
    return templates.TemplateResponse(request, "frame_review.html", {"job_id": job_id})


@app.get("/jobs/{job_id}/ocr-review", response_class=HTMLResponse)
async def ocr_review(request: Request, job_id: str):
    """Serve the OCR review & entity extraction page."""
    return templates.TemplateResponse(request, "ocr_review.html", {"job_id": job_id})


@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):
    """Serve the system documentation page."""
    return templates.TemplateResponse(request, "docs.html")
