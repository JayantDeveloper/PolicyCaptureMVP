"""FastAPI application for PolicyCapture Local."""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
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
    """Initialize database on startup."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the review UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/recorder", response_class=HTMLResponse)
async def recorder(request: Request):
    """Serve the screen recorder page."""
    return templates.TemplateResponse("recorder.html", {"request": request})


@app.get("/jobs/{job_id}/frames", response_class=HTMLResponse)
async def frame_review(request: Request, job_id: str):
    """Serve the frame review/selection page."""
    return templates.TemplateResponse("frame_review.html", {"request": request, "job_id": job_id})
