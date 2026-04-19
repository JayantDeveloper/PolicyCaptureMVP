"""Classification API routes — SVM/LR document classification via OCR pipeline."""
import json
import logging
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse

from packages.shared.database import (
    create_classification_result,
    get_classification_results,
    get_screenshots_for_job,
)
from packages.shared.config import ML_MODELS_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/classify", tags=["classify"])

_ALLOWED_TYPES = {".png", ".jpg", ".jpeg", ".pdf", ".tif", ".tiff", ".bmp"}
_MAX_UPLOAD_MB = 50


@router.post("/upload")
async def classify_upload(
    file: UploadFile = File(...),
    ocr_mode: str = Form("fast"),
):
    """Accept an image or PDF, run OCR + SVM/LR classification, return results."""
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in _ALLOWED_TYPES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(_ALLOWED_TYPES)}")

    image_bytes = await file.read()
    if len(image_bytes) > _MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (max {_MAX_UPLOAD_MB} MB)")
    if not image_bytes:
        raise HTTPException(400, "Empty file")

    try:
        from packages.core.pipeline.classifier import run_ocr
        result = run_ocr(image_bytes, filename=file.filename, mode=ocr_mode)
    except Exception as exc:
        logger.exception("Classification failed for %s", file.filename)
        raise HTTPException(500, f"Classification error: {exc}") from exc

    stored = create_classification_result(
        job_id="manual",
        filename=file.filename or "upload",
        svm_prediction=result["svm_prediction"],
        lr_prediction=result["lr_prediction"],
        features=result.get("features"),
        tfidf_keywords=result.get("tfidf", {}).get("keywords"),
    )

    svm = result["svm_prediction"]
    lr  = result["lr_prediction"]
    consensus = svm["label"] if svm["label"] not in ("", "unavailable", "Unknown") else lr["label"]

    return {
        **result,
        "consensus_label": consensus,
        "stored_id": stored["id"],
    }


@router.post("/job/{job_id}")
async def classify_job_screenshots(job_id: str):
    """Batch-classify all screenshots for an existing job and store results."""
    screenshots = get_screenshots_for_job(job_id)
    if not screenshots:
        raise HTTPException(404, f"No screenshots found for job '{job_id}'")

    try:
        from packages.core.pipeline.classifier import run_ocr
    except ImportError as exc:
        raise HTTPException(500, "Classifier not available") from exc

    classified = []
    errors = []
    for ss in screenshots:
        img_path = ss.get("image_path", "")
        if not img_path or not Path(img_path).exists():
            errors.append({"frame_id": ss["id"], "error": "image not found"})
            continue
        try:
            with open(img_path, "rb") as f:
                clf = run_ocr(f.read(), filename=Path(img_path).name, mode="fast")
            stored = create_classification_result(
                job_id=job_id,
                frame_id=ss["id"],
                filename=Path(img_path).name,
                svm_prediction=clf["svm_prediction"],
                lr_prediction=clf["lr_prediction"],
                features=clf.get("features"),
                tfidf_keywords=clf.get("tfidf", {}).get("keywords"),
            )
            classified.append({
                "frame_id": ss["id"],
                "label": clf["svm_prediction"]["label"],
                "confidence": clf["svm_prediction"]["confidence"],
                "stored_id": stored["id"],
            })
        except Exception as exc:
            logger.warning("Classification failed for screenshot %s: %s", ss["id"], exc)
            errors.append({"frame_id": ss["id"], "error": str(exc)})

    return {
        "job_id": job_id,
        "frames_classified": len(classified),
        "errors": len(errors),
        "results": classified,
    }


@router.get("/results")
async def get_results(
    job_id: str | None = Query(default=None),
    frame_id: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
):
    """Query stored classification results by job_id and/or frame_id."""
    rows = get_classification_results(job_id=job_id, frame_id=frame_id, limit=limit)
    return {"results": rows, "count": len(rows)}


@router.post("/retrain")
async def retrain(version: str = "v1", x_admin_key: str | None = None):
    """Dev-only: retrain ML models by running the specified training script."""
    expected_key = "bah-admin"
    if x_admin_key != expected_key:
        raise HTTPException(403, "Missing or invalid X-Admin-Key header")

    script = PROJECT_ROOT / "ml" / "training" / f"MLTrainingv{version}.py"
    if not script.exists():
        raise HTTPException(404, f"Training script not found: {script.name}")

    try:
        subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT / "ml" / "training"),
        )
    except Exception as exc:
        raise HTTPException(500, f"Failed to launch training: {exc}") from exc

    return JSONResponse(
        status_code=202,
        content={"status": "retraining started", "version": version, "script": script.name},
    )
