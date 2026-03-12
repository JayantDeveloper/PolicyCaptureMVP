"""Pipeline orchestrator for PolicyCapture Local."""

import json
import logging
import os
import time

import cv2

from packages.core.pipeline.validate_video import validate_video
from packages.core.pipeline.sample_frames import sample_frames
from packages.core.pipeline.preprocess_frame import compute_blur_score, compute_stability_score, preprocess_frame
from packages.core.pipeline.detect_relevance import detect_relevance
from packages.core.pipeline.scene_change import detect_scene_changes
from packages.core.pipeline.classify_screenshot import classify_screenshot
from packages.core.pipeline.synthesize_section import synthesize_section
from packages.core.pipeline.generate_report import generate_html_report, generate_pdf_report

from packages.shared.database import (
    update_job_status, create_frame, create_screenshot, create_section, create_report,
)
from packages.shared.utils import generate_id, ensure_dir, get_job_subdir
from packages.shared.config import FRAME_SAMPLE_INTERVAL_SEC

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full PolicyCapture pipeline for a single job."""

    def __init__(self):
        self._stage = "init"

    def _set_stage(self, job_id: str, stage: str, detail: str = ""):
        self._stage = stage
        logger.info("[%s] Stage: %s %s", job_id, stage, detail)
        try:
            update_job_status(job_id, "processing")
        except Exception:
            pass

    def run_pipeline(self, job_id: str, video_path: str) -> dict:
        """Run the full extraction pipeline."""
        start = time.time()
        result = {
            "job_id": job_id,
            "status": "running",
            "stages_completed": [],
            "frame_count": 0,
            "selected_count": 0,
            "sections": [],
            "report_path": "",
            "duration_sec": 0.0,
            "error": None,
        }

        try:
            frames_dir = str(ensure_dir(get_job_subdir(job_id, "frames")))
            screenshots_dir = str(ensure_dir(get_job_subdir(job_id, "screenshots")))
            thumbnails_dir = str(ensure_dir(get_job_subdir(job_id, "thumbnails")))
            reports_dir = str(ensure_dir(get_job_subdir(job_id, "reports")))

            # Also create processed_frames dir for thumbnail versions of all frames
            processed_frames_dir = str(ensure_dir(get_job_subdir(job_id, "processed_frames")))

            # 1. Validate
            self._set_stage(job_id, "validating")
            validation = validate_video(video_path)
            if not validation["valid"]:
                raise ValueError(f"Video validation failed: {validation.get('error')}")
            result["stages_completed"].append("validate")

            # 2. Sample frames
            self._set_stage(job_id, "sampling")
            sampled = sample_frames(video_path, frames_dir, interval_sec=FRAME_SAMPLE_INTERVAL_SEC)
            result["frame_count"] = len(sampled)
            if not sampled:
                raise ValueError("No frames could be sampled from the video")
            result["stages_completed"].append("sample")

            # 3. Preprocess
            self._set_stage(job_id, "preprocessing", f"{len(sampled)} frames")
            prev_img = None
            for frame in sampled:
                pp = preprocess_frame(frame["image_path"])
                frame.update(pp)
                cur_img = cv2.imread(frame["image_path"])
                if prev_img is not None and cur_img is not None:
                    frame["stability_score"] = compute_stability_score(prev_img, cur_img)
                else:
                    frame["stability_score"] = 0.5
                prev_img = cur_img
            result["stages_completed"].append("preprocess")

            # 4. Detect relevance
            self._set_stage(job_id, "detecting_relevance")
            for frame in sampled:
                rel = detect_relevance(frame["image_path"])
                frame.update(rel)
            result["stages_completed"].append("detect_relevance")

            # 5. Scene change detection
            self._set_stage(job_id, "detecting_scene_changes")
            sampled = detect_scene_changes(sampled)
            result["stages_completed"].append("scene_change")

            # Generate thumbnails for ALL frames (for the frame browser UI)
            self._set_stage(job_id, "generating_thumbnails")
            for frame in sampled:
                img = cv2.imread(frame["image_path"])
                if img is not None:
                    thumb_name = f"frame_thumb_{frame['frame_index']:06d}.jpg"
                    thumb_path = os.path.join(processed_frames_dir, thumb_name)
                    cv2.imwrite(thumb_path, cv2.resize(img, (320, 180)),
                                [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame["thumbnail_path"] = thumb_path

            # Persist frames to DB (now includes scene change + visual importance scores)
            for frame in sampled:
                create_frame(
                    frame_id=generate_id(),
                    job_id=job_id,
                    frame_index=frame["frame_index"],
                    timestamp_ms=frame["timestamp_ms"],
                    source_image_path=frame["image_path"],
                    blur_score=frame.get("blur_score", 0),
                    stability_score=frame.get("stability_score", 0),
                    relevance_score=frame.get("relevance_score", 0),
                    matched_keywords=frame.get("matched_keywords", []),
                    extracted_text=frame.get("extracted_text", ""),
                    ocr_confidence=frame.get("ocr_confidence", 0),
                    candidate_score=frame.get("visual_importance", frame.get("candidate_score", 0)),
                )

            update_job_status(job_id, "processing", frame_count=len(sampled))

            # 6. Auto-select: every frame where the screen changed
            self._set_stage(job_id, "auto_selecting")
            unique = [f for f in sampled if f.get("is_scene_change")]
            unique.sort(key=lambda f: f["timestamp_ms"])
            result["selected_count"] = len(unique)
            result["stages_completed"].append("auto_select")
            logger.info("[%s] Auto-selected %d scene-change frames out of %d total",
                        job_id, len(unique), len(sampled))

            # 8. Classify
            self._set_stage(job_id, "classifying")
            for frame in unique:
                cls = classify_screenshot(
                    extracted_text=frame.get("extracted_text", ""),
                    matched_keywords=frame.get("matched_keywords", []),
                    structure_score=frame.get("structure_score", 0.0),
                )
                frame.update(cls)
            result["stages_completed"].append("classify")

            # 9. Synthesize
            self._set_stage(job_id, "synthesizing")
            sections = []
            for frame in unique:
                syn = synthesize_section(frame)
                sections.append(syn)
            result["sections"] = sections
            result["stages_completed"].append("synthesize")

            # Copy selected frames to screenshots dir and create thumbnails
            for i, frame in enumerate(unique):
                src = frame["image_path"]
                dst = os.path.join(screenshots_dir, f"screenshot_{i:03d}.png")
                thumb = os.path.join(thumbnails_dir, f"thumb_{i:03d}.png")

                img = cv2.imread(src)
                if img is not None:
                    cv2.imwrite(dst, img)
                    cv2.imwrite(thumb, cv2.resize(img, (320, 180)))
                frame["screenshot_path"] = dst
                frame["thumbnail_path"] = thumb

                # Persist screenshot
                screenshot_id = generate_id()
                frame["screenshot_id"] = screenshot_id
                create_screenshot(
                    screenshot_id=screenshot_id,
                    job_id=job_id,
                    source_frame_id=f"frame_{frame['frame_index']:06d}",
                    image_path=dst,
                    thumbnail_path=thumb,
                    captured_at_ms=frame.get("timestamp_ms", 0),
                    section_type=frame.get("section_type", "unknown"),
                    confidence=frame.get("confidence", 0),
                    rationale=frame.get("rationale", ""),
                    matched_keywords=json.dumps(frame.get("matched_keywords", [])),
                    extracted_text=frame.get("extracted_text", ""),
                )

                # Persist section
                section = sections[i] if i < len(sections) else {}
                create_section(
                    section_id=generate_id(),
                    job_id=job_id,
                    screenshot_id=screenshot_id,
                    heading=section.get("heading", ""),
                    section_type=frame.get("section_type", "unknown"),
                    summary=section.get("summary", ""),
                    key_points=json.dumps(section.get("key_points", [])),
                    confidence=section.get("confidence", 0),
                    final_order=i,
                )

            update_job_status(job_id, "processing", screenshot_count=len(unique))

            # 10. Generate report
            self._set_stage(job_id, "generating_report")
            html_path = os.path.join(reports_dir, f"report_{job_id}.html")
            pdf_path = os.path.join(reports_dir, f"report_{job_id}.pdf")
            job_meta = {"job_id": job_id, "video_path": video_path, "status": "completed"}
            generate_html_report(job_meta, sections, unique, html_path)
            generate_pdf_report(html_path, pdf_path)
            result["report_path"] = html_path
            result["stages_completed"].append("generate_report")

            # Persist report
            create_report(
                report_id=generate_id(),
                job_id=job_id,
                html_path=html_path,
                pdf_path=pdf_path,
            )

            result["status"] = "completed"
            update_job_status(job_id, "completed", screenshot_count=len(unique))

        except Exception as exc:
            result["status"] = "failed"
            result["error"] = str(exc)
            logger.exception("[%s] Pipeline failed at stage '%s': %s", job_id, self._stage, exc)
            try:
                update_job_status(job_id, "failed")
            except Exception:
                pass

        result["duration_sec"] = round(time.time() - start, 2)
        return result
