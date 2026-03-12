"""Pydantic schemas for PolicyCapture Local."""
from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionType(str, enum.Enum):
    DEMOGRAPHICS = "demographics"
    INCOME = "income"
    HOUSEHOLD = "household"
    ELIGIBILITY = "eligibility"
    POLICY_GUIDANCE = "policy_guidance"
    APPLICATION_STEP = "application_step"
    TABLE = "table"
    UNKNOWN = "unknown"


# --- Core data models ---

class RecordingJob(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    title: str
    source_video_path: Optional[str] = None
    status: JobStatus = JobStatus.PENDING
    duration_ms: Optional[int] = None
    frame_count: Optional[int] = None
    screenshot_count: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""


class FrameMetadata(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    job_id: str
    frame_index: int
    timestamp_ms: int
    source_image_path: str
    blur_score: float = 0.0
    stability_score: float = 0.0
    relevance_score: float = 0.0
    matched_keywords: list[str] = Field(default_factory=list)
    extracted_text: str = ""
    ocr_confidence: float = 0.0
    candidate_score: float = 0.0


class ScreenshotCandidate(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    job_id: str
    source_frame_id: str
    image_path: str
    thumbnail_path: str = ""
    captured_at_ms: int = 0
    section_type: SectionType = SectionType.UNKNOWN
    confidence: float = 0.0
    rationale: str = ""
    matched_keywords: list[str] = Field(default_factory=list)
    extracted_text: str = ""
    accepted: bool = True
    notes: str = ""
    order_index: int = 0


class ExtractedSection(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    job_id: str
    screenshot_id: str
    heading: str = ""
    section_type: SectionType = SectionType.UNKNOWN
    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    final_order: int = 0


class FinalReport(BaseModel):
    model_config = {"from_attributes": True}

    id: str
    job_id: str
    html_path: str = ""
    pdf_path: str = ""
    created_at: str = ""


# --- Request/Response models ---

class CreateJobRequest(BaseModel):
    title: str
    source_video_path: Optional[str] = None


class RegisterVideoRequest(BaseModel):
    source_video_path: str


class UpdateScreenshotRequest(BaseModel):
    accepted: Optional[bool] = None
    notes: Optional[str] = None
    section_type: Optional[SectionType] = None
    order_index: Optional[int] = None
