#!/usr/bin/env python3
"""Seed demo data for PolicyCapture Local.

Creates a demo job with synthetic screenshots to test the UI
without requiring a real video file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Install with: pip install opencv-python-headless")
    sys.exit(1)

from packages.shared.database import init_db, create_job, create_screenshot, create_section
from packages.shared.utils import generate_id, get_job_subdir, ensure_dir
from packages.shared.schemas import SectionType


DEMO_SECTIONS = [
    {
        "section_type": SectionType.DEMOGRAPHICS,
        "heading": "Applicant Demographics",
        "summary": "Personal information section showing applicant details including name, date of birth, and contact information.",
        "key_points": ["Name: Jane Doe", "DOB: 1985-03-15", "Address: 123 Main St, Springfield, IL"],
        "keywords": ["name", "date of birth", "address", "phone", "email"],
        "color": (200, 220, 240),
        "text_lines": [
            "APPLICANT DEMOGRAPHICS",
            "",
            "Name: Jane Doe",
            "Date of Birth: 03/15/1985",
            "Address: 123 Main St",
            "City: Springfield, IL 62701",
            "Phone: (555) 123-4567",
            "Email: jane.doe@email.com",
        ],
    },
    {
        "section_type": SectionType.INCOME,
        "heading": "Income Verification",
        "summary": "Income details showing monthly earnings, employment status, and verification documents.",
        "key_points": ["Monthly Income: $2,450", "Employer: ABC Corp", "Employment Type: Full-time"],
        "keywords": ["income", "wages", "salary", "employment"],
        "color": (220, 240, 200),
        "text_lines": [
            "INCOME VERIFICATION",
            "",
            "Employment Status: Full-time",
            "Employer: ABC Corporation",
            "Monthly Gross Income: $2,450.00",
            "Annual Income: $29,400.00",
            "Pay Frequency: Bi-weekly",
            "Verification: Pay stub attached",
        ],
    },
    {
        "section_type": SectionType.HOUSEHOLD,
        "heading": "Household Composition",
        "summary": "Household members listing showing family size and dependent information.",
        "key_points": ["Household Size: 3", "Dependents: 1 child", "Spouse: John Doe"],
        "keywords": ["household", "members", "dependents", "family"],
        "color": (240, 220, 200),
        "text_lines": [
            "HOUSEHOLD COMPOSITION",
            "",
            "Total Members: 3",
            "",
            "1. Jane Doe (Applicant) - Age 40",
            "2. John Doe (Spouse) - Age 42",
            "3. Emily Doe (Child) - Age 8",
            "",
            "Dependents: 1",
        ],
    },
    {
        "section_type": SectionType.ELIGIBILITY,
        "heading": "Eligibility Determination",
        "summary": "Eligibility screening results showing qualification status for benefits programs.",
        "key_points": ["Status: Potentially Eligible", "Program: Medicaid", "FPL: 138%"],
        "keywords": ["eligible", "eligibility", "qualify", "determination"],
        "color": (200, 240, 220),
        "text_lines": [
            "ELIGIBILITY DETERMINATION",
            "",
            "Program: Medicaid",
            "Federal Poverty Level: 138%",
            "Household Income: $29,400/year",
            "FPL Threshold: $25,820 (3-person)",
            "",
            "Status: POTENTIALLY ELIGIBLE",
            "Next Step: Complete application",
        ],
    },
    {
        "section_type": SectionType.TABLE,
        "heading": "Benefits Summary Table",
        "summary": "Summary table showing available benefits programs and coverage amounts.",
        "key_points": ["Medicaid: Eligible", "SNAP: Review Required", "CHIP: Eligible"],
        "keywords": ["table", "amount", "total", "benefits"],
        "color": (230, 230, 240),
        "text_lines": [
            "BENEFITS SUMMARY",
            "",
            "Program        | Status    | Amount",
            "---------------|-----------|--------",
            "Medicaid       | Eligible  | Full",
            "SNAP           | Review    | $234/mo",
            "CHIP           | Eligible  | Full",
            "TANF           | Ineligible| N/A",
            "WIC            | Eligible  | Standard",
        ],
    },
    {
        "section_type": SectionType.POLICY_GUIDANCE,
        "heading": "Policy Guidance Notes",
        "summary": "Relevant policy guidance for the eligibility determination and application process.",
        "key_points": [
            "42 CFR 435.603 - MAGI methodology",
            "Income disregard: 5% FPL",
            "Retroactive coverage: 3 months",
        ],
        "keywords": ["policy", "regulation", "guidance", "requirement"],
        "color": (240, 230, 220),
        "text_lines": [
            "POLICY GUIDANCE",
            "",
            "Reference: 42 CFR 435.603",
            "MAGI-Based Income Methodology",
            "",
            "Key Rules:",
            "- 5% FPL income disregard applies",
            "- Retroactive coverage up to 3 months",
            "- Presumptive eligibility available",
            "- Annual redetermination required",
        ],
    },
]


def create_synthetic_screenshot(text_lines: list[str], bg_color: tuple, output_path: str, size=(1280, 720)):
    """Create a synthetic screenshot with text content."""
    img = np.full((size[1], size[0], 3), bg_color, dtype=np.uint8)

    # Add a header bar
    cv2.rectangle(img, (0, 0), (size[0], 60), (50, 70, 90), -1)
    cv2.putText(img, "PolicyCapture Local - Demo Screenshot", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add content area border
    cv2.rectangle(img, (40, 80), (size[0] - 40, size[1] - 40), (180, 180, 180), 2)

    # Add text lines
    y = 130
    for line in text_lines:
        if not line:
            y += 20
            continue
        font_scale = 0.7 if not line.isupper() else 0.85
        thickness = 1 if not line.isupper() else 2
        color = (30, 30, 30) if not line.isupper() else (20, 50, 100)
        cv2.putText(img, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        y += 35

    # Add timestamp watermark
    cv2.putText(img, "DEMO DATA", (size[0] - 200, size[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    cv2.imwrite(output_path, img)
    return output_path


def create_thumbnail(image_path: str, thumb_path: str, size=(320, 180)):
    """Create a thumbnail from an image."""
    img = cv2.imread(image_path)
    if img is not None:
        thumb = cv2.resize(img, size)
        cv2.imwrite(thumb_path, thumb)
    return thumb_path


def seed():
    """Seed the database with demo data."""
    init_db()

    job_id = generate_id()
    screenshots_dir = get_job_subdir(job_id, "screenshots")
    thumbnails_dir = get_job_subdir(job_id, "thumbnails")
    ensure_dir(screenshots_dir)
    ensure_dir(thumbnails_dir)

    # Create job
    create_job(
        job_id=job_id,
        title="Demo: Benefits Eligibility Review",
        source_video_path="(demo - synthetic data)",
    )

    # Update job status to completed
    from packages.shared.database import update_job_status
    update_job_status(job_id, "completed", screenshot_count=len(DEMO_SECTIONS))

    print(f"Created demo job: {job_id}")

    # Create screenshots and sections
    for i, section_data in enumerate(DEMO_SECTIONS):
        screenshot_id = generate_id()
        section_id = generate_id()
        timestamp_ms = (i + 1) * 5000  # Every 5 seconds

        # Create synthetic image
        image_path = str(screenshots_dir / f"screenshot_{i:03d}.png")
        thumb_path = str(thumbnails_dir / f"thumb_{i:03d}.png")

        create_synthetic_screenshot(section_data["text_lines"], section_data["color"], image_path)
        create_thumbnail(image_path, thumb_path)

        # Create screenshot record
        create_screenshot(
            screenshot_id=screenshot_id,
            job_id=job_id,
            source_frame_id=f"frame_{i:06d}",
            image_path=image_path,
            thumbnail_path=thumb_path,
            captured_at_ms=timestamp_ms,
            section_type=section_data["section_type"].value,
            confidence=0.85 + (i * 0.02),
            rationale=f"Matched keywords: {', '.join(section_data['keywords'][:3])}",
            matched_keywords=json.dumps(section_data["keywords"]),
            extracted_text=" ".join(section_data["text_lines"]),
        )

        # Create section record
        create_section(
            section_id=section_id,
            job_id=job_id,
            screenshot_id=screenshot_id,
            heading=section_data["heading"],
            section_type=section_data["section_type"].value,
            summary=section_data["summary"],
            key_points=json.dumps(section_data["key_points"]),
            confidence=0.85 + (i * 0.02),
            final_order=i,
        )

        print(f"  Created screenshot: {section_data['heading']}")

    print(f"\nDemo data seeded successfully!")
    print(f"Job ID: {job_id}")
    print(f"Start the server and visit http://localhost:8420")


if __name__ == "__main__":
    seed()
