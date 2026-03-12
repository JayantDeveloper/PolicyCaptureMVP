"""
Section synthesis module for PolicyCapture Local.

Generates structured section content from classified screenshot data.
"""

# TODO: Replace with LLM-based synthesis (e.g., Claude API)

import logging
from typing import List

logger = logging.getLogger(__name__)

_SECTION_HEADINGS = {
    "demographics": "Applicant Demographics",
    "income": "Income & Employment Information",
    "household": "Household Composition",
    "eligibility": "Eligibility Determination",
    "policy_guidance": "Policy Guidance & Regulations",
    "application_step": "Application Process Step",
    "table": "Data Table / Summary",
    "unknown": "Uncategorized Content",
}


def synthesize_section(screenshot_data: dict) -> dict:
    """
    Generate structured section content from a classified screenshot.

    Expected keys in screenshot_data:
        section_type (str): Classification label.
        extracted_text (str): OCR or placeholder text.
        matched_keywords (list[str]): Keywords found in the frame.
        confidence (float): Classification confidence.
        timestamp_ms (int): Source video timestamp.

    Args:
        screenshot_data: Metadata dict for a single classified screenshot.

    Returns:
        dict with keys:
            heading (str): Human-readable section heading.
            summary (str): Brief narrative summary.
            key_points (list[str]): Bullet-point key observations.
            confidence (float): Inherited classification confidence.
            order_suggestion (int): Suggested ordering weight.
    """
    section_type = screenshot_data.get("section_type", "unknown")
    extracted_text = screenshot_data.get("extracted_text", "")
    matched_keywords: List[str] = screenshot_data.get("matched_keywords", [])
    confidence = screenshot_data.get("confidence", 0.0)
    timestamp_ms = screenshot_data.get("timestamp_ms", 0)

    # Heading
    heading = _SECTION_HEADINGS.get(section_type, _SECTION_HEADINGS["unknown"])

    # Summary
    if extracted_text.strip():
        # Truncate long text for summary
        snippet = extracted_text.strip()[:300]
        summary = (
            f"This section (captured at {timestamp_ms / 1000:.1f}s) contains "
            f"content related to {heading.lower()}. Extracted text begins: "
            f'"{snippet}..."'
        )
    else:
        summary = (
            f"This section (captured at {timestamp_ms / 1000:.1f}s) appears to "
            f"contain content related to {heading.lower()}. "
            f"Full text extraction pending OCR integration."
        )

    # Key points from matched keywords
    key_points: List[str] = []
    if matched_keywords:
        key_points.append(
            f"Matched {len(matched_keywords)} relevant keyword(s): "
            + ", ".join(matched_keywords)
        )
    if section_type == "table":
        key_points.append("Visual structure suggests tabular or form data.")
    if confidence >= 0.5:
        key_points.append(f"Classification confidence: {confidence:.0%}.")
    else:
        key_points.append(
            f"Low classification confidence ({confidence:.0%}); manual review recommended."
        )
    if not extracted_text.strip():
        key_points.append("OCR text not yet available — synthesis is preliminary.")

    # Order suggestion based on typical policy document flow
    order_map = {
        "demographics": 10,
        "income": 20,
        "household": 30,
        "eligibility": 40,
        "policy_guidance": 50,
        "application_step": 60,
        "table": 70,
        "unknown": 99,
    }
    order_suggestion = order_map.get(section_type, 99)

    result = {
        "heading": heading,
        "summary": summary,
        "key_points": key_points,
        "confidence": confidence,
        "order_suggestion": order_suggestion,
    }

    logger.debug(
        "Synthesized section '%s' (confidence=%.4f, %d key points)",
        heading,
        confidence,
        len(key_points),
    )
    return result
