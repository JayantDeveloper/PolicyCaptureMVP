"""
Relevance detection module for PolicyCapture Local.

Determines how relevant a frame is to policy-capture documentation using
keyword matching and visual structure analysis.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default relevance keywords — importable and overridable via config.
RELEVANCE_KEYWORDS: List[str] = [
    "policy",
    "eligibility",
    "income",
    "household",
    "application",
    "determination",
    "benefit",
    "enrollment",
    "demographics",
    "address",
    "applicant",
    "date of birth",
    "ssn",
    "submit",
    "upload",
    "form",
    "table",
    "total",
    "guidance",
    "regulation",
    "requirement",
]


def mock_ocr_extract(image_path: str) -> str:
    """
    Placeholder OCR extraction function.

    Args:
        image_path: Path to the image file.

    Returns:
        Extracted text (currently returns empty string).
    """
    # TODO: Replace with Tesseract/PaddleOCR integration
    logger.debug("mock_ocr_extract called for %s — returning empty string", image_path)
    return ""


def detect_relevance_by_keywords(
    text: str,
    keywords: List[str],
) -> Tuple[float, List[str]]:
    """
    Score text relevance based on keyword matches.

    Args:
        text: The text to search (e.g., OCR output).
        keywords: List of keywords to look for.

    Returns:
        Tuple of (score in [0, 1], list of matched keywords).
    """
    if not text or not keywords:
        return 0.0, []

    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]

    # Score is the fraction of keywords matched, capped at 1.0
    score = min(len(matched) / max(len(keywords) * 0.3, 1), 1.0)
    return round(score, 4), matched


def detect_visual_structure(image: np.ndarray) -> float:
    """
    Estimate whether an image contains table or form structure using edge
    and line detection.

    Args:
        image: BGR image as a NumPy array.

    Returns:
        Float in [0, 1] indicating likelihood of structured content.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect horizontal and vertical lines with Hough transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )

    if lines is None:
        return 0.0

    horizontal_count = 0
    vertical_count = 0
    angle_tolerance = 10  # degrees

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))

        if angle < angle_tolerance or angle > (180 - angle_tolerance):
            horizontal_count += 1
        elif abs(angle - 90) < angle_tolerance:
            vertical_count += 1

    # A table/form typically has multiple horizontal AND vertical lines
    h_score = min(horizontal_count / 10.0, 1.0)
    v_score = min(vertical_count / 5.0, 1.0)
    structure_score = round((h_score * 0.6 + v_score * 0.4), 4)

    logger.debug(
        "Visual structure: h_lines=%d, v_lines=%d, score=%.4f",
        horizontal_count,
        vertical_count,
        structure_score,
    )
    return structure_score


def detect_relevance(
    image_path: str,
    extracted_text: str = "",
) -> dict:
    """
    Determine the relevance of a frame image to policy documentation.

    Combines keyword-based text scoring with visual structure detection.

    Args:
        image_path: Path to the frame image.
        extracted_text: Pre-extracted text (if available). If empty, mock OCR
                        is attempted.

    Returns:
        dict with keys:
            relevance_score (float): Combined relevance in [0, 1].
            matched_keywords (list[str]): Keywords found in text.
            has_structure (bool): Whether visual structure was detected.
            structure_score (float): Visual structure score in [0, 1].
            extracted_text (str): The text that was analyzed.
            ocr_confidence (float): Placeholder OCR confidence.
    """
    # Attempt OCR if no text provided
    if not extracted_text:
        extracted_text = mock_ocr_extract(image_path)

    # Keyword analysis
    keyword_score, matched_keywords = detect_relevance_by_keywords(
        extracted_text, RELEVANCE_KEYWORDS
    )

    # Visual structure analysis
    image = cv2.imread(image_path)
    structure_score = 0.0
    if image is not None:
        structure_score = detect_visual_structure(image)
    else:
        logger.warning("Could not read image for structure detection: %s", image_path)

    has_structure = structure_score >= 0.3

    # Combined score: weight text higher when available, else lean on structure
    if extracted_text.strip():
        relevance_score = round(keyword_score * 0.7 + structure_score * 0.3, 4)
    else:
        relevance_score = round(structure_score, 4)

    result = {
        "relevance_score": relevance_score,
        "matched_keywords": matched_keywords,
        "has_structure": has_structure,
        "structure_score": structure_score,
        "extracted_text": extracted_text,
        "ocr_confidence": 0.0,  # placeholder until real OCR is integrated
    }

    logger.debug(
        "Relevance for %s — score=%.4f, keywords=%d, structure=%.4f",
        image_path,
        relevance_score,
        len(matched_keywords),
        structure_score,
    )
    return result
