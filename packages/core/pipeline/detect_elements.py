"""
Website element detection with bounding boxes using OpenCV + Tesseract OCR.

Detects UI elements (text blocks, buttons, input fields, images, navigation bars,
tables) in a screenshot and returns bounding boxes with labels and OCR text.
Uses subprocess to call tesseract directly to avoid pytesseract/pandas dependency issues.
"""

import logging
import os
import subprocess
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _tesseract_available():
    """Check if tesseract binary is installed."""
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


HAS_TESSERACT = _tesseract_available()
if not HAS_TESSERACT:
    logger.warning("tesseract not found in PATH — OCR will be skipped")


def _run_tesseract_tsv(image_path):
    """Run tesseract and parse TSV output for word-level bounding boxes."""
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", "3", "tsv"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return [], ""

        words = []
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return [], ""

        headers = lines[0].split("\t")
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) != len(headers):
                continue
            row = dict(zip(headers, cols))
            try:
                conf = int(row.get("conf", "-1"))
                text = row.get("text", "").strip()
                if conf >= 30 and text:
                    words.append({
                        "x": int(row["left"]),
                        "y": int(row["top"]),
                        "w": int(row["width"]),
                        "h": int(row["height"]),
                        "text": text,
                        "conf": conf,
                    })
            except (ValueError, KeyError):
                continue

        full_text = " ".join(w["text"] for w in words)
        return words, full_text
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Tesseract TSV failed: %s", e)
        return [], ""


def _run_tesseract_text(image_path):
    """Run tesseract to get plain text."""
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", "3"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def _merge_overlapping_boxes(boxes, overlap_thresh=0.3):
    """Merge overlapping bounding boxes using non-maximum suppression."""
    if not boxes:
        return []

    rects = np.array(boxes)
    x1, y1 = rects[:, 0], rects[:, 1]
    x2, y2 = rects[:, 0] + rects[:, 2], rects[:, 1] + rects[:, 3]
    areas = rects[:, 2] * rects[:, 3]

    order = areas.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / np.maximum(areas[order[1:]], 1)

        remaining = np.where(overlap < overlap_thresh)[0]
        order = order[remaining + 1]

    return [boxes[i] for i in keep]


def _classify_element(roi_gray, x, y, w, h, img_h, img_w):
    """Classify a detected region by its visual characteristics."""
    aspect = w / max(h, 1)
    area_ratio = (w * h) / (img_h * img_w)
    position_y = y / img_h

    # Navigation bar: wide, thin, near top
    if aspect > 5 and position_y < 0.15 and area_ratio > 0.02:
        return "navbar"

    # Footer: wide, thin, near bottom
    if aspect > 5 and position_y > 0.85:
        return "footer"

    # Button: small-ish, roughly rectangular
    if 1.5 < aspect < 8 and area_ratio < 0.02 and 20 < h < 80:
        std = np.std(roi_gray)
        if std < 50:
            return "button"

    # Input field: wide, thin, usually with a border
    if aspect > 3 and 20 < h < 60 and area_ratio < 0.03:
        edges = cv2.Canny(roi_gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / max(edges.size, 1)
        if edge_ratio > 0.05:
            return "input_field"

    # Image region: low text density, medium-to-large area
    if area_ratio > 0.03:
        binary = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 10
        )
        text_density = np.sum(binary > 0) / max(binary.size, 1)
        if text_density < 0.1:
            return "image"

    # Table: check for grid-like structure
    edges = cv2.Canny(roi_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=int(w * 0.3), maxLineGap=5)
    if lines is not None and len(lines) > 6:
        h_lines = sum(1 for l in lines if abs(l[0][3] - l[0][1]) < 5)
        v_lines = sum(1 for l in lines if abs(l[0][2] - l[0][0]) < 5)
        if h_lines > 2 and v_lines > 2:
            return "table"

    return "text_block"


def detect_elements(image_path):
    """
    Detect UI elements in a screenshot using OpenCV contour analysis + Tesseract OCR.

    Returns dict with:
        elements: list of {bbox: [x,y,w,h], label: str, text: str, confidence: float}
        annotated_path: path to image with bounding boxes drawn
        element_count: total detected elements
        ocr_available: whether Tesseract was used
        extracted_text: full OCR text from the frame
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Could not read image: %s", image_path)
        return {"elements": [], "annotated_path": "", "element_count": 0,
                "ocr_available": False, "extracted_text": ""}

    img_h, img_w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elements = []

    # --- Step 1: Contour-based element detection ---
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 25, 12)

    # Dilate to merge nearby text into blocks
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 10))
    dilated = cv2.dilate(binary, kernel_h, iterations=2)
    dilated = cv2.dilate(dilated, kernel_v, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = img_h * img_w * 0.001
    max_area = img_h * img_w * 0.8
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if min_area < area < max_area and w > 20 and h > 10:
            boxes.append([x, y, w, h])

    boxes = _merge_overlapping_boxes(boxes)

    for (x, y, w, h) in boxes:
        roi_gray = gray[y:y+h, x:x+w]
        label = _classify_element(roi_gray, x, y, w, h, img_h, img_w)
        elements.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "label": label,
            "text": "",
            "confidence": 0.0,
        })

    # --- Step 2: OCR with Tesseract (subprocess, no pytesseract) ---
    full_text = ""
    if HAS_TESSERACT:
        ocr_words, full_text = _run_tesseract_tsv(image_path)

        # Assign OCR words to detected elements
        for word in ocr_words:
            ox, oy = word["x"], word["y"]
            for elem in elements:
                bx, by, bw, bh = elem["bbox"]
                if bx <= ox <= bx + bw and by <= oy <= by + bh:
                    if elem["text"]:
                        elem["text"] += " " + word["text"]
                    else:
                        elem["text"] = word["text"]
                    elem["confidence"] = max(elem["confidence"], word["conf"] / 100.0)
                    break

        # Fallback: get full text if TSV failed
        if not full_text:
            full_text = _run_tesseract_text(image_path)

    # --- Step 3: Draw annotated image with bounding boxes ---
    annotated = img.copy()
    colors = {
        "navbar":      (255, 165, 0),
        "footer":      (128, 128, 128),
        "button":      (0, 200, 0),
        "input_field": (0, 180, 255),
        "image":       (255, 0, 255),
        "table":       (255, 255, 0),
        "text_block":  (255, 100, 100),
    }

    for elem in elements:
        x, y, w, h = elem["bbox"]
        color = colors.get(elem["label"], (200, 200, 200))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label_text = elem["label"].replace("_", " ").title()
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 6, y), color, -1)
        cv2.putText(annotated, label_text, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Save annotated image alongside the original
    annotated_path = image_path.rsplit(".", 1)[0] + "_annotated.jpg"
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "elements": elements,
        "annotated_path": annotated_path,
        "element_count": len(elements),
        "ocr_available": HAS_TESSERACT,
        "extracted_text": full_text,
    }
