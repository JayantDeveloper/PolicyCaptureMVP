"""
Website element detection with bounding boxes using OpenCV + Tesseract OCR.

Detects UI elements (text blocks, buttons, input fields, images, navigation bars,
tables) in a screenshot and returns bounding boxes with labels and OCR text.
Uses subprocess to call tesseract directly to avoid pytesseract/pandas dependency issues.

Enhanced with:
- Adaptive multi-strategy image preprocessing for better OCR accuracy
- Deskew correction for rotated/skewed captures
- Multi-PSM mode OCR with best-result selection
- Confidence-based retry with alternative preprocessing
- Structured text extraction preserving paragraph/line layout
- Table detection and cell-level OCR extraction
- Batch processing support for multiple images
"""

import logging
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

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


# ---------------------------------------------------------------------------
# Image preprocessing strategies for better OCR
# ---------------------------------------------------------------------------

def _deskew(gray):
    """Correct slight rotation/skew in the image using minAreaRect on text contours."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 50:
        return gray
    try:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        # Only correct small skews (< 5 degrees)
        if abs(angle) > 5 or abs(angle) < 0.1:
            return gray
        h, w = gray.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _preprocess_strategy_standard(img):
    """Standard preprocessing: grayscale + denoise + CLAHE + sharpen."""
    w = img.shape[1]
    if w < 1200:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
    return gray


def _preprocess_strategy_otsu(img):
    """Otsu binarization — good for clean documents with uniform background."""
    w = img.shape[1]
    if w < 1200:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _preprocess_strategy_adaptive(img):
    """Adaptive threshold — good for uneven lighting/web screenshots."""
    w = img.shape[1]
    if w < 1200:
        img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 10)
    return binary


def _preprocess_strategy_heavy(img):
    """Heavy preprocessing: upscale 3x + aggressive denoise + deskew + CLAHE.
    Best for low-res or noisy captures."""
    w = img.shape[1]
    scale = 3.0 if w < 800 else (2.0 if w < 1200 else 1.5)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)
    gray = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Mild unsharp mask instead of harsh kernel
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    return gray


# All strategies ordered from fastest/lightest to heaviest
_PREPROCESSING_STRATEGIES = [
    ("standard", _preprocess_strategy_standard),
    ("otsu", _preprocess_strategy_otsu),
    ("adaptive", _preprocess_strategy_adaptive),
    ("heavy", _preprocess_strategy_heavy),
]


def _save_temp_image(gray):
    """Save a grayscale/binary image to a temp file and return the path."""
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(tmp_path, gray)
    return tmp_path


def _cleanup_temp(path, original_path):
    """Remove temp file if it's not the original."""
    if path and path != original_path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# OCR text post-processing
# ---------------------------------------------------------------------------

def _clean_ocr_text(text):
    """Post-process OCR text to fix common Tesseract errors."""
    if not text:
        return text

    # Fix common OCR substitutions
    replacements = [
        (r'\bl\b', 'I'),           # lone lowercase L → I (context-dependent)
        (r'(?<=[a-z])0(?=[a-z])', 'o'),  # zero between lowercase → o
        (r'(?<=[A-Z])0(?=[A-Z])', 'O'),  # zero between uppercase → O
        (r'\brn\b', 'm'),           # rn → m when it's a standalone word
        (r'[|]', 'I'),              # pipe → I
    ]

    cleaned = text
    # Remove null bytes and control characters (except newlines/tabs)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)
    # Collapse runs of 3+ spaces into double space
    cleaned = re.sub(r' {3,}', '  ', cleaned)
    # Remove lines that are just noise (single special chars, etc.)
    lines = cleaned.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep empty lines (paragraph breaks) and lines with actual content
        if not stripped or len(stripped) > 1 or stripped.isalnum():
            filtered_lines.append(line)
    cleaned = '\n'.join(filtered_lines)
    # Collapse 3+ consecutive blank lines into 2
    cleaned = re.sub(r'\n{4,}', '\n\n\n', cleaned)

    return cleaned.strip()


# ---------------------------------------------------------------------------
# Tesseract runners
# ---------------------------------------------------------------------------

def _run_tesseract_tsv(image_path, psm="3", timeout=60):
    """Run tesseract and parse TSV output for word-level bounding boxes."""
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", psm, "tsv"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return [], "", 0.0

        words = []
        confidences = []
        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return [], "", 0.0

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
                    confidences.append(conf)
            except (ValueError, KeyError):
                continue

        full_text = " ".join(w["text"] for w in words)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return words, full_text, avg_conf
    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Tesseract TSV failed: %s", e)
        return [], "", 0.0


def _run_tesseract_structured(image_path, psm="3", timeout=60):
    """Run tesseract and extract text preserving line/paragraph structure.

    Uses TSV output's block_num, par_num, line_num fields to reconstruct
    the document layout with proper line breaks and paragraph spacing.

    Returns:
        structured_text: str with preserved layout
        words: list of word dicts with positions
        avg_confidence: float (0-100)
    """
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", psm, "-l", "eng",
             "-c", "preserve_interword_spaces=1", "tsv"],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode != 0:
            return "", [], 0.0

        lines = result.stdout.strip().split("\n")
        if len(lines) < 2:
            return "", [], 0.0

        headers = lines[0].split("\t")

        # Parse all rows
        rows = []
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) != len(headers):
                continue
            rows.append(dict(zip(headers, cols)))

        # Group by block > paragraph > line
        structured_lines = []
        current_block = -1
        current_par = -1
        current_line_num = -1
        current_line_words = []
        words = []
        confidences = []

        for row in rows:
            try:
                block = int(row.get("block_num", 0))
                par = int(row.get("par_num", 0))
                line_n = int(row.get("line_num", 0))
                conf = int(row.get("conf", "-1"))
                text = row.get("text", "").strip()
            except (ValueError, KeyError):
                continue

            # Track line transitions
            if block != current_block or par != current_par or line_n != current_line_num:
                # Flush current line
                if current_line_words:
                    line_text = " ".join(current_line_words)
                    # Add paragraph break on block/par change
                    if structured_lines and (block != current_block or par != current_par):
                        structured_lines.append("")  # blank line = paragraph break
                    structured_lines.append(line_text)
                    current_line_words = []
                current_block = block
                current_par = par
                current_line_num = line_n

            if conf >= 20 and text:
                current_line_words.append(text)
                if conf >= 30:
                    confidences.append(conf)
                    words.append({
                        "x": int(row.get("left", 0)),
                        "y": int(row.get("top", 0)),
                        "w": int(row.get("width", 0)),
                        "h": int(row.get("height", 0)),
                        "text": text,
                        "conf": conf,
                        "block": block,
                        "par": par,
                        "line": line_n,
                    })

        # Flush last line
        if current_line_words:
            structured_lines.append(" ".join(current_line_words))

        structured_text = "\n".join(structured_lines)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return structured_text, words, avg_conf

    except (subprocess.TimeoutExpired, Exception) as e:
        logger.warning("Tesseract structured extraction failed: %s", e)
        return "", [], 0.0


def _run_tesseract_text(image_path, psm="3"):
    """Run tesseract to get plain text (fallback)."""
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", psm],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Multi-strategy OCR: try multiple preprocessing + PSM modes, pick best
# ---------------------------------------------------------------------------

def _ocr_with_strategy(img, strategy_name, strategy_fn, psm="3"):
    """Run OCR with a specific preprocessing strategy. Returns (text, words, confidence, strategy_name)."""
    try:
        processed = strategy_fn(img)
        tmp_path = _save_temp_image(processed)
        try:
            text, words, confidence = _run_tesseract_structured(tmp_path, psm=psm)
            return text, words, confidence, strategy_name
        finally:
            _cleanup_temp(tmp_path, "")
    except Exception as e:
        logger.debug("Strategy %s failed: %s", strategy_name, e)
        return "", [], 0.0, strategy_name


def _best_ocr_result(img, quick=False):
    """Try multiple preprocessing strategies and PSM modes, return the best result.

    Args:
        img: BGR image (numpy array)
        quick: If True, only try the standard strategy (faster, for batch/preview)

    Returns:
        (structured_text, words, avg_confidence, strategy_used)
    """
    if quick:
        strategies = [("standard", _preprocess_strategy_standard)]
    else:
        strategies = _PREPROCESSING_STRATEGIES

    # PSM modes to try:
    # 3 = Fully automatic page segmentation (default)
    # 4 = Assume single column of variable-sized text
    # 6 = Assume a single uniform block of text
    psm_modes = ["3", "6"] if not quick else ["3"]

    best = ("", [], 0.0, "none")

    for strategy_name, strategy_fn in strategies:
        for psm in psm_modes:
            text, words, conf, name = _ocr_with_strategy(img, strategy_name, strategy_fn, psm)
            # Score: weight confidence heavily but also reward more extracted text
            text_len = len(text.strip())
            score = conf * 0.7 + min(text_len / 500.0, 1.0) * 30.0
            best_score = best[2] * 0.7 + min(len(best[0].strip()) / 500.0, 1.0) * 30.0

            if score > best_score:
                best = (text, words, conf, f"{name}_psm{psm}")

        # Early exit: if we got high confidence, don't bother with heavier strategies
        if best[2] >= 75.0:
            break

    return best


# ---------------------------------------------------------------------------
# Table detection and extraction
# ---------------------------------------------------------------------------

def _detect_table_regions(gray, img_h, img_w):
    """Detect table regions by finding grids of horizontal + vertical lines.

    Returns list of table bounding boxes: [(x, y, w, h), ...]
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Detect horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, img_w // 20), 1))
    h_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)), iterations=1)

    # Detect vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, img_h // 20)))
    v_lines = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_kernel)
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)), iterations=1)

    # Combine — table regions have both horizontal and vertical lines
    combined = cv2.add(h_lines, v_lines)
    combined_blur = cv2.GaussianBlur(combined, (15, 15), 0)
    _, dense = cv2.threshold(combined_blur, 30, 255, cv2.THRESH_BINARY)
    dense = cv2.dilate(dense, np.ones((20, 20), np.uint8), iterations=2)

    contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tables = []
    min_table_area = img_h * img_w * 0.005  # at least 0.5% of image
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_table_area:
            continue
        if w < 60 or h < 40:
            continue
        # Check that this region actually has grid lines
        roi_h = h_lines[y:y+h, x:x+w]
        roi_v = v_lines[y:y+h, x:x+w]
        h_count = cv2.countNonZero(roi_h)
        v_count = cv2.countNonZero(roi_v)
        h_density = h_count / max(w * h, 1)
        v_density = v_count / max(w * h, 1)
        if h_density > 0.005 and v_density > 0.005:
            tables.append((x, y, w, h))

    # Also detect borderless tables using text alignment heuristics
    borderless = _detect_borderless_tables(gray, img_h, img_w, tables)
    tables.extend(borderless)

    return tables


def _detect_borderless_tables(gray, img_h, img_w, existing_tables):
    """Detect borderless tables by finding aligned text columns.

    Uses text-like region detection and column alignment analysis.
    """
    # Binary threshold to find text regions
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)

    # Dilate horizontally to merge words into lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    h_dilated = cv2.dilate(binary, h_kernel, iterations=1)

    # Find line-like contours
    contours, _ = cv2.findContours(h_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter for text line regions
    text_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > img_w * 0.15 and 8 < h < 60:  # reasonable text line dimensions
            text_lines.append((x, y, w, h))

    if len(text_lines) < 3:
        return []

    # Sort by y-coordinate
    text_lines.sort(key=lambda t: t[1])

    # Look for groups of lines with consistent left-alignment (potential table rows)
    # Group lines that are close vertically and have similar x-starts
    groups = []
    current_group = [text_lines[0]]
    for i in range(1, len(text_lines)):
        prev = current_group[-1]
        curr = text_lines[i]
        y_gap = curr[1] - (prev[1] + prev[3])
        # Lines in a table are typically evenly spaced
        if y_gap < 40 and y_gap >= 0:
            current_group.append(curr)
        else:
            if len(current_group) >= 3:
                groups.append(current_group)
            current_group = [curr]
    if len(current_group) >= 3:
        groups.append(current_group)

    borderless_tables = []
    for group in groups:
        x_min = min(t[0] for t in group)
        y_min = min(t[1] for t in group)
        x_max = max(t[0] + t[2] for t in group)
        y_max = max(t[1] + t[3] for t in group)
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Skip if it overlaps with an existing bordered table
        overlaps = False
        for et in existing_tables:
            # Check for significant overlap
            ox = max(bbox[0], et[0])
            oy = max(bbox[1], et[1])
            ox2 = min(bbox[0] + bbox[2], et[0] + et[2])
            oy2 = min(bbox[1] + bbox[3], et[1] + et[3])
            if ox < ox2 and oy < oy2:
                overlap_area = (ox2 - ox) * (oy2 - oy)
                bbox_area = bbox[2] * bbox[3]
                if overlap_area > bbox_area * 0.3:
                    overlaps = True
                    break
        if not overlaps and bbox[2] > 100 and bbox[3] > 60:
            borderless_tables.append(bbox)

    return borderless_tables


# ---------------------------------------------------------------------------
# Form field, checkbox, and list detection
# ---------------------------------------------------------------------------

def _detect_checkboxes(gray, img_h, img_w):
    """Detect checkbox-like squares in the image.

    Looks for small square-ish contours that could be checkboxes or radio buttons.
    Returns list of dicts: [{bbox, checked, type}, ...]
    """
    # Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    checkboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Checkboxes are small, roughly square
        if w < 10 or h < 10 or w > 50 or h > 50:
            continue
        aspect = w / max(h, 1)
        if not (0.7 < aspect < 1.4):  # roughly square
            continue
        area = cv2.contourArea(cnt)
        rect_area = w * h
        if rect_area == 0:
            continue
        # Squareness: contour area should be close to bounding rect area
        extent = area / rect_area
        if extent < 0.3:
            continue

        # Check if it's filled (checked) by looking at pixel density inside
        roi = gray[y:y+h, x:x+w]
        _, roi_bin = cv2.threshold(roi, 128, 255, cv2.THRESH_BINARY_INV)
        fill_ratio = np.sum(roi_bin > 0) / max(roi_bin.size, 1)

        # Determine if checkbox or radio button (by circularity)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        cb_type = "radio" if circularity > 0.75 else "checkbox"
        checked = fill_ratio > 0.4  # >40% filled = checked

        checkboxes.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "checked": bool(checked),
            "type": cb_type,
            "fill_ratio": round(float(fill_ratio), 3),
        })

    # Deduplicate overlapping detections (keep larger ones)
    if len(checkboxes) > 1:
        checkboxes.sort(key=lambda c: c["bbox"][2] * c["bbox"][3], reverse=True)
        kept = []
        for cb in checkboxes:
            overlaps = False
            for k in kept:
                cx1, cy1, cw1, ch1 = cb["bbox"]
                cx2, cy2, cw2, ch2 = k["bbox"]
                # Check overlap
                ox = max(cx1, cx2)
                oy = max(cy1, cy2)
                ox2 = min(cx1 + cw1, cx2 + cw2)
                oy2 = min(cy1 + ch1, cy2 + ch2)
                if ox < ox2 and oy < oy2:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(cb)
        checkboxes = kept

    return checkboxes


def _detect_form_fields_visual(gray, elements, img_h, img_w):
    """Detect form input fields by looking for rectangular bordered regions.

    Identifies text input boxes, dropdowns, and text areas by their visual
    characteristics (rectangular borders with mostly white/light interior).

    Returns list of dicts: [{bbox, field_type}, ...]
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Find rectangular contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    fields = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        img_area = img_h * img_w

        # Input fields: medium-width, thin height, reasonable size
        if w < 50 or h < 15 or h > 80 or area < img_area * 0.001:
            continue
        if area > img_area * 0.15:
            continue

        aspect = w / max(h, 1)
        # Input field aspect: wider than tall
        if aspect < 2:
            continue

        # Check rectangularity (how close the contour is to a rectangle)
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        # Check if interior is mostly light (empty field)
        roi = gray[y+2:y+h-2, x+2:x+w-2] if h > 6 and w > 6 else gray[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        mean_val = np.mean(roi)
        std_val = np.std(roi)

        # Classify field type
        if h > 50:
            field_type = "textarea"
        elif mean_val > 200 and std_val < 30:
            field_type = "text_input"  # Empty field
        elif mean_val > 150:
            field_type = "text_input_filled"  # Has content
        else:
            continue  # Probably not a form field

        # Check it doesn't overlap significantly with existing elements
        overlaps = False
        for elem in elements:
            ex, ey, ew, eh = elem["bbox"]
            ox1 = max(x, ex)
            oy1 = max(y, ey)
            ox2 = min(x + w, ex + ew)
            oy2 = min(y + h, ey + eh)
            if ox1 < ox2 and oy1 < oy2:
                overlap_area = (ox2 - ox1) * (oy2 - oy1)
                if overlap_area > area * 0.5:
                    overlaps = True
                    break
        if overlaps:
            continue

        fields.append({
            "bbox": [int(x), int(y), int(w), int(h)],
            "field_type": field_type,
        })

    return fields


def _extract_structured_data_from_text(full_text):
    """Extract structured data forms from OCR text.

    Identifies and extracts:
    - Key-value pairs (e.g., "Name: John Smith", "DOB  03/15/1990")
    - Bulleted/numbered lists
    - Section headers
    - Tabular data in text form (aligned columns)

    Returns dict with:
        key_value_pairs: [{key, value, line_num}, ...]
        lists: [{type: "bulleted"|"numbered", items: [str, ...], start_line}, ...]
        section_headers: [{text, line_num}, ...]
        data_rows: [[col1, col2, ...], ...] (tab/space-aligned data)
    """
    if not full_text:
        return {"key_value_pairs": [], "lists": [], "section_headers": [], "data_rows": []}

    lines = full_text.split("\n")
    key_value_pairs = []
    lists = []
    section_headers = []
    data_rows = []

    # --- Key-value pair detection ---
    # Patterns: "Label: Value", "Label  Value" (wide space), "Label .... Value" (dot leaders)
    kv_patterns = [
        # Colon-separated: "Name: John Smith"
        re.compile(r'^(.{2,40}?)\s*[:]\s+(.+)$'),
        # Dot leader: "Name ........ John Smith"
        re.compile(r'^(.{2,40}?)\s*[.]{3,}\s*(.+)$'),
        # Pipe/bar separated: "Name | John Smith"
        re.compile(r'^(.{2,40}?)\s*[|]\s+(.+)$'),
        # Equals: "Name = John Smith"
        re.compile(r'^(.{2,40}?)\s*=\s+(.+)$'),
    ]

    # --- List detection ---
    bullet_pattern = re.compile(r'^\s*[•\-\*\u2022\u25CF\u25CB\u2023\u2043\u25E6>]\s+(.+)$')
    numbered_pattern = re.compile(r'^\s*(?:\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+|(?:i{1,3}|iv|v|vi{0,3}|ix|x)[.)]\s+)(.+)$')

    current_list = None
    current_list_type = None

    for line_num, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # End current list on blank line
            if current_list:
                lists.append({
                    "type": current_list_type,
                    "items": current_list["items"],
                    "start_line": current_list["start_line"],
                })
                current_list = None
            continue

        # Check for section headers (ALL CAPS line, or short bold-looking line)
        if (stripped.isupper() and 3 < len(stripped) < 60 and not any(c.isdigit() for c in stripped[:3])):
            section_headers.append({"text": stripped, "line_num": line_num})
            if current_list:
                lists.append({
                    "type": current_list_type,
                    "items": current_list["items"],
                    "start_line": current_list["start_line"],
                })
                current_list = None
            continue

        # Check for key-value pairs
        kv_matched = False
        for kv_pat in kv_patterns:
            m = kv_pat.match(stripped)
            if m:
                key = m.group(1).strip()
                value = m.group(2).strip()
                # Validate: key should look like a label (not just random text)
                if len(key) >= 2 and len(value) >= 1 and not key[0].isdigit():
                    key_value_pairs.append({
                        "key": key,
                        "value": value,
                        "line_num": line_num,
                    })
                    kv_matched = True
                    break

        if kv_matched:
            if current_list:
                lists.append({
                    "type": current_list_type,
                    "items": current_list["items"],
                    "start_line": current_list["start_line"],
                })
                current_list = None
            continue

        # Check for bulleted list items
        bullet_m = bullet_pattern.match(stripped)
        if bullet_m:
            item_text = bullet_m.group(1).strip()
            if current_list and current_list_type == "bulleted":
                current_list["items"].append(item_text)
            else:
                if current_list:
                    lists.append({
                        "type": current_list_type,
                        "items": current_list["items"],
                        "start_line": current_list["start_line"],
                    })
                current_list = {"items": [item_text], "start_line": line_num}
                current_list_type = "bulleted"
            continue

        # Check for numbered list items
        numbered_m = numbered_pattern.match(stripped)
        if numbered_m:
            item_text = numbered_m.group(1).strip()
            if current_list and current_list_type == "numbered":
                current_list["items"].append(item_text)
            else:
                if current_list:
                    lists.append({
                        "type": current_list_type,
                        "items": current_list["items"],
                        "start_line": current_list["start_line"],
                    })
                current_list = {"items": [item_text], "start_line": line_num}
                current_list_type = "numbered"
            continue

        # Check for tab/space-aligned data rows (potential table-in-text)
        parts = re.split(r'\t|  {2,}', stripped)
        if len(parts) >= 2 and all(len(p.strip()) > 0 for p in parts):
            data_rows.append([p.strip() for p in parts])
            if current_list:
                lists.append({
                    "type": current_list_type,
                    "items": current_list["items"],
                    "start_line": current_list["start_line"],
                })
                current_list = None
            continue

        # End current list if line doesn't match list pattern
        if current_list:
            lists.append({
                "type": current_list_type,
                "items": current_list["items"],
                "start_line": current_list["start_line"],
            })
            current_list = None

    # Flush any remaining list
    if current_list:
        lists.append({
            "type": current_list_type,
            "items": current_list["items"],
            "start_line": current_list["start_line"],
        })

    return {
        "key_value_pairs": key_value_pairs,
        "lists": lists,
        "section_headers": section_headers,
        "data_rows": data_rows,
    }


def _extract_table_cells(gray, table_bbox):
    """Extract individual cells from a table region.

    Uses horizontal and vertical line detection within the table ROI
    to find cell boundaries, then returns cell grid positions.

    Returns list of cell dicts: [{row, col, x, y, w, h}, ...]
    """
    tx, ty, tw, th = table_bbox
    roi = gray[ty:ty+th, tx:tx+tw]

    # Find lines within the table ROI
    edges = cv2.Canny(roi, 50, 150)

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, tw // 10), 1))
    h_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, h_kernel)

    # Vertical lines
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, th // 10)))
    v_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, v_kernel)

    # Find horizontal line positions (y-coordinates)
    h_proj = np.sum(h_mask, axis=1)
    h_threshold = tw * 0.15
    h_positions = []
    in_line = False
    line_start = 0
    for i, val in enumerate(h_proj):
        if val > h_threshold and not in_line:
            in_line = True
            line_start = i
        elif val <= h_threshold and in_line:
            in_line = False
            h_positions.append((line_start + i) // 2)
    if in_line:
        h_positions.append((line_start + len(h_proj) - 1) // 2)

    # Find vertical line positions (x-coordinates)
    v_proj = np.sum(v_mask, axis=0)
    v_threshold = th * 0.15
    v_positions = []
    in_line = False
    line_start = 0
    for i, val in enumerate(v_proj):
        if val > v_threshold and not in_line:
            in_line = True
            line_start = i
        elif val <= v_threshold and in_line:
            in_line = False
            v_positions.append((line_start + i) // 2)
    if in_line:
        v_positions.append((line_start + len(v_proj) - 1) // 2)

    # Add boundaries if missing
    if not h_positions or h_positions[0] > 10:
        h_positions.insert(0, 0)
    if not h_positions or h_positions[-1] < th - 10:
        h_positions.append(th)
    if not v_positions or v_positions[0] > 10:
        v_positions.insert(0, 0)
    if not v_positions or v_positions[-1] < tw - 10:
        v_positions.append(tw)

    # Need at least 2 rows and 2 columns to be a table
    if len(h_positions) < 2 or len(v_positions) < 2:
        return []

    # Build cell grid
    cells = []
    for ri in range(len(h_positions) - 1):
        for ci in range(len(v_positions) - 1):
            y1 = h_positions[ri]
            y2 = h_positions[ri + 1]
            x1 = v_positions[ci]
            x2 = v_positions[ci + 1]
            cell_w = x2 - x1
            cell_h = y2 - y1
            if cell_w < 10 or cell_h < 10:
                continue
            cells.append({
                "row": ri,
                "col": ci,
                "x": tx + x1,
                "y": ty + y1,
                "w": cell_w,
                "h": cell_h,
            })

    return cells


def _ocr_table(gray_img, table_bbox):
    """Extract structured table data by OCRing individual cells.

    Returns:
        {
            "bbox": [x, y, w, h],
            "rows": [[cell_text, ...], ...],
            "num_rows": int,
            "num_cols": int,
            "header": [str, ...] or None,
            "raw_text": str,
        }
    """
    cells = _extract_table_cells(gray_img, table_bbox)
    if not cells:
        # Fallback: OCR the whole table region as a block
        tx, ty, tw, th = table_bbox
        roi = gray_img[ty:ty+th, tx:tx+tw]

        # Preprocess the table region for better OCR
        if roi.size > 0:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            roi = clahe.apply(roi)

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp, roi)
        try:
            result = subprocess.run(
                ["tesseract", tmp, "stdout", "--psm", "6", "-l", "eng"],
                capture_output=True, text=True, timeout=30,
            )
            raw = result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            raw = ""
        finally:
            os.unlink(tmp)

        # Try to parse lines into rows
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        rows = []
        for line in lines:
            parts = re.split(r'\s{2,}|\t', line)
            rows.append(parts)

        return {
            "bbox": list(table_bbox),
            "rows": rows,
            "num_rows": len(rows),
            "num_cols": max((len(r) for r in rows), default=0),
            "header": rows[0] if rows else None,
            "raw_text": raw,
        }

    # Build grid from cells
    max_row = max(c["row"] for c in cells) + 1
    max_col = max(c["col"] for c in cells) + 1
    grid = [[""] * max_col for _ in range(max_row)]

    # Batch OCR cells using a thread pool for speed
    def _ocr_single_cell(cell):
        cx, cy, cw, ch = cell["x"], cell["y"], cell["w"], cell["h"]
        pad = 3
        y1 = max(0, cy + pad)
        y2 = min(gray_img.shape[0], cy + ch - pad)
        x1 = max(0, cx + pad)
        x2 = min(gray_img.shape[1], cx + cw - pad)
        roi = gray_img[y1:y2, x1:x2]

        if roi.size == 0:
            return cell["row"], cell["col"], ""

        # Preprocess cell for better OCR
        if roi.shape[0] < 40 or roi.shape[1] < 40:
            roi = cv2.resize(roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(tmp, roi)
        try:
            result = subprocess.run(
                ["tesseract", tmp, "stdout", "--psm", "7", "-l", "eng"],
                capture_output=True, text=True, timeout=10,
            )
            cell_text = result.stdout.strip() if result.returncode == 0 else ""
        except Exception:
            cell_text = ""
        finally:
            os.unlink(tmp)

        return cell["row"], cell["col"], cell_text

    # Use thread pool for parallel cell OCR (Tesseract is the bottleneck)
    with ThreadPoolExecutor(max_workers=min(4, len(cells))) as executor:
        futures = [executor.submit(_ocr_single_cell, cell) for cell in cells]
        for future in as_completed(futures):
            try:
                row_idx, col_idx, text = future.result()
                grid[row_idx][col_idx] = text
            except Exception:
                pass

    # Build raw text representation
    raw_lines = []
    for row in grid:
        raw_lines.append("\t".join(row))
    raw_text = "\n".join(raw_lines)

    return {
        "bbox": list(table_bbox),
        "rows": grid,
        "num_rows": max_row,
        "num_cols": max_col,
        "header": grid[0] if grid else None,
        "raw_text": raw_text,
    }


# ---------------------------------------------------------------------------
# Element detection helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def detect_elements(image_path, quick=False, run_classifier=False):
    """
    Detect UI elements in a screenshot using OpenCV contour analysis + Tesseract OCR.

    Uses multi-strategy adaptive preprocessing to get the best OCR results:
    tries multiple preprocessing approaches and PSM modes, selects the highest
    confidence result. Falls back to progressively heavier strategies if initial
    results are poor.

    Args:
        image_path: Path to the screenshot image file.
        quick: If True, use only the standard strategy (faster, for batch/preview).

    Returns dict with:
        elements: list of {bbox: [x,y,w,h], label: str, text: str, confidence: float}
        annotated_path: path to image with bounding boxes drawn
        element_count: total detected elements
        ocr_available: whether Tesseract was used
        extracted_text: full OCR text from the frame (structured with line breaks)
        tables: list of extracted table data
        ocr_confidence: average OCR confidence (0-100)
        ocr_strategy: which preprocessing strategy produced the best result
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Could not read image: %s", image_path)
        return {"elements": [], "annotated_path": "", "element_count": 0,
                "ocr_available": False, "extracted_text": "", "tables": [],
                "ocr_confidence": 0.0, "ocr_strategy": "none"}

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

    # --- Step 2: Multi-strategy adaptive OCR ---
    full_text = ""
    tables = []
    ocr_confidence = 0.0
    ocr_strategy = "none"

    if HAS_TESSERACT:
        # Use multi-strategy OCR to get the best result
        structured_text, ocr_words, avg_conf, strategy = _best_ocr_result(img, quick=quick)
        full_text = _clean_ocr_text(structured_text)
        ocr_confidence = avg_conf
        ocr_strategy = strategy

        logger.info("OCR: strategy=%s confidence=%.1f text_len=%d for %s",
                     strategy, avg_conf, len(full_text), os.path.basename(image_path))

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

        # Fallback: get full text if structured failed
        if not full_text:
            # Try plain text with the standard preprocessed image
            processed = _preprocess_strategy_standard(img)
            tmp = _save_temp_image(processed)
            try:
                full_text = _run_tesseract_text(tmp)
                if not full_text:
                    full_text = _run_tesseract_text(image_path)
            finally:
                _cleanup_temp(tmp, image_path)

        # --- Step 2b: Table detection and extraction ---
        table_regions = _detect_table_regions(gray, img_h, img_w)
        for t_bbox in table_regions:
            table_data = _ocr_table(gray, t_bbox)
            if table_data and table_data["num_rows"] > 0:
                tables.append(table_data)

    # --- Step 2c: Detect checkboxes/radio buttons ---
    checkboxes = _detect_checkboxes(gray, img_h, img_w)

    # --- Step 2d: Detect form input fields ---
    form_fields = _detect_form_fields_visual(gray, elements, img_h, img_w)

    # --- Step 2e: Extract structured data from OCR text ---
    structured_data = _extract_structured_data_from_text(full_text)

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

    # Draw table bounding boxes in cyan
    for t in tables:
        bx, by, bw, bh = t["bbox"]
        cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 255, 0), 3)
        cv2.putText(annotated, f"Table ({t['num_rows']}x{t['num_cols']})",
                    (bx + 5, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)

    # Draw checkboxes in green/red
    for cb in checkboxes:
        cbx, cby, cbw, cbh = cb["bbox"]
        cb_color = (0, 200, 0) if cb["checked"] else (0, 0, 200)
        cv2.rectangle(annotated, (cbx, cby), (cbx + cbw, cby + cbh), cb_color, 2)
        label = f"{'[x]' if cb['checked'] else '[ ]'} {cb['type']}"
        cv2.putText(annotated, label, (cbx + cbw + 3, cby + cbh - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, cb_color, 1, cv2.LINE_AA)

    # Draw form fields in blue
    for ff in form_fields:
        fx, fy, fw, fh = ff["bbox"]
        cv2.rectangle(annotated, (fx, fy), (fx + fw, fy + fh), (200, 100, 0), 2)
        cv2.putText(annotated, ff["field_type"].replace("_", " "),
                    (fx + 3, fy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 100, 0), 1, cv2.LINE_AA)

    # Save annotated image alongside the original
    annotated_path = image_path.rsplit(".", 1)[0] + "_annotated.jpg"
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    result = {
        "elements": elements,
        "annotated_path": annotated_path,
        "element_count": len(elements),
        "ocr_available": HAS_TESSERACT,
        "extracted_text": full_text,
        "tables": tables,
        "checkboxes": checkboxes,
        "form_fields": form_fields,
        "structured_data": structured_data,
        "ocr_confidence": ocr_confidence,
        "ocr_strategy": ocr_strategy,
    }

    if run_classifier:
        try:
            from packages.core.pipeline.classifier import run_ocr as _classify
            with open(image_path, "rb") as _f:
                _bytes = _f.read()
            clf = _classify(_bytes, filename=os.path.basename(image_path), mode="fast")
            result["classification"] = {
                "label":      clf["svm_prediction"]["label"],
                "confidence": clf["svm_prediction"]["confidence"],
                "svm":        clf["svm_prediction"],
                "lr":         clf["lr_prediction"],
                "keywords":   clf["tfidf"]["keywords"],
                "features":   clf["features"],
            }
        except Exception as _exc:
            logger.warning("ML classifier failed for %s: %s", image_path, _exc)
            result["classification"] = {"error": str(_exc)}

    return result


# ---------------------------------------------------------------------------
# Batch processing API
# ---------------------------------------------------------------------------

def detect_elements_batch(image_paths, max_workers=4, quick=False, run_classifier=False):
    """Process multiple images in parallel using a thread pool.

    Args:
        image_paths: List of image file paths to process.
        max_workers: Maximum number of concurrent OCR threads.
        quick: If True, use only the standard preprocessing strategy.

    Returns:
        List of result dicts (same format as detect_elements), in input order.
    """
    empty_result = {
        "elements": [], "annotated_path": "", "element_count": 0,
        "ocr_available": False, "extracted_text": "", "tables": [],
        "ocr_confidence": 0.0, "ocr_strategy": "none",
    }
    results = [dict(empty_result) for _ in image_paths]

    def _process(idx, path):
        return idx, detect_elements(path, quick=quick, run_classifier=run_classifier)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, i, p) for i, p in enumerate(image_paths)]
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results[idx] = result
            except Exception as e:
                logger.error("Batch OCR failed: %s", e)

    return results
