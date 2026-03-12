"""
Website element detection with bounding boxes using OpenCV + Tesseract OCR.

Detects UI elements (text blocks, buttons, input fields, images, navigation bars,
tables) in a screenshot and returns bounding boxes with labels and OCR text.
Uses subprocess to call tesseract directly to avoid pytesseract/pandas dependency issues.

Enhanced with:
- Image preprocessing (upscale, sharpen, contrast) for better OCR accuracy
- Structured text extraction preserving paragraph/line layout
- Table detection and cell-level OCR extraction
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


# ---------------------------------------------------------------------------
# Image preprocessing for better OCR
# ---------------------------------------------------------------------------

def _preprocess_for_ocr(image_path):
    """Preprocess an image to improve OCR accuracy.

    Returns path to a temporary preprocessed image file.
    Steps: upscale small images, convert to grayscale, denoise, sharpen,
    adaptive contrast enhancement.
    """
    img = cv2.imread(image_path)
    if img is None:
        return image_path

    w = img.shape[1]

    # Upscale small images (< 1200px wide) — Tesseract works best at 300+ DPI
    if w < 1200:
        scale = 2.0
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) for uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)

    # Save to temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(tmp_path, gray)
    return tmp_path


# ---------------------------------------------------------------------------
# Tesseract runners
# ---------------------------------------------------------------------------

def _run_tesseract_tsv(image_path):
    """Run tesseract and parse TSV output for word-level bounding boxes."""
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", "3", "tsv"],
            capture_output=True, text=True, timeout=60,
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


def _run_tesseract_structured(image_path):
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
            ["tesseract", image_path, "stdout", "--psm", "3", "-l", "eng",
             "-c", "preserve_interword_spaces=1", "tsv"],
            capture_output=True, text=True, timeout=60,
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


def _run_tesseract_text(image_path):
    """Run tesseract to get plain text (fallback)."""
    try:
        result = subprocess.run(
            ["tesseract", image_path, "stdout", "--psm", "3"],
            capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


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
    grid = cv2.bitwise_and(h_lines, v_lines)
    grid = cv2.dilate(grid, np.ones((5, 5), np.uint8), iterations=3)

    # Also try: look for regions where h and v lines overlap densely
    combined = cv2.add(h_lines, v_lines)
    # Threshold for regions with enough line density
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
        # A table should be wider than tall or at least reasonably sized
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

    return tables


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
            # Split on multiple spaces or tabs
            import re
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

    for cell in cells:
        # Crop cell region, add padding
        cx, cy, cw, ch = cell["x"], cell["y"], cell["w"], cell["h"]
        pad = 3
        y1 = max(0, cy + pad)
        y2 = min(gray_img.shape[0], cy + ch - pad)
        x1 = max(0, cx + pad)
        x2 = min(gray_img.shape[1], cx + cw - pad)
        roi = gray_img[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        # OCR individual cell
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

        grid[cell["row"]][cell["col"]] = cell_text

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

def detect_elements(image_path):
    """
    Detect UI elements in a screenshot using OpenCV contour analysis + Tesseract OCR.

    Returns dict with:
        elements: list of {bbox: [x,y,w,h], label: str, text: str, confidence: float}
        annotated_path: path to image with bounding boxes drawn
        element_count: total detected elements
        ocr_available: whether Tesseract was used
        extracted_text: full OCR text from the frame (structured with line breaks)
        tables: list of extracted table data
        ocr_confidence: average OCR confidence (0-100)
    """
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Could not read image: %s", image_path)
        return {"elements": [], "annotated_path": "", "element_count": 0,
                "ocr_available": False, "extracted_text": "", "tables": [],
                "ocr_confidence": 0.0}

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

    # --- Step 2: Enhanced OCR with preprocessing ---
    full_text = ""
    tables = []
    ocr_confidence = 0.0
    preprocessed_path = None

    if HAS_TESSERACT:
        # Preprocess image for better OCR
        preprocessed_path = _preprocess_for_ocr(image_path)

        # Run structured OCR (preserves line/paragraph layout)
        structured_text, ocr_words, avg_conf = _run_tesseract_structured(preprocessed_path)
        full_text = structured_text
        ocr_confidence = avg_conf

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
            full_text = _run_tesseract_text(preprocessed_path)
            if not full_text:
                full_text = _run_tesseract_text(image_path)

        # --- Step 2b: Table detection and extraction ---
        table_regions = _detect_table_regions(gray, img_h, img_w)
        for t_bbox in table_regions:
            table_data = _ocr_table(gray, t_bbox)
            if table_data and table_data["num_rows"] > 0:
                tables.append(table_data)

        # Clean up preprocessed temp file
        if preprocessed_path != image_path and os.path.exists(preprocessed_path):
            try:
                os.unlink(preprocessed_path)
            except OSError:
                pass

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

    # Save annotated image alongside the original
    annotated_path = image_path.rsplit(".", 1)[0] + "_annotated.jpg"
    cv2.imwrite(annotated_path, annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return {
        "elements": elements,
        "annotated_path": annotated_path,
        "element_count": len(elements),
        "ocr_available": HAS_TESSERACT,
        "extracted_text": full_text,
        "tables": tables,
        "ocr_confidence": ocr_confidence,
    }
