"""Report generation for PolicyCapture.

Produces structured HTML + PDF evidence reports in the BAH four-section format:

  Header  : Case ID, PERM ID, recipient, date-of-service, state, case type
  Section 1: Case List — where you were in the system
  Section 2: Program Eligibility — tab opened, date-of-service evidence
  Section 3: Eligibility Evidence — external system captures
  Section 4: State Eligibility Determination — certifications / history
"""

import base64
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ── Section definitions ──────────────────────────────────────────────────────

SECTION_DEFS = [
    {
        "number": 1,
        "title": "Case List",
        "subtitle": "Where you were in the system",
        "fields": ["What screen was captured", "Why it matters"],
    },
    {
        "number": 2,
        "title": "Program Eligibility",
        "subtitle": "What tab was opened",
        "fields": ["Screenshot or saved PDF", "Relevant date-of-service evidence"],
    },
    {
        "number": 3,
        "title": "Eligibility Evidence",
        "subtitle": "External System section capture",
        "fields": [],
    },
    {
        "number": 4,
        "title": "State Eligibility Determination",
        "subtitle": "Certifications / determination history capture for the relevant date of service",
        "fields": [],
    },
]

# BAH pipeline section_type → fixed section number
_SECTION_ROUTING = {
    "table":            1,
    "unknown":          1,
    "demographics":     2,
    "income":           2,
    "household":        2,
    "application_step": 2,
    "eligibility":      3,
    "policy_guidance":  4,
}

# ML classifier label → section number (used when available)
_ML_LABEL_ROUTING = {
    "Case List":                    1,
    "Program Eligibility":          2,
    "Eligibility Evidence":         3,
    "State Eligibility":            4,
    "State Eligibility Determination": 4,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _encode_image_base64(image_path: str) -> str:
    if not image_path or not os.path.isfile(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    return f"data:{mime};base64,{data}"


def _route_screenshot(screenshot: dict, section: dict) -> int:
    """Return the section number (1-4) this screenshot belongs to."""
    # ML label takes priority if present on the screenshot
    ml_label = screenshot.get("ml_label", "")
    if ml_label and ml_label in _ML_LABEL_ROUTING:
        return _ML_LABEL_ROUTING[ml_label]
    # Fall back to BAH pipeline section_type
    section_type = screenshot.get("section_type") or section.get("section_type", "unknown")
    return _SECTION_ROUTING.get(section_type, 1)


def _bucket_items(sections: List[dict], screenshots: List[dict]) -> dict:
    """Group (section, screenshot) pairs into 4 buckets by section number."""
    buckets: dict = {1: [], 2: [], 3: [], 4: []}
    pairs = list(zip(sections, screenshots)) if screenshots else [(s, {}) for s in sections]
    for section, screenshot in pairs:
        bucket = _route_screenshot(screenshot, section)
        buckets[bucket].append((section, screenshot))
    return buckets


def _fmt_ms(ms) -> str:
    """Format milliseconds as MM:SS.s"""
    try:
        total_sec = int(ms) / 1000
        mins = int(total_sec // 60)
        secs = total_sec % 60
        return f"{mins:02d}:{secs:04.1f}"
    except (TypeError, ValueError):
        return "—"


# ── HTML generation ──────────────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    background: #f0f2f5;
    color: #1b1f23;
}
.page { max-width: 900px; margin: 0 auto; padding: 32px 20px; }

/* ── Case header ── */
.case-header {
    background: #003087;
    color: #fff;
    border-radius: 6px 6px 0 0;
    padding: 18px 24px 14px;
    margin-bottom: 0;
}
.case-header .title-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}
.case-header h1 { font-size: 20px; font-weight: 700; letter-spacing: .3px; }
.case-header .generated { font-size: 11px; opacity: .75; margin-top: 4px; }
.case-meta {
    background: #f7f9fc;
    border: 1px solid #d0d7de;
    border-top: none;
    border-radius: 0 0 6px 6px;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0;
    margin-bottom: 28px;
    overflow: hidden;
}
.meta-cell {
    padding: 10px 16px;
    border-right: 1px solid #d0d7de;
    border-bottom: 1px solid #d0d7de;
}
.meta-cell:nth-child(3n) { border-right: none; }
.meta-cell:nth-last-child(-n+3) { border-bottom: none; }
.meta-cell .label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .6px;
    color: #6e7781;
    margin-bottom: 3px;
}
.meta-cell .value { font-size: 13px; font-weight: 600; color: #1b1f23; }
.meta-cell .value.empty { color: #b0b7c0; font-weight: 400; font-style: italic; }

/* ── Section cards ── */
.section-card {
    background: #fff;
    border: 1px solid #d0d7de;
    border-radius: 6px;
    margin-bottom: 24px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    page-break-inside: avoid;
}
.section-card .s-header {
    background: linear-gradient(90deg, #003087 0%, #0055a5 100%);
    padding: 13px 20px;
    display: flex;
    align-items: baseline;
    gap: 10px;
}
.section-card .s-header .s-num {
    background: rgba(255,255,255,.2);
    color: #fff;
    font-size: 11px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 3px;
    white-space: nowrap;
}
.section-card .s-header h2 {
    color: #fff;
    font-size: 16px;
    font-weight: 700;
}
.section-card .s-header .s-subtitle {
    color: rgba(255,255,255,.8);
    font-size: 12px;
    margin-left: auto;
    font-style: italic;
}
.section-body { padding: 0; }

/* ── Capture item (screenshot + details) ── */
.capture {
    border-bottom: 1px solid #eaecef;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
}
.capture:last-child { border-bottom: none; }
.capture-img {
    border-right: 1px solid #eaecef;
    padding: 16px;
    background: #f7f9fc;
    display: flex;
    align-items: flex-start;
    justify-content: center;
}
.capture-img img {
    max-width: 100%;
    max-height: 340px;
    border: 1px solid #d0d7de;
    border-radius: 3px;
    object-fit: contain;
}
.capture-img .no-img {
    color: #b0b7c0;
    font-size: 12px;
    font-style: italic;
    padding: 40px 0;
}
.capture-details { padding: 16px 20px; }
.capture-details .timestamp {
    font-size: 11px;
    color: #6e7781;
    margin-bottom: 10px;
}
.capture-details .field { margin-bottom: 10px; }
.capture-details .field-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: #6e7781;
    margin-bottom: 3px;
}
.capture-details .field-value { font-size: 12px; line-height: 1.55; color: #1b1f23; }
.capture-details .kp-list { padding-left: 16px; margin: 0; }
.capture-details .kp-list li { margin-bottom: 4px; font-size: 12px; line-height: 1.5; }

/* ── Empty section ── */
.empty-section {
    padding: 24px 20px;
    color: #b0b7c0;
    font-size: 12px;
    font-style: italic;
}

/* ── Footer ── */
footer {
    margin-top: 32px;
    padding-top: 14px;
    border-top: 1px solid #d0d7de;
    font-size: 11px;
    color: #b0b7c0;
    text-align: center;
}

@media print {
    body { background: #fff; }
    .page { padding: 0; max-width: none; }
    .section-card { page-break-inside: avoid; box-shadow: none; }
}
"""


def generate_html_report(
    job: dict,
    sections: List[dict],
    screenshots: List[dict],
    output_path: str,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    job_id      = job.get("job_id", "—")
    recipient   = job.get("recipient", "") or "—"
    perm_id     = job.get("perm_id", "") or "—"
    dos         = job.get("date_of_service", "") or "—"
    state       = job.get("state", "") or "—"
    case_type   = job.get("case_type", "") or "—"
    sample      = job.get("sample", "") or "—"
    video_name  = os.path.basename(job.get("video_path") or job.get("source_video_path") or "")

    def _mv(v):
        empty = v in ("—", "", None)
        cls = "empty" if empty else ""
        disp = v if not empty else "not provided"
        return f'<div class="value {cls}">{disp}</div>'

    meta_html = f"""
    <div class="meta-cell"><div class="label">PERM ID</div>{_mv(perm_id)}</div>
    <div class="meta-cell"><div class="label">Recipient</div>{_mv(recipient)}</div>
    <div class="meta-cell"><div class="label">Date of Service</div>{_mv(dos)}</div>
    <div class="meta-cell"><div class="label">State</div>{_mv(state)}</div>
    <div class="meta-cell"><div class="label">Case Type</div>{_mv(case_type)}</div>
    <div class="meta-cell"><div class="label">Sample</div>{_mv(sample)}</div>
    """

    buckets = _bucket_items(sections, screenshots)

    def _render_capture(section: dict, screenshot: dict, idx: int) -> str:
        image_path = screenshot.get("image_path") or screenshot.get("screenshot_path", "")
        ts_ms      = screenshot.get("timestamp_ms") or screenshot.get("captured_at_ms", 0)
        heading    = section.get("heading", "")
        summary    = section.get("summary", "")
        key_points = section.get("key_points", [])
        notes      = screenshot.get("notes", "")

        img_html = ""
        if image_path:
            uri = _encode_image_base64(image_path)
            img_html = f'<img src="{uri}" alt="Capture {idx+1}">' if uri else '<div class="no-img">Image not available</div>'
        else:
            img_html = '<div class="no-img">No screenshot recorded</div>'

        kp_items = "".join(f"<li>{kp}</li>" for kp in key_points) if key_points else ""

        return f"""
        <div class="capture">
            <div class="capture-img">{img_html}</div>
            <div class="capture-details">
                <div class="timestamp">Captured at {_fmt_ms(ts_ms)}</div>
                <div class="field">
                    <div class="field-label">Screen captured</div>
                    <div class="field-value">{heading or "—"}</div>
                </div>
                <div class="field">
                    <div class="field-label">Summary</div>
                    <div class="field-value">{summary or "—"}</div>
                </div>
                {"" if not kp_items else f'<div class="field"><div class="field-label">Key points</div><ul class="kp-list">{kp_items}</ul></div>'}
                {"" if not notes else f'<div class="field"><div class="field-label">Notes</div><div class="field-value">{notes}</div></div>'}
            </div>
        </div>
        """

    sections_html = ""
    for sdef in SECTION_DEFS:
        num    = sdef["number"]
        items  = buckets.get(num, [])
        subtitle = sdef["subtitle"]

        captures_html = ""
        if items:
            for i, (sec, ss) in enumerate(items):
                captures_html += _render_capture(sec, ss, i)
        else:
            captures_html = '<div class="empty-section">No captures recorded for this section.</div>'

        sections_html += f"""
        <div class="section-card">
            <div class="s-header">
                <span class="s-num">Section {num}</span>
                <h2>{sdef["title"]}</h2>
                <span class="s-subtitle">{subtitle}</span>
            </div>
            <div class="section-body">{captures_html}</div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PolicyCapture Report — {perm_id}</title>
    <style>{_CSS}</style>
</head>
<body>
<div class="page">
    <div class="case-header">
        <div class="title-row">
            <div>
                <h1>PolicyCapture Evidence Report</h1>
                <div class="generated">Generated {now}</div>
            </div>
            <div style="text-align:right;font-size:11px;opacity:.7">
                Job: {job_id}<br>
                {"Source: " + video_name if video_name else ""}
            </div>
        </div>
    </div>
    <div class="case-meta">{meta_html}</div>
    {sections_html}
    <footer>PolicyCapture Local &nbsp;&middot;&nbsp; {now} &nbsp;&middot;&nbsp; {len(sections)} capture(s)</footer>
</div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("HTML report written: %s (%d sections)", output_path, len(sections))
    return os.path.abspath(output_path)


# ── PDF generation (ReportLab) ───────────────────────────────────────────────

def generate_pdf_report(
    job: dict,
    sections: List[dict],
    screenshots: List[dict],
    output_path: str,
) -> str:
    """Generate a structured PDF evidence report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak, KeepTogether,
        )
        from reportlab.platypus import Image as RLImage
        from PIL import Image as PILImage
    except ImportError as exc:
        logger.warning("ReportLab/Pillow not available, skipping PDF: %s", exc)
        return output_path

    # ── Colours ──
    NAVY    = colors.HexColor("#003087")
    BLUE    = colors.HexColor("#0055a5")
    LGRAY   = colors.HexColor("#f7f9fc")
    MGRAY   = colors.HexColor("#d0d7de")
    DGRAY   = colors.HexColor("#6e7781")
    BLACK   = colors.HexColor("#1b1f23")
    WHITE   = colors.white

    PAGE_W, PAGE_H = letter
    MARGIN = 0.75 * inch
    CONTENT_W = PAGE_W - 2 * MARGIN

    # ── Styles ──
    base = getSampleStyleSheet()

    def _style(name, parent="Normal", **kw):
        s = ParagraphStyle(name, parent=base[parent])
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    s_title   = _style("s_title",   "Normal",   fontSize=18, textColor=WHITE, fontName="Helvetica-Bold",   spaceAfter=2)
    s_sub     = _style("s_sub",     "Normal",   fontSize=9,  textColor=colors.HexColor("#c5d0e0"), fontName="Helvetica")
    s_label   = _style("s_label",   "Normal",   fontSize=8,  textColor=DGRAY,  fontName="Helvetica",       spaceAfter=1, spaceBefore=6, leading=10)
    s_value   = _style("s_value",   "Normal",   fontSize=10, textColor=BLACK,  fontName="Helvetica-Bold",  spaceAfter=4, leading=14)
    s_empty   = _style("s_empty",   "Normal",   fontSize=9,  textColor=DGRAY,  fontName="Helvetica-Oblique")
    s_sec_hdr = _style("s_sec_hdr", "Normal",   fontSize=13, textColor=WHITE,  fontName="Helvetica-Bold",  spaceAfter=0)
    s_sec_sub = _style("s_sec_sub", "Normal",   fontSize=9,  textColor=colors.HexColor("#c5d0e0"), fontName="Helvetica-Oblique")
    s_ts      = _style("s_ts",      "Normal",   fontSize=8,  textColor=DGRAY,  fontName="Helvetica",       spaceAfter=4)
    s_field_l = _style("s_field_l", "Normal",   fontSize=8,  textColor=DGRAY,  fontName="Helvetica",       spaceAfter=1, spaceBefore=6, leading=10)
    s_field_v = _style("s_field_v", "Normal",   fontSize=10, textColor=BLACK,  fontName="Helvetica",       spaceAfter=4, leading=14)
    s_bullet  = _style("s_bullet",  "Normal",   fontSize=9,  textColor=BLACK,  fontName="Helvetica",       leftIndent=12, leading=13)
    s_footer  = _style("s_footer",  "Normal",   fontSize=8,  textColor=DGRAY,  fontName="Helvetica",       alignment=TA_CENTER)

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    job_id    = job.get("job_id", "—")
    recipient = job.get("recipient", "") or "—"
    perm_id   = job.get("perm_id", "") or "—"
    dos       = job.get("date_of_service", "") or "—"
    state     = job.get("state", "") or "—"
    case_type = job.get("case_type", "") or "—"
    sample    = job.get("sample", "") or "—"

    story = []

    # ── Cover header ──────────────────────────────────────────────────────────
    header_data = [
        [Paragraph("PolicyCapture Evidence Report", s_title),
         Paragraph(f"Generated: {now_str}", s_sub)],
    ]
    header_tbl = Table(header_data, colWidths=[CONTENT_W * 0.65, CONTENT_W * 0.35])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), NAVY),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING",   (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 12),
        ("ALIGN",        (1, 0), (1, 0),   "RIGHT"),
        ("ROUNDEDCORNERS", (0, 0), (-1, -1), 4),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0))

    # ── Case metadata grid ────────────────────────────────────────────────────
    def _meta_cell(label, value):
        empty = value in ("—", "", None)
        v = value if not empty else "not provided"
        return [Paragraph(label, s_label), Paragraph(v, s_empty if empty else s_value)]

    meta_data = [
        [_meta_cell("PERM ID", perm_id),   _meta_cell("Recipient", recipient), _meta_cell("Date of Service", dos)],
        [_meta_cell("State", state),        _meta_cell("Case Type", case_type), _meta_cell("Sample", sample)],
    ]
    col_w = CONTENT_W / 3
    meta_tbl = Table(meta_data, colWidths=[col_w, col_w, col_w], rowHeights=None)
    meta_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), LGRAY),
        ("BOX",          (0, 0), (-1, -1), 0.5, MGRAY),
        ("INNERGRID",    (0, 0), (-1, -1), 0.5, MGRAY),
        ("LEFTPADDING",  (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING",   (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 18))

    # ── Helper: embed image ────────────────────────────────────────────────────
    def _rl_image(image_path: str, max_w: float):
        if not image_path or not os.path.isfile(image_path):
            return None
        try:
            with PILImage.open(image_path) as pil_img:
                orig_w, orig_h = pil_img.size
            if orig_w == 0:
                return None
            aspect = orig_h / orig_w
            w = min(max_w, orig_w)
            h = w * aspect
            # Cap height to half a page
            max_h = PAGE_H * 0.45
            if h > max_h:
                h = max_h
                w = h / aspect
            return RLImage(image_path, width=w, height=h)
        except Exception as exc:
            logger.debug("Could not load image %s: %s", image_path, exc)
            return None

    # ── Sections ───────────────────────────────────────────────────────────────
    buckets = _bucket_items(sections, screenshots)

    for sdef in SECTION_DEFS:
        num   = sdef["number"]
        items = buckets.get(num, [])

        # Section header bar
        sec_hdr_data = [[
            Paragraph(f"Section {num}:  {sdef['title']}", s_sec_hdr),
            Paragraph(sdef["subtitle"], s_sec_sub),
        ]]
        sec_hdr_tbl = Table(sec_hdr_data, colWidths=[CONTENT_W * 0.45, CONTENT_W * 0.55])
        sec_hdr_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), NAVY),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING",   (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 9),
            ("ALIGN",        (1, 0), (1, 0),   "RIGHT"),
        ]))
        story.append(KeepTogether([sec_hdr_tbl]))

        if not items:
            story.append(Spacer(1, 6))
            story.append(Paragraph("No captures recorded for this section.", s_empty))
            story.append(Spacer(1, 16))
            continue

        for sec, ss in items:
            img_path = ss.get("image_path") or ss.get("screenshot_path", "")
            ts_ms    = ss.get("timestamp_ms") or ss.get("captured_at_ms", 0)
            heading  = sec.get("heading", "")
            summary  = sec.get("summary", "")
            kps      = sec.get("key_points", [])
            notes    = ss.get("notes", "")

            capture_parts = [Spacer(1, 8)]

            # Timestamp
            capture_parts.append(Paragraph(f"Captured at {_fmt_ms(ts_ms)}", s_ts))

            # Image
            img = _rl_image(img_path, CONTENT_W)
            if img:
                capture_parts.append(img)
                capture_parts.append(Spacer(1, 6))

            # Detail fields
            if heading:
                capture_parts.append(Paragraph("Screen captured", s_field_l))
                capture_parts.append(Paragraph(heading, s_field_v))

            if summary:
                capture_parts.append(Paragraph("Summary", s_field_l))
                capture_parts.append(Paragraph(summary, s_field_v))

            if kps:
                capture_parts.append(Paragraph("Key points", s_field_l))
                for kp in kps:
                    capture_parts.append(Paragraph(f"• {kp}", s_bullet))

            if notes:
                capture_parts.append(Paragraph("Notes", s_field_l))
                capture_parts.append(Paragraph(notes, s_field_v))

            capture_parts.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=MGRAY, spaceAfter=8))
            story.append(KeepTogether(capture_parts))

        story.append(Spacer(1, 10))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width=CONTENT_W, thickness=0.5, color=MGRAY, spaceBefore=8))
    story.append(Paragraph(f"PolicyCapture Local  ·  {now_str}  ·  {len(sections)} capture(s)", s_footer))

    # ── Build ─────────────────────────────────────────────────────────────────
    def _on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(DGRAY)
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN * 0.45, f"Page {doc.page}")
        canvas.drawString(MARGIN, MARGIN * 0.45, f"PERM ID: {perm_id}  ·  {recipient}")
        canvas.restoreState()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN * 0.75,
        title=f"PolicyCapture Report — {perm_id}",
        author="PolicyCapture Local",
    )
    doc.build(story, onFirstPage=_on_page, onLaterPages=_on_page)

    logger.info("PDF report written: %s", output_path)
    return os.path.abspath(output_path)
