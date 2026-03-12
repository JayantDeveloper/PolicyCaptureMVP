"""
Report generation module for PolicyCapture Local.

Produces self-contained HTML reports (and a PDF stub) from pipeline results.
"""

import base64
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)


def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return a base64-encoded data URI string."""
    if not os.path.isfile(image_path):
        return ""
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(
        ext, "image/png"
    )
    return f"data:{mime};base64,{data}"


_CSS = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background: #f5f6fa;
    color: #2d3436;
}
.container {
    max-width: 960px;
    margin: 0 auto;
    padding: 40px 24px;
}
header {
    border-bottom: 3px solid #0984e3;
    padding-bottom: 16px;
    margin-bottom: 32px;
}
header h1 {
    margin: 0 0 8px 0;
    font-size: 28px;
    color: #0984e3;
}
header .meta {
    font-size: 13px;
    color: #636e72;
}
header .meta span {
    margin-right: 18px;
}
.section-card {
    background: #fff;
    border: 1px solid #dfe6e9;
    border-radius: 6px;
    margin-bottom: 28px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.section-card .section-header {
    background: #f0f3f8;
    padding: 14px 20px;
    border-bottom: 1px solid #dfe6e9;
}
.section-card .section-header h2 {
    margin: 0;
    font-size: 20px;
    color: #2d3436;
}
.section-card .section-header .badge {
    display: inline-block;
    background: #0984e3;
    color: #fff;
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    margin-left: 10px;
    vertical-align: middle;
}
.section-body {
    padding: 20px;
}
.section-body img {
    max-width: 100%;
    border: 1px solid #dfe6e9;
    border-radius: 4px;
    margin-bottom: 16px;
}
.section-body .timestamp {
    font-size: 12px;
    color: #636e72;
    margin-bottom: 12px;
}
.section-body .summary {
    margin-bottom: 14px;
    line-height: 1.6;
}
.section-body ul.key-points {
    margin: 0;
    padding-left: 20px;
}
.section-body ul.key-points li {
    margin-bottom: 6px;
    line-height: 1.5;
}
footer {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid #dfe6e9;
    font-size: 12px;
    color: #b2bec3;
    text-align: center;
}
"""


def generate_html_report(
    job: dict,
    sections: List[dict],
    screenshots: List[dict],
    output_path: str,
) -> str:
    """
    Generate a self-contained HTML evidence report.

    Args:
        job: Job metadata dict (expects keys: job_id, video_path, status).
        sections: List of synthesized section dicts from synthesize_section.
        screenshots: List of screenshot metadata dicts (expects keys:
                     image_path, timestamp_ms, section_type).
        output_path: File path for the output HTML file.

    Returns:
        Absolute path to the generated HTML file.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    job_id = job.get("job_id", "unknown")
    video_path = job.get("video_path", "unknown")

    # Pair sections with their screenshots
    paired = list(zip(sections, screenshots)) if screenshots else [(s, {}) for s in sections]
    # Sort by order_suggestion
    paired.sort(key=lambda p: p[0].get("order_suggestion", 99))

    # Build section HTML
    sections_html_parts: List[str] = []
    for section, screenshot in paired:
        heading = section.get("heading", "Section")
        section_type = screenshot.get("section_type", section.get("section_type", ""))
        summary = section.get("summary", "")
        key_points = section.get("key_points", [])
        timestamp_ms = screenshot.get("timestamp_ms", 0)
        image_path = screenshot.get("image_path", "")

        img_tag = ""
        if image_path:
            data_uri = _encode_image_base64(image_path)
            if data_uri:
                img_tag = f'<img src="{data_uri}" alt="Screenshot at {timestamp_ms}ms">'

        kp_items = "".join(f"<li>{kp}</li>" for kp in key_points)

        sections_html_parts.append(f"""
        <div class="section-card">
            <div class="section-header">
                <h2>{heading}<span class="badge">{section_type}</span></h2>
            </div>
            <div class="section-body">
                {img_tag}
                <div class="timestamp">Video timestamp: {timestamp_ms / 1000:.1f}s</div>
                <div class="summary">{summary}</div>
                <ul class="key-points">{kp_items}</ul>
            </div>
        </div>
        """)

    sections_html = "".join(sections_html_parts)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PolicyCapture Report - {job_id}</title>
    <style>{_CSS}</style>
</head>
<body>
    <div class="container">
        <header>
            <h1>PolicyCapture Evidence Report</h1>
            <div class="meta">
                <span>Job ID: {job_id}</span>
                <span>Source: {os.path.basename(video_path)}</span>
                <span>Generated: {now}</span>
                <span>Sections: {len(sections)}</span>
            </div>
        </header>
        {sections_html}
        <footer>
            Generated by PolicyCapture Local &middot; {now}
        </footer>
    </div>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("HTML report generated: %s (%d sections)", output_path, len(sections))
    return os.path.abspath(output_path)


def generate_pdf_report(html_path: str, output_path: str) -> str:
    """
    Generate a PDF report from an HTML report.

    Currently a stub that copies the HTML file.

    Args:
        html_path: Path to the source HTML report.
        output_path: Desired path for the PDF output.

    Returns:
        Absolute path to the generated file.
    """
    # TODO: Use ReportLab or weasyprint for PDF generation
    logger.warning(
        "PDF generation not yet implemented. Copying HTML to %s as placeholder.",
        output_path,
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    shutil.copy2(html_path, output_path)
    return os.path.abspath(output_path)
