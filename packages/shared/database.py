"""SQLite database layer for PolicyCapture Local."""
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from packages.shared.config import DB_PATH

_local = threading.local()


def _get_conn() -> sqlite3.Connection:
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Initialize the database schema."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_video_path TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            duration_ms INTEGER,
            frame_count INTEGER,
            screenshot_count INTEGER,
            recipient TEXT DEFAULT '',
            perm_id TEXT DEFAULT '',
            date_of_service TEXT DEFAULT '',
            state TEXT DEFAULT '',
            case_type TEXT DEFAULT '',
            sample TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS frames (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            frame_index INTEGER NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            source_image_path TEXT NOT NULL,
            blur_score REAL DEFAULT 0,
            stability_score REAL DEFAULT 0,
            relevance_score REAL DEFAULT 0,
            matched_keywords TEXT DEFAULT '[]',
            extracted_text TEXT DEFAULT '',
            ocr_confidence REAL DEFAULT 0,
            candidate_score REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS screenshots (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            source_frame_id TEXT,
            image_path TEXT NOT NULL,
            thumbnail_path TEXT DEFAULT '',
            captured_at_ms INTEGER DEFAULT 0,
            section_type TEXT DEFAULT 'unknown',
            confidence REAL DEFAULT 0,
            rationale TEXT DEFAULT '',
            matched_keywords TEXT DEFAULT '[]',
            extracted_text TEXT DEFAULT '',
            accepted INTEGER DEFAULT 1,
            notes TEXT DEFAULT '',
            order_index INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS sections (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            screenshot_id TEXT REFERENCES screenshots(id),
            heading TEXT DEFAULT '',
            section_type TEXT DEFAULT 'unknown',
            summary TEXT DEFAULT '',
            key_points TEXT DEFAULT '[]',
            confidence REAL DEFAULT 0,
            final_order INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            html_path TEXT DEFAULT '',
            pdf_path TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_frames_job ON frames(job_id);
        CREATE INDEX IF NOT EXISTS idx_screenshots_job ON screenshots(job_id);
        CREATE INDEX IF NOT EXISTS idx_sections_job ON sections(job_id);
    """)
    conn.commit()

    # Migrate: add BAH metadata columns to existing jobs tables
    _migrate_jobs_metadata(conn)
    # Migrate: add classifier results table
    _ensure_classifier_table(conn)


def _migrate_jobs_metadata(conn: sqlite3.Connection):
    """Add BAH metadata columns if they don't exist (for existing databases)."""
    cursor = conn.execute("PRAGMA table_info(jobs)")
    existing = {row[1] for row in cursor.fetchall()}
    new_cols = {
        "recipient": "TEXT DEFAULT ''",
        "perm_id": "TEXT DEFAULT ''",
        "date_of_service": "TEXT DEFAULT ''",
        "state": "TEXT DEFAULT ''",
        "case_type": "TEXT DEFAULT ''",
        "sample": "TEXT DEFAULT ''",
    }
    for col, definition in new_cols.items():
        if col not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {col} {definition}")
    conn.commit()


def _ensure_classifier_table(conn: sqlite3.Connection):
    """Create the classification_results table if it doesn't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS classification_results (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id                TEXT NOT NULL,
            frame_id              TEXT,
            filename              TEXT NOT NULL,
            classifier_label      TEXT NOT NULL,
            classifier_confidence REAL NOT NULL,
            svm_label             TEXT,
            svm_confidence        REAL,
            lr_label              TEXT,
            lr_confidence         REAL,
            all_svm_probs         TEXT,
            all_lr_probs          TEXT,
            tfidf_top_keywords    TEXT,
            table_row_count       INTEGER,
            table_text_ratio      REAL,
            avg_cells_per_row     REAL,
            created_at            TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_classif_job_id   ON classification_results(job_id);
        CREATE INDEX IF NOT EXISTS idx_classif_frame_id ON classification_results(frame_id);
    """)
    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- Jobs ---

def create_job(job_id: str, title: str, source_video_path: str = "") -> dict:
    conn = _get_conn()
    now = _now()
    conn.execute(
        "INSERT INTO jobs (id, title, source_video_path, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
        (job_id, title, source_video_path, now, now),
    )
    conn.commit()
    return get_job(job_id)


def get_job(job_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return dict(row)


def list_jobs() -> list[dict]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def update_job_status(job_id: str, status: str, **kwargs) -> dict | None:
    conn = _get_conn()
    sets = ["status = ?", "updated_at = ?"]
    vals = [status, _now()]
    for k, v in kwargs.items():
        sets.append(f"{k} = ?")
        vals.append(v)
    vals.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    return get_job(job_id)


def update_job(job_id: str, **kwargs) -> dict | None:
    conn = _get_conn()
    sets = ["updated_at = ?"]
    vals = [_now()]
    for k, v in kwargs.items():
        sets.append(f"{k} = ?")
        vals.append(v)
    vals.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    return get_job(job_id)


def delete_job(job_id: str) -> bool:
    """Delete a job and all related records (frames, screenshots, sections, reports)."""
    conn = _get_conn()
    job = get_job(job_id)
    if not job:
        return False
    conn.execute("DELETE FROM sections WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM reports WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM screenshots WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM frames WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    conn.commit()
    return True


# --- Frames ---

def create_frame(frame_id: str, job_id: str, frame_index: int, timestamp_ms: int,
                 source_image_path: str, **kwargs) -> dict:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO frames (id, job_id, frame_index, timestamp_ms, source_image_path,
           blur_score, stability_score, relevance_score, matched_keywords,
           extracted_text, ocr_confidence, candidate_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            frame_id, job_id, frame_index, timestamp_ms, source_image_path,
            kwargs.get("blur_score", 0), kwargs.get("stability_score", 0),
            kwargs.get("relevance_score", 0),
            json.dumps(kwargs.get("matched_keywords", [])),
            kwargs.get("extracted_text", ""),
            kwargs.get("ocr_confidence", 0),
            kwargs.get("candidate_score", 0),
        ),
    )
    conn.commit()
    return {"id": frame_id, "job_id": job_id}


def get_frames_for_job(job_id: str, min_relevance: float = 0.0, limit: int = 500) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM frames WHERE job_id = ? AND relevance_score >= ? ORDER BY timestamp_ms LIMIT ?",
        (job_id, min_relevance, limit),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["matched_keywords"] = json.loads(d.get("matched_keywords", "[]"))
        result.append(d)
    return result


def get_frame(frame_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["matched_keywords"] = json.loads(d.get("matched_keywords", "[]"))
    return d


# --- Screenshots ---

def create_screenshot(screenshot_id: str, job_id: str, source_frame_id: str,
                      image_path: str, thumbnail_path: str = "", captured_at_ms: int = 0,
                      section_type: str = "unknown", confidence: float = 0.0,
                      rationale: str = "", matched_keywords: str = "[]",
                      extracted_text: str = "") -> dict:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO screenshots (id, job_id, source_frame_id, image_path, thumbnail_path,
           captured_at_ms, section_type, confidence, rationale, matched_keywords,
           extracted_text, accepted, notes, order_index)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, '', ?)""",
        (screenshot_id, job_id, source_frame_id, image_path, thumbnail_path,
         captured_at_ms, section_type, confidence, rationale, matched_keywords,
         extracted_text, 0),
    )
    conn.commit()
    return {"id": screenshot_id, "job_id": job_id}


def get_screenshots_for_job(job_id: str, section_type: str | None = None,
                            accepted_only: bool = False) -> list[dict]:
    conn = _get_conn()
    query = "SELECT * FROM screenshots WHERE job_id = ?"
    params: list = [job_id]
    if section_type:
        query += " AND section_type = ?"
        params.append(section_type)
    if accepted_only:
        query += " AND accepted = 1"
    query += " ORDER BY order_index, captured_at_ms"
    rows = conn.execute(query, params).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["matched_keywords"] = json.loads(d.get("matched_keywords", "[]"))
        d["accepted"] = bool(d.get("accepted", 1))
        result.append(d)
    return result


def update_screenshot(screenshot_id: str, **kwargs) -> dict | None:
    conn = _get_conn()
    sets = []
    vals = []
    for k, v in kwargs.items():
        if k == "matched_keywords" and isinstance(v, list):
            v = json.dumps(v)
        if k == "accepted":
            v = 1 if v else 0
        sets.append(f"{k} = ?")
        vals.append(v)
    if not sets:
        return get_screenshot(screenshot_id)
    vals.append(screenshot_id)
    conn.execute(f"UPDATE screenshots SET {', '.join(sets)} WHERE id = ?", vals)
    conn.commit()
    return get_screenshot(screenshot_id)


def get_screenshot(screenshot_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM screenshots WHERE id = ?", (screenshot_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["matched_keywords"] = json.loads(d.get("matched_keywords", "[]"))
    d["accepted"] = bool(d.get("accepted", 1))
    return d


# --- Sections ---

def create_section(section_id: str, job_id: str, screenshot_id: str, heading: str = "",
                   section_type: str = "unknown", summary: str = "",
                   key_points: str = "[]", confidence: float = 0.0,
                   final_order: int = 0) -> dict:
    conn = _get_conn()
    conn.execute(
        """INSERT INTO sections (id, job_id, screenshot_id, heading, section_type,
           summary, key_points, confidence, final_order)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (section_id, job_id, screenshot_id, heading, section_type,
         summary, key_points, confidence, final_order),
    )
    conn.commit()
    return {"id": section_id, "job_id": job_id}


def get_sections_for_job(job_id: str) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sections WHERE job_id = ? ORDER BY final_order", (job_id,)
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["key_points"] = json.loads(d.get("key_points", "[]"))
        result.append(d)
    return result


# --- Reports ---

def create_report(report_id: str, job_id: str, html_path: str = "",
                  pdf_path: str = "") -> dict:
    conn = _get_conn()
    conn.execute(
        "INSERT INTO reports (id, job_id, html_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?)",
        (report_id, job_id, html_path, pdf_path, _now()),
    )
    conn.commit()
    return {"id": report_id, "job_id": job_id, "html_path": html_path, "pdf_path": pdf_path}


def get_report_for_job(job_id: str) -> dict | None:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM reports WHERE job_id = ? ORDER BY created_at DESC LIMIT 1", (job_id,)
    ).fetchone()
    return dict(row) if row else None


# --- Classification Results ---

def create_classification_result(
    job_id: str,
    filename: str,
    svm_prediction: dict,
    lr_prediction: dict,
    frame_id: str | None = None,
    features: dict | None = None,
    tfidf_keywords: list | None = None,
) -> dict:
    conn = _get_conn()
    features = features or {}
    tfidf_keywords = tfidf_keywords or []
    svm_label = svm_prediction.get("label", "")
    svm_conf  = svm_prediction.get("confidence", 0.0)
    lr_label  = lr_prediction.get("label", "")
    lr_conf   = lr_prediction.get("confidence", 0.0)
    # Use SVM as primary label; fall back to LR if SVM unavailable
    primary_label = svm_label if svm_label not in ("", "unavailable") else lr_label
    primary_conf  = svm_conf  if svm_label not in ("", "unavailable") else lr_conf
    conn.execute(
        """INSERT INTO classification_results
           (job_id, frame_id, filename, classifier_label, classifier_confidence,
            svm_label, svm_confidence, lr_label, lr_confidence,
            all_svm_probs, all_lr_probs, tfidf_top_keywords,
            table_row_count, table_text_ratio, avg_cells_per_row)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            job_id, frame_id, filename,
            primary_label, primary_conf,
            svm_label, svm_conf,
            lr_label, lr_conf,
            json.dumps(svm_prediction.get("all_probs", {})),
            json.dumps(lr_prediction.get("all_probs", {})),
            json.dumps([kw.get("word", kw) if isinstance(kw, dict) else kw for kw in tfidf_keywords[:20]]),
            features.get("table_row_count", 0),
            features.get("table_text_ratio", 0.0),
            features.get("avg_cells_per_row", 0.0),
        ),
    )
    conn.commit()
    row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return {"id": row_id, "job_id": job_id, "frame_id": frame_id}


def get_classification_results(
    job_id: str | None = None,
    frame_id: str | None = None,
    limit: int = 200,
) -> list[dict]:
    conn = _get_conn()
    if job_id and frame_id:
        rows = conn.execute(
            "SELECT * FROM classification_results WHERE job_id = ? AND frame_id = ? ORDER BY created_at DESC LIMIT ?",
            (job_id, frame_id, limit),
        ).fetchall()
    elif job_id:
        rows = conn.execute(
            "SELECT * FROM classification_results WHERE job_id = ? ORDER BY created_at DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
    elif frame_id:
        rows = conn.execute(
            "SELECT * FROM classification_results WHERE frame_id = ? ORDER BY created_at DESC LIMIT ?",
            (frame_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM classification_results ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
