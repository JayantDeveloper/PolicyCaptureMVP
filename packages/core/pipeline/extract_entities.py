"""Regex-based Named Entity Recognition pipeline for OCR text.

Extracts structured entities (dates, currencies, emails, phones, etc.)
from raw OCR output using regex patterns and heuristics. No ML dependencies.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Common words to EXCLUDE from person-name heuristic matches
# ---------------------------------------------------------------------------
_COMMON_TITLE_WORDS: set[str] = {
    "The", "This", "That", "These", "Those", "There", "Their", "They",
    "What", "When", "Where", "Which", "While", "With", "Within", "Without",
    "About", "Above", "After", "Again", "Against", "Along", "Also",
    "Among", "Around", "Before", "Below", "Between", "Beyond", "Both",
    "Could", "Dear", "Does", "Down", "During", "Each", "Either", "Else",
    "Every", "From", "Have", "Here", "How", "Into", "Just", "Like",
    "Many", "More", "Most", "Much", "Must", "Near", "Never", "Next",
    "None", "North", "South", "East", "West", "Note", "Notes", "Notice",
    "Only", "Other", "Over", "Page", "Part", "Please", "Rather", "Same",
    "Shall", "Should", "Since", "Some", "Still", "Such", "Sure", "Take",
    "Than", "Then", "Under", "Until", "Upon", "Very", "Were", "Will",
    "Would", "Your", "Section", "Total", "Amount", "Date", "Time",
    "Subject", "Dear", "Sincerely", "Regards", "Thank", "Thanks",
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "New", "Old", "Not", "Now", "Our", "Out", "Own", "Per", "See",
    "Set", "She", "Too", "Use", "Way", "Was", "You", "All", "And",
    "Are", "But", "For", "Get", "Got", "Had", "Has", "Her", "Him",
    "His", "Its", "Let", "May", "Nor", "Off", "One", "Put", "Ran",
    "Say", "Two", "Who", "Why", "Yes", "Yet",
}

# US state abbreviations (used in address detection)
_US_STATES: set[str] = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC",
}

# Street suffixes for address detection
_STREET_SUFFIXES = (
    r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Rd|Road|"
    r"Ln|Lane|Ct|Court|Pl(?:ace)?|Way|Cir(?:cle)?|Pkwy|Parkway|"
    r"Ter(?:race)?|Trail|Trl|Hwy|Highway)"
)

# Organization suffixes
_ORG_SUFFIXES = (
    r"(?:Inc\.?|LLC|L\.L\.C\.?|Corp(?:oration)?\.?|Ltd\.?|Co\.?|"
    r"LP|L\.P\.?|LLP|L\.L\.P\.?|PLC|Group|Foundation|Association|"
    r"Partners|Consulting|Services|Solutions|Technologies|Systems)"
)

# ---------------------------------------------------------------------------
# Regex patterns — order matters: more specific patterns first to avoid
# partial matches being consumed by greedier patterns.
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 1. URL (before email so http://user@host isn't split)
    (
        "url",
        re.compile(
            r"https?://[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+",
            re.ASCII,
        ),
    ),
    # 2. Email
    (
        "email",
        re.compile(
            r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
            re.ASCII,
        ),
    ),
    # 3. SSN  (XXX-XX-XXXX)
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    # 4. Phone numbers
    (
        "phone",
        re.compile(
            r"(?:"
            r"\+?1[\s\-]?)?"                       # optional country code
            r"(?:\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}"  # (XXX) XXX-XXXX
            r"|\d{3}[\s\-]\d{3}[\s\-]\d{4}"          # XXX-XXX-XXXX
            r"|\+1\d{10}"                             # +1XXXXXXXXXX
            r")\b"
        ),
    ),
    # 5. Currency
    (
        "currency",
        re.compile(
            r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"  # $1,250.00
            r"|\bUSD\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?",
            re.IGNORECASE,
        ),
    ),
    # 6. Percentage
    (
        "percentage",
        re.compile(r"\b\d+(?:\.\d+)?%"),
    ),
    # 7. Dates — multiple formats
    (
        "date",
        re.compile(
            r"\b(?:"
            # MM/DD/YYYY or MM-DD-YYYY
            r"(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-](?:19|20)\d{2}"
            r"|"
            # YYYY-MM-DD
            r"(?:19|20)\d{2}-(?:0?[1-9]|1[0-2])-(?:0?[1-9]|[12]\d|3[01])"
            r"|"
            # Month DD, YYYY  or  Month DD YYYY
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},?\s+(?:19|20)\d{2}"
            r"|"
            # DD-Mon-YYYY  (e.g. 05-Jan-2025)
            r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(?:19|20)\d{2}"
            r")\b"
        ),
    ),
    # 8. Policy number
    (
        "policy_number",
        re.compile(
            r"\b(?:"
            r"POL[\-#]?\s?\d[\d\-]+"
            r"|Policy\s*#?\s*\d[\d\-]+"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 9. Case / reference number
    (
        "case_number",
        re.compile(
            r"\b(?:"
            r"Case\s*(?:#|No\.?)\s*\d[\d\-]+"
            r"|Ref(?:erence)?\s*(?:#|No\.?)\s*\d[\d\-]+"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 10. Address (heuristic: number + words + street suffix + optional state/zip)
    (
        "address",
        re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][a-z]+\s+){0,3}"
            + _STREET_SUFFIXES
            + r"\.?"
            r"(?:,?\s+(?:[A-Z][a-z]+\.?\s*)+,?\s+"
            r"(?:" + "|".join(_US_STATES) + r")"
            r"\s+\d{5}(?:-\d{4})?)?"
        ),
    ),
    # 11. ID number (ID: XXXX, No. XXXX, #XXXX — generic)
    (
        "id_number",
        re.compile(
            r"\b(?:"
            r"ID[\s:]+\d{4,12}"
            r"|No\.\s*\d{4,12}"
            r"|#\d{4,12}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
]

# Organization: ALL CAPS sequences (2+ words) or Name + suffix
_ORG_ALL_CAPS = re.compile(r"\b(?:[A-Z]{2,}(?:\s+[A-Z]{2,})+)\b")
_ORG_SUFFIX = re.compile(
    r"\b(?:[A-Z][A-Za-z&]+(?:\s+[A-Z][A-Za-z&]+)*)\s+" + _ORG_SUFFIXES + r"\b"
)

# Person name: Title-case sequences of 2-3 words
_PERSON_NAME = re.compile(
    r"\b([A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,2})\b"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_ssn(raw: str) -> str:
    """Replace first 5 digits of an SSN with asterisks: 123-45-6789 -> ***-**-6789."""
    return re.sub(r"^\d{3}-\d{2}", "***-**", raw)


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _deduplicate_entities(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove duplicate and overlapping entities, preferring longer matches."""
    # Sort by length descending so longer matches win ties.
    entities.sort(key=lambda e: -(e["end"] - e["start"]))
    kept: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []
    for ent in entities:
        span = (ent["start"], ent["end"])
        if any(_spans_overlap(span, u) for u in used_spans):
            continue
        kept.append(ent)
        used_spans.append(span)
    # Sort by position for final output.
    kept.sort(key=lambda e: e["start"])
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict:
    """Extract named entities from OCR text using regex patterns.

    Returns:
        {
            "entities": [
                {"text": "03/18/2026", "type": "date", "start": 10, "end": 20},
                {"text": "$1,250.00", "type": "currency", "start": 45, "end": 54},
                ...
            ],
            "categories": {
                "date": [...],
                "currency": [...],
                "email": [...],
                "phone": [...],
                "percentage": [...],
                "ssn": [...],          # masked for safety
                "policy_number": [...],
                "case_number": [...],
                "address": [...],
                "person_name": [...],
                "organization": [...],
                "url": [...],
                "id_number": [...],
            },
            "summary": {
                "total_entities": <int>,
                "types_found": [<str>, ...],
            }
        }
    """
    raw_entities: list[dict[str, Any]] = []

    # ---- fixed-pattern entities ----
    for entity_type, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            matched_text = m.group(0).strip()
            if not matched_text:
                continue
            display_text = (
                _mask_ssn(matched_text) if entity_type == "ssn" else matched_text
            )
            raw_entities.append(
                {
                    "text": display_text,
                    "type": entity_type,
                    "start": m.start(),
                    "end": m.end(),
                }
            )

    # ---- organization (all-caps sequences) ----
    for m in _ORG_ALL_CAPS.finditer(text):
        candidate = m.group(0).strip()
        # Skip very short matches and lone state abbreviations
        if len(candidate) < 4 or candidate in _US_STATES:
            continue
        raw_entities.append(
            {
                "text": candidate,
                "type": "organization",
                "start": m.start(),
                "end": m.end(),
            }
        )

    # ---- organization (name + suffix) ----
    for m in _ORG_SUFFIX.finditer(text):
        raw_entities.append(
            {
                "text": m.group(0).strip(),
                "type": "organization",
                "start": m.start(),
                "end": m.end(),
            }
        )

    # ---- person name (title-case heuristic) ----
    for m in _PERSON_NAME.finditer(text):
        candidate = m.group(0)
        words = candidate.split()
        # Every word must NOT be in the common-word exclusion set.
        if any(w in _COMMON_TITLE_WORDS for w in words):
            continue
        # Reject if the candidate overlaps an already-matched span.
        # (We'll do final dedup later, but this avoids adding obvious noise.)
        raw_entities.append(
            {
                "text": candidate,
                "type": "person_name",
                "start": m.start(),
                "end": m.end(),
            }
        )

    # ---- deduplicate & sort ----
    entities = _deduplicate_entities(raw_entities)

    # ---- build categories dict ----
    all_types = [
        "date", "currency", "email", "phone", "percentage", "ssn",
        "policy_number", "case_number", "address", "person_name",
        "organization", "url", "id_number",
    ]
    categories: dict[str, list[str]] = {t: [] for t in all_types}
    for ent in entities:
        categories[ent["type"]].append(ent["text"])

    types_found = [t for t in all_types if categories[t]]

    return {
        "entities": entities,
        "categories": categories,
        "summary": {
            "total_entities": len(entities),
            "types_found": types_found,
        },
    }
