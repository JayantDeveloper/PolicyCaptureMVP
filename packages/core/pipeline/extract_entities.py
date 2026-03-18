"""Regex-based Named Entity Recognition pipeline for OCR text.

Extracts structured entities (dates, currencies, emails, phones, SSNs,
medical codes, EINs, claim numbers, ZIP codes, and more) from raw OCR
output using regex patterns and heuristics. No ML dependencies.
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
    "Effective", "Coverage", "Premium", "Benefits", "Insurance",
    "National", "Federal", "General", "Standard", "Special",
    "Information", "Application", "Authorization", "Certification",
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

# Full US state names (for state_name entity type)
_US_STATE_NAMES: set[str] = {
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "New Hampshire", "New Jersey", "New Mexico", "New York",
    "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon",
    "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "West Virginia", "Wisconsin", "Wyoming", "District of Columbia",
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
    r"Partners|Consulting|Services|Solutions|Technologies|Systems|"
    r"Insurance|Mutual|Health|Medical|Hospital|Clinic|University|"
    r"Agency|Bureau|Department|Commission|Authority|Board)"
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
    # 4. EIN / Tax ID (XX-XXXXXXX)
    (
        "ein",
        re.compile(
            r"\b(?:"
            r"EIN[\s:]+\d{2}-\d{7}"
            r"|(?:Tax\s*(?:ID|Identification)\s*(?:Number|No\.?)?|TIN)[\s:]+\d{2}-\d{7}"
            r"|(?:FEIN|Employer\s+ID)[\s:]+\d{2}-\d{7}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 5. NPI handled separately below (uses capture group)
    # 6. Medical codes (ICD-10, CPT, HCPCS, DRG, NDC)
    (
        "medical_code",
        re.compile(
            r"\b(?:"
            # ICD-10 codes: A00-Z99 with optional decimal (A00.0, M54.5, Z23)
            r"[A-TV-Z]\d{2}(?:\.\d{1,4})?"
            r"|"
            # CPT codes: 5-digit numeric (00100-99499)
            r"(?:CPT|HCPCS)[\s:#]*\d{5}"
            r"|"
            # DRG codes: 3-digit
            r"DRG[\s:#]*\d{3}"
            r"|"
            # NDC codes: various formats (XXXXX-XXXX-XX)
            r"NDC[\s:#]*\d{5}-\d{4}-\d{2}"
            r"|"
            # Revenue code: 4-digit with context
            r"Rev(?:enue)?\s+(?:Code|code)[\s:#]*\d{4}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 7. Claim / authorization number
    (
        "claim_number",
        re.compile(
            r"\b(?:"
            r"(?:Claim|CLM|Auth(?:orization)?|Prior\s+Auth)[\s:#]*[A-Z0-9][\w\-]{4,20}"
            r"|"
            r"(?:Approval|Confirmation|Transaction|Reference)\s*(?:#|No\.?|Number)\s*[A-Z0-9][\w\-]{4,20}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 8. Group / plan number
    (
        "group_number",
        re.compile(
            r"\b(?:"
            r"(?:Group|GRP|Plan|Benefit)\s*(?:#|No\.?|Number|ID)[\s:]*[A-Z0-9][\w\-]{2,15}"
            r"|"
            r"(?:Subscriber|Member|Enrollee)\s*(?:#|No\.?|Number|ID)[\s:]*[A-Z0-9][\w\-]{3,15}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 9. Phone numbers (expanded for international)
    (
        "phone",
        re.compile(
            r"(?:"
            r"\+?1[\s\-]?)?"                          # optional country code
            r"(?:\(\d{3}\)[\s\-]?\d{3}[\s\-]?\d{4}"   # (XXX) XXX-XXXX
            r"|\d{3}[\s\-]\d{3}[\s\-]\d{4}"           # XXX-XXX-XXXX
            r"|\+1\d{10}"                              # +1XXXXXXXXXX
            r"|\d{3}\.\d{3}\.\d{4}"                    # XXX.XXX.XXXX
            r"|\+\d{1,3}[\s\-]\d{1,4}[\s\-]\d{4,10}"  # international
            r")\b"
        ),
    ),
    # 10. Currency (expanded — USD, EUR, GBP, plus written amounts)
    (
        "currency",
        re.compile(
            r"(?:"
            r"\$\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"      # $1,250.00
            r"|\bUSD\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"   # USD 1,250.00
            r"|\bEUR\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?"  # EUR
            r"|\bGBP\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"   # GBP
            r"|\u00a3\s?\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?"   # pound sign
            r"|\u20ac\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?"  # euro sign
            r")",
            re.IGNORECASE,
        ),
    ),
    # 11. Percentage
    (
        "percentage",
        re.compile(r"\b\d+(?:\.\d+)?%"),
    ),
    # 12. Dates — multiple formats (expanded)
    (
        "date",
        re.compile(
            r"\b(?:"
            # MM/DD/YYYY or MM-DD-YYYY or MM.DD.YYYY
            r"(?:0?[1-9]|1[0-2])[/\-.](?:0?[1-9]|[12]\d|3[01])[/\-.](?:19|20)\d{2}"
            r"|"
            # YYYY-MM-DD or YYYY/MM/DD
            r"(?:19|20)\d{2}[/\-](?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])"
            r"|"
            # Month DD, YYYY  or  Month DD YYYY (full month names)
            r"(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},?\s+(?:19|20)\d{2}"
            r"|"
            # Mon DD, YYYY or Mon DD YYYY (abbreviated month names)
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"\.?\s+\d{1,2},?\s+(?:19|20)\d{2}"
            r"|"
            # DD Month YYYY (European style)
            r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+(?:19|20)\d{2}"
            r"|"
            # DD-Mon-YYYY  (e.g. 05-Jan-2025)
            r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(?:19|20)\d{2}"
            r"|"
            # MM/DD/YY (2-digit year)
            r"(?:0?[1-9]|1[0-2])[/\-](?:0?[1-9]|[12]\d|3[01])[/\-]\d{2}(?!\d)"
            r")\b"
        ),
    ),
    # 13. Time values (12h and 24h)
    (
        "time_value",
        re.compile(
            r"\b(?:"
            r"(?:1[0-2]|0?[1-9]):[0-5]\d\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)"
            r"|"
            r"(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d)?"
            r")\b"
        ),
    ),
    # 14. ZIP code (standalone, not part of larger number)
    (
        "zip_code",
        re.compile(
            r"\b(?:"
            r"(?:ZIP|Zip|zip)[\s:]*\d{5}(?:-\d{4})?"
            r"|"
            # 5 or 9-digit ZIP after a state abbreviation
            r"(?<=[A-Z]{2}\s)\d{5}(?:-\d{4})?"
            r")\b"
        ),
    ),
    # 15. Policy number
    (
        "policy_number",
        re.compile(
            r"\b(?:"
            r"POL[\-#]?\s?\d[\d\-]+"
            r"|Policy\s*#?\s*\d[\d\-]+"
            r"|(?:Certificate|Cert)\s*(?:#|No\.?)\s*\d[\d\-]+"
            r"|(?:Contract|Account)\s*(?:#|No\.?)\s*\d[\d\-]+"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 16. Case / reference number
    (
        "case_number",
        re.compile(
            r"\b(?:"
            r"Case\s*(?:#|No\.?)\s*\d[\d\-]+"
            r"|Ref(?:erence)?\s*(?:#|No\.?)\s*\d[\d\-]+"
            r"|(?:Docket|File|Matter)\s*(?:#|No\.?)\s*\d[\d\-]+"
            r"|(?:Ticket|Incident)\s*(?:#|No\.?)\s*[A-Z0-9][\w\-]+"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 17. Address (heuristic: number + words + street suffix + optional state/zip)
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
    # 18. Account / routing number (bank context required)
    (
        "account_number",
        re.compile(
            r"\b(?:"
            r"(?:Account|Acct|A/C)[\s:#]*\d{6,17}"
            r"|(?:Routing|ABA|RTN)[\s:#]*\d{9}"
            r"|(?:IBAN)[\s:]*[A-Z]{2}\d{2}[A-Z0-9]{4,30}"
            r")\b",
            re.IGNORECASE,
        ),
    ),
    # 19. ID number (generic — ID: XXXX, No. XXXX, #XXXX)
    (
        "id_number",
        re.compile(
            r"\b(?:"
            r"ID[\s:]+\d{4,12}"
            r"|No\.\s*\d{4,12}"
            r"|#\d{4,12}"
            r"|(?:License|DL|Driver'?s?\s*License)[\s:#]*[A-Z0-9]{5,15}"
            r"|(?:Passport)[\s:#]*[A-Z0-9]{6,12}"
            r"|(?:VIN)[\s:#]*[A-HJ-NPR-Z0-9]{17}"
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

# ---------------------------------------------------------------------------
# Key-value pair extraction from form-like text
# ---------------------------------------------------------------------------

# Common form field labels in policy/benefits documents
_FORM_FIELD_LABELS: set[str] = {
    # Personal info
    "name", "first name", "last name", "middle name", "full name",
    "maiden name", "suffix", "prefix", "title",
    "date of birth", "dob", "birth date", "age", "gender", "sex",
    "race", "ethnicity", "nationality", "citizenship", "language",
    "ssn", "social security", "social security number",
    "driver license", "drivers license", "dl number", "passport",
    # Contact
    "address", "street", "street address", "mailing address",
    "city", "state", "zip", "zip code", "county", "country",
    "phone", "telephone", "cell", "mobile", "fax", "email",
    "home phone", "work phone", "cell phone", "email address",
    "emergency contact", "contact name", "contact phone",
    # Employment/financial
    "employer", "employer name", "occupation", "job title",
    "income", "salary", "wages", "annual income", "monthly income",
    "gross income", "net income", "hourly rate",
    "bank name", "routing number", "account number",
    # Relationship
    "marital status", "status", "relationship", "spouse",
    "spouse name", "beneficiary", "beneficiary name",
    # Case/policy
    "case number", "case id", "application number", "application id",
    "policy number", "policy id", "member id", "member number",
    "group number", "group id", "subscriber id", "enrollee id",
    "claim number", "authorization number", "approval number",
    "certificate number", "contract number",
    # Dates/terms
    "effective date", "start date", "end date", "expiration date",
    "termination date", "renewal date", "issue date", "filing date",
    "date received", "date submitted", "date of service",
    "date of loss", "date of injury", "date of accident",
    # Coverage/plan
    "plan", "plan name", "plan type", "coverage", "coverage type",
    "premium", "deductible", "copay", "copayment", "coinsurance",
    "out of pocket", "out of pocket max", "annual max", "lifetime max",
    "benefit amount", "benefit period", "waiting period",
    # Medical
    "provider", "physician", "doctor", "facility", "hospital",
    "npi", "tax id", "ein", "tin",
    "diagnosis", "diagnosis code", "procedure", "procedure code",
    "medication", "prescription", "dosage", "pharmacy",
    "primary diagnosis", "secondary diagnosis",
    "referring provider", "attending physician", "rendering provider",
    "place of service", "service date", "admission date", "discharge date",
    # Financial totals
    "total", "subtotal", "amount", "balance", "payment",
    "amount due", "amount paid", "total charges", "total payment",
    "billed amount", "allowed amount", "paid amount",
    "patient responsibility", "coinsurance amount",
    # Administrative
    "signature", "date signed", "applicant", "representative",
    "household size", "number of dependents", "dependents",
    "reason", "description", "comments", "notes",
    "type", "category", "class", "level", "tier",
}

_KV_PATTERNS: list[re.Pattern[str]] = [
    # "Label: Value"
    re.compile(r'^(.{2,50}?)\s*:\s+(.+)$', re.MULTILINE),
    # "Label ......... Value" (dot leaders)
    re.compile(r'^(.{2,50}?)\s*[.]{3,}\s*(.+)$', re.MULTILINE),
    # "Label | Value" (pipe separated)
    re.compile(r'^(.{2,50}?)\s*\|\s+(.+)$', re.MULTILINE),
]


def _extract_key_value_pairs(text: str) -> list[dict[str, Any]]:
    """Extract key-value pairs from text that looks like form data.

    Detects patterns like:
        Name: John Smith
        Date of Birth: 03/15/1990
        Income ......... $45,000
        Status | Active

    Returns list of {key, value} dicts.
    """
    if not text:
        return []

    pairs: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for pattern in _KV_PATTERNS:
        for m in pattern.finditer(text):
            key = m.group(1).strip().rstrip(".")
            value = m.group(2).strip()

            # Validate: key should be short, value should exist
            if len(key) < 2 or len(key) > 50 or len(value) < 1:
                continue
            # Key shouldn't start with a digit (probably a data row, not a label)
            if key[0].isdigit():
                continue
            # Skip if key is just punctuation/noise
            if not any(c.isalpha() for c in key):
                continue

            key_lower = key.lower().strip()
            # Boost confidence if key matches known form labels
            is_known_label = key_lower in _FORM_FIELD_LABELS or any(
                fl in key_lower for fl in _FORM_FIELD_LABELS
            )

            # Deduplicate by key
            if key_lower in seen_keys:
                continue
            seen_keys.add(key_lower)

            pairs.append({
                "key": key,
                "value": value,
                "is_known_field": is_known_label,
            })

    return pairs


def _extract_lists(text: str) -> list[dict[str, Any]]:
    """Extract bulleted and numbered lists from text.

    Returns list of list dicts: [{type, items}, ...]
    """
    if not text:
        return []

    lines = text.split("\n")
    lists: list[dict[str, Any]] = []
    current_items: list[str] = []
    current_type: str | None = None

    bullet_re = re.compile(r'^\s*[•\-\*\u2022\u25CF\u25CB\u2023\u2043\u25E6>]\s+(.+)$')
    number_re = re.compile(r'^\s*(\d{1,3})[.)]\s+(.+)$')
    letter_re = re.compile(r'^\s*([a-zA-Z])[.)]\s+(.+)$')

    for line in lines:
        stripped = line.strip()

        bullet_m = bullet_re.match(stripped)
        number_m = number_re.match(stripped)
        letter_m = letter_re.match(stripped)

        if bullet_m:
            if current_type != "bulleted" and current_items:
                lists.append({"type": current_type, "items": current_items})
                current_items = []
            current_type = "bulleted"
            current_items.append(bullet_m.group(1).strip())
        elif number_m:
            if current_type != "numbered" and current_items:
                lists.append({"type": current_type, "items": current_items})
                current_items = []
            current_type = "numbered"
            current_items.append(number_m.group(2).strip())
        elif letter_m:
            if current_type != "lettered" and current_items:
                lists.append({"type": current_type, "items": current_items})
                current_items = []
            current_type = "lettered"
            current_items.append(letter_m.group(2).strip())
        elif stripped == "":
            # Blank line ends current list
            if current_items and len(current_items) >= 2:
                lists.append({"type": current_type, "items": current_items})
            current_items = []
            current_type = None
        else:
            # Non-list line ends current list
            if current_items and len(current_items) >= 2:
                lists.append({"type": current_type, "items": current_items})
            current_items = []
            current_type = None

    # Flush
    if current_items and len(current_items) >= 2:
        lists.append({"type": current_type, "items": current_items})

    return lists


def _extract_section_headers(text: str) -> list[str]:
    """Extract section headers from text (ALL CAPS lines, short bold lines)."""
    if not text:
        return []

    headers: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # ALL CAPS lines that are reasonably short and don't contain ':'  or '#'
        if (stripped.isupper() and 3 < len(stripped) < 80
                and ":" not in stripped and "#" not in stripped):
            # Skip lines that contain digits (likely data, not headers)
            alpha_chars = sum(1 for c in stripped if c.isalpha())
            if alpha_chars >= 3 and alpha_chars > len(stripped) * 0.5:
                headers.append(stripped)
        # Title Case lines that look like standalone headers
        # Must not contain colons (those are key-value pairs), digits at start,
        # or list markers
        elif (stripped.istitle() and 3 < len(stripped) < 60
              and ":" not in stripped and "#" not in stripped
              and not stripped.endswith((",", ".", ";"))
              and not stripped[0].isdigit()
              and " " in stripped):
            word_count = len(stripped.split())
            if 2 <= word_count <= 6:
                headers.append(stripped)

    return headers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict:
    """Extract named entities and structured data from OCR text.

    Extracts:
    - Named entities (dates, currencies, emails, phones, SSNs, etc.)
    - Key-value pairs from form fields ("Name: John Smith")
    - Bulleted and numbered lists
    - Section headers
    - Organization and person names

    Returns:
        {
            "entities": [
                {"text": "03/18/2026", "type": "date", "start": 10, "end": 20},
                {"text": "$1,250.00", "type": "currency", "start": 45, "end": 54},
                ...
            ],
            "categories": {
                "date": [...], "currency": [...], "email": [...],
                "phone": [...], "percentage": [...], "ssn": [...],
                "policy_number": [...], "case_number": [...],
                "address": [...], "person_name": [...],
                "organization": [...], "url": [...], "id_number": [...],
            },
            "form_data": {
                "key_value_pairs": [{"key": "Name", "value": "John Smith", "is_known_field": true}, ...],
                "lists": [{"type": "bulleted", "items": ["item1", "item2"]}, ...],
                "section_headers": ["DEMOGRAPHICS", "INCOME INFORMATION", ...],
            },
            "summary": {
                "total_entities": <int>,
                "types_found": [<str>, ...],
                "form_fields_found": <int>,
                "lists_found": <int>,
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

    # ---- NPI: extract capture group (pattern uses group) ----
    # Re-run NPI to get the actual number from capture group
    _NPI_RE = re.compile(
        r"\b(?:NPI|National\s+Provider\s+(?:Identifier|ID))[\s:#]*(\d{10})\b",
        re.IGNORECASE,
    )
    for m in _NPI_RE.finditer(text):
        raw_entities.append(
            {
                "text": m.group(1),
                "type": "npi",
                "start": m.start(1),
                "end": m.end(1),
            }
        )

    # ---- state names (full US state names) ----
    for state_name in _US_STATE_NAMES:
        for m in re.finditer(r"\b" + re.escape(state_name) + r"\b", text):
            raw_entities.append(
                {
                    "text": m.group(0),
                    "type": "state",
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
        "date", "time_value", "currency", "email", "phone", "percentage",
        "ssn", "ein", "npi", "medical_code", "claim_number", "group_number",
        "policy_number", "case_number", "address", "zip_code", "state",
        "account_number", "person_name", "organization", "url", "id_number",
    ]
    categories: dict[str, list[str]] = {t: [] for t in all_types}
    for ent in entities:
        categories[ent["type"]].append(ent["text"])

    types_found = [t for t in all_types if categories[t]]

    # ---- extract structured form data ----
    key_value_pairs = _extract_key_value_pairs(text)
    text_lists = _extract_lists(text)
    section_headers = _extract_section_headers(text)

    return {
        "entities": entities,
        "categories": categories,
        "form_data": {
            "key_value_pairs": key_value_pairs,
            "lists": text_lists,
            "section_headers": section_headers,
        },
        "summary": {
            "total_entities": len(entities),
            "types_found": types_found,
            "form_fields_found": len(key_value_pairs),
            "lists_found": len(text_lists),
        },
    }
