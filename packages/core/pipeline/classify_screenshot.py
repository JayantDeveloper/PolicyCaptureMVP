"""
Screenshot classification module for PolicyCapture Local.

Classifies frames into policy-relevant section types using keyword rules.
"""

# TODO: Replace with trained classifier (e.g., fine-tuned BERT or vision model)

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# Rule-based section classification keywords
SECTION_RULES: Dict[str, List[str]] = {
    "demographics": [
        "name", "address", "date of birth", "ssn", "gender", "age",
        "contact", "phone", "email",
    ],
    "income": [
        "income", "wages", "salary", "earnings", "employment", "employer", "pay",
    ],
    "household": [
        "household", "members", "dependents", "family", "spouse", "children",
    ],
    "eligibility": [
        "eligible", "eligibility", "qualify", "qualified", "determination",
        "approved", "denied",
    ],
    "policy_guidance": [
        "policy", "regulation", "rule", "guidance", "requirement", "criteria",
    ],
    "application_step": [
        "step", "submit", "continue", "next", "apply", "application", "form",
        "upload",
    ],
    "table": [
        "table", "row", "column", "total", "amount", "quantity",
    ],
}


def classify_screenshot(
    extracted_text: str,
    matched_keywords: List[str],
    structure_score: float,
) -> dict:
    """
    Classify a screenshot into a policy-relevant section type.

    Args:
        extracted_text: Text extracted from the screenshot (e.g., via OCR).
        matched_keywords: Keywords already matched during relevance detection.
        structure_score: Visual structure score from detect_relevance.

    Returns:
        dict with keys:
            section_type (str): Best-matching section type or "unknown".
            confidence (float): Classification confidence in [0, 1].
            rationale (str): Human-readable explanation.
            matched_rules (dict): Per-section match details.
    """
    text_lower = extracted_text.lower()
    all_keywords_lower = [kw.lower() for kw in matched_keywords]

    matched_rules: Dict[str, List[str]] = {}
    section_scores: Dict[str, float] = {}

    for section_type, rule_keywords in SECTION_RULES.items():
        matches: List[str] = []
        for kw in rule_keywords:
            kw_lower = kw.lower()
            # Check in extracted text or in previously matched keywords
            if kw_lower in text_lower or kw_lower in all_keywords_lower:
                matches.append(kw)

        matched_rules[section_type] = matches
        if rule_keywords:
            section_scores[section_type] = len(matches) / len(rule_keywords)
        else:
            section_scores[section_type] = 0.0

    # Boost table score if visual structure is detected
    if "table" in section_scores and structure_score >= 0.4:
        section_scores["table"] = min(
            section_scores["table"] + structure_score * 0.3, 1.0
        )

    # Determine best section
    if section_scores:
        best_section = max(section_scores, key=section_scores.get)  # type: ignore[arg-type]
        best_score = section_scores[best_section]
    else:
        best_section = "unknown"
        best_score = 0.0

    if best_score < 0.1:
        best_section = "unknown"
        best_score = 0.0

    # Build rationale
    if best_section == "unknown":
        rationale = "No section rules matched with sufficient confidence."
    else:
        matched_kws = matched_rules.get(best_section, [])
        rationale = (
            f"Classified as '{best_section}' based on {len(matched_kws)} "
            f"keyword match(es): {', '.join(matched_kws)}."
        )
        if best_section == "table" and structure_score >= 0.4:
            rationale += f" Boosted by visual structure score ({structure_score:.2f})."

    result = {
        "section_type": best_section,
        "confidence": round(best_score, 4),
        "rationale": rationale,
        "matched_rules": matched_rules,
    }

    logger.debug(
        "Classification: section=%s, confidence=%.4f",
        result["section_type"],
        result["confidence"],
    )
    return result
