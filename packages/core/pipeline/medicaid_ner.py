"""Medicaid-domain Named Entity Recognition pipeline.

Four-layer architecture:
  1. Dictionary layer — seed lexicon of 500+ Medicaid terms mapped to 18 entity types
  2. spaCy NER layer — catches multi-token entities the dictionary misses
  3. GLiNER layer — zero-shot transformer NER for domain-specific entities
  4. Normalization layer — maps variants to canonical labels with acronym resolution

Designed to work alongside the existing regex-based extract_entities.py for
structured data (dates, SSNs, phones, etc.) while this module handles
domain-specific Medicaid policy entities.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# 18 Medicaid entity types + seed lexicons
# ---------------------------------------------------------------------------

ENTITY_LABELS: dict[str, str] = {
    "PROGRAM_BRAND": "State-specific Medicaid program names",
    "PROGRAM_TYPE": "Program categories (Medicaid, CHIP, managed care, etc.)",
    "AGENCY_OR_GOV_BODY": "Government agencies and bodies",
    "ELIGIBILITY_GROUP": "Population/eligibility categories",
    "PERSON_ROLE": "Roles in the Medicaid system",
    "FINANCIAL_TERM": "Financial and eligibility terms",
    "APPLICATION_PROCESS": "Enrollment/application process terms",
    "BENEFIT_OR_SERVICE": "Covered benefits and services",
    "CARE_SETTING": "Healthcare settings and facilities",
    "PROVIDER_TYPE": "Provider categories",
    "PAYMENT_OR_DELIVERY_MODEL": "Payment and delivery models",
    "WAIVER_OR_AUTHORITY": "Federal waivers and authorities",
    "DOCUMENT_OR_RECORD": "Documents, records, and forms",
    "QUALITY_OR_COMPLIANCE": "Quality, fraud, and compliance terms",
    "TECH_OR_SYSTEM": "Technology and system terms",
    "SOCIAL_SUPPORT_OR_COMMUNITY_NEED": "Social determinants and community needs",
    "LOCATION": "Geographic references",
    "ACRONYM": "Medicaid/healthcare acronyms",
}

# Each lexicon maps lowercase terms to their canonical form.
# Multi-word phrases are included and matched via phrase matching.

_LEXICON: dict[str, dict[str, str]] = {
    "PROGRAM_BRAND": {
        "medi-cal": "MEDI_CAL",
        "medi cal": "MEDI_CAL",
        "masshealth": "MASSHEALTH",
        "apple health": "APPLE_HEALTH",
        "soonercare": "SOONERCARE",
        "tenncare": "TENNCARE",
        "familycare": "FAMILYCARE",
        "healthnet": "HEALTHNET",
        "mainecare": "MAINECARE",
        "med-quest": "MED_QUEST",
        "medquest": "MED_QUEST",
        "kancare": "KANCARE",
        "hoosier healthwise": "HOOSIER_HEALTHWISE",
        "hoosier care connect": "HOOSIER_CARE_CONNECT",
        "health first colorado": "HEALTH_FIRST_COLORADO",
        "green mountain care": "GREEN_MOUNTAIN_CARE",
        "cardinal care": "CARDINAL_CARE",
        "healthy connections": "HEALTHY_CONNECTIONS",
        "turquoisecare": "TURQUOISECARE",
        "star plus": "STAR_PLUS",
        "star+plus": "STAR_PLUS",
        "plan first": "PLAN_FIRST",
        "forwardhealth": "FORWARDHEALTH",
        "denalicare": "DENALICARE",
        "husky": "HUSKY",
        "bayou health": "BAYOU_HEALTH",
        "mycare ohio": "MYCARE_OHIO",
        "oregon health plan": "OREGON_HEALTH_PLAN",
    },
    "PROGRAM_TYPE": {
        "medicaid": "MEDICAID",
        "chip": "CHIP",
        "managed care": "MANAGED_CARE",
        "fee-for-service": "FFS",
        "fee for service": "FFS",
        "ffs": "FFS",
        "ltss": "LTSS",
        "long-term services and supports": "LTSS",
        "long term services and supports": "LTSS",
        "hcbs": "HCBS",
        "home and community-based services": "HCBS",
        "home and community based services": "HCBS",
        "epsdt": "EPSDT",
        "early and periodic screening": "EPSDT",
        "early and periodic screening diagnostic and treatment": "EPSDT",
        "pace": "PACE",
        "program of all-inclusive care for the elderly": "PACE",
    },
    "AGENCY_OR_GOV_BODY": {
        "cms": "CMS",
        "centers for medicare and medicaid services": "CMS",
        "centers for medicare & medicaid services": "CMS",
        "macpac": "MACPAC",
        "medicaid and chip payment and access commission": "MACPAC",
        "state medicaid agency": "STATE_MEDICAID_AGENCY",
        "medicaid agency": "STATE_MEDICAID_AGENCY",
        "advisory council": "ADVISORY_COUNCIL",
        "beneficiary advisory council": "BENEFICIARY_ADVISORY_COUNCIL",
        "medicaid advisory committee": "MEDICAID_ADVISORY_COMMITTEE",
        "hhs": "HHS",
        "department of health and human services": "HHS",
        "oig": "OIG",
        "office of inspector general": "OIG",
        "gao": "GAO",
    },
    "ELIGIBILITY_GROUP": {
        "pregnant women": "PREGNANT_WOMEN",
        "pregnant": "PREGNANT_WOMEN",
        "children": "CHILDREN",
        "child": "CHILDREN",
        "infant": "INFANT",
        "elderly": "ELDERLY",
        "aged": "AGED",
        "senior": "AGED",
        "disabled": "DISABLED",
        "blind": "BLIND",
        "dual eligible": "DUAL_ELIGIBLE",
        "dual-eligible": "DUAL_ELIGIBLE",
        "dually eligible": "DUAL_ELIGIBLE",
        "medically needy": "MEDICALLY_NEEDY",
        "categorically needy": "CATEGORICALLY_NEEDY",
        "expansion adult": "EXPANSION_ADULT",
        "expansion adults": "EXPANSION_ADULT",
        "aged blind disabled": "ABD",
        "aged, blind, and disabled": "ABD",
        "abd": "ABD",
        "foster care": "FOSTER_CARE",
        "foster-care": "FOSTER_CARE",
        "parents and caretaker relatives": "PARENTS_CARETAKERS",
        "parent": "PARENTS_CARETAKERS",
        "caretaker": "PARENTS_CARETAKERS",
        "caretaker relative": "PARENTS_CARETAKERS",
        "refugee": "REFUGEE",
        "immigrant": "IMMIGRANT",
        "postpartum": "POSTPARTUM",
        "family": "FAMILY",
        "household": "HOUSEHOLD",
        "adult": "ADULT",
        "teen": "TEEN",
        "adolescent": "ADOLESCENT",
        "disability": "DISABLED",
    },
    "PERSON_ROLE": {
        "beneficiary": "BENEFICIARY",
        "enrollee": "ENROLLEE",
        "member": "MEMBER",
        "applicant": "APPLICANT",
        "caregiver": "CAREGIVER",
        "provider": "PROVIDER",
        "caseworker": "CASEWORKER",
        "navigator": "NAVIGATOR",
        "representative": "REPRESENTATIVE",
        "peer navigator": "PEER_NAVIGATOR",
        "community health worker": "COMMUNITY_HEALTH_WORKER",
        "case aide": "CASE_AIDE",
    },
    "FINANCIAL_TERM": {
        "income": "INCOME",
        "asset": "ASSET",
        "assets": "ASSET",
        "resource": "RESOURCE",
        "resources": "RESOURCE",
        "fpl": "FPL",
        "federal poverty level": "FPL",
        "federal poverty line": "FPL",
        "magi": "MAGI",
        "modified adjusted gross income": "MAGI",
        "premium": "PREMIUM",
        "copayment": "COPAYMENT",
        "copay": "COPAYMENT",
        "coinsurance": "COINSURANCE",
        "deductible": "DEDUCTIBLE",
        "reimbursement": "REIMBURSEMENT",
        "estate recovery": "ESTATE_RECOVERY",
        "spenddown": "SPENDDOWN",
        "spend-down": "SPENDDOWN",
        "spend down": "SPENDDOWN",
        "disregard": "DISREGARD",
        "cost-sharing": "COST_SHARING",
        "cost sharing": "COST_SHARING",
        "out-of-pocket": "OUT_OF_POCKET",
        "out of pocket": "OUT_OF_POCKET",
        "capitation": "CAPITATION",
        "reimbursement rate": "REIMBURSEMENT_RATE",
        "third-party liability": "TPL",
        "third party liability": "TPL",
        "tpl": "TPL",
        "subrogation": "SUBROGATION",
        "recoupment": "RECOUPMENT",
        "overpayment": "OVERPAYMENT",
        "premium assistance": "PREMIUM_ASSISTANCE",
        "earnings": "EARNINGS",
        "wages": "WAGES",
        "self-employment": "SELF_EMPLOYMENT",
        "household size": "HOUSEHOLD_SIZE",
        "household-size": "HOUSEHOLD_SIZE",
        "limit": "LIMIT",
        "threshold": "THRESHOLD",
    },
    "APPLICATION_PROCESS": {
        "application": "APPLICATION",
        "renewal": "RENEWAL",
        "redetermination": "REDETERMINATION",
        "recertification": "RECERTIFICATION",
        "verification": "VERIFICATION",
        "attestation": "ATTESTATION",
        "denial": "DENIAL",
        "approval": "APPROVAL",
        "hearing": "HEARING",
        "appeal": "APPEAL",
        "screening": "SCREENING",
        "determination": "DETERMINATION",
        "ex parte": "EX_PARTE",
        "ex-parte": "EX_PARTE",
        "pending": "PENDING",
        "outreach": "OUTREACH",
        "notice": "NOTICE",
        "fair hearing": "FAIR_HEARING",
        "eligibility review": "ELIGIBILITY_REVIEW",
    },
    "BENEFIT_OR_SERVICE": {
        "dental": "DENTAL",
        "vision": "VISION",
        "hearing": "HEARING_SERVICE",
        "lab": "LAB",
        "laboratory": "LAB",
        "radiology": "RADIOLOGY",
        "behavioral health": "BEHAVIORAL_HEALTH",
        "mental health": "MENTAL_HEALTH",
        "pharmacy": "PHARMACY",
        "transportation": "TRANSPORTATION",
        "non-emergency medical transportation": "NEMT",
        "nemt": "NEMT",
        "hospice": "HOSPICE",
        "home health": "HOME_HEALTH",
        "home-health": "HOME_HEALTH",
        "personal care": "PERSONAL_CARE",
        "personal-care": "PERSONAL_CARE",
        "case management": "CASE_MANAGEMENT",
        "case-management": "CASE_MANAGEMENT",
        "rehabilitation": "REHABILITATION",
        "therapy": "THERAPY",
        "surgery": "SURGERY",
        "inpatient": "INPATIENT",
        "outpatient": "OUTPATIENT",
        "preventive": "PREVENTIVE",
        "preventive care": "PREVENTIVE",
        "immunization": "IMMUNIZATION",
        "prescription": "PRESCRIPTION",
        "formulary": "FORMULARY",
        "durable medical equipment": "DME",
        "dme": "DME",
        "telemedicine": "TELEMEDICINE",
        "telehealth": "TELEMEDICINE",
        "dialysis": "DIALYSIS",
        "substance use disorder": "SUD",
        "substance-use disorder": "SUD",
        "sud": "SUD",
        "opioid use disorder": "OUD",
        "mat": "MAT",
        "medication-assisted treatment": "MAT",
        "medication assisted treatment": "MAT",
        "respite": "RESPITE",
        "respite care": "RESPITE",
        "adult day care": "ADULT_DAY_CARE",
        "skilled nursing": "SKILLED_NURSING",
        "private duty nursing": "PRIVATE_DUTY_NURSING",
        "maternity": "MATERNITY",
        "prenatal": "PRENATAL",
        "postpartum care": "POSTPARTUM_CARE",
        "family planning": "FAMILY_PLANNING",
        "well-child": "WELL_CHILD",
        "well child": "WELL_CHILD",
        "early intervention": "EARLY_INTERVENTION",
        "lead screening": "LEAD_SCREENING",
        "developmental screening": "DEVELOPMENTAL_SCREENING",
        "counseling": "COUNSELING",
        "crisis": "CRISIS",
        "peer support": "PEER_SUPPORT",
        "psychiatric": "PSYCHIATRIC",
        "step therapy": "STEP_THERAPY",
        "step-therapy": "STEP_THERAPY",
        "prior authorization": "PRIOR_AUTH",
        "prior auth": "PRIOR_AUTH",
        "drug utilization review": "DUR",
        "dur": "DUR",
    },
    "CARE_SETTING": {
        "nursing home": "NURSING_HOME",
        "nursing facility": "NURSING_HOME",
        "outpatient clinic": "OUTPATIENT_CLINIC",
        "clinic": "CLINIC",
        "assisted living": "ASSISTED_LIVING",
        "assisted-living": "ASSISTED_LIVING",
        "hospital": "HOSPITAL",
        "inpatient facility": "INPATIENT_FACILITY",
        "community-based": "COMMUNITY_BASED",
        "community based": "COMMUNITY_BASED",
        "psychiatric facility": "PSYCHIATRIC_FACILITY",
        "residential treatment": "RESIDENTIAL_TREATMENT",
        "ambulatory": "AMBULATORY",
        "surgical center": "SURGICAL_CENTER",
        "treatment facility": "TREATMENT_FACILITY",
        "independent living": "INDEPENDENT_LIVING",
    },
    "PROVIDER_TYPE": {
        "hospital": "HOSPITAL",
        "fqhc": "FQHC",
        "federally qualified health center": "FQHC",
        "rhc": "RHC",
        "rural health clinic": "RHC",
        "physician": "PHYSICIAN",
        "pcp": "PCP",
        "primary care provider": "PCP",
        "primary care physician": "PCP",
        "specialist": "SPECIALIST",
        "dentist": "DENTIST",
        "pharmacist": "PHARMACIST",
        "therapist": "THERAPIST",
        "nurse": "NURSE",
        "obgyn": "OBGYN",
        "ob/gyn": "OBGYN",
        "pediatrician": "PEDIATRICIAN",
        "psychiatrist": "PSYCHIATRIST",
        "ambulance": "AMBULANCE",
        "transportation provider": "TRANSPORTATION_PROVIDER",
        "lab provider": "LAB_PROVIDER",
        "rendering provider": "RENDERING_PROVIDER",
        "referring provider": "REFERRING_PROVIDER",
        "attending physician": "ATTENDING_PHYSICIAN",
    },
    "PAYMENT_OR_DELIVERY_MODEL": {
        "capitation": "CAPITATION",
        "fee-for-service": "FFS",
        "fee for service": "FFS",
        "ffs": "FFS",
        "mco": "MCO",
        "managed care organization": "MCO",
        "mmc": "MMC",
        "mltss": "MLTSS",
        "managed long-term services and supports": "MLTSS",
        "managed long term services and supports": "MLTSS",
        "d-snp": "D_SNP",
        "dsnp": "D_SNP",
        "dual special needs plan": "D_SNP",
        "hide-snp": "HIDE_SNP",
        "fide-snp": "FIDE_SNP",
        "aco": "ACO",
        "accountable care organization": "ACO",
        "value-based care": "VBC",
        "value based care": "VBC",
        "care coordination": "CARE_COORDINATION",
        "care-coordination": "CARE_COORDINATION",
        "care integration": "CARE_INTEGRATION",
        "risk-based contract": "RISK_BASED_CONTRACT",
        "encounter": "ENCOUNTER",
        "attribution": "ATTRIBUTION",
        "patient-centered": "PATIENT_CENTERED",
        "utilization review": "UTILIZATION_REVIEW",
        "utilization-review": "UTILIZATION_REVIEW",
        "network adequacy": "NETWORK_ADEQUACY",
        "continuity of care": "CONTINUITY_OF_CARE",
        "carve-out": "CARVE_OUT",
        "carve out": "CARVE_OUT",
    },
    "WAIVER_OR_AUTHORITY": {
        "1115 waiver": "WAIVER_1115",
        "1115 demonstration": "WAIVER_1115",
        "section 1115": "WAIVER_1115",
        "1115": "WAIVER_1115",
        "1915(c) waiver": "WAIVER_1915C",
        "1915(c)": "WAIVER_1915C",
        "1915c waiver": "WAIVER_1915C",
        "1915(b) waiver": "WAIVER_1915B",
        "1915(b)": "WAIVER_1915B",
        "1915b waiver": "WAIVER_1915B",
        "1915(i) option": "OPTION_1915I",
        "1915(i)": "OPTION_1915I",
        "1915(k) option": "OPTION_1915K",
        "1915(k)": "OPTION_1915K",
        "state plan amendment": "SPA",
        "spa": "SPA",
        "state plan": "STATE_PLAN",
        "demonstration": "DEMONSTRATION",
        "waiver renewal": "WAIVER_RENEWAL",
        "federal register": "FEDERAL_REGISTER",
        "cfr": "CFR",
        "aca": "ACA",
        "affordable care act": "ACA",
        "mandatory benefit": "MANDATORY_BENEFIT",
        "optional benefit": "OPTIONAL_BENEFIT",
    },
    "DOCUMENT_OR_RECORD": {
        "form": "FORM",
        "notice": "NOTICE",
        "handbook": "HANDBOOK",
        "billing manual": "BILLING_MANUAL",
        "fee schedule": "FEE_SCHEDULE",
        "fee-schedule": "FEE_SCHEDULE",
        "ehr": "EHR",
        "electronic health record": "EHR",
        "health information exchange": "HIE",
        "hie": "HIE",
        "release of information": "ROI",
        "phi": "PHI",
        "protected health information": "PHI",
        "hipaa": "HIPAA",
        "consent": "CONSENT",
        "authorization": "AUTHORIZATION",
        "forms library": "FORMS_LIBRARY",
        "provider directory": "PROVIDER_DIRECTORY",
        "manual": "MANUAL",
        "publication": "PUBLICATION",
    },
    "QUALITY_OR_COMPLIANCE": {
        "fraud": "FRAUD",
        "abuse": "ABUSE",
        "audit": "AUDIT",
        "grievance": "GRIEVANCE",
        "complaint": "COMPLAINT",
        "quality measure": "QUALITY_MEASURE",
        "quality-measure": "QUALITY_MEASURE",
        "core set": "CORE_SET",
        "core-set": "CORE_SET",
        "child core set": "CHILD_CORE_SET",
        "adult core set": "ADULT_CORE_SET",
        "program integrity": "PROGRAM_INTEGRITY",
        "program-integrity": "PROGRAM_INTEGRITY",
        "sanction": "SANCTION",
        "suspension": "SUSPENSION",
        "oversight": "OVERSIGHT",
        "monitoring": "MONITORING",
        "performance measure": "PERFORMANCE_MEASURE",
        "benchmark": "BENCHMARK",
        "star rating": "STAR_RATING",
        "rac": "RAC",
    },
    "TECH_OR_SYSTEM": {
        "portal": "PORTAL",
        "avrs": "AVRS",
        "interoperability": "INTEROPERABILITY",
        "api": "API",
        "dashboard": "DASHBOARD",
        "claims system": "CLAIMS_SYSTEM",
        "eligibility system": "ELIGIBILITY_SYSTEM",
        "reporting tool": "REPORTING_TOOL",
        "analytics": "ANALYTICS",
        "e-signature": "E_SIGNATURE",
        "electronic signature": "E_SIGNATURE",
        "secure portal": "SECURE_PORTAL",
        "multi-factor authentication": "MFA",
        "mfa": "MFA",
        "workflow": "WORKFLOW",
        "notification": "NOTIFICATION",
    },
    "SOCIAL_SUPPORT_OR_COMMUNITY_NEED": {
        "housing": "HOUSING",
        "food insecurity": "FOOD_INSECURITY",
        "food-insecurity": "FOOD_INSECURITY",
        "transportation": "TRANSPORTATION",
        "caregiver support": "CAREGIVER_SUPPORT",
        "community partner": "COMMUNITY_PARTNER",
        "social drivers of health": "SDOH",
        "social determinants of health": "SDOH",
        "sdoh": "SDOH",
        "hrsn": "HRSN",
        "health-related social needs": "HRSN",
        "health related social needs": "HRSN",
        "employment support": "EMPLOYMENT_SUPPORT",
        "education support": "EDUCATION_SUPPORT",
        "home modification": "HOME_MODIFICATION",
        "homemaker": "HOMEMAKER",
        "meal support": "MEAL_SUPPORT",
        "utility support": "UTILITY_SUPPORT",
        "legal aid": "LEGAL_AID",
        "wellness": "WELLNESS",
        "cbo": "CBO",
        "community-based organization": "CBO",
        "community based organization": "CBO",
    },
    "LOCATION": {
        "state": "STATE",
        "county": "COUNTY",
        "office": "OFFICE",
        "service region": "SERVICE_REGION",
        "service area": "SERVICE_AREA",
        "call center": "CALL_CENTER",
        "call-center": "CALL_CENTER",
    },
    "ACRONYM": {
        "chip": "CHIP",
        "cms": "CMS",
        "hcbs": "HCBS",
        "ltss": "LTSS",
        "mco": "MCO",
        "mltss": "MLTSS",
        "fpl": "FPL",
        "magi": "MAGI",
        "epsdt": "EPSDT",
        "aca": "ACA",
        "spa": "SPA",
        "ehr": "EHR",
        "hie": "HIE",
        "hipaa": "HIPAA",
        "phi": "PHI",
        "nfp": "NFP",
        "rac": "RAC",
        "ndc": "NDC",
        "icn": "ICN",
        "dme": "DME",
        "nemt": "NEMT",
        "sud": "SUD",
        "oud": "OUD",
        "mat": "MAT",
        "dur": "DUR",
        "abd": "ABD",
        "ffs": "FFS",
        "tpl": "TPL",
        "vbc": "VBC",
        "sdoh": "SDOH",
        "hrsn": "HRSN",
        "avrs": "AVRS",
        "mfa": "MFA",
        "cbo": "CBO",
        "d-snp": "D_SNP",
        "dsnp": "D_SNP",
        "fide-snp": "FIDE_SNP",
        "hide-snp": "HIDE_SNP",
        "fqhc": "FQHC",
        "rhc": "RHC",
        "pcp": "PCP",
    },
}

# ---------------------------------------------------------------------------
# Normalization aliases — maps surface forms to (canonical_name, category)
# ---------------------------------------------------------------------------

_NORMALIZATION: dict[str, dict[str, Any]] = {
    # Eligibility group normalization
    "elderly & disabled": {"canonical": "ABD", "label": "ELIGIBILITY_GROUP"},
    "elderly and disabled": {"canonical": "ABD", "label": "ELIGIBILITY_GROUP"},
    "aged blind disabled": {"canonical": "ABD", "label": "ELIGIBILITY_GROUP"},
    "aged, blind, and disabled": {"canonical": "ABD", "label": "ELIGIBILITY_GROUP"},
    "abd": {"canonical": "ABD", "label": "ELIGIBILITY_GROUP"},
    # Waiver normalization
    "1115": {"canonical": "WAIVER_1115", "label": "WAIVER_OR_AUTHORITY"},
    "section 1115": {"canonical": "WAIVER_1115", "label": "WAIVER_OR_AUTHORITY"},
    "1115 demonstration": {"canonical": "WAIVER_1115", "label": "WAIVER_OR_AUTHORITY"},
    "1115 waiver": {"canonical": "WAIVER_1115", "label": "WAIVER_OR_AUTHORITY"},
    "1915(c)": {"canonical": "WAIVER_1915C", "label": "WAIVER_OR_AUTHORITY"},
    "1915c": {"canonical": "WAIVER_1915C", "label": "WAIVER_OR_AUTHORITY"},
    "1915(b)": {"canonical": "WAIVER_1915B", "label": "WAIVER_OR_AUTHORITY"},
    "1915b": {"canonical": "WAIVER_1915B", "label": "WAIVER_OR_AUTHORITY"},
    # Payment model normalization
    "fee for service": {"canonical": "FFS", "label": "PAYMENT_OR_DELIVERY_MODEL"},
    "fee-for-service": {"canonical": "FFS", "label": "PAYMENT_OR_DELIVERY_MODEL"},
    "ffs": {"canonical": "FFS", "label": "PAYMENT_OR_DELIVERY_MODEL"},
    # Provider normalization
    "federally qualified health center": {"canonical": "FQHC", "label": "PROVIDER_TYPE"},
    "fqhc": {"canonical": "FQHC", "label": "PROVIDER_TYPE"},
    "rural health clinic": {"canonical": "RHC", "label": "PROVIDER_TYPE"},
    "rhc": {"canonical": "RHC", "label": "PROVIDER_TYPE"},
    # Application process normalization
    "renewal": {"canonical": "ELIGIBILITY_REVIEW", "label": "APPLICATION_PROCESS"},
    "redetermination": {"canonical": "ELIGIBILITY_REVIEW", "label": "APPLICATION_PROCESS"},
    "recertification": {"canonical": "ELIGIBILITY_REVIEW", "label": "APPLICATION_PROCESS"},
    # Program
    "centers for medicare and medicaid services": {"canonical": "CMS", "label": "AGENCY_OR_GOV_BODY"},
    "centers for medicare & medicaid services": {"canonical": "CMS", "label": "AGENCY_OR_GOV_BODY"},
    # LTSS/HCBS
    "home and community-based services": {"canonical": "HCBS", "label": "PROGRAM_TYPE"},
    "home and community based services": {"canonical": "HCBS", "label": "PROGRAM_TYPE"},
    "long-term services and supports": {"canonical": "LTSS", "label": "PROGRAM_TYPE"},
    "long term services and supports": {"canonical": "LTSS", "label": "PROGRAM_TYPE"},
    "managed long-term services and supports": {"canonical": "MLTSS", "label": "PAYMENT_OR_DELIVERY_MODEL"},
    "managed long term services and supports": {"canonical": "MLTSS", "label": "PAYMENT_OR_DELIVERY_MODEL"},
}

# ---------------------------------------------------------------------------
# Precomputed structures for fast matching
# ---------------------------------------------------------------------------

# Build a single flat lookup: lowercase phrase → (label, canonical)
_PHRASE_LOOKUP: dict[str, tuple[str, str]] = {}
for _label, _terms in _LEXICON.items():
    for _phrase, _canonical in _terms.items():
        _PHRASE_LOOKUP[_phrase.lower()] = (_label, _canonical)

# Sort phrases by length descending so longer matches take priority
_SORTED_PHRASES: list[str] = sorted(_PHRASE_LOOKUP.keys(), key=len, reverse=True)

# Precompile regex patterns for each phrase (word-boundary matching)
# Only compile patterns for phrases >= 3 chars to avoid noise
_PHRASE_PATTERNS: list[tuple[str, re.Pattern[str]]] = []
for _phrase in _SORTED_PHRASES:
    if len(_phrase) < 2:
        continue
    # For acronyms (all uppercase canonical), use case-sensitive matching
    _label, _canonical = _PHRASE_LOOKUP[_phrase]
    if _phrase == _phrase.upper() and len(_phrase) <= 6:
        # Case-sensitive for short acronyms
        _pat = re.compile(r"\b" + re.escape(_phrase.upper()) + r"\b")
    else:
        _pat = re.compile(r"\b" + re.escape(_phrase) + r"\b", re.IGNORECASE)
    _PHRASE_PATTERNS.append((_phrase, _pat))


# ---------------------------------------------------------------------------
# spaCy integration (lazy-loaded)
# ---------------------------------------------------------------------------

_nlp = None

def _get_spacy_nlp():
    """Lazy-load spaCy model for multi-token entity detection."""
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except Exception:
        return None


# Map spaCy labels to our Medicaid labels where appropriate
_SPACY_LABEL_MAP: dict[str, str] = {
    "ORG": "AGENCY_OR_GOV_BODY",
    "GPE": "LOCATION",
    "PERSON": "PERSON_ROLE",
    "LAW": "WAIVER_OR_AUTHORITY",
    "MONEY": "FINANCIAL_TERM",
}


# ---------------------------------------------------------------------------
# GLiNER integration (lazy-loaded)
# ---------------------------------------------------------------------------

_gliner_model = None
_GLINER_AVAILABLE = None  # None = not checked yet

# GLiNER label names (human-readable for the zero-shot model)
_GLINER_LABELS: list[str] = [
    "program brand",
    "program type",
    "government agency",
    "eligibility group",
    "person role",
    "financial term",
    "application process",
    "benefit or service",
    "care setting",
    "provider type",
    "payment model",
    "waiver or authority",
    "document or record",
    "quality or compliance",
    "technology system",
    "social support need",
]

# Map GLiNER label names back to our canonical label names
_GLINER_LABEL_MAP: dict[str, str] = {
    "program brand": "PROGRAM_BRAND",
    "program type": "PROGRAM_TYPE",
    "government agency": "AGENCY_OR_GOV_BODY",
    "eligibility group": "ELIGIBILITY_GROUP",
    "person role": "PERSON_ROLE",
    "financial term": "FINANCIAL_TERM",
    "application process": "APPLICATION_PROCESS",
    "benefit or service": "BENEFIT_OR_SERVICE",
    "care setting": "CARE_SETTING",
    "provider type": "PROVIDER_TYPE",
    "payment model": "PAYMENT_OR_DELIVERY_MODEL",
    "waiver or authority": "WAIVER_OR_AUTHORITY",
    "document or record": "DOCUMENT_OR_RECORD",
    "quality or compliance": "QUALITY_OR_COMPLIANCE",
    "technology system": "TECH_OR_SYSTEM",
    "social support need": "SOCIAL_SUPPORT_OR_COMMUNITY_NEED",
}


def _get_gliner_model():
    """Lazy-load GLiNER model for zero-shot NER."""
    global _gliner_model, _GLINER_AVAILABLE
    if _GLINER_AVAILABLE is False:
        return None
    if _gliner_model is not None:
        return _gliner_model
    try:
        from gliner import GLiNER
        _gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        _GLINER_AVAILABLE = True
        return _gliner_model
    except Exception:
        _GLINER_AVAILABLE = False
        return None


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def _dictionary_extract(text: str) -> list[dict[str, Any]]:
    """Layer 1: Dictionary/lexicon-based extraction.

    Matches all phrases from the seed lexicon against the input text,
    longest-first to avoid partial matches.
    """
    entities: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []

    for phrase, pattern in _PHRASE_PATTERNS:
        label, canonical = _PHRASE_LOOKUP[phrase]

        for m in pattern.finditer(text):
            span = (m.start(), m.end())
            # Skip if overlaps with a previously matched (longer) span
            if any(span[0] < u[1] and u[0] < span[1] for u in used_spans):
                continue

            matched_text = m.group(0)

            # Look up normalization
            norm = _NORMALIZATION.get(matched_text.lower())
            canonical_name = norm["canonical"] if norm else canonical
            norm_label = norm["label"] if norm else label

            # Determine normalized acronym
            normalized_acronym = ""
            if canonical_name == canonical_name.upper() and len(canonical_name) <= 8:
                normalized_acronym = canonical_name
            elif norm and norm["canonical"] == norm["canonical"].upper():
                normalized_acronym = norm["canonical"]

            entities.append({
                "text": matched_text,
                "label": norm_label,
                "canonical_name": canonical_name,
                "start": span[0],
                "end": span[1],
                "confidence": 0.95,  # High confidence for dictionary matches
                "source": "dictionary",
                "normalized_acronym": normalized_acronym,
            })
            used_spans.append(span)

    return entities


def _spacy_extract(text: str) -> list[dict[str, Any]]:
    """Layer 2: spaCy NER for multi-token entities the dictionary misses.

    Catches entities like 'parents and caretaker relatives', org names, etc.
    """
    nlp = _get_spacy_nlp()
    if nlp is None:
        return []

    # Truncate very long texts for spaCy performance
    max_len = 100_000
    proc_text = text[:max_len] if len(text) > max_len else text

    doc = nlp(proc_text)
    entities: list[dict[str, Any]] = []

    for ent in doc.ents:
        mapped_label = _SPACY_LABEL_MAP.get(ent.label_)
        if not mapped_label:
            continue

        # Skip very short or very long entities
        if len(ent.text.strip()) < 2 or len(ent.text) > 100:
            continue

        entities.append({
            "text": ent.text.strip(),
            "label": mapped_label,
            "canonical_name": ent.text.strip().upper().replace(" ", "_")[:40],
            "start": ent.start_char,
            "end": ent.end_char,
            "confidence": 0.70,  # Lower confidence for statistical NER
            "source": "spacy",
            "normalized_acronym": "",
        })

    return entities


def _gliner_extract(text: str, threshold: float = 0.4) -> list[dict[str, Any]]:
    """Layer 3: GLiNER zero-shot NER for domain-specific entities.

    Uses a pretrained transformer model to find entities that match
    our Medicaid label descriptions without any fine-tuning.
    Chunks long text to stay within model limits.
    """
    model = _get_gliner_model()
    if model is None:
        return []

    entities: list[dict[str, Any]] = []

    # GLiNER has a max token limit (~512 tokens ≈ ~2000 chars for safety)
    # Chunk text with overlap to avoid splitting entities
    chunk_size = 1500
    overlap = 200
    chunks: list[tuple[int, str]] = []  # (offset, chunk_text)

    if len(text) <= chunk_size:
        chunks.append((0, text))
    else:
        pos = 0
        while pos < len(text):
            end = min(pos + chunk_size, len(text))
            chunks.append((pos, text[pos:end]))
            if end >= len(text):
                break
            pos += chunk_size - overlap

    for offset, chunk in chunks:
        try:
            preds = model.predict_entities(chunk, _GLINER_LABELS, threshold=threshold)
        except Exception:
            continue

        for pred in preds:
            mapped_label = _GLINER_LABEL_MAP.get(pred["label"])
            if not mapped_label:
                continue

            ent_text = pred["text"].strip()
            if len(ent_text) < 2 or len(ent_text) > 100:
                continue

            # Calculate absolute positions
            # GLiNER returns text but not always char offsets — find in chunk
            start_in_chunk = chunk.find(ent_text)
            if start_in_chunk == -1:
                start_in_chunk = chunk.lower().find(ent_text.lower())
            if start_in_chunk == -1:
                continue

            abs_start = offset + start_in_chunk
            abs_end = abs_start + len(ent_text)

            # Try to normalize via our dictionary/normalization tables
            norm = _NORMALIZATION.get(ent_text.lower())
            lookup = _PHRASE_LOOKUP.get(ent_text.lower())
            if norm:
                canonical = norm["canonical"]
                mapped_label = norm["label"]
            elif lookup:
                mapped_label, canonical = lookup
            else:
                canonical = ent_text.upper().replace(" ", "_")[:40]

            normalized_acronym = ""
            if canonical == canonical.upper() and len(canonical) <= 8:
                normalized_acronym = canonical

            entities.append({
                "text": ent_text,
                "label": mapped_label,
                "canonical_name": canonical,
                "start": abs_start,
                "end": abs_end,
                "confidence": round(pred["score"], 4),
                "source": "gliner",
                "normalized_acronym": normalized_acronym,
            })

    return entities


def _resolve_overlaps(
    entities: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve overlapping spans.

    Priority: dictionary > spacy > gliner, then longer > shorter.
    Special handling: if a long span contains multiple entity types,
    split it (e.g., 'Home and Community-Based Services Waiver').
    """
    # Sort: dictionary first, then by span length descending
    source_priority = {"dictionary": 0, "spacy": 1, "gliner": 2}
    entities.sort(key=lambda e: (
        source_priority.get(e["source"], 2),
        -(e["end"] - e["start"]),
    ))

    kept: list[dict[str, Any]] = []
    used_spans: list[tuple[int, int]] = []

    for ent in entities:
        span = (ent["start"], ent["end"])
        if any(span[0] < u[1] and u[0] < span[1] for u in used_spans):
            continue
        kept.append(ent)
        used_spans.append(span)

    kept.sort(key=lambda e: e["start"])
    return kept


def _categorize_entities(
    entities: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group entities by label type."""
    categories: dict[str, list[dict[str, Any]]] = {
        label: [] for label in ENTITY_LABELS
    }
    for ent in entities:
        label = ent["label"]
        if label in categories:
            categories[label].append(ent)
    return categories


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_medicaid_entities(text: str) -> dict[str, Any]:
    """Extract Medicaid-domain entities from text using four-layer NER.

    Layers:
        1. Dictionary — seed lexicon of 500+ terms across 18 entity types
        2. spaCy — statistical NER for multi-token entities
        3. GLiNER — zero-shot transformer NER for domain-specific entities
        4. Normalization — maps variants to canonical labels

    Returns:
        {
            "entities": [
                {
                    "text": "Home and Community-Based Services",
                    "label": "BENEFIT_OR_SERVICE",
                    "canonical_name": "HCBS",
                    "confidence": 0.95,
                    "source": "dictionary",
                    "normalized_acronym": "HCBS",
                    "start": 10,
                    "end": 43,
                },
                ...
            ],
            "categories": {
                "PROGRAM_BRAND": [...],
                "ELIGIBILITY_GROUP": [...],
                ...
            },
            "summary": {
                "total_entities": <int>,
                "types_found": [<str>, ...],
                "unique_canonical": <int>,
            },
        }
    """
    if not text or not text.strip():
        return {
            "entities": [],
            "categories": {label: [] for label in ENTITY_LABELS},
            "summary": {
                "total_entities": 0,
                "types_found": [],
                "unique_canonical": 0,
            },
        }

    # Layer 1: Dictionary extraction (fastest, highest confidence)
    dict_entities = _dictionary_extract(text)

    # Layer 2: spaCy extraction (statistical NER)
    spacy_entities = _spacy_extract(text)

    # Layer 3: GLiNER zero-shot extraction (transformer-based)
    gliner_entities = _gliner_extract(text)

    # Merge and resolve overlaps (dictionary > spacy > gliner)
    all_entities = dict_entities + spacy_entities + gliner_entities
    resolved = _resolve_overlaps(all_entities)

    # Build categories
    categories = _categorize_entities(resolved)

    # Summary
    types_found = [label for label, ents in categories.items() if ents]
    unique_canonicals = set()
    for ent in resolved:
        if ent.get("canonical_name"):
            unique_canonicals.add(ent["canonical_name"])

    return {
        "entities": resolved,
        "categories": categories,
        "summary": {
            "total_entities": len(resolved),
            "types_found": types_found,
            "unique_canonical": len(unique_canonicals),
        },
    }


def normalize_entity(text: str) -> dict[str, Any] | None:
    """Look up a single entity text and return its normalized form.

    Returns None if no normalization exists.
    """
    key = text.lower().strip()
    if key in _NORMALIZATION:
        norm = _NORMALIZATION[key]
        return {
            "input": text,
            "canonical_name": norm["canonical"],
            "label": norm["label"],
        }
    if key in _PHRASE_LOOKUP:
        label, canonical = _PHRASE_LOOKUP[key]
        return {
            "input": text,
            "canonical_name": canonical,
            "label": label,
        }
    return None
