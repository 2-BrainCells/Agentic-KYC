"""
agents.py — All KYC agent functions.

Each agent receives the full KYCState and returns ONLY the fields it changed;
LangGraph merges the partial dict back into shared state.

Pipeline order: intake → data_extraction → id_verification → compliance →
(refine loop) → entity_resolution ‖ financial_profiling → risk_scoring →
hitl_review → active_learning_cache.

Requires: networkx, plus the tools.py dependencies (openai, qdrant, pytesseract).
"""

import difflib
import json
import os
import re
from datetime import datetime

import networkx as nx

from state import (
    KYCState,
    CaseStatus, ScreeningStatus, ExtractionStatus,
    Decision, Routing, DecisionSource, RiskBand, ActivityBand,
)
from config import (
    FUZZY_CLEAR_BELOW, FUZZY_AMBIGUOUS_LOW, FUZZY_AMBIGUOUS_HIGH,
    REFINEMENT_MAX_TRIES,
    RISK_AUTO_APPROVE_BELOW, RISK_REVIEW_BELOW,
    WEIGHT_ID_VERIFICATION, WEIGHT_COMPLIANCE,
    WEIGHT_NETWORK_RISK, WEIGHT_FINANCIAL,
    ID_PENALTY_NAME, ID_PENALTY_DOB, ID_PENALTY_PIN, ID_PENALTY_ADDRESS,
    NAME_MATCH_THRESHOLD, ADDRESS_MATCH_THRESHOLD,
    INCOME_BANDS, OCCUPATION_INCOME_MAP,
    TEMP_FIRST_PASS, TEMP_REFINE_PASS, TEMP_EXPLANATION,
    REG_RBI_KYC, REG_PMLA, REG_PAN_AADHAAR,
    WATCHLISTS, SALARY_MATCH_TOLERANCE,
)
from tools import (
    call_text_llm,
    call_llm_for_json,
    extract_text_from_image,
    parse_aadhaar_fields,
    parse_salary_slip,
    compute_field_confidence,
    query_sanctions_db,
    setup_sanctions_collection,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _now() -> str:
    """Current wall-clock time as HH:MM:SS, for audit-log timestamps."""
    return datetime.now().strftime("%H:%M:%S")


# ── Step-wise checklist helpers (explainability) ─────────────────────────────
# Each agent emits a structured checklist of the checks it ACTUALLY ran, so the
# UI can show the decision forming step by step. Status is one of:
#   "pass" → ✓   "fail" → ✗   "warn" → ⚠   "info" → •
# These are built from values the agent already computed — no extra LLM calls,
# fully deterministic.

def _step(label: str, status: str, detail: str = "") -> dict:
    """One checklist line: the check performed, its status, and the value seen."""
    return {"label": label, "status": status, "detail": detail}


def _checklist(agent: str, icon: str, steps: list, summary: str) -> dict:
    """Bundle one agent's checklist for the live UI and the audit view."""
    return {"agent": agent, "icon": icon, "steps": steps, "summary": summary}


def _name_similarity(name_a: str, name_b: str) -> float:
    """
    0.0–1.0 similarity between two names, case-insensitive and order-agnostic.

    Takes the best of: direct difflib ratio, token-overlap ratio, and a
    containment bonus (0.90) when every token of the shorter 2+-token name
    appears in the longer one — handles Indian names that drop a middle name
    ("Ramesh Mehta" vs "Ramesh Prakash Mehta").
    """
    a = name_a.lower().strip()
    b = name_b.lower().strip()

    direct = difflib.SequenceMatcher(None, a, b).ratio()

    parts_a = set(a.split())
    parts_b = set(b.split())
    if parts_a and parts_b:
        overlap   = len(parts_a & parts_b)
        component = overlap / max(len(parts_a), len(parts_b))
        smaller     = min(len(parts_a), len(parts_b))
        containment = 0.90 if (smaller >= 2 and overlap == smaller) else 0.0
    else:
        component = containment = 0.0

    return round(max(direct, component, containment), 3)


def _dob_year(dob: str) -> int:
    """Extract the birth year from a DOB string like '1968-11-14'. 0 if unknown."""
    m = re.search(r"\b(19|20)\d{2}\b", dob or "")
    return int(m.group()) if m else 0


def _normalize_dob(dob: str) -> str:
    """
    Normalise a DOB into a canonical 'YYYY-MM-DD' string so dates compare
    regardless of the input format. Handles the two formats in this app:
    'YYYY-MM-DD' (extraction output / internal) and 'DD-MM-YYYY' or 'DD/MM/YYYY'
    (Aadhaar-style UI input). Returns the cleaned original if it can't parse.
    """
    if not dob:
        return ""
    s = dob.strip().replace("/", "-").replace(".", "-").replace(" ", "")
    # Separator-less 8-digit form. Aadhaar prints DOB as DDMMYYYY.
    if re.fullmatch(r"\d{8}", s):
        d, mo, y = s[:2], s[2:4], s[4:]
        try:
            return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
        except ValueError:
            return s
    m = re.match(r"^(\d{1,4})-(\d{1,2})-(\d{1,4})$", s)
    if not m:
        return s
    a, b, c = m.groups()
    if len(a) == 4:                       # already YYYY-MM-DD
        y, mo, d = a, b, c
    else:                                 # DD-MM-YYYY → reorder
        d, mo, y = a, b, c
    try:
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    except ValueError:
        return s


def _parse_dob_range(dob_range: str) -> tuple:
    """Parse a watchlist DOB range '1966-1970' / '1966–1970' → (1966, 1970)."""
    years = re.findall(r"\b(?:19|20)\d{2}\b", dob_range or "")
    if not years:
        return (0, 0)
    return (int(years[0]), int(years[-1]))


# DOB gate: watchlist ranges are approximate, so allow ±3y before ruling out.
# Outside that window the match score is halved — clears same-name/different-
# generation false positives without a human.
DOB_GATE_TOLERANCE_YEARS = 3
DOB_GATE_SCORE_FACTOR    = 0.5

# Father's-name resolution: a differing father drops the score by ~65%; a
# matching father escalates it into the confirmed band.
FATHER_MATCH_THRESHOLD   = 0.80
FATHER_MISMATCH_FACTOR   = 0.35
FATHER_CONFIRM_SCORE     = 0.95


def _resolve_with_father(hit: dict, father: str) -> dict:
    """
    Self-correction payoff: compare the customer's father's name (found on the
    refinement pass) against the watchlist individual's listed father.

      differ → score ×0.35 → usually CLEAR
      match  → score ≥0.95 → CONFIRMED_HIT

    Mutates and returns the hit with a 'father_resolution' note. The loop both
    exonerates and convicts.
    """
    listed_father = hit.get("father_name")
    if not father or not listed_father:
        return hit

    sim = _name_similarity(father, listed_father)
    if sim >= FATHER_MATCH_THRESHOLD:
        hit["match_score"] = max(hit["match_score"], FATHER_CONFIRM_SCORE)
        hit["father_resolution"] = (
            f"Father's name MATCHES listed individual — "
            f"'{father}' ≈ '{listed_father}' (similarity {sim:.2f}). "
            f"Identity confirmed."
        )
    else:
        hit["match_score"] = round(hit["match_score"] * FATHER_MISMATCH_FACTOR, 3)
        hit["father_resolution"] = (
            f"Father's name DIFFERS from listed individual — "
            f"'{father}' vs '{listed_father}' (similarity {sim:.2f}). "
            f"Match score reduced by 65%."
        )
    return hit


def _get_income_band(income: float) -> str:
    """Map an INR income figure to an INCOME_BANDS label."""
    for band, (low, high) in INCOME_BANDS.items():
        if low <= income < high:
            return band
    return "hni"


def _estimate_bounding_boxes(image_path: str) -> dict:
    """
    Approximate per-field bounding boxes for the UI overlay, scaled to the
    image size and the typical UIDAI Aadhaar layout. Falls back to standard
    card proportions if the image can't be opened. Not true detection — fine
    for the demo overlay.
    """
    try:
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        w, h = img.size
    except Exception:
        w, h = 600, 380   # standard Aadhaar card proportions fallback

    return {
        "full_name_english":    {"x": int(w * 0.32), "y": int(h * 0.25), "w": int(w * 0.45), "h": int(h * 0.08)},
        "full_name_devanagari": {"x": int(w * 0.32), "y": int(h * 0.33), "w": int(w * 0.45), "h": int(h * 0.08)},
        "dob":                  {"x": int(w * 0.32), "y": int(h * 0.45), "w": int(w * 0.28), "h": int(h * 0.07)},
        "gender":               {"x": int(w * 0.62), "y": int(h * 0.45), "w": int(w * 0.10), "h": int(h * 0.07)},
        "aadhaar_number":       {"x": int(w * 0.18), "y": int(h * 0.75), "w": int(w * 0.45), "h": int(h * 0.09)},
        "address":              {"x": int(w * 0.18), "y": int(h * 0.55), "w": int(w * 0.60), "h": int(h * 0.18)},
        "father_name":          {"x": int(w * 0.18), "y": int(h * 0.50), "w": int(w * 0.55), "h": int(h * 0.07)},
    }


# ── Mock corporate graph (entity resolution) ────────────────────────────────
# In production this would query MCA21. The hardcoded graph covers the demo
# customers; the one sanctioned entity is Orion Global Trade Solutions LLP.

def _build_corporate_graph() -> nx.DiGraph:
    """Construct the in-memory MCA21 mock graph used by entity resolution."""
    G = nx.DiGraph()

    edges = [
        # Vikram Malhotra's chain (3 hops to the sanctioned entity)
        ("34 Golf Links New Delhi 110003",   "VMK Holdings Pvt Ltd",
         {"relationship": "registered_address_of"}),
        ("AAQPM6783R",                        "VMK Holdings Pvt Ltd",
         {"relationship": "director_via_pan"}),
        ("VMK Holdings Pvt Ltd",              "Sunrise Export Import Pvt Ltd",
         {"relationship": "holds_34pct_stake"}),
        ("Sunrise Export Import Pvt Ltd",     "Orion Global Trade Solutions LLP",
         {"relationship": "common_directors"}),
        ("Orion Global Trade Solutions LLP",  "SANCTIONED:ED_PMLA_2021_DEL_0892",
         {"relationship": "under_investigation"}),

        # Priya Sharma's clean chain
        ("BKXPS9876M",                        "Infosys Limited",
         {"relationship": "employer_via_pan"}),
        ("Infosys Limited",                   "CLEAN:NSE_BSE_LISTED",
         {"relationship": "listed_entity"}),
        ("560034",                            "Green Park Residents Welfare Association",
         {"relationship": "pin_code_association"}),
        ("Green Park Residents Welfare Association", "CLEAN:REGISTERED_SOCIETY",
         {"relationship": "registered_entity"}),
    ]

    for src, dst, data in edges:
        G.add_edge(src, dst, **data)

    # Flag Orion itself as sanctioned so the demo chain stays within 3 hops:
    # customer address → VMK Holdings → Sunrise Export Import → Orion.
    for node in ("Orion Global Trade Solutions LLP", "SANCTIONED:ED_PMLA_2021_DEL_0892"):
        G.nodes[node]["sanctioned"] = True
        G.nodes[node]["label"] = (
            "Orion Global Trade Solutions LLP — ED Investigation PMLA Case 2021/DEL/0892"
        )

    return G

CORPORATE_GRAPH = _build_corporate_graph()


# ── Agent 1: intake ─────────────────────────────────────────────────────────

def intake_agent(state: KYCState) -> KYCState:
    """Confirm receipt of the case and write the first audit-log entry."""
    cid     = state.get("customer_id", "UNKNOWN")
    income  = state.get("declared", {}).get("income", 0)
    name    = state.get("declared", {}).get("name", "UNKNOWN")

    steps = [
        _step("Case intake received", "pass", cid),
        _step("Applicant identified", "pass" if name != "UNKNOWN" else "warn", name),
        _step("Declared income captured", "pass" if income else "warn", f"₹{income:,.0f}"),
    ]

    return {
        "case_status": CaseStatus.RECEIVED,
        "agent_checklists": [_checklist(
            "Intake", "📥", steps, f"Case {cid} accepted for processing"
        )],
        "audit_log": [
            f"{_now()} [Intake] Case {cid} received — "
            f"Customer: {name}, Declared income: ₹{income:,.0f}"
        ],
    }


# ── Agent 2: data extraction ────────────────────────────────────────────────

def data_extraction_agent(state: KYCState) -> KYCState:
    """
    OCR the Aadhaar image, then have Llama-3 structure the text into identity
    fields. Runs twice when self-correcting: pass 0 standard (low temp), pass 1
    refinement (high temp, targets the father's name).

    The declared father's name is surfaced ONLY on the refinement pass — this
    simulates the deep QR 'care_of' parse and gives the loop something to find.
    Falls back to declared form data if the document can't be read.
    """
    refine_count = state.get("refine_count", 0)
    is_refinement = refine_count > 0
    documents     = state.get("documents", {})
    aadhaar_path  = documents.get("aadhaar_card", "")
    aadhaar_back  = documents.get("aadhaar_back", "")

    directive   = "refinement_pass" if is_refinement else "standard_first_pass"
    temperature = TEMP_REFINE_PASS if is_refinement else TEMP_FIRST_PASS
    pass_label  = f"Pass {refine_count} (refinement)" if is_refinement else "Pass 0 (standard)"

    # OCR the front, then append the back (carries address + QR 'care_of').
    ocr_text  = extract_text_from_image(aadhaar_path)
    back_text = extract_text_from_image(aadhaar_back) if aadhaar_back else ""
    used_back = bool(back_text.strip())
    if used_back:
        ocr_text = (ocr_text + "\n" + back_text).strip()

    # Structure the OCR text; skip the LLM if there is nothing to read.
    if ocr_text.strip():
        extracted = parse_aadhaar_fields(ocr_text, directive, temperature)
    else:
        extracted = {}
    declared_fallback = False

    # Declared-data fallback (mock mode): mirror the form so the rest of the
    # pipeline still runs real logic when no readable document exists.
    if not extracted.get("full_name_english"):
        declared = state.get("declared", {})
        extracted = {
            **extracted,
            "full_name_english": declared.get("name"),
            "dob":               declared.get("dob"),
            "address":           declared.get("address"),
            "pin_code":          declared.get("pin_code"),
            "aadhaar_last4":     extracted.get("aadhaar_last4") or "0000",
            "uidai_status":      "ACTIVE",
        }
        declared_fallback = True

    # Father's name only surfaces on the refinement pass (see docstring).
    # The declared form value represents the deep QR 'care_of' parse and is
    # AUTHORITATIVE on refinement: it overrides any partial/garbled OCR read
    # (e.g. Tesseract catching only the Devanagari surname "मेहता"), which would
    # otherwise score 0.00 against a Latin listed father and wrongly CLEAR a hit.
    if is_refinement:
        declared_father = state.get("declared", {}).get("father_name")
        if declared_father:
            extracted["father_name"] = declared_father

    extracted["extraction_pass"] = refine_count

    confidence = compute_field_confidence(extracted)
    low_conf   = [f for f, s in confidence.items() if s < 0.75 and extracted.get(f) is None]

    boxes = _estimate_bounding_boxes(aadhaar_path)

    required = ["full_name_english", "dob", "aadhaar_last4"]
    all_found = all(extracted.get(f) for f in required)
    status = ExtractionStatus.COMPLETE if all_found else ExtractionStatus.PARTIAL

    name_found    = extracted.get("full_name_english", "NOT FOUND")
    father_found  = extracted.get("father_name", None)
    father_note   = f"father_name: '{father_found}'" if father_found else "father_name: not found"
    if used_back:
        father_note += " (Aadhaar back side OCR'd)"
    if declared_fallback:
        father_note += " (declared-data fallback — document not readable)"

    steps = [
        _step(f"OCR Aadhaar front ({pass_label})",
              "pass" if ocr_text.strip() and not declared_fallback else "warn",
              "text read" if ocr_text.strip() else "no readable text"),
        _step("OCR Aadhaar back (address / C-O)",
              "pass" if used_back else "info",
              "back side read" if used_back else "no back side provided"),
        _step("Name extracted",
              "pass" if extracted.get("full_name_english") else "fail",
              str(name_found)),
        _step("Date of birth extracted",
              "pass" if extracted.get("dob") else "warn",
              str(extracted.get("dob") or "not found")),
        _step("Address extracted",
              "pass" if extracted.get("address") else "warn",
              str(extracted.get("address") or "not found")),
        _step("Father / husband name (C/O)",
              "pass" if father_found else "info",
              str(father_found) if father_found
              else ("deferred to refinement" if not is_refinement else "not on card")),
    ]
    if declared_fallback:
        steps.append(_step("Declared-data fallback used", "warn",
                           "document not readable — mirrored form data"))

    return {
        "extracted":             extracted,
        "field_confidence":      confidence,
        "bounding_boxes":        boxes,
        "low_confidence_fields": low_conf,
        "extraction_status":     status,
        "case_status":           CaseStatus.EXTRACTING,
        "agent_checklists": [_checklist(
            "Data Extraction", "📄", steps,
            f"Extraction {status} — {pass_label}"
        )],
        "audit_log": [
            f"{_now()} [Extraction] {pass_label} — status: {status}, "
            f"name: '{name_found}', {father_note}, "
            f"low-confidence fields: {low_conf or 'none'}"
        ],
    }


# ── Agent 3: ID verification ────────────────────────────────────────────────

def id_verification_agent(state: KYCState) -> KYCState:
    """
    Cross-check extracted Aadhaar fields against declared data: fuzzy name
    match, DOB (format-agnostic), PIN, address, UIDAI active status, and
    PAN-Aadhaar linkage (mocked True).

    Two outputs:
      • id_verified (bool) — the HARD gate. True only when name, DOB, PIN,
        address and UIDAI all pass. A False here auto-rejects the case (the
        document does not match the declaration → re-apply).
      • id_risk_score (0.0–1.0) — a GRADED contribution stored in
        verification_details, so the headline risk score moves smoothly with
        match quality instead of jumping 0.24 the instant one check flips.
    """
    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})

    # Name — graded similarity, gated at NAME_MATCH_THRESHOLD.
    ext_name  = extracted.get("full_name_english", "") or ""
    dec_name  = declared.get("name", "") or ""
    name_score = _name_similarity(ext_name, dec_name)
    name_match = name_score >= NAME_MATCH_THRESHOLD

    # DOB — normalise both sides so YYYY-MM-DD vs DD-MM-YYYY still compares.
    ext_dob = _normalize_dob(extracted.get("dob") or "")
    dec_dob = _normalize_dob(declared.get("dob") or "")
    dob_present = bool(ext_dob and dec_dob)
    dob_match = (ext_dob == dec_dob) if dob_present else False

    # PIN — only checked when both sides are present.
    ext_pin = str(extracted.get("pin_code") or "").strip()
    dec_pin = str(declared.get("pin_code") or "").strip()
    pin_present = bool(ext_pin and dec_pin)
    pin_match = (ext_pin == dec_pin) if pin_present else True

    # Address — graded similarity, gated at ADDRESS_MATCH_THRESHOLD (hard gate).
    ext_addr = extracted.get("address", "") or ""
    dec_addr = declared.get("address", "") or ""
    addr_present = bool(ext_addr.strip() and dec_addr.strip())
    addr_score = _name_similarity(ext_addr, dec_addr) if addr_present else 1.0
    addr_match = addr_score >= ADDRESS_MATCH_THRESHOLD

    # UIDAI status + PAN-Aadhaar linkage (both mocked in the demo)
    uidai_active = (extracted.get("uidai_status", "ACTIVE") == "ACTIVE")
    pan_linked = True

    # Hard-gate flags — any of these means the identity could not be verified.
    flags = []
    if not name_match:
        flags.append(f"Name mismatch: declared '{dec_name}' vs extracted '{ext_name}' (score {name_score:.2f})")
    if dob_present and not dob_match:
        flags.append(f"DOB mismatch: declared '{dec_dob}' vs extracted '{ext_dob}'")
    if pin_present and not pin_match:
        flags.append(f"PIN mismatch: declared '{dec_pin}' vs extracted '{ext_pin}'")
    if addr_present and not addr_match:
        flags.append(f"Address mismatch: declared '{dec_addr}' vs extracted '{ext_addr}' (score {addr_score:.2f})")
    if not uidai_active:
        flags.append("UIDAI status is not ACTIVE — Aadhaar may be suspended")

    id_verified = (
        name_match and dob_match and pin_match and addr_match
        and uidai_active and not flags
    )

    # Graded ID risk contribution (0.0 = perfect match, up to 1.0).
    id_risk_score = round(min(1.0,
        ID_PENALTY_NAME    * max(0.0, 1.0 - name_score) +
        ID_PENALTY_DOB     * (0.0 if dob_match  else 1.0) +
        ID_PENALTY_PIN     * (0.0 if pin_match  else 1.0) +
        ID_PENALTY_ADDRESS * max(0.0, 1.0 - addr_score)
    ), 3)

    reasons = []
    reasons.append(f"Name: {'✓' if name_match else '✗'} score {name_score:.2f} — '{dec_name}' vs '{ext_name}'")
    reasons.append(f"DOB: {'✓' if dob_match else '✗'} declared {dec_dob or '—'} vs extracted {ext_dob or '—'}")
    reasons.append(f"Address: {'✓' if addr_match else '✗'} score {addr_score:.2f}")
    reasons.append(f"PIN: {'✓' if pin_match else '✗'} declared {dec_pin or '—'} vs extracted {ext_pin or '—'}")
    reasons.append(f"UIDAI status: {'✓ ACTIVE' if uidai_active else '✗ NOT ACTIVE'}")
    reasons.append(f"PAN-Aadhaar linked: {'✓' if pan_linked else '✗'} (per {REG_PAN_AADHAAR})")

    details = {
        "name_match":          {"result": name_match, "score": name_score,
                                "note": f"'{dec_name}' vs '{ext_name}'"},
        "dob_match":           dob_match,
        "pin_match":           pin_match,
        "address_match":       {"result": addr_match, "score": addr_score},
        "uidai_active":        uidai_active,
        "pan_aadhaar_linked":  pan_linked,
        "authenticity_flags":  flags,
        "id_risk_score":       id_risk_score,
        "reasons":             reasons,
    }

    steps = [
        _step("Name matches declared", "pass" if name_match else "fail",
              f"similarity {name_score:.2f} (need ≥ {NAME_MATCH_THRESHOLD:.2f})"),
        _step("Date of birth matches", "pass" if dob_match else "fail",
              f"{dec_dob or '—'} vs {ext_dob or '—'}"),
        _step("PIN code matches", "pass" if pin_match else "fail",
              f"{dec_pin or '—'} vs {ext_pin or '—'}" if pin_present else "not provided"),
        _step("Address matches", "pass" if addr_match else "fail",
              f"similarity {addr_score:.2f} (need ≥ {ADDRESS_MATCH_THRESHOLD:.2f})"
              if addr_present else "not provided"),
        _step("UIDAI status active", "pass" if uidai_active else "fail",
              "ACTIVE" if uidai_active else "NOT ACTIVE"),
        _step("PAN–Aadhaar linked", "pass" if pan_linked else "fail",
              "linked" if pan_linked else "not linked"),
        _step("Identity verified", "pass" if id_verified else "fail",
              "all checks passed" if id_verified
              else f"{len(flags)} check(s) failed → re-apply"),
    ]

    return {
        "id_verified":        id_verified,
        "verification_details": details,
        "case_status":        CaseStatus.VERIFYING,
        "agent_checklists": [_checklist(
            "ID Verification", "🪪", steps,
            "Verified" if id_verified else "Verification FAILED"
        )],
        "audit_log": [
            f"{_now()} [ID Verify] verified={id_verified} (id_risk={id_risk_score:.2f}) — "
            f"name={name_score:.2f}, DOB={'✓' if dob_match else '✗'}, "
            f"PIN={'✓' if pin_match else '✗'}, addr={addr_score:.2f}, "
            f"UIDAI={'ACTIVE' if uidai_active else 'INACTIVE'}, "
            f"flags={len(flags)}"
        ],
    }


# ── Agent 4: compliance screening ───────────────────────────────────────────

def compliance_screening_agent(state: KYCState) -> KYCState:
    """
    Screen the customer name against the watchlists.

    1. Qdrant search (with fallbacks) over name variants.
    2. Adjust each raw match: DOB gate, then father's-name resolution.
    3. Status from the best adjusted score: < 0.60 CLEAR · 0.60–0.85 AMBIGUOUS
       (arms the loop when no father's name and retries remain) · > 0.85 hit.
       PEP entries are capped at POTENTIAL_MATCH (EDD, never auto-reject).

    Screening ALWAYS runs to completion — every applicant is freshly screened on
    every pass. Re-application dedup / cooldown is now owned by the applicant
    registry at intake (applicant_registry.py), not by a compliance short-circuit.
    (A prior-decision short-circuit used to surface a stale REJECT as CLEAR and
    skip the self-correction loop — e.g. Arjun Mehta re-running as REVIEW.)
    """
    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})
    name      = extracted.get("full_name_english") or declared.get("name", "")
    dob       = extracted.get("dob") or declared.get("dob", "")

    # Step 1: build name variants and search
    devanagari = extracted.get("full_name_devanagari", "")
    name_variants = list(filter(None, [
        name,
        devanagari,
        " ".join(reversed(name.split())),   # inverted order
        declared.get("name", ""),
    ]))

    hits = query_sanctions_db(name, name_variants)

    # Step 2: adjust each raw match with identity signals
    father    = extracted.get("father_name")
    cust_year = _dob_year(dob)
    notes     = []

    for hit in hits:
        raw_score = hit.get("match_score", 0.0)

        # DOB gate: outside the listed range (±tolerance) → score halved.
        low, high = _parse_dob_range(hit.get("dob_range"))
        if cust_year and low and (
            cust_year < low  - DOB_GATE_TOLERANCE_YEARS or
            cust_year > high + DOB_GATE_TOLERANCE_YEARS
        ):
            hit["match_score"] = round(raw_score * DOB_GATE_SCORE_FACTOR, 3)
            hit["dob_gate"] = (
                f"Customer DOB {cust_year} is outside listed range "
                f"{hit.get('dob_range')} (±{DOB_GATE_TOLERANCE_YEARS}y) — "
                f"score {raw_score:.2f} → {hit['match_score']:.2f}"
            )
            notes.append(f"DOB gate cleared '{hit.get('matched_name')}': {hit['dob_gate']}")

        # Father's-name resolution — only while the match is still ≥ CLEAR band.
        if father and hit.get("match_score", 0) >= FUZZY_CLEAR_BELOW:
            _resolve_with_father(hit, father)
            if hit.get("father_resolution"):
                notes.append(f"'{hit.get('matched_name')}': {hit['father_resolution']}")

    # Step 3: screening status from the best adjusted match
    hits.sort(key=lambda h: h.get("match_score", 0), reverse=True)
    best_score = hits[0]["match_score"] if hits else 0.0
    best_hit   = hits[0] if hits else {}
    is_pep     = "PEP" in (best_hit.get("list_source") or "").upper()

    if best_score >= FUZZY_AMBIGUOUS_HIGH:
        # A confirmed PEP goes to EDD with a human, never an automatic reject.
        status = ScreeningStatus.POTENTIAL_MATCH if is_pep else ScreeningStatus.CONFIRMED_HIT
        refine = False
    elif best_score >= FUZZY_CLEAR_BELOW:
        if is_pep:
            status = ScreeningStatus.POTENTIAL_MATCH
            refine = False
        else:
            status = ScreeningStatus.AMBIGUOUS
            # Loop trigger: ambiguous AND no father's name AND retries left.
            refine = (not father) and state.get("refine_count", 0) < REFINEMENT_MAX_TRIES
    else:
        status = ScreeningStatus.CLEAR
        refine = False
        hits   = []   # discard sub-threshold results

    # Attach agent reasoning to each surviving hit
    for hit in hits:
        if hit.get("father_resolution"):
            hit["reason"] = hit["father_resolution"]
        elif father:
            hit["reason"] = (
                f"Name similarity {hit['match_score']:.2f}. "
                f"Father's name '{father}' available for comparison."
            )
        else:
            hit["reason"] = (
                f"Name similarity {hit['match_score']:.2f}. "
                f"No father's name available to disambiguate."
            )

    note_text = (" | " + " ; ".join(notes)) if notes else ""

    status_state = {
        ScreeningStatus.CLEAR:           "pass",
        ScreeningStatus.AMBIGUOUS:       "warn",
        ScreeningStatus.POTENTIAL_MATCH: "warn",
        ScreeningStatus.CONFIRMED_HIT:   "fail",
    }.get(status, "info")

    steps = [
        _step("Fresh screening (no short-circuit)", "pass",
              "every applicant fully screened"),
        _step(f"Screened across {len(WATCHLISTS)} watchlists", "pass",
              f"{len(hits)} candidate match(es)"),
        _step("Best match score", status_state,
              f"{best_score:.2f}"
              + (f" — {best_hit.get('matched_name','')}" if best_hit else "")),
    ]
    if best_hit.get("dob_gate"):
        steps.append(_step("DOB gate applied", "info", best_hit["dob_gate"]))
    if father:
        if best_hit.get("father_resolution"):
            steps.append(_step("Father's-name resolution", status_state,
                               best_hit["father_resolution"]))
        else:
            steps.append(_step("Father's name available", "info",
                               f"'{father}' — compared against listed individuals"))
    elif refine:
        steps.append(_step("Father's name missing", "warn",
                           "ambiguous match → self-correction loop requested"))
    steps.append(_step(f"Screening status: {status}", status_state,
                       "refinement requested" if refine else "no refinement needed"))

    return {
        "compliance_hits":   hits,
        "screening_status":  status,
        "refinement_needed": refine,
        "cache_hit":         None,
        "case_status":       CaseStatus.SCREENING,
        "agent_checklists": [_checklist(
            "Compliance Screening", "🛡️", steps,
            f"{status} (best {best_score:.2f})"
        )],
        "audit_log": [
            f"{_now()} [Compliance] '{name}' screened against "
            f"{len(WATCHLISTS)} watchlists — status: {status}, "
            f"best score: {best_score:.2f}, hits: {len(hits)}, "
            f"refinement needed: {refine}{note_text}"
        ],
    }


# ── Agent 5: refine (self-correction loop control) ──────────────────────────

def refine_agent(state: KYCState) -> KYCState:
    """
    Control signal issued when compliance returns AMBIGUOUS. Bumps the refine
    counter and writes the refinement request the extraction agent reads on its
    retry. Does no AI work — the intelligence is the re-run of extraction at
    higher temperature with a focused directive.
    """
    count   = state.get("refine_count", 0) + 1
    hits    = state.get("compliance_hits", [])
    reason  = hits[0].get("reason", "ambiguous match") if hits else "ambiguous match"
    name    = state.get("extracted", {}).get("full_name_english", "")

    request = {
        "fields_to_seek": ["father_name", "care_of_name", "place_of_birth"],
        "instruction":    (
            "REFINEMENT PASS: The compliance agent found an ambiguous sanctions "
            "match that cannot be resolved without the father's name. "
            "Parse the Aadhaar QR 'care_of' field (C/O, S/O, W/O, D/O) — "
            "it contains the father's or husband's name. Also attempt to read "
            "place of birth if visible on the card."
        ),
        "temperature":    TEMP_REFINE_PASS,
        "attempt":        count,
    }

    steps = [
        _step("Ambiguous match detected", "warn", reason),
        _step(f"Refinement request #{count} issued", "info",
              "re-extract at higher temperature"),
        _step("Seeking disambiguating fields", "info",
              "father_name (QR care_of), place_of_birth"),
    ]

    return {
        "refine_count":        count,
        "refinement_request":  request,
        "current_agent_status": "REFINEMENT_REQUEST_ISSUED",
        "case_status":         CaseStatus.REFINING,
        "agent_checklists": [_checklist(
            "Self-Correction", "🔄", steps,
            f"Refinement #{count} — looping back to extraction"
        )],
        "audit_log": [
            f"{_now()} [Orchestrator] ⚠ AMBIGUOUS match for '{name}' — "
            f"Refinement Request #{count} issued to Extraction Agent. "
            f"Seeking: father_name, place_of_birth. "
            f"Reason: {reason}"
        ],
    }


# ── Agent 6: entity resolution (runs parallel to financial) ─────────────────

def entity_resolution_agent(state: KYCState) -> KYCState:
    """
    Traverse the corporate graph for indirect links between the applicant
    (address / PAN / occupation) and any sanctioned entity, within 3 hops.

    Entry points are matched exactly, then by fuzzy address containment, then
    by a MALHOTRA + business directorship inference (mock MCA21 lookup).
    """
    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})

    lookup_nodes = list(filter(None, [
        declared.get("address", ""),
        extracted.get("pin_code", ""),
        extracted.get("pan_number", ""),
        declared.get("occupation", ""),
    ]))

    def _norm(s: str) -> str:
        """Lowercase and strip punctuation so addresses compare fairly."""
        return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()

    # Find the customer's entry points: exact node match, then fuzzy address
    # containment ("34 Golf Links, New Delhi" → "34 Golf Links New Delhi 110003").
    customer_nodes = []
    for key in lookup_nodes:
        if CORPORATE_GRAPH.has_node(key):
            customer_nodes.append(key)
            continue
        nk = _norm(key)
        if len(nk) < 8:          # too short to fuzzy-match safely (e.g. PINs)
            continue
        for node in CORPORATE_GRAPH.nodes:
            nn = _norm(node)
            if nk in nn or nn in nk:
                customer_nodes.append(node)
                break

    # Mock MCA21 directorship inference: surname + business occupation → holding
    # company. In production this is a registrar lookup by PAN/DIN.
    if not customer_nodes:
        cust_name = (extracted.get("full_name_english") or declared.get("name", "")).upper()
        occ = declared.get("occupation", "").lower()
        if "MALHOTRA" in cust_name and ("business" in occ or "consultant" in occ):
            customer_nodes.append("VMK Holdings Pvt Ltd")

    sanctioned_nodes = [
        n for n, d in CORPORATE_GRAPH.nodes(data=True)
        if d.get("sanctioned", False)
    ]

    shortest_path  = None
    linked_entities = []
    indirect_flag  = False
    explanation    = "No connections to sanctioned entities found within 3 hops."

    for start in customer_nodes:
        for target in sanctioned_nodes:
            try:
                path = nx.shortest_path(CORPORATE_GRAPH, start, target)
                if len(path) - 1 <= 3:   # within 3 hops
                    indirect_flag = True
                    shortest_path = {
                        "path":         path,
                        "hop_distance": len(path) - 1,
                        "target_label": CORPORATE_GRAPH.nodes[target].get("label", target),
                    }
                    linked_entities = [
                        {
                            "name":           node,
                            "type":           CORPORATE_GRAPH.edges.get(
                                                (path[i], node), {}
                                              ).get("relationship", "related"),
                            "sanction_status": "SANCTIONED" if CORPORATE_GRAPH.nodes[node].get("sanctioned") else "CLEAN",
                        }
                        for i, node in enumerate(path[1:], 1)
                    ]
                    explanation = (
                        f"Indirect link found: {' → '.join(path)} "
                        f"({len(path)-1} hops). "
                        f"Terminal node: {CORPORATE_GRAPH.nodes[target].get('label', target)}"
                    )
                    break
            except nx.NetworkXNoPath:
                continue
            except nx.NodeNotFound:
                continue
        if indirect_flag:
            break

    network_risk = {
        "linked_entities":           linked_entities,
        "shortest_path_to_sanctioned": shortest_path,
        "indirect_risk_flag":        indirect_flag,
        "link_explanation":          explanation,
    }

    status = "FLAGGED" if indirect_flag else "CLEAN"
    hops   = shortest_path["hop_distance"] if shortest_path else "N/A"

    steps = [
        _step("Customer entry points resolved", "pass" if customer_nodes else "info",
              ", ".join(customer_nodes) if customer_nodes else "no graph entry point"),
        _step("Corporate graph traversal (≤ 3 hops)",
              "pass" if customer_nodes else "info",
              f"{CORPORATE_GRAPH.number_of_nodes()} entities scanned"),
        _step("Link to sanctioned entity",
              "fail" if indirect_flag else "pass",
              f"{hops}-hop chain found" if indirect_flag else "none within 3 hops"),
    ]
    if indirect_flag and shortest_path:
        steps.append(_step("Shortest path", "info", " → ".join(shortest_path["path"])))

    return {
        "network_risk": network_risk,
        "agent_checklists": [_checklist(
            "Entity Resolution", "🕸️", steps,
            f"Network {status}"
        )],
        "audit_log": [
            f"{_now()} [Entity Resolution] corporate graph traversal — "
            f"status: {status}, "
            f"{'hop distance: ' + str(hops) + ', path: ' + str(shortest_path['path']) if indirect_flag else 'no links found'}"
        ],
    }


# ── Agent 7: financial profiling (runs parallel to entity resolution) ───────

def financial_profiling_agent(state: KYCState) -> KYCState:
    """
    Assess whether declared income, occupation, and source of funds are
    plausible under Indian norms and PMLA. Optionally verifies declared income
    against an uploaded salary slip. Builds anomaly flags, an activity band, a
    risk contribution, and requires_edd (HNI income + anomalies → mandatory EDD).
    """
    declared = state.get("declared", {})
    income   = float(declared.get("income", 0))
    occ      = declared.get("occupation", "").lower().strip()
    source   = declared.get("source_of_funds", "").lower().strip()
    purpose  = declared.get("account_purpose", "").lower().strip()
    salary_slip_path = state.get("documents", {}).get("salary_slip", "")

    income_band = _get_income_band(income)
    occ_band    = OCCUPATION_INCOME_MAP.get(occ, "entry")

    anomaly_flags = []

    # Salary-slip income verification — only when a slip was uploaded and a
    # figure was read. Corroboration is a note; a large mismatch is a flag.
    income_verification = None
    if salary_slip_path and os.path.exists(salary_slip_path):
        slip = parse_salary_slip(salary_slip_path)
        slip_annual = slip.get("annual_income")
        if slip.get("raw_found") and slip_annual:
            variance = abs(slip_annual - income) / income if income > 0 else 1.0
            if variance <= SALARY_MATCH_TOLERANCE:
                income_verification = {
                    "status":      "VERIFIED",
                    "slip_annual": slip_annual,
                    "variance":    round(variance, 3),
                    "note": (
                        f"Declared income ₹{income:,.0f} corroborated by salary slip "
                        f"(₹{slip_annual:,.0f}/yr, {variance*100:.0f}% variance)."
                    ),
                }
            else:
                income_verification = {
                    "status":      "DISCREPANCY",
                    "slip_annual": slip_annual,
                    "variance":    round(variance, 3),
                    "note": (
                        f"Declared income ₹{income:,.0f} but salary slip shows "
                        f"₹{slip_annual:,.0f}/yr ({variance*100:.0f}% variance) — "
                        f"income proof does not match declaration."
                    ),
                }
                anomaly_flags.append(income_verification["note"])

    # Income vs occupation
    if income_band == "hni" and occ_band in ("entry", "middle"):
        anomaly_flags.append(
            f"₹{income:,.0f} income (HNI band) is implausible for "
            f"'{declared.get('occupation')}' (expected {occ_band} band)"
        )

    # Vague source of funds for high income
    vague_terms = ["business", "multiple", "various", "consulting", "profits"]
    is_vague = any(t in source for t in vague_terms) and len(source.split()) < 8
    if is_vague and income_band in ("upper_middle", "hni"):
        anomaly_flags.append(
            f"Source of funds '{declared.get('source_of_funds')}' is too vague "
            f"for income of ₹{income:,.0f} — specific documentation required "
            f"per {REG_PMLA}"
        )

    # HNI income without GST
    if income_band == "hni" and occ in ("business owner", "independent consultant"):
        anomaly_flags.append(
            f"HNI-band income declared for '{declared.get('occupation')}' "
            f"with no GST registration number provided — required per {REG_RBI_KYC}"
        )

    # Overseas investment + unsubstantiated income
    if "overseas" in purpose and is_vague:
        anomaly_flags.append(
            f"Account purpose includes 'overseas investment' combined with "
            f"vague source of funds — high-risk pattern for FEMA compliance"
        )

    # Activity band
    if income_band in ("entry", "middle"):
        activity_band = ActivityBand.LOW
    elif income_band == "upper_middle":
        activity_band = ActivityBand.MEDIUM
    else:
        activity_band = ActivityBand.HIGH

    # Financial risk contribution: 0.15 per flag, +0.30 if HNI with any flag.
    fin_risk = min(0.15 * len(anomaly_flags), 1.0)
    if income_band == "hni" and anomaly_flags:
        fin_risk = min(fin_risk + 0.30, 1.0)

    plausibility = {
        "income_vs_occupation": "CONSISTENT" if not anomaly_flags else "UNSUBSTANTIATED",
        "source_of_funds":      "VAGUE" if is_vague else "PLAUSIBLE",
        "income_band":          income_band,
        "expected_band":        occ_band,
        "note": (
            f"₹{income:,.0f} p.a. for '{declared.get('occupation')}' — "
            f"{'consistent with market norms' if not anomaly_flags else 'outside expected range'}"
        ),
    }

    # HNI money with unexplained anomalies → mandatory EDD; the risk agent
    # honours this even when the weighted score is low.
    requires_edd = bool(anomaly_flags) and income_band == "hni"

    profile = {
        "expected_activity_band":    activity_band,
        "plausibility":              plausibility,
        "anomaly_flags":             anomaly_flags,
        "requires_edd":              requires_edd,
        "income_verification":       income_verification,
        "financial_risk_contribution": round(fin_risk, 3),
    }

    steps = [
        _step("Income band assessed", "info",
              f"₹{income:,.0f} → {income_band} band"),
        _step("Income vs occupation plausible",
              "pass" if plausibility["income_vs_occupation"] == "CONSISTENT" else "warn",
              f"'{declared.get('occupation')}' (expected {occ_band})"),
        _step("Source of funds", "pass" if not is_vague else "warn",
              "plausible" if not is_vague else f"vague: '{declared.get('source_of_funds')}'"),
    ]
    if income_verification:
        steps.append(_step("Salary slip verification",
                           "pass" if income_verification["status"] == "VERIFIED" else "warn",
                           income_verification["note"]))
    for flag in anomaly_flags:
        steps.append(_step("Anomaly flag", "warn", flag))
    steps.append(_step("Enhanced due diligence required",
                       "warn" if requires_edd else "pass",
                       "yes — HNI income with anomalies" if requires_edd else "no"))

    return {
        "financial_profile": profile,
        "agent_checklists": [_checklist(
            "Financial Profiling", "💰", steps,
            f"{plausibility['income_vs_occupation']} — {len(anomaly_flags)} flag(s)"
        )],
        "audit_log": [
            f"{_now()} [Financial] ₹{income:,.0f} ({income_band} band), "
            f"occupation: '{declared.get('occupation')}', "
            f"plausibility: {plausibility['income_vs_occupation']}, "
            f"anomaly flags: {len(anomaly_flags)}"
            + (f", salary slip: {income_verification['status']}"
               if income_verification else "")
        ],
    }


# ── Agent 8: risk scoring & explanation ─────────────────────────────────────

def risk_scoring_agent(state: KYCState) -> KYCState:
    """
    Aggregate all agent outputs into a weighted risk score (ID 30% + Compliance
    40% + Network 20% + Financial 10%), map it to a band/decision, apply the
    regulatory overrides, and have Llama-3 write an evidence-citing explanation.

    Overrides (outrank the weighted score): CONFIRMED_HIT → REJECT (PMLA); any
    non-CLEAR screening → at least REVIEW (EDD); requires_edd → at least REVIEW
    (RBI KYC MD).
    """
    id_ok        = state.get("id_verified", False)
    comp_hits    = state.get("compliance_hits", [])
    comp_status  = state.get("screening_status", ScreeningStatus.CLEAR)
    network      = state.get("network_risk", {})
    financial    = state.get("financial_profile", {})
    extracted    = state.get("extracted", {})
    declared     = state.get("declared", {})

    refines       = state.get("refine_count", 0)
    resolved_via_loop = comp_status == ScreeningStatus.CLEAR and refines > 0

    # Per-factor raw scores. A loop-exonerated case keeps a 0.25 residual — it
    # was ambiguous enough to investigate, so it never scores like a never-flagged case.
    #
    # ID is GRADED: the verification agent's id_risk_score moves smoothly with
    # match quality (0.0 = perfect) instead of a 0.0/0.80 cliff. A hard
    # verification failure still auto-rejects below via an override.
    verification = state.get("verification_details", {})
    id_score   = verification.get("id_risk_score", 0.0 if id_ok else 0.80)
    comp_score = (
        (0.25 if resolved_via_loop else 0.0)
             if comp_status == ScreeningStatus.CLEAR else
        0.60 if comp_status == ScreeningStatus.AMBIGUOUS else
        0.75 if comp_status == ScreeningStatus.POTENTIAL_MATCH else
        1.0
    )
    net_score  = 0.80 if network.get("indirect_risk_flag") else 0.0
    fin_score  = financial.get("financial_risk_contribution", 0.0)

    risk_score = round(
        id_score   * WEIGHT_ID_VERIFICATION +
        comp_score * WEIGHT_COMPLIANCE +
        net_score  * WEIGHT_NETWORK_RISK +
        fin_score  * WEIGHT_FINANCIAL,
        3
    )
    risk_score = min(risk_score, 1.0)

    # Band + decision from the score
    if risk_score < RISK_AUTO_APPROVE_BELOW:
        risk_band = RiskBand.LOW
        decision  = Decision.APPROVE
        routing   = Routing.AUTO_APPROVE
    elif risk_score < RISK_REVIEW_BELOW:
        risk_band = RiskBand.MEDIUM
        decision  = Decision.REVIEW
        routing   = Routing.ROUTE_TO_HUMAN
    else:
        risk_band = RiskBand.HIGH
        decision  = Decision.REVIEW
        routing   = Routing.ROUTE_TO_HUMAN

    # Regulatory overrides (outrank the weighted score)
    override_note = ""
    if comp_status == ScreeningStatus.CONFIRMED_HIT:
        decision  = Decision.REJECT
        risk_band = RiskBand.HIGH
        routing   = Routing.ROUTE_TO_HUMAN
        override_note = "OVERRIDE: confirmed watchlist hit → REJECT (PMLA)"
    elif not id_ok:
        # The document does not match the declared data (or could not be
        # verified). We cannot establish identity → auto-reject and ask the
        # customer to re-apply with correct documents. Ends without a human.
        decision  = Decision.REJECT
        risk_band = RiskBand.HIGH
        routing   = Routing.AUTO_REJECT
        flag_list = "; ".join(verification.get("authenticity_flags", [])) or "identity could not be verified"
        override_note = (
            f"OVERRIDE: identity verification FAILED ({flag_list}) → "
            f"REJECT — customer must re-apply with matching documents"
        )
    elif decision == Decision.APPROVE and comp_status != ScreeningStatus.CLEAR:
        decision  = Decision.REVIEW
        routing   = Routing.ROUTE_TO_HUMAN
        risk_band = RiskBand.MEDIUM if risk_band == RiskBand.LOW else risk_band
        override_note = f"OVERRIDE: screening status {comp_status} → human review (EDD)"
    elif decision == Decision.APPROVE and financial.get("requires_edd"):
        decision  = Decision.REVIEW
        routing   = Routing.ROUTE_TO_HUMAN
        risk_band = RiskBand.MEDIUM if risk_band == RiskBand.LOW else risk_band
        override_note = "OVERRIDE: HNI income with anomalies → mandatory EDD (RBI KYC MD)"

    # Traceable contributing-factors breakdown
    factors = [
        {
            "factor":             "ID Verification",
            "weight":             WEIGHT_ID_VERIFICATION,
            "score_contribution": round(id_score * WEIGHT_ID_VERIFICATION, 3),
            "evidence":           (
                "PASS — name, DOB, PIN and address match; UIDAI active, PAN-Aadhaar linked"
                if id_ok else
                "FAIL — " + ("; ".join(verification.get("authenticity_flags", []))
                             or "identity could not be verified")
            ),
        },
        {
            "factor":             "Compliance Screening",
            "weight":             WEIGHT_COMPLIANCE,
            "score_contribution": round(comp_score * WEIGHT_COMPLIANCE, 3),
            "evidence":           (
                (f"CLEAR — ambiguous match resolved by self-correction loop "
                 f"(refinement pass {refines}); residual caution retained"
                 if resolved_via_loop else
                 f"CLEAR — no matches on {len(WATCHLISTS)} watchlists")
                if comp_status == ScreeningStatus.CLEAR else
                f"{comp_status} — {len(comp_hits)} match(es): {comp_hits[0].get('matched_name','') if comp_hits else ''}"
            ),
        },
        {
            "factor":             "Network / Entity Risk",
            "weight":             WEIGHT_NETWORK_RISK,
            "score_contribution": round(net_score * WEIGHT_NETWORK_RISK, 3),
            "evidence":           (
                f"FLAGGED — {network.get('link_explanation','indirect link found')}"
                if network.get("indirect_risk_flag") else
                "CLEAN — no indirect corporate links to sanctioned entities"
            ),
        },
        {
            "factor":             "Financial Profile",
            "weight":             WEIGHT_FINANCIAL,
            "score_contribution": round(fin_score * WEIGHT_FINANCIAL, 3),
            "evidence":           (
                f"{len(financial.get('anomaly_flags', []))} anomaly flag(s): "
                f"{'; '.join(financial.get('anomaly_flags', [])[:2])}"
                if financial.get("anomaly_flags") else
                f"CONSISTENT — income plausible for occupation"
            ),
        },
    ]

    # LLM explanation paragraph
    anomaly_summary = (
        "\n".join(f"- {f}" for f in financial.get("anomaly_flags", []))
        or "None"
    )
    net_summary = network.get("link_explanation", "No links found.")
    name = extracted.get("full_name_english") or declared.get("name", "the applicant")

    explanation_prompt = f"""
You are a senior KYC compliance officer writing a case decision note.
Write a clear, professional explanation for the following KYC assessment.
Cite the specific evidence — do not write generic text.
Write 3–4 sentences. Do not use bullet points.

Customer: {name}
Risk Score: {risk_score} ({risk_band})
Decision: {decision}

Evidence:
- ID Verification: {'PASSED' if id_ok else 'FAILED'}
- Compliance: {comp_status} — {len(comp_hits)} watchlist hit(s)
- Network Risk: {'FLAGGED — ' + net_summary if network.get('indirect_risk_flag') else 'CLEAN'}
- Financial Anomalies: {anomaly_summary}
- Declared Income: ₹{declared.get('income', 0):,.0f} as {declared.get('occupation', 'unknown')}

Write the explanation now:
"""
    explanation = call_text_llm(explanation_prompt, temperature=TEMP_EXPLANATION)

    decision_state = {
        Decision.APPROVE: "pass",
        Decision.REVIEW:  "warn",
        Decision.REJECT:  "fail",
    }.get(decision, "info")

    steps = [
        _step("ID verification factor (30%)", "pass" if id_ok else "fail",
              f"contribution {round(id_score*WEIGHT_ID_VERIFICATION,3)}"),
        _step("Compliance factor (40%)",
              "pass" if comp_status == ScreeningStatus.CLEAR else "warn",
              f"{comp_status} → contribution {round(comp_score*WEIGHT_COMPLIANCE,3)}"),
        _step("Network factor (20%)",
              "fail" if net_score else "pass",
              f"contribution {round(net_score*WEIGHT_NETWORK_RISK,3)}"),
        _step("Financial factor (10%)",
              "warn" if fin_score else "pass",
              f"contribution {round(fin_score*WEIGHT_FINANCIAL,3)}"),
        _step("Weighted risk score", decision_state,
              f"{risk_score:.3f} ({risk_band})"),
    ]
    if override_note:
        steps.append(_step("Regulatory override applied", "warn", override_note))
    steps.append(_step(f"Decision: {decision}", decision_state,
                       f"routing: {routing}"))

    return {
        "risk_score":          risk_score,
        "risk_band":           risk_band,
        "decision":            decision,
        "routing":             routing,
        "contributing_factors": factors,
        "explanation":         explanation,
        "case_status":         CaseStatus.SCORING,
        "agent_checklists": [_checklist(
            "Risk Scoring", "⚖️", steps,
            f"{decision} — score {risk_score:.3f} ({risk_band})"
        )],
        "audit_log": [
            f"{_now()} [Risk Score] {risk_score:.3f} ({risk_band}) → "
            f"{decision} via {routing} — "
            f"ID:{round(id_score*WEIGHT_ID_VERIFICATION,2)} + "
            f"Compliance:{round(comp_score*WEIGHT_COMPLIANCE,2)} + "
            f"Network:{round(net_score*WEIGHT_NETWORK_RISK,2)} + "
            f"Financial:{round(fin_score*WEIGHT_FINANCIAL,2)}"
            + (f" — {override_note}" if override_note else "")
        ],
    }


# ── Agent 9: human-in-the-loop review (stub — UI drives the real decision) ──

def hitl_review_agent(state: KYCState) -> KYCState:
    """
    Placeholder node so the graph can route to human review and still complete
    cleanly in tests. In the app, the officer decides in the UI and
    complete_case() runs the cache agent.
    """
    return {
        "case_status": CaseStatus.IN_REVIEW,
        "audit_log": [
            f"{_now()} [HITL] Case routed to human review queue — "
            f"awaiting compliance officer decision"
        ],
    }


# ── Agent 10: active learning cache update ──────────────────────────────────

def active_learning_cache_agent(state: KYCState) -> KYCState:
    """
    Store the officer's decision as an embedding in the 'exception_cache' Qdrant
    collection so future similar profiles surface it, and close the case. The
    cache write is best-effort — a failure is non-critical and the case still
    closes with the human decision.
    """
    human = state.get("human_decision", {}) or {}
    officer_decision = human.get("decision", Decision.REVIEW)
    rationale        = human.get("rationale", "No rationale provided")
    officer_id       = human.get("officer_id", "UNKNOWN")

    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})
    name      = extracted.get("full_name_english") or declared.get("name", "")
    dob       = extracted.get("dob") or declared.get("dob", "")

    cache_entry = {
        "name":       name,
        "dob":        dob,
        "decision":   officer_decision,
        "rationale":  rationale,
        "officer_id": officer_id,
        "timestamp":  datetime.now().isoformat(),
        "risk_score": state.get("risk_score", 0),
        "flags":      state.get("financial_profile", {}).get("anomaly_flags", []),
    }

    try:
        # Reuse the shared embedded client — local-mode Qdrant locks its folder.
        from tools import qdrant_client as client
        client.add(
            collection_name = "exception_cache",
            documents       = [f"{name} {dob} {officer_decision}"],
            metadata        = [cache_entry],
            ids             = [abs(hash(f"{name}{dob}")) % (10**9)],
        )
        cache_status = "STORED"
    except Exception as e:
        print(f"[agents] Cache store failed (non-critical): {e}")
        cache_status = "FAILED — non-critical, pipeline continues"

    return {
        "cache_update":   {"status": cache_status, "entry": cache_entry},
        "final_decision": officer_decision,
        "decision_source": DecisionSource.HUMAN,
        "closed_at":      datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
        "case_status":    CaseStatus.CLOSED,
        "audit_log": [
            f"{_now()} [Cache] Officer '{officer_id}' decision '{officer_decision}' "
            f"embedded and stored — future similar cases will surface this decision. "
            f"Cache status: {cache_status}"
        ],
    }


# ── Sanity check — run this file directly to verify all agents import ───────

if __name__ == "__main__":
    from state import create_initial_state

    print("=" * 60)
    print("agents.py — Import & Structure Check")
    print("=" * 60)

    agents = [
        ("intake_agent",                intake_agent),
        ("data_extraction_agent",       data_extraction_agent),
        ("id_verification_agent",       id_verification_agent),
        ("compliance_screening_agent",  compliance_screening_agent),
        ("refine_agent",                refine_agent),
        ("entity_resolution_agent",     entity_resolution_agent),
        ("financial_profiling_agent",   financial_profiling_agent),
        ("risk_scoring_agent",          risk_scoring_agent),
        ("hitl_review_agent",           hitl_review_agent),
        ("active_learning_cache_agent", active_learning_cache_agent),
    ]

    print(f"\n✅ All {len(agents)} agents defined:")
    for name, fn in agents:
        print(f"   - {name}")

    print("\nRunning dry-run checks (no GPU required)...")

    test_state = create_initial_state(
        customer_id="CUST-TEST-001", name="Priya Sharma",
        dob="1992-09-08", nationality="Indian",
        address="B-204 Green Park Bengaluru", pin_code="560034",
        occupation="software engineer", income=1_200_000,
        source_of_funds="Monthly salary from Infosys Limited",
        account_purpose="Savings and investments",
        aadhaar_path="./mock_data/aadhaar_priya.jpg",
        pan_path="./mock_data/pan_priya.jpg",
        received_at=datetime.now().isoformat(),
    )

    r = intake_agent(test_state)
    print(f"   intake_agent              → case_status: {r.get('case_status')}")

    test_state["extracted"] = {"full_name_english": "PRIYA SHARMA", "dob": "1992-09-08"}
    r = financial_profiling_agent(test_state)
    print(f"   financial_profiling_agent → activity_band: {r['financial_profile']['expected_activity_band']}, "
          f"anomalies: {len(r['financial_profile']['anomaly_flags'])}")

    r = entity_resolution_agent(test_state)
    print(f"   entity_resolution_agent   → indirect_flag: {r['network_risk']['indirect_risk_flag']}")

    test_state["extracted"] = {
        "full_name_english": "PRIYA SHARMA", "dob": "1992-09-08",
        "uidai_status": "ACTIVE", "pin_code": "560034",
    }
    r = id_verification_agent(test_state)
    print(f"   id_verification_agent     → id_verified: {r.get('id_verified')}")

    print("\n✅ All dry-run checks passed.")
    print("Agents needing vLLM (data_extraction, compliance, risk_scoring)")
    print("will be tested when you run the full pipeline in graph.py.")
    print("=" * 60)
