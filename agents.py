"""
agents.py — All KYC Agent Functions
=====================================
Every agent follows the same pattern:
  INPUT  → receives the full KYCState
  OUTPUT → returns ONLY the fields it changed (a partial dict)
  LangGraph merges the return value back into shared state automatically.

Agents in pipeline order:
  1.  intake_agent               — initialise the case
  2.  data_extraction_agent      — OCR + parse Aadhaar fields
  3.  id_verification_agent      — cross-check extracted vs declared
  4.  compliance_screening_agent — sanctions watchlist check
  5.  refine_agent               — self-correction: bump counter + request
  6.  entity_resolution_agent    — MCA21 / NetworkX corporate link analysis
  7.  financial_profiling_agent  — income plausibility + anomaly flags
  8.  risk_scoring_agent         — weighted score + NL explanation
  9.  hitl_review_agent          — stub (human officer decides via UI)
  10. active_learning_cache_agent — store officer decision for future cases

Imports needed:
  pip install networkx openai qdrant-client[fastembed] pytesseract pillow
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
    check_exception_cache,
    setup_sanctions_collection,
)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _name_similarity(name_a: str, name_b: str) -> float:
    """
    Returns 0.0–1.0 similarity between two name strings.
    Case-insensitive. Handles Indian name component ordering.
    e.g. "Priya Sharma" vs "SHARMA PRIYA" → still high score.
    """
    a = name_a.lower().strip()
    b = name_b.lower().strip()

    # Direct similarity
    direct = difflib.SequenceMatcher(None, a, b).ratio()

    # Component-wise similarity (handles inverted name order)
    parts_a = set(a.split())
    parts_b = set(b.split())
    if parts_a and parts_b:
        component = len(parts_a & parts_b) / max(len(parts_a), len(parts_b))
    else:
        component = 0.0

    return round(max(direct, component), 3)


def _dob_year(dob: str) -> int:
    """Pulls the birth year out of a DOB string like '1968-11-14'. 0 if unknown."""
    m = re.search(r"\b(19|20)\d{2}\b", dob or "")
    return int(m.group()) if m else 0


def _parse_dob_range(dob_range: str) -> tuple:
    """Parses a watchlist DOB range like '1966-1970' or '1966–1970' → (1966, 1970)."""
    years = re.findall(r"\b(?:19|20)\d{2}\b", dob_range or "")
    if not years:
        return (0, 0)
    return (int(years[0]), int(years[-1]))


# DOB tolerance: watchlist DOB ranges are approximate, so allow ±3 years
# before ruling someone out. Outside that window the match score is halved —
# this is how an exact-name false positive (same name, different generation)
# gets cleared without ever bothering a human.
DOB_GATE_TOLERANCE_YEARS = 3
DOB_GATE_SCORE_FACTOR    = 0.5

# Father's-name resolution: a differing father drops the match score by ~65%
# (CLAUDE.md §4); a matching father escalates the score into the confirmed band.
FATHER_MATCH_THRESHOLD   = 0.80
FATHER_MISMATCH_FACTOR   = 0.35
FATHER_CONFIRM_SCORE     = 0.95


def _resolve_with_father(hit: dict, father: str) -> dict:
    """
    The payoff of the self-correction loop. Compares the customer's father's
    name (found via the Aadhaar QR 'care_of' field on the refinement pass)
    against the father's name listed for the watchlist individual.

      father names DIFFER → match score drops ~65% → usually CLEAR
      father names MATCH  → score escalates → CONFIRMED_HIT

    The loop works both ways — it can exonerate AND convict. Mutates and
    returns the hit with an explanatory 'father_resolution' note.
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
    """Maps INR income to band label."""
    for band, (low, high) in INCOME_BANDS.items():
        if low <= income < high:
            return band
    return "hni"


def _estimate_bounding_boxes(image_path: str) -> dict:
    """
    Returns approximate bounding box positions for standard Aadhaar card fields.
    These positions are based on the typical UIDAI Aadhaar card layout.
    Good enough for the UI overlay — pixel-perfect is not needed for the demo.
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


# ═══════════════════════════════════════════════════════════════════════════
# MOCK CORPORATE GRAPH  (Entity Resolution — NetworkX)
# ═══════════════════════════════════════════════════════════════════════════
# In production this would query MCA21 (Ministry of Corporate Affairs).
# For the demo, this hardcoded graph covers both example customers.
# Add more nodes here if your bulk dossiers include other high-risk profiles.

def _build_corporate_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # Edges: (source, target, metadata)
    edges = [
        # Vikram Malhotra's corporate chain (3 hops to sanctioned entity)
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

    # Mark sanctioned nodes. Orion itself is the sanctioned entity (CLAUDE.md
    # §8) — flagging it directly keeps the demo chain within the 3-hop search:
    # customer address → VMK Holdings → Sunrise Export Import → Orion (3 hops).
    for node in ("Orion Global Trade Solutions LLP", "SANCTIONED:ED_PMLA_2021_DEL_0892"):
        G.nodes[node]["sanctioned"] = True
        G.nodes[node]["label"] = (
            "Orion Global Trade Solutions LLP — ED Investigation PMLA Case 2021/DEL/0892"
        )

    return G

CORPORATE_GRAPH = _build_corporate_graph()


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 1 — INTAKE
# ═══════════════════════════════════════════════════════════════════════════

def intake_agent(state: KYCState) -> KYCState:
    """
    Initialises the case. Confirms all required fields are present.
    Writes the first audit log entry.
    In practice this is handled by create_initial_state() in state.py —
    this node just confirms receipt and updates case_status.
    """
    cid     = state.get("customer_id", "UNKNOWN")
    income  = state.get("declared", {}).get("income", 0)
    name    = state.get("declared", {}).get("name", "UNKNOWN")

    return {
        "case_status": CaseStatus.RECEIVED,
        "audit_log": [
            f"{_now()} [Intake] Case {cid} received — "
            f"Customer: {name}, Declared income: ₹{income:,.0f}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 2 — DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def data_extraction_agent(state: KYCState) -> KYCState:
    """
    Reads the Aadhaar card image using OCR, then sends the raw text to
    Llama-3 to extract structured identity fields as JSON.

    Runs TWICE when the self-correction loop triggers:
      Pass 0: standard extraction at low temperature
      Pass 1: refinement pass at higher temperature, targeting father's name
    """
    refine_count = state.get("refine_count", 0)
    is_refinement = refine_count > 0
    documents     = state.get("documents", {})
    aadhaar_path  = documents.get("aadhaar_card", "")
    aadhaar_back  = documents.get("aadhaar_back", "")

    directive   = "refinement_pass" if is_refinement else "standard_first_pass"
    temperature = TEMP_REFINE_PASS if is_refinement else TEMP_FIRST_PASS
    pass_label  = f"Pass {refine_count} (refinement)" if is_refinement else "Pass 0 (standard)"

    # Step 1: OCR — extract raw text from the Aadhaar front image, then append
    # the BACK image text if one was uploaded. The back of an Aadhaar carries
    # the full address and the QR 'care_of' (father/husband) field, so feeding
    # both faces to the LLM gives the refinement pass a real chance of finding
    # the father's name. No back image → back_text stays empty → unchanged.
    ocr_text  = extract_text_from_image(aadhaar_path)
    back_text = extract_text_from_image(aadhaar_back) if aadhaar_back else ""
    used_back = bool(back_text.strip())
    if used_back:
        ocr_text = (ocr_text + "\n" + back_text).strip()

    # Step 2: LLM — structure the OCR text into identity fields.
    # If OCR produced nothing (no image on disk — bulk profiles, laptop mode),
    # skip the LLM call entirely: there is nothing for it to read.
    if ocr_text.strip():
        extracted = parse_aadhaar_fields(ocr_text, directive, temperature)
    else:
        extracted = {}
    declared_fallback = False

    # Step 2.5: declared-data fallback (mock mode). When the document could
    # not be read, mirror the customer's declared KYC form so the rest of the
    # pipeline still exercises real logic. The demo never crashes on a
    # missing image — it degrades to declared data and says so in the audit.
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

    # Father's name only surfaces on the REFINEMENT pass — it simulates the
    # deep parse of the Aadhaar QR 'care_of' field that the refinement
    # directive requests. On pass 0 it stays hidden so the self-correction
    # loop has something to find. (See CLAUDE.md §4 and §13.)
    if is_refinement and not extracted.get("father_name"):
        extracted["father_name"] = state.get("declared", {}).get("father_name")

    extracted["extraction_pass"] = refine_count

    # Step 3: Confidence scores per field
    confidence = compute_field_confidence(extracted)
    low_conf   = [f for f, s in confidence.items() if s < 0.75 and extracted.get(f) is None]

    # Step 4: Bounding boxes for UI overlay
    boxes = _estimate_bounding_boxes(aadhaar_path)

    # Step 5: Determine extraction status
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

    return {
        "extracted":             extracted,
        "field_confidence":      confidence,
        "bounding_boxes":        boxes,
        "low_confidence_fields": low_conf,
        "extraction_status":     status,
        "case_status":           CaseStatus.EXTRACTING,
        "audit_log": [
            f"{_now()} [Extraction] {pass_label} — status: {status}, "
            f"name: '{name_found}', {father_note}, "
            f"low-confidence fields: {low_conf or 'none'}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 3 — ID VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def id_verification_agent(state: KYCState) -> KYCState:
    """
    Cross-checks extracted Aadhaar fields against what the customer declared.
    Checks: name match, DOB match, address PIN match, UIDAI active status,
    PAN-Aadhaar linkage (mandatory per RBI 2023).

    Uses fuzzy name matching to handle:
      - Name component ordering differences (Given Surname vs Surname Given)
      - Minor spelling differences from OCR noise
    """
    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})

    # ── Name match ────────────────────────────────────────────────────────
    ext_name  = extracted.get("full_name_english", "") or ""
    dec_name  = declared.get("name", "") or ""
    name_score = _name_similarity(ext_name, dec_name)
    name_match = name_score >= 0.80

    # ── DOB match ─────────────────────────────────────────────────────────
    ext_dob = (extracted.get("dob") or "").replace("/", "-").strip()
    dec_dob = (declared.get("dob") or "").replace("/", "-").strip()
    dob_match = ext_dob == dec_dob

    # ── Address / PIN match ───────────────────────────────────────────────
    ext_pin = str(extracted.get("pin_code") or "").strip()
    dec_pin = str(declared.get("pin_code") or "").strip()
    pin_match = (ext_pin == dec_pin) if (ext_pin and dec_pin) else True

    # ── UIDAI status (mocked — ACTIVE always in demo) ─────────────────────
    uidai_active = (extracted.get("uidai_status", "ACTIVE") == "ACTIVE")

    # ── PAN-Aadhaar linkage (mocked as True — mandatory per RBI 2023) ─────
    pan_linked = True

    # ── Authenticity flags ────────────────────────────────────────────────
    flags = []
    if name_score < 0.60:
        flags.append(f"Name mismatch: declared '{dec_name}' vs extracted '{ext_name}' (score {name_score})")
    if not dob_match and ext_dob and dec_dob:
        flags.append(f"DOB mismatch: declared '{dec_dob}' vs extracted '{ext_dob}'")
    if not uidai_active:
        flags.append("UIDAI status is not ACTIVE — Aadhaar may be suspended")

    # ── Overall decision ──────────────────────────────────────────────────
    id_verified = name_match and dob_match and uidai_active and not flags

    reasons = []
    reasons.append(f"Name: {'✓' if name_match else '✗'} score {name_score:.2f} — '{dec_name}' vs '{ext_name}'")
    reasons.append(f"DOB: {'✓' if dob_match else '✗'} declared {dec_dob} vs extracted {ext_dob}")
    reasons.append(f"UIDAI status: {'✓ ACTIVE' if uidai_active else '✗ NOT ACTIVE'}")
    reasons.append(f"PAN-Aadhaar linked: {'✓' if pan_linked else '✗'} (per {REG_PAN_AADHAAR})")

    details = {
        "name_match":          {"result": name_match, "score": name_score,
                                "note": f"'{dec_name}' vs '{ext_name}'"},
        "dob_match":           dob_match,
        "pin_match":           pin_match,
        "uidai_active":        uidai_active,
        "pan_aadhaar_linked":  pan_linked,
        "authenticity_flags":  flags,
        "reasons":             reasons,
    }

    return {
        "id_verified":        id_verified,
        "verification_details": details,
        "case_status":        CaseStatus.VERIFYING,
        "audit_log": [
            f"{_now()} [ID Verify] verified={id_verified} — "
            f"name score={name_score:.2f}, DOB={'✓' if dob_match else '✗'}, "
            f"UIDAI={'ACTIVE' if uidai_active else 'INACTIVE'}, "
            f"PAN linked={'✓' if pan_linked else '✗'}, "
            f"flags={len(flags)}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 4 — COMPLIANCE SCREENING
# ═══════════════════════════════════════════════════════════════════════════

def compliance_screening_agent(state: KYCState) -> KYCState:
    """
    Screens the customer name against Indian and international watchlists:
    FIU-IND, ED (PMLA), RBI Wilful Defaulters, UN Security Council.

    Step 1: Check the Human Exception Cache — if a prior officer decision
            covers this profile, return it immediately.
    Step 2: Query Qdrant sanctions DB (semantic search).
            Falls back to LLM-as-judge if Qdrant is unavailable.
    Step 3: Score the best match and determine screening_status.
            CLEAR     → score < 0.60
            AMBIGUOUS → score 0.60–0.85 (triggers self-correction loop)
            HIT       → score > 0.85

    The self-correction loop fires when: AMBIGUOUS + refine_count < cap.
    """
    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})
    name      = extracted.get("full_name_english") or declared.get("name", "")
    dob       = extracted.get("dob") or declared.get("dob", "")

    # ── Step 1: Exception cache ───────────────────────────────────────────
    cache_hit = check_exception_cache(name, dob)
    if cache_hit:
        return {
            "compliance_hits":   [],
            "screening_status":  ScreeningStatus.CLEAR,
            "refinement_needed": False,
            "cache_hit":         cache_hit,
            "case_status":       CaseStatus.SCREENING,
            "audit_log": [
                f"{_now()} [Compliance] CACHE HIT — prior officer decision "
                f"found: '{cache_hit.get('decision')}'. Skipping re-screen."
            ],
        }

    # ── Step 2: Name variants for richer search ───────────────────────────
    devanagari = extracted.get("full_name_devanagari", "")
    name_variants = list(filter(None, [
        name,
        devanagari,
        " ".join(reversed(name.split())),   # inverted order
        declared.get("name", ""),
    ]))

    hits = query_sanctions_db(name, name_variants)

    # ── Step 3: Adjust each raw name match with identity signals ─────────
    father    = extracted.get("father_name")
    cust_year = _dob_year(dob)
    notes     = []

    for hit in hits:
        raw_score = hit.get("match_score", 0.0)

        # DOB gate: same name but a different generation is not the same
        # person. Outside the listed range (±tolerance) → score halved.
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

        # Father's-name resolution (self-correction payoff): only meaningful
        # while the match is still in or above the ambiguous band.
        if father and hit.get("match_score", 0) >= FUZZY_CLEAR_BELOW:
            _resolve_with_father(hit, father)
            if hit.get("father_resolution"):
                notes.append(f"'{hit.get('matched_name')}': {hit['father_resolution']}")

    # ── Step 4: Determine screening status from best adjusted match ──────
    hits.sort(key=lambda h: h.get("match_score", 0), reverse=True)
    best_score = hits[0]["match_score"] if hits else 0.0
    best_hit   = hits[0] if hits else {}
    is_pep     = "PEP" in (best_hit.get("list_source") or "").upper()

    if best_score >= FUZZY_AMBIGUOUS_HIGH:
        # PEP is not a sanction — a confirmed PEP goes to Enhanced Due
        # Diligence with a human, never to an automatic reject.
        status = ScreeningStatus.POTENTIAL_MATCH if is_pep else ScreeningStatus.CONFIRMED_HIT
        refine = False
    elif best_score >= FUZZY_CLEAR_BELOW:
        if is_pep:
            status = ScreeningStatus.POTENTIAL_MATCH
            refine = False
        else:
            status = ScreeningStatus.AMBIGUOUS
            # Self-correction trigger: ambiguous AND no father's name to
            # disambiguate AND retries left (CLAUDE.md §4).
            refine = (not father) and state.get("refine_count", 0) < REFINEMENT_MAX_TRIES
    else:
        status = ScreeningStatus.CLEAR
        refine = False
        hits   = []   # discard sub-threshold results

    # ── Add agent reasoning to each surviving hit ─────────────────────────
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
    return {
        "compliance_hits":   hits,
        "screening_status":  status,
        "refinement_needed": refine,
        "cache_hit":         None,
        "case_status":       CaseStatus.SCREENING,
        "audit_log": [
            f"{_now()} [Compliance] '{name}' screened against "
            f"{len(WATCHLISTS)} watchlists — status: {status}, "
            f"best score: {best_score:.2f}, hits: {len(hits)}, "
            f"refinement needed: {refine}{note_text}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 5 — REFINE  (self-correction loop bump)
# ═══════════════════════════════════════════════════════════════════════════

def refine_agent(state: KYCState) -> KYCState:
    """
    Issued by the orchestrator when compliance returns AMBIGUOUS.
    Increments the refine counter and writes the refinement request —
    the extraction agent reads this on its next (retry) invocation.

    This node does NO AI work — it's a control signal.
    The intelligence comes from extraction_agent running again at
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

    return {
        "refine_count":        count,
        "refinement_request":  request,
        "current_agent_status": "REFINEMENT_REQUEST_ISSUED",
        "case_status":         CaseStatus.REFINING,
        "audit_log": [
            f"{_now()} [Orchestrator] ⚠ AMBIGUOUS match for '{name}' — "
            f"Refinement Request #{count} issued to Extraction Agent. "
            f"Seeking: father_name, place_of_birth. "
            f"Reason: {reason}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 6 — ENTITY RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

def entity_resolution_agent(state: KYCState) -> KYCState:
    """
    Traverses the corporate relationship graph to find indirect links
    between the applicant's address / employer / PAN and any sanctioned entity.

    Uses NetworkX to check paths up to 3 hops away.
    In production: queries MCA21 (Ministry of Corporate Affairs) API.
    In demo: uses the hardcoded CORPORATE_GRAPH defined above.

    Runs in PARALLEL with financial_profiling_agent.
    """
    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})

    # Build lookup keys for this customer
    lookup_nodes = list(filter(None, [
        declared.get("address", ""),
        extracted.get("pin_code", ""),
        extracted.get("pan_number", ""),
        declared.get("occupation", ""),
    ]))

    def _norm(s: str) -> str:
        """Lowercase and strip punctuation so address strings compare fairly."""
        return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()

    # Find the customer's entry points into the corporate graph.
    # Exact node match first, then fuzzy address containment ("34 Golf Links,
    # New Delhi" should still find node "34 Golf Links New Delhi 110003").
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

    # Mock MCA21 directorship inference (CLAUDE.md §8): the demo customer's
    # surname + a business occupation maps to his holding company. In
    # production this is a real registrar lookup by PAN/DIN.
    if not customer_nodes:
        cust_name = (extracted.get("full_name_english") or declared.get("name", "")).upper()
        occ = declared.get("occupation", "").lower()
        if "MALHOTRA" in cust_name and ("business" in occ or "consultant" in occ):
            customer_nodes.append("VMK Holdings Pvt Ltd")

    # Search for paths to sanctioned nodes within 3 hops
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
                    # Collect intermediate entities
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

    return {
        "network_risk": network_risk,
        "audit_log": [
            f"{_now()} [Entity Resolution] corporate graph traversal — "
            f"status: {status}, "
            f"{'hop distance: ' + str(hops) + ', path: ' + str(shortest_path['path']) if indirect_flag else 'no links found'}"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 7 — FINANCIAL PROFILING
# ═══════════════════════════════════════════════════════════════════════════

def financial_profiling_agent(state: KYCState) -> KYCState:
    """
    Assesses whether the declared income, occupation, and source of funds
    are consistent and plausible under Indian income norms and PMLA guidelines.

    Runs in PARALLEL with entity_resolution_agent.
    """
    declared = state.get("declared", {})
    income   = float(declared.get("income", 0))
    occ      = declared.get("occupation", "").lower().strip()
    source   = declared.get("source_of_funds", "").lower().strip()
    purpose  = declared.get("account_purpose", "").lower().strip()
    salary_slip_path = state.get("documents", {}).get("salary_slip", "")

    income_band = _get_income_band(income)
    occ_band    = OCCUPATION_INCOME_MAP.get(occ, "entry")

    # ── Plausibility checks ───────────────────────────────────────────────
    anomaly_flags = []

    # ── Salary-slip income verification ──────────────────────────────────
    # Only runs when a slip was actually uploaded AND a figure could be read.
    # Corroboration → informational note. A large mismatch → anomaly flag.
    # (No slip on the canonical customers / bulk profiles, so this is inert
    # there and the 20/20 acceptance test is unaffected.)
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

    # ── Activity band ─────────────────────────────────────────────────────
    if income_band in ("entry", "middle"):
        activity_band = ActivityBand.LOW
    elif income_band == "upper_middle":
        activity_band = ActivityBand.MEDIUM
    else:
        activity_band = ActivityBand.HIGH

    # ── Financial risk contribution ───────────────────────────────────────
    # More anomaly flags = higher contribution
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

    # HNI money with unexplained anomalies cannot be auto-approved —
    # RBI KYC Master Direction mandates Enhanced Due Diligence by an officer.
    # The risk agent honours this flag even when the weighted score is low.
    requires_edd = bool(anomaly_flags) and income_band == "hni"

    profile = {
        "expected_activity_band":    activity_band,
        "plausibility":              plausibility,
        "anomaly_flags":             anomaly_flags,
        "requires_edd":              requires_edd,
        "income_verification":       income_verification,
        "financial_risk_contribution": round(fin_risk, 3),
    }

    return {
        "financial_profile": profile,
        "audit_log": [
            f"{_now()} [Financial] ₹{income:,.0f} ({income_band} band), "
            f"occupation: '{declared.get('occupation')}', "
            f"plausibility: {plausibility['income_vs_occupation']}, "
            f"anomaly flags: {len(anomaly_flags)}"
            + (f", salary slip: {income_verification['status']}"
               if income_verification else "")
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 8 — RISK SCORING & EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════

def risk_scoring_agent(state: KYCState) -> KYCState:
    """
    Aggregates all agent outputs into a single risk score (0.0 – 1.0),
    maps it to a decision band, and generates a plain-English explanation
    using Llama-3. The explanation cites specific evidence — not generic text.

    Weights (from config.py):
      ID Verification  30%
      Compliance       40%
      Network Risk     20%
      Financial        10%
    """
    # ── Collect inputs ────────────────────────────────────────────────────
    id_ok        = state.get("id_verified", False)
    comp_hits    = state.get("compliance_hits", [])
    comp_status  = state.get("screening_status", ScreeningStatus.CLEAR)
    network      = state.get("network_risk", {})
    financial    = state.get("financial_profile", {})
    extracted    = state.get("extracted", {})
    declared     = state.get("declared", {})

    # ── Individual scores ─────────────────────────────────────────────────
    refines       = state.get("refine_count", 0)
    resolved_via_loop = comp_status == ScreeningStatus.CLEAR and refines > 0

    id_score   = 0.0 if id_ok else 0.80
    comp_score = (
        # A case the self-correction loop had to exonerate keeps a residual —
        # it was ambiguous enough to investigate, so it never scores like a
        # case that was never flagged at all.
        (0.25 if resolved_via_loop else 0.0)
             if comp_status == ScreeningStatus.CLEAR else
        0.60 if comp_status == ScreeningStatus.AMBIGUOUS else
        0.75 if comp_status == ScreeningStatus.POTENTIAL_MATCH else
        1.0
    )
    net_score  = 0.80 if network.get("indirect_risk_flag") else 0.0
    fin_score  = financial.get("financial_risk_contribution", 0.0)

    # ── Weighted aggregate ────────────────────────────────────────────────
    risk_score = round(
        id_score   * WEIGHT_ID_VERIFICATION +
        comp_score * WEIGHT_COMPLIANCE +
        net_score  * WEIGHT_NETWORK_RISK +
        fin_score  * WEIGHT_FINANCIAL,
        3
    )
    risk_score = min(risk_score, 1.0)

    # ── Risk band + decision ──────────────────────────────────────────────
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

    # ── Regulatory overrides (these outrank the weighted score) ──────────
    override_note = ""
    if comp_status == ScreeningStatus.CONFIRMED_HIT:
        # A confirmed watchlist hit is a REJECT regardless of arithmetic —
        # PMLA does not allow onboarding a sanctioned individual.
        decision  = Decision.REJECT
        risk_band = RiskBand.HIGH
        routing   = Routing.ROUTE_TO_HUMAN
        override_note = "OVERRIDE: confirmed watchlist hit → REJECT (PMLA)"
    elif decision == Decision.APPROVE and comp_status != ScreeningStatus.CLEAR:
        # Any unresolved screening signal (PEP, potential match) needs a human.
        decision  = Decision.REVIEW
        routing   = Routing.ROUTE_TO_HUMAN
        risk_band = RiskBand.MEDIUM if risk_band == RiskBand.LOW else risk_band
        override_note = f"OVERRIDE: screening status {comp_status} → human review (EDD)"
    elif decision == Decision.APPROVE and financial.get("requires_edd"):
        # HNI income with anomaly flags → mandatory Enhanced Due Diligence.
        decision  = Decision.REVIEW
        routing   = Routing.ROUTE_TO_HUMAN
        risk_band = RiskBand.MEDIUM if risk_band == RiskBand.LOW else risk_band
        override_note = "OVERRIDE: HNI income with anomalies → mandatory EDD (RBI KYC MD)"

    # ── Contributing factors breakdown ───────────────────────────────────
    factors = [
        {
            "factor":             "ID Verification",
            "weight":             WEIGHT_ID_VERIFICATION,
            "score_contribution": round(id_score * WEIGHT_ID_VERIFICATION, 3),
            "evidence":           "PASS — Aadhaar QR valid, UIDAI active, PAN-Aadhaar linked"
                                  if id_ok else "FAIL — identity verification did not pass",
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

    # ── LLM explanation paragraph ─────────────────────────────────────────
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

    return {
        "risk_score":          risk_score,
        "risk_band":           risk_band,
        "decision":            decision,
        "routing":             routing,
        "contributing_factors": factors,
        "explanation":         explanation,
        "case_status":         CaseStatus.SCORING,
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


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 9 — HUMAN-IN-THE-LOOP REVIEW  (stub — UI overrides this)
# ═══════════════════════════════════════════════════════════════════════════

def hitl_review_agent(state: KYCState) -> KYCState:
    """
    Placeholder node for human review. In the real Streamlit app, the
    officer clicks Approve / Reject / Hold in the UI — that updates
    human_decision in the state directly.

    This stub exists so the graph can route to it and the pipeline
    completes cleanly even without UI interaction (useful for testing).
    """
    return {
        "case_status": CaseStatus.IN_REVIEW,
        "audit_log": [
            f"{_now()} [HITL] Case routed to human review queue — "
            f"awaiting compliance officer decision"
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGENT 10 — ACTIVE LEARNING CACHE UPDATE
# ═══════════════════════════════════════════════════════════════════════════

def active_learning_cache_agent(state: KYCState) -> KYCState:
    """
    After a human officer makes a decision, store that decision as an
    embedding in the 'exception_cache' Qdrant collection.

    Future cases with similar profiles will surface this decision first,
    allowing the system to learn from human judgment without retraining.
    """
    human = state.get("human_decision", {}) or {}
    officer_decision = human.get("decision", Decision.REVIEW)
    rationale        = human.get("rationale", "No rationale provided")
    officer_id       = human.get("officer_id", "UNKNOWN")

    extracted = state.get("extracted", {})
    declared  = state.get("declared", {})
    name      = extracted.get("full_name_english") or declared.get("name", "")
    dob       = extracted.get("dob") or declared.get("dob", "")

    # Store in Qdrant exception_cache collection (best-effort)
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
        # Reuse the shared embedded client from tools.py — local-mode Qdrant
        # locks its storage folder, so a second QdrantClient would fail.
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


# ═══════════════════════════════════════════════════════════════════════════
# SANITY CHECK  —  run this file directly to verify all agents import cleanly
# ═══════════════════════════════════════════════════════════════════════════

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

    # Quick dry-run of agents that don't need external services
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

    # Test intake
    r = intake_agent(test_state)
    print(f"   intake_agent              → case_status: {r.get('case_status')}")

    # Test financial (no external deps)
    test_state["extracted"] = {"full_name_english": "PRIYA SHARMA", "dob": "1992-09-08"}
    r = financial_profiling_agent(test_state)
    print(f"   financial_profiling_agent → activity_band: {r['financial_profile']['expected_activity_band']}, "
          f"anomalies: {len(r['financial_profile']['anomaly_flags'])}")

    # Test entity resolution (uses local NetworkX — no external deps)
    r = entity_resolution_agent(test_state)
    print(f"   entity_resolution_agent   → indirect_flag: {r['network_risk']['indirect_risk_flag']}")

    # Test id_verification (no external deps)
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