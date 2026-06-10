"""
state.py — Shared Agent State (The System's Memory)
====================================================
Owned by:  BOTH — agree on this file on Day 1, then freeze it.
           Neither person changes a field name without telling the other.

This is the single contract that every agent reads from and writes to.
Every agent receives the full state and returns ONLY the fields it changes.
LangGraph merges the return value back into the shared state automatically.

India-Centric:
  - Primary document : Aadhaar Card (UIDAI)
  - Secondary doc    : PAN Card
  - Currency         : INR (₹)
  - Compliance lists : FIU-IND, ED (Enforcement Directorate), PMLA, RBI
  - Scripts          : English + Devanagari (Hindi)

Compatible with: graph.py  (kyc_orchestration.py skeleton)
Imported by    : agents.py, graph.py, tools.py, app.py (Streamlit)
"""

from __future__ import annotations
from typing import Annotated, Optional, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END  # noqa – re-exported for graph.py


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  —  use these strings everywhere instead of raw literals
#               so a typo never creates a silent wrong decision
# ═══════════════════════════════════════════════════════════════════════════════

class CaseStatus:
    """Tracks where the case is in the pipeline."""
    RECEIVED    = "RECEIVED"
    EXTRACTING  = "EXTRACTING"
    VERIFYING   = "VERIFYING"
    SCREENING   = "SCREENING"
    REFINING    = "REFINING"       # self-correction loop is running
    PROFILING   = "PROFILING"      # entity + financial running in parallel
    SCORING     = "SCORING"
    IN_REVIEW   = "IN_REVIEW"      # waiting for human officer
    CLOSED      = "CLOSED"


class ScreeningStatus:
    """Output of the Compliance Screening Agent."""
    CLEAR           = "CLEAR"
    AMBIGUOUS       = "AMBIGUOUS"       # triggers self-correction loop
    POTENTIAL_MATCH = "POTENTIAL_MATCH"
    CONFIRMED_HIT   = "CONFIRMED_HIT"


class ExtractionStatus:
    COMPLETE = "COMPLETE"
    PARTIAL  = "PARTIAL"    # some fields missing — may need refinement pass
    FAILED   = "FAILED"


class UIDIAStatus:
    ACTIVE    = "ACTIVE"
    INACTIVE  = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class Decision:
    """Final decision values — used by risk agent and human officer."""
    APPROVE            = "APPROVE"
    REVIEW             = "REVIEW"
    REJECT             = "REJECT"
    HOLD_FOR_DOCUMENTS = "HOLD_FOR_DOCUMENTS"   # common Indian compliance outcome


class Routing:
    """How risk agent routes after scoring."""
    AUTO_APPROVE    = "AUTO_APPROVE"
    ROUTE_TO_HUMAN  = "ROUTE_TO_HUMAN"


class DecisionSource:
    AUTO  = "AUTO"    # system decided without human
    HUMAN = "HUMAN"   # officer made or overrode the decision


class RiskBand:
    LOW         = "LOW"          # score < 0.30
    MEDIUM      = "MEDIUM"       # 0.30 – 0.50
    MEDIUM_HIGH = "MEDIUM_HIGH"  # 0.50 – 0.70
    HIGH        = "HIGH"         # > 0.70


class ActivityBand:
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# ═══════════════════════════════════════════════════════════════════════════════
# KYSTATE  —  the shared memory that every agent reads and writes
# ═══════════════════════════════════════════════════════════════════════════════

class KYCState(TypedDict, total=False):
    """
    The complete shared state passed through the LangGraph pipeline.

    total=False means every field is Optional — agents only return the
    fields they change; LangGraph merges everything else automatically.

    IMPORTANT: audit_log uses an `add` reducer (see Annotated below).
    This means when two agents run IN PARALLEL (entity_resolution +
    financial_profiling), both can safely append to audit_log without
    overwriting each other. All other fields are last-write-wins.

    Field groups (scroll down):
      Stage 0  – Customer Intake
      Stage 1  – Orchestrator Control
      Stage 2  – Data Extraction (Aadhaar + PAN)
      Stage 3  – ID Verification
      Stage 4  – Compliance Screening
      Stage 5  – Self-Correction Loop
      Stage 6  – Entity Resolution
      Stage 7  – Financial Profiling
      Stage 8  – Risk Scoring & Explanation
      Stage 9  – Human-in-the-Loop Review
      Stage 10 – Active Learning Cache
      Final    – Closed Case Record
      Always   – Audit Log (parallel-safe)
    """

    # ── STAGE 0: Customer Intake ──────────────────────────────────────────────
    # Written by: intake node
    # Read by: all agents

    customer_id: str
    # e.g. "CUST-IN-2024-001"

    declared: dict
    # Everything the customer typed on the web form:
    # {
    #   "name"          : "Priya Sharma",
    #   "dob"           : "1992-09-08",
    #   "nationality"   : "Indian",
    #   "address"       : "B-204, Green Park Apartments, Koramangala,
    #                      Bengaluru, Karnataka — 560034",
    #   "pin_code"      : "560034",
    #   "occupation"    : "Software Engineer",
    #   "income"        : 1200000,        ← INR, not USD
    #   "source_of_funds": "Monthly salary from Infosys Limited",
    #   "account_purpose": "Savings and mutual fund investments"
    # }

    documents: dict
    # Paths to uploaded files:
    # {
    #   "aadhaar_card"    : "/uploads/CUST-IN-2024-001/aadhaar.jpg",
    #   "pan_card"        : "/uploads/CUST-IN-2024-001/pan.jpg",
    #   "proof_of_address": "/uploads/CUST-IN-2024-001/utility_bill.jpg"
    # }

    received_at: str
    # "2024-11-14T10:15:22 IST"


    # ── STAGE 1: Orchestrator Control ─────────────────────────────────────────
    # Written by: orchestrator / router functions in graph.py
    # Read by: routers and the Streamlit UI for status display

    case_status: str
    # One of CaseStatus constants above
    # Displayed as the current pipeline stage in the UI

    refine_count: int
    # How many self-correction loops have run so far.
    # Starts at 0. Capped at 1 in route_after_compliance() in graph.py.
    # Prevents infinite loops if the model can't find the disambiguating field.


    # ── STAGE 2: Data Extraction Agent ───────────────────────────────────────
    # Written by: extract node (runs twice if self-correction triggered)
    # Read by: id_verify, compliance nodes

    extracted: dict
    # All fields read from the Aadhaar card + PAN card:
    # {
    #   # --- From Aadhaar card face (OCR) ---
    #   "full_name_english"   : "PRIYA SHARMA",
    #   "full_name_devanagari": "प्रिया शर्मा",
    #   "given_name"          : "PRIYA",
    #   "surname"             : "SHARMA",
    #   "father_name"         : "RAMESH SHARMA",   ← from QR 'care_of' field
    #   "dob"                 : "1992-09-08",
    #   "gender"              : "F",
    #   "address"             : "B-204 Green Park Apartments Koramangala
    #                            Bengaluru Karnataka 560034",
    #   "pin_code"            : "560034",
    #
    #   # --- Aadhaar number (always masked — last 4 only) ---
    #   "aadhaar_last4"       : "7823",
    #   "aadhaar_masked"      : "XXXX XXXX 7823",
    #
    #   # --- QR code verification ---
    #   "qr_verified"         : True,    ← QR decoded and matches card face
    #   "uidai_status"        : "ACTIVE", ← mocked UIDAI API call
    #
    #   # --- From PAN card ---
    #   "pan_number"          : "BKXPS9876M",
    #   "pan_name"            : "PRIYA SHARMA",
    #   "pan_dob_match"       : True,
    #
    #   # --- Extraction metadata ---
    #   "extraction_pass"     : 0,        ← 0 = first pass, 1 = refinement
    #   "active_status"       : "ACTIVE"
    # }

    field_confidence: dict
    # Confidence score (0.0 – 1.0) per extracted field:
    # {
    #   "full_name_english"   : 0.98,
    #   "father_name"         : 0.00,   ← 0.00 means not found on this pass
    #   "dob"                 : 0.99,
    #   "aadhaar_last4"       : 0.99,
    #   "pan_number"          : 0.97
    # }

    bounding_boxes: dict
    # Pixel coordinates of each field on the Aadhaar card image.
    # Used to draw the coloured overlay in the Streamlit UI.
    # Format: { "field_name": {"x": int, "y": int, "w": int, "h": int} }
    # {
    #   "full_name_english": {"x": 210, "y": 180, "w": 320, "h": 30},
    #   "dob"              : {"x": 210, "y": 230, "w": 200, "h": 28},
    #   "aadhaar_number"   : {"x": 140, "y": 380, "w": 280, "h": 32}
    # }

    low_confidence_fields: list
    # Fields with confidence < 0.75 — displayed in amber/red in the UI.
    # e.g. ["father_name", "place_of_birth"] on first extraction pass.

    extraction_status: str
    # One of ExtractionStatus constants.
    # PARTIAL means some required fields are missing — may trigger refinement.


    # ── STAGE 3: ID Verification Agent ───────────────────────────────────────
    # Written by: id_verify node
    # Read by: compliance, risk nodes

    id_verified: bool
    # Overall pass/fail — True means the document checks out.
    # False means do not proceed with this application.

    verification_details: dict
    # Full breakdown of what was checked:
    # {
    #   "name_match"         : {"result": True, "score": 0.99,
    #                           "note": "exact match English + Devanagari"},
    #   "dob_match"          : True,
    #   "address_match"      : {"result": True, "score": 0.95},
    #   "qr_checksum_valid"  : True,
    #   "uidai_active"       : True,
    #   "pan_aadhaar_linked" : True,    ← mandatory per RBI 2023 mandate
    #   "authenticity_flags" : [],      ← list of any tamper/anomaly notes
    #   "reasons"            : [
    #       "Name: exact match English and Devanagari scripts",
    #       "DOB: exact match",
    #       "Aadhaar QR valid, UIDAI active",
    #       "PAN-Aadhaar linkage confirmed"
    #   ]
    # }


    # ── STAGE 4: Compliance Screening Agent ──────────────────────────────────
    # Written by: compliance node (may run twice — before and after refinement)
    # Read by: orchestrator router, risk node

    compliance_hits: list
    # List of potential watchlist matches. Empty = CLEAR.
    # Each item:
    # {
    #   "matched_name"  : "Vikram Suresh Malhotra",
    #   "list_source"   : "ED — PMLA Case No. 2019/DEL/0147",
    #   "match_score"   : 0.81,         ← 0.75–0.90 = AMBIGUOUS band
    #   "match_type"    : "FUZZY_GIVEN_SURNAME_MATCH",
    #   "matched_fields": ["given_name: exact", "surname: exact"],
    #   "reason"        : "No father name to disambiguate. DOB within range."
    # }

    screening_status: str
    # One of ScreeningStatus constants.
    # AMBIGUOUS (score 0.75–0.90 with no disambiguator) → triggers loop.

    refinement_needed: bool
    # True when screening_status is AMBIGUOUS and refine_count < cap.
    # The orchestrator router reads this to decide whether to loop back.

    cache_hit: dict
    # If the Human Exception Cache returns a prior decision for this profile:
    # {"decision": "APPROVE_WITH_EDD", "rationale": "3-hop link insufficient"}
    # None if no prior case matches.


    # ── STAGE 5: Self-Correction Loop ────────────────────────────────────────
    # Written by: refine node (the bump_refine_counter node in graph.py)
    # Read by: extract node on second pass

    refinement_request: dict
    # Instructions sent back to the extraction agent:
    # {
    #   "fields_to_seek": ["father_name", "care_of_name"],
    #   "instruction"   : "Parse Aadhaar QR 'care_of' field for relative name.",
    #   "temperature"   : 0.7
    # }

    current_agent_status: str
    # Free-text status shown in the Streamlit UI during processing.
    # e.g. "REFINEMENT_REQUEST_ISSUED"
    # e.g. "Re-parsing Aadhaar QR for father's name..."
    # e.g. "Autonomously resolved — ED match cleared"


    # ── STAGE 6: Entity Resolution Agent ─────────────────────────────────────
    # Written by: entity_resolution node (runs in PARALLEL with financial)
    # Read by: risk node

    network_risk: dict
    # Output of the MCA21 / NetworkX graph traversal:
    # {
    #   "linked_entities": [
    #       {"name": "VMK Holdings Pvt Ltd", "type": "director",
    #        "mca21_status": "ACTIVE", "sanction_status": "CLEAN"},
    #       {"name": "Orion Global Trade Solutions LLP",
    #        "mca21_status": "ACTIVE",
    #        "sanction_status": "UNDER ED INVESTIGATION — PMLA 2021/DEL/0892"}
    #   ],
    #   "shortest_path_to_sanctioned": {
    #       "path": ["Vikram Malhotra", "VMK Holdings",
    #                "Sunrise Export Import", "Orion Global Trade Solutions"],
    #       "hop_distance": 3
    #   },
    #   "indirect_risk_flag": True,
    #   "link_explanation" : "3-hop chain via MCA21 to ED-investigated entity."
    # }
    # For clean customers: {"indirect_risk_flag": False,
    #                       "shortest_path_to_sanctioned": None,
    #                       "linked_entities": [], "link_explanation": "Clean"}


    # ── STAGE 7: Financial Profiling Agent ───────────────────────────────────
    # Written by: financial node (runs in PARALLEL with entity_resolution)
    # Read by: risk node

    financial_profile: dict
    # Output of the income plausibility + anomaly checks:
    # {
    #   "expected_activity_band"  : "HIGH",   ← LOW / MEDIUM / HIGH
    #   "plausibility": {
    #       "income_vs_occupation": "UNSUBSTANTIATED",
    #       "note"                : "₹1.2 Crore for unverified consultant",
    #       "source_of_funds"     : "VAGUE",
    #       "note"                : "No GST number or business registration"
    #   },
    #   "anomaly_flags": [
    #       "₹1.2 Crore declared with no GST registration",
    #       "Source of funds too vague for HNI income band",
    #       "Account purpose 'overseas investment' + unsubstantiated income",
    #       "No verifiable employer or business entity"
    #   ],
    #   "financial_risk_contribution": 0.75   ← this agent's share of risk
    # }


    # ── STAGE 8: Risk Scoring & Explanation Agent ─────────────────────────────
    # Written by: risk node
    # Read by: orchestrator router, Streamlit UI, HITL panel

    risk_score: float
    # Weighted aggregate of all agent contributions. Range: 0.0 – 1.0.
    # Weights: ID 30% + Compliance 40% + Network 20% + Financial 10%

    risk_band: str
    # One of RiskBand constants — derived from risk_score.

    decision: str
    # One of Decision constants — the system's recommendation.

    routing: str
    # One of Routing constants — how the graph routes after scoring.
    # "auto" → END   |   "human" → hitl_review node

    contributing_factors: list
    # Fully traceable breakdown — shown in the HITL review panel:
    # [
    #   {"factor": "ID Verification",   "weight": 0.30,
    #    "evidence": "PASS — Aadhaar QR valid, PAN linked",
    #    "score_contribution": 0.02},
    #   {"factor": "Compliance Screen",  "weight": 0.40,
    #    "evidence": "CLEAR after refinement — ED match ruled out",
    #    "score_contribution": 0.00},
    #   {"factor": "Network Risk",       "weight": 0.20,
    #    "evidence": "FLAGGED — 3-hop MCA21 chain to ED entity",
    #    "score_contribution": 0.18},
    #   {"factor": "Financial Profile",  "weight": 0.10,
    #    "evidence": "HIGH — ₹1.2Cr unsubstantiated, 4 anomaly flags",
    #    "score_contribution": 0.47}
    # ]

    explanation: str
    # The natural-language justification generated by the LLM.
    # Must cite SPECIFIC evidence, not a generic paragraph.
    # This is what the human officer reads in the review dashboard.


    # ── STAGE 9: Human-in-the-Loop Review ────────────────────────────────────
    # Written by: hitl_review node
    # Read by: active_learning_cache node, final case record

    human_decision: dict
    # Only populated for cases routed to human review:
    # {
    #   "officer_id"    : "OFFICER-KYC-014",
    #   "decision"      : "HOLD_FOR_DOCUMENTS",
    #   "override"      : False,
    #   "rationale"     : "3-hop chain insufficient for rejection.
    #                      HNI income requires ITR + GST certificate.",
    #   "reviewed_at"   : "2024-11-14T16:12:44 IST"
    # }


    # ── STAGE 10: Active Learning Cache ──────────────────────────────────────
    # Written by: active_learning_cache node

    cache_update: dict
    # Confirmation that the human decision was stored:
    # {
    #   "status"   : "STORED",
    #   "timestamp": "2024-11-14T16:12:47 IST"
    # }


    # ── FINAL: Closed Case Record ─────────────────────────────────────────────
    # Written by: the last node before END

    final_decision: str
    # The authoritative final outcome — Decision constant.

    decision_source: str
    # DecisionSource.AUTO or DecisionSource.HUMAN

    closed_at: str
    # ISO timestamp when the case was closed.


    # ── ALWAYS: Audit Log (parallel-safe) ────────────────────────────────────
    # Written by: EVERY agent node
    # The Annotated[list, add] reducer means when two agents run in
    # PARALLEL (entity_resolution + financial), both can append their
    # log entries and LangGraph will MERGE them — not overwrite.

    audit_log: Annotated[list, add]
    # Ordered list of every action taken, by whom, and why.
    # This is the explainability backbone — displayed in full in the UI.
    # Each entry is a plain string:
    # "10:15:27 [Extraction]  COMPLETE — QR decoded, 0 low-confidence fields"
    # "10:15:29 [Compliance]  CLEAR — 0 hits on FIU-IND, ED, PMLA, UN"
    # "14:30:21 [Compliance]  AMBIGUOUS — 0.81 match to ED entry → refinement"
    # "14:30:21 [Orchestrator] SELF-CORRECTION → Refinement Request issued"


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER — create the initial state from the Streamlit form inputs
# ═══════════════════════════════════════════════════════════════════════════════

def create_initial_state(
    customer_id: str,
    name: str,
    dob: str,
    nationality: str,
    address: str,
    pin_code: str,
    occupation: str,
    income: float,              # INR
    source_of_funds: str,
    account_purpose: str,
    aadhaar_path: str,
    pan_path: str,
    received_at: str,
) -> KYCState:
    """
    Builds the starting KYCState from what the customer typed on the form.
    Call this in your Streamlit app when the Submit button is clicked.

    Example:
        state = create_initial_state(
            customer_id   = "CUST-IN-2024-001",
            name          = "Priya Sharma",
            dob           = "1992-09-08",
            nationality   = "Indian",
            address       = "B-204, Green Park Apartments, Bengaluru",
            pin_code      = "560034",
            occupation    = "Software Engineer",
            income        = 1200000,
            source_of_funds = "Monthly salary from Infosys Limited",
            account_purpose = "Savings and mutual fund investments",
            aadhaar_path  = "/uploads/CUST-IN-2024-001/aadhaar.jpg",
            pan_path      = "/uploads/CUST-IN-2024-001/pan.jpg",
            received_at   = "2024-11-14T10:15:22 IST",
        )
        final_state = app.invoke(state)
    """
    return KYCState(
        customer_id  = customer_id,
        received_at  = received_at,
        case_status  = CaseStatus.RECEIVED,
        refine_count = 0,

        declared = {
            "name"           : name,
            "dob"            : dob,
            "nationality"    : nationality,
            "address"        : address,
            "pin_code"       : pin_code,
            "occupation"     : occupation,
            "income"         : income,          # ₹ INR
            "source_of_funds": source_of_funds,
            "account_purpose": account_purpose,
        },

        documents = {
            "aadhaar_card": aadhaar_path,
            "pan_card"    : pan_path,
        },

        audit_log = [
            f"{received_at} [Intake] {customer_id} received — "
            f"Aadhaar + PAN, ₹{income:,.0f} declared income"
        ],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK — run this file directly to confirm it imports cleanly
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from datetime import datetime

    test_state = create_initial_state(
        customer_id     = "CUST-IN-2024-TEST",
        name            = "Priya Sharma",
        dob             = "1992-09-08",
        nationality     = "Indian",
        address         = "B-204 Green Park Apartments Koramangala Bengaluru",
        pin_code        = "560034",
        occupation      = "Software Engineer",
        income          = 1_200_000,
        source_of_funds = "Monthly salary from Infosys Limited",
        account_purpose = "Savings and mutual fund investments",
        aadhaar_path    = "/uploads/test/aadhaar.jpg",
        pan_path        = "/uploads/test/pan.jpg",
        received_at     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
    )

    print("✅ KYCState created successfully")
    print(f"   customer_id  : {test_state['customer_id']}")
    print(f"   case_status  : {test_state['case_status']}")
    print(f"   income (INR) : ₹{test_state['declared']['income']:,.0f}")
    print(f"   audit_log    : {test_state['audit_log'][0]}")
    print("\nAll constants:")
    print(f"   CaseStatus   : {[v for k,v in vars(CaseStatus).items() if not k.startswith('_')]}")
    print(f"   Decision     : {[v for k,v in vars(Decision).items() if not k.startswith('_')]}")
    print(f"   RiskBand     : {[v for k,v in vars(RiskBand).items() if not k.startswith('_')]}")