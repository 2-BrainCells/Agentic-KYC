"""
state.py — Shared agent state (the system's memory).

KYCState is the single contract every agent reads from and writes to. Each
agent receives the full state and returns ONLY the fields it changes;
LangGraph merges the return value back in automatically.

India-centric: Aadhaar (primary doc), PAN (secondary), INR currency,
FIU-IND/ED/PMLA/RBI watchlists, English + Devanagari scripts.

Imported by: agents.py, graph.py, tools.py, app.py.
"""

from __future__ import annotations
from typing import Annotated, Optional, TypedDict
from operator import add
from langgraph.graph import StateGraph, START, END  # noqa – re-exported for graph.py


# ── Constants ────────────────────────────────────────────────────────────────
# Use these instead of raw string literals so a typo can't silently break routing.

class CaseStatus:
    """Where the case currently sits in the pipeline."""
    RECEIVED    = "RECEIVED"
    EXTRACTING  = "EXTRACTING"
    VERIFYING   = "VERIFYING"
    SCREENING   = "SCREENING"
    REFINING    = "REFINING"       # self-correction loop running
    PROFILING   = "PROFILING"      # entity + financial running in parallel
    SCORING     = "SCORING"
    IN_REVIEW   = "IN_REVIEW"      # waiting for a human officer
    CLOSED      = "CLOSED"


class ScreeningStatus:
    """Output of the compliance screening agent."""
    CLEAR           = "CLEAR"
    AMBIGUOUS       = "AMBIGUOUS"       # triggers the self-correction loop
    POTENTIAL_MATCH = "POTENTIAL_MATCH"
    CONFIRMED_HIT   = "CONFIRMED_HIT"


class ExtractionStatus:
    """Output of the data extraction agent."""
    COMPLETE = "COMPLETE"
    PARTIAL  = "PARTIAL"    # some required fields missing
    FAILED   = "FAILED"


class UIDIAStatus:
    """Mocked UIDAI Aadhaar status (always ACTIVE in the demo)."""
    ACTIVE    = "ACTIVE"
    INACTIVE  = "INACTIVE"
    SUSPENDED = "SUSPENDED"


class Decision:
    """Final decision values — set by the risk agent or the human officer."""
    APPROVE            = "APPROVE"
    REVIEW             = "REVIEW"
    REJECT             = "REJECT"
    HOLD_FOR_DOCUMENTS = "HOLD_FOR_DOCUMENTS"


class Routing:
    """How the risk agent routes the case after scoring."""
    AUTO_APPROVE    = "AUTO_APPROVE"
    ROUTE_TO_HUMAN  = "ROUTE_TO_HUMAN"
    AUTO_REJECT     = "AUTO_REJECT"   # hard fail (e.g. ID not verified) → end, ask to reapply


class DecisionSource:
    """Who made the final decision."""
    AUTO  = "AUTO"
    HUMAN = "HUMAN"


class RiskBand:
    """Risk-score band labels (thresholds live in config.py)."""
    LOW         = "LOW"          # score < 0.30
    MEDIUM      = "MEDIUM"       # 0.30 – 0.50
    MEDIUM_HIGH = "MEDIUM_HIGH"  # 0.50 – 0.70
    HIGH        = "HIGH"         # > 0.70


class ActivityBand:
    """Expected account-activity band, derived from income."""
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# ── Shared state ─────────────────────────────────────────────────────────────

class KYCState(TypedDict, total=False):
    """
    The shared state passed through the LangGraph pipeline.

    total=False makes every field optional — agents return only the fields they
    change and LangGraph merges the rest. audit_log uses an `add` reducer so the
    two parallel agents (entity_resolution + financial_profiling) can both
    append without overwriting each other; every other field is last-write-wins.
    """

    # Stage 0 — Customer intake (written by intake, read by all)
    customer_id: str             # e.g. "CUST-IN-2024-001"
    declared: dict               # the customer's web-form data (name, dob, income ₹, …)
    documents: dict              # uploaded file paths (aadhaar_card, aadhaar_back, pan_card, salary_slip)
    received_at: str             # ISO timestamp

    # Stage 1 — Orchestrator control (written by graph.py routers/nodes)
    case_status: str             # a CaseStatus value, shown as the pipeline stage in the UI
    refine_count: int            # self-correction loops run so far; capped at REFINEMENT_MAX_TRIES

    # Stage 2 — Data extraction (written by extract; runs twice if self-correcting)
    extracted: dict              # identity fields read from Aadhaar + PAN
    field_confidence: dict       # per-field confidence 0.0–1.0
    bounding_boxes: dict         # per-field pixel boxes for the UI overlay
    low_confidence_fields: list  # fields with confidence < 0.75
    extraction_status: str       # an ExtractionStatus value

    # Stage 3 — ID verification (written by id_verify)
    id_verified: bool            # overall pass/fail
    verification_details: dict   # per-check breakdown + human-readable reasons

    # Stage 4 — Compliance screening (written by compliance; may run twice)
    compliance_hits: list        # watchlist matches; empty = CLEAR
    screening_status: str        # a ScreeningStatus value
    refinement_needed: bool      # True when AMBIGUOUS and refine_count < cap
    cache_hit: dict              # prior officer decision from the exception cache, or None

    # Stage 5 — Self-correction loop (written by refine)
    refinement_request: dict     # instructions for the extraction agent's retry
    current_agent_status: str    # free-text status shown in the UI

    # Stage 6 — Entity resolution (written by entity_resolution, runs in parallel)
    network_risk: dict           # MCA21/NetworkX corporate link analysis

    # Stage 7 — Financial profiling (written by financial, runs in parallel)
    financial_profile: dict      # income plausibility + anomaly flags + requires_edd

    # Stage 8 — Risk scoring & explanation (written by risk)
    risk_score: float            # weighted aggregate 0.0–1.0
    risk_band: str               # a RiskBand value
    decision: str                # a Decision value (the system recommendation)
    routing: str                 # a Routing value
    contributing_factors: list   # traceable per-factor breakdown for the officer
    explanation: str             # LLM-written, evidence-citing justification

    # Stage 9 — Human-in-the-loop review (written by hitl_review)
    human_decision: dict         # officer_id, decision, override, rationale, reviewed_at

    # Stage 10 — Active learning cache (written by active_learning_cache)
    cache_update: dict           # confirmation the human decision was stored

    # Final — closed-case record (written by the last node before END)
    final_decision: str          # authoritative outcome
    decision_source: str         # DecisionSource.AUTO or HUMAN
    closed_at: str               # ISO timestamp

    # Always — parallel-safe audit log (written by every agent)
    audit_log: Annotated[list, add]   # ordered list of plain-string actions


# ── Initial-state helper ─────────────────────────────────────────────────────

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
    received_at: str,
    pan_path: str          = "",   # optional, backward compatible
    aadhaar_back_path: str = "",   # back / address-QR side of the Aadhaar card
    salary_slip_path: str  = "",   # optional income proof, verified by financial agent
) -> KYCState:
    """
    Build the starting KYCState from the customer's submitted form data.

    Call this when the Streamlit Submit button is clicked, then pass the result
    to the compiled graph's invoke().
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
            "aadhaar_card": aadhaar_path,        # front face — photo, name, DOB, number
            "aadhaar_back": aadhaar_back_path,   # back — address + QR 'care_of'
            "pan_card"    : pan_path,            # legacy / optional
            "salary_slip" : salary_slip_path,    # optional income proof
        },

        audit_log = [
            f"{received_at} [Intake] {customer_id} received — "
            f"Aadhaar + PAN, ₹{income:,.0f} declared income"
        ],
    )


# ── Sanity check — run this file directly to confirm it imports cleanly ──────

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
