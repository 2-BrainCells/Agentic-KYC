"""
graph.py — LangGraph pipeline wiring.

Flow: START → extract → id_verify → compliance → (refine ↺ extract) → fan_out →
{entity_resolution ‖ financial_profiling} → risk → AUTO_APPROVE=END or
ROUTE_TO_HUMAN → hitl_review → END.

LangGraph auto-joins the two parallel branches at risk (both have edges to it),
and audit_log's add-reducer lets them append safely. After hitl_review the graph
ends; the UI calls complete_case() when the officer decides.

    from graph import build_graph, complete_case
    app = build_graph(); final = app.invoke(create_initial_state(...))
    if final["routing"] == "ROUTE_TO_HUMAN":
        complete_case(final, "OFFICER-001", "APPROVE", "Clean after review")
"""

from __future__ import annotations
from datetime import datetime

from langgraph.graph import StateGraph, START, END

from state import KYCState, CaseStatus, Routing
from agents import (
    data_extraction_agent,
    id_verification_agent,
    compliance_screening_agent,
    refine_agent,
    entity_resolution_agent,
    financial_profiling_agent,
    risk_scoring_agent,
    hitl_review_agent,
    active_learning_cache_agent,
)


# ── Router functions (the central orchestrator's decisions) ─────────────────

def route_after_compliance(state: KYCState) -> str:
    """Ambiguous match wanting more signal → 'refine'; otherwise → 'fan_out'."""
    if state.get("refinement_needed", False):
        return "refine"
    return "fan_out"


def route_after_risk(state: KYCState) -> str:
    """
    Terminal routings end the graph; only ROUTE_TO_HUMAN goes to the queue.
    AUTO_APPROVE → 'auto' (END). AUTO_REJECT (e.g. failed ID verification) →
    'auto' (END) too — the customer is told to re-apply, no officer needed.
    Everything else → 'human' (review queue).
    """
    routing = state.get("routing", Routing.ROUTE_TO_HUMAN)
    if routing in (Routing.AUTO_APPROVE, Routing.AUTO_REJECT):
        return "auto"
    return "human"


# ── Fan-out node (passthrough that splits into two parallel branches) ───────

def fan_out_node(state: KYCState) -> dict:
    """
    Passthrough so LangGraph can run entity_resolution and financial_profiling
    in parallel; it waits for both before running risk.
    """
    return {
        "case_status": CaseStatus.PROFILING,
        "current_agent_status": "Running entity resolution and financial profiling in parallel",
    }


# ── Build graph ──────────────────────────────────────────────────────────────

def build_graph():
    """Assemble and compile the full KYC pipeline. Call once at startup."""
    g = StateGraph(KYCState)

    # Nodes
    g.add_node("extract",             data_extraction_agent)
    g.add_node("id_verify",           id_verification_agent)
    g.add_node("compliance",          compliance_screening_agent)
    g.add_node("refine",              refine_agent)
    g.add_node("fan_out",             fan_out_node)
    g.add_node("entity_resolution",   entity_resolution_agent)
    g.add_node("financial_profiling", financial_profiling_agent)
    g.add_node("risk",                risk_scoring_agent)
    g.add_node("hitl_review",         hitl_review_agent)

    # Sequential: START → extract → id_verify → compliance
    g.add_edge(START,        "extract")
    g.add_edge("extract",    "id_verify")
    g.add_edge("id_verify",  "compliance")

    # Self-correction loop OR proceed to profiling
    g.add_conditional_edges(
        "compliance",
        route_after_compliance,
        {
            "refine":   "refine",    # ambiguous → self-correction
            "fan_out":  "fan_out",   # clear/resolved → parallel profiling
        }
    )
    g.add_edge("refine", "extract")   # loop back for the second pass

    # Parallel fan-out then join at risk
    g.add_edge("fan_out", "entity_resolution")
    g.add_edge("fan_out", "financial_profiling")
    g.add_edge("entity_resolution",   "risk")
    g.add_edge("financial_profiling", "risk")

    # Final routing: auto-approve (END) or human review
    g.add_conditional_edges(
        "risk",
        route_after_risk,
        {
            "auto":        END,            # low risk → close automatically
            "human":       "hitl_review",  # medium/high → officer queue
        }
    )
    g.add_edge("hitl_review", END)   # UI calls complete_case() next

    return g.compile()


# ── HITL completion helper ──────────────────────────────────────────────────

def complete_case(
    state:     dict,
    officer_id: str,
    decision:   str,
    rationale:  str,
    override:   bool = False,
) -> dict:
    """
    Finish a human-routed case. Writes the officer's decision into state and
    runs active_learning_cache_agent to close the case and store the decision.
    Returns the FULL updated state (final_decision, closed_at, cache_update, and
    everything carried over).

    NOTE: active_learning_cache_agent follows the LangGraph contract and returns
    ONLY the fields it changes. Because we call it directly here (not through the
    graph), no automatic state merge happens — so we merge its partial result
    back over the input state ourselves. Skipping this drops human_decision,
    declared, risk_score, etc. from the stored closed state (the cause of the
    blank officer name / "the applicant" in the review banner).
    """
    updated_state = {
        **state,
        "human_decision": {
            "officer_id":  officer_id,
            "decision":    decision,
            "rationale":   rationale,
            "override":    override,
            "reviewed_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
        }
    }
    cache_result = active_learning_cache_agent(updated_state)

    merged = {**updated_state, **cache_result}
    # audit_log uses an add-reducer inside the graph; outside it we concatenate
    # by hand so the cache agent's line is appended, not replaced.
    merged["audit_log"] = (
        updated_state.get("audit_log", []) + cache_result.get("audit_log", [])
    )
    return merged


# ── Sanity check — run this file directly to exercise the full pipeline ─────

if __name__ == "__main__":
    from state import create_initial_state

    print("=" * 60)
    print("graph.py — Full Pipeline Sanity Check")
    print("=" * 60)

    app = build_graph()
    print("\n✅ Graph compiled successfully\n")

    # Test 1: clean customer → AUTO-APPROVE
    print("Test 1: Clean customer (Priya Sharma)")
    print("-" * 40)

    state_clean = create_initial_state(
        customer_id     = "CUST-TEST-CLEAN",
        name            = "Priya Sharma",
        dob             = "1992-09-08",
        nationality     = "Indian",
        address         = "B-204 Green Park Bengaluru",
        pin_code        = "560034",
        occupation      = "software engineer",
        income          = 1_200_000,
        source_of_funds = "Monthly salary from Infosys Limited",
        account_purpose = "Savings and investments",
        aadhaar_path    = "/uploads/test/priya_aadhaar.jpg",
        pan_path        = "/uploads/test/priya_pan.jpg",
        received_at     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
    )

    result_clean = app.invoke(state_clean)

    print(f"Decision  : {result_clean.get('decision')}")
    print(f"Risk score: {result_clean.get('risk_score')}")
    print(f"Routing   : {result_clean.get('routing')}")
    print(f"Refine cnt: {result_clean.get('refine_count', 0)}")
    print("\nAudit log:")
    for line in result_clean.get("audit_log", []):
        print(f"  {line}")

    # Test 2: high-risk customer → ROUTE_TO_HUMAN
    print("\n" + "=" * 60)
    print("Test 2: High-risk customer (Vikram Malhotra)")
    print("-" * 40)

    state_risk = create_initial_state(
        customer_id     = "CUST-TEST-HIGHRISK",
        name            = "Vikram Malhotra",
        dob             = "1968-11-14",
        nationality     = "Indian",
        address         = "34 Golf Links New Delhi",
        pin_code        = "110003",
        occupation      = "business owner",
        income          = 12_000_000,
        source_of_funds = "Business profits from multiple ventures",
        account_purpose = "Business transactions and overseas investment",
        aadhaar_path    = "/uploads/test/vikram_aadhaar.jpg",
        pan_path        = "/uploads/test/vikram_pan.jpg",
        received_at     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
    )
    # Father's name as it would appear in the Aadhaar QR 'care_of' field — the
    # extraction agent only surfaces it on the refinement pass, so the loop has
    # something to discover.
    state_risk["declared"]["father_name"] = "Ramesh Malhotra"

    result_risk = app.invoke(state_risk)

    print(f"Decision  : {result_risk.get('decision')}")
    print(f"Risk score: {result_risk.get('risk_score')}")
    print(f"Routing   : {result_risk.get('routing')}")
    print(f"Refine cnt: {result_risk.get('refine_count', 0)}")
    print("\nAudit log:")
    for line in result_risk.get("audit_log", []):
        print(f"  {line}")

    # Test 3: HITL completion simulation
    if result_risk.get("routing") == Routing.ROUTE_TO_HUMAN:
        print("\n" + "=" * 60)
        print("Test 3: Officer completes the HITL case")
        print("-" * 40)
        closed = complete_case(
            state      = result_risk,
            officer_id = "OFFICER-KYC-014",
            decision   = "HOLD_FOR_DOCUMENTS",
            rationale  = (
                "3-hop corporate chain noted but applicant not a named ED party. "
                "Requesting ITR and GST certificate before proceeding."
            ),
            override   = False,
        )
        print(f"Final decision : {closed.get('final_decision')}")
        print(f"Decision source: {closed.get('decision_source')}")
        print(f"Closed at      : {closed.get('closed_at')}")
        print(f"Cache update   : {closed.get('cache_update', {}).get('status')}")

    print("\n" + "=" * 60)
    print("All tests complete. Your pipeline is wired and ready.")
    print("Next file: app.py (Streamlit UI)")
    print("=" * 60)
