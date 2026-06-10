"""
graph.py — LangGraph Pipeline
==============================
Wires all agents into a single compiled graph.
This is the last backend file.

Full pipeline:

  START
    │
    ▼
  extract ◄─────────────────────────────────┐
    │                                        │ (self-correction loop)
    ▼                                        │
  id_verify                                  │
    │                                        │
    ▼                                        │
  compliance ──── refinement_needed? ────► refine
    │
    │  CLEAR / resolved
    ▼
  fan_out ──────┬─────────────────────────┐
                │                         │
                ▼                         ▼
        entity_resolution        financial_profiling
                │                         │
                └─────────┬───────────────┘
                           ▼
                          risk
                           │
               ┌───────────┴───────────┐
               │                       │
        AUTO_APPROVE             ROUTE_TO_HUMAN
               │                       │
              END               hitl_review
                                       │
                                      END
                                (Streamlit calls
                                complete_case() here)

Key design decisions:
  - fan_out is a passthrough node that splits into two parallel branches
  - LangGraph automatically JOINS entity_resolution + financial_profiling
    before running risk (because both have edges to risk)
  - audit_log uses Annotated[list, add] reducer so parallel branches
    can both append without overwriting each other
  - HITL: graph ends after hitl_review. Streamlit calls complete_case()
    when the officer decides, which runs active_learning_cache_agent.

Usage:
    from graph import build_graph, complete_case
    from state import create_initial_state

    app   = build_graph()
    state = create_initial_state(...)
    final = app.invoke(state)

    # If routed to human review:
    if final["routing"] == "ROUTE_TO_HUMAN":
        result = complete_case(final, "OFFICER-001", "APPROVE", "Clean after review")
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


# ============================================================
# ROUTER FUNCTIONS  (the Central Orchestrator's decisions)
# ============================================================

def route_after_compliance(state: KYCState) -> str:
    """
    Called after every compliance screening run.

    If the compliance agent flagged an ambiguous match AND
    wants more identity signal → route to refine (self-correction loop).

    Otherwise (clear, confirmed hit, or retries exhausted) →
    route to fan_out to start parallel profiling.
    """
    if state.get("refinement_needed", False):
        return "refine"
    return "fan_out"


def route_after_risk(state: KYCState) -> str:
    """
    Called after risk scoring.

    AUTO_APPROVE  → end the graph immediately
    anything else → send to human review queue
    """
    routing = state.get("routing", Routing.ROUTE_TO_HUMAN)
    if routing == Routing.AUTO_APPROVE:
        return "auto"
    return "human"


# ============================================================
# FAN-OUT NODE  (passthrough that splits into two parallel branches)
# ============================================================

def fan_out_node(state: KYCState) -> dict:
    """
    Passthrough — does no work itself.
    Exists purely so LangGraph can split execution into two
    parallel branches: entity_resolution AND financial_profiling.

    Both branches run at the same time. LangGraph automatically
    waits for BOTH to finish before running risk_scoring_agent.
    """
    return {
        "case_status": CaseStatus.PROFILING,
        "current_agent_status": "Running entity resolution and financial profiling in parallel",
    }


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():
    """
    Assembles and compiles the full KYC pipeline.

    Call this ONCE at app startup:
        app = build_graph()

    Then invoke for each customer:
        final_state = app.invoke(initial_state)
    """
    g = StateGraph(KYCState)

    # ── Register all nodes ────────────────────────────────────
    g.add_node("extract",             data_extraction_agent)
    g.add_node("id_verify",           id_verification_agent)
    g.add_node("compliance",          compliance_screening_agent)
    g.add_node("refine",              refine_agent)
    g.add_node("fan_out",             fan_out_node)
    g.add_node("entity_resolution",   entity_resolution_agent)
    g.add_node("financial_profiling", financial_profiling_agent)
    g.add_node("risk",                risk_scoring_agent)
    g.add_node("hitl_review",         hitl_review_agent)

    # ── Sequential flow: START → extract → id_verify → compliance ─
    g.add_edge(START,        "extract")
    g.add_edge("extract",    "id_verify")
    g.add_edge("id_verify",  "compliance")

    # ── Self-correction loop OR proceed to profiling ──────────────
    #    route_after_compliance returns "refine" or "fan_out"
    g.add_conditional_edges(
        "compliance",
        route_after_compliance,
        {
            "refine":   "refine",    # ambiguous → self-correction
            "fan_out":  "fan_out",   # clear/resolved → parallel profiling
        }
    )

    # Loop: refine bumps counter, sends back to extract for second pass
    g.add_edge("refine", "extract")

    # ── Parallel fan-out: fan_out → BOTH entity + financial ───────
    #    LangGraph runs these two simultaneously
    g.add_edge("fan_out", "entity_resolution")
    g.add_edge("fan_out", "financial_profiling")

    # ── Parallel join: BOTH branches converge on risk ─────────────
    #    LangGraph waits for both before running risk
    g.add_edge("entity_resolution",   "risk")
    g.add_edge("financial_profiling", "risk")

    # ── Final routing: auto-approve or human review ───────────────
    g.add_conditional_edges(
        "risk",
        route_after_risk,
        {
            "auto":        END,            # low risk → close automatically
            "human":       "hitl_review",  # medium/high → officer queue
        }
    )

    # ── Human review → END (Streamlit calls complete_case() next) ─
    g.add_edge("hitl_review", END)

    return g.compile()


# ============================================================
# HITL COMPLETION HELPER
# ============================================================

def complete_case(
    state:     dict,
    officer_id: str,
    decision:   str,
    rationale:  str,
    override:   bool = False,
) -> dict:
    """
    Call this from Streamlit when an officer makes their decision.

    The graph ends after hitl_review. This function picks up from
    there — it writes the human decision into state and runs
    active_learning_cache_agent to close the case and store the
    decision for future reference.

    Args:
        state:      The final state returned by app.invoke()
        officer_id: e.g. "OFFICER-KYC-014"
        decision:   Decision.APPROVE / REJECT / HOLD_FOR_DOCUMENTS
        rationale:  Officer's written reasoning
        override:   True if officer disagrees with system recommendation

    Returns:
        Updated state with final_decision, closed_at, cache_update

    Usage in Streamlit:
        if st.button("Approve"):
            final = complete_case(
                state      = case_state,
                officer_id = "OFFICER-001",
                decision   = "APPROVE",
                rationale  = "Identity verified, corporate link is 3 hops.",
                override   = True,
            )
            st.success(f"Case closed: {final['final_decision']}")
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
    return active_learning_cache_agent(updated_state)


# ============================================================
# SANITY CHECK  — run this file directly to verify the full
# pipeline executes end-to-end with mock data
# ============================================================

if __name__ == "__main__":
    from state import create_initial_state

    print("=" * 60)
    print("graph.py — Full Pipeline Sanity Check")
    print("=" * 60)

    app = build_graph()
    print("\n✅ Graph compiled successfully\n")

    # ── Test 1: Clean customer (should AUTO-APPROVE) ──────────────
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

    # ── Test 2: High-risk customer (should ROUTE_TO_HUMAN) ────────
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
    # Father's name as it would appear in the Aadhaar QR 'care_of' field.
    # The extraction agent only surfaces this on the refinement pass, so the
    # self-correction loop has something to discover (CLAUDE.md §4).
    state_risk["declared"]["father_name"] = "Ramesh Malhotra"

    result_risk = app.invoke(state_risk)

    print(f"Decision  : {result_risk.get('decision')}")
    print(f"Risk score: {result_risk.get('risk_score')}")
    print(f"Routing   : {result_risk.get('routing')}")
    print(f"Refine cnt: {result_risk.get('refine_count', 0)}")
    print("\nAudit log:")
    for line in result_risk.get("audit_log", []):
        print(f"  {line}")

    # ── Test 3: HITL completion simulation ────────────────────────
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