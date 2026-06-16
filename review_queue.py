"""
review_queue.py — Persistent human review inbox (the officer dashboard store).

Cases the pipeline routes to a human (routing == ROUTE_TO_HUMAN) are written to
a JSON file (config.REVIEW_QUEUE_PATH) so a compliance officer can work through
them later, whether they came from a single run or the bulk import. The file
backing means the inbox survives app restarts.

Pure storage — imports nothing from graph.py/agents.py, so it can never create
an import cycle. Each record: {customer_id, status, queued_at, summary, state,
(on close) final_decision, closed_at}.
"""

import json
import os
from datetime import datetime
from typing import Optional

from config import REVIEW_QUEUE_PATH

STATUS_PENDING = "PENDING"
STATUS_CLOSED  = "CLOSED"


def _now() -> str:
    """Current timestamp string used for queued_at / closed_at."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST")


def _load() -> list:
    """Read the queue file. Returns [] if it does not exist or is unreadable."""
    try:
        if not os.path.exists(REVIEW_QUEUE_PATH):
            return []
        with open(REVIEW_QUEUE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[review_queue] load failed: {e}")
        return []


def _save(records: list) -> bool:
    """Write the queue file (creating the folder if needed). Never raises."""
    try:
        folder = os.path.dirname(REVIEW_QUEUE_PATH)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(REVIEW_QUEUE_PATH, "w", encoding="utf-8") as f:
            # default=str keeps it robust against any stray non-JSON value
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[review_queue] save failed: {e}")
        return False


def _summary(state: dict) -> dict:
    """Compact list-view fields pulled out of a full KYCState."""
    declared = state.get("declared", {}) or {}
    return {
        "name":             declared.get("name", ""),
        "occupation":       declared.get("occupation", ""),
        "income":           declared.get("income", 0),
        "decision":         state.get("decision", "REVIEW"),
        "risk_score":       state.get("risk_score", 0.0),
        "risk_band":        state.get("risk_band", "?"),
        "screening_status": state.get("screening_status", ""),
        "refine_count":     state.get("refine_count", 0),
    }


def enqueue_case(state: dict) -> Optional[str]:
    """
    Add a pipeline result to the inbox as PENDING. Deduplicates by customer_id
    (re-running updates the entry rather than stacking duplicates) and never
    reopens a CLOSED case. Returns the customer_id, or None if already closed.
    """
    records = _load()
    cid = state.get("customer_id") or f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    for rec in records:
        if rec.get("customer_id") == cid:
            if rec.get("status") == STATUS_CLOSED:
                return None
            rec["state"]     = state
            rec["summary"]   = _summary(state)
            rec["queued_at"] = _now()
            _save(records)
            return cid

    records.append({
        "customer_id": cid,
        "status":      STATUS_PENDING,
        "queued_at":   _now(),
        "summary":     _summary(state),
        "state":       state,
    })
    _save(records)
    return cid


def list_cases(status: Optional[str] = None) -> list:
    """All cases, or only those with the given status (newest queued first)."""
    records = _load()
    if status:
        records = [r for r in records if r.get("status") == status]
    return list(reversed(records))


def get_case(customer_id: str) -> Optional[dict]:
    """Fetch a single case record by customer_id, or None."""
    for r in _load():
        if r.get("customer_id") == customer_id:
            return r
    return None


def close_case(customer_id: str, closed_state: dict) -> bool:
    """
    Mark a case CLOSED after complete_case() has run on its state, storing the
    officer-decided final state back so the queue keeps a full audit record.
    """
    records = _load()
    for rec in records:
        if rec.get("customer_id") == customer_id:
            rec["state"]          = closed_state
            rec["status"]         = STATUS_CLOSED
            rec["final_decision"] = (closed_state.get("final_decision")
                                     or closed_state.get("decision"))
            rec["closed_at"]      = closed_state.get("closed_at", _now())
            rec["summary"]        = _summary(closed_state)
            return _save(records)
    return False


def clear_queue() -> bool:
    """Empty the inbox — handy for resetting between demo runs."""
    return _save([])


def counts() -> dict:
    """Quick {pending, closed, total} tallies for the dashboard header."""
    records = _load()
    pending = sum(1 for r in records if r.get("status") == STATUS_PENDING)
    closed  = sum(1 for r in records if r.get("status") == STATUS_CLOSED)
    return {"pending": pending, "closed": closed, "total": len(records)}
