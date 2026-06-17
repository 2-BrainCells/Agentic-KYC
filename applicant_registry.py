"""
applicant_registry.py — Persistent applicant registry (the onboarding ledger).

Every application that reaches a FINAL outcome (auto-approved, auto-rejected, or
closed by an officer) is recorded here, keyed by a unique application number. It
powers the re-application guard used at intake:

  • an already-ACCEPTED applicant is told they're already registered, and
  • a recently-REJECTED applicant must wait out a cooldown before retrying.

Storage is a small on-disk SQLite database (config.APPLICANT_DB_PATH). Like
review_queue.py this module is pure storage — it imports ONLY config + stdlib,
so it can never create an import cycle with agents/graph, and every call is
best-effort (a DB hiccup never crashes the demo).

    from applicant_registry import (
        generate_application_number, precheck, record_decision,
    )
    app_no = generate_application_number()
    status, rec, *rest = precheck(name, dob)      # ("OK"|"ALREADY_ACCEPTED"|"REJECTED_COOLDOWN", ...)
    record_decision(app_no, customer_id, name, dob, "APPROVE", 0.06, payload)
"""

import json
import os
import re
import sqlite3
import secrets
import difflib
from datetime import datetime, timedelta

from config import (
    APPLICANT_DB_PATH,
    REAPPLY_COOLDOWN_MINUTES,
    APPLICATION_NUMBER_PREFIX,
)

# Decision ->registry status mapping.
STATUS_ACCEPTED = "ACCEPTED"
STATUS_REJECTED = "REJECTED"
STATUS_HOLD     = "HOLD"

_DECISION_TO_STATUS = {
    "APPROVE":            STATUS_ACCEPTED,
    "REJECT":             STATUS_REJECTED,
    "HOLD_FOR_DOCUMENTS": STATUS_HOLD,
    "REVIEW":             STATUS_HOLD,   # an undecided review shouldn't block re-apply
}

# Name-match tolerance for the dedup lookup (DOB must match exactly).
_NAME_MATCH_RATIO = 0.90


# ── Low-level helpers ───────────────────────────────────────────────────────

def _now() -> str:
    """Wall-clock timestamp string stored on each row."""
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST")


def _norm_name(name: str) -> str:
    """Lowercase + collapse whitespace so 'Arjun  Mehta' == 'arjun mehta'."""
    return re.sub(r"\s+", " ", (name or "").strip().lower())


def _norm_dob(dob: str) -> str:
    """Light DOB canonicaliser — strips separators to YYYYMMDD-ish for equality."""
    return re.sub(r"[^0-9]", "", (dob or "").strip())


def _conn() -> sqlite3.Connection:
    """
    Open the registry DB, creating the folder + table on first use. Rows are
    returned as dict-like sqlite3.Row objects.
    """
    folder = os.path.dirname(APPLICANT_DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    conn = sqlite3.connect(APPLICANT_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            application_number TEXT PRIMARY KEY,
            customer_id        TEXT,
            name               TEXT,
            name_norm          TEXT,
            dob                TEXT,
            dob_norm           TEXT,
            decision           TEXT,
            status             TEXT,
            risk_score         REAL,
            created_at         TEXT,
            decided_at         TEXT,
            payload            TEXT
        )
    """)
    conn.commit()
    return conn


# ── Public API ──────────────────────────────────────────────────────────────

def generate_application_number() -> str:
    """
    Return a unique, human-readable application number, e.g.
    'KYC-APP-20260617094131-a1b'. The timestamp keeps it sortable/readable; the
    3-hex random suffix guarantees uniqueness even within the same second.
    """
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{APPLICATION_NUMBER_PREFIX}-{stamp}-{secrets.token_hex(2)[:3]}"


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Convert a sqlite3.Row to a plain dict, decoding the JSON payload."""
    d = dict(row)
    try:
        d["payload"] = json.loads(d.get("payload") or "{}")
    except Exception:
        d["payload"] = {}
    return d


def precheck(name: str, dob: str):
    """
    Look up a prior FINAL decision for this person (matched by fuzzy name +
    exact DOB) and decide whether they may re-apply.

    Returns a tuple whose first element is the status:
      ("OK", None)                                  ->no blocker, proceed
      ("ALREADY_ACCEPTED", record)                  ->already onboarded
      ("REJECTED_COOLDOWN", record, minutes_left)   ->rejected, still in cooldown

    A rejection whose cooldown has elapsed returns ("OK", None) — the applicant
    may try again. HOLD/REVIEW rows never block. Never raises.
    """
    try:
        target_name = _norm_name(name)
        target_dob  = _norm_dob(dob)
        if not target_name or not target_dob:
            return ("OK", None)

        with _conn() as conn:
            rows = conn.execute(
                "SELECT * FROM applications WHERE dob_norm = ? "
                "ORDER BY decided_at DESC",
                (target_dob,),
            ).fetchall()

        for row in rows:
            rec = _row_to_dict(row)
            same_name = (
                rec.get("name_norm") == target_name
                or difflib.SequenceMatcher(
                    None, rec.get("name_norm", ""), target_name
                ).ratio() >= _NAME_MATCH_RATIO
            )
            if not same_name:
                continue

            if rec.get("status") == STATUS_ACCEPTED:
                return ("ALREADY_ACCEPTED", rec)

            if rec.get("status") == STATUS_REJECTED:
                decided = _parse_ts(rec.get("decided_at"))
                if decided is not None:
                    elapsed = datetime.now() - decided
                    cooldown = timedelta(minutes=REAPPLY_COOLDOWN_MINUTES)
                    if elapsed < cooldown:
                        minutes_left = max(
                            1, int((cooldown - elapsed).total_seconds() // 60) + 1
                        )
                        return ("REJECTED_COOLDOWN", rec, minutes_left)
                # cooldown elapsed (or unparseable timestamp) ->allow re-apply
        return ("OK", None)

    except Exception as e:
        print(f"[applicant_registry] precheck failed (non-critical): {e}")
        return ("OK", None)


def record_decision(
    application_number: str,
    customer_id:        str,
    name:               str,
    dob:                str,
    decision:           str,
    risk_score:         float = 0.0,
    payload:            dict  = None,
) -> bool:
    """
    Upsert a final decision into the registry. Maps the decision to a status
    (APPROVE→ACCEPTED, REJECT→REJECTED, HOLD_FOR_DOCUMENTS→HOLD). Returns True on
    success. Best-effort — a failure is logged and swallowed.
    """
    try:
        status = _DECISION_TO_STATUS.get(decision, STATUS_HOLD)
        now    = _now()
        with _conn() as conn:
            existing = conn.execute(
                "SELECT created_at FROM applications WHERE application_number = ?",
                (application_number,),
            ).fetchone()
            created_at = existing["created_at"] if existing else now
            conn.execute(
                """
                INSERT INTO applications (
                    application_number, customer_id, name, name_norm,
                    dob, dob_norm, decision, status, risk_score,
                    created_at, decided_at, payload
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(application_number) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    name        = excluded.name,
                    name_norm   = excluded.name_norm,
                    dob         = excluded.dob,
                    dob_norm    = excluded.dob_norm,
                    decision    = excluded.decision,
                    status      = excluded.status,
                    risk_score  = excluded.risk_score,
                    decided_at  = excluded.decided_at,
                    payload     = excluded.payload
                """,
                (
                    application_number, customer_id, name, _norm_name(name),
                    dob, _norm_dob(dob), decision, status, float(risk_score or 0.0),
                    created_at, now, json.dumps(payload or {}, ensure_ascii=False,
                                                default=str),
                ),
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[applicant_registry] record_decision failed (non-critical): {e}")
        return False


def get_application(application_number: str):
    """Fetch a single application row as a dict, or None."""
    try:
        with _conn() as conn:
            row = conn.execute(
                "SELECT * FROM applications WHERE application_number = ?",
                (application_number,),
            ).fetchone()
        return _row_to_dict(row) if row else None
    except Exception as e:
        print(f"[applicant_registry] get_application failed: {e}")
        return None


def list_applications(status: str = None) -> list:
    """All applications (newest decided first), optionally filtered by status."""
    try:
        with _conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM applications WHERE status = ? "
                    "ORDER BY decided_at DESC", (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM applications ORDER BY decided_at DESC"
                ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[applicant_registry] list_applications failed: {e}")
        return []


def _parse_ts(ts: str):
    """Parse a stored 'YYYY-MM-DDTHH:MM:SS IST' timestamp back to a datetime."""
    if not ts:
        return None
    try:
        return datetime.strptime(ts.replace(" IST", ""), "%Y-%m-%dT%H:%M:%S")
    except Exception:
        return None


# ── Sanity check — run this file directly to exercise the registry ──────────

if __name__ == "__main__":
    print("applicant_registry.py — self-test")
    app_no = generate_application_number()
    print(f"  generated application number: {app_no}")

    record_decision(app_no, "CUST-TEST-1", "Priya Sharma", "1992-09-08",
                    "APPROVE", 0.06, {"occupation": "software engineer"})
    print(f"  recorded APPROVE for Priya Sharma")

    status, *rest = precheck("Priya Sharma", "1992-09-08")
    print(f"  precheck(Priya 1992-09-08) ->{status}  (expected ALREADY_ACCEPTED)")

    rej_no = generate_application_number()
    record_decision(rej_no, "CUST-TEST-2", "Arjun Mehta", "1974-06-22",
                    "REJECT", 0.40, {})
    status, *rest = precheck("Arjun Mehta", "1974-06-22")
    print(f"  precheck(Arjun 1974-06-22) ->{status} {rest}  "
          f"(expected REJECTED_COOLDOWN, minutes_left)")

    status, *rest = precheck("Someone New", "2000-01-01")
    print(f"  precheck(new person)        ->{status}  (expected OK)")
    print("  [OK] self-test complete")
