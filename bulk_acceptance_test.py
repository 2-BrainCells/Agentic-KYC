"""
bulk_acceptance_test.py — Bulk demo regression test.

Runs all 20 bulk customers through the compiled pipeline and checks each final
decision against the `expected_decision` in mock_data/bulk_customers.json. Run
it after ANY change to agents/tools/config — 20/20 PASS (12/5/3) means the
bulk-import demo climax is safe. Works offline (mock fallbacks) and on the MI300X.

    python bulk_acceptance_test.py
"""

import json
import sys

from config import BULK_DOSSIERS_PATH
from graph import build_graph
from state import create_initial_state
from datetime import datetime


def main() -> int:
    """Run every bulk profile and compare its decision to the expected value."""
    app = build_graph()

    with open(BULK_DOSSIERS_PATH, encoding="utf-8") as f:
        customers = json.load(f)

    passed, failed = 0, 0
    counts = {"APPROVE": 0, "REVIEW": 0, "REJECT": 0}

    print(f"{'ID':<10} {'Name':<28} {'Score':>5}  {'Loop':<4} {'Got':<8} {'Expected':<8} Result")
    print("-" * 80)

    for customer in customers:
        initial = create_initial_state(
            customer_id     = customer.get("customer_id", "BULK-?"),
            name            = customer.get("name", ""),
            dob             = customer.get("dob", ""),
            nationality     = customer.get("nationality", "Indian"),
            address         = customer.get("address", ""),
            pin_code        = customer.get("pin_code", ""),
            occupation      = customer.get("occupation", "other"),
            income          = float(customer.get("income", 0)),
            source_of_funds = customer.get("source_of_funds", ""),
            account_purpose = customer.get("account_purpose", ""),
            aadhaar_path    = customer.get("aadhaar_path", "/uploads/mock.jpg"),
            pan_path        = customer.get("pan_path", "/uploads/mock_pan.jpg"),
            received_at     = datetime.now().strftime("%Y-%m-%dT%H:%M:%S IST"),
        )
        # Same father's-name passthrough the Streamlit bulk loop applies.
        if customer.get("father_name"):
            initial["declared"]["father_name"] = customer["father_name"]

        final    = app.invoke(initial)
        decision = final.get("decision", "?")
        expected = customer.get("expected_decision", "?")
        score    = final.get("risk_score", 0.0)
        refines  = final.get("refine_count", 0)

        counts[decision] = counts.get(decision, 0) + 1
        ok = decision == expected
        passed += ok
        failed += not ok

        print(f"{customer['customer_id']:<10} {customer['name']:<28} "
              f"{score:>5.2f}  {'🔄' if refines else '—':<4} "
              f"{decision:<8} {expected:<8} {'PASS' if ok else '*** FAIL ***'}")

    print("-" * 80)
    print(f"Decisions: {counts.get('APPROVE',0)} APPROVE / "
          f"{counts.get('REVIEW',0)} REVIEW / {counts.get('REJECT',0)} REJECT "
          f"(target: 12 / 5 / 3)")
    print(f"Result: {passed}/{len(customers)} PASS")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
