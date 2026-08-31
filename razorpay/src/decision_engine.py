"""
decision_engine.py
---------------------
Takes scoring_logic's output and actually DOES something with it:
  low    -> no action
  medium -> create a Razorpay Payment Link (soft nudge toward prepayment)
  high   -> flag for human review (never auto-cancel)

MODES:
  DRY_RUN (default) -- simulates the Razorpay call, logs what WOULD have
    happened. No API keys needed. This is what you build and demo with
    today.
  LIVE -- makes a real Razorpay test-mode API call. Requires two
    environment variables:
      RAZORPAY_KEY_ID
      RAZORPAY_KEY_SECRET
    Switch modes by setting:  RAZORPAY_MODE=live
    (Never hardcode keys in this file or commit them to git.)

GUARDRAILS IN THIS FILE:
  - No auto-cancellation anywhere, at any tier. Ever.
  - If a payment-link action fails (bad key, network issue, invalid
    amount), the failure is caught, logged, and the order is escalated
    to the human-review queue as a fallback -- it never fails silently
    and never crashes the batch.
  - Every action attempt (success or failure) is logged with a
    timestamp, independent of the scoring audit log in scoring_logic.py.

WHERE THIS FILE LIVES:
    razorpay/src/decision_engine.py

HOW TO RUN THE BUILT-IN SMOKE TEST:
    cd razorpay
    python src/decision_engine.py
"""

import os
import json
from pathlib import Path
from datetime import datetime, timezone

import scoring_logic as sl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ACTION_LOG_PATH = REPORTS_DIR / "action_log.jsonl"
REVIEW_QUEUE_PATH = REPORTS_DIR / "human_review_queue.jsonl"

RAZORPAY_MODE = os.environ.get("RAZORPAY_MODE", "dry_run")  # "dry_run" or "live"


def _log_action(entry: dict):
    entry["logged_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(ACTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _flag_for_human_review(order: dict, result: dict, reason: str):
    entry = {
        "order_id": order["order_id"],
        "risk_score": result["risk_score"],
        "reason": reason,
        "flagged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "order": order,
        "scoring_result": result,
    }
    with open(REVIEW_QUEUE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _create_payment_link_dry_run(order: dict, amount_inr: float) -> dict:
    """Simulates what a Razorpay Payment Link creation would return.
    No network call, no keys required."""
    return {
        "status": "simulated_success",
        "mode": "dry_run",
        "short_url": f"https://rzp.io/i/DRYRUN_{order['order_id']}",
        "amount": amount_inr,
        "note": "This is a SIMULATED link -- no real Razorpay API call was made.",
    }


def _create_payment_link_live(order: dict, amount_inr: float) -> dict:
    """Real Razorpay test-mode API call. Requires razorpay SDK + env vars.
    Raises on failure -- caller is responsible for catching it."""
    import razorpay  # imported here, not at module top, so dry_run mode
                       # never requires this package to be installed at all

    key_id = os.environ["RAZORPAY_KEY_ID"]
    key_secret = os.environ["RAZORPAY_KEY_SECRET"]
    client = razorpay.Client(auth=(key_id, key_secret))

    payment_link = client.payment_link.create({
        "amount": int(amount_inr * 100),  # Razorpay expects paise, not rupees
        "currency": "INR",
        "description": f"Secure your order {order['order_id']} -- pay now for faster delivery",
        "reference_id": order["order_id"],
        "notes": {"risk_tier": "medium", "source": "cod_risk_scorer"},
    })
    return {
        "status": "success",
        "mode": "live",
        "short_url": payment_link["short_url"],
        "amount": amount_inr,
        "razorpay_id": payment_link["id"],
    }


def execute_action(order: dict) -> dict:
    """
    Main entry point: scores the order, then executes (or simulates)
    the appropriate action. Returns a dict describing what happened.
    Never raises for an action failure -- always returns a result dict,
    even in the failure path, so a batch run never crashes mid-way.
    """
    result = sl.score_order(order)
    tier = result["tier"]

    outcome = {
        "order_id": order["order_id"],
        "risk_score": result["risk_score"],
        "tier": tier,
        "action_taken": None,
        "action_status": None,
        "details": None,
    }

    if tier == "low":
        outcome["action_taken"] = "none"
        outcome["action_status"] = "auto_approved"

    elif tier == "medium":
        try:
            if RAZORPAY_MODE == "live":
                link_result = _create_payment_link_live(order, order["order_value"])
            else:
                link_result = _create_payment_link_dry_run(order, order["order_value"])
            outcome["action_taken"] = "payment_link_nudge"
            outcome["action_status"] = "success"
            outcome["details"] = link_result
        except Exception as e:
            # Graceful failure: don't crash, don't silently drop it --
            # escalate to human review as a safe fallback instead.
            outcome["action_taken"] = "payment_link_nudge"
            outcome["action_status"] = "failed"
            outcome["details"] = {"error": str(e)}
            _flag_for_human_review(
                order, result,
                reason=f"Automated nudge action failed ({e}); escalated as fallback."
            )

    elif tier == "high":
        _flag_for_human_review(order, result, reason="High risk score, requires human review.")
        outcome["action_taken"] = "flagged_for_human_review"
        outcome["action_status"] = "success"

    _log_action(outcome)
    return outcome


if __name__ == "__main__":
    print(f"Decision engine running in {RAZORPAY_MODE.upper()} mode.\n")

    # Three test orders, deliberately chosen to hit all three tiers
    test_orders = [
        {  # should score LOW -- established customer, safe category
            "order_id": "DEMO_LOW_001", "customer_id": "CUST_LOYAL_001",
            "delivery_pincode": "560001", "order_value": 400.0,
            "discount_pct": 5.0, "item_category": "Grocery",
            "order_hour": 11, "distance_km": 3.0,
        },
        {  # should score MEDIUM/HIGH -- new customer, risky category, late night, high discount
            "order_id": "DEMO_RISKY_001", "customer_id": "CUST_BRAND_NEW_XYZ",
            "delivery_pincode": "560099", "order_value": 1500.0,
            "discount_pct": 60.0, "item_category": "Fashion",
            "order_hour": 1, "distance_km": 30.0,
        },
        {  # deliberately malformed-ish to test graceful failure path
           # (still valid dict, but simulates a failure by forcing live
           # mode without keys set -- only meaningful if RAZORPAY_MODE=live)
            "order_id": "DEMO_FAILURE_TEST", "customer_id": "CUST_TEST_FAIL",
            "delivery_pincode": "560050", "order_value": 900.0,
            "discount_pct": 45.0, "item_category": "Beauty",
            "order_hour": 23, "distance_km": 12.0,
        },
    ]

    for order in test_orders:
        outcome = execute_action(order)
        print(json.dumps(outcome, indent=2, default=str))
        print("-" * 60)

    print(f"\nAction log written to: {ACTION_LOG_PATH}")
    print(f"Human review queue written to: {REVIEW_QUEUE_PATH}")