"""
stress_test.py
-----------------
Deliberately feeds edge cases through scoring_logic + decision_engine to
surface real failure modes -- this is what generates genuine "what broke
and how I handled it" material, rather than relying only on bugs found
by accident.

WHERE THIS FILE LIVES:
    razorpay/src/stress_test.py

HOW TO RUN IT:
    cd razorpay
    python src/stress_test.py
"""

import json
import scoring_logic as sl
import decision_engine as de

EDGE_CASES = [
    {
        "name": "Brand new customer AND brand new pincode (double cold-start)",
        "order": {
            "order_id": "EDGE_001", "customer_id": "CUST_NEVER_SEEN_XYZ",
            "delivery_pincode": "999999", "order_value": 800.0,
            "discount_pct": 20.0, "item_category": "Fashion",
            "order_hour": 12, "distance_km": 10.0,
        },
    },
    {
        "name": "Unknown item_category (not in training data)",
        "order": {
            "order_id": "EDGE_002", "customer_id": "CUST100872",
            "delivery_pincode": "560093", "order_value": 500.0,
            "discount_pct": 10.0, "item_category": "Pets",  # never seen in training
            "order_hour": 14, "distance_km": 5.0,
        },
    },
    {
        "name": "Extreme order value (unusually high)",
        "order": {
            "order_id": "EDGE_003", "customer_id": "CUST100872",
            "delivery_pincode": "560093", "order_value": 250000.0,
            "discount_pct": 10.0, "item_category": "Electronics",
            "order_hour": 14, "distance_km": 5.0,
        },
    },
    {
        "name": "Zero discount, zero distance (minimum boundary values)",
        "order": {
            "order_id": "EDGE_004", "customer_id": "CUST100872",
            "delivery_pincode": "560093", "order_value": 300.0,
            "discount_pct": 0.0, "item_category": "Grocery",
            "order_hour": 10, "distance_km": 0.0,
        },
    },
    {
        "name": "Maximum discount boundary (100%)",
        "order": {
            "order_id": "EDGE_005", "customer_id": "CUST_NEW_002",
            "delivery_pincode": "560001", "order_value": 999.0,
            "discount_pct": 100.0, "item_category": "Beauty",
            "order_hour": 3, "distance_km": 22.0,
        },
    },
    {
        "name": "Same customer scored twice in a row (does profile update sanely?)",
        "order": {
            "order_id": "EDGE_006", "customer_id": "CUST_REPEAT_TEST",
            "delivery_pincode": "560002", "order_value": 700.0,
            "discount_pct": 15.0, "item_category": "Home",
            "order_hour": 16, "distance_km": 8.0,
        },
    },
    {
        "name": "Missing required field (should fail loudly, not silently)",
        "order": {
            "order_id": "EDGE_007", "customer_id": "CUST_NEW_003",
            "delivery_pincode": "560003",
            # order_value deliberately missing
            "discount_pct": 15.0, "item_category": "Home",
            "order_hour": 16, "distance_km": 8.0,
        },
    },
]


def run_stress_test():
    print("=" * 70)
    print("STRESS TEST -- feeding deliberate edge cases through the system")
    print("=" * 70)

    results_summary = []

    for case in EDGE_CASES:
        print(f"\n--- {case['name']} ---")
        try:
            outcome = de.execute_action(case["order"])
            print(f"  -> SUCCEEDED: score={outcome['risk_score']}, "
                  f"tier={outcome['tier']}, action={outcome['action_taken']}")
            results_summary.append({"case": case["name"], "outcome": "handled", "detail": outcome})
        except Exception as e:
            print(f"  -> RAISED EXCEPTION: {type(e).__name__}: {e}")
            results_summary.append({"case": case["name"], "outcome": "exception",
                                     "detail": f"{type(e).__name__}: {e}"})

    # EDGE_006 run twice to check profile evolves sanely across repeated calls
    print(f"\n--- Same customer scored a SECOND time (checking profile increments) ---")
    second_order = dict(EDGE_CASES[5]["order"])
    second_order["order_id"] = "EDGE_006_SECOND_CALL"
    outcome2 = de.execute_action(second_order)
    print(f"  -> score={outcome2['risk_score']}, tier={outcome2['tier']}")
    print("  (Compare customer_orders_before between the two calls in the "
          "audit log to confirm it incremented, not reset or stayed static.)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results_summary:
        print(f"  [{r['outcome'].upper():>10}] {r['case']}")

    n_exceptions = sum(1 for r in results_summary if r["outcome"] == "exception")
    print(f"\n{n_exceptions} case(s) raised an unhandled exception.")
    if n_exceptions > 0:
        print("These are your real 'what broke' findings -- document the "
              "exact exception, WHY it happened, and what you changed to "
              "fix or guard against it.")
    else:
        print("All edge cases were handled without crashing. Still worth "
              "checking: did any of them produce a NONSENSICAL score/tier "
              "rather than crashing? A graceful-looking failure that gives "
              "a wrong answer silently is arguably worse than a loud crash.")


if __name__ == "__main__":
    run_stress_test()