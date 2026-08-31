"""
batch_runner.py
------------------
Runs a batch of orders through scoring_logic + decision_engine together,
and prints/saves a funnel summary. This is your best demo-video asset --
"here's 200 orders going through the whole system at once."

WHERE THIS FILE LIVES:
    razorpay/src/batch_runner.py

HOW TO RUN IT:
    cd razorpay
    python src/batch_runner.py --n 200

INPUT:  data/val.csv by default ("demo" mode samples the same file with
        a fresh random draw each run, for variety across demo takes).
        data/test.csv is DELIBERATELY NOT AN OPTION in this script --
        see the guard below. The held-out test set was evaluated exactly
        once in final_test_evaluation.py and must never be touched again,
        for any reason, including demos. If you need a demo dataset that
        isn't even validation data, generate a genuinely fresh synthetic
        batch with generate_data.py using a different seed/output path,
        rather than reaching for test.csv.
OUTPUT: reports/batch_run_summary.json
        Appends to reports/action_log.jsonl and reports/human_review_queue.jsonl
        (the same files decision_engine.py already writes to)
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from collections import Counter

import decision_engine as de

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_ORDER_FIELDS = [
    "order_id", "customer_id", "delivery_pincode", "order_value",
    "discount_pct", "item_category", "order_hour", "distance_km",
]

# Structural guard: "test" is not a valid choice for --source at all,
# enforced by argparse below, not just a warning a script could ignore.
ALLOWED_SOURCES = ["train", "val", "demo"]


def run_batch(n: int, source: str = "val"):
    if source not in ALLOWED_SOURCES:
        raise ValueError(
            f"'{source}' is not an allowed demo source. This script will "
            f"never load test.csv -- the held-out test set is locked after "
            f"its one-time evaluation in final_test_evaluation.py. "
            f"Allowed sources: {ALLOWED_SOURCES}"
        )

    # "demo" reuses val.csv but with an unfixed random draw each run, so
    # repeated demo/video takes show varied orders. "val" (explicit) uses
    # a fixed seed for reproducible output when you need consistency.
    actual_file = "val" if source == "demo" else source
    df = pd.read_csv(DATA_DIR / f"{actual_file}.csv", dtype={"customer_id": str, "delivery_pincode": str})

    random_state = None if source == "demo" else 7
    sample = df.sample(n=min(n, len(df)), random_state=random_state)

    tier_counts = Counter()
    action_counts = Counter()
    status_counts = Counter()
    results = []

    for _, row in sample.iterrows():
        order = {field: row[field] for field in RAW_ORDER_FIELDS}
        outcome = de.execute_action(order)
        tier_counts[outcome["tier"]] += 1
        action_counts[outcome["action_taken"]] += 1
        status_counts[outcome["action_status"]] += 1
        results.append(outcome)

    summary = {
        "batch_size": len(sample),
        "source_file": f"{actual_file}.csv ({'demo sample, unfixed seed' if source == 'demo' else 'fixed seed'})",
        "tier_breakdown": dict(tier_counts),
        "action_breakdown": dict(action_counts),
        "status_breakdown": dict(status_counts),
    }

    print("NOTE: data/test.csv is never used by this script. The held-out "
          "test set was evaluated once in final_test_evaluation.py and "
          "stays locked.\n")
    print("\n=========== BATCH RUN SUMMARY ===========")
    print(f"Processed {summary['batch_size']} orders from {summary['source_file']}\n")
    print("Tier breakdown:")
    for tier, count in summary["tier_breakdown"].items():
        pct = count / summary["batch_size"] * 100
        print(f"  {tier:<8} {count:>4}  ({pct:.1f}%)")
    print("\nAction breakdown:")
    for action, count in summary["action_breakdown"].items():
        print(f"  {action:<25} {count:>4}")
    print("\nStatus breakdown:")
    for status, count in summary["status_breakdown"].items():
        print(f"  {status:<20} {count:>4}")
    print("==========================================")

    with open(REPORTS_DIR / "batch_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {REPORTS_DIR / 'batch_run_summary.json'}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of orders to process")
    parser.add_argument("--source", type=str, default="val",
                         choices=ALLOWED_SOURCES,
                         help="Which data to sample demo orders from. "
                              "'test' is intentionally not a valid choice -- "
                              "the held-out test set is locked.")
    args = parser.parse_args()
    run_batch(args.n, args.source)