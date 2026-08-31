"""
evaluate_model.py
-------------------
Step 4 of the pipeline: proper evaluation of the baseline model on the
VALIDATION set (test.csv remains untouched -- see train_model.py notes).

WHAT THIS SCRIPT DOES:
  1. Loads the fitted pipeline + val.csv.
  2. Plots a precision-recall curve across all thresholds.
  3. Plots a calibration curve (are predicted probabilities trustworthy,
     or just a good ranking?).
  4. Defines an explicit cost matrix (cost of a false negative vs. a
     false positive) and sweeps thresholds to find the one that
     MINIMIZES TOTAL EXPECTED COST -- not the one that maximizes
     accuracy or F1. This is what "honest metrics including false-
     positive cost" in the brief is actually asking for.
  5. Saves the chosen threshold to disk so later steps (decision engine,
     API) use the same, deliberately-chosen number.

WHY THE COST MATRIX NUMBERS BELOW ARE WHAT THEY ARE (edit if you disagree,
but do so on purpose, and say why in your README):
  - False Negative (a risky order slips through, gets returned/refused):
    cost = shipping cost there-and-back + reverse logistics handling.
    ~₹150 is a reasonable assumed average for a mid-value COD parcel.
  - False Positive (a genuinely fine order gets flagged/friction-ed):
    Because our decision engine's medium-risk action is a SOFT nudge
    (an optional "pay now for faster delivery" link, not a rejection or
    a hard block), most flagged good customers simply ignore it and
    continue as normal. Modeled as: (small probability of genuine
    annoyance/drop-off from the nudge, ~5%) x (average order value ~₹800)
    ≈ ₹40. If the action were a hard block instead of a soft nudge, this
    number should be much higher -- the cost model must match the actual
    action taken, which is why this constant lives next to the decision
    engine's threshold tiers, not in isolation.
  These are deliberately simple, average, order-value-independent numbers
  for this stage -- a documented simplification, not a hidden shortcut.
  A stretch-goal improvement is making FP cost scale with each order's
  actual order_value instead of using one flat number.

WHERE THIS FILE LIVES:
    razorpay/src/evaluate_model.py

HOW TO RUN IT:
    cd razorpay
    python src/evaluate_model.py

INPUT:   models/baseline_logreg.joblib, data/val.csv
OUTPUTS: reports/pr_curve_baseline.png
         reports/calibration_baseline.png
         reports/cost_vs_threshold_baseline.png
         models/chosen_threshold.json
"""

import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # save plots to file, no display window needed
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    precision_recall_curve, precision_score, recall_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "order_value", "discount_pct", "order_hour", "is_late_night",
    "distance_km", "customer_orders_before", "customer_return_rate_smoothed",
    "is_first_order_for_customer", "pincode_orders_before",
    "pincode_return_rate_smoothed",
]
CATEGORICAL_FEATURES = ["item_category"]
LABEL_COLUMN = "is_returned"

# ---- Cost matrix: see the module docstring for justification ----
COST_FALSE_NEGATIVE = 150.0   # ₹ cost of a missed risky order
COST_FALSE_POSITIVE = 40.0    # ₹ cost of a soft nudge on a good order (see docstring)


def plot_pr_curve(y_true, y_proba, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (validation set)")
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved PR curve to {out_path}")


def plot_calibration(y_true, y_proba, out_path):
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Actual fraction returned")
    plt.title("Calibration Curve (validation set)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved calibration curve to {out_path}")
    print("(Points close to the dashed diagonal = probabilities are "
          "trustworthy, not just a good ranking. Points far from it "
          "mean 'risk score of 0.7' doesn't really mean '70% chance'.)")


def find_cost_minimizing_threshold(y_true, y_proba, cost_fn, cost_fp, out_path):
    thresholds = np.arange(0.01, 1.00, 0.01)
    total_costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_cost = fp * cost_fp + fn * cost_fn
        total_costs.append(total_cost)
    total_costs = np.array(total_costs)

    best_idx = np.argmin(total_costs)
    best_threshold = thresholds[best_idx]
    best_cost = total_costs[best_idx]

    # cost at the naive default threshold, for a direct before/after comparison
    default_idx = np.argmin(np.abs(thresholds - 0.50))
    default_cost = total_costs[default_idx]

    plt.figure(figsize=(6, 5))
    plt.plot(thresholds, total_costs)
    plt.axvline(best_threshold, color="green", linestyle="--", label=f"Chosen threshold = {best_threshold:.2f}")
    plt.axvline(0.50, color="red", linestyle=":", label="Default threshold = 0.50")
    plt.xlabel("Decision threshold")
    plt.ylabel("Total expected cost on validation set (₹)")
    plt.title("Cost vs. Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cost-vs-threshold curve to {out_path}")

    print(f"\nDefault threshold (0.50): total cost = ₹{default_cost:,.0f}")
    print(f"Chosen threshold ({best_threshold:.2f}): total cost = ₹{best_cost:,.0f}")
    print(f"Savings from cost-aware thresholding: ₹{default_cost - best_cost:,.0f} "
          f"({(default_cost - best_cost) / default_cost:.1%} reduction) "
          f"on this validation batch of {len(y_true)} orders")

    return best_threshold


def main():
    pipeline = joblib.load(MODELS_DIR / "baseline_logreg.joblib")
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["order_timestamp"])

    X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[LABEL_COLUMN]
    val_proba = pipeline.predict_proba(X_val)[:, 1]

    plot_pr_curve(y_val, val_proba, REPORTS_DIR / "pr_curve_baseline.png")
    plot_calibration(y_val, val_proba, REPORTS_DIR / "calibration_baseline.png")

    best_threshold = find_cost_minimizing_threshold(
        y_val, val_proba, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE,
        REPORTS_DIR / "cost_vs_threshold_baseline.png",
    )

    # Report precision/recall AT the chosen threshold, still on validation,
    # so you can see the actual behavior change vs. the 0.5 default.
    y_pred_chosen = (val_proba >= best_threshold).astype(int)
    print(f"\nAt chosen threshold {best_threshold:.2f} on validation set:")
    print(f"  Precision: {precision_score(y_val, y_pred_chosen):.3f}")
    print(f"  Recall:    {recall_score(y_val, y_pred_chosen):.3f}")
    print("  (Compare this recall to the 0.181 we got at threshold=0.50 "
          "in train_model.py -- this is the concrete improvement from "
          "cost-aware thresholding.)")

    # Save the chosen threshold + cost assumptions for later steps to reuse
    threshold_config = {
        "chosen_threshold": float(best_threshold),
        "cost_false_negative": COST_FALSE_NEGATIVE,
        "cost_false_positive": COST_FALSE_POSITIVE,
        "selected_on": "validation set",
        "note": "Threshold chosen to minimize total expected cost, not accuracy/F1.",
    }
    with open(MODELS_DIR / "chosen_threshold.json", "w", encoding="utf-8") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"\nSaved chosen threshold + cost assumptions to {MODELS_DIR / 'chosen_threshold.json'}")


if __name__ == "__main__":
    main()