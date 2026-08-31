"""
final_test_evaluation.py
--------------------------
THE ONE-TIME, LOCKED, HELD-OUT TEST SET EVALUATION.

RULES THIS SCRIPT FOLLOWS (do not violate these, even to "just check"):
  - Loads the ALREADY-FITTED baseline Logistic Regression pipeline.
    Does NOT retrain, does NOT refit, does NOT touch train.csv or val.csv.
  - Loads the ALREADY-CHOSEN threshold + cost matrix from
    models/chosen_threshold.json (produced by evaluate_model.py on the
    VALIDATION set). Does NOT re-sweep or re-tune the threshold here.
  - Evaluates ONLY on data/test.csv -- the one dataset that has not
    influenced a single decision until this exact moment.
  - Does NOT evaluate or mention the XGBoost/stronger model. That
    comparison already happened, on validation, and the baseline won.
    This script exists to report ONE model's ONE honest final number.

If you find yourself wanting to run this script again with a different
threshold or model "to see if it's better" -- don't. That defeats the
entire purpose of a held-out test set. If you genuinely need to change
the model or threshold, that decision must be made by going back to
evaluate_model.py / train_stronger_model.py using validation data only,
and then this script gets ONE new final run, documented as such.

WHERE THIS FILE LIVES:
    razorpay/src/final_test_evaluation.py

HOW TO RUN IT (once):
    cd razorpay
    python src/final_test_evaluation.py

INPUT:   models/baseline_logreg.joblib, models/chosen_threshold.json, data/test.csv
OUTPUT:  reports/final_test_evaluation.json
         reports/final_test_evaluation.md   (ready to paste into your README)
"""

import json
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix,
)

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


def main():
    print("=" * 70)
    print("FINAL LOCKED TEST-SET EVALUATION -- this is the one-time,")
    print("held-out result. No retraining, no threshold tuning happens")
    print("in this script.")
    print("=" * 70)

    # Load the fixed model and fixed threshold -- single source of truth,
    # both produced entirely from train/validation data, never test data.
    pipeline = joblib.load(MODELS_DIR / "baseline_logreg.joblib")
    with open(MODELS_DIR / "chosen_threshold.json") as f:
        threshold_config = json.load(f)

    threshold = threshold_config["chosen_threshold"]
    cost_fn = threshold_config["cost_false_negative"]
    cost_fp = threshold_config["cost_false_positive"]

    print(f"\nModel: baseline_logreg.joblib (Logistic Regression)")
    print(f"Threshold: {threshold} (chosen on validation set, see chosen_threshold.json)")
    print(f"Cost matrix: FN=₹{cost_fn}, FP=₹{cost_fp}")

    test_df = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["order_timestamp"])
    print(f"\nTest set: {len(test_df)} orders, "
          f"{test_df['order_timestamp'].min()} to {test_df['order_timestamp'].max()}")
    print(f"Test set positive rate: {test_df[LABEL_COLUMN].mean():.2%}")

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[LABEL_COLUMN]

    # Predict -- this is the only model interaction this script performs
    test_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (test_proba >= threshold).astype(int)

    # Threshold-independent metrics (computed straight from probabilities)
    roc_auc = roc_auc_score(y_test, test_proba)
    pr_auc = average_precision_score(y_test, test_proba)

    # Threshold-dependent metrics (at the fixed, pre-chosen threshold)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    total_cost = fp * cost_fp + fn * cost_fn

    # What the cost would have been doing nothing at all (never flagging
    # anything) -- a useful "worst case / no-model" baseline comparison.
    no_model_cost = test_df[LABEL_COLUMN].sum() * cost_fn

    print("\n--- FINAL RESULTS (test set, threshold-independent) ---")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC:  {pr_auc:.3f}")

    print(f"\n--- FINAL RESULTS (test set, at threshold={threshold}) ---")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Confusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\nTotal cost with model: ₹{total_cost:,.0f}")
    print(f"Total cost with NO model (never flag anything): ₹{no_model_cost:,.0f}")
    print(f"Cost reduction vs. doing nothing: ₹{no_model_cost - total_cost:,.0f} "
          f"({(no_model_cost - total_cost) / no_model_cost:.1%})")

    # ------------------------------------------------------------------
    # Save results -- both machine-readable and README-ready
    # ------------------------------------------------------------------
    results = {
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model": "baseline_logreg.joblib",
        "threshold": threshold,
        "cost_false_negative": cost_fn,
        "cost_false_positive": cost_fp,
        "test_set_size": len(test_df),
        "test_positive_rate": float(test_df[LABEL_COLUMN].mean()),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "total_cost_with_model_inr": round(total_cost, 2),
        "total_cost_no_model_inr": round(no_model_cost, 2),
        "cost_reduction_pct": round((no_model_cost - total_cost) / no_model_cost, 4),
    }
    with open(REPORTS_DIR / "final_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    markdown = f"""## Final Held-Out Test Set Evaluation

**Model:** Logistic Regression baseline (selected over XGBoost, which scored lower on validation PR-AUC and cost)
**Threshold:** {threshold} (chosen via cost-minimization on the validation set, never on test)
**Test set:** {len(test_df)} orders, entirely unseen by the model, the threshold, and the model-selection decision

| Metric | Value |
|---|---|
| ROC-AUC | {roc_auc:.3f} |
| PR-AUC | {pr_auc:.3f} |
| Precision | {precision:.3f} |
| Recall | {recall:.3f} |
| F1 | {f1:.3f} |
| False Positives | {fp} |
| False Negatives | {fn} |
| Total cost (with model) | ₹{total_cost:,.0f} |
| Total cost (no model / never flag) | ₹{no_model_cost:,.0f} |
| Cost reduction vs. no model | {(no_model_cost - total_cost) / no_model_cost:.1%} |

Cost matrix: ₹{cost_fn} per missed risky order (false negative), ₹{cost_fp} per unnecessary soft nudge on a good order (false positive).
"""
    with open(REPORTS_DIR / "final_test_evaluation.md", "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"\nSaved machine-readable results to {REPORTS_DIR / 'final_test_evaluation.json'}")
    print(f"Saved README-ready summary to {REPORTS_DIR / 'final_test_evaluation.md'}")
    print("\nThis is your official number. Do not re-run this script with a "
          "different threshold or model in an attempt to improve it.")


if __name__ == "__main__":
    main()