"""
train_stronger_model.py
-------------------------
Step 5: train a gradient-boosted model (XGBoost, or scikit-learn's
HistGradientBoostingClassifier as an automatic fallback if xgboost
isn't installed) and compare it FAIRLY against the baseline -- same
train/val split, same cost matrix, same "find its own best threshold"
logic, not just accuracy at a fixed threshold.

WHY "FAIRLY": each model gets ITS OWN cost-minimizing threshold from
the same sweep logic used in evaluate_model.py. Comparing model A at
threshold 0.5 against model B at its optimal threshold would be biased
towards B for no good reason -- we give both models the same chance.

STILL NOT TOUCHING test.csv. This script only uses train.csv and val.csv.

WHERE THIS FILE LIVES:
    razorpay/src/train_stronger_model.py

HOW TO RUN IT:
    cd razorpay
    pip install xgboost   (recommended -- falls back automatically if skipped)
    python src/train_stronger_model.py

OUTPUT:
    models/stronger_model.joblib
    Printed side-by-side comparison against the baseline's known numbers.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_score, recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

NUMERIC_FEATURES = [
    "order_value", "discount_pct", "order_hour", "is_late_night",
    "distance_km", "customer_orders_before", "customer_return_rate_smoothed",
    "is_first_order_for_customer", "pincode_orders_before",
    "pincode_return_rate_smoothed",
]
CATEGORICAL_FEATURES = ["item_category"]
LABEL_COLUMN = "is_returned"

# Same cost matrix as evaluate_model.py -- keep these two files in sync
COST_FALSE_NEGATIVE = 150.0
COST_FALSE_POSITIVE = 40.0

# Baseline's known numbers from evaluate_model.py, hardcoded here purely
# for the printed side-by-side comparison at the end of this script.
BASELINE_PR_AUC = 0.451
BASELINE_BEST_COST = 22930.0
BASELINE_BEST_THRESHOLD = 0.23


def get_model():
    """Prefer XGBoost; fall back to sklearn's HistGradientBoostingClassifier
    (which needs no extra install) if xgboost isn't available."""
    try:
        from xgboost import XGBClassifier
        print("Using XGBoost.")
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("xgboost not installed -- falling back to "
              "HistGradientBoostingClassifier (run `pip install xgboost` "
              "and re-run this script to use the real thing).")
        return HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, random_state=42,
        )


def build_pipeline():
    # Tree-based models don't need feature scaling, unlike the logistic
    # regression baseline -- numeric features pass through unchanged.
    preprocessor = ColumnTransformer(transformers=[
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline(steps=[("preprocess", preprocessor), ("model", get_model())])


def find_best_threshold_and_cost(y_true, y_proba, cost_fn, cost_fp):
    thresholds = np.arange(0.01, 1.00, 0.01)
    best_threshold, best_cost = 0.5, float("inf")
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_threshold = cost, t
    return best_threshold, best_cost


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["order_timestamp"])
    val_df = pd.read_csv(DATA_DIR / "val.csv", parse_dates=["order_timestamp"])

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[LABEL_COLUMN]
    X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[LABEL_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]

    pr_auc = average_precision_score(y_val, val_proba)
    roc_auc = roc_auc_score(y_val, val_proba)
    best_threshold, best_cost = find_best_threshold_and_cost(
        y_val, val_proba, COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE
    )
    y_pred_at_best = (val_proba >= best_threshold).astype(int)
    precision = precision_score(y_val, y_pred_at_best)
    recall = recall_score(y_val, y_pred_at_best)

    print("\n=========== STRONGER MODEL vs BASELINE (validation set) ===========")
    print(f"{'Metric':<28}{'Baseline (LogReg)':<22}{'Stronger model':<22}")
    print(f"{'PR-AUC':<28}{BASELINE_PR_AUC:<22.3f}{pr_auc:<22.3f}")
    print(f"{'Best threshold':<28}{BASELINE_BEST_THRESHOLD:<22.2f}{best_threshold:<22.2f}")
    print(f"{'Total cost at best thresh.':<28}₹{BASELINE_BEST_COST:<21,.0f}₹{best_cost:<21,.0f}")
    print(f"{'Precision at best thresh.':<28}{'0.379':<22}{precision:<22.3f}")
    print(f"{'Recall at best thresh.':<28}{'0.634':<22}{recall:<22.3f}")
    print("=====================================================================")

    if best_cost < BASELINE_BEST_COST:
        savings = BASELINE_BEST_COST - best_cost
        print(f"\n-> Stronger model WINS: ₹{savings:,.0f} lower cost on this "
              f"validation batch ({savings / BASELINE_BEST_COST:.1%} improvement "
              f"over the already cost-optimized baseline).")
    else:
        print("\n-> Baseline is still competitive or better. This is a "
              "genuinely fine, reportable outcome -- a simple, fully "
              "interpretable Logistic Regression matching or beating a "
              "gradient-boosted model is worth stating honestly, not "
              "hidden. Keep the baseline as your primary model in that case.")

    model_path = MODELS_DIR / "stronger_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved stronger model pipeline to {model_path}")
    print("\nDecide which model to lock in as FINAL before running the "
          "one-time test-set evaluation. Do not run both models against "
          "test.csv 'just to see' -- pick one, based on this validation "
          "comparison, then evaluate that one choice on test.csv exactly once.")


if __name__ == "__main__":
    main()