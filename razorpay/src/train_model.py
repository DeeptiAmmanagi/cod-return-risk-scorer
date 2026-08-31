"""
train_model.py
----------------
Step 3 of the pipeline: time-based train/val/test split, then a baseline
Logistic Regression model.

WHY TIME-BASED, NOT RANDOM, SPLIT:
  A random shuffle split can put an order from March in training and an
  order from January (that happened AFTER it in a customer's real history
  in some derived sense) in a set treated as "the future" -- for time-
  ordered business data, this understates real-world difficulty. Splitting
  by time (earliest 70% = train, next 15% = val, last 15% = test) means
  you're always predicting forward in time, exactly like a real deployed
  system would.

WHY WE DO NOT TOUCH THE TEST SET HERE:
  The test set is saved now and left alone. Only the validation set is
  used for the quick sanity-check metrics in this script. The test set
  is reserved exclusively for the final, one-time "evaluation metrics"
  step later in the plan -- looking at it now, even by accident, would
  let it quietly influence your choices and undermine the "held-out"
  claim you need to be able to make honestly.

WHY NO class_weight='balanced' YET:
  Rebalancing changes what the model's predicted probabilities mean,
  which would complicate the calibration check and cost-based threshold
  selection planned for a later step. The baseline here is trained
  "as-is" so its raw probability outputs are honest and interpretable;
  imbalance is handled deliberately later, via the cost matrix, not
  smuggled in early through a training-time trick.

WHERE THIS FILE LIVES:
    razorpay/src/train_model.py

HOW TO RUN IT:
    cd razorpay
    python src/train_model.py

INPUT:   data/cod_orders_features.csv
OUTPUTS: data/train.csv, data/val.csv, data/test.csv   (the three splits)
         models/baseline_logreg.joblib                  (the fitted pipeline)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = DATA_DIR / "cod_orders_features.csv"

# Feature columns the model is actually allowed to see.
# Deliberately explicit (not "everything except the label") so it's obvious
# at a glance that identifiers (order_id, customer_id, delivery_pincode,
# order_timestamp) are excluded -- they'd either leak or just be noise.
NUMERIC_FEATURES = [
    "order_value", "discount_pct", "order_hour", "is_late_night",
    "distance_km", "customer_orders_before", "customer_return_rate_smoothed",
    "is_first_order_for_customer", "pincode_orders_before",
    "pincode_return_rate_smoothed",
]
CATEGORICAL_FEATURES = ["item_category"]
LABEL_COLUMN = "is_returned"


def time_based_split(df, train_frac=0.70, val_frac=0.15):
    """
    df must already be sorted by order_timestamp ascending.
    Returns (train_df, val_df, test_df) as a strict chronological split.
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def build_pipeline():
    """
    A single sklearn Pipeline bundling preprocessing + model together.
    This matters: it guarantees the EXACT same scaling/encoding logic
    is applied at training time and at prediction time later (in
    evaluate_model.py, and eventually the FastAPI scoring endpoint) --
    there's no way to accidentally apply different preprocessing in
    production than what the model was trained on.
    """
    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    model = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def quick_report(y_true, y_pred_proba, threshold=0.5, label="validation"):
    y_pred = (y_pred_proba >= threshold).astype(int)
    print(f"\n--- Quick sanity metrics on {label} set (threshold={threshold}) ---")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
    print(f"F1:        {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_pred_proba):.3f}")
    print(f"PR-AUC:    {average_precision_score(y_true, y_pred_proba):.3f}")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"Confusion matrix -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print("(This is a SANITY CHECK only, at a default 0.5 threshold, on the "
          "VALIDATION set. Final honest numbers with a cost-justified "
          "threshold on the untouched TEST set come in a later step.)")


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["order_timestamp"])
    df = df.sort_values("order_timestamp").reset_index(drop=True)

    train_df, val_df, test_df = time_based_split(df)

    print(f"Total orders: {len(df)}")
    print(f"  Train: {len(train_df)}  ({train_df['order_timestamp'].min()} to {train_df['order_timestamp'].max()})")
    print(f"  Val:   {len(val_df)}  ({val_df['order_timestamp'].min()} to {val_df['order_timestamp'].max()})")
    print(f"  Test:  {len(test_df)}  ({test_df['order_timestamp'].min()} to {test_df['order_timestamp'].max()})")

    # Save the splits so evaluate_model.py can load test.csv directly,
    # without needing to redo (and risk subtly redefining) the split logic.
    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"\nSaved train.csv, val.csv, test.csv to {DATA_DIR}")
    print("Reminder: do NOT open/inspect test.csv results until the final "
          "evaluation step. It exists now purely so the split is fixed and reproducible.")

    # Train the baseline on TRAIN only
    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[LABEL_COLUMN]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Quick sanity check on VALIDATION only -- test set stays untouched
    X_val = val_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_val = val_df[LABEL_COLUMN]
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    quick_report(y_val, val_proba, threshold=0.5, label="validation")

    # Save the fitted pipeline for reuse in evaluate_model.py and later the API
    model_path = MODELS_DIR / "baseline_logreg.joblib"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved fitted baseline pipeline to {model_path}")


if __name__ == "__main__":
    main()