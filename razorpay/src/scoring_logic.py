"""
scoring_logic.py
------------------
The actual scoring logic, kept as plain Python with NO FastAPI dependency.
api.py is a thin wrapper around score_order() below. This split means:
  - This file can be unit-tested without running a web server.
  - decision_engine.py can call score_order() directly, without needing
    the API to be running.

THE DESIGN DECISION THIS FILE RESOLVES:
  feature_engineering.py computes customer_return_rate_smoothed and
  pincode_return_rate_smoothed from the FULL historical dataset. A live
  API scoring a brand-new incoming order can't recompute that the same
  way -- there's no "future" data to look back from mid-request.

  SOLUTION: at startup, build an in-memory "current profile" lookup for
  every customer_id and delivery_pincode seen in the historical data,
  using EXACTLY their last known (most time-recent) smoothed rate and
  order count. This simulates a real production "customer risk profile"
  table that gets updated over time. For a customer/pincode NEVER seen
  before, we fall back to the global baseline rate and orders_before=0
  -- the same safe, documented behavior as a genuinely new customer in
  training. No guessing, no silent assumptions.

TWO THRESHOLDS, TWO DIFFERENT COSTS:
  - LOW_RISK_THRESHOLD (from chosen_threshold.json): the cost-optimal
    point for taking ANY automated action at all (a cheap, soft nudge).
  - HIGH_RISK_THRESHOLD (set separately, below): a deliberately more
    conservative bar for escalating to a HUMAN reviewer, because human
    review has real capacity limits and a much higher per-case cost than
    an automated nudge. These being different numbers is intentional,
    not an oversight.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_PATH = REPORTS_DIR / "audit_log.jsonl"

NUMERIC_FEATURES = [
    "order_value", "discount_pct", "order_hour", "is_late_night",
    "distance_km", "customer_orders_before", "customer_return_rate_smoothed",
    "is_first_order_for_customer", "pincode_orders_before",
    "pincode_return_rate_smoothed",
]
CATEGORICAL_FEATURES = ["item_category"]

# Escalation to human review is deliberately more conservative than the
# cost-optimal automated-action threshold. Tune this based on how much
# human review capacity actually exists -- it's a capacity/cost decision,
# not a statistical one, so it's a separate constant on purpose.
HIGH_RISK_THRESHOLD = 0.55


# ----------------------------------------------------------------------
# Load everything ONCE at import time (not per-request)
# ----------------------------------------------------------------------

_pipeline = joblib.load(MODELS_DIR / "baseline_logreg.joblib")

with open(MODELS_DIR / "chosen_threshold.json", encoding="utf-8") as f:
    _threshold_config = json.load(f)
LOW_RISK_THRESHOLD = _threshold_config["chosen_threshold"]

_history_df = pd.read_csv(
    DATA_DIR / "cod_orders_features.csv",
    parse_dates=["order_timestamp"],
    dtype={"customer_id": str, "delivery_pincode": str},  # prevent pandas from
    # silently inferring these ID columns as int64 just because pincodes look numeric
)
_history_df = _history_df.sort_values("order_timestamp")

GLOBAL_BASELINE_RATE = float(_history_df["is_returned"].mean())

# ----------------------------------------------------------------------
# Out-of-distribution guardrail
# ----------------------------------------------------------------------
# Found via deliberate stress testing: an order_value far outside the
# training range (e.g. Rs. 2,50,000 vs a training max of ~Rs. 15,000)
# causes StandardScaler to extrapolate massively, producing a raw
# probability like 5.8e-27 -- the model isn't "confident it's safe", it's
# extrapolating into territory it has never seen and cannot honestly
# reason about. Silently trusting that score would mean auto-approving
# a wildly anomalous order with false confidence.
#
# Fix: any order outside a documented, data-derived bound on order_value
# or distance_km is flagged as out-of-distribution and forced to HIGH
# risk / human review, regardless of what the raw model score says. The
# raw score is still reported for transparency, but the tier decision
# does not trust it.
ORDER_VALUE_BOUNDS = (
    float(_history_df["order_value"].min()),
    float(_history_df["order_value"].max()) * 1.5,  # some margin above the max seen
)
DISTANCE_KM_BOUNDS = (
    0.0,
    float(_history_df["distance_km"].max()) * 1.5,
)


def is_out_of_distribution(order: dict) -> str | None:
    """Returns a human-readable reason string if the order is out of the
    model's trained range, else None."""
    ov = float(order["order_value"])
    if ov < ORDER_VALUE_BOUNDS[0] or ov > ORDER_VALUE_BOUNDS[1]:
        return (f"order_value {ov} is outside the training range "
                f"{ORDER_VALUE_BOUNDS} -- model score is not trustworthy here.")
    dist = float(order["distance_km"])
    if dist < DISTANCE_KM_BOUNDS[0] or dist > DISTANCE_KM_BOUNDS[1]:
        return (f"distance_km {dist} is outside the training range "
                f"{DISTANCE_KM_BOUNDS} -- model score is not trustworthy here.")
    return None

# Last known state per customer / pincode, as of the end of the historical data.
_customer_profiles = (
    _history_df.groupby("customer_id")
    .last()[["customer_orders_before", "customer_return_rate_smoothed"]]
    .to_dict(orient="index")
)
_pincode_profiles = (
    _history_df.groupby("delivery_pincode")
    .last()[["pincode_orders_before", "pincode_return_rate_smoothed"]]
    .to_dict(orient="index")
)

print(f"[scoring_logic] Loaded model, threshold={LOW_RISK_THRESHOLD}, "
      f"{len(_customer_profiles)} known customers, "
      f"{len(_pincode_profiles)} known pincodes, "
      f"global baseline rate={GLOBAL_BASELINE_RATE:.3f}")


# ----------------------------------------------------------------------
# Profile lookups with safe, documented defaults for unseen entities
# ----------------------------------------------------------------------

def get_customer_profile(customer_id: str) -> dict:
    customer_id = str(customer_id)   # normalize in case caller sends a non-string
    if customer_id in _customer_profiles:
        prev = _customer_profiles[customer_id]
        return {
            "customer_orders_before": int(prev["customer_orders_before"]) + 1,
            "customer_return_rate_smoothed": float(prev["customer_return_rate_smoothed"]),
            "is_first_order_for_customer": 0,
        }
    # Genuinely new customer -- same safe default as a first-ever order in training
    return {
        "customer_orders_before": 0,
        "customer_return_rate_smoothed": GLOBAL_BASELINE_RATE,
        "is_first_order_for_customer": 1,
    }


def get_pincode_profile(pincode: str) -> dict:
    pincode = str(pincode)   # normalize in case caller sends a non-string
    if pincode in _pincode_profiles:
        prev = _pincode_profiles[pincode]
        return {
            "pincode_orders_before": int(prev["pincode_orders_before"]) + 1,
            "pincode_return_rate_smoothed": float(prev["pincode_return_rate_smoothed"]),
        }
    return {
        "pincode_orders_before": 0,
        "pincode_return_rate_smoothed": GLOBAL_BASELINE_RATE,
    }


# ----------------------------------------------------------------------
# Build the exact feature row the model expects
# ----------------------------------------------------------------------

def build_feature_row(order: dict) -> pd.DataFrame:
    """
    `order` must contain: customer_id, delivery_pincode, order_value,
    discount_pct, item_category, order_hour, distance_km
    """
    cust = get_customer_profile(order["customer_id"])
    pin = get_pincode_profile(order["delivery_pincode"])
    order_hour = int(order["order_hour"])

    row = {
        "order_value": float(order["order_value"]),
        "discount_pct": float(order["discount_pct"]),
        "order_hour": order_hour,
        "is_late_night": 1 if (order_hour >= 22 or order_hour <= 4) else 0,
        "distance_km": float(order["distance_km"]),
        "item_category": order["item_category"],
        **cust,
        **pin,
    }
    return pd.DataFrame([row])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


# ----------------------------------------------------------------------
# Interpretability: honest, direct-from-coefficients top factors
# (Logistic Regression is linear, so this is exact, not approximated --
# no SHAP dependency needed for this model.)
# ----------------------------------------------------------------------

def get_top_factors(X_row: pd.DataFrame, n=3) -> list:
    preprocessor = _pipeline.named_steps["preprocess"]
    model = _pipeline.named_steps["model"]

    transformed = preprocessor.transform(X_row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = transformed.flatten()

    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_[0]
    contributions = transformed * coefs

    top_idx = np.argsort(-np.abs(contributions))[:n]
    return [
        {
            "feature": str(feature_names[i]),
            "contribution": round(float(contributions[i]), 4),
            "direction": "increases_risk" if contributions[i] > 0 else "decreases_risk",
        }
        for i in top_idx
    ]


# ----------------------------------------------------------------------
# Audit logging -- every scored order, every decision, timestamped
# ----------------------------------------------------------------------

def _json_safe(obj):
    """Fallback for json.dumps: convert numpy scalar types to native Python.
    A logging bug should never be able to take down the actual scoring
    response -- this is a defensive backstop on top of the real fix
    (explicit dtype=str on ID columns above)."""
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def log_audit(order: dict, result: dict):
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "order": order,
        "result": result,
    }
    with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=_json_safe) + "\n")


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------

def score_order(order: dict) -> dict:
    """
    order: dict with keys order_id, customer_id, delivery_pincode,
           order_value, discount_pct, item_category, order_hour, distance_km
    Returns a dict with risk_score, tier, action, top_factors, threshold_used.
    Never raises on missing history -- only on genuinely malformed input
    (missing required raw fields), which the API layer should reject
    with a 422 before it ever reaches here.
    """
    X_row = build_feature_row(order)
    risk_score = float(_pipeline.predict_proba(X_row)[:, 1][0])

    ood_reason = is_out_of_distribution(order)
    if ood_reason is not None:
        tier, action = "high", "flag_for_human_review"
    elif risk_score < LOW_RISK_THRESHOLD:
        tier, action = "low", "auto_approve"
    elif risk_score < HIGH_RISK_THRESHOLD:
        tier, action = "medium", "send_prepayment_nudge"
    else:
        tier, action = "high", "flag_for_human_review"

    result = {
        "order_id": order["order_id"],
        "risk_score": round(risk_score, 4),
        "tier": tier,
        "action": action,
        "top_factors": get_top_factors(X_row),
        "low_risk_threshold": LOW_RISK_THRESHOLD,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "out_of_distribution": ood_reason,
    }

    log_audit(order, result)
    return result


if __name__ == "__main__":
    # Quick manual smoke test -- run this file directly to sanity check
    # scoring logic without needing FastAPI/uvicorn at all.
    test_order = {
        "order_id": "TEST0001",
        "customer_id": "CUST_NEVER_SEEN_BEFORE",
        "delivery_pincode": "560099",
        "order_value": 1200.0,
        "discount_pct": 35.0,
        "item_category": "Fashion",
        "order_hour": 23,
        "distance_km": 18.0,
    }
    print(json.dumps(score_order(test_order), indent=2))