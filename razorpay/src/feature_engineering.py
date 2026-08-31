"""
feature_engineering.py
------------------------
Turns the raw synthetic COD orders (data/cod_orders.csv) into a modeling-
ready feature table by computing TIME-RESPECTING historical aggregates.

THE CORE IDEA:
  For every order, we compute "based on everything that happened with this
  customer / this pincode STRICTLY BEFORE this order, how risky did they
  look?" Using anything from after this order's timestamp would be leakage
  -- information a real production system could not possibly have had at
  scoring time.

WHY SMOOTHING MATTERS:
  A brand-new customer's very first order has ZERO history. A customer
  with exactly 1 prior order that was returned has a "100% return rate"
  on paper, but that's one data point, not a reliable signal. We use
  Bayesian/Laplace-style smoothing: blend the customer's own history with
  a global baseline rate, weighted by how much personal history exists.
  As history grows, the personal signal dominates; with little/no history,
  it falls back toward the (also time-respecting) global average.

WHERE THIS FILE LIVES:
    razorpay/src/feature_engineering.py

HOW TO RUN IT:
    cd razorpay
    python src/feature_engineering.py

INPUT:  data/cod_orders.csv
OUTPUT: data/cod_orders_features.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_PATH = DATA_DIR / "cod_orders.csv"
OUTPUT_PATH = DATA_DIR / "cod_orders_features.csv"

# Smoothing strength: higher K = more weight given to the global baseline
# before personal history is trusted. K=5 means "it takes roughly 5 prior
# orders before a customer's own history starts to dominate the estimate."
CUSTOMER_SMOOTHING_K = 5
PINCODE_SMOOTHING_K = 20   # pincodes naturally accumulate more orders faster,
                            # so we require more evidence before trusting them


def main():
    df = pd.read_csv(INPUT_PATH, parse_dates=["order_timestamp"])

    # CRITICAL: sort by time first. Every "before this row" calculation
    # below depends on the DataFrame already being in chronological order.
    df = df.sort_values("order_timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 1. Time-respecting GLOBAL baseline (used as the smoothing prior)
    # ------------------------------------------------------------------
    # global_orders_before / global_returns_before for row i = counts from
    # ALL rows before row i in the whole dataset, regardless of customer
    # or pincode. This is itself time-respecting: row 0 has zero prior
    # global history, row 100 has 100 prior orders' worth of history, etc.
    n = len(df)
    df["global_orders_before"] = np.arange(n)
    df["global_returns_before"] = df["is_returned"].cumsum() - df["is_returned"]

    # Avoid divide-by-zero for the very first row(s); fall back to a
    # reasonable assumed baseline (documented, not hidden) until real
    # global history accumulates.
    ASSUMED_BASELINE_RATE = 0.15
    with np.errstate(invalid="ignore", divide="ignore"):
        df["global_return_rate_before"] = np.where(
            df["global_orders_before"] > 0,
            df["global_returns_before"] / df["global_orders_before"],
            ASSUMED_BASELINE_RATE,
        )

    # ------------------------------------------------------------------
    # 2. Per-CUSTOMER time-respecting history
    # ------------------------------------------------------------------
    # groupby().cumcount() gives, for each row, how many EARLIER rows exist
    # in the same group (in the order they appear in the sorted DataFrame)
    # -- exactly "this customer's prior order count."
    grouped_cust = df.groupby("customer_id")["is_returned"]
    df["customer_orders_before"] = grouped_cust.cumcount()
    # cumsum() includes the current row; subtracting is_returned removes it,
    # leaving only strictly-prior returns for this customer.
    df["customer_returns_before"] = grouped_cust.cumsum() - df["is_returned"]

    df["customer_return_rate_smoothed"] = (
        df["customer_returns_before"] + CUSTOMER_SMOOTHING_K * df["global_return_rate_before"]
    ) / (df["customer_orders_before"] + CUSTOMER_SMOOTHING_K)

    df["is_first_order_for_customer"] = (df["customer_orders_before"] == 0).astype(int)

    # ------------------------------------------------------------------
    # 3. Per-PINCODE time-respecting history (same logic, different group)
    # ------------------------------------------------------------------
    grouped_pin = df.groupby("delivery_pincode")["is_returned"]
    df["pincode_orders_before"] = grouped_pin.cumcount()
    df["pincode_returns_before"] = grouped_pin.cumsum() - df["is_returned"]

    df["pincode_return_rate_smoothed"] = (
        df["pincode_returns_before"] + PINCODE_SMOOTHING_K * df["global_return_rate_before"]
    ) / (df["pincode_orders_before"] + PINCODE_SMOOTHING_K)

    # ------------------------------------------------------------------
    # 4. Simple derived features (no history needed, no leakage risk)
    # ------------------------------------------------------------------
    df["is_late_night"] = df["order_hour"].apply(lambda h: 1 if (h >= 22 or h <= 4) else 0)

    # ------------------------------------------------------------------
    # 5. Assemble final feature table
    # ------------------------------------------------------------------
    final_columns = [
        # identifiers / audit trail -- keep these, just don't feed the
        # raw IDs into the model as features
        "order_id", "customer_id", "delivery_pincode", "order_timestamp",
        # raw order-level features
        "order_value", "discount_pct", "item_category", "order_hour",
        "is_late_night", "distance_km",
        # engineered, time-respecting historical features
        "customer_orders_before", "customer_return_rate_smoothed",
        "is_first_order_for_customer",
        "pincode_orders_before", "pincode_return_rate_smoothed",
        # label
        "is_returned",
    ]
    features_df = df[final_columns].copy()
    features_df.to_csv(OUTPUT_PATH, index=False)

    # ------------------------------------------------------------------
    # 6. Verification block -- prove to yourself this actually worked
    # ------------------------------------------------------------------
    print(f"Saved {len(features_df)} rows to {OUTPUT_PATH}\n")

    print("Spot-check: a customer's FIRST order must always show 0 prior orders.")
    first_orders = features_df[features_df["is_first_order_for_customer"] == 1]
    print(f"  {len(first_orders)} first-orders found; "
          f"customer_orders_before max among them = "
          f"{first_orders['customer_orders_before'].max()} (must be 0)\n")

    print("Correlation of engineered features with the label "
          "(should now be noticeably stronger than the raw columns):")
    corr_cols = [
        "order_value", "discount_pct", "distance_km",
        "customer_return_rate_smoothed", "pincode_return_rate_smoothed",
        "customer_orders_before", "is_returned",
    ]
    print(features_df[corr_cols].corr()["is_returned"].sort_values(ascending=False))

    print("\nIf customer_return_rate_smoothed and pincode_return_rate_smoothed "
          "show noticeably higher correlation than the raw order-level columns, "
          "the feature engineering step did its job -- you've recovered signal "
          "that was hidden in the raw data via customer/pincode identity.")


if __name__ == "__main__":
    main()