"""
generate_data.py
-----------------
Generates synthetic Cash-on-Delivery (COD) order data for the
AI Risk Manager (COD return-risk scorer) project.

WHAT THIS SCRIPT DOES:
  1. Creates a pool of synthetic customers and delivery pincodes,
     each with a HIDDEN risk tendency (never exposed to the model).
  2. Generates orders over a simulated time period, assigning each
     order to a customer/pincode and a set of visible, realistic
     order-level features.
  3. Computes a latent "true" return-risk probability per order from
     a weighted combination of several factors + random noise, then
     samples the actual is_returned label from that probability.
  4. Saves the RAW, feature-engineering-ready dataset to data/cod_orders.csv.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO:
  - It does not compute historical/aggregate features (e.g. "customer's
    past return count", "pincode's historical return rate"). Those are
    the job of the next step (feature engineering) and require respecting
    time order to avoid leakage. Doing them here would blur that lesson.
  - It does not expose the hidden risk-affinity values to the main
    dataset. They are saved separately, purely for your own reference/
    debugging -- never feed them into a model.

WHERE THIS FILE LIVES:
    razorpay/src/generate_data.py

HOW TO RUN IT:
    cd razorpay
    # activate your virtual environment first, e.g.:
    #   source .venv/bin/activate        (macOS/Linux)
    #   .venv\\Scripts\\activate           (Windows)
    python src/generate_data.py

OUTPUT:
    razorpay/data/cod_orders.csv              <- the real dataset (use this)
    razorpay/data/_debug_true_affinities.csv  <- hidden ground-truth values,
                                                  for YOUR understanding only
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# ----------------------------------------------------------------------
# CONFIG -- tweak these as you like, but understand what each one does
# ----------------------------------------------------------------------

SEED = 42                      # reproducibility: same seed -> same dataset every run
N_CUSTOMERS = 2000              # size of the customer pool
N_PINCODES = 120                 # size of the delivery-pincode pool
N_ORDERS = 8000                  # total number of synthetic orders to generate
DATE_RANGE_DAYS = 200            # orders are spread across this many days
NOISE_STD = 0.6                  # size of the random noise term in the risk logit
                                  # (increase this if the labels feel "too easy" to predict)

rng = np.random.default_rng(SEED)

# Resolve paths relative to THIS FILE, not the current working directory.
# This means the script saves to the right place no matter where you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# STEP 1: Build the hidden customer and pincode risk pools
# ----------------------------------------------------------------------

def build_customers(n):
    """
    Each customer gets a hidden 'risk_affinity' in [0, 1] drawn from a
    Beta distribution skewed toward low risk, with a tail of genuinely
    risky customers -- this mimics real populations (most people are
    fine, a minority drive most of the returns).
    """
    customer_ids = [f"CUST{100000 + i}" for i in range(n)]
    risk_affinity = rng.beta(a=2.0, b=6.0, size=n)  # mean ~0.25, right-skewed
    return pd.DataFrame({
        "customer_id": customer_ids,
        "_true_risk_affinity": risk_affinity,   # HIDDEN -- leading underscore = do not use as a feature
    })


def build_pincodes(n):
    """
    Each pincode gets a hidden 'base_return_rate' -- representing
    regional factors like delivery reliability, trust in COD, and
    typical courier experience in that area.
    """
    pincodes = [f"{560000 + i}" for i in range(n)]
    base_rate = rng.beta(a=2.0, b=7.0, size=n)  # mean ~0.22, right-skewed
    return pd.DataFrame({
        "delivery_pincode": pincodes,
        "_true_pincode_base_rate": base_rate,   # HIDDEN
    })


customers_df = build_customers(N_CUSTOMERS)
pincodes_df = build_pincodes(N_PINCODES)


# ----------------------------------------------------------------------
# STEP 2: Simulate realistic customer ordering behavior
# ----------------------------------------------------------------------
# Real customer bases have a heavy-tailed order distribution -- a small
# number of repeat buyers order more than average, most order rarely.
# We use a log-normal "popularity" weight per customer, CAPPED at the
# 99.5th percentile so no single customer can dominate the dataset
# (an earlier, uncapped Zipf version let one customer account for 12%
# of all orders -- unrealistic, and it distorts the historical features
# computed in the next step).

raw_popularity = rng.lognormal(mean=0.0, sigma=1.1, size=N_CUSTOMERS)
cap = np.percentile(raw_popularity, 99.5)
raw_popularity = np.clip(raw_popularity, None, cap)
customer_pick_weights = raw_popularity / raw_popularity.sum()


# ----------------------------------------------------------------------
# STEP 3: Category risk weights (based on real-world COD return patterns)
# ----------------------------------------------------------------------
# Fashion and beauty categories have well-documented higher COD return
# rates (sizing/fit issues, impulse buys) versus electronics/grocery.

CATEGORIES = ["Fashion", "Beauty", "Electronics", "Home", "Grocery", "Other"]
CATEGORY_PROBS = [0.30, 0.15, 0.20, 0.15, 0.10, 0.10]
CATEGORY_RISK_WEIGHT = {
    "Fashion": 0.9,
    "Beauty": 0.6,
    "Electronics": -0.4,
    "Home": -0.1,
    "Grocery": -0.7,
    "Other": 0.0,
}


# ----------------------------------------------------------------------
# STEP 4: Generate the orders
# ----------------------------------------------------------------------

start_date = datetime(2025, 1, 1)

# Assign each order a customer and a pincode
order_customer_idx = rng.choice(N_CUSTOMERS, size=N_ORDERS, p=customer_pick_weights)
order_pincode_idx = rng.integers(0, N_PINCODES, size=N_ORDERS)

# Spread orders across the date range, then sort chronologically
order_offsets_days = rng.integers(0, DATE_RANGE_DAYS, size=N_ORDERS)
order_offsets_seconds = rng.integers(0, 86400, size=N_ORDERS)  # random time within the day
order_timestamps = [
    start_date + timedelta(days=int(d), seconds=int(s))
    for d, s in zip(order_offsets_days, order_offsets_seconds)
]

# Order-level visible features
order_value = np.round(np.exp(rng.normal(6.5, 0.6, size=N_ORDERS)), 2)   # log-normal, roughly ₹300 - ₹3000+
order_value = np.clip(order_value, 150, 15000)

discount_pct = np.round(rng.beta(1.5, 4.0, size=N_ORDERS) * 70, 1)        # skewed toward lower discounts, up to ~70%

item_category = rng.choice(CATEGORIES, size=N_ORDERS, p=CATEGORY_PROBS)

distance_km = np.round(np.abs(rng.normal(12, 8, size=N_ORDERS)), 1)
distance_km = np.clip(distance_km, 1, 80)

order_hour = rng.integers(0, 24, size=N_ORDERS)

# Build the orders DataFrame by joining in the hidden customer/pincode values
orders_df = pd.DataFrame({
    "order_id": [f"ORD{500000 + i}" for i in range(N_ORDERS)],
    "customer_id": customers_df["customer_id"].values[order_customer_idx],
    "delivery_pincode": pincodes_df["delivery_pincode"].values[order_pincode_idx],
    "order_timestamp": order_timestamps,
    "order_value": order_value,
    "discount_pct": discount_pct,
    "item_category": item_category,
    "order_hour": order_hour,
    "distance_km": distance_km,
})

# Sort chronologically now -- this matters for every later step
orders_df = orders_df.sort_values("order_timestamp").reset_index(drop=True)

# Bring in the HIDDEN values temporarily, only to compute the label.
# We will drop these columns before saving the main dataset.
orders_df = orders_df.merge(customers_df, on="customer_id", how="left")
orders_df = orders_df.merge(pincodes_df, on="delivery_pincode", how="left")


# ----------------------------------------------------------------------
# STEP 5: Compute the latent risk logit and sample the label
# ----------------------------------------------------------------------
# This is the heart of the simulation. The label is NOT a deterministic
# function of any single column -- it's a weighted combination of several
# signals plus real randomness, then passed through a sigmoid to get a
# probability, then a coin-flip (Bernoulli sample) from that probability.
#
# This mirrors reality: even a genuinely risky order sometimes doesn't
# get returned, and a seemingly safe order sometimes does. A model that
# just memorizes exact rules will fail here -- which is exactly what you
# want, because it forces honest evaluation later.

category_weight = orders_df["item_category"].map(CATEGORY_RISK_WEIGHT).values

# Late-night orders (10pm - 4am) are a mild real-world proxy for lower
# purchase intent / impulse buying.
is_late_night = ((orders_df["order_hour"] >= 22) | (orders_df["order_hour"] <= 4)).astype(int)

# Standardize order_value and distance so their weights are meaningful
order_value_z = (orders_df["order_value"] - orders_df["order_value"].mean()) / orders_df["order_value"].std()
distance_z = (orders_df["distance_km"] - orders_df["distance_km"].mean()) / orders_df["distance_km"].std()
discount_z = (orders_df["discount_pct"] - orders_df["discount_pct"].mean()) / orders_df["discount_pct"].std()

logit = (
    7.0 * orders_df["_true_risk_affinity"].values          # dominant, but hidden -- proxied later via history
    + 5.0 * orders_df["_true_pincode_base_rate"].values     # dominant, but hidden -- proxied later via history
    + 0.5 * category_weight
    + 0.35 * discount_z.values
    + 0.25 * distance_z.values
    + 0.30 * is_late_night.values
    - 0.15 * order_value_z.values                            # slightly higher-value orders are a bit more "considered"
    - 5.0                                                     # intercept, tuned so overall return rate is realistic (~12-18%)
    + rng.normal(0, NOISE_STD, size=N_ORDERS)                # genuine randomness -- the noise term
)

probability = 1 / (1 + np.exp(-logit))
is_returned = rng.binomial(1, probability)

orders_df["is_returned"] = is_returned

print(f"Overall return rate in generated data: {orders_df['is_returned'].mean():.2%}")
print("(A realistic COD return rate is roughly 10-20% -- if yours is wildly outside that, "
      "adjust the intercept in the logit formula above.)")


# ----------------------------------------------------------------------
# STEP 6: Save -- separate the real dataset from the hidden debug values
# ----------------------------------------------------------------------

debug_cols = ["order_id", "customer_id", "delivery_pincode",
              "_true_risk_affinity", "_true_pincode_base_rate", "is_returned"]
orders_df[debug_cols].to_csv(DATA_DIR / "_debug_true_affinities.csv", index=False)

visible_cols = [
    "order_id", "customer_id", "delivery_pincode", "order_timestamp",
    "order_value", "discount_pct", "item_category", "order_hour",
    "distance_km", "is_returned",
]
final_df = orders_df[visible_cols].copy()
final_df.to_csv(DATA_DIR / "cod_orders.csv", index=False)

print(f"\nSaved {len(final_df)} orders to: {DATA_DIR / 'cod_orders.csv'}")
print(f"Saved hidden debug values to:    {DATA_DIR / '_debug_true_affinities.csv'}")
print("\nNext step: feature engineering -- compute historical, "
      "time-respecting aggregates from cod_orders.csv (do NOT touch the debug file).")