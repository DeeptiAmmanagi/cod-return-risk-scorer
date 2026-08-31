"""
dashboard.py
-------------
Streamlit dashboard: model metrics, a live "test an order" box, the
funnel from your last batch run, and the audit/review logs.

NOT TESTED IN A SANDBOX -- I don't have Streamlit available to run this
myself. Run it yourself and report back any errors immediately; treat
this file as a strong first draft, not verified-working code, unlike
every other script so far in this project.

WHERE THIS FILE LIVES:
    razorpay/src/dashboard.py

HOW TO RUN IT:
    cd razorpay/src
    pip install streamlit
    streamlit run dashboard.py
"""

import json
import pandas as pd
import streamlit as st
from pathlib import Path

import scoring_logic as sl
import decision_engine as de

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data"

st.set_page_config(page_title="COD Return-Risk Scorer", layout="wide")
st.title("COD Return-Risk Scorer -- AI Risk Manager")
st.caption("Defense-only. Recommends actions, never auto-cancels orders.")

# ----------------------------------------------------------------------
# Section 1: Final model metrics (from the one-time locked test evaluation)
# ----------------------------------------------------------------------
st.header("1. Model performance (held-out test set)")

eval_path = REPORTS_DIR / "final_test_evaluation.json"
if eval_path.exists():
    with open(eval_path, encoding="utf-8") as f:
        metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("Cost reduction vs. no model", f"{metrics['cost_reduction_pct']:.1%}")

    with st.expander("Full metrics"):
        st.json(metrics)
else:
    st.warning("final_test_evaluation.json not found -- run "
               "src/final_test_evaluation.py first.")

col_a, col_b, col_c = st.columns(3)
for col, filename, caption in [
    (col_a, "pr_curve_baseline.png", "Precision-Recall Curve"),
    (col_b, "calibration_baseline.png", "Calibration Curve"),
    (col_c, "cost_vs_threshold_baseline.png", "Cost vs. Threshold"),
]:
    img_path = REPORTS_DIR / filename
    if img_path.exists():
        col.image(str(img_path), caption=caption, use_container_width=True)

st.divider()

# ----------------------------------------------------------------------
# Section 2: Live "test an order" box
# ----------------------------------------------------------------------
st.header("2. Score a live order")

with st.form("score_order_form"):
    c1, c2, c3 = st.columns(3)
    order_id = c1.text_input("Order ID", value="DEMO_LIVE_001")
    customer_id = c2.text_input("Customer ID", value="CUST_NEW_DEMO")
    delivery_pincode = c3.text_input("Delivery pincode", value="560099")

    c4, c5, c6 = st.columns(3)
    order_value = c4.number_input("Order value (₹)", min_value=1.0, value=1200.0)
    discount_pct = c5.number_input("Discount %", min_value=0.0, max_value=100.0, value=30.0)
    item_category = c6.selectbox(
        "Item category",
        ["Fashion", "Beauty", "Electronics", "Home", "Grocery", "Other"],
    )

    c7, c8 = st.columns(2)
    order_hour = c7.slider("Order hour (24h)", 0, 23, 14)
    distance_km = c8.number_input("Delivery distance (km)", min_value=0.0, value=10.0)

    submitted = st.form_submit_button("Score this order")

if submitted:
    order = {
        "order_id": order_id,
        "customer_id": customer_id,
        "delivery_pincode": delivery_pincode,
        "order_value": order_value,
        "discount_pct": discount_pct,
        "item_category": item_category,
        "order_hour": order_hour,
        "distance_km": distance_km,
    }
    outcome = de.execute_action(order)

    tier_color = {"low": "green", "medium": "orange", "high": "red"}.get(outcome["tier"], "gray")
    st.markdown(f"### Risk score: **{outcome['risk_score']:.3f}** &nbsp; "
                f"Tier: :{tier_color}[**{outcome['tier'].upper()}**]")
    st.write(f"**Action taken:** {outcome['action_taken']} "
             f"(status: {outcome['action_status']})")
    if outcome.get("details"):
        st.json(outcome["details"])

    result = sl.score_order(order)
    st.write("**Top contributing factors:**")
    for factor in result["top_factors"]:
        st.write(f"- `{factor['feature']}` -> {factor['direction']} "
                 f"(contribution: {factor['contribution']:+.3f})")

st.divider()

# ----------------------------------------------------------------------
# Section 3: Last batch run funnel
# ----------------------------------------------------------------------
st.header("3. Last batch run funnel")

batch_path = REPORTS_DIR / "batch_run_summary.json"
if batch_path.exists():
    with open(batch_path, encoding="utf-8") as f:
        batch_summary = json.load(f)

    tier_df = pd.DataFrame(
        list(batch_summary["tier_breakdown"].items()),
        columns=["Tier", "Count"],
    )
    st.bar_chart(tier_df.set_index("Tier"))
    st.write(f"Batch size: {batch_summary['batch_size']} orders "
             f"from `{batch_summary['source_file']}`")
else:
    st.info("No batch run yet -- run `python src/batch_runner.py --n 200` first.")

st.divider()

# ----------------------------------------------------------------------
# Section 4: Audit trail and human review queue
# ----------------------------------------------------------------------
st.header("4. Audit trail")

tab1, tab2 = st.tabs(["Recent scoring decisions", "Human review queue"])

with tab1:
    audit_path = REPORTS_DIR / "audit_log.jsonl"
    if audit_path.exists():
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        recent = [json.loads(l) for l in lines[-20:]]
        st.dataframe(pd.json_normalize(recent), use_container_width=True)
    else:
        st.info("No scoring activity logged yet.")

with tab2:
    review_path = REPORTS_DIR / "human_review_queue.jsonl"
    if review_path.exists():
        lines = review_path.read_text(encoding="utf-8").strip().split("\n")
        if lines and lines[0]:
            queue = [json.loads(l) for l in lines]
            st.dataframe(pd.json_normalize(queue), use_container_width=True)
        else:
            st.success("Review queue is empty.")
    else:
        st.success("Review queue is empty.")