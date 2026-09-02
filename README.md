# COD Return-Risk Scorer
### Track 2 — AI Risk Manager | Razorpay AI Buildathon 2026

**Stop the merchant losing money to fraud, returns and chargebacks — this project targets one class of loss: Cash-on-Delivery (COD) orders that get returned or refused at the doorstep.**

I built a working detector that scores every COD order's return-risk before shipping, and a defense-only action layer that responds proportionally — a soft nudge, or a human review flag — and **never auto-cancels an order.**

---

## Results (held-out test set, evaluated once, never re-tuned against)

| Metric | Value |
|---|---|
| ROC-AUC | 0.751 |
| PR-AUC | 0.447 |
| Precision (at threshold 0.23) | 0.326 |
| Recall (at threshold 0.23) | 0.675 |
| F1 | 0.440 |
| Cost reduction vs. taking no action | **30.3%** |

Model chosen: **Logistic Regression**, selected over XGBoost after a fair, validation-set comparison (XGBoost scored lower on both PR-AUC and total cost — a real result, not a foregone conclusion, and I report it honestly rather than assuming the more complex model would win).

Threshold (0.23) was chosen by minimizing total expected cost — not accuracy or F1 — using an explicit, documented cost matrix (₹150 per missed risky order, ₹40 per unnecessary soft nudge on a good order). See `reports/cost_vs_threshold_baseline.png`.

---

## Architecture

```
Synthetic COD orders (data/generate_data.py)
        │
Feature engineering: time-respecting historical aggregates,
Bayesian-smoothed customer/pincode return rates (data/feature_engineering.py)
        │
Time-based train/val/test split (70/15/15) — test set touched exactly once
        │
Baseline Logistic Regression vs. XGBoost, compared fairly on validation
        │
Cost-based threshold selection (minimizes ₹ cost, not accuracy)
        │
FastAPI scoring endpoint (src/api.py) ── Decision engine (src/decision_engine.py)
        │                                        │
   Returns risk score,                   LOW    → auto-approve
   tier, top factors                     MEDIUM → Razorpay Payment Link nudge
                                          HIGH   → human review queue (never auto-cancel)
        │
Audit log (every decision, every action, timestamped) → Streamlit dashboard
```

---

## Key design decisions

- **Time-based split, not random.** Training only ever sees the past relative to what it's evaluated on — matches how the model would actually be deployed.
- **Smoothed historical features.** A customer's return rate is blended with a global baseline weighted by how much history actually exists, so a single early return doesn't get treated as "100% risk."
- **Cost-aware thresholding, not accuracy-maximizing.** The operating threshold minimizes real ₹ cost using an explicit, justified cost matrix tied to the actual action taken (a soft nudge, not a rejection).
- **Two separate thresholds for two separate costs.** The cost-optimal threshold (0.23) governs any automated action at all; a separate, more conservative threshold (0.55) governs escalation to a human reviewer, because human review capacity is a different, scarcer resource than an automated nudge.
- **Out-of-distribution guardrail.** Orders far outside the training range are forced to human review regardless of model score — see "What broke" below for why this exists.
- **Never auto-cancel, at any tier, under any condition.**

---

## What broke, and how I found and fixed it

I deliberately stress-tested the system (`src/stress_test.py`) with edge cases rather than waiting to find bugs by accident: a brand-new customer with a brand-new pincode, an unseen item category, boundary discount values, a missing required field, and — critically — an order value far outside anything seen in training (₹2,50,000 against a training max of ~₹15,000).

That last case looked fine at first glance (the model returned tier="low", implying "safe") but the raw probability was **5.8 × 10⁻²⁷** — not genuinely confident, but a StandardScaler wildly extrapolating on an input it had never seen anything like. Left unfixed, the system would have auto-approved an obviously anomalous order with entirely unjustified confidence — a silent failure that looks correct but isn't, which is worse than a loud crash.

**Fix:** added an explicit out-of-distribution check (`is_out_of_distribution()` in `scoring_logic.py`) using data-derived bounds on `order_value` and `distance_km`. Any order outside those bounds is forced to the `high` tier and routed to human review, regardless of what the model's raw score says. Verified the fix catches the extreme case while leaving normal orders' scores completely unchanged.

Two smaller bugs also worth noting, both fixed the same day they were found:
- A Windows-specific `UnicodeEncodeError` when writing the ₹ symbol to a report file (fixed by explicit `encoding="utf-8"` on all file writes).
- `delivery_pincode` silently read back as `int64` instead of a string by pandas (fixed with explicit `dtype=str` on ID columns), which had been breaking customer/pincode profile lookups.

---

## Guardrails

- No fully automatic order cancellation, ever, at any risk tier.
- Out-of-distribution inputs are force-routed to human review rather than trusted.
- Failed automated actions (e.g. a Payment Link API call failing) are caught, logged, and escalated to human review as a fallback — never silently dropped, never crash the batch.
- Every scoring decision and every action attempt is logged with a timestamp and the features that drove it (`reports/audit_log.jsonl`, `reports/action_log.jsonl`).
- Model limitations are documented, not hidden: it was trained on synthetic data with explicitly stated generative assumptions (see `src/generate_data.py` docstring).
- No offense-capable output anywhere — this system only detects and routes, it does not generate fraud patterns or evasion techniques.

---

## How to run it

```bash
# 1. Enter the project
cd razorpay

# 2. Setup
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# 3. Build the data and model pipeline, in order
python src/generate_data.py
python src/feature_engineering.py
python src/train_model.py
python src/evaluate_model.py
python src/train_stronger_model.py       # compares XGBoost, baseline wins here
python src/final_test_evaluation.py      # ONE-TIME locked test evaluation

# 4. Stress test (optional but recommended before demoing)
python src/stress_test.py

# 5. Run the system
cd src
uvicorn api:app --reload --port 8000     # API at http://127.0.0.1:8000/docs
# in a separate terminal:
python batch_runner.py --n 200           # demo batch through the decision engine
streamlit run dashboard.py               # full dashboard at http://localhost:8501
```

---

## Tech stack

Python, pandas, scikit-learn, XGBoost, FastAPI, Streamlit, Razorpay test-mode Payment Links API (dry-run mode available with no API keys required — set `RAZORPAY_MODE=live` with `RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET` for real test-mode calls).

---

## Limitations & honest disclosure

- Trained on synthetic data with documented, deliberately-designed generative assumptions (see `generate_data.py`) — not validated against real merchant data.
- Cost matrix values (₹150 / ₹40) are reasoned estimates, not empirically measured; they're explicitly stated as such rather than presented as ground truth.
- The out-of-distribution guardrail uses simple range bounds, not a proper multivariate OOD detector — a reasonable first line of defense, with room to grow into something more rigorous (e.g. Mahalanobis distance or an isolation forest) if extended.
- Real deployment would need the customer/pincode profile store to update incrementally with real outcomes (returns/refusals), not just order counts — the current version approximates this using the static historical snapshot at model-build time.

## Repo structure

```
razorpay/
├── data/                       # generated + engineered datasets, train/val/test splits
├── models/                     # fitted pipeline, chosen threshold + cost config
├── reports/                    # evaluation plots, final metrics, audit/action logs
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── train_stronger_model.py
│   ├── final_test_evaluation.py
│   ├── scoring_logic.py        # core scoring + OOD guardrail logic
│   ├── api.py                  # FastAPI wrapper
│   ├── decision_engine.py      # action execution (Razorpay integration)
│   ├── batch_runner.py         # demo batch runner (never touches test.csv)
│   ├── stress_test.py          # deliberate edge-case testing
│   └── dashboard.py            # Streamlit UI
├── requirements.txt
└── README.md
```
