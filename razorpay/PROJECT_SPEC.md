# AI Risk Manager — Project Specification

## Track

Track 2 — AI Risk Manager  
Razorpay AI Buildathon 2026

## Objective

Build a defense-only AI system that predicts the risk of a Cash-on-Delivery (COD) order being returned or refused before shipping.

The system:

1. Receives COD order information.
2. Calculates a return-risk probability.
3. Assigns a LOW, MEDIUM, or HIGH risk tier.
4. Identifies the main factors contributing to the score.
5. Recommends a proportional defense action.
6. Logs every scoring decision and action for auditing.
7. Routes anomalous or high-risk cases to human review.

The system never automatically cancels an order based solely on the model prediction.

---

## Target

`return_or_refuse`

- `0` = Delivered successfully
- `1` = Returned or refused

---

## Input Features

- `order_value`
- `discount_percent`
- `customer_order_count`
- `customer_return_rate`
- `customer_cod_return_rate`
- `pincode_return_rate`
- `product_category`
- `orders_last_30_days`
- `hour_of_day`
- `first_time_customer`
- `previous_return_count`
- `payment_history_score`

Historical customer and pincode features are calculated using time-respecting data so that future information does not leak into earlier orders.

---

## Data

Real merchant transaction data was not available.

Therefore, the project uses synthetic COD order data designed to contain realistic relationships between customer behaviour, order characteristics, and return risk.

Risk is generated from multiple interacting signals and noise rather than allowing a single feature to trivially determine the target.

The synthetic-data assumptions are documented in `src/generate_data.py`.

---

## Data Splitting

The dataset is split chronologically:

- 70% training
- 15% validation
- 15% held-out test

A time-based split is used instead of a random split to better simulate deployment, where the model predicts future orders using information available in the past.

The held-out test set is evaluated once after model and threshold decisions are finalized.

---

## Feature Engineering

Historical customer and pincode behaviour is represented using time-respecting aggregates.

Return-rate features use Bayesian-style smoothing toward the global baseline so that customers or pincodes with very little history do not receive extreme risk estimates from a single observation.

---

## Machine Learning

### Baseline

Logistic Regression.

### Stronger comparison model

XGBoost.

Both models were compared fairly on the validation set.

Logistic Regression was selected because it achieved better validation PR-AUC and lower expected business cost for this dataset.

The more complex model was not assumed to be better simply because it was XGBoost.

---

## Final Model Performance

Held-out test set:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.751 |
| PR-AUC | 0.447 |
| Precision | 0.326 |
| Recall | 0.675 |
| F1 | 0.440 |
| Cost reduction vs. no action | **30.3%** |

The final evaluation is stored in:

`reports/final_test_evaluation.md`

---

## Cost-Aware Decision Threshold

The system does not select its operating threshold using accuracy alone.

The automated-action threshold was selected by minimizing expected business cost on the validation set.

### Cost assumptions

- False Negative: ₹150
- False Positive / unnecessary soft nudge: ₹40

The chosen automated-action threshold is:

`0.23`

This threshold is based on expected business loss rather than simply maximizing accuracy or F1.

---

## Risk Tiers

### LOW

Low predicted return risk.

Action:

`auto_approve`

No additional customer intervention is required.

### MEDIUM

Moderate predicted return risk.

Action:

`send_pre_payment_nudge`

The system can send/offer a Razorpay Payment Link nudge rather than blocking the order.

### HIGH

High predicted return risk.

Action:

`flag_for_human_review`

A human reviews the order.

The system never automatically cancels the order.

---

## Human Review Threshold

Two thresholds are deliberately used for different operational purposes.

### Automated action threshold

`0.23`

Controls whether the system takes an automated defensive action.

### Human escalation threshold

`0.55`

Used for the more conservative high-risk human-review decision.

Human review is treated as a scarcer operational resource than an automated nudge.

---

## Out-of-Distribution Guardrail

The model can produce highly confident-looking predictions when given inputs far outside its training distribution.

During deliberate stress testing, an order worth ₹2,50,000 produced an extremely small raw model probability despite the value being far outside the training range.

This exposed a silent failure mode: the model could appear confident on an input it had never meaningfully learned from.

### Fix

An explicit out-of-distribution check was added to:

`src/scoring_logic.py`

The guardrail uses data-derived bounds for important numeric features such as:

- `order_value`
- `distance_km`

When an order falls outside the accepted training-derived range:

- the raw model score is still recorded;
- the model score is not trusted for the final tier;
- the order is forced to `HIGH`;
- the order is routed to human review;
- the reason is recorded as `out_of_distribution`.

Normal in-distribution orders continue to receive their original model scores.

---

## Stress Testing

The system was deliberately tested against edge cases using:

`src/stress_test.py`

Test cases included:

1. Brand-new customer and brand-new pincode.
2. Previously unseen item category.
3. Extremely high order value.
4. Zero discount and zero delivery distance.
5. Maximum discount boundary (100%).
6. Repeated scoring for the same customer.
7. Missing required input field.

The missing-field case intentionally fails loudly with a `KeyError` rather than silently producing an incorrect result.

The extreme order-value case exposed the OOD failure described above and resulted in the addition of the OOD guardrail.

After the fix, the extreme order was correctly forced to:

`HIGH → flagged_for_human_review`

---

## Guardrails

The system follows these defense-only principles:

- No automatic order cancellation.
- High-risk cases are routed to human review.
- Out-of-distribution inputs are not blindly trusted.
- Failed automated actions are caught and escalated rather than silently dropped.
- Every scoring decision is logged.
- Every action attempt is logged.
- Model limitations and synthetic-data assumptions are explicitly documented.
- The system does not generate fraud patterns, evasion techniques, or other offense-capable outputs.

---

## Auditability

Each scoring decision records information including:

- timestamp
- order/customer identifiers
- input features
- risk score
- risk tier
- selected action
- top contributing factors
- OOD status/reason
- action result

Relevant logs are stored under:

`reports/`

The Streamlit dashboard reads the same audit information and provides visibility into recent decisions and the human-review queue.

---

## System Architecture

```text
Synthetic COD Orders
        |
        v
Feature Engineering
(time-respecting historical features)
        |
        v
Chronological Train / Validation / Test Split
        |
        v
Logistic Regression + XGBoost Comparison
        |
        v
Cost-Based Threshold Selection
        |
        v
FastAPI Scoring Endpoint
        |
        v
Risk Scoring + OOD Guardrail
        |
        v
Decision Engine
   /       |        \
 LOW    MEDIUM      HIGH
  |        |          |
Approve  Payment    Human Review
         Link
         Nudge
        |
        v
Audit / Action Logs
        |
        v
Streamlit Dashboard