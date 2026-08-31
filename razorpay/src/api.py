"""
api.py
-------
Thin FastAPI wrapper around scoring_logic.py. All real logic (model
loading, profile lookups, tiering, interpretability, audit logging) is
already tested in scoring_logic.py -- this file just exposes it over HTTP.

WHERE THIS FILE LIVES:
    razorpay/src/api.py

HOW TO RUN IT:
    cd razorpay/src
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

THEN TEST IT:
    Open http://127.0.0.1:8000/docs for interactive Swagger UI, or:

    curl -X POST http://127.0.0.1:8000/score-order \\
      -H "Content-Type: application/json" \\
      -d '{
            "order_id": "ORD999001",
            "customer_id": "CUST_NEW_001",
            "delivery_pincode": "560099",
            "order_value": 1200,
            "discount_pct": 35,
            "item_category": "Fashion",
            "order_hour": 23,
            "distance_km": 18
          }'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import scoring_logic as sl

app = FastAPI(
    title="COD Return-Risk Scoring API",
    description="Scores Cash-on-Delivery orders for return/refusal risk. "
                 "Defense-only: recommends actions, never auto-cancels.",
    version="1.0.0",
)


class OrderRequest(BaseModel):
    order_id: str
    customer_id: str
    delivery_pincode: str
    order_value: float = Field(..., gt=0)
    discount_pct: float = Field(..., ge=0, le=100)
    item_category: str
    order_hour: int = Field(..., ge=0, le=23)
    distance_km: float = Field(..., ge=0)


class TopFactor(BaseModel):
    feature: str
    contribution: float
    direction: str


class ScoreResponse(BaseModel):
    order_id: str
    risk_score: float
    tier: str
    action: str
    top_factors: list[TopFactor]
    low_risk_threshold: float
    high_risk_threshold: float


@app.get("/health")
def health():
    """Basic liveness check -- also confirms the model actually loaded."""
    return {
        "status": "ok",
        "low_risk_threshold": sl.LOW_RISK_THRESHOLD,
        "high_risk_threshold": sl.HIGH_RISK_THRESHOLD,
        "known_customers": len(sl._customer_profiles),
        "known_pincodes": len(sl._pincode_profiles),
    }


@app.post("/score-order", response_model=ScoreResponse)
def score_order_endpoint(order: OrderRequest):
    """
    Scores a single COD order for return/refusal risk.
    Never auto-cancels an order -- returns a recommended action only;
    the decision engine (a separate step) is responsible for actually
    executing it, with its own guardrails.
    """
    try:
        result = sl.score_order(order.model_dump())
    except Exception as e:
        # Fail safe and loud, not silent -- a scoring failure should be
        # visible, not swallowed into a default "low risk" result.
        raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)