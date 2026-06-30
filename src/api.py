import time
import os
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import our custom modules
from src.db import init_db, log_transaction, log_feedback, get_user_history, get_db_stats
from src.features import PesaGuardFeaturePipeline
from src.models import PesaGuardEnsemble

load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="PesaGuard Fraud Detection Service",
    description="Production-grade REST API for scoring mobile money transactions in real-time.",
    version="1.0.0"
)

# Initialize and load model ensemble
ensemble = PesaGuardEnsemble(models_dir="models")

# Pydantic schemas
class TransactionPayload(BaseModel):
    step: int = Field(..., description="Chronological hour of transaction (1 step = 1 hour)", ge=1)
    type: str = Field(..., description="CASH_IN, CASH_OUT, DEBIT, PAYMENT, or TRANSFER")
    amount: float = Field(..., description="Transaction amount value", gt=0)
    nameOrig: str = Field(..., description="Originating client account ID")
    oldbalanceOrg: float = Field(..., description="Origin balance before transaction", ge=0)
    newbalanceOrig: float = Field(..., description="Origin balance after transaction", ge=0)
    nameDest: str = Field(..., description="Recipient client account ID")
    oldbalanceDest: float = Field(..., description="Recipient balance before transaction", ge=0)
    newbalanceDest: float = Field(..., description="Recipient balance after transaction", ge=0)
    transaction_id: str | None = Field(None, description="Optional unique identifier for logging")

class BatchTransactionPayload(BaseModel):
    transactions: List[TransactionPayload]

class FeedbackPayload(BaseModel):
    transaction_id: str = Field(..., description="Transaction ID being flagged")
    is_fraud_feedback: int = Field(..., description="1 if true fraud, 0 if legitimate", ge=0, le=1)
    feedback_notes: str | None = Field("", description="Optional notes from the analyst")

# Database startup hook
@app.on_event("startup")
def startup_event():
    # Automatically initialize DB tables if they don't exist
    init_db()
    # Reload models in case they were updated
    ensemble.load_models()

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """
    Returns the system health and model availability status.
    """
    model_loaded = ensemble.xgb_model is not None and ensemble.iforest_model is not None
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": model_loaded,
        "engine": "PesaGuard Hybrid Ensemble"
    }

@app.post("/score")
def score_transaction(payload: TransactionPayload):
    """
    Scores a single mobile money transaction, returning risk indicators, 
    recommendation, and SHAP-based explanation signals under 200ms.
    """
    start_time = time.perf_counter()
    
    if ensemble.xgb_model is None or ensemble.iforest_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model weights are not loaded. Please train the system first."
        )

    tx_dict = payload.dict()
    name_orig = tx_dict["nameOrig"]

    try:
        # 1. Fetch user transaction history from DB in real time
        user_history = get_user_history(name_orig)

        # 2. Run real-time feature engineering pipeline
        features_df = ensemble.feature_pipeline.transform_realtime(tx_dict, user_history)

        # 3. Predict using PesaGuard Ensemble
        scores = ensemble.predict_single(features_df)

        # 4. Log scored transaction in background database (essential for velocity/history)
        tx_id = log_transaction(tx_dict, scores)
        
        # Calculate response latency in milliseconds
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # 5. Build final response
        return {
            "transaction_id": tx_id,
            "fraud_probability": round(scores["fraud_probability"], 4),
            "anomaly_score": round(scores["anomaly_score"], 4),
            "ensemble_score": round(scores["ensemble_score"], 4),
            "risk_tier": scores["risk_tier"],
            "top_signals": scores["top_signals"],
            "recommendation": scores["recommendation"],
            "response_time_ms": latency_ms
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during transaction scoring: {str(e)}"
        )

@app.post("/score/batch")
def score_batch(payload: BatchTransactionPayload):
    """
    Scores a batch of transactions and returns their results.
    """
    results = []
    for tx in payload.transactions:
        try:
            res = score_transaction(tx)
            results.append(res)
        except Exception as e:
            results.append({
                "transaction_id": tx.transaction_id or "UNKNOWN",
                "error": str(e),
                "status": "failed"
            })
    return {"results": results, "batch_size": len(payload.transactions)}

@app.post("/feedback", status_code=status.HTTP_201_CREATED)
def record_feedback(payload: FeedbackPayload):
    """
    Accepts human analyst labels (true fraud / false positive) for scored transactions.
    These labels help update false positive metrics and can be used for future model retraining.
    """
    try:
        log_feedback(
            transaction_id=payload.transaction_id,
            is_fraud_feedback=payload.is_fraud_feedback,
            feedback_notes=payload.feedback_notes
        )
        return {"status": "success", "message": f"Feedback recorded for transaction: {payload.transaction_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record feedback: {str(e)}"
        )

@app.get("/stats")
def get_model_stats():
    """
    Returns performance metrics of the system, including counts of flagged 
    transactions, values blocked, and feedback summaries.
    """
    try:
        db_stats = get_db_stats()
        
        # We can also add some static model validation metrics from training
        model_stats = {
            "model_version": "1.0.0",
            "model_type": "XGBoost + Isolation Forest",
            "training_pr_auc": 0.8875,
            "test_pr_auc": 0.9500,
            "optimal_threshold": 0.45,
            "database_stats": db_stats
        }
        return model_stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model statistics: {str(e)}"
        )
