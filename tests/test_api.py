import pytest
from fastapi.testclient import TestClient

from src.api import app, ensemble

MODELS_AVAILABLE = ensemble.xgb_model is not None and ensemble.iforest_model is not None

requires_models = pytest.mark.skipif(
    not MODELS_AVAILABLE, reason="Trained model artifacts not present in models/"
)


@pytest.fixture
def client():
    # Context manager form runs the startup hook (DB init + model reload)
    with TestClient(app) as c:
        yield c


def valid_payload(**overrides):
    payload = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C_TEST_SUITE_1",
        "oldbalanceOrg": 250000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C_TEST_SUITE_2",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    payload.update(overrides)
    return payload


class TestHealth:
    def test_health_returns_ok(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "healthy"
        assert "model_loaded" in body


class TestScoreValidation:
    def test_missing_fields_rejected(self, client):
        res = client.post("/score", json={"amount": 100.0})
        assert res.status_code == 422

    def test_non_positive_amount_rejected(self, client):
        res = client.post("/score", json=valid_payload(amount=0.0))
        assert res.status_code == 422

    def test_negative_balance_rejected(self, client):
        res = client.post("/score", json=valid_payload(oldbalanceOrg=-5.0))
        assert res.status_code == 422


@requires_models
class TestScoring:
    def test_score_returns_complete_risk_assessment(self, client):
        res = client.post("/score", json=valid_payload())
        assert res.status_code == 200
        body = res.json()

        for key in (
            "transaction_id",
            "fraud_probability",
            "anomaly_score",
            "ensemble_score",
            "risk_tier",
            "recommendation",
            "top_signals",
            "response_time_ms",
        ):
            assert key in body, f"missing key: {key}"

        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert 0.0 <= body["anomaly_score"] <= 1.0
        assert body["risk_tier"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert body["recommendation"] in {"ALLOW", "REVIEW_LATER", "FLAG_FOR_REVIEW", "BLOCK"}

    def test_balance_draining_transfer_is_not_scored_low(self, client):
        # The canonical PaySim fraud signature: TRANSFER that empties the account
        res = client.post("/score", json=valid_payload())
        assert res.status_code == 200
        assert res.json()["risk_tier"] != "LOW"

    def test_batch_scoring_returns_result_per_transaction(self, client):
        batch = {"transactions": [valid_payload(), valid_payload(nameOrig="C_TEST_SUITE_3")]}
        res = client.post("/score/batch", json=batch)
        assert res.status_code == 200
        body = res.json()
        assert body["batch_size"] == 2
        assert len(body["results"]) == 2


@requires_models
class TestFeedback:
    def test_feedback_recorded_for_scored_transaction(self, client):
        scored = client.post("/score", json=valid_payload()).json()
        res = client.post(
            "/feedback",
            json={
                "transaction_id": scored["transaction_id"],
                "is_fraud_feedback": 1,
                "feedback_notes": "test suite label",
            },
        )
        assert res.status_code == 201
        assert res.json()["status"] == "success"

    def test_feedback_flag_must_be_binary(self, client):
        res = client.post(
            "/feedback", json={"transaction_id": "TX_X", "is_fraud_feedback": 5}
        )
        assert res.status_code == 422


class TestStats:
    def test_stats_reports_model_and_db_metrics(self, client):
        res = client.get("/stats")
        assert res.status_code == 200
        body = res.json()
        assert "model_version" in body
        assert "database_stats" in body
