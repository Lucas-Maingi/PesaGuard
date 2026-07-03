import pandas as pd
import pytest

from src.features import PesaGuardFeaturePipeline

EXPECTED_FEATURES = [
    "amount_deviation",
    "velocity_1h",
    "velocity_24h",
    "amount_to_balance_ratio",
    "hour_of_day",
    "is_round_amount",
    "merchant_risk_score",
    "cross_border",
    "time_since_last_transaction",
    "amount_percentile",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "is_transfer",
    "is_cash_out",
]


def make_batch_df():
    return pd.DataFrame(
        [
            # step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest
            {"step": 1, "type": "PAYMENT", "amount": 500.0, "nameOrig": "C1", "oldbalanceOrg": 1000.0,
             "newbalanceOrig": 500.0, "nameDest": "M1", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
            {"step": 2, "type": "TRANSFER", "amount": 5000.0, "nameOrig": "C2", "oldbalanceOrg": 5000.0,
             "newbalanceOrig": 0.0, "nameDest": "C3", "oldbalanceDest": 0.0, "newbalanceDest": 5000.0},
            {"step": 3, "type": "CASH_OUT", "amount": 5000.0, "nameOrig": "C3", "oldbalanceOrg": 5000.0,
             "newbalanceOrig": 0.0, "nameDest": "C4", "oldbalanceDest": 100.0, "newbalanceDest": 5100.0},
            {"step": 4, "type": "PAYMENT", "amount": 250.0, "nameOrig": "C1", "oldbalanceOrg": 500.0,
             "newbalanceOrig": 250.0, "nameDest": "M2", "oldbalanceDest": 0.0, "newbalanceDest": 0.0},
        ]
    )


@pytest.fixture
def fitted_pipeline():
    df = make_batch_df()
    y = pd.Series([0, 1, 1, 0])
    return PesaGuardFeaturePipeline().fit(df, y)


class TestFit:
    def test_learns_risk_score_per_destination_prefix(self, fitted_pipeline):
        # Destinations: M1/M2 (merchants, 0 fraud), C3/C4 (customers, 2 frauds)
        assert fitted_pipeline.merchant_risk_scores_["M"] == 0.0
        assert fitted_pipeline.merchant_risk_scores_["C"] == 1.0

    def test_default_risk_is_overall_fraud_rate(self, fitted_pipeline):
        assert fitted_pipeline.default_merchant_risk_ == pytest.approx(0.5)


class TestBatchTransform:
    def test_produces_expected_feature_columns(self, fitted_pipeline):
        out = fitted_pipeline.transform(make_batch_df())
        assert list(out.columns) == EXPECTED_FEATURES
        assert len(out) == 4

    def test_transaction_type_flags(self, fitted_pipeline):
        out = fitted_pipeline.transform(make_batch_df())
        # Rows are sorted by step: PAYMENT, TRANSFER, CASH_OUT, PAYMENT
        assert out["is_transfer"].tolist() == [0, 1, 0, 0]
        assert out["is_cash_out"].tolist() == [0, 0, 1, 0]

    def test_merchant_destination_flagged_as_cross_border(self, fitted_pipeline):
        out = fitted_pipeline.transform(make_batch_df())
        assert out["cross_border"].tolist() == [1, 0, 0, 1]

    def test_balance_drain_has_ratio_near_one(self, fitted_pipeline):
        out = fitted_pipeline.transform(make_batch_df())
        # The TRANSFER empties the full 5000 balance
        assert out["amount_to_balance_ratio"].iloc[1] == pytest.approx(1.0, abs=1e-3)


class TestRealtimeTransform:
    def make_tx(self, **overrides):
        tx = {
            "step": 100,
            "type": "TRANSFER",
            "amount": 9000.0,
            "nameOrig": "C10",
            "oldbalanceOrg": 9000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C99",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        }
        tx.update(overrides)
        return tx

    def test_no_history_uses_first_transaction_defaults(self, fitted_pipeline):
        out = fitted_pipeline.transform_realtime(self.make_tx(), pd.DataFrame())
        row = out.iloc[0]
        assert len(out) == 1
        assert row["velocity_1h"] == 1
        assert row["velocity_24h"] == 1
        assert row["time_since_last_transaction"] == 999.0
        assert row["amount_percentile"] == 0.5
        assert row["amount_deviation"] == 0.0

    def test_history_drives_velocity_and_deviation(self, fitted_pipeline):
        history = pd.DataFrame({"step": [98, 99, 100], "amount": [100.0, 120.0, 110.0]})
        out = fitted_pipeline.transform_realtime(self.make_tx(), history)
        row = out.iloc[0]
        # Two transactions within the last hour (steps 99, 100) plus the current one
        assert row["velocity_1h"] == 3
        assert row["velocity_24h"] == 4
        assert row["time_since_last_transaction"] == 0.0
        # 9000 vs an average of ~110 is a massive positive deviation
        assert row["amount_deviation"] > 100
        # Larger than all historical amounts
        assert row["amount_percentile"] == 1.0

    def test_stale_history_outside_30_day_window_ignored(self, fitted_pipeline):
        history = pd.DataFrame({"step": [1, 2], "amount": [100.0, 120.0]})
        out = fitted_pipeline.transform_realtime(self.make_tx(step=1000), history)
        row = out.iloc[0]
        assert row["velocity_1h"] == 1
        assert row["amount_deviation"] == 0.0

    def test_round_amount_detection(self, fitted_pipeline):
        round_tx = fitted_pipeline.transform_realtime(self.make_tx(amount=50000.0), pd.DataFrame())
        odd_tx = fitted_pipeline.transform_realtime(self.make_tx(amount=49871.37), pd.DataFrame())
        assert round_tx.iloc[0]["is_round_amount"] == 1
        assert odd_tx.iloc[0]["is_round_amount"] == 0

    def test_realtime_columns_match_batch_columns(self, fitted_pipeline):
        batch_cols = list(fitted_pipeline.transform(make_batch_df()).columns)
        realtime_cols = list(fitted_pipeline.transform_realtime(self.make_tx(), pd.DataFrame()).columns)
        assert realtime_cols == batch_cols
