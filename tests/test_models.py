import pandas as pd
import pytest

from src.models import PesaGuardEnsemble


@pytest.fixture
def empty_ensemble(tmp_path):
    """Ensemble pointed at a directory with no model artifacts."""
    return PesaGuardEnsemble(models_dir=str(tmp_path))


class TestModelLoading:
    def test_missing_artifacts_leave_models_unloaded(self, empty_ensemble):
        assert empty_ensemble.xgb_model is None
        assert empty_ensemble.iforest_model is None
        assert empty_ensemble.shap_explainer is None

    def test_predict_without_models_raises(self, empty_ensemble):
        with pytest.raises(ValueError, match="not loaded"):
            empty_ensemble.predict_single(pd.DataFrame([{"amount": 1.0}]))


class TestAnomalyScoreScaling:
    def test_scores_are_clipped_to_unit_interval(self, empty_ensemble):
        assert empty_ensemble.scale_anomaly_score(-10.0) == 1.0
        assert empty_ensemble.scale_anomaly_score(10.0) == 0.0

    def test_more_anomalous_decisions_score_higher(self, empty_ensemble):
        # Isolation Forest: negative decision scores mean outliers
        outlier = empty_ensemble.scale_anomaly_score(-0.15)
        inlier = empty_ensemble.scale_anomaly_score(0.15)
        assert outlier > inlier

    def test_scores_stay_within_bounds_across_range(self, empty_ensemble):
        for decision in [-0.5, -0.2, 0.0, 0.2, 0.5]:
            score = empty_ensemble.scale_anomaly_score(decision)
            assert 0.0 <= score <= 1.0


class TestRiskTiers:
    @pytest.mark.parametrize(
        "score,tier,recommendation",
        [
            (0.0, "LOW", "ALLOW"),
            (0.14, "LOW", "ALLOW"),
            (0.15, "MEDIUM", "REVIEW_LATER"),
            (0.44, "MEDIUM", "REVIEW_LATER"),
            (0.45, "HIGH", "FLAG_FOR_REVIEW"),
            (0.74, "HIGH", "FLAG_FOR_REVIEW"),
            (0.75, "CRITICAL", "BLOCK"),
            (1.0, "CRITICAL", "BLOCK"),
        ],
    )
    def test_tier_boundaries(self, empty_ensemble, score, tier, recommendation):
        assert empty_ensemble.get_risk_tier_and_recommendation(score) == (tier, recommendation)


class TestFeatureDescriptions:
    def test_first_transaction_flag_described(self, empty_ensemble):
        desc = empty_ensemble.get_feature_description("time_since_last_transaction", 999.0)
        assert "First transaction" in desc

    def test_unknown_feature_falls_back_gracefully(self, empty_ensemble):
        desc = empty_ensemble.get_feature_description("some_new_feature", 1.23)
        assert "some_new_feature" in desc
