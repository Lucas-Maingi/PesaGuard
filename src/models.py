import os
import numpy as np
import pandas as pd
import joblib
import shap

class PesaGuardEnsemble:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.xgb_model = None
        self.iforest_model = None
        self.feature_pipeline = None
        self.shap_explainer = None
        
        # Load models if they exist
        self.load_models()

    def load_models(self):
        xgb_path = os.path.join(self.models_dir, "xgb_model.joblib")
        iforest_path = os.path.join(self.models_dir, "iforest_model.joblib")
        pipeline_path = os.path.join(self.models_dir, "feature_pipeline.joblib")

        if os.path.exists(xgb_path) and os.path.exists(iforest_path) and os.path.exists(pipeline_path):
            self.xgb_model = joblib.load(xgb_path)
            self.iforest_model = joblib.load(iforest_path)
            self.feature_pipeline = joblib.load(pipeline_path)
            
            # Initialize SHAP explainer for XGBoost. 
            # In scikit-learn's CalibratedClassifierCV wrapper, the base estimator is inside `estimator` or `calibrated_classifiers_[0].estimator`
            base_xgb = self.xgb_model
            if hasattr(self.xgb_model, 'calibrated_classifiers_'):
                base_xgb = self.xgb_model.calibrated_classifiers_[0].estimator
            elif hasattr(self.xgb_model, 'estimator'):
                base_xgb = self.xgb_model.estimator
                
            # Unwrap any meta-estimators (like FrozenEstimator or Pipeline) to get the raw model for SHAP
            while hasattr(base_xgb, 'estimator'):
                base_xgb = base_xgb.estimator
                
            self.shap_explainer = shap.TreeExplainer(base_xgb)
            print("PesaGuard models loaded successfully.")
        else:
            print("Warning: PesaGuard model files not found. Model scoring will be unavailable until trained.")

    def scale_anomaly_score(self, decision_score):
        """
        Converts raw Isolation Forest decision score into [0, 1] range.
        Standard Isolation Forest outputs decision_function in [-0.5, 0.5] range.
        - Negative values indicate outliers.
        - Positive values indicate inliers.
        We invert and normalize so that 1 is highly anomalous, 0 is normal.
        """
        # Isolation forest decision scores usually cluster between -0.4 and 0.2
        # We cap and map to [0, 1]
        score = -decision_score  # Invert so higher is anomalous
        # Normalize assuming range is approx -0.2 to 0.4
        mapped = (score - (-0.2)) / (0.4 - (-0.2))
        return float(np.clip(mapped, 0.0, 1.0))

    def get_risk_tier_and_recommendation(self, score):
        if score < 0.15:
            return "LOW", "ALLOW"
        elif score < 0.45:
            return "MEDIUM", "REVIEW_LATER"
        elif score < 0.75:
            return "HIGH", "FLAG_FOR_REVIEW"
        else:
            return "CRITICAL", "BLOCK"

    def predict_single(self, features_df: pd.DataFrame) -> dict:
        """
        Predicts fraud metrics for a single engineered feature row.
        """
        if self.xgb_model is None or self.iforest_model is None:
            raise ValueError("Models are not loaded. Please train models first.")

        # Get XGBoost probability
        xgb_prob = float(self.xgb_model.predict_proba(features_df)[0, 1])

        # Get Isolation Forest anomaly score
        iforest_decision = float(self.iforest_model.decision_function(features_df)[0])
        anomaly_score = self.scale_anomaly_score(iforest_decision)

        # Ensemble Score (0.7 * Supervised + 0.3 * Unsupervised)
        ensemble_score = float(0.7 * xgb_prob + 0.3 * anomaly_score)

        # Risk tier and recommendation
        risk_tier, recommendation = self.get_risk_tier_and_recommendation(ensemble_score)

        # Generate SHAP explanations for top driving signals
        top_signals = self.generate_shap_signals(features_df)

        return {
            "fraud_probability": xgb_prob,
            "anomaly_score": anomaly_score,
            "ensemble_score": ensemble_score,
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "top_signals": top_signals
        }

    def generate_shap_signals(self, features_df: pd.DataFrame) -> list:
        """
        Computes SHAP values to explain the supervised model prediction.
        Returns the top 3 driving features and a description of their impact.
        """
        if self.shap_explainer is None:
            return []

        try:
            # SHAP expects the base model features
            shap_vals = self.shap_explainer.shap_values(features_df)[0]
            feature_names = features_df.columns.tolist()

            # Pair features with their SHAP values and current values
            signals = []
            for name, val in zip(feature_names, shap_vals):
                raw_val = features_df[name].values[0]
                signals.append({
                    "name": name,
                    "shap_val": float(val),
                    "raw_val": float(raw_val)
                })

            # Sort by absolute SHAP value in descending order (highest impact)
            signals = sorted(signals, key=lambda x: abs(x["shap_val"]), reverse=True)

            top_signals = []
            for sig in signals[:3]:
                # Only include positive impact features (driving the score up)
                # or features with significant negative impact if they reduce fraud risk
                impact = "high" if abs(sig["shap_val"]) > 0.5 else "medium" if abs(sig["shap_val"]) > 0.15 else "low"
                
                desc = self.get_feature_description(sig["name"], sig["raw_val"])
                top_signals.append({
                    "signal": sig["name"],
                    "description": desc,
                    "impact": impact,
                    "shap_value": sig["shap_val"]
                })
                
            return top_signals
        except Exception as e:
            print(f"Error computing SHAP: {e}")
            return []

    def get_feature_description(self, feature_name, value):
        """
        Converts feature name and value into human-readable descriptions for operators.
        """
        if feature_name == "velocity_1h":
            return f"Velocity check: {int(value)} transaction(s) in last hour."
        elif feature_name == "velocity_24h":
            return f"Velocity check: {int(value)} transaction(s) in last 24 hours."
        elif feature_name == "amount_deviation":
            if value > 0:
                return f"Transaction amount is {value:.2f} standard deviations above user average."
            else:
                return f"Transaction amount is in line with user average (Z-score: {value:.2f})."
        elif feature_name == "hour_of_day":
            return f"Transaction initiated at {int(value):02d}:00 (high-risk hour)."
        elif feature_name == "amount_to_balance_ratio":
            return f"Amount-to-balance ratio is {value*100:.1f}% of current balance."
        elif feature_name == "merchant_risk_score":
            return f"Destination account category historical fraud rate: {value*100:.2f}%."
        elif feature_name == "cross_border":
            return "Transaction involves an external merchant account transfer."
        elif feature_name == "time_since_last_transaction":
            if value == 999.0:
                return "First transaction recorded for this account (no history)."
            else:
                return f"Time elapsed since last transaction: {value:.1f} step hours."
        elif feature_name == "amount":
            return f"Transaction amount value: {value:,.2f} units."
        elif feature_name == "is_transfer":
            return "Transaction type is TRANSFER (high fraud prevalence)."
        elif feature_name == "is_cash_out":
            return "Transaction type is CASH_OUT (high cash withdrawal risk)."
        else:
            return f"Feature {feature_name} has value {value:.2f}."
