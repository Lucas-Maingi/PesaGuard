import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from imblearn.over_sampling import SMOTE
from src.features import PesaGuardFeaturePipeline

def train_system():
    # 1. Load data
    sample_path = "data/paysim_sample.csv"
    if not os.path.exists(sample_path):
        print(f"Error: Sample data not found at {sample_path}. Run download_data.py first.")
        return

    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df):,} rows from {sample_path}")

    # Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Created models/ directory.")

    # 2. Chronological Split (64/16/20)
    # The data is already sorted by step (chronological order)
    n = len(df)
    train_idx = int(n * 0.64)
    val_idx = int(n * 0.80)

    train_df = df.iloc[:train_idx]
    val_df = df.iloc[train_idx:val_idx]
    test_df = df.iloc[val_idx:]

    print(f"Splits:")
    print(f"  - Train: {len(train_df):,} rows (Steps {train_df['step'].min()} - {train_df['step'].max()}, Fraud: {train_df['isFraud'].sum()})")
    print(f"  - Val:   {len(val_df):,} rows (Steps {val_df['step'].min()} - {val_df['step'].max()}, Fraud: {val_df['isFraud'].sum()})")
    print(f"  - Test:  {len(test_df):,} rows (Steps {test_df['step'].min()} - {test_df['step'].max()}, Fraud: {test_df['isFraud'].sum()})")

    # Split features and labels
    X_train, y_train = train_df.drop(columns=['isFraud']), train_df['isFraud']
    X_val, y_val = val_df.drop(columns=['isFraud']), val_df['isFraud']
    X_test, y_test = test_df.drop(columns=['isFraud']), test_df['isFraud']

    # 3. Fit and Transform Feature Pipeline
    print("\nFitting feature pipeline on training split...")
    pipeline = PesaGuardFeaturePipeline()
    pipeline.fit(X_train, y_train)

    print("Transforming train, validation, and test features...")
    X_train_trans = pipeline.transform(X_train)
    X_val_trans = pipeline.transform(X_val)
    X_test_trans = pipeline.transform(X_test)

    # 4. Fit Unsupervised Isolation Forest (trained on train features)
    print("\nTraining Isolation Forest on training split...")
    iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iforest.fit(X_train_trans)

    # 5. Class Imbalance Comparison
    print("\n" + "="*50)
    print(" COMPARING CLASS IMBALANCE STRATEGIES ")
    print("="*50)

    # Strategy A: XGBoost scale_pos_weight
    neg_count = sum(y_train == 0)
    pos_count = sum(y_train == 1)
    scale_weight = neg_count / (pos_count + 1e-5)
    print(f"Strategy A: XGBoost built-in class weight (scale_pos_weight = {scale_weight:.2f})")
    
    xgb_weighted = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=-1
    )
    xgb_weighted.fit(X_train_trans, y_train)
    
    val_probs_weighted = xgb_weighted.predict_proba(X_val_trans)[:, 1]
    test_probs_weighted = xgb_weighted.predict_proba(X_test_trans)[:, 1]
    
    auc_weighted = roc_auc_score(y_test, test_probs_weighted)
    pr_auc_weighted = average_precision_score(y_test, test_probs_weighted)
    print(f"  Test ROC-AUC: {auc_weighted:.4f}")
    print(f"  Test PR-AUC (Average Precision): {pr_auc_weighted:.4f}")

    # Strategy B: SMOTE Oversampling
    print(f"\nStrategy B: SMOTE oversampling on training split...")
    try:
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_trans, y_train)
        print(f"  Resampled size: {len(X_train_resampled):,} rows (Fraud: {sum(y_train_resampled == 1):,})")

        xgb_smote = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1
        )
        xgb_smote.fit(X_train_resampled, y_train_resampled)
        
        val_probs_smote = xgb_smote.predict_proba(X_val_trans)[:, 1]
        test_probs_smote = xgb_smote.predict_proba(X_test_trans)[:, 1]
        
        auc_smote = roc_auc_score(y_test, test_probs_smote)
        pr_auc_smote = average_precision_score(y_test, test_probs_smote)
        print(f"  Test ROC-AUC: {auc_smote:.4f}")
        print(f"  Test PR-AUC (Average Precision): {pr_auc_smote:.4f}")
    except Exception as e:
        print(f"  SMOTE training failed: {e}")
        pr_auc_smote = -1.0

    # Determine best model
    best_strategy = "Weighted XGBoost"
    best_xgb = xgb_weighted
    if pr_auc_smote > pr_auc_weighted:
        best_strategy = "SMOTE XGBoost"
        best_xgb = xgb_smote
        print(f"\nSMOTE worked better than scale_pos_weight.")
    else:
        print(f"\nscale_pos_weight worked better than SMOTE.")
        
    print(f"Selected strategy for production: {best_strategy}")
    print("="*50)

    # 6. Probability Calibration using Validation Split
    print("\nCalibrating model probabilities using Isotonic Regression on Validation split...")
    # Calibrate the base model. CalibratedClassifierCV wraps the fitted model.
    # We wrap the best fitted model in FrozenEstimator to prevent it from being re-fit.
    calibrated_model = CalibratedClassifierCV(estimator=FrozenEstimator(best_xgb), method='isotonic')
    calibrated_model.fit(X_val_trans, y_val)

    # 7. Evaluate Ensemble on Test Set
    print("\nEvaluating final ensemble on Test split...")
    cal_test_probs = calibrated_model.predict_proba(X_test_trans)[:, 1]
    
    # Isolation forest anomaly scores
    test_anomaly_raw = iforest.decision_function(X_test_trans)
    # Map decision scores to [0,1]
    score_min, score_max = -0.4, 0.2
    test_anomaly = 1.0 - (test_anomaly_raw - score_min) / (score_max - score_min + 1e-5)
    test_anomaly = np.clip(test_anomaly, 0.0, 1.0)
    
    # Ensemble Formula: 0.7 * xgb_prob + 0.3 * anomaly_score
    ensemble_scores = 0.7 * cal_test_probs + 0.3 * test_anomaly
    
    final_auc = roc_auc_score(y_test, ensemble_scores)
    final_pr_auc = average_precision_score(y_test, ensemble_scores)
    print(f"Final Ensemble Test metrics:")
    print(f"  - ROC-AUC: {final_auc:.4f}")
    print(f"  - PR-AUC (Average Precision): {final_pr_auc:.4f}")

    # Print classification report at ensemble score threshold 0.45 (HIGH/CRITICAL boundary)
    y_pred = (ensemble_scores >= 0.45).astype(int)
    print("\nClassification Report (Ensemble Threshold = 0.45):")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Fraud']))

    # Business Metrics analysis
    # At threshold 0.45, what % of fraud is caught (Recall)? What is false alarm rate?
    fraud_caught = sum((ensemble_scores >= 0.45) & (y_test == 1))
    total_fraud = sum(y_test == 1)
    recall = fraud_caught / total_fraud if total_fraud > 0 else 0
    
    false_positives = sum((ensemble_scores >= 0.45) & (y_test == 0))
    total_legit = sum(y_test == 0)
    fpr = false_positives / total_legit if total_legit > 0 else 0
    
    print("Business Metrics summary:")
    print(f"  - Fraud caught (Recall): {fraud_caught} of {total_fraud} cases ({recall*100:.2f}%)")
    print(f"  - False positive rate (FPR): {false_positives} of {total_legit} legitimate transactions ({fpr*100:.4f}%)")
    print(f"  - Alert Precision: {fraud_caught / (fraud_caught + false_positives)*100:.2f}% of alerts are true fraud.")

    # 8. Save Artifacts
    print("\nSaving trained artifacts to models/...")
    joblib.dump(calibrated_model, "models/xgb_model.joblib")
    joblib.dump(iforest, "models/iforest_model.joblib")
    joblib.dump(pipeline, "models/feature_pipeline.joblib")
    print("Saved xgb_model.joblib, iforest_model.joblib, and feature_pipeline.joblib.")
    
    # Save a small sample of test data to models/test_sample.csv for Streamlit streaming
    test_df_with_preds = test_df.copy()
    test_df_with_preds['ensemble_score'] = ensemble_scores
    test_df_with_preds['xgb_prob'] = cal_test_probs
    test_df_with_preds['anomaly_score'] = test_anomaly
    test_df_with_preds.to_csv("data/paysim_test_sample.csv", index=False)
    print("Saved validation test dataset with predictions to data/paysim_test_sample.csv for streaming simulation.")

if __name__ == "__main__":
    train_system()
