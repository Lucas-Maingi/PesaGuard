import os
import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# PesaGuard: Exploratory Data Analysis & Feature Engineering Pipeline\n\n"
        "This notebook contains the research phase for **PesaGuard**, a production-grade real-time fraud detection system. "
        "We will perform exploratory data analysis on the PaySim mobile money dataset, implement a custom scikit-learn "
        "feature engineering pipeline, and address severe class imbalance."
    ))
    
    # Imports Cell
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.base import BaseEstimator, TransformerMixin\n"
        "from sklearn.pipeline import Pipeline\n\n"
        "sns.set_theme(style='darkgrid')\n"
        "print('Imports complete!')"
    ))
    
    # Load Data Cell
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load Dataset\n\n"
        "We configure the notebook to run on either the local 100K sample or the full 6.3M PaySim dataset (e.g. on Google Colab)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Toggle this to True if running on Google Colab with the full dataset\n"
        "USE_FULL_DATASET = False\n"
        "data_path = 'data/paysim_sample.csv'\n\n"
        "if USE_FULL_DATASET:\n"
        "    data_path = 'PS_20174392719_1491204439457_log.csv'\n\n"
        "if not os.path.exists(data_path):\n"
        "    print(f'Error: Data not found at {data_path}. Please run download_data.py first.')\n"
        "else:\n"
        "    df = pd.read_csv(data_path)\n"
        "    print(f'Loaded dataset from {data_path}')\n"
        "    print(f'Shape: {df.shape}')"
    ))
    
    # Basic EDA Cell
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n\n"
        "Let's visualize the class distribution, transaction type patterns, amounts, and hourly fraud rates."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Class distribution\n"
        "fraud_counts = df['isFraud'].value_counts()\n"
        "fraud_pct = df['isFraud'].value_counts(normalize=True) * 100\n"
        "print('Class Imbalance:')\n"
        "print(f'Legitimate: {fraud_counts[0]:,} ({fraud_pct[0]:.4f}%)')\n"
        "print(f'Fraudulent: {fraud_counts[1]:,} ({fraud_pct[1]:.4f}%)')\n\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "sns.countplot(x='isFraud', data=df, ax=ax)\n"
        "ax.set_title('Transaction Class Counts (0 = Legit, 1 = Fraud)')\n"
        "plt.yscale('log') # Log scale to visualize the tiny fraud class\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Fraud by Transaction Type\n"
        "fraud_by_type = df.groupby('type')['isFraud'].agg(['count', 'sum'])\n"
        "fraud_by_type['rate (%)'] = (fraud_by_type['sum'] / fraud_by_type['count']) * 100\n"
        "print('Fraud rates by transaction type:')\n"
        "print(fraud_by_type)\n\n"
        "fig, ax = plt.subplots(figsize=(8, 4))\n"
        "sns.barplot(x=fraud_by_type.index, y='rate (%)', data=fraud_by_type, ax=ax)\n"
        "ax.set_title('Fraud Rate (%) by Transaction Type')\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Amount distribution by class\n"
        "fig, ax = plt.subplots(figsize=(10, 5))\n"
        "sns.boxplot(x='isFraud', y='amount', data=df, ax=ax)\n"
        "ax.set_title('Transaction Amounts by Class')\n"
        "plt.yscale('log')\n"
        "plt.show()"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Hourly fraud patterns\n"
        "df['hour'] = df['step'] % 24\n"
        "hourly_stats = df.groupby('hour')['isFraud'].agg(['count', 'sum'])\n"
        "hourly_stats['rate (%)'] = (hourly_stats['sum'] / hourly_stats['count']) * 100\n\n"
        "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)\n"
        "sns.lineplot(x=hourly_stats.index, y='count', data=hourly_stats, ax=ax1, marker='o', color='blue')\n"
        "ax1.set_ylabel('Total Volume')\n"
        "ax1.set_title('Transaction Volume vs. Fraud Rate by Hour of Day')\n\n"
        "sns.lineplot(x=hourly_stats.index, y='rate (%)', data=hourly_stats, ax=ax2, marker='s', color='red')\n"
        "ax2.set_ylabel('Fraud Rate (%)')\n"
        "ax2.set_xlabel('Hour of Day (step % 24)')\n\n"
        "plt.xticks(range(24))\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Feature Engineering Pipeline Cell
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Custom Feature Engineering Pipeline\n\n"
        "To build a production-grade system, we implement our feature engineering inside a scikit-learn custom transformer. "
        "This ensures that we can deploy the pipeline directly to production to score live single transactions. "
        "We support both **Offline Batch Mode** (running aggregations on the historical dataframe) and **Online Real-Time Mode** "
        "(enriching a single incoming transaction using the customer's state from the database)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "class PesaGuardFeaturePipeline(BaseEstimator, TransformerMixin):\n"
        "    def __init__(self):\n"
        "        # For offline training, we will build a history dictionary for each account\n"
        "        self.user_history_ = {}\n"
        "        self.merchant_fraud_counts_ = {}\n"
        "        self.merchant_total_counts_ = {}\n"
        "        self.merchant_risk_scores_ = {}\n\n"
        "    def fit(self, X, y=None):\n"
        "        # Fit learns history from the training set to prevent leakage\n"
        "        df_fit = X.copy()\n"
        "        if y is not None:\n"
        "            df_fit['isFraud'] = y\n"
        "        else:\n"
        "            df_fit['isFraud'] = 0\n\n"
        "        # Learn merchant risk scores\n"
        "        # Destination accounts starting with 'M' are merchants\n"
        "        df_fit['is_merchant'] = df_fit['nameDest'].str.startswith('M')\n"
        "        df_fit['dest_prefix'] = df_fit['nameDest'].str[0]\n\n"
        "        # Compute fraud counts per destination category\n"
        "        merchant_stats = df_fit.groupby('dest_prefix')['isFraud'].agg(['count', 'sum'])\n"
        "        for prefix, row in merchant_stats.iterrows():\n"
        "            # Frequency-based risk score\n"
        "            self.merchant_risk_scores_[prefix] = row['sum'] / row['count'] if row['count'] > 0 else 0.0\n\n"
        "        # Pre-compute historical statistics for users present in fit data\n"
        "        # (For offline testing, we sort by step to emulate sequential arrival)\n"
        "        df_sorted = df_fit.sort_values(by='step')\n"
        "        for idx, row in df_sorted.iterrows():\n"
        "            user = row['nameOrig']\n"
        "            amt = row['amount']\n"
        "            step = row['step']\n"
        "            \n"
        "            if user not in self.user_history_:\n"
        "                self.user_history_[user] = {'amounts': [], 'steps': []}\n"
        "            \n"
        "            self.user_history_[user]['amounts'].append(amt)\n"
        "            self.user_history_[user]['steps'].append(step)\n"
        "            \n"
        "        return self\n\n"
        "    def transform(self, X):\n"
        "        # Transforms data. In production, this can score batched or single rows.\n"
        "        df_trans = X.copy().sort_values(by='step')\n"
        "        \n"
        "        engineered_rows = []\n"
        "        \n"
        "        # Temporary history to update sequentially during transformation\n"
        "        temp_history = {u: {'amounts': list(h['amounts']), 'steps': list(h['steps'])} \n"
        "                        for u, h in self.user_history_.items()}\n"
        "        \n"
        "        for idx, row in df_trans.iterrows():\n"
        "            user = row['nameOrig']\n"
        "            dest = row['nameDest']\n"
        "            amt = row['amount']\n"
        "            step = row['step']\n"
        "            balance_orig = row['oldbalanceOrg']\n"
        "            \n"
        "            # 1. Basic features\n"
        "            hour_of_day = step % 24\n"
        "            is_round_amount = 1 if (amt % 100 == 0 or amt % 1000 == 0) and amt > 0 else 0\n"
        "            cross_border = 1 if row['nameDest'].startswith('M') else 0 # simple logic for merchant/external transfer\n"
        "            amount_to_balance_ratio = amt / (balance_orig + 1e-5) # add small epsilon to prevent div by zero\n"
        "            \n"
        "            # Get destination merchant risk score\n"
        "            dest_prefix = dest[0] if len(dest) > 0 else 'C'\n"
        "            merchant_risk_score = self.merchant_risk_scores_.get(dest_prefix, 0.0)\n\n"
        "            # 2. User history features\n"
        "            user_history = temp_history.get(user, {'amounts': [], 'steps': []})\n"
        "            user_amts = user_history['amounts']\n"
        "            user_steps = user_history['steps']\n"
        "            \n"
        "            if len(user_amts) > 0:\n"
        "                # 30-day average (720 steps)\n"
        "                recent_amts = [a for a, s in zip(user_amts, user_steps) if step - s <= 720]\n"
        "                recent_steps = [s for s in user_steps if step - s <= 720]\n"
        "                \n"
        "                # Average amount\n"
        "                avg_amt = np.mean(recent_amts) if len(recent_amts) > 0 else amt\n"
        "                std_amt = np.std(recent_amts) if len(recent_amts) > 1 else 0.0\n"
        "                amount_deviation = (amt - avg_amt) / (std_amt + 1e-5) # Z-score\n"
        "                \n"
        "                # Velocity\n"
        "                velocity_1h = sum(1 for s in recent_steps if step - s <= 1)\n"
        "                velocity_24h = sum(1 for s in recent_steps if step - s <= 24)\n"
        "                \n"
        "                # Time since last transaction (step is in hours)\n"
        "                time_since_last_transaction = step - user_steps[-1]\n"
        "                \n"
        "                # Amount percentile in history\n"
        "                amount_percentile = np.percentile(recent_amts, 50) # simple median reference as fallback\n"
        "                if len(recent_amts) > 2:\n"
        "                    amount_percentile = sum(1 for a in recent_amts if amt >= a) / len(recent_amts)\n"
        "                else:\n"
        "                    amount_percentile = 0.5\n"
        "            else:\n"
        "                # First transaction for this user\n"
        "                amount_deviation = 0.0\n"
        "                velocity_1h = 1\n"
        "                velocity_24h = 1\n"
        "                time_since_last_transaction = 999.0 # flag for first transaction\n"
        "                amount_percentile = 0.5\n"
        "                \n"
        "            # Record engineered feature row\n"
        "            feat_row = {\n"
        "                'amount_deviation': amount_deviation,\n"
        "                'velocity_1h': velocity_1h,\n"
        "                'velocity_24h': velocity_24h,\n"
        "                'amount_to_balance_ratio': amount_to_balance_ratio,\n"
        "                'hour_of_day': hour_of_day,\n"
        "                'is_round_amount': is_round_amount,\n"
        "                'merchant_risk_score': merchant_risk_score,\n"
        "                'cross_border': cross_border,\n"
        "                'time_since_last_transaction': time_since_last_transaction,\n"
        "                'amount_percentile': amount_percentile,\n"
        "                'amount': amt,\n"
        "                'oldbalanceOrg': balance_orig,\n"
        "                'newbalanceOrig': row['newbalanceOrig'],\n"
        "                'oldbalanceDest': row['oldbalanceDest'],\n"
        "                'newbalanceDest': row['newbalanceDest'],\n"
        "                'is_transfer': 1 if row['type'] == 'TRANSFER' else 0,\n"
        "                'is_cash_out': 1 if row['type'] == 'CASH_OUT' else 0\n"
        "            }\n"
        "            engineered_rows.append(feat_row)\n"
        "            \n"
        "            # Update history dynamically\n"
        "            if user not in temp_history:\n"
        "                temp_history[user] = {'amounts': [], 'steps': []}\n"
        "            temp_history[user]['amounts'].append(amt)\n"
        "            temp_history[user]['steps'].append(step)\n"
        "            \n"
        "        return pd.DataFrame(engineered_rows)\n\n"
        "pipeline = PesaGuardFeaturePipeline()\n"
        "print('Feature engineering class compiled!')"
    ))
    
    # Train-test split
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Time-Based Splitting & Fitting\n\n"
        "Fraud patterns evolve. Random splits cause data leakage (using future data to predict past fraud). "
        "We use a chronological split (split by `step`): the first 80% of steps for training, and the final 20% for validation."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Chronological split\n"
        "split_step = int(df['step'].max() * 0.8)\n"
        "train_df = df[df['step'] <= split_step]\n"
        "test_df = df[df['step'] > split_step]\n\n"
        "X_train, y_train = train_df.drop(columns=['isFraud']), train_df['isFraud']\n"
        "X_test, y_test = test_df.drop(columns=['isFraud']), test_df['isFraud']\n\n"
        "print(f'Training steps: <= {split_step} (Size: {len(X_train)} rows, Fraud cases: {y_train.sum()})')\n"
        "print(f'Testing steps: > {split_step} (Size: {len(X_test)} rows, Fraud cases: {y_test.sum()})')\n\n"
        "# Fit and transform features\n"
        "pipeline.fit(X_train, y_train)\n"
        "X_train_trans = pipeline.transform(X_train)\n"
        "X_test_trans = pipeline.transform(X_test)\n\n"
        "print('Features engineered successfully!')\n"
        "print('Transformed training features columns:')\n"
        "print(X_train_trans.columns.tolist())\n"
        "print(X_train_trans.head(3))"
    ))
    
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Class Imbalance: SMOTE vs. XGBoost Class Weights\n\n"
        "Because our fraud rate is only 0.11%, standard classifiers will simply guess 'legitimate' for all rows. "
        "In the next notebook, we will test:\n"
        "1. **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic fraud samples during training.\n"
        "2. **scale_pos_weight** in XGBoost to penalize errors on fraud cases proportional to the imbalance.\n"
        "We will save our engineered training sets for the model notebook."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Save processed features for the modeling notebook\n"
        "if not os.path.exists('data/processed'):\n"
        "    os.makedirs('data/processed')\n\n"
        "X_train_trans.to_csv('data/processed/X_train_feats.csv', index=False)\n"
        "X_test_trans.to_csv('data/processed/X_test_feats.csv', index=False)\n"
        "y_train.to_csv('data/processed/y_train.csv', index=False)\n"
        "y_test.to_csv('data/processed/y_test.csv', index=False)\n"
        "print('Processed datasets saved for Modeling phase!')"
    ))
    
    nb.cells = cells
    
    # Save notebook
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/eda_and_feature_engineering.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/eda_and_feature_engineering.ipynb")

def create_model_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Title Cell
    cells.append(nbf.v4.new_markdown_cell(
        "# PesaGuard: Model Training, Evaluation & Calibration\n\n"
        "This notebook handles Phase 2 of **PesaGuard**. We train:\n"
        "1. **Isolation Forest** (unsupervised anomaly detector)\n"
        "2. **XGBoost Classifier** (supervised model, optimized for class imbalance)\n"
        "3. **Ensemble Scoring & Calibration** (weighted combination of both models)\n\n"
        "We evaluate using Precision-Recall curves, compute SHAP explainability, and calibrate probabilities."
    ))
    
    # Imports Cell
    cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import joblib\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from xgboost import XGBClassifier\n"
        "from sklearn.ensemble import IsolationForest\n"
        "from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, classification_report\n"
        "from sklearn.calibration import CalibratedClassifierCV\n"
        "import shap\n\n"
        "print('Imports complete!')"
    ))
    
    # Load Processed Data Cell
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load Processed Data\n\n"
        "We load the engineered features saved in the previous notebook."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "X_train = pd.read_csv('data/processed/X_train_feats.csv')\n"
        "X_test = pd.read_csv('data/processed/X_test_feats.csv')\n"
        "y_train = pd.read_csv('data/processed/y_train.csv').iloc[:, 0]\n"
        "y_test = pd.read_csv('data/processed/y_test.csv').iloc[:, 0]\n\n"
        "print(f'X_train shape: {X_train.shape}')\n"
        "print(f'X_test shape: {X_test.shape}')"
    ))
    
    # Train Unsupervised Isolation Forest Cell
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Train Isolation Forest Anomaly Detector\n\n"
        "Isolation Forest identifies outliers. It outputs an anomaly score (higher score = more anomalous)."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Isolation Forest only requires feature matrix (unsupervised)\n"
        "iforest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)\n"
        "iforest.fit(X_train)\n\n"
        "# Get anomaly scores (Isolation Forest outputs decision_function where more negative = more anomalous)\n"
        "# We invert and scale it to [0, 1] range where 1 is highly anomalous\n"
        "def scale_anomaly_scores(model, X):\n"
        "    scores = model.decision_function(X)\n"
        "    # Map scores so that higher value = more anomalous\n"
        "    # Raw decision function outputs range roughly from -0.5 to +0.5\n"
        "    scaled = 1.0 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-5)\n"
        "    return scaled\n\n"
        "train_anomaly = scale_anomaly_scores(iforest, X_train)\n"
        "test_anomaly = scale_anomaly_scores(iforest, X_test)\n\n"
        "print(f'Isolation Forest trained! Sample test anomaly scores: {test_anomaly[:5]}')"
    ))
    
    # Train XGBoost Classifier
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Train XGBoost Supervised Classifier\n\n"
        "We handle class imbalance using XGBoost's `scale_pos_weight` parameter which sets the weight of "
        "positive class (fraud) proportional to the negative class ratio."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Calculate class ratio for scale_pos_weight\n"
        "neg_count = sum(y_train == 0)\n"
        "pos_count = sum(y_train == 1)\n"
        "scale_weight = neg_count / (pos_count + 1e-5)\n"
        "print(f'Imbalance scale weight: {scale_weight:.2f}')\n\n"
        "# Train XGBClassifier\n"
        "xgb = XGBClassifier(\n"
        "    n_estimators=150,\n"
        "    max_depth=5,\n"
        "    learning_rate=0.05,\n"
        "    scale_pos_weight=scale_weight,\n"
        "    random_state=42,\n"
        "    n_jobs=-1\n"
        ")\n"
        "xgb.fit(X_train, y_train)\n\n"
        "test_probs = xgb.predict_proba(X_test)[:, 1]\n"
        "print('XGBoost trained!')"
    ))
    
    # Ensemble Model
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Ensemble Scoring: PesaGuard Score\n\n"
        "We combine supervised and unsupervised models:\n"
        "$$\\text{Ensemble Score} = 0.7 \\times \\text{XGB Prob} + 0.3 \\times \\text{Anomaly Score}$$\n"
        "Let's write this function and evaluate the performance of XGBoost, Isolation Forest, and the Ensemble."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "def compute_ensemble_score(xgb_prob, anomaly_score):\n"
        "    return 0.7 * xgb_prob + 0.3 * anomaly_score\n\n"
        "ensemble_probs = compute_ensemble_score(test_probs, test_anomaly)\n\n"
        "# Metrics comparisons\n"
        "for name, y_scores in [('XGBoost Only', test_probs), ('Anomaly Only', test_anomaly), ('Ensemble', ensemble_probs)]:\n"
        "    auc = roc_auc_score(y_test, y_scores)\n"
        "    pr_auc = average_precision_score(y_test, y_scores)\n"
        "    print(f'{name:<15} | ROC-AUC: {auc:.4f} | PR-AUC (Average Precision): {pr_auc:.4f}')"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Plot Precision-Recall Trade-off Curve\n"
        "precision, recall, thresholds = precision_recall_curve(y_test, ensemble_probs)\n\n"
        "plt.figure(figsize=(8, 5))\n"
        "plt.plot(recall, precision, label=f'Ensemble (PR-AUC = {average_precision_score(y_test, ensemble_probs):.4f})', color='green')\n"
        "plt.xlabel('Recall (Detection Rate)')\n"
        "plt.ylabel('Precision (True Fraud Rate in Flags)')\n"
        "plt.title('Precision-Recall Curve (Ensemble Model)')\n"
        "plt.legend(loc='best')\n"
        "plt.show()"
    ))
    
    # Probability Calibration Cell
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Probability Calibration\n\n"
        "XGBoost trained with `scale_pos_weight` shifts output probabilities upwards (exaggerating true fraud probability). "
        "We use Isotonic Regression (CalibratedClassifierCV) to calibrate the probabilities so the outputs represent actual risk percentages."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Calibrate XGBoost\n"
        "calibrated_xgb = CalibratedClassifierCV(estimator=xgb, method='isotonic', cv='prefit')\n"
        "calibrated_xgb.fit(X_test, y_test) # prefitted calibrator on hold-out set\n\n"
        "cal_test_probs = calibrated_xgb.predict_proba(X_test)[:, 1]\n"
        "cal_ensemble = compute_ensemble_score(cal_test_probs, test_anomaly)\n\n"
        "print(f'Original mean prediction prob: {test_probs.mean():.4f}')\n"
        "print(f'Calibrated mean prediction prob: {cal_test_probs.mean():.4f}')\n"
        "print(f'Actual fraud rate in test set: {y_test.mean():.4f}')"
    ))
    
    # SHAP Explainer
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Model Explainability using SHAP\n\n"
        "Explain predictions to analysts via SHAP waterfall charts. We will construct a TreeExplainer for XGBoost."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Shap analysis\n"
        "explainer = shap.TreeExplainer(xgb)\n"
        "# We take a sample of test rows to speed up explanation\n"
        "sample_X = X_test.head(100)\n"
        "shap_values = explainer(sample_X)\n\n"
        "# Summary plot\n"
        "plt.figure(figsize=(10, 6))\n"
        "shap.summary_plot(shap_values, sample_X, show=False)\n"
        "plt.title('SHAP Feature Importance Summary for XGBoost')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # Save Artifacts
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Save Model Weights & Pipeline Artifacts\n\n"
        "We save our trained model objects to the `models/` directory for deployment in our FastAPI microservice."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "if not os.path.exists('models'):\n"
        "    os.makedirs('models')\n\n"
        "joblib.dump(iforest, 'models/iforest_model.joblib')\n"
        "joblib.dump(calibrated_xgb, 'models/xgb_model.joblib')\n"
        "# Note: We will export the feature pipeline object we learn in the previous notebook as well\n"
        "print('Trained models saved successfully!')"
    ))
    
    nb.cells = cells
    
    # Save notebook
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/model_training.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Created notebooks/model_training.ipynb")

if __name__ == "__main__":
    create_eda_notebook()
    create_model_notebook()
