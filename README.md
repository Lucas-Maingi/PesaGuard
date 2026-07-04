# PesaGuard: Real-Time Mobile Money Fraud Detection System

[![CI](https://github.com/Lucas-Maingi/PesaGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/Lucas-Maingi/PesaGuard/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Docker](https://img.shields.io/badge/docker-compose%20ready-2496ED)

PesaGuard is an enterprise-grade, low-latency Real-Time Fraud Detection System designed for international fintechs, African banks, and mobile money operators (such as M-Pesa, MTN MoMo, and Airtel Money). 

The platform leverages a hybrid machine learning approach, combining a **supervised XGBoost classifier** (trained on SMOTE-oversampled transactions) with an **unsupervised Isolation Forest anomaly detector**. It scores incoming transactions in **under 100ms** and streams them to a visual operations console for human analysts.

---

## 🎥 Live Demo

**▶️ Try it live:** https://huggingface.co/spaces/lucas-maingi/pesaguard

The live demo runs the full stack — the FastAPI scoring API and the Streamlit analyst console — in a single always-on container.

![PesaGuard analyst console](docs/demo.gif)

> _Recording the GIF? See [`docs/HOW_TO_RECORD_DEMO.md`](docs/HOW_TO_RECORD_DEMO.md)._

---

## 🏗️ System Architecture

PesaGuard is designed for strict separation of concerns, featuring a stateless FastAPI backend and a dynamic database layer that supports SQLite for local development and Supabase (PostgreSQL) for production.

```
                   +----------------------------------+
                   |    Streamlit Analyst Console     |
                   |      (Real-time Dashboard)       |
                   +----------------+-----------------+
                                    |
                                    | HTTP REST (POST /score)
                                    v
                   +----------------------------------+
                   |           FastAPI API            |
                   |      (Microservice Gateway)      |
                   +-------+------------------+-------+
                           |                  |
    1. Fetch History       |                  | 3. Log Scored Tx
    2. Write Feedback      v                  v
                   +---------------+  +---------------+
                   |  SQLite Local |  |  Supabase DB  |
                   | (pesaguard.db)|  | (PostgreSQL)  |
                   +---------------+  +---------------+
```

### Key Components:
1.  **Feature Pipeline (`src/features.py`)**: Custom scikit-learn pipeline implementing both **Offline Batch Mode** (training data) and **Online Real-Time Mode** (enriching a single transaction using the sender's recent transaction history from the database).
2.  **Model Ensemble (`src/models.py`)**: Computes the threat score using the formula:
    $$\text{PesaGuard Score} = 0.7 \times \text{Calibrated XGBoost Prob} + 0.3 \times \text{Isolation Forest Anomaly}$$
3.  **FastAPI Backend (`src/api.py`)**: Serves scoring and feedback endpoints, keeping internal execution latency $< 100\text{ms}$.
4.  **Ops Dashboard (`src/dashboard.py`)**: Streamlit interface containing a live transaction feed, an alert queue, interactive SHAP risk-contributions, and a geographic transaction risk map centered in East and West Africa.

---

## 📊 Model Performance & Business Metrics

Models are trained and evaluated on a **strict chronological split** (64% train / 16% validation / 20% test — no shuffling, so no future information leaks into training) of the [PaySim synthetic mobile money dataset](https://www.kaggle.com/datasets/ealaxi/paysim1), where fraud represents just **0.116%** of transactions.

> **⚠️ A note on honest benchmarking:** PaySim is a *simulator*. Its fraud patterns (balance-draining `TRANSFER` → `CASH_OUT` chains) are far more separable than real-world fraud, which is why tree ensembles reach near-perfect ROC-AUC on it — a property of the dataset, not proof of production-grade performance. For that reason we treat **PR-AUC (Average Precision)** as the primary metric for this heavily imbalanced problem, and all numbers below as an *upper bound*. On real transaction data, expect materially lower scores plus concept drift — which is exactly why the system ships with an analyst feedback loop (`POST /feedback`) to support monitored retraining.

### Class Imbalance Strategy Comparison (PaySim test split):
*   **Weighted XGBoost (`scale_pos_weight`)**: PR-AUC (Average Precision): `0.8409` | ROC-AUC: `0.9999`
*   **SMOTE Resampled XGBoost**: PR-AUC (Average Precision): **`0.8875`** | ROC-AUC: `1.0000`
*   *Conclusion*: **SMOTE resampled XGBoost** was selected due to its superior Average Precision, meaning fewer false alarms for operations teams. (`train.py` re-runs this comparison on every training run and keeps whichever strategy wins.)

### Final Ensemble Performance (PaySim test split):
*   **PR-AUC (Average Precision)**: **`0.9500`**
*   **ROC-AUC**: `1.0000` *(see benchmarking note above — near-perfect separability is expected on PaySim)*

### Business Impact Translation (Threshold = 0.45, PaySim test split):
At the operational threshold of `0.45`, the system achieves on the held-out test window:

| Metric | Performance | Business Meaning |
| :--- | :--- | :--- |
| **Recall (Detection Rate)** | **100.0%** | Catches **10 out of 10** fraudulent transactions in the test window, stopping theft before cash-out. |
| **False Positive Rate (FPR)** | **0.0050%** | Only **1 in 20,000** legitimate transactions is wrongly flagged. |
| **Alert Precision** | **80.00%** | **4 out of 5 alerts** triggered in the console are true fraud, minimizing analyst review fatigue. |

#### 🌍 Illustrative Scenario: Why FPR Is the Metric That Matters at Mobile-Money Scale
> *The following is a back-of-envelope illustration using Safaricom's publicly reported scale figures. PesaGuard is a portfolio project and has **not** been deployed at Safaricom or any operator — the point is to show how the FPR/recall trade-off translates into operational load.*

Safaricom's M-Pesa reports over **40 million active subscribers** and on the order of **40 million transactions per day**.
*   A fraud system with a $1\%$ False Positive Rate at that volume would trigger **~400,000 false alerts daily** — enough to overwhelm any operations team and lock out legitimate users.
*   A system operating at PesaGuard's PaySim-measured FPR of $0.0050\%$ would generate **~2,000 false alerts daily** at the same volume. That two-orders-of-magnitude difference is the difference between an alert queue analysts can actually work and one they ignore.

---

## 🛡️ Fraud Signatures Identified in EDA (PaySim)

Exploratory analysis of the dataset surfaced two signatures that mirror documented real-world mobile-money fraud patterns:
1.  **High-Risk Transaction Types**: In PaySim, **100% of fraud occurs in `TRANSFER` and `CASH_OUT` transactions** — modeling the classic pattern where a compromised account `TRANSFER`s its balance to a mule account, followed by an immediate agent `CASH_OUT`. PesaGuard exploits this with fast-path handling for payment and cash-in types to optimize throughput.
2.  **Low-Traffic Hour Exploitation**: While transaction volume peaks during the day, the *fraud rate* peaks between **midnight and 6:00 AM** — consistent with scammers striking while victims sleep through SMS notifications. This makes transaction time-of-day a critical feature.

---

## ⚠️ Known Limitations & Road to Production

Being honest about the gap between a portfolio system and a production deployment:

*   **Synthetic data ceiling**: All metrics come from PaySim. Real mobile-money fraud is adversarial and drifts over time; production deployment would require re-benchmarking on operator data behind their compliance walls, and the numbers *will* be lower.
*   **Latency measured locally**: The <100ms scoring path is measured on a single machine without concurrent load. Production readiness would require load testing (e.g., Locust) and horizontal scaling of the stateless API tier.
*   **Feedback loop is logged, not automated**: Analyst feedback is captured via `POST /feedback`, but retraining is a manual `train.py` run — no automated retraining pipeline or model registry promotion yet.
*   **History features depend on the serving database**: Velocity/deviation features are computed from transactions the API itself has logged. A production system would source these from the operator's core transaction log or a feature store.

## 🚀 Running the System Locally

### Prerequisites:
- Python 3.10+
- Installed packages: `pip install -r requirements.txt`

### 1. Initialize the Database & Train the Models:
This will create `data/pesaguard.db`, compile the feature pipeline, train the ensemble, calibrate probabilities, and save the binaries.
```bash
python train.py
```

### 2. Launch the FastAPI Scoring Service:
Runs on port 8000.
```bash
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

### 3. Launch the Streamlit Monitoring Dashboard:
Opens the visual UI in your web browser.
```bash
streamlit run src/dashboard.py
```

### 🐳 Alternative: Run the Full Stack with Docker Compose
Builds a lean scoring-API image (from `requirements-api.txt`) and a dashboard image, wired together on an internal network:
```bash
docker compose up --build
```
*   API: `http://localhost:8000` (interactive docs at `/docs`)
*   Dashboard: `http://localhost:8501`

Set `SUPABASE_DB_URL` in your environment (or a `.env` file) to switch the API from container-local SQLite to PostgreSQL.

---

## 🔌 API Documentation & Example Calls

### Health Check:
`GET /health`
```bash
curl -X GET http://127.0.0.1:8000/health
```

### Score a Transaction:
`POST /score`
```bash
curl -X POST http://127.0.0.1:8000/score \
-H "Content-Type: application/json" \
-d '{
  "step": 12,
  "type": "TRANSFER",
  "amount": 250000.0,
  "nameOrig": "C1928374",
  "oldbalanceOrg": 250000.0,
  "newbalanceOrig": 0.0,
  "nameDest": "C9988776",
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}'
```

**Response Example (under 100ms)**:
```json
{
  "transaction_id": "TX_C1928374_12_1782765318",
  "fraud_probability": 1.0,
  "anomaly_score": 0.0816,
  "ensemble_score": 0.7245,
  "risk_tier": "HIGH",
  "top_signals": [
    {
      "signal": "amount_to_balance_ratio",
      "description": "Amount-to-balance ratio is 100.0% of current balance.",
      "impact": "high",
      "shap_value": 6.2873
    },
    {
      "signal": "is_transfer",
      "description": "Transaction type is TRANSFER (high fraud prevalence).",
      "impact": "high",
      "shap_value": 1.9357
    }
  ],
  "recommendation": "FLAG_FOR_REVIEW",
  "response_time_ms": 91
}
```

### Log Analyst Feedback:
`POST /feedback`
```bash
curl -X POST http://127.0.0.1:8000/feedback \
-H "Content-Type: application/json" \
-d '{
  "transaction_id": "TX_C1928374_12_1782765318",
  "is_fraud_feedback": 1,
  "feedback_notes": "Customer confirmed unauthorized transfer, SIM swap suspected."
}'
```
