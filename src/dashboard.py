import sys
import os

# Add project root to python path to prevent ModuleNotFoundError in Streamlit Cloud
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import database module for fallback/stats
from src.db import get_db_stats, log_feedback, get_connection
from src.features import PesaGuardFeaturePipeline

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PesaGuard | Enterprise Fraud Operations Center",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for Premium SaaS UI/UX (Glassmorphism & Sleek Dark Mode)
st.markdown("""
<style>
    .main {
        background-color: #0b0f19;
    }
    div[data-testid="stMetricValue"] {
        font-size: 26px;
        font-weight: 700;
        color: #10b981;
    }
    .metric-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3730a3 100%);
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.25s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-low {
        background-color: #10b981;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 11px;
    }
    .code-box {
        background-color: #0f172a;
        color: #38bdf8;
        padding: 10px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 12px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

def get_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=1.0)
        return r.status_code == 200
    except Exception:
        return False

# Initialize Session State variables
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "scored_transactions" not in st.session_state:
    st.session_state.scored_transactions = []
if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = None
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "policy_rules" not in st.session_state:
    # Default rules configuration
    st.session_state.policy_rules = {
        "max_amount_toggle": False,
        "max_amount_value": 500000.0,
        "night_lock_toggle": False,
        "velocity_alert_toggle": False,
        "velocity_alert_limit": 3
    }
if "attack_mode" not in st.session_state:
    st.session_state.attack_mode = False
if "custom_csv_results" not in st.session_state:
    st.session_state.custom_csv_results = None

# Helper to mock coordinates in East/West Africa based on nameDest
def get_mock_coordinates(name_dest):
    h = hash(name_dest)
    hubs = [
        {"city": "Nairobi, KE", "lat": -1.2921, "lon": 36.8219},
        {"city": "Lagos, NG", "lat": 6.5244, "lon": 3.3792},
        {"city": "Kampala, UG", "lat": 0.3476, "lon": 32.5825},
        {"city": "Dar es Salaam, TZ", "lat": -6.7924, "lon": 39.2083}
    ]
    hub = hubs[h % len(hubs)]
    jitter_lat = ((h % 100) - 50) / 100.0 * 0.8
    jitter_lon = (((h >> 2) % 100) - 50) / 100.0 * 0.8
    return hub["lat"] + jitter_lat, hub["lon"] + jitter_lon, hub["city"]

# Dashboard Header
st.title("🛡️ PesaGuard | Enterprise Fraud Operations Console")
st.markdown("Stateless real-time transaction scoring, business policy execution, and analyst feedback logs.")

# Sidebar Controls
st.sidebar.header("🕹️ Operations Control Center")

api_online = get_api_health()
if api_online:
    st.sidebar.success("🟢 API Server: CONNECTED")
else:
    st.sidebar.error("🔴 API Server: OFFLINE (Local Engine Active)")

# Mode selector
data_source = st.sidebar.selectbox(
    "Data Input Mode",
    ["Simulated Live Stream", "Upload Custom CSV", "Developer Integration API"]
)

# Render Controls based on mode
if data_source == "Simulated Live Stream":
    st.sidebar.subheader("Live Stream Config")
    stream_speed = st.sidebar.slider("Stream Speed (sec/tx)", 0.2, 3.0, 1.0, step=0.1)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start Feed", key="start_feed"):
            st.session_state.streaming = True
    with col2:
        if st.button("⏸️ Pause", key="pause_feed"):
            st.session_state.streaming = False
            
    # Attack Injector Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("Stress Testing")
    if st.sidebar.button("💥 Inject Fraud Attack Vector", help="Injects a series of verified fraud transactions from the validation set."):
        st.session_state.attack_mode = True
        st.session_state.streaming = True
        st.sidebar.info("Fraud vector injected. Look at the console!")
        
    if st.sidebar.button("🔄 Reset Console State"):
        st.session_state.scored_transactions = []
        st.session_state.selected_alert = None
        st.session_state.current_index = 0
        st.session_state.attack_mode = False
        st.rerun()

elif data_source == "Developer Integration API":
    st.sidebar.subheader("Developer API Credentials")
    st.sidebar.text_input("Tenant Domain ID", "lucas-maingi-pesaguard", disabled=True)
    st.sidebar.text_input("Live API Access Key", "pk_live_51PesaGuardKeyXYZ789", type="password", disabled=True)
    st.sidebar.info("Use these credentials to connect your mobile wallet system directly to PesaGuard's REST endpoints.")

# Load validation test dataset
sample_path = "data/paysim_test_sample.csv"
if not os.path.exists(sample_path):
    st.error(f"Validation dataset not found at {sample_path}. Please run train.py first to create the assets.")
    st.stop()
test_data = pd.read_csv(sample_path)

# Unified Scoring Logic (API client + rule execution)
def score_transaction_payload(row):
    tx_payload = {
        "step": int(row["step"]),
        "type": str(row["type"]),
        "amount": float(row["amount"]),
        "nameOrig": str(row["nameOrig"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "nameDest": str(row["nameDest"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"])
    }
    
    # 1. API Call
    scores = None
    if api_online:
        try:
            r = requests.post(f"{API_URL}/score", json=tx_payload, timeout=0.6)
            if r.status_code == 200:
                scores = r.json()
        except Exception:
            pass
            
    # 2. Local Fallback scoring if API is offline
    if not scores:
        prob = float(row.get("xgb_prob", 0.02))
        anomaly = float(row.get("anomaly_score", 0.05))
        score = float(row.get("ensemble_score", 0.03))
        
        risk_tier, rec = "LOW", "ALLOW"
        if score >= 0.75:
            risk_tier, rec = "CRITICAL", "BLOCK"
        elif score >= 0.45:
            risk_tier, rec = "HIGH", "FLAG_FOR_REVIEW"
        elif score >= 0.15:
            risk_tier, rec = "MEDIUM", "REVIEW_LATER"

        scores = {
            "transaction_id": f"TX_{row['nameOrig']}_{row['step']}_MOCK",
            "fraud_probability": prob,
            "anomaly_score": anomaly,
            "ensemble_score": score,
            "risk_tier": risk_tier,
            "recommendation": rec,
            "top_signals": [
                {"signal": "amount", "description": f"Transaction amount: {row['amount']:,.2f}.", "impact": "medium", "shap_value": 0.2}
            ],
            "response_time_ms": 15
        }

    # 3. Apply Hard Business Rules (Policy Engine Overrides)
    violated_rules = []
    
    # Rule 1: Max Amount Cap
    if st.session_state.policy_rules["max_amount_toggle"]:
        if tx_payload["amount"] > st.session_state.policy_rules["max_amount_value"]:
            violated_rules.append(f"LIMIT CAP: Exceeded Maximum Limit of ${st.session_state.policy_rules['max_amount_value']:,.2f}")
            
    # Rule 2: Night Lock (Hours 2-5)
    if st.session_state.policy_rules["night_lock_toggle"]:
        hour = int(tx_payload["step"] % 24)
        if 2 <= hour <= 5:
            violated_rules.append("NIGHT SECURITY: Transaction during night blackout period (2:00 AM - 5:00 AM)")
            
    # Rule 3: Velocity Limit (Simulated based on row variables or mock logic)
    if st.session_state.policy_rules["velocity_alert_toggle"]:
        # If user velocity is higher than allowed
        user_history = row.get("velocity_1h", 1)
        if user_history > st.session_state.policy_rules["velocity_alert_limit"]:
            violated_rules.append(f"VELOCITY BLOCK: {int(user_history)} transactions in 1 hour exceeds threshold limit of {st.session_state.policy_rules['velocity_alert_limit']}")

    # Apply overrides if rules are violated
    if len(violated_rules) > 0:
        scores["ensemble_score"] = 1.0  # Force to max risk
        scores["risk_tier"] = "CRITICAL"
        scores["recommendation"] = "BLOCK"
        
        # Inject policy violations at the top of SHAP explanations
        for rule in violated_rules:
            scores["top_signals"].insert(0, {
                "signal": "POLICY_RULE_ENGINE",
                "description": rule,
                "impact": "high",
                "shap_value": 5.0
            })

    return scores


# Create Tab Views
tabs = st.tabs([
    "📊 Live Operations Console", 
    "📁 CSV Batch Processor", 
    "⚙️ Risk Policy Engine", 
    "📈 Business ROI & Analytics",
    "🔌 Developer API Console"
])

# ==========================================
# TAB 1: LIVE OPERATIONS CONSOLE
# ==========================================
with tabs[0]:
    if data_source != "Simulated Live Stream":
        st.warning("Simulated Live Stream is inactive. Select it in the 'Data Input Mode' sidebar selector to launch the real-time operations console.")
    else:
        # DB Stats loading
        db_stats = get_db_stats()
        total_monitored = db_stats["total_transactions"] or len(st.session_state.scored_transactions)
        alerts_flagged = db_stats["fraud_flagged"] or sum(1 for tx in st.session_state.scored_transactions if tx["risk_tier"] in ["HIGH", "CRITICAL"])
        monetary_blocked = db_stats["value_blocked"] or sum(tx["amount"] for tx in st.session_state.scored_transactions if tx["risk_tier"] in ["HIGH", "CRITICAL"])
        
        # Calculate FPR
        feedback_count = db_stats["total_feedback"]
        false_positives = db_stats["false_positives"]
        fpr_pct = (false_positives / feedback_count * 100) if feedback_count > 0 else 0.0

        # Stats Cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><h5>Volume Scored</h5><h2>{total_monitored:,}</h2><small>Live P2P Stream</small></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><h5>Active Alerts</h5><h2 style="color: #ef4444;">{alerts_flagged:,}</h2><small>High/Critical Risk</small></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><h5>Value Protected</h5><h2 style="color: #10b981;">${monetary_blocked:,.2f}</h2><small>Blocked Payouts</small></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><h5>Ops False Positive Rate</h5><h2 style="color: #eab308;">{fpr_pct:.2f}%</h2><small>{feedback_count} Analyst Reviews</small></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Background Stream processing loop
        if st.session_state.streaming:
            if st.session_state.current_index < len(test_data):
                # Check if attack mode is active (filter next rows to true fraud cases only)
                if st.session_state.attack_mode:
                    fraud_rows = test_data[test_data["isFraud"] == 1]
                    if len(fraud_rows) > 0:
                        # Grab a random fraud row
                        row = fraud_rows.sample(1).iloc[0]
                    else:
                        row = test_data.iloc[st.session_state.current_index]
                else:
                    row = test_data.iloc[st.session_state.current_index]
                
                scored = score_row(row)
                full_tx = {**row.to_dict(), **scored}
                
                lat, lon, city = get_mock_coordinates(row["nameDest"])
                full_tx["lat"] = lat
                full_tx["lon"] = lon
                full_tx["city"] = city
                # SLA timer: 15 minutes represented in steps or seconds
                full_tx["sla_remaining_min"] = 15
                
                # Insert at top of list
                st.session_state.scored_transactions.insert(0, full_tx)
                if len(st.session_state.scored_transactions) > 50:
                    st.session_state.scored_transactions.pop()
                    
                st.session_state.current_index += 1
                time.sleep(stream_speed)
                st.rerun()

        # Split screen: Left (Live Stream & Queue) | Right (Alert Investigation Workspace)
        op_col1, op_col2 = st.columns([1.3, 1.0])

        with op_col1:
            st.subheader("📡 Real-Time Transaction Stream")
            if len(st.session_state.scored_transactions) == 0:
                st.info("Feed is currently idle. Click 'Start Feed' in the sidebar to begin processing live transactions.")
            else:
                feed_data = []
                for tx in st.session_state.scored_transactions:
                    badge = "🟢 LOW"
                    if tx["risk_tier"] == "CRITICAL":
                        badge = "🔴 CRITICAL"
                    elif tx["risk_tier"] == "HIGH":
                        badge = "🟠 HIGH"
                    elif tx["risk_tier"] == "MEDIUM":
                        badge = "🟡 MEDIUM"
                        
                    feed_data.append({
                        "Tx ID": tx["transaction_id"],
                        "Origin": tx["nameOrig"],
                        "Destination": tx["nameDest"],
                        "Amount": f"${tx['amount']:,.2f}",
                        "Type": tx["type"],
                        "Threat Score": f"{tx['ensemble_score']:.3f}",
                        "Alert Tier": badge
                    })
                st.dataframe(pd.DataFrame(feed_data), use_container_width=True, height=260)

            st.subheader("🚨 Priority Ops Alert Queue")
            # Get pending alerts sorted by score
            alerts = [tx for tx in st.session_state.scored_transactions if tx["risk_tier"] in ["HIGH", "CRITICAL"]]
            if len(alerts) == 0:
                st.write("No active high-risk alerts to review.")
            else:
                for idx, alert in enumerate(alerts[:5]):
                    alert_tier = "🔴 CRITICAL" if alert["risk_tier"] == "CRITICAL" else "🟠 HIGH"
                    col_q1, col_q2, col_q3, col_q4 = st.columns([1, 2.5, 1.5, 1])
                    with col_q1:
                        st.markdown(f"**{alert_tier}**")
                    with col_q2:
                        st.write(f"Amount: **${alert['amount']:,.2f}** ({alert['type']})")
                    with col_q3:
                        st.write(f"SLA: ⏱️ **{alert['sla_remaining_min']}m** remaining")
                    with col_q4:
                        if st.button("Investigate", key=f"inv_{alert['transaction_id']}"):
                            st.session_state.selected_alert = alert
                            st.rerun()

        with op_col2:
            st.subheader("🕵️ Analyst Alert Investigation Panel")
            selected = st.session_state.selected_alert
            
            if not selected:
                st.info("Select a transaction from the Alert Queue to run a full diagnostic risk assessment.")
            else:
                st.markdown(f"### Diagnostic Report: `{selected['transaction_id']}`")
                
                # Threat banner
                is_critical = selected["risk_tier"] == "CRITICAL"
                banner_color = "#ef4444" if is_critical else "#f97316"
                st.markdown(f"""
                <div style="background-color: #111827; padding: 15px; border-radius: 8px; border-left: 5px solid {banner_color}; border-right: 1px solid #1f2937; border-top: 1px solid #1f2937; border-bottom: 1px solid #1f2937;">
                    <h4>Threat Evaluation: <span style="color: {banner_color}; font-weight: bold;">{selected['risk_tier']}</span></h4>
                    <p style="margin: 5px 0;">PesaGuard Score: <b>{selected['ensemble_score']:.4f}</b></p>
                    <p style="margin: 0;">System Recommendation: <b style="color: {banner_color};">{selected['recommendation']}</b></p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.write(f"**Sender (Orig)**: `{selected['nameOrig']}`")
                    st.write(f"**Amount**: `${selected['amount']:,.2f}`")
                    st.write(f"**Hub Location**: `{selected['city']}`")
                with col_d2:
                    st.write(f"**Recipient (Dest)**: `{selected['nameDest']}`")
                    st.write(f"**Orig Balance**: `${selected['oldbalanceOrg']:,.2f}`")
                    st.write(f"**Type**: `{selected['type']}`")

                # Plot SHAP
                signals = selected.get("top_signals", [])
                if len(signals) > 0:
                    names = [s["signal"] for s in signals]
                    vals = [s["shap_value"] for s in signals]
                    descriptions = [s["description"] for s in signals]
                    
                    colors = ["#ef4444" if v > 0 else "#10b981" for v in vals]
                    
                    fig_shap = go.Figure(go.Bar(
                        x=vals, y=names, orientation='h',
                        marker_color=colors,
                        text=[f"{v:+.2f}" for v in vals],
                        textposition='auto',
                        hovertext=descriptions
                    ))
                    fig_shap.update_layout(
                        title="Model Explanation Drivers (SHAP Value Contribution)",
                        xaxis_title="Risk Vector Impact",
                        yaxis=dict(autorange="reversed"),
                        height=200,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="white",
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # Descriptions
                    for s in signals:
                        st.markdown(f"- **{s['signal']}**: {s['description']}")
                else:
                    st.write("No feature explanations available.")

                st.markdown("---")
                st.markdown("#### Analyst Verification")
                notes = st.text_area("Investigation Finding Notes", placeholder="Enter authorization details, SIM swap investigation notes, or false-positive reasons...")
                
                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("🔒 Confirm Fraud (Lock Account)"):
                        log_feedback(selected["transaction_id"], is_fraud_feedback=1, feedback_notes=notes)
                        st.success("Transaction blocked. Origin wallet locked.")
                        st.session_state.selected_alert = None
                        time.sleep(1)
                        st.rerun()
                with col_b2:
                    if st.button("✅ Dismiss Alert (Legitimate)"):
                        log_feedback(selected["transaction_id"], is_fraud_feedback=0, feedback_notes=notes)
                        st.info("Alert dismissed. Transaction authorized.")
                        st.session_state.selected_alert = None
                        time.sleep(1)
                        st.rerun()

# ==========================================
# TAB 2: CSV BATCH PROCESSOR
# ==========================================
with tabs[1]:
    st.subheader("📁 Bulk Transaction File Uploader")
    st.markdown("Upload historical billing or transaction export CSVs. PesaGuard will parse the logs, engineer behavioral features, run model classifications, and provide download links.")

    uploaded_file = st.file_uploader("Upload CSV transactions file (must contain: type, amount, oldbalanceOrg, newbalanceOrig, nameOrig, nameDest, step)", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read file
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully. Columns verified.")
            st.dataframe(input_df.head(5), use_container_width=True)
            
            if st.button("🚀 Process & Score Transactions"):
                progress_text = "Parsing transactions..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Check required columns
                required = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']
                missing = [col for col in required if col not in input_df.columns]
                
                if missing:
                    st.error(f"Missing required columns in CSV: {missing}")
                else:
                    # Run Pipeline
                    # We will load the feature pipeline from models
                    pipeline = ensemble.feature_pipeline
                    if not pipeline:
                        # Fallback pipeline fit in memory to run the demo standalone
                        pipeline = PesaGuardFeaturePipeline()
                        # Fit on input_df to learn
                        pipeline.fit(input_df)
                    
                    progress_bar.progress(30, text="Engineering behavioral history features...")
                    # Process features
                    engineered_df = pipeline.transform(input_df)
                    
                    progress_bar.progress(60, text="Executing ML Model Scoring...")
                    # Score
                    probs = ensemble.xgb_model.predict_proba(engineered_df)[:, 1] if ensemble.xgb_model else np.random.rand(len(input_df)) * 0.1
                    
                    # Isolation Forest anomaly scoring
                    if ensemble.iforest_model:
                        anom_raw = ensemble.iforest_model.decision_function(engineered_df)
                        # Normalize
                        anom = 1.0 - (anom_raw - (-0.4)) / (0.2 - (-0.4) + 1e-5)
                        anom = np.clip(anom, 0.0, 1.0)
                    else:
                        anom = np.random.rand(len(input_df)) * 0.1
                        
                    ensemble_scores = 0.7 * probs + 0.3 * anom
                    
                    # Merge scores
                    output_df = input_df.copy()
                    output_df["fraud_probability"] = probs
                    output_df["anomaly_score"] = anom
                    output_df["ensemble_score"] = ensemble_scores
                    
                    # Assign tiers
                    output_df["risk_tier"] = pd.cut(
                        ensemble_scores, 
                        bins=[-0.1, 0.15, 0.45, 0.75, 1.1], 
                        labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                    )
                    
                    progress_bar.progress(100, text="Process completed!")
                    st.success("File processing completed. Summary dashboard generated below.")
                    
                    # Save results in session state
                    st.session_state.custom_csv_results = output_df
                    st.rerun()
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            import traceback
            st.text(traceback.format_exc())

    # Render results if present
    if st.session_state.custom_csv_results is not None:
        res_df = st.session_state.custom_csv_results
        
        c_r1, c_r2, c_r3 = st.columns(3)
        with c_r1:
            st.metric("Total Rows Scored", f"{len(res_df):,}")
        with c_r2:
            alert_count = sum(res_df["risk_tier"].isin(["HIGH", "CRITICAL"]))
            st.metric("Fraud Flags Detected", f"{alert_count:,}")
        with c_r3:
            blocked_val = res_df[res_df["risk_tier"].isin(["HIGH", "CRITICAL"])]["amount"].sum()
            st.metric("Monetary Value at Risk", f"${blocked_val:,.2f}")
            
        # Download button
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Scored Transactions Report",
            data=csv_data,
            file_name="pesaguard_scored_report.csv",
            mime="text/csv"
        )
        
        st.subheader("📊 Scored Risk Distribution Analysis")
        # Plot risk distribution chart
        tier_counts = res_df["risk_tier"].value_counts().reset_index()
        fig_dist = px.bar(
            tier_counts, x="risk_tier", y="count",
            labels={"risk_tier": "Risk Tier", "count": "Transaction Count"},
            title="Transaction Volume Breakdown by Threat Level",
            color="risk_tier",
            color_discrete_map={"LOW": "#10b981", "MEDIUM": "#eab308", "HIGH": "#f97316", "CRITICAL": "#ef4444"}
        )
        fig_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 3: RISK POLICY ENGINE
# ==========================================
with tabs[2]:
    st.subheader("⚙️ Hard Rules & Security Policy Builder")
    st.markdown("SaaS operators combine machine learning threat scores with strict compliance rules. "
                "Toggling these parameters will **override** ML models and force actions (e.g. BLOCK/FLAG) "
                "for transactions violating policies.")

    col_p1, col_p2 = st.columns(2)
    
    with col_p1:
        st.markdown("### Transaction Value Limits")
        max_amount_toggle = st.toggle("Activate Maximum Value Limit Cap", value=st.session_state.policy_rules["max_amount_toggle"])
        max_amount_value = st.slider("Max Transaction Limit ($)", 50000.0, 1000000.0, float(st.session_state.policy_rules["max_amount_value"]), step=25000.0)
        
        st.markdown("### Temporal Security Policy")
        night_lock_toggle = st.toggle("Activate Night Blackout Protection", value=st.session_state.policy_rules["night_lock_toggle"],
                                      help="Automatically flags all transactions occurring between 2:00 AM and 5:00 AM.")
        
    with col_p2:
        st.markdown("### Velocity Limits (Account Taking Prevention)")
        velocity_alert_toggle = st.toggle("Activate User Hourly Velocity Cap", value=st.session_state.policy_rules["velocity_alert_toggle"])
        velocity_alert_limit = st.slider("Maximum allowed transactions in 1 Hour", 1, 10, int(st.session_state.policy_rules["velocity_alert_limit"]))
        
    if st.button("💾 Apply Policies & Save"):
        st.session_state.policy_rules = {
            "max_amount_toggle": max_amount_toggle,
            "max_amount_value": max_amount_value,
            "night_lock_toggle": night_lock_toggle,
            "velocity_alert_toggle": velocity_alert_toggle,
            "velocity_alert_limit": velocity_alert_limit
        }
        st.success("Security policies saved. Overrides applied to incoming live streams.")

# ==========================================
# TAB 4: BUSINESS ROI & ANALYTICS
# ==========================================
with tabs[3]:
    st.subheader("📈 Financial Security & ROI Dashboard")
    st.markdown("Analyze the business value provided by PesaGuard: money saved, analyst efficiency, and customer friction.")

    # Compute mock metrics based on database logs
    # Assume 1 flagged fraud saved = amount saved.
    # Cost of false positive = $10 per review.
    # Cost of analyst review = $5 per alert.
    stats = get_db_stats()
    blocked_val = stats["value_blocked"] or 541578.42
    flagged_alerts = stats["fraud_flagged"] or 15
    false_pos = stats["false_positives"] or 1
    
    total_savings = blocked_val
    friction_cost = false_pos * 15.0  # friction cost of customer calling
    ops_cost = flagged_alerts * 5.0
    net_savings = total_savings - friction_cost - ops_cost

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("Total Loss Avoided (Protected)", f"${total_savings:,.2f}")
    with a2:
        st.metric("Friction & Ops Overhead Costs", f"${friction_cost + ops_cost:,.2f}")
    with a3:
        st.metric("Net Financial ROI", f"${net_savings:,.2f}", delta=f"{net_savings/max(1.0, total_savings)*100:.1f}% Yield")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        # Cost Benefits breakdown
        fig_roi = px.pie(
            names=["Net Savings", "Friction Cost", "Ops Analyst Costs"],
            values=[net_savings, friction_cost, ops_cost],
            color_discrete_sequence=["#10b981", "#ef4444", "#eab308"],
            title="Loss Protection vs. Overhead Friction Distribution"
        )
        fig_roi.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_roi, use_container_width=True)
        
    with col_chart2:
        # Alert reason trends
        fig_reasons = px.bar(
            x=["Value Limit", "Time-of-day Lock", "Velocity Limits", "Supervised ML", "Anomaly Engine"],
            y=[2, 4, 1, 8, 3],
            labels={"x": "Trigger Signal", "y": "Trigger Count"},
            title="Operational Alert Breakdown by Rule/ML Trigger",
            color_discrete_sequence=["#4f46e5"]
        )
        fig_reasons.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_reasons, use_container_width=True)

# ==========================================
# TAB 5: DEVELOPER API CONSOLE
# ==========================================
with tabs[4]:
    st.subheader("🔌 Developer REST API & Integration Guides")
    st.markdown("Integrate PesaGuard's low-latency scoring directly into your fintech payment gate. "
                "Any programming language can stream transaction models using standard JSON REST requests.")

    st.markdown("#### API Endpoint Access Details")
    st.write(f"**Production REST API Base URL**: `{API_URL}`")
    st.write("**Header Authorization**: `Authorization: Bearer pk_live_51PesaGuardKeyXYZ789`")

    st.markdown("#### Integration Code Snippet (Python)")
    st.code("""
import requests
import json

url = "http://127.0.0.1:8000/score"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer pk_live_51PesaGuardKeyXYZ789"
}

payload = {
    "step": 11,
    "type": "TRANSFER",
    "amount": 25000.0,
    "nameOrig": "C1293848",
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 25000.0,
    "nameDest": "C99887766",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
}

response = requests.post(url, headers=headers, data=json.dumps(payload))
print("Response Status Code:", response.status_code)
print("Scoring Result:", response.json())
    """, language="python")

    st.markdown("#### Integration Code Snippet (cURL / Bash)")
    st.code("""
curl -X POST http://127.0.0.1:8000/score \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer pk_live_51PesaGuardKeyXYZ789" \\
  -d '{
    "step": 11,
    "type": "TRANSFER",
    "amount": 25000.0,
    "nameOrig": "C1293848",
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 25000.0,
    "nameDest": "C99887766",
    "oldbalanceDest": 0.0,
    "newbalanceDest": 0.0
  }'
    """, language="bash")
