import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import os
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import database module for fallback/stats
from src.db import get_db_stats, log_feedback

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PesaGuard | Real-Time Fraud Monitoring",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (CSS) for Glassmorphism Dark Theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #00ffcc;
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 6px;
        font-weight: 600;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-2px);
    }
    .badge-critical {
        background-color: #ef4444;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-high {
        background-color: #f97316;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-medium {
        background-color: #eab308;
        color: black;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-low {
        background-color: #22c55e;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
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

# Initialize Session State
if "streaming" not in st.session_state:
    st.session_state.streaming = False
if "scored_transactions" not in st.session_state:
    st.session_state.scored_transactions = []
if "selected_alert" not in st.session_state:
    st.session_state.selected_alert = None
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Helper to mock coordinates in East/West Africa based on nameDest
def get_mock_coordinates(name_dest):
    # Deterministic hash of string
    h = hash(name_dest)
    
    # Hub coordinates: Nairobi, Lagos, Kampala, Dar es Salaam
    hubs = [
        {"city": "Nairobi", "lat": -1.2921, "lon": 36.8219},
        {"city": "Lagos", "lat": 6.5244, "lon": 3.3792},
        {"city": "Kampala", "lat": 0.3476, "lon": 32.5825},
        {"city": "Dar es Salaam", "lat": -6.7924, "lon": 39.2083}
    ]
    
    hub = hubs[h % len(hubs)]
    
    # Add random jitter within 150km
    jitter_lat = ((h % 100) - 50) / 100.0 * 0.8
    jitter_lon = (((h >> 2) % 100) - 50) / 100.0 * 0.8
    
    return hub["lat"] + jitter_lat, hub["lon"] + jitter_lon, hub["city"]

# Dashboard Header
st.title("🛡️ PesaGuard | Real-Time Fraud Operations Center")
st.markdown("Commercial SaaS dashboard for M-Pesa, Airtel Money, and MTN mobile money operators.")

# Sidebar Control Center
st.sidebar.header("🛠️ Simulation Control Center")

api_online = get_api_health()
if api_online:
    st.sidebar.success("🟢 REST API Status: ONLINE")
else:
    st.sidebar.error("🔴 REST API Status: OFFLINE (Running local model fallback)")

# Simulation settings
st.sidebar.markdown("---")
stream_speed = st.sidebar.slider("Streaming Speed (seconds per transaction)", 0.2, 3.0, 1.0, step=0.1)

# Start/Stop Buttons
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("▶️ Start Feed", key="start_btn"):
        st.session_state.streaming = True
with col2:
    if st.button("⏸️ Pause", key="pause_btn"):
        st.session_state.streaming = False

if st.sidebar.button("🔄 Clear Console Log"):
    st.session_state.scored_transactions = []
    st.session_state.selected_alert = None
    st.session_state.current_index = 0
    st.rerun()

# Load test sample dataset
sample_path = "data/paysim_test_sample.csv"
if not os.path.exists(sample_path):
    st.error(f"Test sample dataset not found at {sample_path}. Please run train.py first to generate the model assets and validation dataset.")
    st.stop()

test_data = pd.read_csv(sample_path)

# Scoring Logic (API integration with local fallback)
def score_row(row):
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
    
    if api_online:
        try:
            r = requests.post(f"{API_URL}/score", json=tx_payload, timeout=0.5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
            
    # Local fallback scoring using pre-computed model predictions from train.py
    lat, lon, hub_city = get_mock_coordinates(row["nameDest"])
    
    # Check if pre-computed scores are present in row
    prob = float(row.get("xgb_prob", 0.05))
    anomaly = float(row.get("anomaly_score", 0.1))
    score = float(row.get("ensemble_score", 0.065))
    
    risk_tier = "LOW"
    recommendation = "ALLOW"
    if score >= 0.75:
        risk_tier = "CRITICAL"
        recommendation = "BLOCK"
    elif score >= 0.45:
        risk_tier = "HIGH"
        recommendation = "FLAG_FOR_REVIEW"
    elif score >= 0.15:
        risk_tier = "MEDIUM"
        recommendation = "REVIEW_LATER"

    # Mock signals for fallback
    top_signals = [
        {"signal": "amount", "description": f"Transaction amount of {row['amount']:,.2f} is high value.", "impact": "medium", "shap_value": 0.3},
        {"signal": "is_transfer", "description": f"Transaction type is TRANSFER.", "impact": "medium", "shap_value": 0.25}
    ]
    if row["type"] in ["TRANSFER", "CASH_OUT"] and row["amount"] > 100000:
        top_signals.insert(0, {"signal": "amount_deviation", "description": "Transaction amount is 3.5 standard deviations above client average", "impact": "high", "shap_value": 1.2})
        
    return {
        "transaction_id": f"TX_{row['nameOrig']}_{row['step']}_MOCK",
        "fraud_probability": prob,
        "anomaly_score": anomaly,
        "ensemble_score": score,
        "risk_tier": risk_tier,
        "top_signals": top_signals,
        "recommendation": recommendation,
        "response_time_ms": 12
    }

# 1. Main Metrics Row
db_stats = get_db_stats()
total_scored = db_stats["total_transactions"] or len(st.session_state.scored_transactions)
fraud_flagged = db_stats["fraud_flagged"] or sum(1 for tx in st.session_state.scored_transactions if tx["risk_tier"] in ["HIGH", "CRITICAL"])
value_blocked = db_stats["value_blocked"] or sum(tx["amount"] for tx in st.session_state.scored_transactions if tx["risk_tier"] in ["HIGH", "CRITICAL"])

# Calculate False Positive Rate
feedback_count = db_stats["total_feedback"]
false_positives = db_stats["false_positives"]
fpr_pct = (false_positives / feedback_count * 100) if feedback_count > 0 else 0.0

m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("Transactions Monitored", f"{total_scored:,}", help="Total number of mobile money transactions processed through PesaGuard")
with m_col2:
    st.metric("Fraud Alerts Triggered", f"{fraud_flagged:,}", help="Count of transactions flagged as HIGH or CRITICAL risk")
with m_col3:
    st.metric("Suspicious Value Blocked", f"${value_blocked:,.2f}", help="Total currency value withheld or flagged for operations review")
with m_col4:
    st.metric("Ops False Positive Rate", f"{fpr_pct:.2f}%", help="Percentage of analyst-reviewed alerts found to be legitimate transactions", delta=f"{feedback_count} Reviews")

st.markdown("---")

# Main Screen Split: Left (Streaming Feed) | Right (Operations Review Panel)
feed_col, review_col = st.columns([1.3, 1.0])

# Background Streaming Thread
if st.session_state.streaming:
    if st.session_state.current_index < len(test_data):
        row = test_data.iloc[st.session_state.current_index]
        scored = score_row(row)
        
        # Merge row details with score results
        full_tx = {**row.to_dict(), **scored}
        lat, lon, city = get_mock_coordinates(row["nameDest"])
        full_tx["lat"] = lat
        full_tx["lon"] = lon
        full_tx["city"] = city
        
        # Append to the top of our log
        st.session_state.scored_transactions.insert(0, full_tx)
        
        # Keep log size reasonable (max 100)
        if len(st.session_state.scored_transactions) > 100:
            st.session_state.scored_transactions.pop()
            
        st.session_state.current_index += 1
        time.sleep(stream_speed)
        st.rerun()

with feed_col:
    st.subheader("📡 Live Transaction Stream")
    
    if len(st.session_state.scored_transactions) == 0:
        st.info("Simulation inactive. Click 'Start Feed' in the sidebar to stream transactions.")
    else:
        # Create a clean dataframe for representation
        feed_rows = []
        for tx in st.session_state.scored_transactions:
            # Color risk badges
            badge = "LOW"
            if tx["risk_tier"] == "CRITICAL":
                badge = "🔴 CRITICAL"
            elif tx["risk_tier"] == "HIGH":
                badge = "🟠 HIGH"
            elif tx["risk_tier"] == "MEDIUM":
                badge = "🟡 MEDIUM"
            else:
                badge = "🟢 LOW"
                
            feed_rows.append({
                "Tx ID": tx["transaction_id"],
                "Time Hour": int(tx["step"]),
                "Sender": tx["nameOrig"],
                "Receiver": tx["nameDest"],
                "Amount": f"${tx['amount']:,.2f}",
                "Type": tx["type"],
                "Risk": badge,
                "Score": f"{tx['ensemble_score']:.3f}"
            })
            
        feed_df = pd.DataFrame(feed_rows)
        st.dataframe(feed_df, use_container_width=True, height=350)
        
    # Queue section for HIGH/CRITICAL alerts
    st.subheader("🚨 Investigation Queue (High & Critical Risk)")
    alerts = [tx for tx in st.session_state.scored_transactions if tx["risk_tier"] in ["HIGH", "CRITICAL"]]
    
    if len(alerts) == 0:
        st.write("No active high-risk alerts in session buffer.")
    else:
        # Show queue with button selection
        for idx, alert in enumerate(alerts[:8]):
            alert_label = "🔴 CRITICAL" if alert["risk_tier"] == "CRITICAL" else "🟠 HIGH"
            col_a, col_b, col_c, col_d = st.columns([1, 2, 2, 1])
            with col_a:
                st.write(f"**{alert_label}**")
            with col_b:
                st.write(f"Amount: **${alert['amount']:,.2f}** ({alert['type']})")
            with col_c:
                st.write(f"Orig: {alert['nameOrig']}")
            with col_d:
                if st.button("Review", key=f"rev_{alert['transaction_id']}"):
                    st.session_state.selected_alert = alert
                    st.rerun()

with review_col:
    st.subheader("🕵️ Alert Investigation Workspace")
    
    selected = st.session_state.selected_alert
    
    if selected is None:
        st.info("Select a transaction from the Investigation Queue to load its explanation parameters.")
    else:
        st.markdown(f"### Alert Details: `{selected['transaction_id']}`")
        
        # Risk Badge and Recommendation Card
        badge_class = "badge-critical" if selected["risk_tier"] == "CRITICAL" else "badge-high"
        st.markdown(f"""
        <div style="background-color:#1e293b; padding:15px; border-radius:8px; border-left: 5px solid {'#ef4444' if selected['risk_tier']=='CRITICAL' else '#f97316'};">
            <h4>Risk Assessment: <span class="{badge_class}">{selected['risk_tier']} RISK</span></h4>
            <p style="margin-top: 5px;">Ensemble Threat Score: <b>{selected['ensemble_score']:.4f}</b></p>
            <p>Action Recommendation: <b style="color: {'#ef4444' if selected['recommendation']=='BLOCK' else '#f97316'}">{selected['recommendation']}</b></p>
            <small>FastAPI latency: {selected['response_time_ms']} ms | Source Hub: {selected['city']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Transaction Metadata")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.write(f"**Sender**: `{selected['nameOrig']}`")
            st.write(f"**Amount**: `${selected['amount']:,.2f}`")
            st.write(f"**Type**: `{selected['type']}`")
        with col_m2:
            st.write(f"**Receiver**: `{selected['nameDest']}`")
            st.write(f"**Origin Balance**: `${selected['oldbalanceOrg']:,.2f}`")
            st.write(f"**Hour (Step)**: `{selected['step']}`")
            
        st.markdown("#### Model Decision Signals (SHAP Explainability)")
        
        # Draw Plotly horizontal bar chart representing SHAP Waterfall
        signals = selected.get("top_signals", [])
        if len(signals) > 0:
            names = [s["signal"] for s in signals]
            vals = [s.get("shap_value", 0.0) for s in signals]
            descriptions = [s["description"] for s in signals]
            
            # Map colors based on direction of impact
            colors = ['#ef4444' if v > 0 else '#22c55e' for v in vals]
            
            fig = go.Figure(go.Bar(
                x=vals,
                y=names,
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.2f}" for v in vals],
                textposition='auto',
                hoverinfo='text',
                hovertext=descriptions
            ))
            fig.update_layout(
                title="Feature Risk Contributions (SHAP Values)",
                xaxis_title="Impact on Fraud Risk (Positive = Drives Risk Up)",
                yaxis=dict(autorange="reversed"),
                height=220,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Display bullet point descriptions
            for s in signals:
                st.markdown(f"- **{s['signal']}**: {s['description']}")
        else:
            st.write("No SHAP values logged for this transaction.")
            
        # human analyst action form
        st.markdown("#### Human Analyst Action Panel")
        feedback_notes = st.text_area("Investigation Notes", placeholder="Enter findings, e.g. 'Customer confirmed unauthorized transaction' or 'Legitimate merchant payout'...", key="notes_input")
        
        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🚨 Confirm Fraud (Block Account)", key="confirm_fraud_btn"):
                # Save feedback to DB
                log_feedback(selected["transaction_id"], is_fraud_feedback=1, feedback_notes=feedback_notes)
                st.success("Alert Blocked. Account locked.")
                st.session_state.selected_alert = None
                time.sleep(1)
                st.rerun()
        with col_act2:
            if st.button("✅ Dismiss Alert (False Positive)", key="dismiss_alert_btn"):
                # Save feedback to DB
                log_feedback(selected["transaction_id"], is_fraud_feedback=0, feedback_notes=feedback_notes)
                st.info("Alert dismissed as false alarm.")
                st.session_state.selected_alert = None
                time.sleep(1)
                st.rerun()

st.markdown("---")

# 2. Bottom Row: Time-Series & Map
t_col1, t_col2 = st.columns([1.2, 1.0])

with t_col1:
    st.subheader("📈 Fraud Alert Timeline")
    # Plot alert detection rate over the simulated session
    if len(st.session_state.scored_transactions) > 0:
        hist_df = pd.DataFrame(st.session_state.scored_transactions)
        
        # Group by step and get counts of alerts
        hist_stats = hist_df.groupby("step").agg(
            total_tx=("isFraud", "count"),
            flagged=("risk_tier", lambda x: sum(1 for v in x if v in ["HIGH", "CRITICAL"]))
        ).reset_index()
        
        fig_time = px.line(
            hist_stats, x="step", y=["total_tx", "flagged"],
            labels={"value": "Transaction Count", "step": "Hour of Day (step)", "variable": "Metric"},
            title="Monitored Transactions vs. Flagged Fraud Cases",
            color_discrete_map={"total_tx": "#4f46e5", "flagged": "#ef4444"}
        )
        fig_time.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Time-series chart will populate when streaming data commences.")

with t_col2:
    st.subheader("🗺️ African Mobile Money Risk Heatmap")
    # Plot Map of current streaming transactions centered in Kenya/Nigeria
    if len(st.session_state.scored_transactions) > 0:
        map_df = pd.DataFrame(st.session_state.scored_transactions)
        
        # Color coding points by risk tier
        color_map = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}
        map_df["color"] = map_df["risk_tier"].map(color_map)
        
        fig_map = px.scatter_mapbox(
            map_df, 
            lat="lat", 
            lon="lon",
            color="risk_tier",
            color_discrete_map=color_map,
            size="amount", 
            size_max=35,
            zoom=2.5,
            center=dict(lat=-1.2921, lon=25.0), # Centered in Central/East Africa
            mapbox_style="carto-darkmatter",
            hover_name="city",
            hover_data={
                "transaction_id": True,
                "amount": ":$,.2f",
                "risk_tier": True,
                "ensemble_score": ":.3f",
                "lat": False,
                "lon": False
            },
            title="Transactions Mapped by Hub and Risk Tier"
        )
        fig_map.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("Risk map will populate when transactions begin streaming.")
