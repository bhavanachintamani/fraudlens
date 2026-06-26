"""
app.py
------
Live-style dashboard: every time you click "Generate next batch", a new
set of synthetic transactions streams in, gets scored by the anomaly
detector, and any flagged transactions are surfaced with a clear reason.

Run:
    streamlit run app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st

from anomaly_detector import AnomalyDetector
from data_simulator import generate_transaction_batch

st.set_page_config(page_title="FraudLens | Real-Time Anomaly Detection", page_icon="🔍", layout="wide")

if "all_transactions" not in st.session_state:
    st.session_state.all_transactions = pd.DataFrame()
if "detector" not in st.session_state:
    st.session_state.detector = AnomalyDetector()
    # Warm up the model on an initial baseline batch so scoring is meaningful
    baseline = generate_transaction_batch(n=300, fraud_rate=0.05, seed=1)
    st.session_state.detector.fit(baseline)
if "batch_count" not in st.session_state:
    st.session_state.batch_count = 0

st.title("🔍 FraudLens")
st.caption("Real-time anomaly detection on streaming transactions — IsolationForest + explainable rule checks.")

col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    batch_size = st.slider("Transactions per batch", 10, 200, 50)
with col2:
    fraud_rate = st.slider("Injected anomaly rate", 0.0, 0.3, 0.06, step=0.01)
with col3:
    generate = st.button("▶ Generate next batch", type="primary")

if generate:
    st.session_state.batch_count += 1
    new_batch = generate_transaction_batch(
        n=batch_size, fraud_rate=fraud_rate, seed=st.session_state.batch_count
    )
    scored = st.session_state.detector.score(new_batch)
    scored["zscore_flag"] = AnomalyDetector.zscore_flags(scored)
    scored["batch"] = st.session_state.batch_count
    st.session_state.all_transactions = pd.concat(
        [st.session_state.all_transactions, scored], ignore_index=True
    )

df = st.session_state.all_transactions

if df.empty:
    st.info("Click 'Generate next batch' to start streaming transactions.")
else:
    flagged = df[df["is_flagged"]]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total transactions", len(df))
    m2.metric("Flagged by model", len(flagged))
    m3.metric("Flag rate", f"{len(flagged) / len(df) * 100:.1f}%")
    m4.metric(
        "True fraud caught",
        f"{len(flagged[flagged['is_fraud'] == 1])}/{len(df[df['is_fraud'] == 1])}",
    )

    st.subheader("Transaction stream")
    fig = px.scatter(
        df,
        x="distance_from_home_km",
        y="amount",
        color="is_flagged",
        symbol="is_fraud",
        hover_data=["transaction_id", "hour_of_day", "anomaly_score"],
        color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
        title="Transaction amount vs. distance from home (red = flagged as anomalous)",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🚩 Flagged transactions (most recent batch first)")
    display_cols = [
        "transaction_id", "batch", "amount", "hour_of_day",
        "distance_from_home_km", "transactions_last_hour",
        "anomaly_score", "zscore_flag", "is_fraud",
    ]
    st.dataframe(
        flagged.sort_values("batch", ascending=False)[display_cols],
        use_container_width=True,
    )

    with st.expander("ℹ️ How detection works"):
        st.markdown(
            "- **IsolationForest** learns the 'normal' shape of transactions "
            "(amount, time of day, distance from home, transaction frequency) "
            "and flags points that sit far outside that shape.\n"
            "- **Z-score rule check** is a simple, explainable backstop: flags "
            "any transaction whose amount is a statistical outlier on its own. "
            "Real fraud systems often keep a rule layer like this for "
            "transparency/audit purposes, even alongside ML models.\n"
            "- `is_fraud` is the *ground truth* injected by the simulator, shown "
            "here only so you can evaluate how well the detector performs."
        )
