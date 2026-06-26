# 🔍 FraudLens — Real-Time Anomaly Detection Dashboard

FraudLens simulates a live stream of financial transactions and flags anomalous/fraudulent ones in real time, using an unsupervised ML model (IsolationForest) plus a simple explainable rule-based backstop — mirroring how real fraud-detection systems layer ML and rules together.

## Why this project

This is a strong project for "business impact" framing in interviews: anomaly detection has obvious revenue/risk implications, and the project shows you understand both ML detection AND the importance of explainability/transparency layers that real systems need.

## Features

- 🔄 Streaming simulation — generate new transaction batches on demand, watch flags update live
- 🧠 IsolationForest unsupervised anomaly detection (no labeled fraud data needed to train)
- 📏 Explainable z-score rule check as a transparent secondary signal
- 📊 Interactive scatter plot (amount vs. distance-from-home) with flagged points highlighted
- 🎯 Live precision tracking against injected ground-truth fraud labels

## Architecture

```
data_simulator.py → generates synthetic transaction batches
                          ↓
anomaly_detector.py → IsolationForest (ML) + z-score rule (explainable)
                          ↓
app.py (Streamlit) → live dashboard, scatter plot, flagged transaction table
```

## Tech Stack

`Python` · `Streamlit` · `scikit-learn` · `Plotly` · `Pandas`

## Setup

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Click **"Generate next batch"** repeatedly to simulate a live transaction stream.

## Results / Metrics (fill in after running)

- Fraud capture rate at default settings: __ / __ true fraud cases caught
- False positive rate: __%
- Avg detection latency per batch: __ ms

## Future improvements

- Swap synthetic data for a real public fraud dataset (e.g., Kaggle Credit Card Fraud)
- Add a supervised model (XGBoost) trained on the ground-truth labels for comparison
- Persist flagged transactions to a database instead of in-memory session state
