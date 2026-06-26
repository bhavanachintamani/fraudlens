"""
anomaly_detector.py
--------------------
Two detection approaches, both common in real fraud systems:

1. IsolationForest (unsupervised ML) - learns "normal" transaction
   shape and flags points that don't fit it.
2. Z-score rule-based check - simple, explainable, fast - the kind
   of layer real systems keep alongside ML models for transparency.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURES = ["amount", "hour_of_day", "distance_from_home_km", "transactions_last_hour"]


class AnomalyDetector:
    def __init__(self, contamination=0.06):
        self.model = IsolationForest(
            n_estimators=150, contamination=contamination, random_state=42
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame):
        self.model.fit(df[FEATURES])
        self._fitted = True

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'anomaly_score' (higher = more anomalous) and 'is_flagged' columns."""
        if not self._fitted:
            self.fit(df)

        raw_scores = self.model.decision_function(df[FEATURES])  # higher = more normal
        anomaly_score = -raw_scores  # flip so higher = more anomalous
        predictions = self.model.predict(df[FEATURES])  # -1 = anomaly, 1 = normal

        result = df.copy()
        result["anomaly_score"] = np.round(anomaly_score, 4)
        result["is_flagged"] = predictions == -1
        return result

    @staticmethod
    def zscore_flags(df: pd.DataFrame, threshold: float = 2.5) -> pd.Series:
        """Simple explainable rule: flag if 'amount' is a statistical outlier."""
        z = (df["amount"] - df["amount"].mean()) / df["amount"].std()
        return z.abs() > threshold
