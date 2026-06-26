"""
data_simulator.py
------------------
Generates a stream of synthetic credit-card-style transactions, with
a small injected fraction of anomalous/fraudulent transactions so the
detector has something real to catch.
"""

import numpy as np
import pandas as pd


def generate_transaction_batch(n=50, fraud_rate=0.05, seed=None):
    rng = np.random.default_rng(seed)

    n_fraud = max(1, int(n * fraud_rate))
    n_normal = n - n_fraud

    normal = pd.DataFrame(
        {
            "amount": rng.normal(60, 30, n_normal).clip(1, None),
            "hour_of_day": rng.normal(14, 4, n_normal).clip(0, 23),
            "distance_from_home_km": rng.exponential(5, n_normal),
            "transactions_last_hour": rng.poisson(1, n_normal),
            "is_fraud": 0,
        }
    )

    fraud = pd.DataFrame(
        {
            "amount": rng.normal(450, 200, n_fraud).clip(1, None),
            "hour_of_day": rng.choice([1, 2, 3, 23], n_fraud),
            "distance_from_home_km": rng.exponential(300, n_fraud),
            "transactions_last_hour": rng.poisson(6, n_fraud),
            "is_fraud": 1,
        }
    )

    batch = pd.concat([normal, fraud], ignore_index=True)
    batch = batch.sample(frac=1, random_state=seed).reset_index(drop=True)
    batch["transaction_id"] = [f"TXN-{i:06d}" for i in range(len(batch))]
    return batch
