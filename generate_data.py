"""
generate_data.py
-----------------
Generates a synthetic, ANONYMIZED financial transactions dataset that
mimics the structure of real-world fraud datasets (like the popular
Kaggle "Credit Card Fraud Detection" dataset which uses PCA-anonymized
features V1...V28 + Time + Amount + Class).

Why synthetic data?
- Real fraud datasets are sensitive / not distributable.
- This script creates a realistic, class-imbalanced dataset (only
  ~0.5% fraud) so the whole pipeline (EDA -> imbalance handling ->
  anomaly detection -> dashboard) works end-to-end out of the box.

Run:
    python src/generate_data.py
Output:
    data/transactions.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

N_SAMPLES = 20000          # total transactions
FRAUD_RATIO = 0.006        # ~0.6% fraud -> realistic class imbalance
N_ANON_FEATURES = 10       # anonymized PCA-like features (V1...V10)


def generate_transactions(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO):
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud

    # ---- Normal transactions ----
    normal_features = np.random.normal(loc=0.0, scale=1.0,
                                        size=(n_normal, N_ANON_FEATURES))
    normal_amount = np.random.exponential(scale=60, size=n_normal)
    normal_time = np.random.randint(0, 172800, size=n_normal)  # 2 days in seconds

    # ---- Fraudulent transactions (shifted distribution = anomalies) ----
    fraud_features = np.random.normal(loc=2.5, scale=1.8,
                                       size=(n_fraud, N_ANON_FEATURES))
    fraud_amount = np.random.exponential(scale=350, size=n_fraud) + 50
    fraud_time = np.random.randint(0, 172800, size=n_fraud)

    # ---- Combine ----
    features = np.vstack([normal_features, fraud_features])
    amount = np.concatenate([normal_amount, fraud_amount])
    time = np.concatenate([normal_time, fraud_time])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    col_names = [f"V{i+1}" for i in range(N_ANON_FEATURES)]
    df = pd.DataFrame(features, columns=col_names)
    df["Time"] = time
    df["Amount"] = np.round(amount, 2)
    df["Class"] = labels.astype(int)  # 1 = fraud, 0 = normal

    # Shuffle rows so fraud is scattered like in real data
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Add a synthetic transaction ID
    df.insert(0, "TransactionID", [f"TXN{100000+i}" for i in range(len(df))])

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_transactions()
    out_path = os.path.join("data", "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} transactions -> {out_path}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}% of data)")
