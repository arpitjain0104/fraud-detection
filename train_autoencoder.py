"""
train_autoencoder.py
---------------------
OPTIONAL deep-learning approach: trains an AutoEncoder ONLY on normal
(non-fraud) transactions. The idea: the autoencoder learns to
reconstruct normal transactions very well. Fraudulent transactions
(which it has never seen) will have HIGH reconstruction error ->
flagged as anomalies.

This script needs TensorFlow, which is a heavier install. It is kept
separate from train_isolation_forest.py so the core project works
even if you skip this file.

Install (only if you want to run this file):
    pip install tensorflow

Run:
    python src/train_autoencoder.py
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_data, get_feature_columns, scale_features

RANDOM_SEED = 42

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_autoencoder(input_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(16, activation="relu")(inputs)
    x = layers.Dense(8, activation="relu")(x)
    bottleneck = layers.Dense(4, activation="relu")(x)
    x = layers.Dense(8, activation="relu")(bottleneck)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = models.Model(inputs, outputs)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train():
    if not TF_AVAILABLE:
        print("TensorFlow is not installed.")
        print("This script is OPTIONAL. Run:  pip install tensorflow")
        print("Then re-run: python src/train_autoencoder.py")
        return

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_data("data/transactions.csv")
    feature_cols = get_feature_columns(df)
    X_scaled, scaler = scale_features(df, feature_cols)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    # Train ONLY on normal transactions
    X_train_normal = X_train[y_train == 0]

    autoencoder = build_autoencoder(input_dim=X_train_normal.shape[1])
    autoencoder.fit(
        X_train_normal, X_train_normal,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        shuffle=True,
        verbose=1,
    )

    # Reconstruction error on test set
    reconstructions = autoencoder.predict(X_test, verbose=0)
    mse = np.mean(np.square(X_test.values - reconstructions), axis=1)

    # Threshold = 95th percentile of error on normal training data
    train_recon = autoencoder.predict(X_train_normal, verbose=0)
    train_mse = np.mean(np.square(X_train_normal.values - train_recon), axis=1)
    threshold = np.percentile(train_mse, 95)

    preds = (mse > threshold).astype(int)

    print("=== AutoEncoder Anomaly Detection ===")
    print(f"Reconstruction error threshold: {threshold:.4f}")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC (using reconstruction error): {roc_auc_score(y_test, mse):.4f}")

    autoencoder.save("models/autoencoder.keras")
    joblib.dump(threshold, "models/autoencoder_threshold.joblib")
    print("\nSaved model -> models/autoencoder.keras")


if __name__ == "__main__":
    train()
