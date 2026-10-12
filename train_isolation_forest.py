"""
train_isolation_forest.py
--------------------------
Trains an Isolation Forest for UNSUPERVISED anomaly detection on the
transactions dataset, then (for comparison) trains a supervised
Random Forest using SMOTE to fix class imbalance.

Saves:
    models/isolation_forest.joblib
    models/scaler.joblib
    models/rf_smote.joblib
    outputs/metrics.json
    outputs/predictions.csv   (used by the Streamlit dashboard)

Run:
    python src/train_isolation_forest.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE

from preprocess import load_data, get_feature_columns, scale_features

RANDOM_SEED = 42


def train():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    df = load_data("data/transactions.csv")
    feature_cols = get_feature_columns(df)

    X_scaled, scaler = scale_features(df, feature_cols)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    # ------------------------------------------------------------
    # 1) UNSUPERVISED: Isolation Forest (anomaly detection)
    #    contamination = expected proportion of fraud in the data
    # ------------------------------------------------------------
    contamination = max(y_train.mean(), 0.001)
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    iso_forest.fit(X_train)

    # IsolationForest outputs -1 (anomaly) / 1 (normal) -> map to 1/0
    raw_preds = iso_forest.predict(X_test)
    iso_preds = np.where(raw_preds == -1, 1, 0)

    # anomaly score: lower score_samples = more abnormal
    iso_scores = -iso_forest.score_samples(X_test)  # higher = more suspicious

    iso_report = classification_report(y_test, iso_preds, output_dict=True)
    iso_cm = confusion_matrix(y_test, iso_preds).tolist()
    iso_auc = roc_auc_score(y_test, iso_scores)

    print("=== Isolation Forest (Unsupervised Anomaly Detection) ===")
    print(classification_report(y_test, iso_preds))
    print(f"ROC-AUC (using anomaly score): {iso_auc:.4f}")

    # ------------------------------------------------------------
    # 2) CLASS IMBALANCE TECHNIQUE: SMOTE + supervised Random Forest
    #    (shown as a stronger benchmark / comparison model)
    # ------------------------------------------------------------
    smote = SMOTE(random_state=RANDOM_SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"\nBefore SMOTE: {y_train.value_counts().to_dict()}")
    print(f"After SMOTE:  {pd.Series(y_train_res).value_counts().to_dict()}")

    rf = RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1, class_weight=None
    )
    rf.fit(X_train_res, y_train_res)

    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    rf_report = classification_report(y_test, rf_preds, output_dict=True)
    rf_cm = confusion_matrix(y_test, rf_preds).tolist()
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_ap = average_precision_score(y_test, rf_probs)

    print("\n=== Random Forest + SMOTE (Supervised, class-imbalance handled) ===")
    print(classification_report(y_test, rf_preds))
    print(f"ROC-AUC: {rf_auc:.4f} | Average Precision (PR-AUC): {rf_ap:.4f}")

    # ------------------------------------------------------------
    # Save models + scaler
    # ------------------------------------------------------------
    joblib.dump(iso_forest, "models/isolation_forest.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(rf, "models/rf_smote.joblib")

    # ------------------------------------------------------------
    # Save metrics for reference
    # ------------------------------------------------------------
    metrics = {
        "isolation_forest": {
            "roc_auc": iso_auc,
            "confusion_matrix": iso_cm,
            "classification_report": iso_report,
        },
        "random_forest_smote": {
            "roc_auc": rf_auc,
            "average_precision": rf_ap,
            "confusion_matrix": rf_cm,
            "classification_report": rf_report,
        },
    }
    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------
    # Save a predictions file the DASHBOARD will read to show alerts
    # ------------------------------------------------------------
    results = df.loc[X_test.index].copy()
    results["iso_forest_pred"] = iso_preds
    results["iso_forest_score"] = iso_scores
    results["rf_smote_pred"] = rf_preds
    results["rf_smote_prob"] = rf_probs
    results.to_csv("outputs/predictions.csv", index=False)

    print("\nSaved models -> models/")
    print("Saved metrics -> outputs/metrics.json")
    print("Saved predictions (for dashboard) -> outputs/predictions.csv")


if __name__ == "__main__":
    train()
