"""
preprocess.py
-------------
Shared preprocessing utilities used by training + dashboard scripts.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = None  # will be set dynamically based on dataset


def load_data(path="data/transactions.csv"):
    df = pd.read_csv(path)
    return df


def get_feature_columns(df):
    """All columns except ID and target are model features."""
    drop_cols = ["TransactionID", "Class"]
    return [c for c in df.columns if c not in drop_cols]


def scale_features(df, feature_cols, scaler=None):
    """
    Scales Amount + Time + anonymized columns using StandardScaler.
    Returns scaled dataframe (features only) and the fitted scaler.
    """
    X = df[feature_cols].copy()
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    return X_scaled_df, scaler
