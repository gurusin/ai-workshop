import os

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest

from .data import (
    FEATURE_NAMES,
    FRAUD_TYPES,
    classify_fraud_type,
    generate_customers,
    get_or_create_training_data,
)

# ── Path to cached training data ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def build() -> dict:
    """
    Train an IsolationForest anomaly detector on betting behaviour features.

    Returns
    -------
    dict with keys:
        model, X_train, score_min, score_range, p70, p85,
        shap_explainer (or None), metrics
    """
    df = get_or_create_training_data(DATA_DIR, n=1000, seed=42)
    X: np.ndarray = df[FEATURE_NAMES].values

    # ── Fit model ────────────────────────────────────────────────────────────
    model = IsolationForest(
        n_estimators=200,
        contamination=0.12,
        random_state=42,
        max_samples="auto",
    )
    model.fit(X)

    # ── Normalise scores to [0, 1] where 1 = most anomalous ─────────────────
    raw_scores: np.ndarray = model.score_samples(X)   # negative; lower = more anomalous
    score_min   = float(raw_scores.min())
    score_range = float(raw_scores.max() - raw_scores.min())
    anomaly_score = 1.0 - (raw_scores - score_min) / score_range

    # ── Percentile thresholds for tier assignment ─────────────────────────────
    p70 = float(np.percentile(anomaly_score, 70))
    p85 = float(np.percentile(anomaly_score, 85))

    # ── SHAP explainer ────────────────────────────────────────────────────────
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = None

    metrics = {
        "n_train":      len(X),
        "contamination": 0.12,
        "n_features":   len(FEATURE_NAMES),
    }

    return {
        "model":          model,
        "X_train":        X,
        "score_min":      score_min,
        "score_range":    score_range,
        "p70":            p70,
        "p85":            p85,
        "shap_explainer": explainer,
        "metrics":        metrics,
    }


def score_batch(artifact: dict, n: int = 40, seed: int | None = None) -> dict:
    """
    Score a fresh batch of synthetic customers with the trained artifact.

    Parameters
    ----------
    artifact : dict
        Output of build().
    n : int
        Number of customers to score.
    seed : int or None
        RNG seed; randomly chosen if None.

    Returns
    -------
    dict with keys: df, X_raw, shap_values (or None), seed
    """
    if seed is None:
        seed = int(np.random.randint(1000, 99999))

    # ── Generate & relabel customers ─────────────────────────────────────────
    df = generate_customers(n=n, seed=seed)
    prefix = f"BET-{seed % 10000:04d}"
    df["customer_id"] = [f"{prefix}-{i:03d}" for i in range(n)]

    X_score: np.ndarray = df[FEATURE_NAMES].values

    # ── Score ─────────────────────────────────────────────────────────────────
    model        = artifact["model"]
    score_min    = artifact["score_min"]
    score_range  = artifact["score_range"]
    p70          = artifact["p70"]
    p85          = artifact["p85"]
    X_train      = artifact["X_train"]
    explainer    = artifact["shap_explainer"]

    raw_scores    = model.score_samples(X_score)
    anomaly_score = 1.0 - (raw_scores - score_min) / score_range
    anomaly_score = np.clip(anomaly_score, 0.0, 1.0)

    df["anomaly_score"] = anomaly_score

    # ── Risk tier ─────────────────────────────────────────────────────────────
    def _tier(s: float) -> str:
        if s >= p85:
            return "High Risk"
        if s >= p70:
            return "Medium Risk"
        return "Low Risk"

    df["risk_tier"] = [_tier(s) for s in anomaly_score]

    # ── Fraud type classification ─────────────────────────────────────────────
    df["fraud_type"] = [
        classify_fraud_type(row, anomaly_score=row["anomaly_score"])
        for _, row in df.iterrows()
    ]

    # ── Top driver ────────────────────────────────────────────────────────────
    shap_values = None

    if explainer is not None:
        try:
            shap_values = explainer.shap_values(X_score)
            # shap_values shape: (n_samples, n_features)
            sv = shap_values if shap_values.ndim == 2 else shap_values[:, :, 0]
            top_drivers = [
                FEATURE_NAMES[int(np.argmax(np.abs(sv[i])))]
                for i in range(n)
            ]
        except Exception:
            shap_values = None
            top_drivers = None
    else:
        top_drivers = None

    if top_drivers is None:
        # Fallback: feature with highest absolute z-score vs training mean/std
        train_mean = X_train.mean(axis=0)
        train_std  = X_train.std(axis=0) + 1e-9
        z_scores   = np.abs((X_score - train_mean) / train_std)
        top_drivers = [FEATURE_NAMES[int(np.argmax(z_scores[i]))] for i in range(n)]

    df["top_driver"] = top_drivers

    # ── Convenience percentage column ─────────────────────────────────────────
    df["anomaly_pct"] = (df["anomaly_score"] * 100).round(1)

    return {
        "df":          df,
        "X_raw":       X_score,
        "shap_values": shap_values,
        "seed":        seed,
    }
