"""
Responsible Gambling Early Warning System
train.py — Model training and batch-scoring artefacts.
"""

import os

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .data import (
    FEATURE_NAMES,
    INTERVENTIONS,
    assign_risk_tier,
    generate_customers,
    get_or_create_training_data,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

# rg-warning/data/  (parent of the model package, then /data)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build() -> dict:
    """
    Train an XGBClassifier on 1 000 synthetic customers (seed=42).

    Returns
    -------
    dict with keys:
        model           – fitted XGBClassifier
        X_train         – training feature DataFrame
        shap_explainer  – shap.TreeExplainer fitted on the model
        metrics         – dict of evaluation metrics
    """
    df = get_or_create_training_data(DATA_DIR, n=1000, seed=42)

    X = df[FEATURE_NAMES]
    y = df["at_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "auc":          roc_auc_score(y_test, y_prob),
        "precision":    precision_score(y_test, y_pred, zero_division=0),
        "recall":       recall_score(y_test, y_pred, zero_division=0),
        "f1":           f1_score(y_test, y_pred, zero_division=0),
        "n_train":      len(X_train),
        "n_test":       len(X_test),
        "at_risk_rate": float(y.mean()),
    }

    shap_explainer = shap.TreeExplainer(model)

    return {
        "model":          model,
        "X_train":        X_train,
        "shap_explainer": shap_explainer,
        "metrics":        metrics,
    }


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_batch(artifact: dict, n: int = 40, seed: int | None = None) -> dict:
    """
    Score a fresh batch of *n* synthetic customers.

    Parameters
    ----------
    artifact : dict
        Output of :func:`build`.
    n : int
        Number of customers to score.
    seed : int or None
        RNG seed.  If None a random seed is chosen.

    Returns
    -------
    dict with keys:
        df          – scored DataFrame (customer_id, features, risk_prob,
                      risk_tier, intervention, top_driver)
        X_raw       – raw feature DataFrame (n × len(FEATURE_NAMES))
        shap_values – numpy array of SHAP values (n × len(FEATURE_NAMES))
        seed        – seed that was used
    """
    if seed is None:
        seed = int(np.random.randint(1000, 99999))

    model: XGBClassifier       = artifact["model"]
    shap_explainer: shap.TreeExplainer = artifact["shap_explainer"]

    raw_df = generate_customers(n, seed)

    # Re-format customer IDs to batch-specific scheme
    raw_df["customer_id"] = [
        f"PLY-{seed % 10_000:04d}-{i:03d}" for i in range(n)
    ]

    X_raw = raw_df[FEATURE_NAMES].copy()

    risk_prob = model.predict_proba(X_raw)[:, 1]

    df = raw_df[["customer_id"] + FEATURE_NAMES].copy()
    df["risk_prob"] = risk_prob

    df["risk_tier"]     = df["risk_prob"].apply(assign_risk_tier)
    df["intervention"]  = df["risk_tier"].map(
        lambda tier: INTERVENTIONS[tier]["action"]
    )

    # SHAP explanations
    shap_values = shap_explainer.shap_values(X_raw)

    # If the explainer returns a list (binary classification), take class-1 values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Top driver per customer: feature with the highest absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_driver_indices = abs_shap.argmax(axis=1)
    df["top_driver"] = [FEATURE_NAMES[idx] for idx in top_driver_indices]

    return {
        "df":          df,
        "X_raw":       X_raw,
        "shap_values": shap_values,
        "seed":        seed,
    }
