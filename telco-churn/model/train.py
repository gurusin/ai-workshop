import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import shap
from lime import lime_tabular

from .data import FEATURE_NAMES, CAMPAIGNS, assign_campaign, get_or_create_training_data, generate_customers

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def build() -> dict:
    """Train on the saved CSV (Dataset 1). Returns locked model artifacts."""
    df_train_full = get_or_create_training_data(DATA_DIR, n=800, seed=42)
    X_all = df_train_full[FEATURE_NAMES].values
    y_all = df_train_full["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "auc":        round(roc_auc_score(y_test, y_prob), 3),
        "precision":  round(precision_score(y_test, y_pred, zero_division=0), 3),
        "recall":     round(recall_score(y_test, y_pred, zero_division=0), 3),
        "f1":         round(f1_score(y_test, y_pred, zero_division=0), 3),
        "n_train":    int(len(X_train)),
        "n_test":     int(len(X_test)),
        "churn_rate": round(float(y_all.mean()), 3),
        "training_file": os.path.join(DATA_DIR, "training_data.csv"),
    }

    shap_explainer = shap.TreeExplainer(model)
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=FEATURE_NAMES,
        class_names=["Retained", "Churned"],
        mode="classification",
        random_state=42,
    )

    return {
        "model":          model,
        "X_train":        X_train,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "metrics":        metrics,
    }


def score_batch(artifact: dict, n: int = 30, seed: int = None) -> dict:
    """Score a fresh random batch. Called on each UI invocation — never cached."""
    if seed is None:
        seed = int(np.random.randint(0, 99999))

    model = artifact["model"]
    shap_explainer = artifact["shap_explainer"]

    df = generate_customers(n=n, seed=seed)
    df["customer_id"] = [f"MVNO-{seed % 10000:04d}-{i:03d}" for i in range(1, n + 1)]

    X_score = df[FEATURE_NAMES].values
    proba = model.predict_proba(X_score)[:, 1]

    df["churn_probability"] = proba.round(4)
    df["churn_pct"] = (proba * 100).round(1)
    df["predicted_churn"] = (proba > 0.5).astype(int)

    p40 = float(np.percentile(proba, 40))
    p70 = float(np.percentile(proba, 70))
    df["risk_segment"] = pd.cut(
        proba,
        bins=[0.0, p40, p70, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    df["campaign"] = df.apply(assign_campaign, axis=1)

    shap_values = shap_explainer(df[FEATURE_NAMES])
    shap_matrix = shap_values.values
    top_idx = np.abs(shap_matrix).argmax(axis=1)
    df["top_driver"] = [FEATURE_NAMES[i] for i in top_idx]

    return {
        "df":          df,
        "X_raw":       X_score,
        "shap_values": shap_values,
        "seed":        seed,
    }
