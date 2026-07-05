import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import shap
from lime import lime_tabular

from .data import FEATURE_NAMES, generate_customers


def build(
    n_train: int = 600,
    n_score: int = 400,
    train_seed: int = 42,
    score_seed: int = 99,
) -> dict:
    # ── Dataset 1: training + held-out test ──────────────────────────────────
    df_train_full = generate_customers(n=n_train, seed=train_seed)
    X_all = df_train_full[FEATURE_NAMES].values
    y_all = df_train_full["churned"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=train_seed, stratify=y_all
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=train_seed,
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
        "n_score":    n_score,
    }

    # Explainers fitted on training data only
    shap_explainer = shap.TreeExplainer(model)
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=FEATURE_NAMES,
        class_names=["Retained", "Churned"],
        mode="classification",
        random_state=train_seed,
    )

    # ── Dataset 2: separate scoring dataset — never seen during training ──────
    df_score = generate_customers(n=n_score, seed=score_seed)
    X_score = df_score[FEATURE_NAMES].values

    proba = model.predict_proba(X_score)[:, 1]
    df_score["churn_probability"] = proba.round(4)
    df_score["churn_pct"] = (proba * 100).round(1)
    df_score["predicted_churn"] = (proba > 0.5).astype(int)

    p40 = float(np.percentile(proba, 40))
    p70 = float(np.percentile(proba, 70))
    df_score["risk_segment"] = pd.cut(
        proba,
        bins=[0.0, p40, p70, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    shap_values = shap_explainer(df_score[FEATURE_NAMES])

    return {
        "model":          model,
        "df":             df_score,          # business dashboard uses scoring data only
        "X":              df_score[FEATURE_NAMES],
        "X_raw":          X_score,
        "X_train":        X_train,           # LIME needs original training distribution
        "shap_values":    shap_values,
        "shap_explainer": shap_explainer,
        "lime_explainer": lime_explainer,
        "metrics":        metrics,
    }
