"""
Responsible Gambling Early Warning System
data.py — Synthetic data generation and shared constants.
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "loss_chasing_score",
    "late_night_session_pct",
    "avg_session_mins",
    "deposit_freq_7d",
    "deposit_amount_increase_pct",
    "withdrawal_reversal_count",
    "consecutive_days_played",
    "net_loss_30d",
    "reality_check_dismissal_rate",
    "high_volatility_game_pct",
    "stake_variance_norm",
    "rg_tool_usage",
    "time_since_break_days",
    "sessions_per_day",
    "bet_count_7d",
]

FEATURE_DISPLAY = {
    "loss_chasing_score":           "Loss Chasing Score",
    "late_night_session_pct":       "Late-Night Session %",
    "avg_session_mins":             "Avg Session Length (mins)",
    "deposit_freq_7d":              "Deposit Frequency (7d)",
    "deposit_amount_increase_pct":  "Deposit Amount Increase %",
    "withdrawal_reversal_count":    "Withdrawal Reversals",
    "consecutive_days_played":      "Consecutive Days Played",
    "net_loss_30d":                 "Net Loss – Last 30d ($)",
    "reality_check_dismissal_rate": "Reality Check Dismissal Rate",
    "high_volatility_game_pct":     "High-Volatility Game %",
    "stake_variance_norm":          "Stake Variance (normalised)",
    "rg_tool_usage":                "RG Tool Usage",
    "time_since_break_days":        "Days Since Last Break",
    "sessions_per_day":             "Sessions per Day",
    "bet_count_7d":                 "Bet Count (7d)",
}

# (min, max, step) tuples for UI sliders / validation
FEATURE_RANGES = {
    "loss_chasing_score":           (0.0,   1.0,   0.01),
    "late_night_session_pct":       (0.0,   1.0,   0.01),
    "avg_session_mins":             (10,    180,   1),
    "deposit_freq_7d":              (0,     8,     1),
    "deposit_amount_increase_pct":  (0,     100,   1),
    "withdrawal_reversal_count":    (0,     5,     1),
    "consecutive_days_played":      (0,     60,    1),
    "net_loss_30d":                 (0,     500,   1),
    "reality_check_dismissal_rate": (0.0,   1.0,   0.01),
    "high_volatility_game_pct":     (0.0,   1.0,   0.01),
    "stake_variance_norm":          (0.0,   1.0,   0.01),
    "rg_tool_usage":                (0,     1,     1),
    "time_since_break_days":        (0,     60,    1),
    "sessions_per_day":             (0.5,   6.0,   0.1),
    "bet_count_7d":                 (0,     80,    1),
}

# ---------------------------------------------------------------------------
# Intervention tiers
# ---------------------------------------------------------------------------

INTERVENTIONS = {
    "Low": {
        "icon":   "✅",
        "color":  "#166534",
        "bg":     "#dcfce7",
        "action": "Continue monitoring — no intervention required",
    },
    "Medium": {
        "icon":   "⚠️",
        "color":  "#92400e",
        "bg":     "#fef3c7",
        "action": "Send RG information email + offer self-assessment tool",
    },
    "High": {
        "icon":   "🚨",
        "color":  "#b91c1c",
        "bg":     "#fee2e2",
        "action": "Mandatory outreach + offer stake limits + cooling-off period",
    },
    "Critical": {
        "icon":   "🛑",
        "color":  "#7f1d1d",
        "bg":     "#fecaca",
        "action": "Restrict account + mandatory referral to support services",
    },
}

# ---------------------------------------------------------------------------
# Risk-tier helper
# ---------------------------------------------------------------------------

def assign_risk_tier(prob: float) -> str:
    """Map a scalar probability to a named risk tier."""
    if prob >= 0.8:
        return "Critical"
    if prob >= 0.6:
        return "High"
    if prob >= 0.3:
        return "Medium"
    return "Low"


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int, seed: int) -> pd.DataFrame:
    """
    Generate *n* synthetic customer records using *seed* for reproducibility.

    Returns a DataFrame with columns:
        customer_id, <15 features>, at_risk
    """
    rng = np.random.default_rng(seed)

    loss_chasing_score           = rng.uniform(0.0, 1.0, n)
    late_night_session_pct       = rng.uniform(0.0, 1.0, n)
    avg_session_mins             = rng.uniform(10,  180,  n)
    deposit_freq_7d              = rng.integers(0,   9,   n)           # [0, 8]
    deposit_amount_increase_pct  = rng.uniform(0,   100,  n)
    withdrawal_reversal_count    = rng.integers(0,   6,   n)           # [0, 5]
    consecutive_days_played      = rng.integers(0,   61,  n)           # [0, 60]
    net_loss_30d                 = rng.uniform(0,   500,  n)
    reality_check_dismissal_rate = rng.uniform(0.0, 1.0, n)
    high_volatility_game_pct     = rng.uniform(0.0, 1.0, n)
    stake_variance_norm          = rng.uniform(0.0, 1.0, n)
    rg_tool_usage                = rng.integers(0,   2,   n)           # 0 or 1
    time_since_break_days        = rng.integers(0,   61,  n)           # [0, 60]
    sessions_per_day             = rng.uniform(0.5, 6.0, n)
    bet_count_7d                 = rng.integers(0,   81,  n)           # [0, 80]

    logit = (
        1.5  * loss_chasing_score
        + 1.2  * late_night_session_pct
        + 0.008 * avg_session_mins
        + 0.2  * deposit_freq_7d
        + 0.01 * deposit_amount_increase_pct
        + 0.5  * withdrawal_reversal_count
        + 0.02 * consecutive_days_played
        + 0.002 * net_loss_30d
        + 1.2  * reality_check_dismissal_rate
        + 1.0  * high_volatility_game_pct
        + 0.4  * stake_variance_norm
        - 1.5  * rg_tool_usage
        + 0.02 * time_since_break_days
        + 0.3  * sessions_per_day
        + 0.02 * bet_count_7d
        - 10.0
        + rng.normal(0, 0.4, n)
    )

    churn_prob = _sigmoid(logit)
    at_risk    = (churn_prob > 0.5).astype(int)

    customer_ids = [f"PLY-{i:05d}" for i in range(n)]

    df = pd.DataFrame({
        "customer_id":                  customer_ids,
        "loss_chasing_score":           loss_chasing_score,
        "late_night_session_pct":       late_night_session_pct,
        "avg_session_mins":             avg_session_mins,
        "deposit_freq_7d":              deposit_freq_7d,
        "deposit_amount_increase_pct":  deposit_amount_increase_pct,
        "withdrawal_reversal_count":    withdrawal_reversal_count,
        "consecutive_days_played":      consecutive_days_played,
        "net_loss_30d":                 net_loss_30d,
        "reality_check_dismissal_rate": reality_check_dismissal_rate,
        "high_volatility_game_pct":     high_volatility_game_pct,
        "stake_variance_norm":          stake_variance_norm,
        "rg_tool_usage":                rg_tool_usage,
        "time_since_break_days":        time_since_break_days,
        "sessions_per_day":             sessions_per_day,
        "bet_count_7d":                 bet_count_7d,
        "at_risk":                      at_risk,
    })

    return df


def get_or_create_training_data(
    data_dir: str,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return training data as a DataFrame.

    If *data_dir*/training_data.csv exists it is loaded directly;
    otherwise fresh data is generated, saved, and returned.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "training_data.csv")

    if os.path.exists(path):
        return pd.read_csv(path)

    df = generate_customers(n, seed)
    df.to_csv(path, index=False)
    return df
