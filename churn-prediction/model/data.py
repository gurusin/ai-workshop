import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "days_since_last_bet",
    "bets_last_30d",
    "bets_prev_30d",
    "avg_stake_last_30d",
    "avg_stake_prev_30d",
    "sessions_last_30d",
    "win_rate",
    "tenure_days",
    "deposit_count_last_30d",
    "live_bet_pct",
    "sport_diversity",
    "promo_redemption_rate",
    "support_contacts_last_90d",
    "bet_frequency_trend",
    "stake_trend",
]

FEATURE_DISPLAY = {
    "days_since_last_bet": "Days Since Last Bet",
    "bets_last_30d": "Bets (Last 30d)",
    "bets_prev_30d": "Bets (Prior 30d)",
    "avg_stake_last_30d": "Avg Stake Last 30d ($)",
    "avg_stake_prev_30d": "Avg Stake Prior 30d ($)",
    "sessions_last_30d": "Sessions (Last 30d)",
    "win_rate": "Win Rate",
    "tenure_days": "Account Tenure (Days)",
    "deposit_count_last_30d": "Deposits (Last 30d)",
    "live_bet_pct": "Live Bet %",
    "sport_diversity": "Sports Variety",
    "promo_redemption_rate": "Promo Redemption Rate",
    "support_contacts_last_90d": "Support Contacts (90d)",
    "bet_frequency_trend": "Bet Frequency Trend",
    "stake_trend": "Stake Size Trend",
}

FEATURE_RANGES = {
    "days_since_last_bet":     (0, 62, 1),
    "bets_last_30d":           (0, 55, 1),
    "bets_prev_30d":           (0, 55, 1),
    "avg_stake_last_30d":      (5.0, 250.0, 5.0),
    "avg_stake_prev_30d":      (5.0, 250.0, 5.0),
    "sessions_last_30d":       (0, 45, 1),
    "win_rate":                (0.28, 0.72, 0.01),
    "tenure_days":             (30, 1825, 30),
    "deposit_count_last_30d":  (0, 12, 1),
    "live_bet_pct":            (0.0, 1.0, 0.05),
    "sport_diversity":         (1, 8, 1),
    "promo_redemption_rate":   (0.0, 1.0, 0.05),
    "support_contacts_last_90d": (0, 5, 1),
    "bet_frequency_trend":     (0.0, 3.0, 0.05),
    "stake_trend":             (0.0, 3.0, 0.05),
}


def generate_customers(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(30, 1825, n)
    days_since = rng.integers(0, 62, n)
    bets_last = rng.integers(0, 55, n)
    bets_prev = rng.integers(0, 55, n)
    stake_last = rng.uniform(5.0, 250.0, n).round(2)
    stake_prev = rng.uniform(5.0, 250.0, n).round(2)
    sessions = rng.integers(0, 45, n)
    win_rate = rng.uniform(0.28, 0.72, n).round(3)
    deposits = rng.integers(0, 12, n)
    live_pct = rng.uniform(0.0, 1.0, n).round(3)
    sport_div = rng.integers(1, 9, n)
    promo = rng.uniform(0.0, 1.0, n).round(3)
    support = rng.integers(0, 6, n)

    freq_trend = np.clip(bets_last / (bets_prev + 0.5), 0.0, 3.0).round(3)
    stake_trend = np.clip(stake_last / (stake_prev + 1.0), 0.0, 3.0).round(3)

    # Churn ground truth: probabilistic logistic — recency & declining frequency drive churn
    logit = (
        0.09 * days_since
        - 0.07 * bets_last
        - 0.04 * sessions
        + 0.20 * support
        - 0.14 * freq_trend
        - 0.001 * tenure
        - 0.55
        + rng.normal(0, 0.35, n)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))

    return pd.DataFrame({
        "customer_id":              [f"CST-{i:04d}" for i in range(1, n + 1)],
        "days_since_last_bet":      days_since,
        "bets_last_30d":            bets_last,
        "bets_prev_30d":            bets_prev,
        "avg_stake_last_30d":       stake_last,
        "avg_stake_prev_30d":       stake_prev,
        "sessions_last_30d":        sessions,
        "win_rate":                 win_rate,
        "tenure_days":              tenure,
        "deposit_count_last_30d":   deposits,
        "live_bet_pct":             live_pct,
        "sport_diversity":          sport_div,
        "promo_redemption_rate":    promo,
        "support_contacts_last_90d": support,
        "bet_frequency_trend":      freq_trend,
        "stake_trend":              stake_trend,
        "churned":                  (churn_prob > 0.5).astype(int),
    })
