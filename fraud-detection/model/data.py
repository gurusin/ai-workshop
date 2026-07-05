import os

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "win_rate_30d",
    "avg_stake",
    "stake_cv",
    "avg_odds",
    "bet_timing_mins",
    "round_stake_pct",
    "market_concentration",
    "account_age_days",
    "withdrawal_deposit_ratio",
    "max_bets_single_event",
    "win_streak_max",
    "unique_sports_count",
    "bet_count_30d",
    "deposit_round_pct",
    "late_line_move_pct",
]

FEATURE_DISPLAY = {
    "win_rate_30d":             "Win Rate (30d)",
    "avg_stake":                "Avg Stake ($)",
    "stake_cv":                 "Stake Coefficient of Variation",
    "avg_odds":                 "Avg Odds Backed",
    "bet_timing_mins":          "Avg Mins Before Event (Bet Timing)",
    "round_stake_pct":          "Round Stake %",
    "market_concentration":     "Market Concentration",
    "account_age_days":         "Account Age (Days)",
    "withdrawal_deposit_ratio": "Withdrawal / Deposit Ratio",
    "max_bets_single_event":    "Max Bets on Single Event",
    "win_streak_max":           "Max Win Streak",
    "unique_sports_count":      "Unique Sports Count",
    "bet_count_30d":            "Bet Count (30d)",
    "deposit_round_pct":        "Round Deposit %",
    "late_line_move_pct":       "Late Line Move %",
}

FEATURE_RANGES = {
    "win_rate_30d":             (0.20, 0.95, 0.01),
    "avg_stake":                (5.0, 500.0, 5.0),
    "stake_cv":                 (0.01, 2.0, 0.01),
    "avg_odds":                 (1.2, 8.0, 0.1),
    "bet_timing_mins":          (0, 180, 1),
    "round_stake_pct":          (0.0, 1.0, 0.05),
    "market_concentration":     (0.1, 1.0, 0.05),
    "account_age_days":         (1, 1500, 1),
    "withdrawal_deposit_ratio": (0.0, 1.0, 0.05),
    "max_bets_single_event":    (1, 10, 1),
    "win_streak_max":           (1, 20, 1),
    "unique_sports_count":      (1, 8, 1),
    "bet_count_30d":            (2, 300, 1),
    "deposit_round_pct":        (0.0, 1.0, 0.05),
    "late_line_move_pct":       (0.0, 1.0, 0.05),
}

FRAUD_TYPES = {
    "Match-Fixing Risk": {"color": "#7f1d1d", "bg": "#fecaca",  "icon": "🎯"},
    "Bot / Syndicate":   {"color": "#1e3a5f", "bg": "#dbeafe",  "icon": "🤖"},
    "Bonus Abuse":       {"color": "#92400e", "bg": "#fef3c7",  "icon": "🎁"},
    "Arbitrage":         {"color": "#5b21b6", "bg": "#ede9fe",  "icon": "⚖️"},
    "Unusual Pattern":   {"color": "#374151", "bg": "#f3f4f6",  "icon": "❓"},
    "Normal":            {"color": "#166534", "bg": "#dcfce7",  "icon": "✅"},
}


def classify_fraud_type(row, anomaly_score: float = 0.0) -> str:
    """Rule-based fraud type classification. Returns a key from FRAUD_TYPES."""
    if row["win_rate_30d"] > 0.68 and row["bet_timing_mins"] < 8:
        return "Match-Fixing Risk"
    if row["stake_cv"] < 0.15 and row["bet_count_30d"] > 80:
        return "Bot / Syndicate"
    if row["account_age_days"] < 30 and row["withdrawal_deposit_ratio"] > 0.80:
        return "Bonus Abuse"
    if row["stake_cv"] < 0.25 and row["win_rate_30d"] > 0.60 and row["unique_sports_count"] <= 2:
        return "Arbitrage"
    if anomaly_score > 0.6:
        return "Unusual Pattern"
    return "Normal"


def generate_customers(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic mix of normal and anomalous betting customers."""
    rng = np.random.default_rng(seed)

    # ── Baseline normal distributions ────────────────────────────────────────
    win_rate_30d             = rng.uniform(0.35, 0.55, n)
    avg_stake                = rng.uniform(10.0, 150.0, n)
    stake_cv                 = rng.uniform(0.4, 1.8, n)
    avg_odds                 = rng.uniform(1.5, 4.5, n)
    bet_timing_mins          = rng.uniform(30.0, 180.0, n)
    round_stake_pct          = rng.uniform(0.3, 0.7, n)
    market_concentration     = rng.uniform(0.2, 0.6, n)
    account_age_days         = rng.uniform(90.0, 1500.0, n)
    withdrawal_deposit_ratio = rng.uniform(0.1, 0.6, n)
    max_bets_single_event    = rng.integers(1, 4, n).astype(float)   # 1–3
    win_streak_max           = rng.integers(1, 7, n).astype(float)   # 1–6
    unique_sports_count      = rng.integers(2, 8, n).astype(float)   # 2–7
    bet_count_30d            = rng.uniform(5.0, 60.0, n)
    deposit_round_pct        = rng.uniform(0.3, 0.7, n)
    late_line_move_pct       = rng.uniform(0.1, 0.4, n)

    # ── Inject anomalous patterns by index ───────────────────────────────────
    for i in range(n):
        # Match-fixing: high win rate + very early bets (low timing)
        if i % 7 == 0:
            win_rate_30d[i]    = rng.uniform(0.70, 0.90)
            bet_timing_mins[i] = rng.uniform(0.0, 8.0)

        # Bot / Syndicate: very consistent stakes + high volume
        if i % 11 == 0:
            stake_cv[i]      = rng.uniform(0.02, 0.12)
            bet_count_30d[i] = rng.uniform(80.0, 250.0)

        # Bonus Abuse: new account + high withdrawal ratio
        if i % 13 == 0:
            account_age_days[i]         = rng.uniform(3.0, 25.0)
            withdrawal_deposit_ratio[i] = rng.uniform(0.82, 0.98)

        # Arbitrage: very consistent stakes + solid win rate on 1–2 sports
        if i % 17 == 0:
            stake_cv[i]           = rng.uniform(0.05, 0.20)
            win_rate_30d[i]       = rng.uniform(0.62, 0.72)
            unique_sports_count[i] = float(rng.integers(1, 3))   # 1 or 2

    # ── Clip all values to declared valid ranges ──────────────────────────────
    win_rate_30d             = np.clip(win_rate_30d.round(3),             0.20, 0.95)
    avg_stake                = np.clip(avg_stake.round(2),                5.0, 500.0)
    stake_cv                 = np.clip(stake_cv.round(3),                 0.01, 2.0)
    avg_odds                 = np.clip(avg_odds.round(2),                 1.2, 8.0)
    bet_timing_mins          = np.clip(bet_timing_mins.round(1),          0.0, 180.0)
    round_stake_pct          = np.clip(round_stake_pct.round(3),          0.0, 1.0)
    market_concentration     = np.clip(market_concentration.round(3),     0.1, 1.0)
    account_age_days         = np.clip(account_age_days.round(0),         1.0, 1500.0)
    withdrawal_deposit_ratio = np.clip(withdrawal_deposit_ratio.round(3), 0.0, 1.0)
    max_bets_single_event    = np.clip(max_bets_single_event,             1.0, 10.0)
    win_streak_max           = np.clip(win_streak_max,                    1.0, 20.0)
    unique_sports_count      = np.clip(unique_sports_count,               1.0, 8.0)
    bet_count_30d            = np.clip(bet_count_30d.round(0),            2.0, 300.0)
    deposit_round_pct        = np.clip(deposit_round_pct.round(3),        0.0, 1.0)
    late_line_move_pct       = np.clip(late_line_move_pct.round(3),       0.0, 1.0)

    return pd.DataFrame({
        "customer_id":              [f"BET-{i:05d}" for i in range(n)],
        "win_rate_30d":             win_rate_30d,
        "avg_stake":                avg_stake,
        "stake_cv":                 stake_cv,
        "avg_odds":                 avg_odds,
        "bet_timing_mins":          bet_timing_mins,
        "round_stake_pct":          round_stake_pct,
        "market_concentration":     market_concentration,
        "account_age_days":         account_age_days,
        "withdrawal_deposit_ratio": withdrawal_deposit_ratio,
        "max_bets_single_event":    max_bets_single_event,
        "win_streak_max":           win_streak_max,
        "unique_sports_count":      unique_sports_count,
        "bet_count_30d":            bet_count_30d,
        "deposit_round_pct":        deposit_round_pct,
        "late_line_move_pct":       late_line_move_pct,
    })


def get_or_create_training_data(
    data_dir: str,
    n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Load cached training CSV if it exists; otherwise generate and save it."""
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "fraud_training.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    df = generate_customers(n=n, seed=seed)
    df.to_csv(csv_path, index=False)
    return df
