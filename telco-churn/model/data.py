import os
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "days_since_last_call",
    "days_since_last_data",
    "data_gb_last_30d",
    "data_gb_prev_30d",
    "calls_last_30d",
    "calls_prev_30d",
    "sms_last_30d",
    "avg_monthly_spend",
    "tenure_months",
    "contract_months_remaining",
    "support_tickets_90d",
    "payment_delays_12m",
    "plan_changed_6m",
    "num_lines",
    "roaming_days_30d",
]

FEATURE_DISPLAY = {
    "days_since_last_call":      "Days Since Last Call",
    "days_since_last_data":      "Days Since Last Data Use",
    "data_gb_last_30d":          "Data Usage GB (Last 30d)",
    "data_gb_prev_30d":          "Data Usage GB (Prior 30d)",
    "calls_last_30d":            "Calls (Last 30d)",
    "calls_prev_30d":            "Calls (Prior 30d)",
    "sms_last_30d":              "SMS Sent (Last 30d)",
    "avg_monthly_spend":         "Avg Monthly Spend ($)",
    "tenure_months":             "Tenure (Months)",
    "contract_months_remaining": "Contract Months Remaining",
    "support_tickets_90d":       "Support Tickets (90d)",
    "payment_delays_12m":        "Payment Delays (12m)",
    "plan_changed_6m":           "Plan Changed Last 6m",
    "num_lines":                 "Number of Lines",
    "roaming_days_30d":          "Roaming Days (30d)",
}

FEATURE_RANGES = {
    "days_since_last_call":      (0, 45, 1),
    "days_since_last_data":      (0, 30, 1),
    "data_gb_last_30d":          (0.5, 50.0, 0.5),
    "data_gb_prev_30d":          (0.5, 50.0, 0.5),
    "calls_last_30d":            (0, 200, 5),
    "calls_prev_30d":            (0, 200, 5),
    "sms_last_30d":              (0, 150, 5),
    "avg_monthly_spend":         (15.0, 120.0, 5.0),
    "tenure_months":             (1, 72, 1),
    "contract_months_remaining": (0, 24, 1),
    "support_tickets_90d":       (0, 8, 1),
    "payment_delays_12m":        (0, 6, 1),
    "plan_changed_6m":           (0, 1, 1),
    "num_lines":                 (1, 5, 1),
    "roaming_days_30d":          (0, 20, 1),
}

CAMPAIGNS = {
    "VIP Retention":    {"color": "#5b21b6", "bg": "#ede9fe", "icon": "👑",
                         "desc": "Personal callback + exclusive loyalty deal for high-value churners"},
    "Win-Back Offer":   {"color": "#b91c1c", "bg": "#fee2e2", "icon": "🎁",
                         "desc": "Promotional data bundle or tariff discount to reverse exit intent"},
    "Early Renewal":    {"color": "#0c4a6e", "bg": "#e0f2fe", "icon": "🔒",
                         "desc": "Lock in current rate with incentive before contract window closes"},
    "Service Recovery": {"color": "#92400e", "bg": "#fef3c7", "icon": "🛠️",
                         "desc": "Proactive outreach + service credit for high-friction customers"},
    "Renewal Reminder": {"color": "#065f46", "bg": "#d1fae5", "icon": "📅",
                         "desc": "Contract renewal nudge via SMS + email — time-sensitive offer"},
    "Data Upsell":      {"color": "#1e40af", "bg": "#dbeafe", "icon": "📶",
                         "desc": "Upgrade to higher data tier at a reduced introductory rate"},
    "Engagement Boost": {"color": "#065f46", "bg": "#ecfdf5", "icon": "⚡",
                         "desc": "Usage incentive or add-on feature trial for cooling engagement"},
    "Loyalty Reward":   {"color": "#374151", "bg": "#f3f4f6", "icon": "🏅",
                         "desc": "Loyalty points, referral bonus, or value-add for retained base"},
}


def assign_campaign(row) -> str:
    risk = str(row["risk_segment"])
    arpu = float(row["avg_monthly_spend"])
    contract = float(row["contract_months_remaining"])
    tickets = float(row["support_tickets_90d"])
    data_last = float(row["data_gb_last_30d"])
    data_prev = float(row["data_gb_prev_30d"])
    data_trend = data_last / (data_prev + 0.5)

    if risk == "High":
        if contract <= 1:
            return "Early Renewal"
        if arpu >= 70:
            return "VIP Retention"
        if tickets >= 3:
            return "Service Recovery"
        return "Win-Back Offer"
    if risk == "Medium":
        if contract <= 2:
            return "Renewal Reminder"
        if data_trend < 0.65:
            return "Data Upsell"
        return "Engagement Boost"
    return "Loyalty Reward"


def generate_customers(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 72, n)
    contract = rng.integers(0, 25, n)
    days_call = rng.integers(0, 45, n)
    days_data = rng.integers(0, 30, n)
    data_last = rng.uniform(0.5, 50.0, n).round(2)
    data_prev = rng.uniform(0.5, 50.0, n).round(2)
    calls_last = rng.integers(0, 200, n)
    calls_prev = rng.integers(0, 200, n)
    sms = rng.integers(0, 150, n)
    arpu = rng.uniform(15.0, 120.0, n).round(2)
    tickets = rng.integers(0, 9, n)
    delays = rng.integers(0, 7, n)
    plan_changed = rng.integers(0, 2, n)
    num_lines = rng.integers(1, 6, n)
    roaming = rng.integers(0, 21, n)

    data_trend = np.clip(data_last / (data_prev + 0.5), 0.0, 4.0)
    call_trend = np.clip(calls_last / (calls_prev + 0.5), 0.0, 4.0)

    logit = (
        0.03 * days_call
        + 0.04 * days_data
        - 0.03 * data_trend
        - 0.02 * call_trend
        + 0.12 * tickets
        + 0.10 * delays
        - 0.015 * tenure
        - 0.02 * contract
        - 0.006 * arpu
        - 0.15 * num_lines
        - 0.12 * plan_changed
        - 0.90
        + rng.normal(0, 0.35, n)
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))

    return pd.DataFrame({
        "customer_id":               [f"MVNO-{i:05d}" for i in range(1, n + 1)],
        "days_since_last_call":      days_call,
        "days_since_last_data":      days_data,
        "data_gb_last_30d":          data_last,
        "data_gb_prev_30d":          data_prev,
        "calls_last_30d":            calls_last,
        "calls_prev_30d":            calls_prev,
        "sms_last_30d":              sms,
        "avg_monthly_spend":         arpu,
        "tenure_months":             tenure,
        "contract_months_remaining": contract,
        "support_tickets_90d":       tickets,
        "payment_delays_12m":        delays,
        "plan_changed_6m":           plan_changed,
        "num_lines":                 num_lines,
        "roaming_days_30d":          roaming,
        "churned":                   (churn_prob > 0.5).astype(int),
    })


def get_or_create_training_data(data_dir: str, n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Load training CSV if it exists, otherwise generate and save it."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "training_data.csv")
    if not os.path.exists(path):
        df = generate_customers(n=n, seed=seed)
        df.to_csv(path, index=False)
    return pd.read_csv(path)
