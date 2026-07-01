DEMO_MESSAGE = (
    "Big game tonight! Your personalised multi-bet is ready — we've picked the best odds just for you. "
    "Deposit $50 now and we'll match it. Don't miss out, this offer expires at 8PM!"
)

DEMO_USERS = [
    {
        "name": "James",
        "age": 34,
        "risk_profile": "standard",
        "deposit_count_today": 1,
        "description": "Casual weekend bettor, no flags",
    },
    {
        "name": "Sarah",
        "age": 29,
        "risk_profile": "at_risk",
        "deposit_count_today": 4,
        "description": "Deposit frequency increasing, RG team flagged",
    },
    {
        "name": "Michael",
        "age": 45,
        "risk_profile": "self_excluded",
        "deposit_count_today": 0,
        "description": "Voluntarily self-excluded 3 months ago",
    },
    {
        "name": "Priya",
        "age": 27,
        "risk_profile": "new_user",
        "deposit_count_today": 1,
        "description": "Registered 5 days ago, first deposit",
    },
    {
        "name": "David",
        "age": 38,
        "risk_profile": "vip",
        "deposit_count_today": 2,
        "description": "High-value VIP, consistent bettor, no behavioural flags",
    },
]

RISK_PROFILE_LABELS = {
    "standard": "Standard",
    "at_risk": "At-Risk",
    "self_excluded": "Self-Excluded",
    "new_user": "New User",
    "vip": "VIP",
}

RISK_PROFILE_COLORS = {
    "standard": "#6c757d",
    "at_risk": "#fd7e14",
    "self_excluded": "#dc3545",
    "new_user": "#0d6efd",
    "vip": "#6f42c1",
}
