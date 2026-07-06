import os

ROOT = os.path.dirname(os.path.abspath(__file__))
IS_CLOUD = os.environ.get("HOME") == "/home/appuser"

APPS = [
    {
        "id": "telco_churn",
        "icon": "📱",
        "title": "MVNO Telco Churn Prediction & Martech Campaigns",
        "desc": (
            "Proprietary XGBoost model trained on private synthetic MVNO data scores every customer "
            "on 15 behavioural signals — usage recency, data consumption trends, support friction, "
            "payment delays — and maps each churn risk directly to an actionable martech campaign: "
            "VIP Retention, Win-Back, Early Renewal, Data Upsell, and more. Each invocation scores "
            "a fresh random customer batch with per-customer SHAP explanations."
        ),
        "why": "Demonstrates end-to-end ML for revenue retention: proprietary model, private training data, and campaign-ready outputs for CRM and martech teams.",
        "tags": ["XGBoost", "SHAP", "MVNO", "Martech", "Proprietary Model", "Synthetic Data"],
        "status": "live",
    },
    {
        "id": "rai_eval",
        "icon": "🛡️",
        "title": "Responsible AI Content Evaluation Pipeline",
        "desc": (
            "Intercepts AI-generated promotional content before it reaches customers and evaluates it "
            "across Harm, Fairness, Compliance, and Tone dimensions relative to each user's risk profile. "
            "Passes, refines, or blocks — all before the message lands."
        ),
        "why": "Maps directly to regulatory obligations under ACMA and state gambling codes.",
        "tags": ["LangGraph", "Multi-Agent", "Responsible AI", "Groq", "Streamlit"],
        "status": "live",
    },
    {
        "id": "sports_intel",
        "icon": "📡",
        "title": "Live Sports Intelligence Agent",
        "desc": (
            "A LangGraph ReAct agent that calls live tools — The Odds API for bookmaker lines, ESPN for "
            "injury news, OpenWeatherMap for outdoor venues — then synthesises the data into structured "
            "betting signals with a full agent reasoning trace."
        ),
        "why": "Shows domain-relevant agentic reasoning with real-time data in a sports betting context.",
        "tags": ["LangGraph", "ReAct Agent", "Tool Use", "The Odds API", "ESPN"],
        "status": "live",
    },
    {
        "id": "rg_warning",
        "icon": "⚠️",
        "title": "Responsible Gambling Early Warning System",
        "desc": (
            "Classifies at-risk customers from behavioural signals — loss chasing, late-night sessions, "
            "deposit frequency spikes — with SHAP explainability showing exactly why each customer was flagged."
        ),
        "why": "Directly addresses harm minimisation obligations. The ML behind the compliance.",
        "tags": ["XGBoost", "SHAP", "Synthetic Data", "Streamlit"],
        "status": "live",
    },
    {
        "id": "churn",
        "icon": "📉",
        "title": "Customer Churn Prediction with Explainability",
        "desc": (
            "Predicts which customers are about to lapse from 15 behavioural signals, with per-customer "
            "SHAP waterfall charts and LIME explanations showing exactly why each customer was flagged. "
            "Full ML lifecycle — feature engineering, XGBoost training, and interactive serving."
        ),
        "why": "Adds ML lifecycle depth beyond LLM work. Per-customer explainability supports CRM action and compliance audit trails.",
        "tags": ["XGBoost", "SHAP", "LIME", "scikit-learn", "Synthetic Data"],
        "status": "live",
    },
    {
        "id": "fraud",
        "icon": "🔍",
        "title": "Bet Anomaly & Fraud Detection",
        "desc": (
            "Unsupervised anomaly detection on betting patterns to flag match-fixing signals and "
            "arbitrage exploitation, with an explainability layer surfacing the key features."
        ),
        "why": "Addresses a real risk the industry actively manages. ML breadth beyond the LLM stack.",
        "tags": ["Isolation Forest", "SHAP", "Anomaly Detection", "Synthetic Data"],
        "status": "live",
    },
]

CARD_ACCENT = {
    "telco_churn":  "linear-gradient(90deg,#6366f1,#8b5cf6)",
    "rai_eval":     "linear-gradient(90deg,#0ea5e9,#06b6d4)",
    "sports_intel": "linear-gradient(90deg,#f59e0b,#f97316)",
    "rg_warning":   "linear-gradient(90deg,#ef4444,#f97316)",
    "churn":        "linear-gradient(90deg,#10b981,#059669)",
    "fraud":        "linear-gradient(90deg,#8b5cf6,#6366f1)",
}
