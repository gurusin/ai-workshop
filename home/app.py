import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Sportsbet AI Portfolio",
    page_icon="🏆",
    layout="wide",
)


def _url(env_key: str, default: str = "#") -> str:
    try:
        v = st.secrets.get(env_key)
        if v:
            return v
    except Exception:
        pass
    return os.environ.get(env_key, default)


RAI_EVAL_URL = _url("RAI_EVAL_URL")
SPORTS_INTEL_URL = _url("SPORTS_INTEL_URL")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main { padding-top: 0; }
    .hero {
        background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0f2840 100%);
        padding: 56px 48px 44px;
        border-radius: 0 0 24px 24px;
        margin: -4px -1rem 40px;
        text-align: center;
        color: white;
    }
    .hero h1 { font-size: 2.4rem; font-weight: 800; margin: 0 0 12px; letter-spacing: -0.5px; }
    .hero p  { font-size: 1.1rem; color: #a8c4e0; margin: 0; max-width: 680px; margin: 0 auto; }
    .hero-sub {
        display: inline-block;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 4px 16px;
        font-size: 0.8rem;
        color: #7fb3d3;
        margin-top: 16px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #374151;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0 0 20px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e5e7eb;
    }
    .app-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 28px 24px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 12px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .app-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    .card-icon { font-size: 2.4rem; line-height: 1; }
    .card-title { font-size: 1.1rem; font-weight: 700; color: #111827; line-height: 1.3; }
    .card-desc  { font-size: 0.88rem; color: #6b7280; line-height: 1.6; flex: 1; }
    .card-why {
        background: #f0f9ff;
        border-left: 3px solid #0ea5e9;
        border-radius: 0 6px 6px 0;
        padding: 8px 12px;
        font-size: 0.8rem;
        color: #0369a1;
        line-height: 1.5;
    }
    .tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .tag {
        background: #f3f4f6;
        color: #374151;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-live {
        display: inline-flex; align-items: center; gap: 5px;
        background: #dcfce7; color: #166534;
        border: 1px solid #86efac;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.72rem; font-weight: 700;
    }
    .badge-soon {
        display: inline-flex; align-items: center; gap: 5px;
        background: #fef9c3; color: #854d0e;
        border: 1px solid #fde68a;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.72rem; font-weight: 700;
    }
    .launch-btn {
        display: inline-block;
        background: #0a1628;
        color: white !important;
        text-decoration: none !important;
        border-radius: 8px;
        padding: 9px 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-top: 4px;
        transition: background 0.15s;
    }
    .launch-btn:hover { background: #1a3a5c; }
    .launch-btn-disabled {
        display: inline-block;
        background: #f3f4f6;
        color: #9ca3af !important;
        border-radius: 8px;
        padding: 9px 20px;
        font-size: 0.85rem;
        font-weight: 600;
        text-align: center;
        margin-top: 4px;
        cursor: not-allowed;
    }
    .footer {
        margin-top: 60px;
        padding: 24px;
        text-align: center;
        color: #9ca3af;
        font-size: 0.82rem;
        border-top: 1px solid #f3f4f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏆 Sportsbet AI Portfolio</h1>
        <p>A collection of AI engineering projects built for the sports betting domain —
           demonstrating responsible AI, agentic reasoning, and ML engineering.</p>
        <div class="hero-sub">AI Lead Engineer · Sportsbet Australia</div>
    </div>
    """,
    unsafe_allow_html=True,
)


APPS = [
    {
        "icon": "🛡️",
        "title": "Responsible AI Content Evaluation Pipeline",
        "description": (
            "Intercepts AI-generated promotional content before delivery and evaluates it across four "
            "dimensions — Harm, Fairness, Compliance, Tone — relative to each customer's risk profile. "
            "Passes, refines, or blocks content using a multi-agent LangGraph pipeline."
        ),
        "why": "Maps to Sportsbet's core regulatory obligations under ACMA and state gambling codes.",
        "tags": ["LangGraph", "Multi-Agent", "Responsible AI", "Streamlit"],
        "status": "live",
        "url": RAI_EVAL_URL,
    },
    {
        "icon": "📡",
        "title": "Live Sports Intelligence Agent",
        "description": (
            "A LangGraph ReAct agent that ingests live odds, injury news, and weather data to surface "
            "structured betting intelligence signals — line movement anomalies, injury alerts, and "
            "weather impacts — with full agent reasoning trace."
        ),
        "why": "Shows domain-relevant agentic reasoning with real-time data in a Sportsbet-native context.",
        "tags": ["LangGraph", "ReAct Agent", "Tool Use", "The Odds API", "Streamlit"],
        "status": "live",
        "url": SPORTS_INTEL_URL,
    },
    {
        "icon": "⚠️",
        "title": "Responsible Gambling Early Warning System",
        "description": (
            "Classifies at-risk customers from behavioural patterns — chasing losses, late-night sessions, "
            "sudden bet-size escalation, deposit frequency spikes — with SHAP explainability showing why "
            "each customer was flagged."
        ),
        "why": "Directly addresses Sportsbet's regulatory obligations for harm minimisation.",
        "tags": ["XGBoost", "SHAP", "Synthetic Data", "Streamlit"],
        "status": "soon",
        "url": "#",
    },
    {
        "icon": "📉",
        "title": "Customer Churn Prediction with Explainability",
        "description": (
            "XGBoost model scoring 600 customers on 15 behavioural signals — recency, frequency trend, "
            "stake trajectory, engagement depth. Per-customer SHAP waterfall charts and LIME explanations "
            "show exactly why each customer was flagged. Full ML lifecycle: feature engineering, training, serving."
        ),
        "why": "Adds ML lifecycle depth beyond LLM work. Per-customer explainability supports CRM action and compliance audit trails.",
        "tags": ["XGBoost", "SHAP", "LIME", "scikit-learn", "Streamlit"],
        "status": "live",
        "url": _url("CHURN_URL"),
    },
    {
        "icon": "🔍",
        "title": "Bet Anomaly & Fraud Detection",
        "description": (
            "Unsupervised anomaly detection on betting patterns to flag potential match-fixing signals "
            "or arbitrage exploitation, with an explainability layer surfacing the key contributing features."
        ),
        "why": "Addresses a real risk Sportsbet actively manages and demonstrates ML breadth beyond LLM work.",
        "tags": ["Isolation Forest", "Autoencoder", "Anomaly Detection", "Streamlit"],
        "status": "soon",
        "url": "#",
    },
]

live_apps = [a for a in APPS if a["status"] == "live"]
soon_apps = [a for a in APPS if a["status"] == "soon"]

st.markdown('<div class="section-title">Live Apps</div>', unsafe_allow_html=True)

live_cols = st.columns(len(live_apps))
for col, app in zip(live_cols, live_apps):
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in app["tags"])
    btn_html = (
        f'<a href="{app["url"]}" target="_blank" class="launch-btn">↗ Launch App</a>'
        if app["url"] != "#"
        else '<span class="launch-btn-disabled">Configure URL →</span>'
    )
    with col:
        st.markdown(
            f"""
            <div class="app-card">
                <div>
                    <div class="card-icon">{app['icon']}</div>
                    <div style="margin-top:10px;">
                        <span class="badge-live">● LIVE</span>
                    </div>
                </div>
                <div class="card-title">{app['title']}</div>
                <div class="card-desc">{app['description']}</div>
                <div class="card-why"><strong>Why it matters:</strong> {app['why']}</div>
                <div class="tags">{tags_html}</div>
                {btn_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-title">Coming Soon</div>', unsafe_allow_html=True)

soon_cols = st.columns(len(soon_apps))
for col, app in zip(soon_cols, soon_apps):
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in app["tags"])
    with col:
        st.markdown(
            f"""
            <div class="app-card" style="opacity:0.75;">
                <div>
                    <div class="card-icon">{app['icon']}</div>
                    <div style="margin-top:10px;">
                        <span class="badge-soon">◌ COMING SOON</span>
                    </div>
                </div>
                <div class="card-title">{app['title']}</div>
                <div class="card-desc">{app['description']}</div>
                <div class="card-why"><strong>Why it matters:</strong> {app['why']}</div>
                <div class="tags">{tags_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="footer">
        Built with LangGraph · Streamlit · Open-source LLMs &nbsp;|&nbsp;
        Responsible AI Portfolio for Sportsbet Australia
    </div>
    """,
    unsafe_allow_html=True,
)
