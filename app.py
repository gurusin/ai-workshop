import os
import sys

import streamlit as st
from dotenv import load_dotenv

from config import ROOT

# ── Environment setup (must run before page imports that touch these paths) ──
load_dotenv(os.path.join(ROOT, "responsible-ai-eval", ".env"))
sys.path.insert(0, os.path.join(ROOT, "responsible-ai-eval"))
sys.path.insert(0, os.path.join(ROOT, "sports-intelligence-agent"))

from pages.home import render_home  # noqa: E402
from pages.rai_eval import render_rai_eval  # noqa: E402
from pages.sports_intel import render_sports_intel  # noqa: E402
from pages.telco_churn import render_telco_churn  # noqa: E402
from pages.churn_prediction import render_churn  # noqa: E402
from pages.rg_warning import render_rg_warning  # noqa: E402
from pages.fraud_detection import render_fraud_detection  # noqa: E402

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Sudarshana's AI Playground",
    page_icon="🎮",
    layout="wide",
)

# ── Header nav redirect ───────────────────────────────────────────────────────
if st.query_params.get("nav") == "home":
    st.session_state["page"] = "home"
    st.query_params.clear()
    st.rerun()

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main { padding-top: 0; }
    [data-testid="stHeader"] { display: none; }

    /* ── Section label ──────────────────────────────────────────────────────── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        color: #9ca3af;
        margin: 0 0 20px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e5e7eb;
    }

    /* ── Landing page cards ─────────────────────────────────────────────────── */
    .lp-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0 0 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        display: flex;
        flex-direction: column;
        min-height: 340px;
        overflow: hidden;
        transition: box-shadow 0.18s, transform 0.18s;
        cursor: default;
    }
    .lp-card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transform: translateY(-3px);
    }
    .lp-card-accent {
        height: 5px;
        border-radius: 14px 14px 0 0;
    }
    .lp-card-body {
        padding: 18px 18px 0;
        display: flex;
        flex-direction: column;
        flex: 1;
    }
    .lp-card-top {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    .lp-card-icon { font-size: 2rem; line-height: 1; }
    .lp-badge-live {
        background: #dcfce7; color: #166534;
        border: 1px solid #86efac;
        border-radius: 20px; padding: 2px 9px;
        font-size: 0.67rem; font-weight: 700;
        white-space: nowrap;
    }
    .lp-badge-soon {
        background: #fef9c3; color: #854d0e;
        border: 1px solid #fde68a;
        border-radius: 20px; padding: 2px 9px;
        font-size: 0.67rem; font-weight: 700;
        white-space: nowrap;
    }
    .lp-card-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.35;
        margin-bottom: 8px;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .lp-card-desc {
        font-size: 0.8rem;
        color: #6b7280;
        line-height: 1.65;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
        margin-bottom: 12px;
        flex: 1;
    }
    .lp-card-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 8px; }
    .lp-tag {
        background: #f3f4f6;
        color: #374151;
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 0.65rem;
        font-weight: 600;
    }

    /* ── Inner page card styles ──────────────────────────────────────────────── */
    .app-card { background:white; border:1px solid #e5e7eb; border-radius:16px; padding:24px 22px 18px; box-shadow:0 1px 4px rgba(0,0,0,0.05); }
    .card-icon  { font-size:2.2rem; line-height:1; margin-bottom:10px; }
    .card-title { font-size:1rem; font-weight:700; color:#111827; line-height:1.35; margin-bottom:8px; }
    .card-desc  { font-size:0.84rem; color:#6b7280; line-height:1.6; margin-bottom:12px; }
    .card-why   { background:#f0f9ff; border-left:3px solid #38bdf8; border-radius:0 6px 6px 0; padding:7px 11px; font-size:0.79rem; color:#0369a1; line-height:1.5; margin-bottom:12px; }
    .tags       { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:14px; }
    .tag        { background:#f3f4f6; color:#374151; border-radius:20px; padding:2px 9px; font-size:0.7rem; font-weight:600; }
    .badge-live { display:inline-flex; align-items:center; gap:5px; background:#dcfce7; color:#166534; border:1px solid #86efac; border-radius:20px; padding:2px 9px; font-size:0.7rem; font-weight:700; margin-bottom:10px; }
    .badge-soon { display:inline-flex; align-items:center; gap:5px; background:#fef9c3; color:#854d0e; border:1px solid #fde68a; border-radius:20px; padding:2px 9px; font-size:0.7rem; font-weight:700; margin-bottom:10px; }

    /* ── Score bar (RAI eval) ────────────────────────────────────────────────── */
    .score-bar-wrap { background:#e9ecef; border-radius:4px; height:8px; margin:2px 0 6px 0; }
    .score-bar-fill { height:8px; border-radius:4px; }

    /* ── Footer ──────────────────────────────────────────────────────────────── */
    .footer {
        margin-top: 60px; padding: 20px;
        text-align: center; color: #9ca3af;
        font-size: 0.8rem; border-top: 1px solid #f3f4f6;
    }

    /* ── Landing page button row ──────────────────────────────────────────────── */
    div[data-testid="stHorizontalBlock"] div[data-testid="column"] > div[data-testid="stVerticalBlock"] > div:last-child { margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Router ────────────────────────────────────────────────────────────────────
_PAGE_RENDERERS = {
    "home":        render_home,
    "telco_churn": render_telco_churn,
    "rai_eval":    render_rai_eval,
    "sports_intel": render_sports_intel,
    "churn":       render_churn,
    "rg_warning":  render_rg_warning,
    "fraud":       render_fraud_detection,
}

_page = st.session_state.get("page", "home")
renderer = _PAGE_RENDERERS.get(_page)

if renderer:
    renderer()
else:
    st.session_state["page"] = "home"
    st.rerun()
