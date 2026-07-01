import os
import sys
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, "responsible-ai-eval", ".env"))
sys.path.insert(0, os.path.join(ROOT, "responsible-ai-eval"))
sys.path.insert(0, os.path.join(ROOT, "sports-intelligence-agent"))

IS_CLOUD = os.environ.get("HOME") == "/home/appuser"

st.set_page_config(
    page_title="Sudarshana's AI Playground",
    page_icon="🎮",
    layout="wide",
)

# Header link click → go home
if st.query_params.get("nav") == "home":
    st.session_state["page"] = "home"
    st.query_params.clear()
    st.rerun()

# ── Shared CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main { padding-top: 0; }
    [data-testid="stHeader"] { display: none; }

    /* Section headings */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #9ca3af;
        margin: 0 0 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e5e7eb;
    }

    /* Cards */
    .app-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 24px 22px 18px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        height: 100%;
    }
    .card-icon  { font-size: 2.2rem; line-height: 1; margin-bottom: 10px; }
    .card-title { font-size: 1rem; font-weight: 700; color: #111827; line-height: 1.35; margin-bottom: 8px; }
    .card-desc  { font-size: 0.84rem; color: #6b7280; line-height: 1.6; margin-bottom: 12px; }
    .card-why {
        background: #f0f9ff;
        border-left: 3px solid #38bdf8;
        border-radius: 0 6px 6px 0;
        padding: 7px 11px;
        font-size: 0.79rem;
        color: #0369a1;
        line-height: 1.5;
        margin-bottom: 12px;
    }
    .tags { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 14px; }
    .tag {
        background: #f3f4f6; color: #374151;
        border-radius: 20px; padding: 2px 9px;
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.2px;
    }
    .badge-live {
        display: inline-flex; align-items: center; gap: 5px;
        background: #dcfce7; color: #166534;
        border: 1px solid #86efac;
        border-radius: 20px; padding: 2px 9px;
        font-size: 0.7rem; font-weight: 700;
        margin-bottom: 10px;
    }
    .badge-soon {
        display: inline-flex; align-items: center; gap: 5px;
        background: #fef9c3; color: #854d0e;
        border: 1px solid #fde68a;
        border-radius: 20px; padding: 2px 9px;
        font-size: 0.7rem; font-weight: 700;
        margin-bottom: 10px;
    }

    /* Score bar */
    .score-bar-wrap { background:#e9ecef; border-radius:4px; height:8px; margin:2px 0 6px 0; }
    .score-bar-fill { height:8px; border-radius:4px; }

    .footer {
        margin-top: 60px; padding: 20px;
        text-align: center; color: #9ca3af;
        font-size: 0.8rem; border-top: 1px solid #f3f4f6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

APPS = [
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
        "status": "soon",
    },
    {
        "id": "churn",
        "icon": "📉",
        "title": "Customer Churn Prediction with Explainability",
        "desc": (
            "Predicts which customers are about to lapse, with per-customer SHAP/LIME explanations. "
            "Full ML lifecycle — feature engineering, training, and serving."
        ),
        "why": "Adds ML depth beyond LLM work. Explainability meets commercial retention.",
        "tags": ["scikit-learn", "LIME", "SHAP", "ML Lifecycle"],
        "status": "soon",
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
        "tags": ["Isolation Forest", "Autoencoder", "Anomaly Detection"],
        "status": "soon",
    },
]


def render_header(sub_title: str = ""):
    crumb = (
        f' <span style="color:#3a6080;margin:0 6px;">／</span>'
        f'<span style="color:#7fb3d3;font-size:0.82rem;font-weight:400;">{sub_title}</span>'
        if sub_title else ""
    )
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);margin:-4px -1rem 28px -1rem;padding:13px 28px;display:flex;align-items:center;">'
        f'<a href="?nav=home" style="color:white;text-decoration:none;font-size:1.05rem;font-weight:800;letter-spacing:-0.3px;">🎮 Sudarshana\'s AI Playground</a>'
        f'{crumb}</div>',
        unsafe_allow_html=True,
    )


def render_home():
    render_header()
    st.markdown(
        "<p style='color:#6b7280;font-size:0.92rem;margin:-10px 0 28px;line-height:1.6;'>"
        "Where LangGraph agents scout live odds, LLMs referee promotional content, "
        "and responsible AI isn't just a slide — it <em>actually blocks things</em>. "
        "If something breaks mid-demo, we'll call it emergent behaviour."
        "</p>",
        unsafe_allow_html=True,
    )

    live = [a for a in APPS if a["status"] == "live"]
    soon = [a for a in APPS if a["status"] == "soon"]

    st.markdown('<div class="section-label">Live demos</div>', unsafe_allow_html=True)
    live_cols = st.columns(len(live), gap="large")
    for col, app in zip(live_cols, live):
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in app["tags"])
        with col:
            st.markdown(
                f"""
                <div class="app-card">
                    <div class="card-icon">{app['icon']}</div>
                    <div class="badge-live">● LIVE</div>
                    <div class="card-title">{app['title']}</div>
                    <div class="card-desc">{app['desc']}</div>
                    <div class="card-why"><strong>Why it matters:</strong> {app['why']}</div>
                    <div class="tags">{tags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"▶ Open — {app['title'][:28]}…", key=f"nav_{app['id']}", type="primary", use_container_width=True):
                st.session_state["page"] = app["id"]
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Coming soon</div>', unsafe_allow_html=True)
    soon_cols = st.columns(len(soon), gap="large")
    for col, app in zip(soon_cols, soon):
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in app["tags"])
        with col:
            st.markdown(
                f"""
                <div class="app-card" style="opacity:0.72;">
                    <div class="card-icon">{app['icon']}</div>
                    <div class="badge-soon">◌ COMING SOON</div>
                    <div class="card-title">{app['title']}</div>
                    <div class="card-desc">{app['desc']}</div>
                    <div class="card-why"><strong>Why it matters:</strong> {app['why']}</div>
                    <div class="tags">{tags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Coming Soon ⏳", key=f"nav_{app['id']}", disabled=True, use_container_width=True)

    st.markdown(
        '<div class="footer">Built with LangGraph · Streamlit · Open-source LLMs</div>',
        unsafe_allow_html=True,
    )




# ══════════════════════════════════════════════════════════════════════════════
# RAI EVAL PAGE
# ══════════════════════════════════════════════════════════════════════════════

RAI_MODEL_OPTIONS = {
    "Llama 3.3 70B — Groq": {
        "provider": "groq",
        "fast_model": "llama-3.1-8b-instant",
        "smart_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "key_label": "Groq API Key",
        "key_help": "Free at console.groq.com → API Keys",
        "note": "8B evaluators · 70B refiner · recommended default",
    },
    "Gemma 4 31B — Cerebras": {
        "provider": "cerebras",
        "fast_model": "gemma-4-31b",
        "smart_model": "gemma-4-31b",
        "env_key": "CEREBRAS_API_KEY",
        "key_label": "Cerebras API Key",
        "key_help": "Free at cloud.cerebras.ai → API Keys",
        "note": "Google Gemma 4 31B · ultra-fast Cerebras inference",
    },
    "Llama 3.3 70B — SambaNova": {
        "provider": "sambanova",
        "fast_model": "gemma-4-31B-it",
        "smart_model": "Meta-Llama-3.3-70B-Instruct",
        "env_key": "SAMBANOVA_API_KEY",
        "key_label": "SambaNova API Key",
        "key_help": "Free at cloud.sambanova.ai → API Keys",
        "note": "Gemma 4 31B evaluators · Llama 3.3 70B refiner",
    },
    "DeepSeek V3 — SambaNova": {
        "provider": "sambanova",
        "fast_model": "DeepSeek-V3.1",
        "smart_model": "DeepSeek-V3.2",
        "env_key": "SAMBANOVA_API_KEY",
        "key_label": "SambaNova API Key",
        "key_help": "Free at cloud.sambanova.ai → API Keys",
        "note": "DeepSeek V3.1 evaluators · V3.2 refiner",
    },
    "Ollama (local)": {
        "provider": "ollama",
        "fast_model": "llama3.2",
        "smart_model": "llama3.2",
        "env_key": None,
        "key_label": None,
        "key_help": None,
        "note": "Runs on your machine · no API key · requires Ollama",
    },
}

DECISION_CONFIG = {
    "pass":   {"label": "✅ PASS",   "color": "#198754", "bg": "#d1e7dd"},
    "refine": {"label": "🔄 REFINE", "color": "#856404", "bg": "#fff3cd"},
    "block":  {"label": "🚫 BLOCK",  "color": "#842029", "bg": "#f8d7da"},
}


def _score_bar(score: float, max_val: float = 10.0) -> str:
    pct = int((score / max_val) * 100)
    color = "#198754" if pct < 40 else "#ffc107" if pct < 70 else "#dc3545"
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar-fill" style="background:{color};width:{pct}%;"></div>'
        f'</div>'
    )


def _run_rai_pipeline(user: dict, content: str):
    from pipeline.nodes import configure as rai_configure
    from pipeline.graph import pipeline
    from examples.data import DEMO_MESSAGE  # noqa: F401 — side-effect import to ensure path works

    model_cfg = st.session_state.get("rai_model_cfg", list(RAI_MODEL_OPTIONS.values())[0])
    rai_configure(
        provider=model_cfg["provider"],
        fast_model=model_cfg["fast_model"],
        smart_model=model_cfg["smart_model"],
    )

    initial = {
        "original_content": content,
        "current_content": content,
        "user_name": user["name"],
        "user_age": user["age"],
        "user_risk_profile": user["risk_profile"],
        "deposit_count_today": user.get("deposit_count_today", 0),
        "content_type": "",
        "harm_score": 0.0, "harm_reasoning": "",
        "fairness_score": 0.0, "fairness_reasoning": "",
        "compliance_score": 0.0, "compliance_reasoning": "",
        "tone_score": 0.0, "tone_reasoning": "",
        "overall_score": 0.0,
        "decision": "",
        "block_reason": None,
        "iteration_count": 0,
        "refined_content": None,
        "final_content": None,
        "evaluation_trace": [],
    }
    return pipeline.invoke(initial)


def _render_rai_user_card(col, user: dict, result, placeholder=False):
    from examples.data import RISK_PROFILE_LABELS, RISK_PROFILE_COLORS

    risk = user["risk_profile"]
    risk_color = RISK_PROFILE_COLORS[risk]
    risk_label = RISK_PROFILE_LABELS[risk]

    with col:
        st.markdown(
            f"""<div style="border:1px solid #dee2e6;border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="font-size:1.05rem;font-weight:700;">{user['name']}</div>
            <div style="font-size:0.82rem;color:#6c757d;margin-bottom:6px;">Age {user['age']} · {user.get('description','')}</div>
            <span style="background:{risk_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.73rem;font-weight:600;">{risk_label}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if placeholder:
            st.caption("Waiting…")
            return
        if result is None:
            return

        decision = result["decision"]
        cfg = DECISION_CONFIG[decision]
        overall = result["overall_score"]

        st.markdown(
            f"""<div style="background:{cfg['bg']};border:1px solid {cfg['color']};border-radius:8px;
            padding:10px;text-align:center;margin-bottom:8px;">
            <div style="font-size:1.3rem;font-weight:800;color:{cfg['color']};">{cfg['label']}</div>
            <div style="font-size:0.88rem;color:{cfg['color']};">Score: {overall}/100</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown(_score_bar(overall, 100), unsafe_allow_html=True)

        with st.expander("Details"):
            dims = [
                ("Harm",       result["harm_score"],       result["harm_reasoning"]),
                ("Fairness",   result["fairness_score"],   result["fairness_reasoning"]),
                ("Compliance", result["compliance_score"], result["compliance_reasoning"]),
                ("Tone",       result["tone_score"],       result["tone_reasoning"]),
            ]
            for dim, score, reason in dims:
                st.markdown(f"**{dim}** — {score}/10")
                st.markdown(_score_bar(score), unsafe_allow_html=True)
                st.caption(reason)
            if result.get("block_reason"):
                st.error(f"Block reason: {result['block_reason']}")
            if result.get("refined_content"):
                st.markdown("---")
                st.markdown("**Refined message:**")
                st.info(result["refined_content"])
                st.caption(f"Iterations: {result['iteration_count']}")


def render_rai_eval():
    from datetime import datetime
    from examples.data import DEMO_MESSAGE, DEMO_USERS, RISK_PROFILE_LABELS

    render_header("Responsible AI Content Evaluation Pipeline")
    st.markdown("<h2 style='margin:0 0 4px;'>🛡️ Responsible AI Content Evaluation Pipeline</h2>", unsafe_allow_html=True)
    st.divider()

    st.markdown(
        """
        <div style="background:#f8f9fa;border-left:4px solid #0d6efd;border-radius:4px;
                    padding:14px 18px;margin-bottom:24px;">
        <strong>The problem this solves:</strong> AI systems in sports betting generate personalised content
        at scale — promotions, bet recommendations, odds updates. Without a safeguard layer, the same
        AI-generated message reaches every customer regardless of whether they're a casual bettor, someone
        flagged by the RG team, or self-excluded. This pipeline <strong>intercepts content before delivery</strong>,
        evaluates it across four dimensions relative to each user's risk profile, then
        <span style="color:#198754;font-weight:600;">passes</span>,
        <span style="color:#856404;font-weight:600;">refines</span>, or
        <span style="color:#842029;font-weight:600;">blocks</span> it.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ RAI Pipeline Config")
        _rai_opts = [k for k in RAI_MODEL_OPTIONS if not (IS_CLOUD and RAI_MODEL_OPTIONS[k]["provider"] == "ollama")]
        selected = st.selectbox("Evaluation model", _rai_opts, key="rai_model_select")
        model_cfg = RAI_MODEL_OPTIONS[selected]
        st.session_state["rai_model_cfg"] = model_cfg
        st.caption(f"_{model_cfg['note']}_")

        if model_cfg["env_key"] and not os.environ.get(model_cfg["env_key"]):
            key = st.text_input(model_cfg["key_label"], type="password", help=model_cfg["key_help"], key="rai_api_key")
            if key:
                os.environ[model_cfg["env_key"]] = key
            else:
                st.warning(f"Enter your {model_cfg['key_label']} to run the pipeline.")
                st.stop()

    tab1, tab2, tab3 = st.tabs(["Multi-User Demo", "Custom Evaluation", "Evaluation Trace"])

    # ── Tab 1 ──────────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Step 1 — The AI-generated message")
        message = st.text_area("Message sent to all users:", value=DEMO_MESSAGE, height=80, key="rai_demo_msg")

        st.subheader("Step 2 — Evaluate against all user profiles")
        run_btn = st.button("▶ Run pipeline for all users", type="primary", key="rai_run_all")
        cols = st.columns(5)

        if run_btn:
            results_store = {}
            placeholders = [col.empty() for col in cols]
            for i, (col, user) in enumerate(zip(cols, DEMO_USERS)):
                with placeholders[i]:
                    _render_rai_user_card(col, user, None, placeholder=True)

            for i, user in enumerate(DEMO_USERS):
                with st.spinner(f"Evaluating {user['name']}…"):
                    result = _run_rai_pipeline(user, message)
                    results_store[user["name"]] = result
                placeholders[i].empty()
                _render_rai_user_card(cols[i], user, result)

            st.success("Pipeline complete.")

            refined_users = [(u, results_store[u["name"]]) for u in DEMO_USERS
                             if results_store.get(u["name"], {}).get("refined_content")]
            if refined_users:
                st.subheader("Refined messages")
                orig_col, *ref_cols = st.columns(1 + len(refined_users))
                with orig_col:
                    st.markdown("**Original**")
                    st.warning(message)
                for col, (user, res) in zip(ref_cols, refined_users):
                    with col:
                        st.markdown(f"**→ {user['name']}**")
                        st.success(res["refined_content"])

            st.session_state["rai_trace_runs"] = [
                {
                    "label": f"{u['name']} ({RISK_PROFILE_LABELS[u['risk_profile']]})",
                    "model": selected,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "decision": results_store[u["name"]]["decision"],
                    "score": results_store[u["name"]]["overall_score"],
                    "trace": results_store[u["name"]]["evaluation_trace"],
                }
                for u in DEMO_USERS
            ]
        else:
            for col, user in zip(cols, DEMO_USERS):
                _render_rai_user_card(col, user, None)

    # ── Tab 2 ──────────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Evaluate any message against a single user profile")
        c1, c2 = st.columns([2, 1])
        with c1:
            custom_content = st.text_area("AI-generated content:", height=120, placeholder="Paste any message here…", key="rai_custom_msg")
        with c2:
            risk_profile = st.selectbox(
                "User risk profile",
                ["standard", "at_risk", "self_excluded", "new_user", "vip"],
                format_func=lambda k: {"standard":"Standard","at_risk":"At-Risk","self_excluded":"Self-Excluded","new_user":"New User","vip":"VIP"}[k],
                key="rai_custom_risk",
            )
            user_age = st.number_input("User age", min_value=18, max_value=90, value=30, key="rai_custom_age")
            deposits = st.number_input("Deposits today", min_value=0, max_value=20, value=1, key="rai_custom_dep")

        if st.button("▶ Evaluate", type="primary", key="rai_custom_run") and custom_content.strip():
            custom_user = {"name": "Custom User", "age": user_age, "risk_profile": risk_profile, "deposit_count_today": deposits}
            with st.spinner("Running pipeline…"):
                result = _run_rai_pipeline(custom_user, custom_content)

            decision = result["decision"]
            cfg = DECISION_CONFIG[decision]
            st.markdown(
                f"""<div style="background:{cfg['bg']};border:1px solid {cfg['color']};border-radius:8px;
                padding:16px;text-align:center;margin:16px 0;">
                <div style="font-size:1.8rem;font-weight:800;color:{cfg['color']};">{cfg['label']}</div>
                <div style="font-size:0.95rem;color:{cfg['color']};">Overall risk score: {result['overall_score']}/100</div>
                </div>""",
                unsafe_allow_html=True,
            )
            d1, d2, d3, d4 = st.columns(4)
            for col, (dim, score, reason) in zip(
                [d1, d2, d3, d4],
                [("Harm", result["harm_score"], result["harm_reasoning"]),
                 ("Fairness", result["fairness_score"], result["fairness_reasoning"]),
                 ("Compliance", result["compliance_score"], result["compliance_reasoning"]),
                 ("Tone", result["tone_score"], result["tone_reasoning"])],
            ):
                with col:
                    st.metric(dim, f"{score}/10")
                    st.markdown(_score_bar(score), unsafe_allow_html=True)
                    st.caption(reason)

            if result.get("block_reason"):
                st.error(f"Block reason: {result['block_reason']}")
            if result.get("refined_content"):
                st.markdown("---")
                col_o, col_r = st.columns(2)
                with col_o:
                    st.markdown("**Original**"); st.warning(custom_content)
                with col_r:
                    st.markdown(f"**Refined (iter {result['iteration_count']})**"); st.success(result["refined_content"])

    # ── Tab 3 ──────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Evaluation Trace")
        runs = st.session_state.get("rai_trace_runs")
        if not runs:
            st.info("No trace yet — run an evaluation first.")
        else:
            for run in runs:
                cfg = DECISION_CONFIG[run["decision"]]
                with st.expander(
                    f"**{run['label']}** · {cfg['label']} · score {run['score']}/100 · {run['model']} · {run['timestamp']}",
                    expanded=len(runs) == 1,
                ):
                    for entry in run["trace"]:
                        st.json(entry)


# ══════════════════════════════════════════════════════════════════════════════
# SPORTS INTELLIGENCE AGENT PAGE
# ══════════════════════════════════════════════════════════════════════════════

SPORTS_MODEL_OPTIONS = {
    "Llama 3.3 70B — Groq": {
        "provider": "groq",
        "model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "key_label": "Groq API Key",
        "key_help": "Free at console.groq.com → API Keys",
        "note": "Llama 3.3 70B · strong tool-calling · recommended",
    },
    "Llama 3.3 70B — SambaNova": {
        "provider": "sambanova",
        "model": "Meta-Llama-3.3-70B-Instruct",
        "env_key": "SAMBANOVA_API_KEY",
        "key_label": "SambaNova API Key",
        "key_help": "Free at cloud.sambanova.ai → API Keys",
        "note": "Llama 3.3 70B on ultra-fast SambaNova inference",
    },
    "DeepSeek V3 — SambaNova": {
        "provider": "sambanova",
        "model": "DeepSeek-V3-0324",
        "env_key": "SAMBANOVA_API_KEY",
        "key_label": "SambaNova API Key",
        "key_help": "Free at cloud.sambanova.ai → API Keys",
        "note": "DeepSeek V3 · strong reasoning and tool use",
    },
    "Ollama (local)": {
        "provider": "ollama",
        "model": "llama3.2",
        "env_key": None,
        "key_label": None,
        "key_help": None,
        "note": "Runs locally · no API key · requires Ollama",
    },
}

SPORTS_CATALOGUE = {
    "🏈 NFL": ("americanfootball_nfl", "nfl"),
    "🏀 NBA": ("basketball_nba",       "nba"),
    "⚾ MLB": ("baseball_mlb",         "mlb"),
    "🏒 NHL": ("icehockey_nhl",        "nhl"),
    "⚽ EPL": ("soccer_epl",           "epl"),
    "🏉 AFL": ("aussierules_afl",       "afl"),
    "🏉 NRL": ("rugbyleague_nrl",       "nrl"),
}

SPORTS_DEFAULT_QUERIES = {
    "americanfootball_nfl": "Analyse tonight's NFL games. Surface line movement signals, injury concerns not yet priced in, and weather impacts on outdoor venues.",
    "basketball_nba": "Scan tonight's NBA slate for line discrepancies and injury news affecting spreads or totals.",
    "baseball_mlb": "Review today's MLB odds for value spots. Check weather for outdoor ballparks and flag pitcher injury news.",
    "icehockey_nhl": "Analyse current NHL odds. Identify spread discrepancies and any goaltender injury news.",
    "soccer_epl": "Scan EPL odds for line movement and team news affecting Asian handicap and total goals markets.",
    "aussierules_afl": "Analyse AFL odds. Flag line discrepancies and key player injuries that may affect the line.",
    "rugbyleague_nrl": "Review NRL odds for sharp money signals and injury news affecting the line.",
}


def render_sports_intel():
    from langchain_core.messages import AIMessage, ToolMessage
    from agent.nodes import configure as sports_configure
    from agent.graph import run_query

    render_header("Live Sports Intelligence Agent")
    st.markdown("<h2 style='margin:0 0 4px;'>📡 Live Sports Intelligence Agent</h2>", unsafe_allow_html=True)
    st.divider()

    st.markdown(
        """
        <div style="background:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:4px;
                    padding:14px 18px;margin-bottom:24px;">
        <strong>What this does:</strong> A <strong>LangGraph ReAct agent</strong> that calls live data tools —
        The Odds API for bookmaker lines, ESPN for injury news (free, no key), OpenWeatherMap for outdoor venues —
        then synthesises everything into structured betting intelligence signals.
        The full agent reasoning trace is visible after each run.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Sports Agent Config")

        _sport_opts = [k for k in SPORTS_MODEL_OPTIONS if not (IS_CLOUD and SPORTS_MODEL_OPTIONS[k]["provider"] == "ollama")]
        selected = st.selectbox("Agent model", _sport_opts, key="sports_model_select")
        sports_cfg = SPORTS_MODEL_OPTIONS[selected]
        st.caption(f"_{sports_cfg['note']}_")

        if sports_cfg["env_key"] and not os.environ.get(sports_cfg["env_key"]):
            llm_key = st.text_input(sports_cfg["key_label"], type="password", help=sports_cfg["key_help"], key="sports_llm_key")
            if llm_key:
                os.environ[sports_cfg["env_key"]] = llm_key
            else:
                st.warning(f"Enter {sports_cfg['key_label']} to run the agent.")

        st.divider()
        st.caption(
            "**Data sources:** The Odds API and OpenWeatherMap keys are loaded from `.env`. "
            "ESPN news is free — no key needed."
        )

    if sports_cfg["env_key"] and not os.environ.get(sports_cfg["env_key"]):
        st.info("👈 Enter your LLM API key in the sidebar to run the agent.")
        return

    sports_configure(provider=sports_cfg["provider"], model=sports_cfg["model"])

    col_sport, _ = st.columns([1, 2])
    with col_sport:
        sport_label = st.selectbox("Sport", list(SPORTS_CATALOGUE.keys()), key="sports_sport")

    sport_key, league_code = SPORTS_CATALOGUE[sport_label]
    default_q = SPORTS_DEFAULT_QUERIES.get(sport_key, "Analyse current odds and surface any betting signals.")

    query = st.text_area("Intelligence query", value=default_q, height=88, key="sports_query")

    if st.button("▶ Run Intelligence Agent", type="primary", key="sports_run"):
        if not os.environ.get("ODDS_API_KEY"):
            st.warning("No Odds API key — agent will run but cannot fetch live odds.")

        with st.spinner("Agent is gathering live data and reasoning…"):
            try:
                result = run_query(sport_key=sport_key, sport_label=sport_label, query=query)
            except Exception as exc:
                st.error(f"Agent error: {exc}")
                return

        messages = result.get("messages", [])
        trace = result.get("trace", [])

        tc_id_to_name: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id_to_name[tc["id"]] = tc["name"]

        final_report = ""
        tool_results: dict[str, str] = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                final_report = msg.content
            elif isinstance(msg, ToolMessage):
                name = getattr(msg, "name", None) or tc_id_to_name.get(msg.tool_call_id, "tool")
                tool_results[name] = msg.content

        tab_r, tab_o, tab_n, tab_t = st.tabs(["📋 Intelligence Report", "📊 Live Odds", "📰 News & Weather", "🔍 Agent Trace"])

        with tab_r:
            if final_report:
                st.markdown(final_report)
            else:
                st.warning("No final report produced. Check the Agent Trace tab.")

        with tab_o:
            odds = tool_results.get("get_live_odds", "")
            if odds:
                st.code(odds, language=None)
            else:
                st.info("No odds data fetched in this run.")

        with tab_n:
            news = tool_results.get("get_sports_news", "")
            wx = tool_results.get("get_weather", "")
            if news:
                st.markdown("**News & injury reports:**")
                st.code(news, language=None)
            if wx:
                st.markdown("**Weather:**")
                st.code(wx, language=None)
            if not news and not wx:
                st.info("No news or weather data fetched.")

        with tab_t:
            st.caption("Step-by-step agent reasoning — each tool call and intermediate decision.")
            for i, entry in enumerate(trace):
                tool_calls = entry.get("tool_calls", [])
                preview = entry.get("content_preview", "")
                label = (
                    f"Step {i+1}: calls {', '.join(tc['name'] for tc in tool_calls)}"
                    if tool_calls else f"Step {i+1}: final synthesis"
                )
                with st.expander(label, expanded=(i == 0)):
                    for tc in tool_calls:
                        st.markdown(f"**`{tc['name']}`**")
                        st.json(tc["args"])
                    if preview:
                        st.markdown(f"> {preview}")

            with st.expander("Raw message history"):
                for msg in messages:
                    st.json({
                        "type": type(msg).__name__,
                        "content": (msg.content or "")[:500],
                        "tool_calls": getattr(msg, "tool_calls", None),
                        "tool_call_id": getattr(msg, "tool_call_id", None),
                    })

    with st.expander("How this agent works"):
        st.markdown(
            """
            **Architecture: LangGraph ReAct agent**

            ```
            Human query
                │
                ▼
            agent_node  ──── tool calls? ──yes──►  ToolNode (get_live_odds / get_sports_news / get_weather)
                │                                       │
                │◄──────────────────────────────────────┘
                │
                └── no tool calls ──► Final synthesis report ──► END
            ```

            **Tools:**
            - `get_live_odds` — The Odds API · flags ≥1.5pt spread discrepancies across bookmakers
            - `get_sports_news` — ESPN public API (free) · auto-flags injury headlines
            - `get_weather` — OpenWeatherMap · flags wind, freezing temps, precipitation

            **Signal types:** Line Movement · Injury Alert · Weather Impact · Value Spot
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

_page = st.session_state.get("page", "home")

if _page == "home":
    render_home()
elif _page == "rai_eval":
    render_rai_eval()
elif _page == "sports_intel":
    render_sports_intel()
else:
    st.session_state["page"] = "home"
    st.rerun()
