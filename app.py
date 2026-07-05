import importlib.util
import os
import sys
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(ROOT, "responsible-ai-eval", ".env"))
sys.path.insert(0, os.path.join(ROOT, "responsible-ai-eval"))
sys.path.insert(0, os.path.join(ROOT, "sports-intelligence-agent"))


def _load_pkg_module(pkg_name: str, pkg_dir: str, submodule: str):
    """
    Load <pkg_dir>/<submodule>.py under sys.modules key <pkg_name>.<submodule>.

    The parent package (<pkg_name>) is also registered so that relative imports
    inside the submodule (e.g. `from .data import ...`) resolve correctly.
    """
    full_name = f"{pkg_name}.{submodule}"
    if full_name in sys.modules:
        return sys.modules[full_name]

    # Register the parent package if not yet present
    if pkg_name not in sys.modules:
        pkg_spec = importlib.util.spec_from_file_location(
            pkg_name,
            os.path.join(pkg_dir, "__init__.py"),
            submodule_search_locations=[pkg_dir],
        )
        pkg = importlib.util.module_from_spec(pkg_spec)
        pkg.__path__ = [pkg_dir]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        pkg_spec.loader.exec_module(pkg)

    # Load the submodule
    file_path = os.path.join(pkg_dir, submodule + ".py")
    spec = importlib.util.spec_from_file_location(full_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod

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
# CHURN PREDICTION PAGE
# ══════════════════════════════════════════════════════════════════════════════

def render_telco_churn():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_header("MVNO Telco Churn & Martech Campaigns")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>📱 MVNO Telco Churn Prediction &amp; Martech Campaigns</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#f0fdf4;border:1px solid #86efac;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#166534;">
                &#9711;&nbsp; Proprietary model &mdash; no external API
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#eff6ff;border:1px solid #93c5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#1e40af;">
                &#128274;&nbsp; Trained on private synthetic MVNO data (CSV)
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#9881;&nbsp; XGBoost &middot; SHAP &mdash; fully local
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#fff7ed;border:1px solid #fdba74;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#c2410c;">
                &#127919;&nbsp; Campaign-ready output for CRM &amp; martech
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Business Problem ──────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#7fb3d3;margin-bottom:8px;">Business Problem</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                MVNO subscriber loss &mdash; identified early enough to act
            </div>
            <div style="font-size:0.88rem;color:#c8ddef;line-height:1.75;max-width:860px;">
                MVNOs operate on thin margins with high acquisition costs. A subscriber who reduces
                data consumption, stops calling, and then ports out represents a full CAC write-off.
                Standard reporting flags churn after it happens. This model flags it
                <strong>3&ndash;6 weeks before port-out</strong> so the CRM team can engage with the
                right offer at the right moment. Every prediction maps directly to a martech campaign
                action &mdash; no manual triage required.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#38bdf8;">3&ndash;6 wks</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        lead time before port-out
                    </div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#34d399;">15 signals</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        usage, friction, spend &amp; contract
                    </div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f59e0b;">8 campaigns</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        auto-assigned per risk profile
                    </div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f472b6;">per-customer</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        SHAP explanation for every score
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Three info panels ─────────────────────────────────────────────────────
    _tc1, _tc2, _tc3 = st.columns(3)

    with _tc1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Training Data (CSV file)
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">
                    800 subscribers &middot; 15 features &middot; saved to file
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:10px;">
                    A static CSV of 800 synthetic MVNO subscribers is generated once and persisted to
                    <code>telco-churn/data/training_data.csv</code>. The model always trains from this
                    fixed file &mdash; ensuring reproducible weights across runs.
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    At runtime, a separate random batch is generated fresh on each invocation and scored
                    by the locked model &mdash; simulating a live customer population the model has never seen.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;
                            border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;">
                        <span style="color:#0369a1;font-weight:700;">&#9679;</span>
                        <strong>Recency</strong> &mdash; days since call &amp; data
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#7c3aed;font-weight:700;">&#9679;</span>
                        <strong>Usage</strong> &mdash; data GB &amp; call volumes, trends
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#059669;font-weight:700;">&#9679;</span>
                        <strong>Revenue</strong> &mdash; ARPU, plan changes
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#d97706;font-weight:700;">&#9679;</span>
                        <strong>Contract</strong> &mdash; tenure, months remaining
                    </div>
                    <div>
                        <span style="color:#dc2626;font-weight:700;">&#9679;</span>
                        <strong>Friction</strong> &mdash; support tickets, payment delays
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _tc2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    ML Architecture
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    XGBoost &middot; SHAP &middot; per-invocation scoring
                </div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0f172a;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div><strong>Load training CSV</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                800 subscribers from file &middot; fixed seed
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1e3a5f;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div><strong>XGBoost classifier</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                300 trees &middot; depth 5 &middot; 80/20 stratified split
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">3</div>
                        <div><strong>SHAP TreeExplainer</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                per-customer top driver &middot; global importance
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0369a1;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">4</div>
                        <div><strong>Random batch scoring</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                fresh 30 subscribers per invocation &middot; new seed each run
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#c2410c;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">5</div>
                        <div><strong>Campaign assignment</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                risk tier + feature profile &rarr; campaign action
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _tc3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Martech Campaign Logic
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    8 campaigns mapped by risk &amp; feature profile
                </div>
                <div style="font-size:0.82rem;color:#374151;line-height:1.6;">
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fee2e2;color:#b91c1c;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">High risk</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">
                            VIP Retention &middot; Win-Back Offer<br>Early Renewal &middot; Service Recovery
                        </div>
                    </div>
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">Medium risk</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">
                            Renewal Reminder &middot; Data Upsell<br>Engagement Boost
                        </div>
                    </div>
                    <div>
                        <span style="background:#d1fae5;color:#065f46;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">Low risk</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">
                            Loyalty Reward
                        </div>
                    </div>
                    <div style="margin-top:12px;padding-top:12px;border-top:1px solid #f3f4f6;
                                font-size:0.78rem;color:#6b7280;">
                        High-risk routing prioritises contract status, then ARPU, then support friction.
                        Medium-risk routing checks contract window and data trend.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

    # ── Technical Implementation ──────────────────────────────────────────────
    st.markdown(
        """
        <div style="border:1px solid #e5e7eb;border-radius:13px;padding:26px 28px 22px;
                    background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:28px;">
            <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
                        color:#9ca3af;margin-bottom:6px;">Technical Implementation</div>
            <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:18px;">
                Static training data &rarr; locked model &rarr; fresh random batch per invocation
            </div>
            <div style="font-family:monospace;font-size:0.82rem;color:#374151;
                        background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
                        padding:18px 22px;line-height:1.9;overflow-x:auto;white-space:pre;">
DATASET 1 — Training &amp; Evaluation  (seed=42 · 800 subscribers · saved to CSV)
  · telco-churn/data/training_data.csv  — generated once, persisted, always reloaded
  · 640 subscribers → XGBoost training  (300 trees · depth 5 · LR 0.08)
  · 160 subscribers → held-out test  → AUC · precision · recall · F1
  · SHAP TreeExplainer fitted on training distribution
          |
          v  model weights locked
          |
DATASET 2 — Live Scoring Batch  (random seed · 30 subscribers · per invocation)
  · Generated fresh on every "Generate New Batch" click  — never cached
  · model.predict_proba()  → churn probability per subscriber
  · Percentile tiers       → High (top 30%) · Medium · Low (bottom 40%)
  · SHAP values            → top driving feature per subscriber
  · Campaign assignment    → risk tier + feature profile → martech action
          |
          v
Streamlit UI
  · Campaign Dashboard  — full subscriber table with risk + campaign per row
  · Model Performance   — AUC, precision, recall, F1, SHAP global importance
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load model (cached) ───────────────────────────────────────────────────
    @st.cache_resource(show_spinner="Training MVNO churn model on private dataset…")
    def _load_telco():
        _tc_train = _load_pkg_module("telco_churn_model", os.path.join(ROOT, "telco-churn", "model"), "train")
        return _tc_train.build()

    artifact = _load_telco()
    metrics = artifact["metrics"]

    _tc_data = _load_pkg_module("telco_churn_model", os.path.join(ROOT, "telco-churn", "model"), "data")
    TC_FEATURES = _tc_data.FEATURE_NAMES
    TC_DISPLAY  = _tc_data.FEATURE_DISPLAY
    CAMPAIGNS   = _tc_data.CAMPAIGNS

    # ── Per-invocation random batch ───────────────────────────────────────────
    if "telco_seed" not in st.session_state:
        st.session_state["telco_seed"] = int(np.random.randint(1000, 99999))

    tab1, tab2 = st.tabs(["📊 Campaign Dashboard", "📈 Model Performance"])

    # ── Tab 1: Campaign Dashboard ─────────────────────────────────────────────
    with tab1:
        ctrl1, ctrl2 = st.columns([3, 1])
        with ctrl1:
            st.caption(
                f"Scoring batch seed: `{st.session_state['telco_seed']}` &nbsp;·&nbsp; "
                f"30 subscribers generated fresh &nbsp;·&nbsp; "
                f"model trained on private CSV ({metrics['n_train']:,} subscribers)"
            )
        with ctrl2:
            if st.button("🔄 Generate New Batch", use_container_width=True):
                st.session_state["telco_seed"] = int(np.random.randint(1000, 99999))
                st.rerun()

        _tc_train = _load_pkg_module("telco_churn_model", os.path.join(ROOT, "telco-churn", "model"), "train")
        batch = _tc_train.score_batch(artifact, n=30, seed=st.session_state["telco_seed"])
        df = batch["df"]

        total = len(df)
        high  = int((df["risk_segment"] == "High").sum())
        med   = int((df["risk_segment"] == "Medium").sum())
        low   = int((df["risk_segment"] == "Low").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Subscribers Scored", f"{total}")
        m2.metric("High Risk", f"{high}", delta=f"{high/total:.0%} of batch", delta_color="inverse")
        m3.metric("Medium Risk", f"{med}", delta=f"{med/total:.0%} of batch", delta_color="off")
        m4.metric("Low Risk", f"{low}", delta=f"{low/total:.0%} of batch", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Campaign distribution ─────────────────────────────────────────────
        campaign_counts = df["campaign"].value_counts().to_dict()
        camp_items = []
        for name, info in CAMPAIGNS.items():
            count = campaign_counts.get(name, 0)
            if count > 0:
                camp_items.append(
                    f'<div style="display:inline-flex;align-items:center;gap:8px;'
                    f'background:{info["bg"]};border:1px solid {info["color"]}33;'
                    f'border-radius:10px;padding:8px 14px;margin:4px;">'
                    f'<span style="font-size:1.1rem;">{info["icon"]}</span>'
                    f'<div><div style="font-size:0.78rem;font-weight:700;color:{info["color"]};">'
                    f'{name}</div>'
                    f'<div style="font-size:0.72rem;color:#6b7280;">{count} subscriber{"s" if count != 1 else ""}</div>'
                    f'</div></div>'
                )
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;'
            'color:#9ca3af;margin-bottom:8px;">Campaign Distribution This Batch</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:20px;">'
            + "".join(camp_items) + "</div>",
            unsafe_allow_html=True,
        )

        # ── Subscriber table ──────────────────────────────────────────────────
        risk_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        display = df[["customer_id", "risk_segment", "churn_pct", "top_driver",
                       "campaign", "avg_monthly_spend", "contract_months_remaining",
                       "support_tickets_90d"]].copy()
        display["risk_segment"] = display["risk_segment"].apply(
            lambda r: f"{risk_icon.get(str(r), '')} {r}"
        )
        display["top_driver"] = display["top_driver"].map(TC_DISPLAY)
        display["campaign"] = display["campaign"].apply(
            lambda c: f"{CAMPAIGNS[c]['icon']} {c}" if c in CAMPAIGNS else c
        )
        display.columns = [
            "Subscriber", "Risk", "Churn %", "Top Driver",
            "Campaign Action", "ARPU ($)", "Contract Remaining", "Support Tickets",
        ]

        st.dataframe(
            display,
            column_config={
                "Churn %": st.column_config.ProgressColumn(
                    "Churn %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "ARPU ($)": st.column_config.NumberColumn("ARPU ($)", format="$%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=600,
        )
        st.caption(
            "**Top Driver** — the feature with the largest absolute SHAP value for that subscriber. "
            "**Campaign Action** — auto-assigned from risk tier and feature profile."
        )

    # ── Tab 2: Model Performance ──────────────────────────────────────────────
    with tab2:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("AUC", metrics["auc"], help="Area under the ROC curve.")
        p2.metric("Precision", metrics["precision"], help="Of flagged churners, fraction that actually churned.")
        p3.metric("Recall", metrics["recall"], help="Of actual churners, fraction the model caught.")
        p4.metric("F1 Score", metrics["f1"], help="Harmonic mean of precision and recall.")

        st.markdown("<br>", unsafe_allow_html=True)

        pd1, pd2 = st.columns(2)
        with pd1:
            st.markdown("**Dataset 1 — Training & evaluation**")
            st.markdown(
                f"""
                | | |
                |---|---|
                | Training file | `telco-churn/data/training_data.csv` |
                | Training subscribers | {metrics['n_train']:,} |
                | Held-out test subscribers | {metrics['n_test']:,} |
                | Base churn rate | {metrics['churn_rate']:.1%} |
                | Algorithm | XGBoost (300 trees, depth 5) |
                | Explainability | SHAP TreeExplainer |

                **Dataset 2 — Scoring batch (unseen)**

                | | |
                |---|---|
                | Scoring subscribers | 30 (random per invocation) |
                | Never seen during training | ✓ |
                | Used for | Campaign Dashboard |
                """
            )
        with pd2:
            st.markdown("**Features used**")
            st.markdown("\n".join(f"- {TC_DISPLAY[f]}" for f in TC_FEATURES))

        st.markdown("<br>**Global feature importance (mean |SHAP| across current batch)**")
        st.caption("Shows which signals the model relies on most for this scoring batch.")
        import shap as _shap
        try:
            plt.close("all")
            _shap.plots.bar(batch["shap_values"], max_display=15, show=False)
            fig_shap = plt.gcf()
            fig_shap.set_size_inches(8, 5)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP global plot unavailable: {exc}")


def render_churn():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_header("Customer Churn Prediction")
    st.markdown("<h2 style='margin:0 0 12px;'>📉 Customer Churn Prediction with Explainability</h2>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#f0fdf4;border:1px solid #86efac;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#166534;">
                &#9711;&nbsp; Proprietary model &mdash; no external API
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#eff6ff;border:1px solid #93c5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#1e40af;">
                &#128274;&nbsp; Trained on private synthetic data
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#9881;&nbsp; XGBoost &middot; SHAP &middot; LIME &mdash; runs fully local
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#7fb3d3;margin-bottom:8px;">Business Problem</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                Customer lifetime value erosion — caught too late to act
            </div>
            <div style="font-size:0.88rem;color:#c8ddef;line-height:1.75;max-width:860px;">
                A customer is acquired at a measurable CAC. They bet regularly for months,
                then gradually go quiet — fewer sessions, smaller stakes, longer gaps between bets.
                By the time the 30-day inactivity threshold is crossed and the customer is formally
                <em>churned</em>, they have often already moved to a competitor and no retention offer
                will bring them back. The window to act closes before the signal is obvious.
                This system scores every customer on <strong>15 behavioural signals</strong> — recency,
                frequency trend, stake trajectory, and engagement depth — and surfaces a churn probability
                <strong>2–4 weeks before the customer lapses</strong>, when a targeted offer, CRM outreach,
                or promotional adjustment can still change the outcome. Every prediction is accompanied by
                a per-customer explanation so CRM teams know <em>why</em> someone was flagged, not just
                <em>that</em> they were.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#38bdf8;">2–4 weeks</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        lead time before confirmed churn — when retention action is still viable
                    </div>
                </div>
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#34d399;">15 signals</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        behavioural features per customer across recency, frequency, monetary and engagement
                    </div>
                </div>
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f59e0b;">per-customer</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        SHAP + LIME explanations — CRM sees the exact signals that drove each flag
                    </div>
                </div>
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f472b6;">auditable</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        rationale for every prediction — supports compliance review and model governance
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Three info panels rendered individually via st.columns ────────────────
    _pc1, _pc2, _pc3 = st.columns(3)

    with _pc1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);height:100%;">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Synthetic Training Data
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">
                    600 customers &middot; 15 features &middot; no real PII
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:10px;">
                    Real customer behavioural data cannot be published. The generator encodes genuine
                    domain logic &mdash; rising recency, declining frequency, and support escalations
                    increase churn probability &mdash; so the model behaves as it would in production
                    without exposing any customer records.
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    Churn ground truth is derived from a logistic function over feature values, then
                    Gaussian noise is added to prevent the label from being perfectly separable.
                    The result is a realistic class distribution (~33% churned) with a model that
                    generalises rather than memorises.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;
                            border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;">
                        <span style="color:#7c3aed;font-weight:700;">&#9679;</span>
                        <strong>Recency</strong> &mdash; days since last bet
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#2563eb;font-weight:700;">&#9679;</span>
                        <strong>Frequency</strong> &mdash; bets &amp; sessions, last 30d vs prior 30d
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#059669;font-weight:700;">&#9679;</span>
                        <strong>Monetary</strong> &mdash; avg stake, stake size trend
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#d97706;font-weight:700;">&#9679;</span>
                        <strong>Engagement</strong> &mdash; live bet %, sport diversity, promo redemption
                    </div>
                    <div>
                        <span style="color:#dc2626;font-weight:700;">&#9679;</span>
                        <strong>Friction</strong> &mdash; support contacts, deposit count
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _pc2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);height:100%;">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    ML Architecture
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    XGBoost &middot; SHAP TreeExplainer &middot; LIME Tabular
                </div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0a1628;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">1</div>
                        <div><strong>Synthetic data generation</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                600 customers &middot; logistic churn label &middot; Gaussian noise
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1a3a5c;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">2</div>
                        <div><strong>Feature engineering</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                frequency_trend = bets_last &divide; bets_prev
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#2563eb;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">3</div>
                        <div><strong>XGBoost classifier</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                300 trees &middot; depth 5 &middot; LR 0.08 &middot; 80/20 split
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">4</div>
                        <div><strong>SHAP TreeExplainer</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                global importance &middot; per-customer waterfall
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#059669;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">5</div>
                        <div><strong>LIME Tabular</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                local linear approximation &middot; per-customer weights
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#d97706;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">6</div>
                        <div><strong>Streamlit serving</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                cohort table &middot; deep-dive &middot; what-if &middot; dashboard
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _pc3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);height:100%;">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Design Justification
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    Every technical choice justified by the domain
                </div>
                <div style="font-size:0.82rem;color:#374151;line-height:1.65;">
                    <div style="margin-bottom:13px;padding-bottom:13px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#ede9fe;color:#5b21b6;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">XGBoost</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            Handles mixed feature types and non-linear interactions natively &mdash;
                            the dominant algorithm for tabular prediction in production environments.
                        </div>
                    </div>
                    <div style="margin-bottom:13px;padding-bottom:13px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#dbeafe;color:#1d4ed8;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">SHAP</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            Globally consistent Shapley values. Regulators and risk teams can ask
                            &ldquo;why was this customer flagged?&rdquo; and receive a theoretically
                            grounded, reproducible answer.
                        </div>
                    </div>
                    <div style="margin-bottom:13px;padding-bottom:13px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#d1fae5;color:#065f46;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">LIME</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            Model-agnostic local explanation &mdash; useful as a cross-check and
                            intuitive for teams who trust linear &ldquo;if&ndash;then&rdquo; reasoning.
                        </div>
                    </div>
                    <div>
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">Percentile tiers</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            High/Medium/Low use cohort-relative percentile cuts, not fixed thresholds
                            &mdash; guaranteeing an actionable cohort regardless of overall churn rate.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:28px;'></div>", unsafe_allow_html=True)

    # ── Technical Architecture ────────────────────────────────────────────────
    st.markdown(
        """
        <div style="border:1px solid #e5e7eb;border-radius:13px;padding:26px 28px 22px;
                    background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:28px;">
            <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
                        color:#9ca3af;margin-bottom:6px;">Technical Implementation</div>
            <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:18px;">
                End-to-end ML pipeline — from synthetic data to per-customer explanation
            </div>
            <div style="font-family:monospace;font-size:0.82rem;color:#374151;
                        background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
                        padding:18px 22px;line-height:1.9;overflow-x:auto;white-space:pre;">
DATASET 1 — Training & Evaluation  (seed=42 · 600 customers)
  · Logistic churn label + Gaussian noise → realistic ~33% churn rate
  · 480 customers → XGBoost training  (300 trees · depth 5 · LR 0.08)
  · 120 customers → held-out test → AUC · precision · recall · F1
  · SHAP TreeExplainer and LIME Tabular fitted on training distribution
          │
          ▼  model weights locked — no further training
          │
DATASET 2 — Business Scoring  (seed=99 · 400 customers · unseen)
  · Completely separate synthetic population, never seen during training
  · model.predict_proba() → churn score per customer
  · Percentile tiers → High (top 30%) · Medium · Low (bottom 40%)
  · SHAP values computed → per-customer waterfall explanations
          │
          ▼
Streamlit UI  (app.py)
  · Cohort risk table — Dataset 2 customers ranked by churn score
  · Customer deep-dive — SHAP waterfall + LIME bar chart
  · What-if simulator — re-score on modified feature values
  · Model performance dashboard — metrics from Dataset 1 test split
            </div>
            <div style="display:flex;gap:16px;margin-top:18px;flex-wrap:wrap;">
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;">Stack</div>
                    <div style="font-size:0.82rem;color:#374151;line-height:1.8;">
                        Python · XGBoost · SHAP · LIME<br>scikit-learn · pandas · Streamlit
                    </div>
                </div>
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;">Key files</div>
                    <div style="font-size:0.82rem;color:#374151;line-height:1.8;">
                        <code>model/data.py</code> — data generation<br>
                        <code>model/train.py</code> — training pipeline<br>
                        <code>app.py</code> — Streamlit serving
                    </div>
                </div>
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;">Explainability</div>
                    <div style="font-size:0.82rem;color:#374151;line-height:1.8;">
                        SHAP: game-theoretic, auditable<br>
                        LIME: linear, human-readable<br>
                        Both per-customer, real-time
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="Training model on synthetic customer data…")
    def _load_churn():
        _cp_train = _load_pkg_module("churn_prediction_model", os.path.join(ROOT, "churn-prediction", "model"), "train")
        return _cp_train.build()

    artifact = _load_churn()
    df = artifact["df"]

    _cp_data = _load_pkg_module("churn_prediction_model", os.path.join(ROOT, "churn-prediction", "model"), "data")
    FEATURE_NAMES  = _cp_data.FEATURE_NAMES
    FEATURE_DISPLAY = _cp_data.FEATURE_DISPLAY
    FEATURE_RANGES  = _cp_data.FEATURE_RANGES

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cohort Risk Overview",
        "🔍 Customer Deep-Dive",
        "🎛️ What-If Simulator",
        "📈 Model Performance",
    ])

    # ── Tab 1: Cohort Risk Overview ──────────────────────────────────────────
    with tab1:
        total = len(df)
        high = int((df["risk_segment"] == "High").sum())
        med  = int((df["risk_segment"] == "Medium").sum())
        low  = int((df["risk_segment"] == "Low").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{total:,}")
        m2.metric("High Risk", f"{high}", delta=f"{high/total:.0%} of cohort", delta_color="inverse")
        m3.metric("Medium Risk", f"{med}", delta=f"{med/total:.0%} of cohort", delta_color="off")
        m4.metric("Low Risk", f"{low}", delta=f"{low/total:.0%} of cohort", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        f1, f2 = st.columns([2, 1])
        with f1:
            risk_filter = st.multiselect(
                "Filter by risk segment", ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"], key="cohort_risk_filter",
            )
        with f2:
            sort_col = st.selectbox(
                "Sort by", ["Churn Risk %", "Days Since Last Bet", "Bets (30d)"],
                key="cohort_sort",
            )

        sort_map = {
            "Churn Risk %": "churn_pct",
            "Days Since Last Bet": "days_since_last_bet",
            "Bets (30d)": "bets_last_30d",
        }
        sort_asc = sort_col == "Bets (30d)"

        filtered = df[df["risk_segment"].isin(risk_filter)].sort_values(
            sort_map[sort_col], ascending=sort_asc
        )

        display = filtered[[
            "customer_id", "risk_segment", "churn_pct",
            "days_since_last_bet", "bets_last_30d", "avg_stake_last_30d",
            "sessions_last_30d", "bet_frequency_trend",
        ]].copy()
        display.columns = [
            "Customer ID", "Risk", "Churn Risk %",
            "Days Since Bet", "Bets (30d)", "Avg Stake ($)",
            "Sessions (30d)", "Freq Trend",
        ]

        st.dataframe(
            display,
            column_config={
                "Churn Risk %": st.column_config.ProgressColumn(
                    "Churn Risk %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Risk": st.column_config.TextColumn("Risk"),
                "Avg Stake ($)": st.column_config.NumberColumn("Avg Stake ($)", format="$%.2f"),
                "Freq Trend": st.column_config.NumberColumn("Freq Trend", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=420,
        )

        st.caption(
            "**Churn Risk %** — model probability (0–100). "
            "**Freq Trend** — bets last 30d ÷ prior 30d; values below 1.0 indicate declining activity."
        )

    # ── Tab 2: Customer Deep-Dive ────────────────────────────────────────────
    with tab2:
        sorted_ids = df.sort_values("churn_pct", ascending=False)["customer_id"].tolist()
        selected_id = st.selectbox(
            "Select customer (sorted by churn risk, highest first)",
            sorted_ids, key="deepdive_customer",
        )

        row = df[df["customer_id"] == selected_id].iloc[0]
        cust_idx = df.index[df["customer_id"] == selected_id][0]
        prob = row["churn_pct"]
        risk = str(row["risk_segment"])
        risk_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#198754"}.get(risk, "#6c757d")
        risk_bg = {"High": "#f8d7da", "Medium": "#fff3cd", "Low": "#d1e7dd"}.get(risk, "#f8f9fa")

        st.markdown(
            f"""
            <div style="background:{risk_bg};border:1.5px solid {risk_color};border-radius:10px;
                        padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:24px;">
                <div style="text-align:center;min-width:90px;">
                    <div style="font-size:2rem;font-weight:800;color:{risk_color};">{prob:.1f}%</div>
                    <div style="font-size:0.78rem;color:{risk_color};font-weight:600;">Churn Risk</div>
                </div>
                <div style="border-left:2px solid {risk_color};height:40px;opacity:0.4;"></div>
                <div>
                    <div style="font-size:1.05rem;font-weight:700;">{selected_id}</div>
                    <span style="background:{risk_color};color:white;padding:2px 10px;border-radius:12px;
                                 font-size:0.75rem;font-weight:700;">{risk.upper()} RISK</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        avg = df[FEATURE_NAMES].mean()
        profile_cols = st.columns(5)
        key_feats = [
            ("days_since_last_bet", "Days Since Bet", "{:.0f}"),
            ("bets_last_30d", "Bets (30d)", "{:.0f}"),
            ("sessions_last_30d", "Sessions (30d)", "{:.0f}"),
            ("bet_frequency_trend", "Freq Trend", "{:.2f}"),
            ("support_contacts_last_90d", "Support (90d)", "{:.0f}"),
        ]
        for col, (feat, label, fmt) in zip(profile_cols, key_feats):
            val = row[feat]
            avg_val = avg[feat]
            delta = val - avg_val
            col.metric(label, fmt.format(val), f"{delta:+.1f} vs avg")

        st.markdown("<br>", unsafe_allow_html=True)

        shap_col, lime_col = st.columns(2)

        with shap_col:
            st.markdown("**SHAP Waterfall — why this customer was scored here**")
            st.caption("Red bars push churn probability up; blue bars push it down.")
            try:
                plt.close("all")
                shap.plots.waterfall(artifact["shap_values"][cust_idx], max_display=12, show=False)
                fig = plt.gcf()
                fig.set_size_inches(6, 5)
                st.pyplot(fig, use_container_width=True)
                plt.close("all")
            except Exception as exc:
                st.warning(f"SHAP plot unavailable: {exc}")

        with lime_col:
            st.markdown("**LIME Local Explanation — linear approximation around this customer**")
            st.caption("Positive weight (red) increases churn probability; negative (green) decreases it.")
            try:
                cust_features = df[FEATURE_NAMES].iloc[cust_idx].values
                lime_exp = artifact["lime_explainer"].explain_instance(
                    cust_features,
                    artifact["model"].predict_proba,
                    num_features=12,
                )
                lime_list = lime_exp.as_list()
                feats_lime = [f[0] for f in lime_list]
                vals_lime  = [f[1] for f in lime_list]
                colors_lime = ["#dc3545" if v > 0 else "#198754" for v in vals_lime]

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.barh(feats_lime, vals_lime, color=colors_lime, edgecolor="none")
                ax2.axvline(0, color="black", linewidth=0.8)
                ax2.set_xlabel("LIME weight")
                ax2.set_title("Local Feature Contributions", fontsize=10)
                ax2.tick_params(axis="y", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close("all")
            except Exception as exc:
                st.warning(f"LIME plot unavailable: {exc}")

        with st.expander("Full feature profile for this customer"):
            profile_data = {
                FEATURE_DISPLAY.get(f, f): [row[f], round(avg[f], 2)]
                for f in FEATURE_NAMES
            }
            import pandas as _pd
            profile_df = _pd.DataFrame(profile_data, index=["This Customer", "Cohort Avg"]).T
            st.dataframe(profile_df, use_container_width=True)

    # ── Tab 3: What-If Simulator ─────────────────────────────────────────────
    with tab3:
        st.markdown(
            """
            <div style="background:#fff7ed;border-left:4px solid #f97316;border-radius:4px;
                        padding:10px 16px;margin-bottom:20px;font-size:0.88rem;color:#7c2d12;">
            Select a customer as your starting point, then adjust the sliders to model a retention
            intervention or predict how their score would shift with different behaviour.
            </div>
            """,
            unsafe_allow_html=True,
        )

        wi_id = st.selectbox(
            "Start from customer", sorted_ids, key="whatif_customer",
        )
        wi_row = df[df["customer_id"] == wi_id].iloc[0]
        wi_idx = df.index[df["customer_id"] == wi_id][0]
        X_base = df[FEATURE_NAMES].iloc[wi_idx].values.reshape(1, -1).copy().astype(float)

        feat_idx = {f: i for i, f in enumerate(FEATURE_NAMES)}

        st.markdown("#### Adjust key behavioural features")
        c1, c2 = st.columns(2)
        with c1:
            days_wi = st.slider(
                "Days since last bet",
                0, 62, int(wi_row["days_since_last_bet"]),
                help="Higher = customer has been inactive longer",
                key="wi_days",
            )
            bets_wi = st.slider(
                "Bets (last 30d)",
                0, 55, int(wi_row["bets_last_30d"]),
                key="wi_bets",
            )
            sessions_wi = st.slider(
                "Sessions (last 30d)",
                0, 45, int(wi_row["sessions_last_30d"]),
                key="wi_sessions",
            )
            deposits_wi = st.slider(
                "Deposits (last 30d)",
                0, 12, int(wi_row["deposit_count_last_30d"]),
                key="wi_deposits",
            )
        with c2:
            freq_wi = st.slider(
                "Bet frequency trend (last÷prior 30d)",
                0.0, 3.0, float(wi_row["bet_frequency_trend"]),
                step=0.05,
                help="< 1.0 = declining. > 1.0 = accelerating.",
                key="wi_freq",
            )
            support_wi = st.slider(
                "Support contacts (last 90d)",
                0, 5, int(wi_row["support_contacts_last_90d"]),
                key="wi_support",
            )
            promo_wi = st.slider(
                "Promo redemption rate",
                0.0, 1.0, float(wi_row["promo_redemption_rate"]),
                step=0.05,
                key="wi_promo",
            )
            live_wi = st.slider(
                "Live bet %",
                0.0, 1.0, float(wi_row["live_bet_pct"]),
                step=0.05,
                key="wi_live",
            )

        X_mod = X_base.copy()
        X_mod[0, feat_idx["days_since_last_bet"]]      = days_wi
        X_mod[0, feat_idx["bets_last_30d"]]            = bets_wi
        X_mod[0, feat_idx["sessions_last_30d"]]        = sessions_wi
        X_mod[0, feat_idx["deposit_count_last_30d"]]   = deposits_wi
        X_mod[0, feat_idx["bet_frequency_trend"]]      = freq_wi
        X_mod[0, feat_idx["support_contacts_last_90d"]]= support_wi
        X_mod[0, feat_idx["promo_redemption_rate"]]    = promo_wi
        X_mod[0, feat_idx["live_bet_pct"]]             = live_wi

        orig_prob_wi = artifact["model"].predict_proba(X_base)[0, 1]
        new_prob_wi  = artifact["model"].predict_proba(X_mod)[0, 1]
        delta_wi = new_prob_wi - orig_prob_wi

        st.markdown("#### Score impact")
        r1, r2, r3 = st.columns(3)
        r1.metric("Original Churn Risk", f"{orig_prob_wi:.1%}")
        r2.metric(
            "Modified Churn Risk",
            f"{new_prob_wi:.1%}",
            f"{delta_wi:+.1%}",
            delta_color="inverse",
        )
        r3.metric(
            "Risk change",
            f"{abs(delta_wi):.1%} {'reduction' if delta_wi < 0 else 'increase'}",
        )

        if delta_wi < -0.10:
            st.success("These changes would meaningfully reduce churn risk — a valid retention intervention.")
        elif delta_wi > 0.10:
            st.error("These changes increase churn risk — this trajectory needs CRM attention.")
        else:
            st.info("Modest change in risk score. Try adjusting recency or frequency trend for larger impact.")

    # ── Tab 4: Model Performance ─────────────────────────────────────────────
    with tab4:
        metrics = artifact["metrics"]
        st.markdown("#### Model metrics (held-out test set)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("ROC-AUC", metrics["auc"],
                  help="Area under the ROC curve — 1.0 is perfect. Above 0.80 is production-viable.")
        p2.metric("Precision", metrics["precision"],
                  help="Of customers flagged as churn, what fraction actually churned.")
        p3.metric("Recall", metrics["recall"],
                  help="Of customers who churned, what fraction the model caught.")
        p4.metric("F1 Score", metrics["f1"],
                  help="Harmonic mean of precision and recall.")

        st.markdown("<br>", unsafe_allow_html=True)

        ti1, ti2 = st.columns([1, 1])
        with ti1:
            st.markdown("**Dataset 1 — Training & evaluation**")
            st.markdown(
                f"""
                | | |
                |---|---|
                | Training customers | {metrics['n_train']:,} |
                | Held-out test customers | {metrics['n_test']:,} |
                | Base churn rate (training set) | {metrics['churn_rate']:.1%} |
                | Algorithm | XGBoost (300 trees, depth 5) |
                | Explainability | SHAP TreeExplainer + LIME Tabular |

                **Dataset 2 — Business scoring (unseen)**

                | | |
                |---|---|
                | Scoring customers | {metrics['n_score']:,} |
                | Never seen during training | ✓ |
                | Used for | Cohort view · Deep-dive · What-if |
                """
            )
        with ti2:
            st.markdown("**Features used**")
            st.markdown(
                "\n".join(f"- {FEATURE_DISPLAY[f]}" for f in FEATURE_NAMES)
            )

        st.markdown("<br>**Global feature importance (mean |SHAP| across all customers)**")
        st.caption("Shows which signals the model relies on most — regardless of direction.")
        try:
            plt.close("all")
            shap.plots.bar(artifact["shap_values"], max_display=15, show=False)
            fig3 = plt.gcf()
            fig3.set_size_inches(8, 5)
            st.pyplot(fig3, use_container_width=True)
            plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP global plot unavailable: {exc}")

        with st.expander("How this model works"):
            st.markdown(
                """
                **Two-dataset pipeline**

                ```
                ┌─────────────────────────────────────────────────────────────┐
                │  DATASET 1 — Training & Evaluation  (seed=42, 600 customers)│
                └─────────────────────────────────────────────────────────────┘
                        │
                        ▼
                Feature engineering
                  · bet_frequency_trend = bets_last_30d ÷ bets_prev_30d
                  · stake_trend         = avg_stake_last_30d ÷ avg_stake_prev_30d
                        │
                        ▼
                80/20 stratified split
                  · 480 customers → model training
                  · 120 customers → held-out evaluation (AUC, precision, recall, F1)
                        │
                        ▼
                XGBoost classifier  (300 estimators, max depth 5, LR 0.08)
                SHAP TreeExplainer  ← fitted on training distribution
                LIME Tabular        ← fitted on training distribution

                ┌─────────────────────────────────────────────────────────────┐
                │  DATASET 2 — Business Scoring  (seed=99, 400 customers)     │
                │  Completely unseen during training                          │
                └─────────────────────────────────────────────────────────────┘
                        │
                        ▼
                model.predict_proba()  →  churn score per customer
                Percentile tiers       →  High / Medium / Low risk segments
                SHAP values            →  per-customer explanations
                        │
                        ▼
                Streamlit UI
                  · Cohort risk table     · Customer deep-dive
                  · What-if simulator     · Model performance dashboard
                ```

                **Why two datasets?**
                Evaluating a model on data it was trained on inflates every metric.
                The training set (Dataset 1) is used only to fit the model and measure
                its generalisation on the held-out 20%. The business dashboard (Dataset 2)
                is generated with a different random seed and never touches the training
                pipeline — so every churn score shown is a genuine out-of-sample prediction.
                """
            )


def render_rg_warning():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_header("Responsible Gambling Early Warning System")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>⚠️ Responsible Gambling Early Warning System</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
            <span style="display:inline-flex;align-items:center;gap:6px;background:#f0fdf4;border:1px solid #86efac;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#166534;">
                &#9711;&nbsp; Proprietary model &mdash; no external API
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#eff6ff;border:1px solid #93c5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#1e40af;">
                &#128274;&nbsp; Trained on private synthetic data (CSV)
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#fef2f2;border:1px solid #fca5a5;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#991b1b;">
                &#9878;&nbsp; Compliance &mdash; ACMA harm minimisation obligations
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#9881;&nbsp; XGBoost &middot; SHAP &mdash; fully local
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1c0a00 0%,#431407 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#fca5a5;margin-bottom:8px;">Regulatory Obligation</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                Identify harm before it becomes self-exclusion
            </div>
            <div style="font-size:0.88rem;color:#fecaca;line-height:1.75;max-width:860px;">
                Standard harm minimisation detects problem gambling after it has escalated — a self-exclusion
                request, a support call, a complaint. By then, significant financial and psychological harm
                has already occurred. This system reads <strong>15 behavioural signals</strong> continuously
                and flags at-risk customers <strong>3&ndash;6 weeks before self-exclusion</strong>, when a
                proportionate, graduated intervention can still change the trajectory. Every flag carries a
                SHAP explanation so compliance teams can document the rationale and demonstrate due diligence.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fca5a5;">3&ndash;6 wks</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">lead time before self-exclusion</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fdba74;">4 tiers</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">Low · Medium · High · Critical</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#86efac;">absolute</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">probability thresholds &mdash; not relative percentiles</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#c4b5fd;">auditable</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">SHAP explanation per flag for compliance review</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rg1, rg2, rg3 = st.columns(3)
    with rg1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Training Data (CSV)</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">1,000 players &middot; 15 signals &middot; private file</div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    A static CSV of 1,000 synthetic players is generated once and persisted.
                    The model trains from this fixed file. A separate random batch is scored
                    fresh each session &mdash; simulating live player monitoring.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;"><span style="color:#b91c1c;font-weight:700;">&#9679;</span> <strong>Chasing</strong> &mdash; loss chasing score, stake variance</div>
                    <div style="margin-bottom:3px;"><span style="color:#d97706;font-weight:700;">&#9679;</span> <strong>Time patterns</strong> &mdash; late-night %, session length</div>
                    <div style="margin-bottom:3px;"><span style="color:#7c3aed;font-weight:700;">&#9679;</span> <strong>Financial</strong> &mdash; deposit freq, net loss, escalation</div>
                    <div style="margin-bottom:3px;"><span style="color:#0369a1;font-weight:700;">&#9679;</span> <strong>Control avoidance</strong> &mdash; withdrawal reversals, reality check</div>
                    <div><span style="color:#059669;font-weight:700;">&#9679;</span> <strong>Game choice</strong> &mdash; high-volatility game %</div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
    with rg2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">ML Architecture</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">XGBoost &middot; SHAP &middot; absolute thresholds</div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1c0a00;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div><strong>Load training CSV</strong><br><span style="color:#6b7280;font-size:0.78rem;">1,000 players &middot; fixed seed &middot; persisted file</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#431407;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div><strong>XGBoost binary classifier</strong><br><span style="color:#6b7280;font-size:0.78rem;">300 trees &middot; depth 5 &middot; 80/20 stratified split</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7f1d1d;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">3</div>
                        <div><strong>Absolute threshold tiering</strong><br><span style="color:#6b7280;font-size:0.78rem;">prob &lt;0.3 Low &middot; &lt;0.6 Medium &middot; &lt;0.8 High &middot; &ge;0.8 Critical</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">4</div>
                        <div><strong>SHAP TreeExplainer</strong><br><span style="color:#6b7280;font-size:0.78rem;">top driver per player &middot; global importance</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#b91c1c;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">5</div>
                        <div><strong>Intervention assignment</strong><br><span style="color:#6b7280;font-size:0.78rem;">tier &rarr; graduated action for compliance team</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
    with rg3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Intervention Framework</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">Graduated response proportionate to risk</div>
                <div style="font-size:0.82rem;line-height:1.65;">
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#dcfce7;color:#166534;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">Low &lt;0.3</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Continue monitoring &mdash; no action required</div>
                    </div>
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">Medium 0.3&ndash;0.6</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">RG information email + self-assessment tool offer</div>
                    </div>
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fee2e2;color:#b91c1c;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">High 0.6&ndash;0.8</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Mandatory outreach + stake limits + cooling-off</div>
                    </div>
                    <div>
                        <span style="background:#fecaca;color:#7f1d1d;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">Critical &ge;0.8</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Restrict account + mandatory support referral</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Training RG early warning model on private dataset…")
    def _load_rg():
        _rg_train = _load_pkg_module("rg_warning_model", os.path.join(ROOT, "rg-warning", "model"), "train")
        return _rg_train.build()

    artifact = _load_rg()
    metrics  = artifact["metrics"]
    _rg_data = _load_pkg_module("rg_warning_model", os.path.join(ROOT, "rg-warning", "model"), "data")
    RG_FEATURES = _rg_data.FEATURE_NAMES
    RG_DISPLAY  = _rg_data.FEATURE_DISPLAY
    INTERVENTIONS = _rg_data.INTERVENTIONS

    if "rg_seed" not in st.session_state:
        st.session_state["rg_seed"] = int(np.random.randint(1000, 99999))

    tab1, tab2 = st.tabs(["🛡️ Player Risk Dashboard", "📈 Model Performance"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"Scoring batch seed: `{st.session_state['rg_seed']}` &nbsp;·&nbsp; 40 players scored &nbsp;·&nbsp; model trained on private CSV ({metrics['n_train']:,} players)")
        with c2:
            if st.button("🔄 Generate New Batch", key="rg_new_batch", use_container_width=True):
                st.session_state["rg_seed"] = int(np.random.randint(1000, 99999))
                st.rerun()

        _rg_train = _load_pkg_module("rg_warning_model", os.path.join(ROOT, "rg-warning", "model"), "train")
        batch = _rg_train.score_batch(artifact, n=40, seed=st.session_state["rg_seed"])
        df = batch["df"]

        total    = len(df)
        critical = int((df["risk_tier"] == "Critical").sum())
        high     = int((df["risk_tier"] == "High").sum())
        medium   = int((df["risk_tier"] == "Medium").sum())
        low      = int((df["risk_tier"] == "Low").sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Players Scored", total)
        m2.metric("🛑 Critical", critical, delta=f"{critical/total:.0%}", delta_color="inverse")
        m3.metric("🚨 High", high,     delta=f"{high/total:.0%}",     delta_color="inverse")
        m4.metric("⚠️ Medium",  medium,  delta=f"{medium/total:.0%}",  delta_color="off")
        m5.metric("✅ Low",     low,     delta=f"{low/total:.0%}",     delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        tier_icon = {"Critical": "🛑", "High": "🚨", "Medium": "⚠️", "Low": "✅"}
        display = df[["customer_id", "risk_tier", "risk_prob", "top_driver", "intervention",
                       "loss_chasing_score", "late_night_session_pct", "withdrawal_reversal_count"]].copy()
        display["risk_tier"]   = display["risk_tier"].apply(lambda r: f"{tier_icon.get(r,'')} {r}")
        display["top_driver"]  = display["top_driver"].map(RG_DISPLAY)
        display["risk_prob"]   = (display["risk_prob"] * 100).round(1)
        display.columns = ["Player", "Risk Tier", "Risk %", "Top Signal", "Intervention",
                           "Loss Chasing", "Late-Night %", "Withdrawal Reversals"]

        st.dataframe(
            display,
            column_config={
                "Risk %": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.1f%%"),
                "Loss Chasing": st.column_config.NumberColumn("Loss Chasing", format="%.2f"),
                "Late-Night %": st.column_config.NumberColumn("Late-Night %", format="%.0%%"),
            },
            hide_index=True,
            use_container_width=True,
            height=580,
        )
        st.caption("**Risk %** — model probability of problem gambling. **Top Signal** — highest absolute SHAP feature. **Intervention** — graduated action mapped to risk tier.")

    with tab2:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("AUC",       f"{metrics['auc']:.3f}")
        p2.metric("Precision", f"{metrics['precision']:.3f}")
        p3.metric("Recall",    f"{metrics['recall']:.3f}")
        p4.metric("F1",        f"{metrics['f1']:.3f}")
        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Training dataset**")
            st.markdown(f"""
| | |
|---|---|
| Training file | `rg-warning/data/training_data.csv` |
| Training players | {metrics['n_train']:,} |
| Test players | {metrics['n_test']:,} |
| At-risk rate | {metrics['at_risk_rate']:.1%} |
| Algorithm | XGBoost (300 trees, depth 5) |
| Explainability | SHAP TreeExplainer |
""")
        with d2:
            st.markdown("**Features used**")
            st.markdown("\n".join(f"- {RG_DISPLAY[f]}" for f in RG_FEATURES))
        st.markdown("<br>**Global feature importance (mean |SHAP| — current batch)**")
        import shap as _shap
        try:
            plt.close("all")
            _shap.plots.bar(batch["shap_values"], max_display=15, show=False)
            fig = plt.gcf(); fig.set_size_inches(8, 5)
            st.pyplot(fig, use_container_width=True); plt.close("all")
        except Exception as e:
            st.warning(f"SHAP plot unavailable: {e}")


def render_fraud_detection():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    render_header("Bet Anomaly & Fraud Detection")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>🔍 Bet Anomaly &amp; Fraud Detection</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
            <span style="display:inline-flex;align-items:center;gap:6px;background:#f0fdf4;border:1px solid #86efac;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#166534;">
                &#9711;&nbsp; Proprietary model &mdash; no external API
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#eff6ff;border:1px solid #93c5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#1e40af;">
                &#128274;&nbsp; Trained on private synthetic data (CSV)
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#129302;&nbsp; Unsupervised &mdash; no fraud labels required
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#fff7ed;border:1px solid #fdba74;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#c2410c;">
                &#128269;&nbsp; Isolation Forest &middot; SHAP &middot; analyst review workflow
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0c0a00 0%,#1c1a00 50%,#1a0a00 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#fde68a;margin-bottom:8px;">Business Problem</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                Flag suspicious betting patterns before financial exposure grows
            </div>
            <div style="font-size:0.88rem;color:#fef3c7;line-height:1.75;max-width:860px;">
                Match-fixing, arbitrage exploitation, bonus abuse, and syndicate bots all share a common
                problem: <strong>confirmed fraud labels arrive weeks after the activity</strong>. By then
                the financial exposure has already occurred. Isolation Forest learns what normal betting
                looks like across <strong>15 behavioural signals</strong> and flags accounts that deviate
                significantly &mdash; no labels required. A rule-based fraud type classifier then gives
                investigators a starting hypothesis, directing investigation effort to the right signal.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fde68a;">unsupervised</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">no fraud labels needed to start</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#86efac;">5 types</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">match-fixing · arb · bonus abuse · bot · unusual</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#c4b5fd;">per-account</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">SHAP driver for every anomaly score</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fdba74;">analyst-first</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">score + hypothesis &rarr; human investigation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fd1, fd2, fd3 = st.columns(3)
    with fd1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Training Data (CSV)</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">1,000 accounts &middot; 15 signals &middot; private file</div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    Synthetic accounts with realistic normal behaviour plus injected anomalous
                    patterns (match-fixing, bots, bonus abuse, arbitrage). Isolation Forest
                    trains on all accounts and learns to separate the outliers.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;"><span style="color:#b91c1c;font-weight:700;">&#9679;</span> <strong>Win patterns</strong> &mdash; win rate, win streak</div>
                    <div style="margin-bottom:3px;"><span style="color:#1e40af;font-weight:700;">&#9679;</span> <strong>Staking</strong> &mdash; avg stake, coefficient of variation</div>
                    <div style="margin-bottom:3px;"><span style="color:#5b21b6;font-weight:700;">&#9679;</span> <strong>Market selection</strong> &mdash; odds, concentration, sports count</div>
                    <div style="margin-bottom:3px;"><span style="color:#d97706;font-weight:700;">&#9679;</span> <strong>Timing</strong> &mdash; mins before start, late line moves</div>
                    <div><span style="color:#059669;font-weight:700;">&#9679;</span> <strong>Account signals</strong> &mdash; age, withdrawal ratio, bet volume</div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
    with fd2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">ML Architecture</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">Isolation Forest &middot; SHAP &middot; rule-based type</div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0c0a00;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div><strong>Load training CSV</strong><br><span style="color:#6b7280;font-size:0.78rem;">1,000 accounts &middot; mixed normal + anomalous</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1c1a00;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div><strong>Isolation Forest</strong><br><span style="color:#6b7280;font-size:0.78rem;">200 trees &middot; contamination=0.12 &middot; no labels</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">3</div>
                        <div><strong>Score normalisation</strong><br><span style="color:#6b7280;font-size:0.78rem;">score_samples() &rarr; [0,1] &middot; p70/p85 tier thresholds</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#b91c1c;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">4</div>
                        <div><strong>SHAP driver</strong><br><span style="color:#6b7280;font-size:0.78rem;">top anomaly-driving feature per account</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#d97706;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">5</div>
                        <div><strong>Fraud type classifier</strong><br><span style="color:#6b7280;font-size:0.78rem;">rule-based hypothesis &rarr; investigation focus</span></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )
    with fd3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Fraud Type Classification</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">5 hypotheses &mdash; analyst starting point</div>
                <div style="font-size:0.82rem;color:#374151;line-height:1.6;">
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fecaca;color:#7f1d1d;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#127919; Match-Fixing</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">High win rate + bets very close to event start</div>
                    </div>
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#dbeafe;color:#1e3a5f;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#129302; Bot / Syndicate</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Very consistent stake sizes + high volume</div>
                    </div>
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#127873; Bonus Abuse</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">New account + high withdrawal-to-deposit ratio</div>
                    </div>
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#ede9fe;color:#5b21b6;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#9878; Arbitrage</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Systematic stakes + high win rate + &le;2 sports</div>
                    </div>
                    <div>
                        <span style="background:#f3f4f6;color:#374151;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#10067; Unusual Pattern</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">High anomaly score without clear rule match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Fitting Isolation Forest on private betting dataset…")
    def _load_fraud():
        _fd_train = _load_pkg_module("fraud_detection_model", os.path.join(ROOT, "fraud-detection", "model"), "train")
        return _fd_train.build()

    artifact = _load_fraud()
    metrics  = artifact["metrics"]
    _fd_data = _load_pkg_module("fraud_detection_model", os.path.join(ROOT, "fraud-detection", "model"), "data")
    FD_FEATURES = _fd_data.FEATURE_NAMES
    FD_DISPLAY  = _fd_data.FEATURE_DISPLAY
    FRAUD_TYPES = _fd_data.FRAUD_TYPES

    if "fraud_seed" not in st.session_state:
        st.session_state["fraud_seed"] = int(np.random.randint(1000, 99999))

    tab1, tab2 = st.tabs(["🔍 Anomaly Dashboard", "📈 Model Insights"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(f"Scoring batch seed: `{st.session_state['fraud_seed']}` &nbsp;·&nbsp; 40 accounts scored &nbsp;·&nbsp; Isolation Forest trained on private CSV ({metrics['n_train']:,} accounts)")
        with c2:
            if st.button("🔄 Generate New Batch", key="fraud_new_batch", use_container_width=True):
                st.session_state["fraud_seed"] = int(np.random.randint(1000, 99999))
                st.rerun()

        _fd_train = _load_pkg_module("fraud_detection_model", os.path.join(ROOT, "fraud-detection", "model"), "train")
        batch = _fd_train.score_batch(artifact, n=40, seed=st.session_state["fraud_seed"])
        df = batch["df"]

        total  = len(df)
        high   = int((df["risk_tier"] == "High Risk").sum())
        medium = int((df["risk_tier"] == "Medium Risk").sum())
        low    = int((df["risk_tier"] == "Low Risk").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accounts Scored", total)
        m2.metric("🔴 High Risk",   high,   delta=f"{high/total:.0%}",   delta_color="inverse")
        m3.metric("🟡 Medium Risk", medium, delta=f"{medium/total:.0%}", delta_color="off")
        m4.metric("🟢 Low Risk",    low,    delta=f"{low/total:.0%}",    delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        fraud_counts = df[df["fraud_type"] != "Normal"]["fraud_type"].value_counts().to_dict()
        if fraud_counts:
            items = []
            for name, info in FRAUD_TYPES.items():
                count = fraud_counts.get(name, 0)
                if count > 0:
                    items.append(
                        f'<div style="display:inline-flex;align-items:center;gap:8px;background:{info["bg"]};'
                        f'border:1px solid {info["color"]}33;border-radius:10px;padding:8px 14px;margin:4px;">'
                        f'<span style="font-size:1.1rem;">{info["icon"]}</span>'
                        f'<div><div style="font-size:0.78rem;font-weight:700;color:{info["color"]};">{name}</div>'
                        f'<div style="font-size:0.72rem;color:#6b7280;">{count} account{"s" if count!=1 else ""}</div>'
                        f'</div></div>'
                    )
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#9ca3af;margin-bottom:8px;">Flagged Fraud Types This Batch</div>'
                '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:20px;">' + "".join(items) + "</div>",
                unsafe_allow_html=True,
            )

        risk_icon = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
        display = df[["customer_id", "risk_tier", "anomaly_pct", "fraud_type", "top_driver",
                       "win_rate_30d", "stake_cv", "account_age_days"]].copy()
        display["risk_tier"]  = display["risk_tier"].apply(lambda r: f"{risk_icon.get(r,'')} {r}")
        display["fraud_type"] = display["fraud_type"].apply(
            lambda f: f"{FRAUD_TYPES[f]['icon']} {f}" if f in FRAUD_TYPES else f
        )
        display["top_driver"] = display["top_driver"].map(FD_DISPLAY)
        display.columns = ["Account", "Risk Tier", "Anomaly %", "Fraud Type", "Top Driver",
                           "Win Rate", "Stake CV", "Account Age (days)"]

        st.dataframe(
            display,
            column_config={
                "Anomaly %": st.column_config.ProgressColumn("Anomaly %", min_value=0, max_value=100, format="%.1f%%"),
                "Win Rate":  st.column_config.NumberColumn("Win Rate",  format="%.2f"),
                "Stake CV":  st.column_config.NumberColumn("Stake CV",  format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=580,
        )
        st.caption("**Anomaly %** — Isolation Forest score (higher = more anomalous). **Fraud Type** — rule-based hypothesis for investigation focus. **Top Driver** — feature most responsible for the anomaly score.")

    with tab2:
        st.markdown(f"""
**Model details**

| | |
|---|---|
| Training file | `fraud-detection/data/fraud_training.csv` |
| Training accounts | {metrics['n_train']:,} |
| Contamination parameter | {metrics['contamination']:.0%} |
| Algorithm | Isolation Forest (200 trees) |
| Explainability | SHAP TreeExplainer (z-score fallback) |
| Features | {metrics['n_features']} behavioural signals |

**Anomaly score interpretation**

| Score range | Risk tier | Suggested action |
|---|---|---|
| 0–p70 | Low Risk | No action |
| p70–p85 | Medium Risk | Watch list — monitor 7 days |
| p85–100 | High Risk | Analyst review within 48h |
""")
        if batch["shap_values"] is not None:
            st.markdown("<br>**Global feature importance — mean |SHAP| across current batch**")
            import shap as _shap
            try:
                plt.close("all")
                _shap.plots.bar(batch["shap_values"], max_display=15, show=False)
                fig = plt.gcf(); fig.set_size_inches(8, 5)
                st.pyplot(fig, use_container_width=True); plt.close("all")
            except Exception as e:
                st.warning(f"SHAP plot unavailable: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

_page = st.session_state.get("page", "home")

if _page == "home":
    render_home()
elif _page == "telco_churn":
    render_telco_churn()
elif _page == "rai_eval":
    render_rai_eval()
elif _page == "sports_intel":
    render_sports_intel()
elif _page == "churn":
    render_churn()
elif _page == "rg_warning":
    render_rg_warning()
elif _page == "fraud":
    render_fraud_detection()
else:
    st.session_state["page"] = "home"
    st.rerun()
