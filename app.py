import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Detect Streamlit Community Cloud (no local Ollama available)
IS_CLOUD = os.environ.get("HOME") == "/home/appuser"

st.set_page_config(
    page_title="Responsible AI Evaluation Pipeline",
    page_icon="🛡️",
    layout="wide",
)

# ── Model catalogue ───────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    # ── Groq (free — console.groq.com) ────────────────────────────────────────
    "Llama 3.3 70B — Groq": {
        "provider": "groq",
        "fast_model": "llama-3.1-8b-instant",
        "smart_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "key_label": "Groq API Key",
        "key_help": "Free at console.groq.com → API Keys",
        "note": "8B evaluators · 70B refiner · recommended default",
    },
    # ── Cerebras (free — cloud.cerebras.ai) ───────────────────────────────────
    "Gemma 4 31B — Cerebras": {
        "provider": "cerebras",
        "fast_model": "gemma-4-31b",
        "smart_model": "gemma-4-31b",
        "env_key": "CEREBRAS_API_KEY",
        "key_label": "Cerebras API Key",
        "key_help": "Free at cloud.cerebras.ai → API Keys",
        "note": "Google Gemma 4 31B · ultra-fast Cerebras inference",
    },
    # ── SambaNova (free — cloud.sambanova.ai) ─────────────────────────────────
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
    "Gemma 4 31B — SambaNova": {
        "provider": "sambanova",
        "fast_model": "gemma-4-31B-it",
        "smart_model": "gemma-4-31B-it",
        "env_key": "SAMBANOVA_API_KEY",
        "key_label": "SambaNova API Key",
        "key_help": "Free at cloud.sambanova.ai → API Keys",
        "note": "Google's latest open model · strong instruction following",
    },
    # ── Ollama (local — no key) ────────────────────────────────────────────────
    "Ollama (local)": {
        "provider": "ollama",
        "fast_model": "llama3.2",
        "smart_model": "llama3.2",
        "env_key": None,
        "key_label": None,
        "key_help": None,
        "note": "Runs on your machine · no API key · requires Ollama + ollama pull llama3.2",
    },
}

# ── Model selector ────────────────────────────────────────────────────────────
st.title("🛡️ Responsible AI Content Evaluation Pipeline")

st.markdown(
    """
    <div style="background:#f8f9fa;border-left:4px solid #0d6efd;border-radius:4px;padding:14px 18px;margin-bottom:20px;">
    <strong>The problem this solves:</strong> AI systems in sports betting generate personalised content at
    scale — promotions, bet recommendations, odds updates — tailored to individual users. Without a safeguard
    layer, the same AI-generated message reaches every customer identically, regardless of whether they are a
    casual bettor, someone flagged by the responsible gambling team, or a customer who has self-excluded.
    <br><br>
    This pipeline <strong>intercepts AI-generated content before delivery</strong>, evaluates it across four
    dimensions (Harm, Fairness, Compliance, Tone) relative to each user's risk profile, then either
    <span style="color:#198754;font-weight:600;">passes</span> it,
    <span style="color:#856404;font-weight:600;">refines</span> it into a safer version, or
    <span style="color:#842029;font-weight:600;">blocks</span> it entirely — all before the message reaches the customer.
    </div>
    """,
    unsafe_allow_html=True,
)

_available_models = [k for k in MODEL_OPTIONS if not (IS_CLOUD and MODEL_OPTIONS[k]["provider"] == "ollama")]
selected_model_name = st.selectbox(
    "Evaluation model",
    options=_available_models,
    index=0,
    help="All models are open-source weights. Groq models use your existing key.",
)
model_cfg = MODEL_OPTIONS[selected_model_name]
st.caption(f"_{model_cfg['note']}_")

# ── API key gate (per model) ──────────────────────────────────────────────────
if model_cfg["env_key"] and not os.environ.get(model_cfg["env_key"]):
    with st.sidebar:
        key = st.text_input(
            model_cfg["key_label"],
            type="password",
            help=model_cfg["key_help"],
        )
        if key:
            os.environ[model_cfg["env_key"]] = key
        else:
            st.warning(f"Enter your {model_cfg['key_label']} to run the pipeline.")
            st.stop()

# ── Configure pipeline models ─────────────────────────────────────────────────
from pipeline.nodes import configure as configure_models
from pipeline.graph import pipeline
from examples.data import DEMO_MESSAGE, DEMO_USERS, RISK_PROFILE_LABELS, RISK_PROFILE_COLORS

configure_models(
    provider=model_cfg["provider"],
    fast_model=model_cfg["fast_model"],
    smart_model=model_cfg["smart_model"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────
DECISION_CONFIG = {
    "pass":   {"label": "✅ PASS",   "color": "#198754", "bg": "#d1e7dd"},
    "refine": {"label": "🔄 REFINE", "color": "#856404", "bg": "#fff3cd"},
    "block":  {"label": "🚫 BLOCK",  "color": "#842029", "bg": "#f8d7da"},
}


def score_bar(score: float, max_val: float = 10.0) -> str:
    pct = int((score / max_val) * 100)
    color = "#198754" if pct < 40 else "#ffc107" if pct < 70 else "#dc3545"
    return f"""
    <div style="background:#e9ecef;border-radius:4px;height:8px;margin:2px 0 6px 0;">
      <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;"></div>
    </div>"""


def run_pipeline(user: dict, content: str) -> dict:
    initial_state = {
        "original_content": content,
        "current_content": content,
        "user_name": user["name"],
        "user_age": user["age"],
        "user_risk_profile": user["risk_profile"],
        "deposit_count_today": user.get("deposit_count_today", 0),
        "content_type": "",
        "harm_score": 0.0,
        "harm_reasoning": "",
        "fairness_score": 0.0,
        "fairness_reasoning": "",
        "compliance_score": 0.0,
        "compliance_reasoning": "",
        "tone_score": 0.0,
        "tone_reasoning": "",
        "overall_score": 0.0,
        "decision": "",
        "block_reason": None,
        "iteration_count": 0,
        "refined_content": None,
        "final_content": None,
        "evaluation_trace": [],
    }
    return pipeline.invoke(initial_state)


def render_user_card(col, user: dict, result: dict | None, placeholder=False):
    risk = user["risk_profile"]
    risk_color = RISK_PROFILE_COLORS[risk]
    risk_label = RISK_PROFILE_LABELS[risk]

    with col:
        st.markdown(
            f"""<div style="border:1px solid #dee2e6;border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="font-size:1.1rem;font-weight:700;">{user['name']}</div>
            <div style="font-size:0.85rem;color:#6c757d;margin-bottom:6px;">Age {user['age']} · {user['description']}</div>
            <span style="background:{risk_color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75rem;font-weight:600;">{risk_label}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        if placeholder:
            st.markdown("*Waiting…*")
            return

        if result is None:
            return

        decision = result["decision"]
        cfg = DECISION_CONFIG[decision]
        overall = result["overall_score"]

        st.markdown(
            f"""<div style="background:{cfg['bg']};border:1px solid {cfg['color']};border-radius:8px;
            padding:10px;text-align:center;margin-bottom:8px;">
            <div style="font-size:1.4rem;font-weight:800;color:{cfg['color']};">{cfg['label']}</div>
            <div style="font-size:0.9rem;color:{cfg['color']};">Risk score: {overall}/100</div>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown(score_bar(overall, 100), unsafe_allow_html=True)

        with st.expander("Details"):
            dims = [
                ("Harm",       result["harm_score"],       result["harm_reasoning"]),
                ("Fairness",   result["fairness_score"],   result["fairness_reasoning"]),
                ("Compliance", result["compliance_score"], result["compliance_reasoning"]),
                ("Tone",       result["tone_score"],       result["tone_reasoning"]),
            ]
            for dim, score, reason in dims:
                st.markdown(f"**{dim}** — {score}/10")
                st.markdown(score_bar(score), unsafe_allow_html=True)
                st.caption(reason)

            if result.get("block_reason"):
                st.error(f"Block reason: {result['block_reason']}")

            if result.get("refined_content"):
                st.markdown("---")
                st.markdown("**Refined message sent instead:**")
                st.info(result["refined_content"])
                st.caption(f"Refinement iterations: {result['iteration_count']}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Multi-User Demo", "Custom Evaluation", "Evaluation Trace"])

# ── Tab 1: Multi-User Demo ────────────────────────────────────────────────────
with tab1:
    st.subheader("Step 1 — The AI-generated message")
    message = st.text_area(
        "Message sent by the AI content engine to all users:",
        value=DEMO_MESSAGE,
        height=80,
    )

    st.subheader("Step 2 — Evaluate against all user profiles")
    run_btn = st.button("▶ Run pipeline for all users", type="primary")

    cols = st.columns(5)
    results_store = {}

    if run_btn:
        placeholders = [col.empty() for col in cols]

        for i, (col, user) in enumerate(zip(cols, DEMO_USERS)):
            with placeholders[i]:
                render_user_card(col, user, None, placeholder=True)

        for i, user in enumerate(DEMO_USERS):
            with st.spinner(f"Evaluating {user['name']}…"):
                result = run_pipeline(user, message)
                results_store[user["name"]] = result

            placeholders[i].empty()
            with cols[i]:
                render_user_card(cols[i], user, result)

        st.success("Pipeline complete for all users.")

        st.subheader("Refined messages (for REFINE decisions)")
        refined_users = [(u, results_store[u["name"]]) for u in DEMO_USERS
                         if results_store.get(u["name"], {}).get("refined_content")]
        if refined_users:
            orig_col, *ref_cols = st.columns(1 + len(refined_users))
            with orig_col:
                st.markdown("**Original message**")
                st.warning(message)
            for col, (user, res) in zip(ref_cols, refined_users):
                with col:
                    st.markdown(f"**→ {user['name']}**")
                    st.success(res["refined_content"])
        else:
            st.info("No refined messages — all were either PASS or BLOCK.")

        # Store traces for all users
        st.session_state["trace_runs"] = [
            {
                "label": f"{user['name']} ({RISK_PROFILE_LABELS[user['risk_profile']]})",
                "model": selected_model_name,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "decision": results_store[user["name"]]["decision"],
                "score": results_store[user["name"]]["overall_score"],
                "trace": results_store[user["name"]]["evaluation_trace"],
            }
            for user in DEMO_USERS
        ]

    else:
        for col, user in zip(cols, DEMO_USERS):
            render_user_card(col, user, None)

# ── Tab 2: Custom Evaluation ──────────────────────────────────────────────────
with tab2:
    st.subheader("Evaluate any message against a single user profile")

    c1, c2 = st.columns([2, 1])
    with c1:
        custom_content = st.text_area("AI-generated content to evaluate:", height=120,
                                       placeholder="Paste any AI-generated message here…")
    with c2:
        risk_profile = st.selectbox("User risk profile", list(RISK_PROFILE_LABELS.keys()),
                                     format_func=lambda k: RISK_PROFILE_LABELS[k])
        user_age = st.number_input("User age", min_value=18, max_value=90, value=30)
        deposits_today = st.number_input("Deposits today", min_value=0, max_value=20, value=1)

    if st.button("▶ Evaluate", type="primary") and custom_content.strip():
        custom_user = {
            "name": "Custom User",
            "age": user_age,
            "risk_profile": risk_profile,
            "deposit_count_today": deposits_today,
        }
        with st.spinner("Running pipeline…"):
            result = run_pipeline(custom_user, custom_content)

        decision = result["decision"]
        cfg = DECISION_CONFIG[decision]

        st.markdown(
            f"""<div style="background:{cfg['bg']};border:1px solid {cfg['color']};
            border-radius:8px;padding:16px;text-align:center;margin:16px 0;">
            <div style="font-size:2rem;font-weight:800;color:{cfg['color']};">{cfg['label']}</div>
            <div style="font-size:1rem;color:{cfg['color']};">Overall risk score: {result['overall_score']}/100</div>
            </div>""",
            unsafe_allow_html=True,
        )

        d1, d2, d3, d4 = st.columns(4)
        for col, (dim, score, reason) in zip(
            [d1, d2, d3, d4],
            [
                ("Harm",       result["harm_score"],       result["harm_reasoning"]),
                ("Fairness",   result["fairness_score"],   result["fairness_reasoning"]),
                ("Compliance", result["compliance_score"], result["compliance_reasoning"]),
                ("Tone",       result["tone_score"],       result["tone_reasoning"]),
            ],
        ):
            with col:
                st.metric(dim, f"{score}/10")
                st.markdown(score_bar(score), unsafe_allow_html=True)
                st.caption(reason)

        if result.get("block_reason"):
            st.error(f"Block reason: {result['block_reason']}")

        if result.get("refined_content"):
            st.markdown("---")
            col_orig, col_refined = st.columns(2)
            with col_orig:
                st.markdown("**Original**")
                st.warning(custom_content)
            with col_refined:
                st.markdown(f"**Refined (iteration {result['iteration_count']})**")
                st.success(result["refined_content"])

        st.session_state["trace_runs"] = [
            {
                "label": f"Custom — {RISK_PROFILE_LABELS[risk_profile]}, age {user_age}",
                "model": selected_model_name,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "decision": decision,
                "score": result["overall_score"],
                "trace": result["evaluation_trace"],
            }
        ]

# ── Tab 3: Evaluation Trace ───────────────────────────────────────────────────
with tab3:
    st.subheader("Evaluation Trace")
    st.caption("Full node-by-node log of every pipeline run. Switch here after running an evaluation.")

    runs = st.session_state.get("trace_runs")
    if not runs:
        st.info("No trace yet — run an evaluation in Tab 1 or Tab 2 first.")
    else:
        for run in runs:
            decision = run["decision"]
            cfg = DECISION_CONFIG[decision]
            with st.expander(
                f"**{run['label']}** · {cfg['label']} · score {run['score']}/100 · {run['model']} · {run['timestamp']}",
                expanded=len(runs) == 1,
            ):
                for entry in run["trace"]:
                    st.json(entry)
