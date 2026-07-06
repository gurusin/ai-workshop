import os
from datetime import datetime

import streamlit as st

from config import IS_CLOUD
from utils.ui import render_header, score_bar

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


def _run_pipeline(user: dict, content: str) -> dict:
    from pipeline.nodes import configure as rai_configure
    from pipeline.graph import pipeline

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


def _render_user_card(col, user: dict, result, placeholder: bool = False) -> None:
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
        st.markdown(score_bar(overall, 100), unsafe_allow_html=True)

        with st.expander("Details"):
            dims = [
                ("Harm",       result["harm_score"],       result["harm_reasoning"]),
                ("Fairness",   result["fairness_score"],   result["fairness_reasoning"]),
                ("Compliance", result["compliance_score"], result["compliance_reasoning"]),
                ("Tone",       result["tone_score"],       result["tone_reasoning"]),
            ]
            for dim, s, reason in dims:
                st.markdown(f"**{dim}** — {s}/10")
                st.markdown(score_bar(s), unsafe_allow_html=True)
                st.caption(reason)
            if result.get("block_reason"):
                st.error(f"Block reason: {result['block_reason']}")
            if result.get("refined_content"):
                st.markdown("---")
                st.markdown("**Refined message:**")
                st.info(result["refined_content"])
                st.caption(f"Iterations: {result['iteration_count']}")


def render_rai_eval() -> None:
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
                    _render_user_card(col, user, None, placeholder=True)

            for i, user in enumerate(DEMO_USERS):
                with st.spinner(f"Evaluating {user['name']}…"):
                    result = _run_pipeline(user, message)
                    results_store[user["name"]] = result
                placeholders[i].empty()
                _render_user_card(cols[i], user, result)

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
                _render_user_card(col, user, None)

    with tab2:
        st.subheader("Evaluate any message against a single user profile")
        c1, c2 = st.columns([2, 1])
        with c1:
            custom_content = st.text_area(
                "AI-generated content:", height=120, placeholder="Paste any message here…", key="rai_custom_msg"
            )
        with c2:
            risk_profile = st.selectbox(
                "User risk profile",
                ["standard", "at_risk", "self_excluded", "new_user", "vip"],
                format_func=lambda k: {
                    "standard": "Standard", "at_risk": "At-Risk",
                    "self_excluded": "Self-Excluded", "new_user": "New User", "vip": "VIP",
                }[k],
                key="rai_custom_risk",
            )
            user_age = st.number_input("User age", min_value=18, max_value=90, value=30, key="rai_custom_age")
            deposits = st.number_input("Deposits today", min_value=0, max_value=20, value=1, key="rai_custom_dep")

        if st.button("▶ Evaluate", type="primary", key="rai_custom_run") and custom_content.strip():
            custom_user = {
                "name": "Custom User", "age": user_age,
                "risk_profile": risk_profile, "deposit_count_today": deposits,
            }
            with st.spinner("Running pipeline…"):
                result = _run_pipeline(custom_user, custom_content)

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
            for col, (dim, s, reason) in zip(
                [d1, d2, d3, d4],
                [
                    ("Harm",       result["harm_score"],       result["harm_reasoning"]),
                    ("Fairness",   result["fairness_score"],   result["fairness_reasoning"]),
                    ("Compliance", result["compliance_score"], result["compliance_reasoning"]),
                    ("Tone",       result["tone_score"],       result["tone_reasoning"]),
                ],
            ):
                with col:
                    st.metric(dim, f"{s}/10")
                    st.markdown(score_bar(s), unsafe_allow_html=True)
                    st.caption(reason)

            if result.get("block_reason"):
                st.error(f"Block reason: {result['block_reason']}")
            if result.get("refined_content"):
                st.markdown("---")
                col_o, col_r = st.columns(2)
                with col_o:
                    st.markdown("**Original**")
                    st.warning(custom_content)
                with col_r:
                    st.markdown(f"**Refined (iter {result['iteration_count']})**")
                    st.success(result["refined_content"])

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
