import os

import streamlit as st

from config import IS_CLOUD
from utils.ui import render_header

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


def render_sports_intel() -> None:
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
            llm_key = st.text_input(
                sports_cfg["key_label"], type="password", help=sports_cfg["key_help"], key="sports_llm_key"
            )
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

    sport_key, _ = SPORTS_CATALOGUE[sport_label]
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

        tab_r, tab_o, tab_n, tab_t = st.tabs(
            ["📋 Intelligence Report", "📊 Live Odds", "📰 News & Weather", "🔍 Agent Trace"]
        )

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
