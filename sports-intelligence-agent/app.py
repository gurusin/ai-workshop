import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

IS_CLOUD = os.environ.get("HOME") == "/home/appuser"

st.set_page_config(
    page_title="Live Sports Intelligence Agent",
    page_icon="📡",
    layout="wide",
)

# ── Model catalogue ────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
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
        "note": "Runs locally · no API key · requires Ollama + ollama pull llama3.2",
    },
}

# ── Sport catalogue ────────────────────────────────────────────────────────────
SPORTS = {
    "🏈 NFL (American Football)":    ("americanfootball_nfl",   "nfl"),
    "🏀 NBA (Basketball)":           ("basketball_nba",         "nba"),
    "⚾ MLB (Baseball)":             ("baseball_mlb",           "mlb"),
    "🏒 NHL (Ice Hockey)":           ("icehockey_nhl",          "nhl"),
    "⚽ EPL (English Premier League)":("soccer_epl",            "epl"),
    "🏉 AFL (Australian Rules)":     ("aussierules_afl",        "afl"),
    "🏉 NRL (Rugby League)":         ("rugbyleague_nrl",        "nrl"),
}

DEFAULT_QUERIES = {
    "americanfootball_nfl": (
        "Analyse tonight's NFL games. Identify any line movement signals, "
        "injury concerns that may not be priced in, and weather impacts on outdoor venues."
    ),
    "basketball_nba": (
        "Scan tonight's NBA slate for line discrepancies across bookmakers "
        "and any significant injury news affecting the spread or total."
    ),
    "baseball_mlb": (
        "Review today's MLB odds for value spots. Check weather for outdoor ballparks "
        "and flag any pitcher injury news."
    ),
    "icehockey_nhl": (
        "Analyse current NHL odds. Identify games with notable spread discrepancies "
        "and any goaltender injury news."
    ),
    "soccer_epl": (
        "Scan EPL odds for line movement signals and team news affecting "
        "the Asian handicap and total goals markets."
    ),
    "aussierules_afl": (
        "Analyse current AFL odds. Check for line discrepancies and "
        "key player injury news that may affect the line."
    ),
    "rugbyleague_nrl": (
        "Review NRL odds for sharp money signals and injury news "
        "affecting the line or first-try-scorer markets."
    ),
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📡 Live Sports Intelligence Agent")
st.markdown(
    """
    <div style="background:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:4px;
                padding:14px 18px;margin-bottom:24px;">
    <strong>What this does:</strong> A <strong>LangGraph ReAct agent</strong> that calls live data tools —
    The Odds API for bookmaker lines, ESPN for injury/team news, and OpenWeatherMap for outdoor venues —
    then synthesises the data into structured betting intelligence signals:
    line movement anomalies, injury alerts, and weather impacts.
    The full agent reasoning trace is shown after each run.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("LLM Model")
    _model_opts = [k for k in MODEL_OPTIONS if not (IS_CLOUD and MODEL_OPTIONS[k]["provider"] == "ollama")]
    selected_model_name = st.selectbox("Agent model", _model_opts, index=0)
    model_cfg = MODEL_OPTIONS[selected_model_name]
    st.caption(f"_{model_cfg['note']}_")

    if model_cfg["env_key"] and not os.environ.get(model_cfg["env_key"]):
        llm_key = st.text_input(
            model_cfg["key_label"],
            type="password",
            help=model_cfg["key_help"],
        )
        if llm_key:
            os.environ[model_cfg["env_key"]] = llm_key
        else:
            st.warning(f"Enter {model_cfg['key_label']} to run the agent.")

    st.divider()
    st.subheader("Data Sources")

    odds_key_env = os.environ.get("ODDS_API_KEY", "")
    odds_key = st.text_input(
        "The Odds API Key *",
        value=odds_key_env,
        type="password",
        help="Required for live odds. Free at https://the-odds-api.com (500 req/month free tier).",
    )
    if odds_key:
        os.environ["ODDS_API_KEY"] = odds_key

    weather_key_env = os.environ.get("WEATHER_API_KEY", "")
    weather_key = st.text_input(
        "OpenWeatherMap Key (optional)",
        value=weather_key_env,
        type="password",
        help="Optional. Free at https://openweathermap.org/api — enables weather impact signals.",
    )
    if weather_key:
        os.environ["WEATHER_API_KEY"] = weather_key

    st.divider()
    st.caption(
        "ESPN news is fetched for free — no key needed. "
        "The Odds API free tier gives 500 requests/month."
    )

# ── Configure agent model ─────────────────────────────────────────────────────
from agent.nodes import configure as configure_agent
from agent.graph import run_query

configure_agent(provider=model_cfg["provider"], model=model_cfg["model"])

# ── Guard: LLM key required ───────────────────────────────────────────────────
if model_cfg["env_key"] and not os.environ.get(model_cfg["env_key"]):
    st.info("👈 Enter your LLM API key in the sidebar to run the agent.")
    st.stop()

# ── Main interface ────────────────────────────────────────────────────────────
col_sport, col_gap = st.columns([1, 2])
with col_sport:
    sport_label = st.selectbox("Sport", list(SPORTS.keys()), index=0)

sport_key, league_code = SPORTS[sport_label]
default_query = DEFAULT_QUERIES.get(sport_key, "Analyse current odds and surface any betting signals.")

query = st.text_area(
    "Intelligence query",
    value=default_query,
    height=90,
    help="Tell the agent what to analyse. It will call the appropriate tools automatically.",
)

run_btn = st.button("▶ Run Intelligence Agent", type="primary", use_container_width=False)

# ── Run agent ─────────────────────────────────────────────────────────────────
if run_btn:
    if not os.environ.get("ODDS_API_KEY"):
        st.warning(
            "The Odds API key is not set. The agent will still run but cannot fetch live odds. "
            "Enter your key in the sidebar for full functionality."
        )

    with st.spinner("Agent is gathering live data and reasoning…"):
        try:
            result = run_query(sport_key=sport_key, sport_label=sport_label, query=query)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.stop()

    messages = result.get("messages", [])
    trace = result.get("trace", [])

    # ── Parse messages for display ─────────────────────────────────────────────
    from langchain_core.messages import AIMessage, ToolMessage

    # Map tool_call_id → tool_name
    tc_id_to_name: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id_to_name[tc["id"]] = tc["name"]

    # Separate final report vs tool results
    final_report = ""
    tool_results: dict[str, str] = {}

    for msg in messages:
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            final_report = msg.content
        elif isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or tc_id_to_name.get(msg.tool_call_id, "tool")
            tool_results[name] = msg.content

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tab_report, tab_odds, tab_news, tab_trace = st.tabs([
        "📋 Intelligence Report",
        "📊 Live Odds",
        "📰 News Feed",
        "🔍 Agent Trace",
    ])

    with tab_report:
        if final_report:
            st.markdown(final_report)
        else:
            st.warning(
                "The agent did not produce a final report. "
                "This can happen when the model hits the recursion limit or if the LLM key is invalid. "
                "Check the Agent Trace tab for details."
            )

    with tab_odds:
        odds_content = tool_results.get("get_live_odds", "")
        if odds_content:
            st.markdown("**Raw odds data fetched by the agent:**")
            st.code(odds_content, language=None)
        else:
            st.info("No odds data fetched in this run.")

    with tab_news:
        news_content = tool_results.get("get_sports_news", "")
        weather_content = tool_results.get("get_weather", "")
        if news_content:
            st.markdown("**News & injury reports:**")
            st.code(news_content, language=None)
        if weather_content:
            st.markdown("**Weather data:**")
            st.code(weather_content, language=None)
        if not news_content and not weather_content:
            st.info("No news or weather data fetched in this run.")

    with tab_trace:
        st.caption(
            "Full step-by-step agent trace. Each entry shows what tool the agent called "
            "and with which arguments, plus a preview of its reasoning between steps."
        )
        if not trace:
            st.info("No trace entries recorded.")
        else:
            for i, entry in enumerate(trace):
                tool_calls = entry.get("tool_calls", [])
                preview = entry.get("content_preview", "")

                if tool_calls:
                    label = f"Step {i + 1}: Agent → calls {', '.join(tc['name'] for tc in tool_calls)}"
                elif preview:
                    label = f"Step {i + 1}: Agent → final synthesis"
                else:
                    label = f"Step {i + 1}: Agent"

                with st.expander(label, expanded=(i == 0)):
                    if tool_calls:
                        for tc in tool_calls:
                            st.markdown(f"**Tool call:** `{tc['name']}`")
                            st.json(tc["args"])
                    if preview:
                        st.markdown("**Agent reasoning preview:**")
                        st.markdown(f"> {preview}")

        # Also show raw message history
        with st.expander("Raw message history (JSON)"):
            for msg in messages:
                st.json({
                    "type": type(msg).__name__,
                    "content": (msg.content or "")[:500],
                    "tool_calls": getattr(msg, "tool_calls", None),
                    "tool_call_id": getattr(msg, "tool_call_id", None),
                })

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("How this agent works"):
    st.markdown(
        """
        **Architecture: LangGraph ReAct agent**

        ```
        START
          │
          ▼
        ┌─────────────────────────────────────────────────┐
        │  agent_node                                     │
        │  LLM (Llama 3.3 70B) with bound tools          │
        │  Decides: call a tool, or synthesise final report│
        └──────────┬──────────────────────────────────────┘
                   │ tool calls?
           yes ────┘                      no ──► END (final report)
           │
           ▼
        ┌─────────────────────────────────────────────────┐
        │  tools_node                                     │
        │  Executes: get_live_odds / get_sports_news /    │
        │            get_weather                          │
        └──────────┬──────────────────────────────────────┘
                   │
                   └──────────► back to agent_node
        ```

        **Tools:**
        - `get_live_odds(sport_key)` — The Odds API · moneyline, spreads, totals across bookmakers · flags discrepancies ≥1.5 pts
        - `get_sports_news(league)` — ESPN public API · injury reports auto-flagged
        - `get_weather(city)` — OpenWeatherMap · flags wind >15 mph, freezing temps, precipitation

        **Signal types detected:** Line Movement · Injury Alert · Weather Impact · Value Spot
        """
    )
