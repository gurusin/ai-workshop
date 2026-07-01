import os
from langchain_core.messages import SystemMessage
from .state import AgentState
from .tools import get_live_odds, get_sports_news, get_weather
from .prompts import SYSTEM_PROMPT

TOOLS = [get_live_odds, get_sports_news, get_weather]

_config: dict = {"provider": "groq", "model": "llama-3.3-70b-versatile"}
_llm_with_tools = None


def configure(provider: str, model: str) -> None:
    global _config, _llm_with_tools
    _config = {"provider": provider, "model": model}
    _llm_with_tools = None


def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is not None:
        return _llm_with_tools

    provider = _config["provider"]
    model = _config["model"]

    if provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, temperature=0, max_tokens=2048)

    elif provider == "sambanova":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=2048,
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
        )

    elif provider == "cerebras":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model,
            temperature=0,
            max_tokens=2048,
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            base_url="https://api.cerebras.ai/v1",
        )

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model=model, temperature=0, num_predict=2048)

    else:
        raise ValueError(f"Unknown provider: {provider}")

    _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = _get_llm().invoke(messages)

    trace_entry = {
        "node": "agent",
        "tool_calls": [
            {"name": tc["name"], "args": tc["args"]}
            for tc in (response.tool_calls or [])
        ],
        "content_preview": (response.content or "")[:300],
    }

    return {
        "messages": [response],
        "trace": (state.get("trace") or []) + [trace_entry],
    }
