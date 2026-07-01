from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

from .state import AgentState
from .nodes import agent_node, TOOLS


def build_agent():
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def run_query(sport_key: str, sport_label: str, query: str) -> AgentState:
    initial: AgentState = {
        "messages": [HumanMessage(content=query)],
        "sport_key": sport_key,
        "sport_label": sport_label,
        "query": query,
        "trace": [],
    }
    return get_agent().invoke(initial, config={"recursion_limit": 12})
