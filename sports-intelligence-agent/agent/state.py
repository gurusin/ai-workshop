from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    sport_key: str
    sport_label: str
    query: str
    trace: List[dict]
