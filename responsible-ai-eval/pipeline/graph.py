from langgraph.graph import StateGraph, END
from .state import PipelineState
from .nodes import (
    excluded_check_node,
    classifier_node,
    evaluator_node,
    scorer_node,
    router_node,
    refiner_node,
)


def _route_excluded(state: PipelineState) -> str:
    return "end" if state["decision"] == "block" else "classifier"


def _route_decision(state: PipelineState) -> str:
    decision = state["decision"]
    if decision == "pass":
        return "end"
    if decision == "block":
        return "end"
    return "refiner"  # refine


def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("excluded_check", excluded_check_node)
    graph.add_node("classifier", classifier_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("router", router_node)
    graph.add_node("refiner", refiner_node)

    graph.set_entry_point("excluded_check")

    graph.add_conditional_edges(
        "excluded_check",
        _route_excluded,
        {"end": END, "classifier": "classifier"},
    )

    graph.add_edge("classifier", "evaluator")
    graph.add_edge("evaluator", "scorer")
    graph.add_edge("scorer", "router")

    graph.add_conditional_edges(
        "router",
        _route_decision,
        {"end": END, "refiner": "refiner"},
    )

    # Refiner loops back to evaluator for re-scoring
    graph.add_edge("refiner", "evaluator")

    return graph.compile()


pipeline = build_pipeline()
