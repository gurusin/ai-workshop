from typing import TypedDict, Optional, List, Annotated
import operator


class PipelineState(TypedDict):
    # Input
    original_content: str
    current_content: str
    user_name: str
    user_age: int
    user_risk_profile: str  # standard | at_risk | self_excluded | new_user | vip
    deposit_count_today: int

    # Classification
    content_type: str  # promotional | odds | recommendation | general

    # Evaluation scores (0–10 each)
    harm_score: float
    harm_reasoning: str
    fairness_score: float
    fairness_reasoning: str
    compliance_score: float
    compliance_reasoning: str
    tone_score: float
    tone_reasoning: str

    # Aggregated (0–100)
    overall_score: float

    # Decision
    decision: str  # pass | refine | block
    block_reason: Optional[str]

    # Refinement loop
    iteration_count: int
    refined_content: Optional[str]

    # Final output
    final_content: Optional[str]

    # Trace accumulates across all nodes (operator.add appends each entry)
    evaluation_trace: Annotated[List[dict], operator.add]
