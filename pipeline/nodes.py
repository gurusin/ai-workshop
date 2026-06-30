import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .state import PipelineState
from .prompts import (
    CLASSIFIER_PROMPT,
    HARM_EVALUATOR_PROMPT,
    FAIRNESS_EVALUATOR_PROMPT,
    COMPLIANCE_EVALUATOR_PROMPT,
    TONE_EVALUATOR_PROMPT,
    REFINER_PROMPT,
)

WEIGHTS = {"harm": 0.35, "fairness": 0.25, "compliance": 0.25, "tone": 0.15}

_config = {
    "provider": "groq",
    "fast_model": "llama-3.1-8b-instant",
    "smart_model": "llama-3.3-70b-versatile",
}

_fast = None
_smart = None


def configure(provider: str, fast_model: str, smart_model: str):
    global _config, _fast, _smart
    _config = {"provider": provider, "fast_model": fast_model, "smart_model": smart_model}
    _fast = None
    _smart = None


def _build_llm(model: str, temperature: float):
    provider = _config["provider"]
    kwargs = {"temperature": temperature}

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, max_tokens=512, **kwargs)

    if provider == "cerebras":
        import os
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            max_tokens=512,
            api_key=os.environ.get("CEREBRAS_API_KEY"),
            base_url="https://api.cerebras.ai/v1",
            **kwargs,
        )

    if provider == "sambanova":
        import os
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            max_tokens=512,
            api_key=os.environ.get("SAMBANOVA_API_KEY"),
            base_url="https://api.sambanova.ai/v1",
            **kwargs,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, num_predict=512, **kwargs)

    raise ValueError(f"Unknown provider: {provider}")


def _get_fast():
    global _fast
    if _fast is None:
        _fast = _build_llm(_config["fast_model"], temperature=0)
    return _fast


def _get_smart():
    global _smart
    if _smart is None:
        _smart = _build_llm(_config["smart_model"], temperature=0.3)
    return _smart


def _invoke(llm, prompt: str) -> str:
    for attempt in range(6):
        try:
            return llm.invoke(prompt).content
        except Exception as e:
            msg = str(e).lower()
            if "rate_limit" in msg or "429" in str(e) or "rate limit" in msg:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
            else:
                raise


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON: {text[:200]}")


def excluded_check_node(state: PipelineState) -> dict:
    if state["user_risk_profile"] == "self_excluded":
        return {
            "decision": "block",
            "block_reason": "User is self-excluded from wagering. No promotional content may be delivered under NCPF obligations.",
            "overall_score": 100.0,
            "harm_score": 0.0,
            "harm_reasoning": "Not evaluated — automatic block.",
            "fairness_score": 0.0,
            "fairness_reasoning": "Not evaluated — automatic block.",
            "compliance_score": 10.0,
            "compliance_reasoning": "Sending any promotional content to a self-excluded user is a direct NCPF violation.",
            "tone_score": 0.0,
            "tone_reasoning": "Not evaluated — automatic block.",
            "content_type": "promotional",
            "final_content": None,
            "evaluation_trace": [{"node": "excluded_check", "result": "auto_block", "reason": "self_excluded user"}],
        }
    return {"evaluation_trace": [{"node": "excluded_check", "result": "pass"}]}


def classifier_node(state: PipelineState) -> dict:
    prompt = CLASSIFIER_PROMPT.format(content=state["current_content"])
    result = _parse_json(_invoke(_get_fast(), prompt))
    content_type = result.get("content_type", "promotional")
    return {
        "content_type": content_type,
        "evaluation_trace": [{"node": "classifier", "content_type": content_type, "reason": result.get("reason", "")}],
    }


def evaluator_node(state: PipelineState) -> dict:
    content = state["current_content"]
    content_type = state["content_type"]
    risk = state["user_risk_profile"]
    deposit_count = state["deposit_count_today"]

    def eval_harm():
        prompt = HARM_EVALUATOR_PROMPT.format(
            content=content,
            content_type=content_type,
            user_name=state["user_name"],
            user_age=state["user_age"],
            risk_profile=risk,
            deposit_count=deposit_count,
        )
        return _parse_json(_invoke(_get_fast(), prompt))

    def eval_fairness():
        prompt = FAIRNESS_EVALUATOR_PROMPT.format(
            content=content, content_type=content_type, risk_profile=risk
        )
        return _parse_json(_invoke(_get_fast(), prompt))

    def eval_compliance():
        prompt = COMPLIANCE_EVALUATOR_PROMPT.format(
            content=content, content_type=content_type, risk_profile=risk, deposit_count=deposit_count
        )
        return _parse_json(_invoke(_get_fast(), prompt))

    def eval_tone():
        prompt = TONE_EVALUATOR_PROMPT.format(
            content=content, content_type=content_type, risk_profile=risk
        )
        return _parse_json(_invoke(_get_fast(), prompt))

    tasks = {
        "harm": eval_harm,
        "fairness": eval_fairness,
        "compliance": eval_compliance,
        "tone": eval_tone,
    }

    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): key for key, fn in tasks.items()}
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()

    return {
        "harm_score": float(results["harm"].get("score", 5)),
        "harm_reasoning": results["harm"].get("reasoning", ""),
        "fairness_score": float(results["fairness"].get("score", 5)),
        "fairness_reasoning": results["fairness"].get("reasoning", ""),
        "compliance_score": float(results["compliance"].get("score", 5)),
        "compliance_reasoning": results["compliance"].get("reasoning", ""),
        "tone_score": float(results["tone"].get("score", 5)),
        "tone_reasoning": results["tone"].get("reasoning", ""),
        "evaluation_trace": [
            {
                "node": "evaluator",
                "iteration": state["iteration_count"],
                "scores": {
                    "harm": results["harm"].get("score"),
                    "fairness": results["fairness"].get("score"),
                    "compliance": results["compliance"].get("score"),
                    "tone": results["tone"].get("score"),
                },
            }
        ],
    }


def scorer_node(state: PipelineState) -> dict:
    raw = (
        WEIGHTS["harm"] * state["harm_score"]
        + WEIGHTS["fairness"] * state["fairness_score"]
        + WEIGHTS["compliance"] * state["compliance_score"]
        + WEIGHTS["tone"] * state["tone_score"]
    )
    overall = round(raw * 10, 1)
    return {
        "overall_score": overall,
        "evaluation_trace": [{"node": "scorer", "overall_score": overall}],
    }


def router_node(state: PipelineState) -> dict:
    score = state["overall_score"]
    iteration = state["iteration_count"]

    if score <= 30:
        return {
            "decision": "pass",
            "final_content": state["current_content"],
            "evaluation_trace": [{"node": "router", "decision": "pass", "score": score}],
        }

    if score <= 60:
        if iteration >= 3:
            return {
                "decision": "block",
                "block_reason": f"Content could not be made compliant after {iteration} refinement attempts.",
                "final_content": None,
                "evaluation_trace": [{"node": "router", "decision": "block", "reason": "max_iterations", "score": score}],
            }
        return {
            "decision": "refine",
            "evaluation_trace": [{"node": "router", "decision": "refine", "score": score, "iteration": iteration}],
        }

    return {
        "decision": "block",
        "block_reason": "Content risk score too high to refine safely.",
        "final_content": None,
        "evaluation_trace": [{"node": "router", "decision": "block", "reason": "high_score", "score": score}],
    }


def refiner_node(state: PipelineState) -> dict:
    prompt = REFINER_PROMPT.format(
        original_content=state["original_content"],
        content_type=state["content_type"],
        risk_profile=state["user_risk_profile"],
        harm_score=state["harm_score"],
        harm_reasoning=state["harm_reasoning"],
        fairness_score=state["fairness_score"],
        fairness_reasoning=state["fairness_reasoning"],
        compliance_score=state["compliance_score"],
        compliance_reasoning=state["compliance_reasoning"],
        tone_score=state["tone_score"],
        tone_reasoning=state["tone_reasoning"],
    )
    refined = _invoke(_get_smart(), prompt).strip()
    new_iteration = state["iteration_count"] + 1
    return {
        "current_content": refined,
        "refined_content": refined,
        "iteration_count": new_iteration,
        "evaluation_trace": [{"node": "refiner", "iteration": new_iteration, "refined_content": refined}],
    }
