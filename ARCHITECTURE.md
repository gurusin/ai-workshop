# Responsible AI Evaluation Pipeline — Architecture

## Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (`app.py`) |
| Orchestration | LangGraph `StateGraph` (`pipeline/graph.py`) |
| LLM — fast evaluators | Llama 3.1 8B via Groq (`llama-3.1-8b-instant`) |
| LLM — refiner | Llama 3.3 70B via Groq (`llama-3.3-70b-versatile`) |
| State contract | `PipelineState` TypedDict (`pipeline/state.py`) |
| Prompts | `pipeline/prompts.py` |

---

## Pipeline Flow

```
INPUT: message + user profile
          │
          ▼
┌─────────────────────┐
│  excluded_check     │  Pure code — no AI
│                     │  Is user self-excluded?
└──────┬──────────────┘
       │ YES → BLOCK immediately (NCPF obligation)
       │ NO
       ▼
┌─────────────────────┐
│  classifier         │  Llama 8B
│                     │  What type of content is this?
│                     │  → promotional / odds /
│                     │    recommendation / general
└──────┬──────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│  evaluator  (4 parallel LLM calls)      │  Llama 8B × 4
│                                         │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │   Harm   │  │      Fairness        │ │
│  │  35% wt  │  │       25% wt         │ │
│  └──────────┘  └──────────────────────┘ │
│  ┌──────────┐  ┌──────────────────────┐ │
│  │Compliance│  │        Tone          │ │
│  │  25% wt  │  │       15% wt         │ │
│  └──────────┘  └──────────────────────┘ │
│                                         │
│  Each returns: { score: 0–10,           │
│                  reasoning: "..." }     │
└──────┬──────────────────────────────────┘
       │
       ▼
┌─────────────────────┐
│  scorer             │  Pure maths — no AI
│                     │  overall = (harm×0.35 + fairness×0.25
│                     │           + compliance×0.25 + tone×0.15) × 10
│                     │  → 0–100 risk score
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│  router             │  Pure code — no AI
│                     │  ≤ 30  → PASS
│                     │  31–60 → REFINE (or BLOCK if 3 attempts)
│                     │  > 60  → BLOCK
└──┬───────┬──────────┘
   │PASS   │BLOCK         REFINE
   ▼       ▼               │
 [END]   [END]             ▼
                  ┌─────────────────────┐
                  │  refiner            │  Llama 3.3 70B
                  │                     │  Rewrites the message
                  │                     │  using all 4 evaluator
                  │                     │  reasoning strings
                  └──────┬──────────────┘
                         │ loops back to evaluator
                         │ (max 3 iterations, then BLOCK)
                         ▼
                    [evaluator]
```

---

## Node Reference

### `excluded_check_node`
- **Type:** Pure code
- **Logic:** If `user_risk_profile == "self_excluded"` → immediately sets `decision = block`. No LLM called.
- **Why:** NCPF (National Consumer Protection Framework) requires this to be an absolute, unconditional block.

---

### `classifier_node`
- **Model:** Llama 3.1 8B
- **Input:** The message text
- **Output:** `content_type` — one of `promotional | odds | recommendation | general`
- **Why it matters:** The content type is injected into every evaluator prompt so each evaluator knows what it's judging (a bonus offer is held to different standards than an odds explanation).

---

### `evaluator_node`
- **Model:** Llama 3.1 8B × 4 (parallel)
- **Runs:** 4 simultaneous LLM calls via `ThreadPoolExecutor`
- **Each call returns:** `{ "score": 0–10, "reasoning": "<natural language explanation>" }`

| Evaluator | What it checks | Weight |
|---|---|---|
| **Harm** | Chasing losses, exploiting vulnerable users, false urgency that targets impulse control | 35% |
| **Fairness** | Implied guaranteed wins, misleading odds framing, hiding house edge, multi-leg risk not disclosed | 25% |
| **Compliance** | Missing NCPF footers, violations for at-risk/new users, inducements without terms disclosure | 25% |
| **Tone** | FOMO language (`"expires tonight"`, `"don't miss out"`), emotional manipulation, streak framing | 15% |

**Why the same message scores differently per user:**
The user's `risk_profile`, `age`, and `deposit_count_today` are injected into the harm and compliance prompts. The model is explicitly instructed: *"the same message scores higher for an at-risk user."* The LLM reasons about context — it is not matching rules.

---

### `scorer_node`
- **Type:** Pure maths
- **Formula:**
  ```
  overall = (harm×0.35 + fairness×0.25 + compliance×0.25 + tone×0.15) × 10
  ```
- **Output:** A single 0–100 risk score

---

### `router_node`
- **Type:** Pure code
- **Thresholds:**
  - `overall ≤ 30` → **PASS** — send original message
  - `31 ≤ overall ≤ 60` → **REFINE** — rewrite and re-evaluate (max 3 iterations, then BLOCK)
  - `overall > 60` → **BLOCK** — too risky to fix

---

### `refiner_node`
- **Model:** Llama 3.3 70B
- **Input:** Original message + all 4 evaluator scores + all 4 reasoning strings
- **Output:** A rewritten message that fixes every flagged issue while preserving marketing intent
- **Loop:** The refined message goes back to `evaluator_node` for re-scoring. If it passes (≤ 30), it is sent. If it still fails after 3 attempts, the pipeline blocks.

---

## Shared State

Every node reads from and writes to a single `PipelineState` TypedDict that LangGraph threads through the graph. Key fields:

```
original_content      The original AI-generated message (never mutated)
current_content       The working copy (replaced by refiner on each iteration)
user_*                User profile fields (name, age, risk_profile, deposit_count_today)
content_type          Set by classifier, read by all evaluators
harm/fairness/        Scores (0–10) and reasoning strings set by evaluator,
compliance/tone       read by scorer and refiner
overall_score         0–100 composite set by scorer, read by router
decision              pass | refine | block — set by router or excluded_check
refined_content       The latest rewritten message from the refiner
iteration_count       How many refine loops have run
evaluation_trace      Append-only log of every node's output (shown in Tab 3)
```

---

## How to prove it is AI and not hardcoded

1. **The refiner generates novel sentences** — no rewritten message exists anywhere in the codebase before runtime.
2. **Reasoning is natural language** — the `harm_reasoning`, `compliance_reasoning` etc. in the Details panel are written by the model, not looked up from a table.
3. **Same message, different reasoning per user** — the reasoning text for James vs Sarah is contextually different even though the code path is identical.
4. **Novel inputs work** — paste a message in another language or on an unrelated sport; the pipeline still scores and reasons correctly without any special-case code.
5. **Groq Activity log** — `console.groq.com → Activity` shows real API calls with timestamps matching every button click.
6. **Tab 3 (Evaluation Trace)** — shows the raw JSON the model returned, including reasoning text, for every node in the run.
