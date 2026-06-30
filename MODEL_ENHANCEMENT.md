# Model Evaluation & Enhancement Guide

## Part 1 — How does Llama evaluate wagering content without being trained on it?

**Groq is not the model — it is the inference engine.**

Groq is a chip manufacturer that built extremely fast AI inference hardware. The model running on it is **Meta's Llama**, trained on hundreds of billions of words of internet text, books, legal documents, news, academic papers, and regulatory filings.

By the time Llama finished training, it had already absorbed:

- Gambling harm research papers
- NCPF and responsible gambling legislation
- Advertising standards and consumer protection guidelines
- Marketing psychology literature (FOMO, urgency, social proof)
- Thousands of examples of manipulative vs neutral language

When our prompt says *"score this message for FOMO language and urgency pressure"* — Llama already knows what those concepts are. The prompt does not teach it; it **directs** knowledge the model already has.

```
Our prompt  →  "You are a harm assessor. Score urgency language 0–10."
                        │
                        ▼
              Llama already knows:
              - What urgency language looks like
              - Why it is harmful for at-risk users
              - What NCPF obligations are
              - What "Don't miss out" does psychologically
                        │
                        ▼
              Output: { "score": 7, "reasoning": "..." }
```

This is called **zero-shot prompting** — no examples needed, just instructions. It works because general-purpose LLMs have broad world knowledge baked in at training time.

**The limitation:** Llama knows NCPF in general, but it does not know your specific internal compliance rules, your past violation cases, your regulator's latest guidance, or your brand's tone standards. That is what Part 2 addresses.

---

## Part 2 — How do you teach it your local data?

There are three approaches, in order of complexity.

---

### Option A — Few-Shot Examples
**Effort:** Hours | **Infrastructure:** None | **Do it today**

Add real labeled examples directly into the prompt. The model pattern-matches against them.

```python
# pipeline/prompts.py — add examples to the harm evaluator prompt
HARM_EVALUATOR_PROMPT = """...

Examples from our past compliance reviews:

Message: "You're on a hot streak! Keep it going with a $100 deposit!"
User: at_risk → Score: 9
Reason: Streak language directly exploits momentum bias in a vulnerable user.

Message: "Tonight's NRL odds are now available."
User: at_risk → Score: 1
Reason: Factual, no emotional pressure, no urgency.

Now evaluate:
Content: {content}
..."""
```

**Pros**
- Takes 30 minutes to implement
- Costs nothing extra
- Immediately improves scoring consistency

**Cons**
- Longer prompts = more tokens = slightly higher cost
- Limited to ~5–10 examples before prompts become unwieldy

---

### Option B — RAG: Retrieval Augmented Generation
**Effort:** 1–2 days | **Infrastructure:** Vector database | **Recommended**

Keep the model weights unchanged. At runtime, pull relevant chunks from your own knowledge base and inject them into the prompt automatically.

```
Your local documents               Runtime pipeline
────────────────────               ──────────────────────────────────────
NCPF guidelines.pdf   ──embed──►  Vector DB  ──search──►  Top 3 relevant
Internal policy.docx              (Chroma /               chunks injected
Past violations.csv               Pinecone /              into the prompt
Brand tone guide.md               FAISS)                  before LLM call
Regulator rulings.pdf
```

**How it works**

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Step 1 — One-time: ingest your compliance documents
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectordb = Chroma.from_documents(your_compliance_docs, embedding_model)

# Step 2 — At evaluation time: retrieve relevant chunks
relevant_chunks = vectordb.similarity_search(content, k=3)

# Step 3 — Inject into the evaluator prompt
context = "\n".join([doc.page_content for doc in relevant_chunks])
prompt = f"Relevant compliance rules:\n{context}\n\nNow evaluate:\n{content}"
```

**What documents to ingest**

| Document | Why |
|---|---|
| NCPF guidelines | Exact regulatory language the model can cite |
| Past violations & decisions | Real examples with outcomes |
| Internal brand tone guide | Your specific standards, not generic ones |
| Regulator correspondence | Specific rulings that apply to your context |
| Product terms & conditions | What disclosures are actually required |

**Pros**
- Model stays unchanged — no training required
- Knowledge is always up to date (re-ingest when documents change)
- Traceable — you can see exactly which document drove a decision
- Compliance team can update knowledge base without touching code

**Cons**
- Adds ~200ms latency per evaluation call
- Requires embedding pipeline setup (~1 day of work)
- Needs a vector database running alongside the app

---

### Option C — Fine-Tuning
**Effort:** Weeks | **Infrastructure:** GPU / paid service | **Long-term investment**

Actually retrain the model's weights on your labeled data. The model internalises your standards rather than looking them up at runtime.

```
Your labeled dataset
────────────────────
{ message: "...", user: at_risk,  harm_score: 8, reasoning: "..." }  ×500 rows
{ message: "...", user: standard, harm_score: 2, reasoning: "..." }  ×500 rows
{ message: "...", user: new_user, harm_score: 5, reasoning: "..." }  ×500 rows
        │
        ▼
Fine-tuning run (QLoRA — parameter-efficient method)
        │
        ▼
Custom Llama weights that score
exactly as your compliance team would
```

**Practical path for this pipeline**

1. Run the pipeline in production for several weeks
2. Have your compliance team review and correct the AI scores
3. Build a dataset of `(message, user_profile, correct_score, correct_reasoning)` rows
4. Fine-tune Llama 3.1 8B using **QLoRA** (efficient — runs on a single GPU)
5. Host the fine-tuned model via Together AI, Modal, or locally via Ollama

**Fine-tuning services (no GPU required)**

| Service | Notes |
|---|---|
| Together AI | Upload dataset → fine-tune via UI or API |
| Modal | Serverless GPU, pay per minute |
| Hugging Face AutoTrain | Simple UI-driven fine-tuning |
| Replicate | Fine-tune and deploy in one place |

**Pros**
- Model becomes genuinely expert in your domain
- No longer needs long prompts to explain context
- Fastest inference — shorter prompts, lower latency

**Cons**
- Needs 500+ labeled examples to be worthwhile
- Model must be retrained when compliance rules change
- More complex deployment (self-hosted weights)

---

## Which approach should you choose?

```
Now (this week)      →  Option A: Add 5–10 past violation examples to each evaluator prompt
Next sprint          →  Option B: RAG pipeline ingesting NCPF docs + internal policy files
Long term (6+ mo)    →  Option C: Fine-tune once you have 500+ labeled compliance decisions
```

**RAG is the recommended sweet spot** — it delivers 80% of fine-tuning's domain accuracy at 10% of the effort. The knowledge base stays editable by compliance staff in plain documents, with no model retraining required when rules change.

---

## How the three approaches combine

In a mature production system, all three are used together:

```
Incoming message
      │
      ▼
RAG retrieves relevant NCPF chunks + past violation precedents
      │
      ▼
Prompt includes: retrieved context + few-shot examples + user profile
      │
      ▼
Fine-tuned Llama (already knows your internal standards)
      │
      ▼
Score + reasoning grounded in your actual compliance history
```

Each layer adds a different kind of knowledge:
- **Fine-tuning** → internalised domain expertise (style, thresholds)
- **RAG** → live, up-to-date facts and specific regulatory text
- **Few-shot** → concrete examples of edge cases and past decisions

---

## Part 3 — Self-Hosting: Owning Your Model

### What "open weights" means

Unlike GPT-4 or Claude, Meta publishes the actual model file — the billions of numerical parameters that make up Llama. You can download it, modify it, and run it anywhere. Groq is just one place to run it. Once you fine-tune, you own the resulting weight file outright and can host it yourself with no external dependency.

```
Meta releases          You fine-tune it        You host your copy
Llama 3.1 8B  ──────►  on your data   ──────►  on your own infra
(open weights)          (new weights)           (no internet dependency)
```

---

### Why self-hosting matters for a regulated gambling operator

| Concern | Groq (current) | Self-hosted fine-tuned model |
|---|---|---|
| Customer PII in prompts | Sent to Groq's servers | Never leaves your infra |
| Compliance audit trail | Groq's logs | Your logs, your control |
| Model behaviour changes | Groq can update the model | Your weights are frozen |
| Internet dependency | Yes | None — runs fully offline |
| Cost at scale | Per-token billing | Fixed infrastructure cost |
| Regulatory approval | Third-party AI vendor | First-party system |

For a regulated gambling operator, the self-hosted path is often **required** by legal and infosec teams once you move from prototype to production — because customer betting behaviour, risk profiles, and deposit data are all sensitive.

---

### Where to host your model

| Option | What it means |
|---|---|
| **On-premises GPU servers** | Runs in your data centre — fully air-gapped, zero cloud |
| **Private cloud (AWS / Azure / GCP)** | VPC-isolated, no data leaves your cloud account |
| **Ollama** | Already in this project — runs on a local machine, no setup |
| **vLLM** | Production-grade self-hosted inference server, high throughput |
| **TGI (Hugging Face)** | Another production inference server, Docker-based |

---

### The phased path to full self-hosting

```
Phase 1 — Prototype (now)
  Use Groq — fast to iterate, validate the concept, no infra cost

Phase 2 — Fine-tuning
  Collect 500+ labeled evaluations from your compliance team
  Run QLoRA fine-tuning via Together AI or Modal (they supply the GPU)
  Receive your custom weight files back

Phase 3 — Production (self-hosted)
  Download your fine-tuned weights
  Deploy on internal GPU servers using vLLM
  Point the pipeline at your internal endpoint
  Zero external API calls — fully offline capable
```

---

### The code change is minimal

Switching from Groq to your self-hosted model requires changing only a few lines in `pipeline/nodes.py`. vLLM exposes an OpenAI-compatible API so all LangChain code works unchanged.

```python
# Current — calls Groq's servers
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.1-8b-instant")

# Self-hosted — calls your internal GPU server
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="sportsbet-compliance-llama-v1",   # your fine-tuned model name
    base_url="http://your-internal-gpu-server:8000/v1",
    api_key="none"                            # no external key — it is yours
)
```

Everything else — the LangGraph graph, the evaluator nodes, the Streamlit UI — stays identical.

---

### Full production architecture

```
                    ┌─────────────────────────────────┐
                    │         Streamlit UI             │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │        LangGraph pipeline        │
                    │   excluded_check → classifier    │
                    │   → evaluator → scorer → router  │
                    │   → refiner (loop)               │
                    └────────────────┬────────────────┘
                                     │ HTTP (internal network only)
                    ┌────────────────▼────────────────┐
                    │     vLLM inference server        │
                    │     (your data centre / VPC)     │
                    │                                  │
                    │  sportsbet-compliance-llama-v1   │
                    │  ← your fine-tuned weights       │
                    └────────────────┬────────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │     ChromaDB vector store        │
                    │     (RAG — NCPF docs,            │
                    │      past violations,            │
                    │      internal policy)            │
                    └─────────────────────────────────┘

No traffic ever leaves your network.
```
