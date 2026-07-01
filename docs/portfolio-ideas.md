# Portfolio Project Ideas — Sportsbet AI Lead Engineer

## Already Built
- Responsible AI Content Evaluation Pipeline (Streamlit)
- LangGraph Multi-Agent Evaluation Orchestrator
- Model Observability & Drift Monitoring Layer

---

## Additional Ideas

### 1. Responsible Gambling Early Warning System
Detect at-risk customers from behavioral patterns — chasing losses, late-night sessions, sudden bet-size escalation, deposit frequency spikes. A classifier trained on synthetic behavioral data with a SHAP explainability layer showing *why* a customer was flagged.

**Why it matters:** Directly maps to Sportsbet's regulatory obligations under ACMA/state gambling codes. Extends the responsible AI narrative to a real compliance problem in the gambling domain.

**Tech:** scikit-learn / XGBoost, SHAP, synthetic behavioral data, Streamlit dashboard.

---

### 2. Live Sports Intelligence Agent
A LangGraph agent that ingests real-time sports feeds (public APIs — ESPN, The Odds API, sports news RSS) and surfaces structured betting signals: injury alerts, line movement anomalies, weather impacts on outdoor events.

**Why it matters:** Shows domain-relevant agentic reasoning using LangGraph in a Sportsbet-native context.

**Tech:** LangGraph, LangChain, public sports/odds APIs, structured output parsing.

---

### 3. Customer Churn Prediction with Explainability
ML model predicting which customers are at risk of lapsing, with SHAP/LIME per-customer explanations. Adds genuine ML lifecycle depth — feature engineering, model training, serving — that the content evaluation pipeline lacks.

**Why it matters:** Extends the responsible AI theme with explainability, and addresses ML lifecycle depth (a stated gap in the fit analysis).

**Tech:** scikit-learn / XGBoost, SHAP / LIME, synthetic customer data, Streamlit.

---

### 4. Bet Anomaly & Fraud Detection
Unsupervised anomaly detection on betting patterns to flag potential match-fixing signals or arbitrage exploitation.

**Why it matters:** Addresses a real risk Sportsbet actively manages. Shows ML engineering breadth beyond LLM-based work.

**Tech:** Isolation Forest or autoencoder, synthetic betting data, explainability layer.

---

## Recommended Priority

1. **Responsible Gambling Early Warning System** — strongest differentiator; industry-specific, regulation-driven, extends responsible AI identity
2. **Live Sports Intelligence Agent** — best LangGraph showcase in a Sportsbet-native domain
3. **Customer Churn Prediction** — plugs ML lifecycle gap from fit analysis
4. **Bet Anomaly Detection** — broadens ML breadth, lower priority
