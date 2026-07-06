import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import streamlit as st

from config import ROOT
from utils.module_loader import load_pkg_module
from utils.ui import render_header


def render_churn() -> None:
    render_header("Customer Churn Prediction")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>📉 Customer Churn Prediction with Explainability</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#f0fdf4;border:1px solid #86efac;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#166534;">
                &#9711;&nbsp; Proprietary model &mdash; no external API
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#eff6ff;border:1px solid #93c5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#1e40af;">
                &#128274;&nbsp; Trained on private synthetic data
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#9881;&nbsp; XGBoost &middot; SHAP &middot; LIME &mdash; runs fully local
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0a1628 0%,#1a3a5c 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#7fb3d3;margin-bottom:8px;">Business Problem</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                Customer lifetime value erosion — caught too late to act
            </div>
            <div style="font-size:0.88rem;color:#c8ddef;line-height:1.75;max-width:860px;">
                A customer is acquired at a measurable CAC. They bet regularly for months,
                then gradually go quiet — fewer sessions, smaller stakes, longer gaps between bets.
                By the time the 30-day inactivity threshold is crossed and the customer is formally
                <em>churned</em>, they have often already moved to a competitor and no retention offer
                will bring them back. The window to act closes before the signal is obvious.
                This system scores every customer on <strong>15 behavioural signals</strong> — recency,
                frequency trend, stake trajectory, and engagement depth — and surfaces a churn probability
                <strong>2–4 weeks before the customer lapses</strong>, when a targeted offer, CRM outreach,
                or promotional adjustment can still change the outcome. Every prediction is accompanied by
                a per-customer explanation so CRM teams know <em>why</em> someone was flagged, not just
                <em>that</em> they were.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#38bdf8;">2–4 weeks</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        lead time before confirmed churn — when retention action is still viable
                    </div>
                </div>
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#34d399;">15 signals</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        behavioural features per customer across recency, frequency, monetary and engagement
                    </div>
                </div>
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f59e0b;">per-customer</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        SHAP + LIME explanations — CRM sees the exact signals that drove each flag
                    </div>
                </div>
                <div style="flex:1;min-width:175px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f472b6;">auditable</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        rationale for every prediction — supports compliance review and model governance
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _pc1, _pc2, _pc3 = st.columns(3)

    with _pc1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);height:100%;">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Synthetic Training Data
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">
                    600 customers &middot; 15 features &middot; no real PII
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:10px;">
                    Real customer behavioural data cannot be published. The generator encodes genuine
                    domain logic &mdash; rising recency, declining frequency, and support escalations
                    increase churn probability &mdash; so the model behaves as it would in production
                    without exposing any customer records.
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    Churn ground truth is derived from a logistic function over feature values, then
                    Gaussian noise is added to prevent the label from being perfectly separable.
                    The result is a realistic class distribution (~33% churned) with a model that
                    generalises rather than memorises.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;
                            border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;">
                        <span style="color:#7c3aed;font-weight:700;">&#9679;</span>
                        <strong>Recency</strong> &mdash; days since last bet
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#2563eb;font-weight:700;">&#9679;</span>
                        <strong>Frequency</strong> &mdash; bets &amp; sessions, last 30d vs prior 30d
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#059669;font-weight:700;">&#9679;</span>
                        <strong>Monetary</strong> &mdash; avg stake, stake size trend
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#d97706;font-weight:700;">&#9679;</span>
                        <strong>Engagement</strong> &mdash; live bet %, sport diversity, promo redemption
                    </div>
                    <div>
                        <span style="color:#dc2626;font-weight:700;">&#9679;</span>
                        <strong>Friction</strong> &mdash; support contacts, deposit count
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _pc2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);height:100%;">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    ML Architecture
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    XGBoost &middot; SHAP TreeExplainer &middot; LIME Tabular
                </div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0a1628;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">1</div>
                        <div><strong>Synthetic data generation</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                600 customers &middot; logistic churn label &middot; Gaussian noise
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1a3a5c;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">2</div>
                        <div><strong>Feature engineering</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                frequency_trend = bets_last &divide; bets_prev
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#2563eb;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">3</div>
                        <div><strong>XGBoost classifier</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                300 trees &middot; depth 5 &middot; LR 0.08 &middot; 80/20 split
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">4</div>
                        <div><strong>SHAP TreeExplainer</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                global importance &middot; per-customer waterfall
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#059669;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">5</div>
                        <div><strong>LIME Tabular</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                local linear approximation &middot; per-customer weights
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#d97706;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;margin-top:1px;">6</div>
                        <div><strong>Streamlit serving</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                cohort table &middot; deep-dive &middot; what-if &middot; dashboard
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _pc3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);height:100%;">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Design Justification
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    Every technical choice justified by the domain
                </div>
                <div style="font-size:0.82rem;color:#374151;line-height:1.65;">
                    <div style="margin-bottom:13px;padding-bottom:13px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#ede9fe;color:#5b21b6;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">XGBoost</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            Handles mixed feature types and non-linear interactions natively &mdash;
                            the dominant algorithm for tabular prediction in production environments.
                        </div>
                    </div>
                    <div style="margin-bottom:13px;padding-bottom:13px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#dbeafe;color:#1d4ed8;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">SHAP</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            Globally consistent Shapley values. Regulators and risk teams can ask
                            &ldquo;why was this customer flagged?&rdquo; and receive a theoretically
                            grounded, reproducible answer.
                        </div>
                    </div>
                    <div style="margin-bottom:13px;padding-bottom:13px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#d1fae5;color:#065f46;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">LIME</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            Model-agnostic local explanation &mdash; useful as a cross-check and
                            intuitive for teams who trust linear &ldquo;if&ndash;then&rdquo; reasoning.
                        </div>
                    </div>
                    <div>
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">Percentile tiers</span>
                        <div style="margin-top:6px;color:#4b5563;">
                            High/Medium/Low use cohort-relative percentile cuts, not fixed thresholds
                            &mdash; guaranteeing an actionable cohort regardless of overall churn rate.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:28px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="border:1px solid #e5e7eb;border-radius:13px;padding:26px 28px 22px;
                    background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:28px;">
            <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
                        color:#9ca3af;margin-bottom:6px;">Technical Implementation</div>
            <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:18px;">
                End-to-end ML pipeline — from synthetic data to per-customer explanation
            </div>
            <div style="font-family:monospace;font-size:0.82rem;color:#374151;
                        background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
                        padding:18px 22px;line-height:1.9;overflow-x:auto;white-space:pre;">
DATASET 1 — Training & Evaluation  (seed=42 · 600 customers)
  · Logistic churn label + Gaussian noise → realistic ~33% churn rate
  · 480 customers → XGBoost training  (300 trees · depth 5 · LR 0.08)
  · 120 customers → held-out test → AUC · precision · recall · F1
  · SHAP TreeExplainer and LIME Tabular fitted on training distribution
          │
          ▼  model weights locked — no further training
          │
DATASET 2 — Business Scoring  (seed=99 · 400 customers · unseen)
  · Completely separate synthetic population, never seen during training
  · model.predict_proba() → churn score per customer
  · Percentile tiers → High (top 30%) · Medium · Low (bottom 40%)
  · SHAP values computed → per-customer waterfall explanations
          │
          ▼
Streamlit UI
  · Cohort risk table — Dataset 2 customers ranked by churn score
  · Customer deep-dive — SHAP waterfall + LIME bar chart
  · What-if simulator — re-score on modified feature values
  · Model performance dashboard — metrics from Dataset 1 test split
            </div>
            <div style="display:flex;gap:16px;margin-top:18px;flex-wrap:wrap;">
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;">Stack</div>
                    <div style="font-size:0.82rem;color:#374151;line-height:1.8;">
                        Python · XGBoost · SHAP · LIME<br>scikit-learn · pandas · Streamlit
                    </div>
                </div>
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;">Key files</div>
                    <div style="font-size:0.82rem;color:#374151;line-height:1.8;">
                        <code>model/data.py</code> — data generation<br>
                        <code>model/train.py</code> — training pipeline<br>
                        <code>app.py</code> — Streamlit serving
                    </div>
                </div>
                <div style="flex:1;min-width:160px;">
                    <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:6px;">Explainability</div>
                    <div style="font-size:0.82rem;color:#374151;line-height:1.8;">
                        SHAP: game-theoretic, auditable<br>
                        LIME: linear, human-readable<br>
                        Both per-customer, real-time
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="Training model on synthetic customer data…")
    def _load_churn():
        cp_train = load_pkg_module("churn_prediction_model", os.path.join(ROOT, "churn-prediction", "model"), "train")
        return cp_train.build()

    artifact = _load_churn()
    df = artifact["df"]

    cp_data = load_pkg_module("churn_prediction_model", os.path.join(ROOT, "churn-prediction", "model"), "data")
    FEATURE_NAMES   = cp_data.FEATURE_NAMES
    FEATURE_DISPLAY = cp_data.FEATURE_DISPLAY
    FEATURE_RANGES  = cp_data.FEATURE_RANGES

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cohort Risk Overview",
        "🔍 Customer Deep-Dive",
        "🎛️ What-If Simulator",
        "📈 Model Performance",
    ])

    with tab1:
        total = len(df)
        high = int((df["risk_segment"] == "High").sum())
        med  = int((df["risk_segment"] == "Medium").sum())
        low  = int((df["risk_segment"] == "Low").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", f"{total:,}")
        m2.metric("High Risk", f"{high}", delta=f"{high/total:.0%} of cohort", delta_color="inverse")
        m3.metric("Medium Risk", f"{med}", delta=f"{med/total:.0%} of cohort", delta_color="off")
        m4.metric("Low Risk", f"{low}", delta=f"{low/total:.0%} of cohort", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        f1, f2 = st.columns([2, 1])
        with f1:
            risk_filter = st.multiselect(
                "Filter by risk segment", ["High", "Medium", "Low"],
                default=["High", "Medium", "Low"], key="cohort_risk_filter",
            )
        with f2:
            sort_col = st.selectbox(
                "Sort by", ["Churn Risk %", "Days Since Last Bet", "Bets (30d)"],
                key="cohort_sort",
            )

        sort_map = {
            "Churn Risk %": "churn_pct",
            "Days Since Last Bet": "days_since_last_bet",
            "Bets (30d)": "bets_last_30d",
        }
        sort_asc = sort_col == "Bets (30d)"

        filtered = df[df["risk_segment"].isin(risk_filter)].sort_values(
            sort_map[sort_col], ascending=sort_asc
        )
        display = filtered[[
            "customer_id", "risk_segment", "churn_pct",
            "days_since_last_bet", "bets_last_30d", "avg_stake_last_30d",
            "sessions_last_30d", "bet_frequency_trend",
        ]].copy()
        display.columns = [
            "Customer ID", "Risk", "Churn Risk %",
            "Days Since Bet", "Bets (30d)", "Avg Stake ($)",
            "Sessions (30d)", "Freq Trend",
        ]

        st.dataframe(
            display,
            column_config={
                "Churn Risk %": st.column_config.ProgressColumn(
                    "Churn Risk %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "Risk": st.column_config.TextColumn("Risk"),
                "Avg Stake ($)": st.column_config.NumberColumn("Avg Stake ($)", format="$%.2f"),
                "Freq Trend": st.column_config.NumberColumn("Freq Trend", format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=420,
        )
        st.caption(
            "**Churn Risk %** — model probability (0–100). "
            "**Freq Trend** — bets last 30d ÷ prior 30d; values below 1.0 indicate declining activity."
        )

    with tab2:
        sorted_ids = df.sort_values("churn_pct", ascending=False)["customer_id"].tolist()
        selected_id = st.selectbox(
            "Select customer (sorted by churn risk, highest first)",
            sorted_ids, key="deepdive_customer",
        )

        row = df[df["customer_id"] == selected_id].iloc[0]
        cust_idx = df.index[df["customer_id"] == selected_id][0]
        prob = row["churn_pct"]
        risk = str(row["risk_segment"])
        risk_color = {"High": "#dc3545", "Medium": "#ffc107", "Low": "#198754"}.get(risk, "#6c757d")
        risk_bg    = {"High": "#f8d7da", "Medium": "#fff3cd", "Low": "#d1e7dd"}.get(risk, "#f8f9fa")

        st.markdown(
            f"""
            <div style="background:{risk_bg};border:1.5px solid {risk_color};border-radius:10px;
                        padding:16px 20px;margin-bottom:20px;display:flex;align-items:center;gap:24px;">
                <div style="text-align:center;min-width:90px;">
                    <div style="font-size:2rem;font-weight:800;color:{risk_color};">{prob:.1f}%</div>
                    <div style="font-size:0.78rem;color:{risk_color};font-weight:600;">Churn Risk</div>
                </div>
                <div style="border-left:2px solid {risk_color};height:40px;opacity:0.4;"></div>
                <div>
                    <div style="font-size:1.05rem;font-weight:700;">{selected_id}</div>
                    <span style="background:{risk_color};color:white;padding:2px 10px;border-radius:12px;
                                 font-size:0.75rem;font-weight:700;">{risk.upper()} RISK</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        avg = df[FEATURE_NAMES].mean()
        profile_cols = st.columns(5)
        key_feats = [
            ("days_since_last_bet",       "Days Since Bet",  "{:.0f}"),
            ("bets_last_30d",             "Bets (30d)",      "{:.0f}"),
            ("sessions_last_30d",         "Sessions (30d)",  "{:.0f}"),
            ("bet_frequency_trend",       "Freq Trend",      "{:.2f}"),
            ("support_contacts_last_90d", "Support (90d)",   "{:.0f}"),
        ]
        for col, (feat, label, fmt) in zip(profile_cols, key_feats):
            val = row[feat]
            avg_val = avg[feat]
            col.metric(label, fmt.format(val), f"{val - avg_val:+.1f} vs avg")

        st.markdown("<br>", unsafe_allow_html=True)

        shap_col, lime_col = st.columns(2)

        with shap_col:
            st.markdown("**SHAP Waterfall — why this customer was scored here**")
            st.caption("Red bars push churn probability up; blue bars push it down.")
            try:
                plt.close("all")
                shap.plots.waterfall(artifact["shap_values"][cust_idx], max_display=12, show=False)
                fig = plt.gcf()
                fig.set_size_inches(6, 5)
                st.pyplot(fig, use_container_width=True)
                plt.close("all")
            except Exception as exc:
                st.warning(f"SHAP plot unavailable: {exc}")

        with lime_col:
            st.markdown("**LIME Local Explanation — linear approximation around this customer**")
            st.caption("Positive weight (red) increases churn probability; negative (green) decreases it.")
            try:
                cust_features = df[FEATURE_NAMES].iloc[cust_idx].values
                lime_exp = artifact["lime_explainer"].explain_instance(
                    cust_features,
                    artifact["model"].predict_proba,
                    num_features=12,
                )
                lime_list = lime_exp.as_list()
                feats_lime  = [f[0] for f in lime_list]
                vals_lime   = [f[1] for f in lime_list]
                colors_lime = ["#dc3545" if v > 0 else "#198754" for v in vals_lime]

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.barh(feats_lime, vals_lime, color=colors_lime, edgecolor="none")
                ax2.axvline(0, color="black", linewidth=0.8)
                ax2.set_xlabel("LIME weight")
                ax2.set_title("Local Feature Contributions", fontsize=10)
                ax2.tick_params(axis="y", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig2, use_container_width=True)
                plt.close("all")
            except Exception as exc:
                st.warning(f"LIME plot unavailable: {exc}")

        with st.expander("Full feature profile for this customer"):
            import pandas as _pd
            profile_data = {
                FEATURE_DISPLAY.get(f, f): [row[f], round(avg[f], 2)]
                for f in FEATURE_NAMES
            }
            profile_df = _pd.DataFrame(profile_data, index=["This Customer", "Cohort Avg"]).T
            st.dataframe(profile_df, use_container_width=True)

    with tab3:
        st.markdown(
            """
            <div style="background:#fff7ed;border-left:4px solid #f97316;border-radius:4px;
                        padding:10px 16px;margin-bottom:20px;font-size:0.88rem;color:#7c2d12;">
            Select a customer as your starting point, then adjust the sliders to model a retention
            intervention or predict how their score would shift with different behaviour.
            </div>
            """,
            unsafe_allow_html=True,
        )

        wi_id = st.selectbox("Start from customer", sorted_ids, key="whatif_customer")
        wi_row = df[df["customer_id"] == wi_id].iloc[0]
        wi_idx = df.index[df["customer_id"] == wi_id][0]
        X_base = df[FEATURE_NAMES].iloc[wi_idx].values.reshape(1, -1).copy().astype(float)
        feat_idx = {f: i for i, f in enumerate(FEATURE_NAMES)}

        st.markdown("#### Adjust key behavioural features")
        c1, c2 = st.columns(2)
        with c1:
            days_wi     = st.slider("Days since last bet",       0,    62, int(wi_row["days_since_last_bet"]),      help="Higher = customer has been inactive longer", key="wi_days")
            bets_wi     = st.slider("Bets (last 30d)",           0,    55, int(wi_row["bets_last_30d"]),            key="wi_bets")
            sessions_wi = st.slider("Sessions (last 30d)",       0,    45, int(wi_row["sessions_last_30d"]),        key="wi_sessions")
            deposits_wi = st.slider("Deposits (last 30d)",       0,    12, int(wi_row["deposit_count_last_30d"]),   key="wi_deposits")
        with c2:
            freq_wi    = st.slider("Bet frequency trend (last÷prior 30d)", 0.0, 3.0, float(wi_row["bet_frequency_trend"]),      step=0.05, help="< 1.0 = declining. > 1.0 = accelerating.", key="wi_freq")
            support_wi = st.slider("Support contacts (last 90d)",           0,   5,  int(wi_row["support_contacts_last_90d"]),  key="wi_support")
            promo_wi   = st.slider("Promo redemption rate",                 0.0, 1.0, float(wi_row["promo_redemption_rate"]),   step=0.05, key="wi_promo")
            live_wi    = st.slider("Live bet %",                            0.0, 1.0, float(wi_row["live_bet_pct"]),            step=0.05, key="wi_live")

        X_mod = X_base.copy()
        X_mod[0, feat_idx["days_since_last_bet"]]       = days_wi
        X_mod[0, feat_idx["bets_last_30d"]]             = bets_wi
        X_mod[0, feat_idx["sessions_last_30d"]]         = sessions_wi
        X_mod[0, feat_idx["deposit_count_last_30d"]]    = deposits_wi
        X_mod[0, feat_idx["bet_frequency_trend"]]       = freq_wi
        X_mod[0, feat_idx["support_contacts_last_90d"]] = support_wi
        X_mod[0, feat_idx["promo_redemption_rate"]]     = promo_wi
        X_mod[0, feat_idx["live_bet_pct"]]              = live_wi

        orig_prob_wi = artifact["model"].predict_proba(X_base)[0, 1]
        new_prob_wi  = artifact["model"].predict_proba(X_mod)[0, 1]
        delta_wi = new_prob_wi - orig_prob_wi

        st.markdown("#### Score impact")
        r1, r2, r3 = st.columns(3)
        r1.metric("Original Churn Risk", f"{orig_prob_wi:.1%}")
        r2.metric("Modified Churn Risk", f"{new_prob_wi:.1%}", f"{delta_wi:+.1%}", delta_color="inverse")
        r3.metric("Risk change", f"{abs(delta_wi):.1%} {'reduction' if delta_wi < 0 else 'increase'}")

        if delta_wi < -0.10:
            st.success("These changes would meaningfully reduce churn risk — a valid retention intervention.")
        elif delta_wi > 0.10:
            st.error("These changes increase churn risk — this trajectory needs CRM attention.")
        else:
            st.info("Modest change in risk score. Try adjusting recency or frequency trend for larger impact.")

    with tab4:
        metrics = artifact["metrics"]
        st.markdown("#### Model metrics (held-out test set)")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("ROC-AUC",   metrics["auc"],       help="Area under the ROC curve — 1.0 is perfect. Above 0.80 is production-viable.")
        p2.metric("Precision", metrics["precision"],  help="Of customers flagged as churn, what fraction actually churned.")
        p3.metric("Recall",    metrics["recall"],     help="Of customers who churned, what fraction the model caught.")
        p4.metric("F1 Score",  metrics["f1"],         help="Harmonic mean of precision and recall.")

        st.markdown("<br>", unsafe_allow_html=True)

        ti1, ti2 = st.columns([1, 1])
        with ti1:
            st.markdown("**Dataset 1 — Training & evaluation**")
            st.markdown(
                f"""
                | | |
                |---|---|
                | Training customers | {metrics['n_train']:,} |
                | Held-out test customers | {metrics['n_test']:,} |
                | Base churn rate (training set) | {metrics['churn_rate']:.1%} |
                | Algorithm | XGBoost (300 trees, depth 5) |
                | Explainability | SHAP TreeExplainer + LIME Tabular |

                **Dataset 2 — Business scoring (unseen)**

                | | |
                |---|---|
                | Scoring customers | {metrics['n_score']:,} |
                | Never seen during training | ✓ |
                | Used for | Cohort view · Deep-dive · What-if |
                """
            )
        with ti2:
            st.markdown("**Features used**")
            st.markdown("\n".join(f"- {FEATURE_DISPLAY[f]}" for f in FEATURE_NAMES))

        st.markdown("<br>**Global feature importance (mean |SHAP| across all customers)**")
        st.caption("Shows which signals the model relies on most — regardless of direction.")
        try:
            plt.close("all")
            shap.plots.bar(artifact["shap_values"], max_display=15, show=False)
            fig3 = plt.gcf()
            fig3.set_size_inches(8, 5)
            st.pyplot(fig3, use_container_width=True)
            plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP global plot unavailable: {exc}")

        with st.expander("How this model works"):
            st.markdown(
                """
                **Two-dataset pipeline**

                ```
                ┌─────────────────────────────────────────────────────────────┐
                │  DATASET 1 — Training & Evaluation  (seed=42, 600 customers)│
                └─────────────────────────────────────────────────────────────┘
                        │
                        ▼
                Feature engineering
                  · bet_frequency_trend = bets_last_30d ÷ bets_prev_30d
                  · stake_trend         = avg_stake_last_30d ÷ avg_stake_prev_30d
                        │
                        ▼
                80/20 stratified split
                  · 480 customers → model training
                  · 120 customers → held-out evaluation (AUC, precision, recall, F1)
                        │
                        ▼
                XGBoost classifier  (300 estimators, max depth 5, LR 0.08)
                SHAP TreeExplainer  ← fitted on training distribution
                LIME Tabular        ← fitted on training distribution

                ┌─────────────────────────────────────────────────────────────┐
                │  DATASET 2 — Business Scoring  (seed=99, 400 customers)     │
                │  Completely unseen during training                          │
                └─────────────────────────────────────────────────────────────┘
                        │
                        ▼
                model.predict_proba()  →  churn score per customer
                Percentile tiers       →  High / Medium / Low risk segments
                SHAP values            →  per-customer explanations
                        │
                        ▼
                Streamlit UI
                  · Cohort risk table     · Customer deep-dive
                  · What-if simulator     · Model performance dashboard
                ```
                """
            )
