import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
import streamlit as st

from config import ROOT
from utils.module_loader import load_pkg_module
from utils.ui import render_header


def render_telco_churn() -> None:
    render_header("MVNO Telco Churn & Martech Campaigns")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>📱 MVNO Telco Churn Prediction &amp; Martech Campaigns</h2>",
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
                &#128274;&nbsp; Trained on private synthetic MVNO data (CSV)
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#9881;&nbsp; XGBoost &middot; SHAP &mdash; fully local
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;
                         background:#fff7ed;border:1px solid #fdba74;border-radius:20px;
                         padding:5px 14px;font-size:0.78rem;font-weight:700;color:#c2410c;">
                &#127919;&nbsp; Campaign-ready output for CRM &amp; martech
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#7fb3d3;margin-bottom:8px;">Business Problem</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                MVNO subscriber loss &mdash; identified early enough to act
            </div>
            <div style="font-size:0.88rem;color:#c8ddef;line-height:1.75;max-width:860px;">
                MVNOs operate on thin margins with high acquisition costs. A subscriber who reduces
                data consumption, stops calling, and then ports out represents a full CAC write-off.
                Standard reporting flags churn after it happens. This model flags it
                <strong>3&ndash;6 weeks before port-out</strong> so the CRM team can engage with the
                right offer at the right moment. Every prediction maps directly to a martech campaign
                action &mdash; no manual triage required.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#38bdf8;">3&ndash;6 wks</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        lead time before port-out
                    </div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#34d399;">15 signals</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        usage, friction, spend &amp; contract
                    </div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f59e0b;">8 campaigns</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        auto-assigned per risk profile
                    </div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);
                            border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#f472b6;">per-customer</div>
                    <div style="font-size:0.78rem;color:#a8c4e0;margin-top:5px;line-height:1.5;">
                        SHAP explanation for every score
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _tc1, _tc2, _tc3 = st.columns(3)

    with _tc1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Training Data (CSV file)
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">
                    800 subscribers &middot; 15 features &middot; saved to file
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:10px;">
                    A static CSV of 800 synthetic MVNO subscribers is generated once and persisted to
                    <code>telco-churn/data/training_data.csv</code>. The model always trains from this
                    fixed file &mdash; ensuring reproducible weights across runs.
                </div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    At runtime, a separate random batch is generated fresh on each invocation and scored
                    by the locked model &mdash; simulating a live customer population the model has never seen.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;
                            border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;">
                        <span style="color:#0369a1;font-weight:700;">&#9679;</span>
                        <strong>Recency</strong> &mdash; days since call &amp; data
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#7c3aed;font-weight:700;">&#9679;</span>
                        <strong>Usage</strong> &mdash; data GB &amp; call volumes, trends
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#059669;font-weight:700;">&#9679;</span>
                        <strong>Revenue</strong> &mdash; ARPU, plan changes
                    </div>
                    <div style="margin-bottom:3px;">
                        <span style="color:#d97706;font-weight:700;">&#9679;</span>
                        <strong>Contract</strong> &mdash; tenure, months remaining
                    </div>
                    <div>
                        <span style="color:#dc2626;font-weight:700;">&#9679;</span>
                        <strong>Friction</strong> &mdash; support tickets, payment delays
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _tc2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    ML Architecture
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    XGBoost &middot; SHAP &middot; per-invocation scoring
                </div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0f172a;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div><strong>Load training CSV</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                800 subscribers from file &middot; fixed seed
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1e3a5f;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div><strong>XGBoost classifier</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                300 trees &middot; depth 5 &middot; 80/20 stratified split
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">3</div>
                        <div><strong>SHAP TreeExplainer</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                per-customer top driver &middot; global importance
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0369a1;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">4</div>
                        <div><strong>Random batch scoring</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                fresh 30 subscribers per invocation &middot; new seed each run
                            </span>
                        </div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#c2410c;
                                    color:white;display:flex;align-items:center;justify-content:center;
                                    font-size:0.7rem;font-weight:700;flex-shrink:0;">5</div>
                        <div><strong>Campaign assignment</strong><br>
                            <span style="color:#6b7280;font-size:0.78rem;">
                                risk tier + feature profile &rarr; campaign action
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with _tc3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;
                        background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;
                            text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">
                    Martech Campaign Logic
                </div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">
                    8 campaigns mapped by risk &amp; feature profile
                </div>
                <div style="font-size:0.82rem;color:#374151;line-height:1.6;">
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fee2e2;color:#b91c1c;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">High risk</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">
                            VIP Retention &middot; Win-Back Offer<br>Early Renewal &middot; Service Recovery
                        </div>
                    </div>
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">Medium risk</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">
                            Renewal Reminder &middot; Data Upsell<br>Engagement Boost
                        </div>
                    </div>
                    <div>
                        <span style="background:#d1fae5;color:#065f46;border-radius:5px;
                                     padding:2px 8px;font-size:0.72rem;font-weight:700;">Low risk</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">
                            Loyalty Reward
                        </div>
                    </div>
                    <div style="margin-top:12px;padding-top:12px;border-top:1px solid #f3f4f6;
                                font-size:0.78rem;color:#6b7280;">
                        High-risk routing prioritises contract status, then ARPU, then support friction.
                        Medium-risk routing checks contract window and data trend.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="border:1px solid #e5e7eb;border-radius:13px;padding:26px 28px 22px;
                    background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);margin-bottom:28px;">
            <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
                        color:#9ca3af;margin-bottom:6px;">Technical Implementation</div>
            <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:18px;">
                Static training data &rarr; locked model &rarr; fresh random batch per invocation
            </div>
            <div style="font-family:monospace;font-size:0.82rem;color:#374151;
                        background:#f8fafc;border:1px solid #e5e7eb;border-radius:8px;
                        padding:18px 22px;line-height:1.9;overflow-x:auto;white-space:pre;">
DATASET 1 — Training &amp; Evaluation  (seed=42 · 800 subscribers · saved to CSV)
  · telco-churn/data/training_data.csv  — generated once, persisted, always reloaded
  · 640 subscribers → XGBoost training  (300 trees · depth 5 · LR 0.08)
  · 160 subscribers → held-out test  → AUC · precision · recall · F1
  · SHAP TreeExplainer fitted on training distribution
          |
          v  model weights locked
          |
DATASET 2 — Live Scoring Batch  (random seed · 30 subscribers · per invocation)
  · Generated fresh on every "Generate New Batch" click  — never cached
  · model.predict_proba()  → churn probability per subscriber
  · Percentile tiers       → High (top 30%) · Medium · Low (bottom 40%)
  · SHAP values            → top driving feature per subscriber
  · Campaign assignment    → risk tier + feature profile → martech action
          |
          v
Streamlit UI
  · Campaign Dashboard  — full subscriber table with risk + campaign per row
  · Model Performance   — AUC, precision, recall, F1, SHAP global importance
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="Training MVNO churn model on private dataset…")
    def _load_telco():
        tc_train = load_pkg_module("telco_churn_model", os.path.join(ROOT, "telco-churn", "model"), "train")
        return tc_train.build()

    artifact = _load_telco()
    metrics = artifact["metrics"]

    tc_data = load_pkg_module("telco_churn_model", os.path.join(ROOT, "telco-churn", "model"), "data")
    TC_FEATURES = tc_data.FEATURE_NAMES
    TC_DISPLAY = tc_data.FEATURE_DISPLAY
    CAMPAIGNS = tc_data.CAMPAIGNS

    if "telco_seed" not in st.session_state:
        st.session_state["telco_seed"] = int(np.random.randint(1000, 99999))

    tab1, tab2 = st.tabs(["📊 Campaign Dashboard", "📈 Model Performance"])

    with tab1:
        ctrl1, ctrl2 = st.columns([3, 1])
        with ctrl1:
            st.caption(
                f"Scoring batch seed: `{st.session_state['telco_seed']}` &nbsp;·&nbsp; "
                f"30 subscribers generated fresh &nbsp;·&nbsp; "
                f"model trained on private CSV ({metrics['n_train']:,} subscribers)"
            )
        with ctrl2:
            if st.button("🔄 Generate New Batch", use_container_width=True):
                st.session_state["telco_seed"] = int(np.random.randint(1000, 99999))
                st.rerun()

        tc_train = load_pkg_module("telco_churn_model", os.path.join(ROOT, "telco-churn", "model"), "train")
        batch = tc_train.score_batch(artifact, n=30, seed=st.session_state["telco_seed"])
        df = batch["df"]

        total = len(df)
        high  = int((df["risk_segment"] == "High").sum())
        med   = int((df["risk_segment"] == "Medium").sum())
        low   = int((df["risk_segment"] == "Low").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Subscribers Scored", f"{total}")
        m2.metric("High Risk", f"{high}", delta=f"{high/total:.0%} of batch", delta_color="inverse")
        m3.metric("Medium Risk", f"{med}", delta=f"{med/total:.0%} of batch", delta_color="off")
        m4.metric("Low Risk", f"{low}", delta=f"{low/total:.0%} of batch", delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        campaign_counts = df["campaign"].value_counts().to_dict()
        camp_items = []
        for name, info in CAMPAIGNS.items():
            count = campaign_counts.get(name, 0)
            if count > 0:
                camp_items.append(
                    f'<div style="display:inline-flex;align-items:center;gap:8px;'
                    f'background:{info["bg"]};border:1px solid {info["color"]}33;'
                    f'border-radius:10px;padding:8px 14px;margin:4px;">'
                    f'<span style="font-size:1.1rem;">{info["icon"]}</span>'
                    f'<div><div style="font-size:0.78rem;font-weight:700;color:{info["color"]};">'
                    f'{name}</div>'
                    f'<div style="font-size:0.72rem;color:#6b7280;">{count} subscriber{"s" if count != 1 else ""}</div>'
                    f'</div></div>'
                )
        st.markdown(
            '<div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;'
            'color:#9ca3af;margin-bottom:8px;">Campaign Distribution This Batch</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:20px;">'
            + "".join(camp_items) + "</div>",
            unsafe_allow_html=True,
        )

        risk_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
        display = df[["customer_id", "risk_segment", "churn_pct", "top_driver",
                       "campaign", "avg_monthly_spend", "contract_months_remaining",
                       "support_tickets_90d"]].copy()
        display["risk_segment"] = display["risk_segment"].apply(lambda r: f"{risk_icon.get(str(r), '')} {r}")
        display["top_driver"] = display["top_driver"].map(TC_DISPLAY)
        display["campaign"] = display["campaign"].apply(
            lambda c: f"{CAMPAIGNS[c]['icon']} {c}" if c in CAMPAIGNS else c
        )
        display.columns = [
            "Subscriber", "Risk", "Churn %", "Top Driver",
            "Campaign Action", "ARPU ($)", "Contract Remaining", "Support Tickets",
        ]

        st.dataframe(
            display,
            column_config={
                "Churn %": st.column_config.ProgressColumn(
                    "Churn %", min_value=0, max_value=100, format="%.1f%%"
                ),
                "ARPU ($)": st.column_config.NumberColumn("ARPU ($)", format="$%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=600,
        )
        st.caption(
            "**Top Driver** — the feature with the largest absolute SHAP value for that subscriber. "
            "**Campaign Action** — auto-assigned from risk tier and feature profile."
        )

    with tab2:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("AUC", metrics["auc"], help="Area under the ROC curve.")
        p2.metric("Precision", metrics["precision"], help="Of flagged churners, fraction that actually churned.")
        p3.metric("Recall", metrics["recall"], help="Of actual churners, fraction the model caught.")
        p4.metric("F1 Score", metrics["f1"], help="Harmonic mean of precision and recall.")

        st.markdown("<br>", unsafe_allow_html=True)

        pd1, pd2 = st.columns(2)
        with pd1:
            st.markdown("**Dataset 1 — Training & evaluation**")
            st.markdown(
                f"""
                | | |
                |---|---|
                | Training file | `telco-churn/data/training_data.csv` |
                | Training subscribers | {metrics['n_train']:,} |
                | Held-out test subscribers | {metrics['n_test']:,} |
                | Base churn rate | {metrics['churn_rate']:.1%} |
                | Algorithm | XGBoost (300 trees, depth 5) |
                | Explainability | SHAP TreeExplainer |

                **Dataset 2 — Scoring batch (unseen)**

                | | |
                |---|---|
                | Scoring subscribers | 30 (random per invocation) |
                | Never seen during training | ✓ |
                | Used for | Campaign Dashboard |
                """
            )
        with pd2:
            st.markdown("**Features used**")
            st.markdown("\n".join(f"- {TC_DISPLAY[f]}" for f in TC_FEATURES))

        st.markdown("<br>**Global feature importance (mean |SHAP| across current batch)**")
        st.caption("Shows which signals the model relies on most for this scoring batch.")
        try:
            plt.close("all")
            shap.plots.bar(batch["shap_values"], max_display=15, show=False)
            fig_shap = plt.gcf()
            fig_shap.set_size_inches(8, 5)
            st.pyplot(fig_shap, use_container_width=True)
            plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP global plot unavailable: {exc}")
