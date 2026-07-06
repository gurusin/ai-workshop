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


def render_rg_warning() -> None:
    render_header("Responsible Gambling Early Warning System")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>⚠️ Responsible Gambling Early Warning System</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:18px;">
            <span style="display:inline-flex;align-items:center;gap:6px;background:#f0fdf4;border:1px solid #86efac;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#166534;">
                &#9711;&nbsp; Proprietary model &mdash; no external API
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#eff6ff;border:1px solid #93c5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#1e40af;">
                &#128274;&nbsp; Trained on private synthetic data (CSV)
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#fef2f2;border:1px solid #fca5a5;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#991b1b;">
                &#9878;&nbsp; Compliance &mdash; ACMA harm minimisation obligations
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#9881;&nbsp; XGBoost &middot; SHAP &mdash; fully local
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#1c0a00 0%,#431407 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#fca5a5;margin-bottom:8px;">Regulatory Obligation</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                Identify harm before it becomes self-exclusion
            </div>
            <div style="font-size:0.88rem;color:#fecaca;line-height:1.75;max-width:860px;">
                Standard harm minimisation detects problem gambling after it has escalated — a self-exclusion
                request, a support call, a complaint. By then, significant financial and psychological harm
                has already occurred. This system reads <strong>15 behavioural signals</strong> continuously
                and flags at-risk customers <strong>3&ndash;6 weeks before self-exclusion</strong>, when a
                proportionate, graduated intervention can still change the trajectory. Every flag carries a
                SHAP explanation so compliance teams can document the rationale and demonstrate due diligence.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fca5a5;">3&ndash;6 wks</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">lead time before self-exclusion</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fdba74;">4 tiers</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">Low · Medium · High · Critical</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#86efac;">absolute</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">probability thresholds &mdash; not relative percentiles</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#c4b5fd;">auditable</div>
                    <div style="font-size:0.78rem;color:#fecaca;margin-top:5px;line-height:1.5;">SHAP explanation per flag for compliance review</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    rg1, rg2, rg3 = st.columns(3)
    with rg1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Training Data (CSV)</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">1,000 players &middot; 15 signals &middot; private file</div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    A static CSV of 1,000 synthetic players is generated once and persisted.
                    The model trains from this fixed file. A separate random batch is scored
                    fresh each session &mdash; simulating live player monitoring.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;"><span style="color:#b91c1c;font-weight:700;">&#9679;</span> <strong>Chasing</strong> &mdash; loss chasing score, stake variance</div>
                    <div style="margin-bottom:3px;"><span style="color:#d97706;font-weight:700;">&#9679;</span> <strong>Time patterns</strong> &mdash; late-night %, session length</div>
                    <div style="margin-bottom:3px;"><span style="color:#7c3aed;font-weight:700;">&#9679;</span> <strong>Financial</strong> &mdash; deposit freq, net loss, escalation</div>
                    <div style="margin-bottom:3px;"><span style="color:#0369a1;font-weight:700;">&#9679;</span> <strong>Control avoidance</strong> &mdash; withdrawal reversals, reality check</div>
                    <div><span style="color:#059669;font-weight:700;">&#9679;</span> <strong>Game choice</strong> &mdash; high-volatility game %</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rg2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">ML Architecture</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">XGBoost &middot; SHAP &middot; absolute thresholds</div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1c0a00;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div><strong>Load training CSV</strong><br><span style="color:#6b7280;font-size:0.78rem;">1,000 players &middot; fixed seed &middot; persisted file</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#431407;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div><strong>XGBoost binary classifier</strong><br><span style="color:#6b7280;font-size:0.78rem;">300 trees &middot; depth 5 &middot; 80/20 stratified split</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7f1d1d;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">3</div>
                        <div><strong>Absolute threshold tiering</strong><br><span style="color:#6b7280;font-size:0.78rem;">prob &lt;0.3 Low &middot; &lt;0.6 Medium &middot; &lt;0.8 High &middot; &ge;0.8 Critical</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">4</div>
                        <div><strong>SHAP TreeExplainer</strong><br><span style="color:#6b7280;font-size:0.78rem;">top driver per player &middot; global importance</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#b91c1c;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">5</div>
                        <div><strong>Intervention assignment</strong><br><span style="color:#6b7280;font-size:0.78rem;">tier &rarr; graduated action for compliance team</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with rg3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Intervention Framework</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">Graduated response proportionate to risk</div>
                <div style="font-size:0.82rem;line-height:1.65;">
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#dcfce7;color:#166534;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">Low &lt;0.3</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Continue monitoring &mdash; no action required</div>
                    </div>
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">Medium 0.3&ndash;0.6</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">RG information email + self-assessment tool offer</div>
                    </div>
                    <div style="margin-bottom:10px;padding-bottom:10px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fee2e2;color:#b91c1c;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">High 0.6&ndash;0.8</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Mandatory outreach + stake limits + cooling-off</div>
                    </div>
                    <div>
                        <span style="background:#fecaca;color:#7f1d1d;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">Critical &ge;0.8</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Restrict account + mandatory support referral</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Training RG early warning model on private dataset…")
    def _load_rg():
        rg_train = load_pkg_module("rg_warning_model", os.path.join(ROOT, "rg-warning", "model"), "train")
        return rg_train.build()

    artifact = _load_rg()
    metrics = artifact["metrics"]
    rg_data = load_pkg_module("rg_warning_model", os.path.join(ROOT, "rg-warning", "model"), "data")
    RG_FEATURES   = rg_data.FEATURE_NAMES
    RG_DISPLAY    = rg_data.FEATURE_DISPLAY
    INTERVENTIONS = rg_data.INTERVENTIONS

    if "rg_seed" not in st.session_state:
        st.session_state["rg_seed"] = int(np.random.randint(1000, 99999))

    tab1, tab2 = st.tabs(["🛡️ Player Risk Dashboard", "📈 Model Performance"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(
                f"Scoring batch seed: `{st.session_state['rg_seed']}` &nbsp;·&nbsp; "
                f"40 players scored &nbsp;·&nbsp; model trained on private CSV ({metrics['n_train']:,} players)"
            )
        with c2:
            if st.button("🔄 Generate New Batch", key="rg_new_batch", use_container_width=True):
                st.session_state["rg_seed"] = int(np.random.randint(1000, 99999))
                st.rerun()

        rg_train = load_pkg_module("rg_warning_model", os.path.join(ROOT, "rg-warning", "model"), "train")
        batch = rg_train.score_batch(artifact, n=40, seed=st.session_state["rg_seed"])
        df = batch["df"]

        total    = len(df)
        critical = int((df["risk_tier"] == "Critical").sum())
        high     = int((df["risk_tier"] == "High").sum())
        medium   = int((df["risk_tier"] == "Medium").sum())
        low      = int((df["risk_tier"] == "Low").sum())

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Players Scored", total)
        m2.metric("🛑 Critical", critical, delta=f"{critical/total:.0%}", delta_color="inverse")
        m3.metric("🚨 High",     high,     delta=f"{high/total:.0%}",     delta_color="inverse")
        m4.metric("⚠️ Medium",   medium,   delta=f"{medium/total:.0%}",   delta_color="off")
        m5.metric("✅ Low",      low,      delta=f"{low/total:.0%}",      delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        tier_icon = {"Critical": "🛑", "High": "🚨", "Medium": "⚠️", "Low": "✅"}
        display = df[["customer_id", "risk_tier", "risk_prob", "top_driver", "intervention",
                       "loss_chasing_score", "late_night_session_pct", "withdrawal_reversal_count"]].copy()
        display["risk_tier"]  = display["risk_tier"].apply(lambda r: f"{tier_icon.get(r, '')} {r}")
        display["top_driver"] = display["top_driver"].map(RG_DISPLAY)
        display["risk_prob"]  = (display["risk_prob"] * 100).round(1)
        display.columns = ["Player", "Risk Tier", "Risk %", "Top Signal", "Intervention",
                           "Loss Chasing", "Late-Night %", "Withdrawal Reversals"]

        st.dataframe(
            display,
            column_config={
                "Risk %": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.1f%%"),
                "Loss Chasing": st.column_config.NumberColumn("Loss Chasing", format="%.2f"),
                "Late-Night %": st.column_config.NumberColumn("Late-Night %", format="%.0%%"),
            },
            hide_index=True,
            use_container_width=True,
            height=580,
        )
        st.caption(
            "**Risk %** — model probability of problem gambling. "
            "**Top Signal** — highest absolute SHAP feature. "
            "**Intervention** — graduated action mapped to risk tier."
        )

    with tab2:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("AUC",       f"{metrics['auc']:.3f}")
        p2.metric("Precision", f"{metrics['precision']:.3f}")
        p3.metric("Recall",    f"{metrics['recall']:.3f}")
        p4.metric("F1",        f"{metrics['f1']:.3f}")
        st.markdown("<br>", unsafe_allow_html=True)
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Training dataset**")
            st.markdown(
                f"""
| | |
|---|---|
| Training file | `rg-warning/data/training_data.csv` |
| Training players | {metrics['n_train']:,} |
| Test players | {metrics['n_test']:,} |
| At-risk rate | {metrics['at_risk_rate']:.1%} |
| Algorithm | XGBoost (300 trees, depth 5) |
| Explainability | SHAP TreeExplainer |
"""
            )
        with d2:
            st.markdown("**Features used**")
            st.markdown("\n".join(f"- {RG_DISPLAY[f]}" for f in RG_FEATURES))

        st.markdown("<br>**Global feature importance (mean |SHAP| — current batch)**")
        try:
            plt.close("all")
            shap.plots.bar(batch["shap_values"], max_display=15, show=False)
            fig = plt.gcf()
            fig.set_size_inches(8, 5)
            st.pyplot(fig, use_container_width=True)
            plt.close("all")
        except Exception as exc:
            st.warning(f"SHAP plot unavailable: {exc}")
