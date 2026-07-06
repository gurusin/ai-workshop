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


def render_fraud_detection() -> None:
    render_header("Bet Anomaly & Fraud Detection")
    st.markdown(
        "<h2 style='margin:0 0 12px;'>🔍 Bet Anomaly &amp; Fraud Detection</h2>",
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
            <span style="display:inline-flex;align-items:center;gap:6px;background:#faf5ff;border:1px solid #c4b5fd;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#5b21b6;">
                &#129302;&nbsp; Unsupervised &mdash; no fraud labels required
            </span>
            <span style="display:inline-flex;align-items:center;gap:6px;background:#fff7ed;border:1px solid #fdba74;border-radius:20px;padding:5px 14px;font-size:0.78rem;font-weight:700;color:#c2410c;">
                &#128269;&nbsp; Isolation Forest &middot; SHAP &middot; analyst review workflow
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0c0a00 0%,#1c1a00 50%,#1a0a00 100%);
                    border-radius:14px;padding:30px 34px;margin-bottom:22px;color:white;">
            <div style="font-size:0.68rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                        color:#fde68a;margin-bottom:8px;">Business Problem</div>
            <div style="font-size:1.25rem;font-weight:800;line-height:1.35;margin-bottom:14px;">
                Flag suspicious betting patterns before financial exposure grows
            </div>
            <div style="font-size:0.88rem;color:#fef3c7;line-height:1.75;max-width:860px;">
                Match-fixing, arbitrage exploitation, bonus abuse, and syndicate bots all share a common
                problem: <strong>confirmed fraud labels arrive weeks after the activity</strong>. By then
                the financial exposure has already occurred. Isolation Forest learns what normal betting
                looks like across <strong>15 behavioural signals</strong> and flags accounts that deviate
                significantly &mdash; no labels required. A rule-based fraud type classifier then gives
                investigators a starting hypothesis, directing investigation effort to the right signal.
            </div>
            <div style="display:flex;gap:20px;margin-top:22px;flex-wrap:wrap;">
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fde68a;">unsupervised</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">no fraud labels needed to start</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#86efac;">5 types</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">match-fixing · arb · bonus abuse · bot · unusual</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#c4b5fd;">per-account</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">SHAP driver for every anomaly score</div>
                </div>
                <div style="flex:1;min-width:150px;background:rgba(255,255,255,0.09);border-radius:10px;padding:14px 18px;">
                    <div style="font-size:1.5rem;font-weight:800;color:#fdba74;">analyst-first</div>
                    <div style="font-size:0.78rem;color:#fef3c7;margin-top:5px;line-height:1.5;">score + hypothesis &rarr; human investigation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fd1, fd2, fd3 = st.columns(3)
    with fd1:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Training Data (CSV)</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:10px;">1,000 accounts &middot; 15 signals &middot; private file</div>
                <div style="font-size:0.82rem;color:#6b7280;line-height:1.68;margin-bottom:16px;">
                    Synthetic accounts with realistic normal behaviour plus injected anomalous
                    patterns (match-fixing, bots, bonus abuse, arbitrage). Isolation Forest
                    trains on all accounts and learns to separate the outliers.
                </div>
                <div style="font-size:0.8rem;color:#374151;line-height:1.9;border-top:1px solid #f3f4f6;padding-top:12px;">
                    <div style="margin-bottom:3px;"><span style="color:#b91c1c;font-weight:700;">&#9679;</span> <strong>Win patterns</strong> &mdash; win rate, win streak</div>
                    <div style="margin-bottom:3px;"><span style="color:#1e40af;font-weight:700;">&#9679;</span> <strong>Staking</strong> &mdash; avg stake, coefficient of variation</div>
                    <div style="margin-bottom:3px;"><span style="color:#5b21b6;font-weight:700;">&#9679;</span> <strong>Market selection</strong> &mdash; odds, concentration, sports count</div>
                    <div style="margin-bottom:3px;"><span style="color:#d97706;font-weight:700;">&#9679;</span> <strong>Timing</strong> &mdash; mins before start, late line moves</div>
                    <div><span style="color:#059669;font-weight:700;">&#9679;</span> <strong>Account signals</strong> &mdash; age, withdrawal ratio, bet volume</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fd2:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">ML Architecture</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">Isolation Forest &middot; SHAP &middot; rule-based type</div>
                <div style="font-size:0.82rem;color:#374151;">
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#0c0a00;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">1</div>
                        <div><strong>Load training CSV</strong><br><span style="color:#6b7280;font-size:0.78rem;">1,000 accounts &middot; mixed normal + anomalous</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#1c1a00;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">2</div>
                        <div><strong>Isolation Forest</strong><br><span style="color:#6b7280;font-size:0.78rem;">200 trees &middot; contamination=0.12 &middot; no labels</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#7c3aed;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">3</div>
                        <div><strong>Score normalisation</strong><br><span style="color:#6b7280;font-size:0.78rem;">score_samples() &rarr; [0,1] &middot; p70/p85 tier thresholds</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;margin-bottom:13px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#b91c1c;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">4</div>
                        <div><strong>SHAP driver</strong><br><span style="color:#6b7280;font-size:0.78rem;">top anomaly-driving feature per account</span></div>
                    </div>
                    <div style="margin-left:13px;border-left:1px dashed #d1d5db;height:10px;"></div>
                    <div style="display:flex;align-items:flex-start;gap:11px;">
                        <div style="width:26px;height:26px;border-radius:50%;background:#d97706;color:white;display:flex;align-items:center;justify-content:center;font-size:0.7rem;font-weight:700;flex-shrink:0;">5</div>
                        <div><strong>Fraud type classifier</strong><br><span style="color:#6b7280;font-size:0.78rem;">rule-based hypothesis &rarr; investigation focus</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fd3:
        st.markdown(
            """
            <div style="border:1px solid #e5e7eb;border-radius:13px;padding:22px 20px;background:white;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
                <div style="font-size:0.67rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#9ca3af;margin-bottom:10px;">Fraud Type Classification</div>
                <div style="font-size:0.97rem;font-weight:700;color:#111827;margin-bottom:16px;">5 hypotheses &mdash; analyst starting point</div>
                <div style="font-size:0.82rem;color:#374151;line-height:1.6;">
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fecaca;color:#7f1d1d;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#127919; Match-Fixing</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">High win rate + bets very close to event start</div>
                    </div>
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#dbeafe;color:#1e3a5f;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#129302; Bot / Syndicate</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Very consistent stake sizes + high volume</div>
                    </div>
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#fef3c7;color:#92400e;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#127873; Bonus Abuse</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">New account + high withdrawal-to-deposit ratio</div>
                    </div>
                    <div style="margin-bottom:9px;padding-bottom:9px;border-bottom:1px solid #f3f4f6;">
                        <span style="background:#ede9fe;color:#5b21b6;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#9878; Arbitrage</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">Systematic stakes + high win rate + &le;2 sports</div>
                    </div>
                    <div>
                        <span style="background:#f3f4f6;color:#374151;border-radius:5px;padding:2px 8px;font-size:0.72rem;font-weight:700;">&#10067; Unusual Pattern</span>
                        <div style="margin-top:5px;color:#4b5563;font-size:0.8rem;">High anomaly score without clear rule match</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:24px;'></div>", unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Fitting Isolation Forest on private betting dataset…")
    def _load_fraud():
        fd_train = load_pkg_module("fraud_detection_model", os.path.join(ROOT, "fraud-detection", "model"), "train")
        return fd_train.build()

    artifact = _load_fraud()
    metrics = artifact["metrics"]
    fd_data = load_pkg_module("fraud_detection_model", os.path.join(ROOT, "fraud-detection", "model"), "data")
    FD_FEATURES = fd_data.FEATURE_NAMES
    FD_DISPLAY  = fd_data.FEATURE_DISPLAY
    FRAUD_TYPES = fd_data.FRAUD_TYPES

    if "fraud_seed" not in st.session_state:
        st.session_state["fraud_seed"] = int(np.random.randint(1000, 99999))

    tab1, tab2 = st.tabs(["🔍 Anomaly Dashboard", "📈 Model Insights"])

    with tab1:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption(
                f"Scoring batch seed: `{st.session_state['fraud_seed']}` &nbsp;·&nbsp; "
                f"40 accounts scored &nbsp;·&nbsp; Isolation Forest trained on private CSV ({metrics['n_train']:,} accounts)"
            )
        with c2:
            if st.button("🔄 Generate New Batch", key="fraud_new_batch", use_container_width=True):
                st.session_state["fraud_seed"] = int(np.random.randint(1000, 99999))
                st.rerun()

        fd_train = load_pkg_module("fraud_detection_model", os.path.join(ROOT, "fraud-detection", "model"), "train")
        batch = fd_train.score_batch(artifact, n=40, seed=st.session_state["fraud_seed"])
        df = batch["df"]

        total  = len(df)
        high   = int((df["risk_tier"] == "High Risk").sum())
        medium = int((df["risk_tier"] == "Medium Risk").sum())
        low    = int((df["risk_tier"] == "Low Risk").sum())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accounts Scored", total)
        m2.metric("🔴 High Risk",   high,   delta=f"{high/total:.0%}",   delta_color="inverse")
        m3.metric("🟡 Medium Risk", medium, delta=f"{medium/total:.0%}", delta_color="off")
        m4.metric("🟢 Low Risk",    low,    delta=f"{low/total:.0%}",    delta_color="normal")

        st.markdown("<br>", unsafe_allow_html=True)

        fraud_counts = df[df["fraud_type"] != "Normal"]["fraud_type"].value_counts().to_dict()
        if fraud_counts:
            items = []
            for name, info in FRAUD_TYPES.items():
                count = fraud_counts.get(name, 0)
                if count > 0:
                    items.append(
                        f'<div style="display:inline-flex;align-items:center;gap:8px;background:{info["bg"]};'
                        f'border:1px solid {info["color"]}33;border-radius:10px;padding:8px 14px;margin:4px;">'
                        f'<span style="font-size:1.1rem;">{info["icon"]}</span>'
                        f'<div><div style="font-size:0.78rem;font-weight:700;color:{info["color"]};">{name}</div>'
                        f'<div style="font-size:0.72rem;color:#6b7280;">{count} account{"s" if count != 1 else ""}</div>'
                        f'</div></div>'
                    )
            st.markdown(
                '<div style="font-size:0.72rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;'
                'color:#9ca3af;margin-bottom:8px;">Flagged Fraud Types This Batch</div>'
                '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:20px;">' + "".join(items) + "</div>",
                unsafe_allow_html=True,
            )

        risk_icon = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}
        display = df[["customer_id", "risk_tier", "anomaly_pct", "fraud_type", "top_driver",
                       "win_rate_30d", "stake_cv", "account_age_days"]].copy()
        display["risk_tier"]  = display["risk_tier"].apply(lambda r: f"{risk_icon.get(r, '')} {r}")
        display["fraud_type"] = display["fraud_type"].apply(
            lambda f: f"{FRAUD_TYPES[f]['icon']} {f}" if f in FRAUD_TYPES else f
        )
        display["top_driver"] = display["top_driver"].map(FD_DISPLAY)
        display.columns = ["Account", "Risk Tier", "Anomaly %", "Fraud Type", "Top Driver",
                           "Win Rate", "Stake CV", "Account Age (days)"]

        st.dataframe(
            display,
            column_config={
                "Anomaly %": st.column_config.ProgressColumn("Anomaly %", min_value=0, max_value=100, format="%.1f%%"),
                "Win Rate":  st.column_config.NumberColumn("Win Rate",  format="%.2f"),
                "Stake CV":  st.column_config.NumberColumn("Stake CV",  format="%.2f"),
            },
            hide_index=True,
            use_container_width=True,
            height=580,
        )
        st.caption(
            "**Anomaly %** — Isolation Forest score (higher = more anomalous). "
            "**Fraud Type** — rule-based hypothesis for investigation focus. "
            "**Top Driver** — feature most responsible for the anomaly score."
        )

    with tab2:
        st.markdown(
            f"""
**Model details**

| | |
|---|---|
| Training file | `fraud-detection/data/fraud_training.csv` |
| Training accounts | {metrics['n_train']:,} |
| Contamination parameter | {metrics['contamination']:.0%} |
| Algorithm | Isolation Forest (200 trees) |
| Explainability | SHAP TreeExplainer (z-score fallback) |
| Features | {metrics['n_features']} behavioural signals |

**Anomaly score interpretation**

| Score range | Risk tier | Suggested action |
|---|---|---|
| 0–p70 | Low Risk | No action |
| p70–p85 | Medium Risk | Watch list — monitor 7 days |
| p85–100 | High Risk | Analyst review within 48h |
"""
        )
        if batch["shap_values"] is not None:
            st.markdown("<br>**Global feature importance — mean |SHAP| across current batch**")
            try:
                plt.close("all")
                shap.plots.bar(batch["shap_values"], max_display=15, show=False)
                fig = plt.gcf()
                fig.set_size_inches(8, 5)
                st.pyplot(fig, use_container_width=True)
                plt.close("all")
            except Exception as exc:
                st.warning(f"SHAP plot unavailable: {exc}")
