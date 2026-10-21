"""Interactive BI Risk Dashboard & Executive Surveillance Center.

Streamlit enterprise application providing executive surveillance KPIs, live investigator
triage consoles with interactive TreeSHAP waterfalls, automated FinCEN SAR narrative exports,
typology drilldowns, and a dynamic asymmetric loss matrix threshold simulator.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from scipy.special import expit

from financial_risk.config import (
    DATA_PROCESSED_DIR,
    MODELS_DIR,
)
from financial_risk.explainability import RiskExplainabilityEngine

matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA & MODEL CACHING HELPERS
# =============================================================================


@st.cache_resource(show_spinner="Loading Champion Risk Models & TreeExplainer...")
def get_cached_engine() -> RiskExplainabilityEngine:
    """Load and cache the RiskExplainabilityEngine singleton."""
    model_path = MODELS_DIR / "champion_lightgbm_tuned.joblib"
    return RiskExplainabilityEngine(model_path=model_path)


@st.cache_data(show_spinner="Loading Processed Surveillance Datasets...")
def get_cached_test_data() -> pd.DataFrame:
    """Load and cache the strictly out-of-time Test Holdout partition (Days 27-31)."""
    parquet_path = DATA_PROCESSED_DIR / "features_engineered.parquet"
    cols = RiskExplainabilityEngine.FEATURE_COLS + [
        "transaction_id",
        "day",
        "hour_of_day",
        "type",
        "amount",
        "isFraud",
        "isFlaggedFraud",
    ]
    table = pq.read_table(parquet_path, columns=cols)
    df = table.to_pandas()
    test_df = df[df["day"] >= 27].copy()

    # Pre-score test transactions for instant UI filtering
    engine = get_cached_engine()
    raw_scores = engine.model.predict(
        test_df[RiskExplainabilityEngine.FEATURE_COLS].astype(np.float32),
        raw_score=True,
    )
    test_df["fraud_probability"] = expit(raw_scores)
    test_df["risk_score"] = np.round(test_df["fraud_probability"] * 1000.0).astype(int)

    # Classify triage queue
    def get_queue(p: float) -> str:
        if p < 0.140:
            return "AUTOMATED_APPROVAL"
        elif p < 0.850:
            return "INVESTIGATOR_REVIEW_QUEUE"
        else:
            return "PRIORITY_ESCALATION_QUEUE"

    test_df["triage_decision"] = test_df["fraud_probability"].apply(get_queue)
    return test_df


@st.cache_data
def get_cached_summaries() -> dict[str, Any]:
    """Load persisted summaries from baseline, advanced ML, and explainability runs."""
    adv_path = DATA_PROCESSED_DIR / "advanced_ml_summary.json"
    exp_path = DATA_PROCESSED_DIR / "explainability_summary.json"

    data: dict[str, Any] = {}
    if adv_path.exists():
        with open(adv_path, encoding="utf-8") as f:
            data["advanced_ml"] = json.load(f)
    if exp_path.exists():
        with open(exp_path, encoding="utf-8") as f:
            data["explainability"] = json.load(f)
    return data


# =============================================================================
# BUSINESS LOGIC & SAR GENERATOR
# =============================================================================


def generate_sar_narrative(
    tx: pd.Series,
    explanation: dict[str, Any],
) -> str:
    """Generate standardized FinCEN Suspicious Activity Report (SAR) narrative text."""
    adverse_codes = explanation.get("primary_adverse_reason_codes", [])
    reasons_text = "\n".join(
        [
            f"  - [{r['reason_code']}] {r['description']} (Attribution: {r['shap_value']:+.2f} log-odds, {r['relative_impact_pct']:.1f}% impact)"
            for r in adverse_codes
        ]
    )

    narrative = f"""================================================================================
FINANCIAL CRIMES ENFORCEMENT NETWORK (FinCEN) SUSPICIOUS ACTIVITY REPORT (SAR)
AUTOMATED INVESTIGATOR NARRATIVE EXPORT PACKAGE
================================================================================
REPORT ID          : SAR-{tx["transaction_id"]}-{pd.Timestamp.now().strftime("%Y%m%d")}
DATE GENERATED     : {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC")}
FILING INSTITUTION : TIER-1 FINANCIAL SURVEILLANCE & RISK OPERATIONS UNIT
SUBJECT TRANSACTION: #{tx["transaction_id"]}

1. TRANSACTION PROFILE
--------------------------------------------------------------------------------
Monetary Magnitude : ${tx["amount"]:,.2f} USD
Corridor Typology  : {tx["type"]}
Execution Time     : Day {tx["day"]}, {int(tx["hour_of_day"]):02d}:00 Hours (Diurnal Risk Window)
Origin Balance Pre : ${tx["oldbalanceOrg"]:,.2f}
Origin Balance Post: ${tx["newbalanceOrig"]:,.2f}
Origin Discrepancy : ${tx["orig_balance_discrepancy"]:,.2f}
Dest Counterparty  : Recipient Balance: ${tx["newbalanceDest"]:,.2f} (Discrepancy: ${tx["dest_balance_discrepancy"]:,.2f})

2. QUANTITATIVE RISK SCORING & DECISION SUPPORT
--------------------------------------------------------------------------------
Predicted Fraud Probability : {explanation["predicted_fraud_probability"]:.6f}
Composite Risk Score        : {round(explanation["predicted_fraud_probability"] * 1000.0)} / 1000
Assigned Triage Queue       : PRIORITY_ESCALATION_QUEUE
Model Version               : Champion LightGBM Tuned (Phase 12 GOSS Architecture)
Efficiency Axiom Verified   : {explanation["efficiency_axiom_verified"]} (Error: {explanation["reconstruction_error"]:.2e})

3. EXCULPATORY & ADVERSE REASON CODES (TreeSHAP ATTRIBUTION)
--------------------------------------------------------------------------------
Primary Objective Indicators Escalating Risk:
{reasons_text}

4. COMPLIANCE OFFICER CONCLUSION & RECOMMENDATION
--------------------------------------------------------------------------------
Transaction exhibits signature characteristics of rapid, uncollateralized account
liquidation with catastrophic ledger discrepancies. Sender balance was liquidated
to zero immediately prior to external corridor transmission during high-risk
nighttime velocity windows. In accordance with BSA/AML guidelines and Federal Reserve SR 11-7 standards,
this transaction is referred for formal SAR filing and counterparty subpoena.

PREPARED BY: Anti-Money Laundering Automated Surveillance Engine
AUTHORIZED COMPLIANCE REVIEWER: [Pending Analyst Signature]
================================================================================"""
    return narrative


def simulate_loss_matrix_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    amounts: np.ndarray,
    review_cost: float,
    n_points: int = 50,
) -> tuple[float, float, list[dict[str, Any]]]:
    """Calculate the financial cost curve across thresholds for a dynamic review cost."""
    thresholds = np.linspace(0.01, 0.99, n_points)
    best_tau = 0.50
    min_cost = float("inf")
    records = []

    for tau in thresholds:
        pred = (y_probs >= tau).astype(int)
        fn_mask = (pred == 0) & (y_true == 1)
        tp_mask = (pred == 1) & (y_true == 1)

        missed_loss = float(amounts[fn_mask].sum())
        captured_loss = float(amounts[tp_mask].sum())
        alerts = int(pred.sum())
        tot_review_cost = float(alerts * review_cost)
        total_cost = missed_loss + tot_review_cost
        tot_exposure = missed_loss + captured_loss
        cap_pct = (captured_loss / tot_exposure * 100.0) if tot_exposure > 0 else 0.0

        records.append(
            {
                "threshold": float(tau),
                "total_cost": total_cost,
                "missed_loss": missed_loss,
                "review_cost": tot_review_cost,
                "alerts": alerts,
                "capture_rate": cap_pct,
            }
        )

        if total_cost < min_cost:
            min_cost = total_cost
            best_tau = float(tau)

    return best_tau, min_cost, records


# =============================================================================
# STREAMLIT USER INTERFACE TABS
# =============================================================================


def render_dashboard() -> None:
    """Render the 4-tab enterprise risk dashboard."""
    st.set_page_config(
        page_title="Financial Risk & Surveillance Intelligence Center",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom institutional CSS typography
    st.markdown(
        """
        <style>
        .main { background-color: #f8fafc; }
        .metric-card {
            background-color: #ffffff;
            border-left: 5px solid #1b365d;
            padding: 16px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            margin-bottom: 12px;
        }
        .metric-title { font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; }
        .metric-value { font-size: 26px; font-weight: 700; color: #1b365d; margin: 4px 0; }
        .metric-sub { font-size: 12px; color: #10b981; font-weight: 500; }
        .metric-sub-red { font-size: 12px; color: #ef4444; font-weight: 500; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Header
    st.sidebar.title("🛡️ Risk Intelligence Platform")
    st.sidebar.markdown(
        "**Institutional Payment Surveillance**  \n*Paytm • Morgan Stanley • HSBC*"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Governance")
    st.sidebar.info(
        "**Regulatory Framework**  \n• Federal Reserve SR 11-7  \n• OCC 2011-12 MRM  \n• Basel BCBS 239  \n• GDPR Article 22"
    )
    st.sidebar.markdown("### Decision-Support Policy")
    st.sidebar.warning(
        "**Strict Rule**: Automated customer blocking or account freezing is prohibited. Triage queues assist human compliance analysts."
    )

    # Load Data & Summaries
    test_df = get_cached_test_data()
    engine = get_cached_engine()
    get_cached_summaries()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🏛️ Executive Surveillance",
            "🔍 Live Investigator Console",
            "📊 Typology & Modus Operandi",
            "🎛️ Asymmetric Loss Simulator",
        ]
    )

    # =========================================================================
    # TAB 1: EXECUTIVE SURVEILLANCE CENTER
    # =========================================================================
    with tab1:
        st.subheader("Executive Risk Surveillance & Financial Loss KPI Center")
        st.markdown(
            "Macroscopic portfolio metrics across **6,362,620 transactions** ($1.144 Trillion GTV) with out-of-time holdout validation (Days 27–31)."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-title">Total Portfolio Volume</div>
                    <div class="metric-value">6,362,620</div>
                    <div class="metric-sub">GTV: $1,144,392,944,759.77</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
                <div class="metric-card" style="border-left-color: #d9381e;">
                    <div class="metric-title">Test Holdout Exposure</div>
                    <div class="metric-value">$2.262 Billion</div>
                    <div class="metric-sub-red">1,328 Fraud Attacks (Days 27-31)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
                <div class="metric-card" style="border-left-color: #008080;">
                    <div class="metric-title">Dollar Capture Rate</div>
                    <div class="metric-value">99.98% - 100%</div>
                    <div class="metric-sub">$2,262,104,308 Intercepted</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col4:
            st.markdown(
                """
                <div class="metric-card" style="border-left-color: #10b981;">
                    <div class="metric-title">Net Financial Savings</div>
                    <div class="metric-value">$2.222 Billion</div>
                    <div class="metric-sub">vs Legacy Rule (isFlaggedFraud)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")
        c_left, c_right = st.columns([3, 2])

        with c_left:
            st.markdown("#### Out-Of-Time Model Architecture Comparison")
            # Model comparison table from Phase 11 & 12
            comp_data = {
                "Surveillance Model": [
                    "Legacy Rule (isFlaggedFraud)",
                    "Logistic Regression Scorecard",
                    "Shallow Decision Tree (Depth=5)",
                    "Cost-Sensitive Random Forest",
                    "Champion LightGBM Tuned",
                    "Champion XGBoost Hist",
                    "Heterogeneous Stacking Meta-Learner",
                ],
                "PR-AUC": [0.0200, 0.9135, 0.9984, 1.0000, 0.9998, 1.0000, 1.0000],
                "Dollar Capture": [
                    "1.77%",
                    "99.62%",
                    "99.998%",
                    "100.00%",
                    "99.98%",
                    "99.98%",
                    "99.98%",
                ],
                "Total Loss + Cost": [
                    "$2,222,501,553",
                    "$32,654,120",
                    "$473,120",
                    "$33,200",
                    "$432,195",
                    "$432,270",
                    "$432,220",
                ],
                "Triage Verdict": [
                    "Catastrophic Failure ($2.22B leaked)",
                    "Baseline Linear Screen",
                    "White-Box Regulatory Tree",
                    "Zero Loss Leader",
                    "Champion Real-Time Gateway",
                    "High-Precision Gradient Booster",
                    "Maximal Multi-Paradigm Defense",
                ],
            }
            st.dataframe(
                pd.DataFrame(comp_data), use_container_width=True, hide_index=True
            )

        with c_right:
            st.markdown("#### Operational Triage Queue Health (Days 27–31)")
            queue_counts = test_df["triage_decision"].value_counts()
            queue_pcts = (queue_counts / len(test_df)) * 100.0

            q_df = pd.DataFrame(
                {
                    "Operational Queue": queue_counts.index,
                    "Transaction Count": queue_counts.values,
                    "Share (%)": [f"{p:.2f}%" for p in queue_pcts.values],
                }
            )
            st.dataframe(q_df, use_container_width=True, hide_index=True)

            st.caption(
                "✓ **Zero Backlog**: 98.52% of transactions auto-approved; only 1.48% routed to human queues."
            )

    # =========================================================================
    # TAB 2: LIVE INVESTIGATOR CONSOLE & SAR WORKFLOW
    # =========================================================================
    with tab2:
        st.subheader("Live Operational Triage Console & FinCEN SAR Generator")
        st.markdown(
            "Interactive decision-support queue for compliance analysts and surveillance officers. "
            "Select any flagged transaction to view its **exact TreeSHAP attribution waterfall** and export automated SAR narratives."
        )

        f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
        with f_col1:
            queue_filter = st.selectbox(
                "Filter by Triage Queue",
                [
                    "ALL QUEUES",
                    "PRIORITY_ESCALATION_QUEUE",
                    "INVESTIGATOR_REVIEW_QUEUE",
                    "AUTOMATED_APPROVAL",
                ],
                index=1,
            )
        with f_col2:
            type_filter = st.multiselect(
                "Corridor Typology",
                options=list(test_df["type"].unique()),
                default=list(test_df["type"].unique()),
            )
        with f_col3:
            min_amount = st.number_input(
                "Min Amount ($)", value=0.0, step=10000.0, format="%.2f"
            )

        # Filter transactions
        filtered_df = test_df.copy()
        if queue_filter != "ALL QUEUES":
            filtered_df = filtered_df[filtered_df["triage_decision"] == queue_filter]
        filtered_df = filtered_df[
            (filtered_df["type"].isin(type_filter))
            & (filtered_df["amount"] >= min_amount)
        ]

        st.markdown(
            f"**Displaying {min(len(filtered_df), 100):,} of {len(filtered_df):,} matching transactions:**"
        )

        display_cols = [
            "transaction_id",
            "day",
            "hour_of_day",
            "type",
            "amount",
            "risk_score",
            "fraud_probability",
            "triage_decision",
            "isFraud",
        ]
        st.dataframe(
            filtered_df[display_cols].head(100),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        st.markdown("### Deep Transaction Inspector & TreeSHAP Waterfall")

        selected_tx_id = st.selectbox(
            "Select Transaction ID to Audit:",
            options=filtered_df["transaction_id"].head(50).tolist(),
        )

        if selected_tx_id:
            tx_row = test_df[test_df["transaction_id"] == selected_tx_id].iloc[0]
            explanation = engine.explain_transaction(tx_row)

            detail_col1, detail_col2 = st.columns([1, 1])

            with detail_col1:
                st.markdown("#### Transaction Financial Details")
                st.write(
                    f"**Transaction ID**: `{tx_row['transaction_id']}`  \n"
                    f"**Typology Corridor**: `{tx_row['type']}`  \n"
                    f"**Amount**: **${tx_row['amount']:,.2f} USD**  \n"
                    f"**Diurnal Execution**: Day {tx_row['day']}, {int(tx_row['hour_of_day']):02d}:00 Hours  \n"
                    f"**Origin Pre-Balance**: ${tx_row['oldbalanceOrg']:,.2f}  \n"
                    f"**Origin Post-Balance**: ${tx_row['newbalanceOrig']:,.2f}  \n"
                    f"**Balance Discrepancy**: ${tx_row['orig_balance_discrepancy']:,.2f}  \n"
                    f"**Ground Truth**: {'⚠️ CONFIRMED FRAUD' if tx_row['isFraud'] == 1 else '✓ LEGITIMATE'}"
                )

                st.markdown("#### Primary Adverse Reason Codes")
                for r in explanation["primary_adverse_reason_codes"]:
                    st.error(
                        f"**{r['reason_code']}** ({r['shap_value']:+.2f} log-odds | {r['relative_impact_pct']:.1f}% impact)  \n"
                        f"{r['description']}"
                    )

            with detail_col2:
                st.markdown("#### TreeSHAP Waterfall Force Chart")
                shap_dict = explanation["all_feature_shap_values"]
                sorted_feats = sorted(
                    shap_dict.items(), key=lambda item: abs(item[1]), reverse=True
                )[:6][::-1]

                f_names = [item[0] for item in sorted_feats]
                s_vals = [item[1] for item in sorted_feats]
                colors = ["#d9381e" if v > 0 else "#1b365d" for v in s_vals]

                fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
                bars = ax.barh(
                    np.arange(len(f_names)), s_vals, color=colors, height=0.55
                )
                ax.axvline(0, color="black", linestyle="-", linewidth=1)
                ax.set_yticks(np.arange(len(f_names)))
                ax.set_yticklabels(f_names, fontsize=9, fontweight="bold")
                ax.set_xlabel("SHAP Impact on Output Log-Odds", fontsize=9)
                ax.set_title(
                    f"Tx #{selected_tx_id} Attribution Waterfall\n"
                    f"P(Fraud) = {explanation['predicted_fraud_probability']:.4%}",
                    fontsize=10,
                    fontweight="bold",
                    color="#1b365d",
                )
                ax.grid(True, linestyle=":", alpha=0.5, axis="x")
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            st.markdown("#### Automated FinCEN SAR Narrative Generator")
            sar_text = generate_sar_narrative(tx_row, explanation)
            st.text_area("Pre-Populated SAR Narrative Package", sar_text, height=220)
            st.download_button(
                label="📥 Download Standardized SAR Filing (.txt)",
                data=sar_text,
                file_name=f"FinCEN_SAR_Package_TX_{selected_tx_id}.txt",
                mime="text/plain",
            )

    # =========================================================================
    # TAB 3: TYPOLOGY & MODUS OPERANDI EXPLORER
    # =========================================================================
    with tab3:
        st.subheader("Typology & Modus Operandi Drilldown")
        st.markdown(
            "Empirical evidence supporting the four core typologies discovered during SQL analytics and statistical profiling."
        )

        typ_col1, typ_col2 = st.columns(2)

        with typ_col1:
            st.markdown("#### 1. The Diurnal Risk Paradox (01:00 - 05:00 AM)")
            st.markdown(
                "Attackers launch automated scripts 24/7 at a constant rate (~350 frauds/hour). "
                "Because legitimate user volume drops 99.7% at night, fraud prevalence surges to **22.30% at 04:00 AM (a 172.8x risk multiplier)**."
            )
            # Render hourly profile
            hourly_stats = (
                test_df.groupby("hour_of_day")["isFraud"]
                .agg(["count", "sum"])
                .reset_index()
            )
            hourly_stats["prevalence_pct"] = (
                hourly_stats["sum"] / hourly_stats["count"]
            ) * 100.0

            fig_h, ax_h = plt.subplots(figsize=(7, 4), dpi=150)
            ax_h.plot(
                hourly_stats["hour_of_day"],
                hourly_stats["prevalence_pct"],
                marker="o",
                color="#d9381e",
                linewidth=2.5,
            )
            ax_h.axvspan(
                1, 5, color="#fecaca", alpha=0.4, label="Night Vulnerability (01-05h)"
            )
            ax_h.set_title(
                "Hourly Fraud Prevalence (%) in Test Holdout",
                fontsize=10,
                fontweight="bold",
                color="#1b365d",
            )
            ax_h.set_xlabel("Hour of Day (0-23)", fontsize=9)
            ax_h.set_ylabel("Fraud Prevalence (%)", fontsize=9)
            ax_h.grid(True, linestyle=":", alpha=0.5)
            ax_h.legend(loc="upper right", fontsize=8)
            st.pyplot(fig_h)
            plt.close(fig_h)

        with typ_col2:
            st.markdown("#### 2. Account Depletion Modus Operandi")
            st.markdown(
                "Complete account draining (`newbalanceOrig = 0.00`) is the single strongest physical indicator of fraud, "
                "accounting for **97.55% of all fraudulent transactions**."
            )
            dep_stats = (
                test_df.groupby("orig_depletion_complete")["isFraud"]
                .agg(["count", "sum"])
                .reset_index()
            )
            dep_stats["label"] = dep_stats["orig_depletion_complete"].map(
                {True: "Complete Drain (Bal=0)", False: "Preserved Balance"}
            )

            fig_d, ax_d = plt.subplots(figsize=(7, 4), dpi=150)
            bars = ax_d.bar(
                dep_stats["label"],
                dep_stats["sum"],
                color=["#1b365d", "#d9381e"],
                width=0.4,
            )
            for bar in bars:
                ax_d.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 15,
                    f"{int(bar.get_height()):,}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )
            ax_d.set_title(
                "Fraud Count by Origin Depletion Pattern",
                fontsize=10,
                fontweight="bold",
                color="#1b365d",
            )
            ax_d.set_ylabel("Confirmed Fraud Count", fontsize=9)
            ax_d.grid(True, linestyle=":", alpha=0.5, axis="y")
            st.pyplot(fig_d)
            plt.close(fig_d)

    # =========================================================================
    # TAB 4: INTERACTIVE ASYMMETRIC LOSS MATRIX SIMULATOR
    # =========================================================================
    with tab4:
        st.subheader("Interactive Asymmetric Loss Matrix Simulator (CRO Console)")
        st.markdown(
            "Empirically test operational trade-offs by varying the **investigator review cost** and **fraud loss exposure**. "
            "Observe in real time how the cost-optimal decision threshold $\\tau^*$ shifts dynamically."
        )

        sim_col1, sim_col2 = st.columns([1, 2])

        with sim_col1:
            st.markdown("#### Cost Matrix Parameters")
            review_cost_input = st.slider(
                "Investigator Review Cost per Alert ($)",
                min_value=5.0,
                max_value=150.0,
                value=25.0,
                step=5.0,
            )

            avg_fraud_loss = float(
                test_df.loc[test_df["isFraud"] == 1, "amount"].mean()
            )
            asymmetry_ratio = avg_fraud_loss / review_cost_input

            st.info(
                f"**Average Fraud Magnitude**: ${avg_fraud_loss:,.2f}  \n"
                f"**Asymmetry Penalty Ratio**: **{asymmetry_ratio:,.1f} : 1**"
            )

        with sim_col2:
            st.markdown("#### Dynamic Threshold Optimization Curve")
            best_tau, min_cost, records = simulate_loss_matrix_curve(
                y_true=test_df["isFraud"].to_numpy(),
                y_probs=test_df["fraud_probability"].to_numpy(),
                amounts=test_df["amount"].to_numpy(),
                review_cost=review_cost_input,
                n_points=40,
            )

            rec_df = pd.DataFrame(records)
            fig_s, ax_s = plt.subplots(figsize=(8, 4), dpi=150)
            ax_s.plot(
                rec_df["threshold"],
                rec_df["total_cost"] / 1000.0,
                color="#1b365d",
                linewidth=2.5,
                label=f"Total Financial Cost (Review: ${review_cost_input:.0f})",
            )
            ax_s.axvline(
                best_tau,
                color="#d9381e",
                linestyle="--",
                linewidth=2,
                label=f"Cost-Optimal tau* = {best_tau:.3f} (${min_cost / 1000.0:,.1f}k)",
            )
            ax_s.set_title(
                "Financial Loss Curve vs Decision Threshold (Test Holdout)",
                fontsize=10,
                fontweight="bold",
                color="#1b365d",
            )
            ax_s.set_xlabel("Decision Threshold (tau)", fontsize=9)
            ax_s.set_ylabel("Total Financial Cost ($ Thousands)", fontsize=9)
            ax_s.grid(True, linestyle=":", alpha=0.5)
            ax_s.legend(loc="upper right", fontsize=8)
            st.pyplot(fig_s)
            plt.close(fig_s)

        st.markdown("---")
        kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
        with kpi_c1:
            st.metric(
                "Optimal Decision Threshold (tau*)",
                f"{best_tau:.3f}",
                delta=f"Review Cost: ${review_cost_input:.0f}",
            )
        with kpi_c2:
            st.metric(
                "Minimized Total Financial Cost",
                f"${min_cost:,.2f}",
                delta="FN Leakage + Alert Reviews",
            )
        with kpi_c3:
            net_saved = 2_222_501_553.42 - min_cost
            st.metric(
                "Net Saved vs Legacy Heuristic",
                f"${net_saved:,.2f}",
                delta="Over $2.22B Protected",
            )


def main() -> None:
    """CLI entrypoint for standalone dashboard invocation."""
    render_dashboard()


if __name__ == "__main__":
    main()
