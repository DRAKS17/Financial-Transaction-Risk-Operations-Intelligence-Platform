"""Exploratory Data Analysis and Statistical Hypothesis Testing Engine.

Conducts non-parametric statistical hypothesis testing (Mann-Whitney U, Chi-Square, Kolmogorov-Smirnov),
evaluates predictive discriminatory power via Weight of Evidence (WoE) and Information Value (IV),
computes rank correlation matrices, and generates high-resolution surveillance visualizations.
"""

from __future__ import annotations

import argparse
import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from scipy import stats

from financial_risk.config import (
    DATA_PROCESSED_DIR,
    DOCS_DIR,
    FIGURES_DIR,
)

# Use non-interactive backend for server/CLI rendering
matplotlib.use("Agg")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def decimal_serializer(obj: Any) -> Any:
    """Helper serializer for json.dumps to handle Decimals, dates, and numpy types."""
    if isinstance(obj, (Decimal, np.floating)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class FinancialRiskEDA:
    """Orchestrates statistical hypothesis testing, information value ranking, and risk visualization."""

    def __init__(
        self,
        features_path: Path = DATA_PROCESSED_DIR / "features_engineered.parquet",
        cases_path: Path = DATA_PROCESSED_DIR / "investigation_cases_cleaned.parquet",
        figures_dir: Path = FIGURES_DIR,
        summary_path: Path = DATA_PROCESSED_DIR / "eda_statistical_summary.json",
        report_path: Path = DOCS_DIR / "eda_statistical_report.md",
    ):
        self.features_path = features_path
        self.cases_path = cases_path
        self.figures_dir = figures_dir
        self.summary_path = summary_path
        self.report_path = report_path
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def load_transactions_data(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Load feature engineered transactions parquet into a Pandas DataFrame."""
        logger.info(f"Loading transaction features from {self.features_path}...")
        table = pq.read_table(self.features_path, columns=columns)
        df = table.to_pandas()
        logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} columns.")
        return df

    def load_investigations_data(self) -> pd.DataFrame:
        """Load operational investigation cases parquet into a Pandas DataFrame."""
        logger.info(f"Loading investigation cases from {self.cases_path}...")
        table = pq.read_table(self.cases_path)
        df = table.to_pandas()
        logger.info(f"Loaded {len(df):,} operational cases.")
        return df

    def run_mann_whitney_tests(
        self, df: pd.DataFrame, continuous_cols: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Execute two-sided Mann-Whitney U rank-sum tests for continuous features against isFraud."""
        if continuous_cols is None:
            continuous_cols = [
                "amount",
                "log_amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "orig_balance_discrepancy",
                "oldbalanceDest",
                "newbalanceDest",
                "dest_balance_discrepancy",
            ]

        logger.info(
            f"Executing Mann-Whitney U tests on {len(continuous_cols)} features..."
        )
        fraud_mask = df["isFraud"] == 1
        n_fraud = int(fraud_mask.sum())
        n_benign = int((~fraud_mask).sum())

        results = {}
        for col in continuous_cols:
            if col not in df.columns:
                continue
            fraud_vals = df.loc[fraud_mask, col].to_numpy()
            benign_vals = df.loc[~fraud_mask, col].to_numpy()

            u_stat, p_val = stats.mannwhitneyu(
                fraud_vals, benign_vals, alternative="two-sided"
            )
            # Rank-biserial correlation effect size: r = 1 - (2U / (n1 * n2))
            rank_biserial_r = float(1.0 - (2.0 * u_stat / (n_fraud * n_benign)))

            fraud_median = float(np.median(fraud_vals))
            benign_median = float(np.median(benign_vals))
            fraud_iqr = float(
                np.percentile(fraud_vals, 75) - np.percentile(fraud_vals, 25)
            )
            benign_iqr = float(
                np.percentile(benign_vals, 75) - np.percentile(benign_vals, 25)
            )

            results[col] = {
                "u_statistic": float(u_stat),
                "p_value": float(p_val),
                "rank_biserial_correlation": round(rank_biserial_r, 4),
                "is_statistically_significant": bool(p_val < 0.001),
                "fraud_median": round(fraud_median, 2),
                "benign_median": round(benign_median, 2),
                "fraud_iqr": round(fraud_iqr, 2),
                "benign_iqr": round(benign_iqr, 2),
                "effect_size_magnitude": (
                    "Large"
                    if abs(rank_biserial_r) >= 0.5
                    else "Medium"
                    if abs(rank_biserial_r) >= 0.3
                    else "Small"
                ),
            }
            logger.info(
                f"  - {col:<26}: U = {u_stat:>14,.0f}, p = {p_val:.2e}, r = {rank_biserial_r:>+.4f} ({results[col]['effect_size_magnitude']})"
            )

        return results

    def run_chi_square_tests(
        self, df: pd.DataFrame, categorical_cols: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Execute Chi-Square tests of independence against isFraud with Cramér's V effect size."""
        if categorical_cols is None:
            categorical_cols = [
                "type",
                "is_night_hours",
                "is_weekend",
                "orig_depletion_complete",
                "is_zero_balance_orig",
                "is_merchant_dest",
                "is_transfer_or_cashout",
            ]

        logger.info(
            f"Executing Chi-Square tests on {len(categorical_cols)} categorical features..."
        )
        n_total = len(df)
        results = {}

        for col in categorical_cols:
            if col not in df.columns:
                continue
            contingency = pd.crosstab(df[col], df["isFraud"])
            chi2_stat, p_val, dof, _ = stats.chi2_contingency(contingency)

            min_dim = min(contingency.shape) - 1
            cramers_v = (
                float(np.sqrt(chi2_stat / (n_total * min_dim))) if min_dim > 0 else 0.0
            )

            results[col] = {
                "chi2_statistic": round(float(chi2_stat), 2),
                "p_value": float(p_val),
                "degrees_of_freedom": int(dof),
                "cramers_v": round(cramers_v, 4),
                "is_statistically_significant": bool(p_val < 0.001),
                "association_strength": (
                    "Strong"
                    if cramers_v >= 0.10
                    else "Moderate"
                    if cramers_v >= 0.05
                    else "Weak"
                ),
            }
            logger.info(
                f"  - {col:<26}: chi2 = {chi2_stat:>12,.2f}, p = {p_val:.2e}, dof = {dof}, V = {cramers_v:.4f} ({results[col]['association_strength']})"
            )

        return results

    def run_kolmogorov_smirnov_tests(
        self, df: pd.DataFrame, continuous_cols: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """Execute Two-Sample Kolmogorov-Smirnov tests evaluating maximum cumulative separation."""
        if continuous_cols is None:
            continuous_cols = [
                "amount",
                "log_amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "orig_balance_discrepancy",
                "oldbalanceDest",
                "newbalanceDest",
            ]

        logger.info(
            f"Executing Two-Sample K-S tests on {len(continuous_cols)} features..."
        )
        fraud_mask = df["isFraud"] == 1
        results = {}

        for col in continuous_cols:
            if col not in df.columns:
                continue
            fraud_vals = df.loc[fraud_mask, col].to_numpy()
            benign_vals = df.loc[~fraud_mask, col].to_numpy()

            ks_stat, p_val = stats.ks_2samp(fraud_vals, benign_vals)

            results[col] = {
                "ks_statistic_d": round(float(ks_stat), 4),
                "p_value": float(p_val),
                "is_statistically_significant": bool(p_val < 0.001),
                "separation_power": (
                    "Exceptional (D >= 0.50)"
                    if ks_stat >= 0.50
                    else "High (D >= 0.30)"
                    if ks_stat >= 0.30
                    else "Moderate (D >= 0.15)"
                    if ks_stat >= 0.15
                    else "Low"
                ),
            }
            logger.info(
                f"  - {col:<26}: KS D = {ks_stat:.4f}, p = {p_val:.2e} ({results[col]['separation_power']})"
            )

        return results

    def compute_information_values(
        self, df: pd.DataFrame, features: list[str] | None = None, n_bins: int = 10
    ) -> dict[str, dict[str, Any]]:
        """Calculate Weight of Evidence (WoE) and Information Value (IV) for credit/fraud scoring."""
        if features is None:
            features = [
                "amount",
                "log_amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "orig_depletion_complete",
                "orig_balance_discrepancy",
                "oldbalanceDest",
                "newbalanceDest",
                "is_transfer_or_cashout",
                "is_night_hours",
                "is_merchant_dest",
                "dest_inbound_prior_count",
            ]

        logger.info(
            f"Calculating Information Value (IV) across {len(features)} risk features..."
        )
        target = df["isFraud"].to_numpy()
        total_0 = int((target == 0).sum())
        total_1 = int((target == 1).sum())

        results = {}
        for col in features:
            if col not in df.columns:
                continue
            series = df[col]
            if pd.api.types.is_numeric_dtype(series) and series.nunique() > n_bins:
                try:
                    binned = pd.qcut(series, q=n_bins, duplicates="drop")
                except ValueError:
                    binned = pd.cut(series, bins=n_bins, duplicates="drop")
            else:
                binned = series

            cross = pd.crosstab(binned, target)
            if 0 not in cross.columns or 1 not in cross.columns:
                continue

            dist_0 = (cross[0] / total_0).to_numpy()
            dist_1 = (cross[1] / total_1).to_numpy()

            # Laplace smoothing to guard against division by zero
            dist_0 = np.clip(dist_0, 1e-6, 1.0)
            dist_1 = np.clip(dist_1, 1e-6, 1.0)

            woe = np.log(dist_0 / dist_1)
            iv = float(((dist_0 - dist_1) * woe).sum())

            predictive_tier = (
                "Extraordinary (IV >= 0.50)"
                if iv >= 0.50
                else "Strong (0.30 <= IV < 0.50)"
                if iv >= 0.30
                else "Medium (0.10 <= IV < 0.30)"
                if iv >= 0.10
                else "Weak (0.02 <= IV < 0.10)"
                if iv >= 0.02
                else "Unpredictive (IV < 0.02)"
            )

            results[col] = {
                "information_value": round(iv, 4),
                "predictive_tier": predictive_tier,
                "num_bins": len(dist_0),
            }
            logger.info(f"  - {col:<26}: IV = {iv:>8.4f} ({predictive_tier})")

        return dict(
            sorted(
                results.items(), key=lambda x: x[1]["information_value"], reverse=True
            )
        )

    def compute_correlation_matrix(
        self, df: pd.DataFrame, num_cols: list[str] | None = None
    ) -> pd.DataFrame:
        """Calculate Spearman rank correlation matrix across numeric predictors."""
        if num_cols is None:
            num_cols = [
                "amount",
                "log_amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "amount_to_orig_balance_ratio",
                "orig_balance_discrepancy",
                "oldbalanceDest",
                "newbalanceDest",
                "dest_balance_discrepancy",
                "dest_inbound_prior_count",
                "hour_of_day",
                "isFraud",
            ]
        valid_cols = [c for c in num_cols if c in df.columns]
        logger.info(
            f"Computing Spearman rank correlation matrix on {len(valid_cols)} predictors..."
        )
        corr_matrix = df[valid_cols].corr(method="spearman")
        return corr_matrix

    def compute_investigation_metrics(self) -> dict[str, Any]:
        """Compute operational investigation and SLA metrics from cleaned cases parquet."""
        cases_df = self.load_investigations_data()

        total_cases = len(cases_df)
        closed_cases = cases_df[cases_df["is_case_closed"]]
        open_cases = cases_df[~cases_df["is_case_closed"]]
        sla_breaches = int(cases_df["is_sla_breached"].sum())

        tat_mean = float(closed_cases["turnaround_time_hours"].mean())
        tat_median = float(closed_cases["turnaround_time_hours"].median())
        tat_p90 = float(closed_cases["turnaround_time_hours"].quantile(0.90))

        priority_summary = (
            cases_df.groupby("priority")
            .agg(
                total_cases=("case_id", "count"),
                sla_breaches=("is_sla_breached", "sum"),
                closed_count=("is_case_closed", "sum"),
            )
            .reset_index()
        )
        priority_summary["breach_rate_pct"] = (
            priority_summary["sla_breaches"] / priority_summary["closed_count"] * 100.0
        ).round(2)

        squad_summary = (
            cases_df.groupby("assigned_team")
            .agg(
                total_cases=("case_id", "count"),
                sla_breaches=("is_sla_breached", "sum"),
                mean_tat=(
                    "turnaround_time_hours",
                    lambda s: round(float(s.dropna().mean()), 2),
                ),
            )
            .reset_index()
        )

        return {
            "total_cases": total_cases,
            "closed_count": len(closed_cases),
            "open_backlog": len(open_cases),
            "sla_breach_count": sla_breaches,
            "overall_sla_breach_rate_pct": round(
                sla_breaches / len(closed_cases) * 100.0, 2
            ),
            "turnaround_time_hours": {
                "mean": round(tat_mean, 2),
                "median_p50": round(tat_median, 2),
                "p90": round(tat_p90, 2),
            },
            "priority_breakdown": priority_summary.to_dict(orient="records"),
            "squad_breakdown": squad_summary.to_dict(orient="records"),
        }

    # =========================================================================
    # VISUALIZATION PIPELINE (HIGH-RESOLUTION FIGURES)
    # =========================================================================

    def plot_fig_01_amount_distribution(self, df: pd.DataFrame) -> Path:
        """Plot Log-Transformed Amount Distribution & Boxplot (Figure 1)."""
        output_file = self.figures_dir / "fig_01_amount_distribution.png"
        logger.info(f"Generating Figure 1: {output_file.name}...")

        sns.set_theme(style="whitegrid", palette="muted")
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        fraud_log = df.loc[df["isFraud"] == 1, "log_amount"].to_numpy()
        benign_log = df.loc[df["isFraud"] == 0, "log_amount"].to_numpy()

        # Panel A: KDE Distribution Comparison
        sns.kdeplot(
            benign_log,
            ax=axes[0],
            color="#2b5c8f",
            fill=True,
            alpha=0.3,
            label=f"Benign (n = {len(benign_log):,})",
        )
        sns.kdeplot(
            fraud_log,
            ax=axes[0],
            color="#d9381e",
            fill=True,
            alpha=0.5,
            label=f"Fraud (n = {len(fraud_log):,})",
        )
        axes[0].set_title(
            "A. Transaction Amount Distribution (Log1p Scale)",
            fontsize=13,
            fontweight="bold",
        )
        axes[0].set_xlabel("log1p(Amount)", fontsize=11)
        axes[0].set_ylabel("Kernel Density", fontsize=11)
        axes[0].legend(loc="upper left", frameon=True)

        # Panel B: Boxplot of log amounts
        plot_df = pd.DataFrame(
            {
                "log_amount": np.concatenate([benign_log[:100000], fraud_log]),
                "Target": ["Benign (Sample 100k)"] * 100000
                + ["Fraud (All 8,213)"] * len(fraud_log),
            }
        )
        sns.boxplot(
            x="Target",
            y="log_amount",
            hue="Target",
            data=plot_df,
            ax=axes[1],
            palette=["#2b5c8f", "#d9381e"],
            fliersize=1.5,
            legend=False,
        )
        axes[1].set_title(
            "B. Dispersion Comparison (Median & IQR)", fontsize=13, fontweight="bold"
        )
        axes[1].set_xlabel("Transaction Class", fontsize=11)
        axes[1].set_ylabel("log1p(Amount)", fontsize=11)

        plt.suptitle(
            "Empirical Evidence of Value Disparity in Fraudulent Transactions",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_fig_02_channel_vulnerability(self, df: pd.DataFrame) -> Path:
        """Plot Channel Fraud Prevalence Rate & Gross Dollar Exposure (Figure 2)."""
        output_file = self.figures_dir / "fig_02_channel_vulnerability.png"
        logger.info(f"Generating Figure 2: {output_file.name}...")

        channel_agg = (
            df.groupby("type")
            .agg(
                total_count=("isFraud", "count"),
                fraud_count=("isFraud", "sum"),
                total_gtv=("amount", "sum"),
                fraud_loss=(
                    "amount",
                    lambda s: s[df.loc[s.index, "isFraud"] == 1].sum(),
                ),
            )
            .reset_index()
        )
        channel_agg["fraud_rate_pct"] = (
            channel_agg["fraud_count"] / channel_agg["total_count"] * 100.0
        )
        channel_agg["fraud_loss_billions"] = channel_agg["fraud_loss"] / 1e9

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        # Panel A: Fraud Rate %
        sns.barplot(
            x="type",
            y="fraud_rate_pct",
            hue="type",
            data=channel_agg,
            ax=axes[0],
            palette="Reds_d",
            legend=False,
        )
        axes[0].set_title(
            "A. Fraud Prevalence Rate by Channel (%)", fontsize=13, fontweight="bold"
        )
        axes[0].set_xlabel("Transaction Channel", fontsize=11)
        axes[0].set_ylabel("Fraud Rate (%)", fontsize=11)
        for p in axes[0].patches:
            axes[0].annotate(
                f"{p.get_height():.3f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

        # Panel B: Fraud Exposure $
        sns.barplot(
            x="type",
            y="fraud_loss_billions",
            hue="type",
            data=channel_agg,
            ax=axes[1],
            palette="Blues_d",
            legend=False,
        )
        axes[1].set_title(
            "B. Gross Fraud Loss Exposure ($ Billions)", fontsize=13, fontweight="bold"
        )
        axes[1].set_xlabel("Transaction Channel", fontsize=11)
        axes[1].set_ylabel("Fraud Loss ($ Billions)", fontsize=11)
        for p in axes[1].patches:
            axes[1].annotate(
                f"${p.get_height():.2f}B",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

        plt.suptitle(
            "Channel Vulnerability: Strict Concentration in TRANSFER & CASH_OUT",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_fig_03_diurnal_hourly_risk(self, df: pd.DataFrame) -> Path:
        """Plot 24-Hour Diurnal Diachronic Risk Profile (Figure 3)."""
        output_file = self.figures_dir / "fig_03_diurnal_hourly_risk.png"
        logger.info(f"Generating Figure 3: {output_file.name}...")

        hourly = (
            df.groupby("hour_of_day")
            .agg(
                total_tx=("isFraud", "count"),
                fraud_tx=("isFraud", "sum"),
            )
            .reset_index()
        )
        hourly["fraud_rate_pct"] = hourly["fraud_tx"] / hourly["total_tx"] * 100.0

        fig, ax1 = plt.subplots(figsize=(14, 6), dpi=300)

        # Volume bars
        color_bars = "#90caf9"
        ax1.bar(
            hourly["hour_of_day"],
            hourly["total_tx"] / 1000.0,
            color=color_bars,
            alpha=0.6,
            label="Transaction Volume (k)",
        )
        ax1.set_xlabel("Hour of Day (00:00 to 23:00)", fontsize=12, fontweight="bold")
        ax1.set_ylabel(
            "Total Transaction Volume (Thousands)",
            color="#1565c0",
            fontsize=12,
            fontweight="bold",
        )
        ax1.tick_params(axis="y", labelcolor="#1565c0")
        ax1.set_xticks(range(24))

        # Night shading (00:00 - 06:00)
        ax1.axvspan(
            -0.5,
            5.5,
            color="#ffebee",
            alpha=0.5,
            label="Nocturnal High-Risk Window (00-06h)",
        )

        # Rate curve
        ax2 = ax1.twinx()
        color_line = "#d32f2f"
        ax2.plot(
            hourly["hour_of_day"],
            hourly["fraud_rate_pct"],
            color=color_line,
            linewidth=2.5,
            marker="o",
            label="Fraud Prevalence Rate (%)",
        )
        ax2.set_ylabel(
            "Fraud Prevalence Rate (%)",
            color=color_line,
            fontsize=12,
            fontweight="bold",
        )
        ax2.tick_params(axis="y", labelcolor=color_line)
        ax2.grid(False)

        # Annotate Peak
        peak_row = hourly.loc[hourly["fraud_rate_pct"].idxmax()]
        ax2.annotate(
            f"Peak Risk: {peak_row['fraud_rate_pct']:.2f}%\nat Hour {int(peak_row['hour_of_day'])}:00",
            xy=(peak_row["hour_of_day"], peak_row["fraud_rate_pct"]),
            xytext=(peak_row["hour_of_day"] + 1.5, peak_row["fraud_rate_pct"] - 3),
            arrowprops={
                "facecolor": "#d32f2f",
                "shrink": 0.05,
                "width": 1.5,
                "headwidth": 8,
            },
            fontweight="bold",
            fontsize=10,
            color="#b71c1c",
        )

        plt.title(
            "Diurnal Surveillance Paradox: Invariant Attack Velocity Meets Collapsing Nocturnal Traffic",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_fig_04_correlation_heatmap(self, corr_matrix: pd.DataFrame) -> Path:
        """Plot Spearman Rank Correlation Matrix (Figure 4)."""
        output_file = self.figures_dir / "fig_04_correlation_heatmap.png"
        logger.info(f"Generating Figure 4: {output_file.name}...")

        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Spearman Rank Correlation (rho)"},
            annot=True,
            fmt=".2f",
            annot_kws={"size": 8},
            ax=ax,
        )
        ax.set_title(
            "Predictor Multicollinearity & Ground-Truth Rank Association Matrix",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_fig_05_investigation_operations(self) -> Path:
        """Plot Operational Investigation Turnaround Time & SLA Breach Rates (Figure 5)."""
        output_file = self.figures_dir / "fig_05_investigation_operations.png"
        logger.info(f"Generating Figure 5: {output_file.name}...")

        cases_df = self.load_investigations_data()
        closed_cases = cases_df[cases_df["is_case_closed"]].copy()

        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

        # Panel A: Violin plot of TAT across priorities
        sns.violinplot(
            x="priority",
            y="turnaround_time_hours",
            hue="priority",
            data=closed_cases,
            ax=axes[0],
            palette="Set2",
            cut=0,
            inner="quartile",
            legend=False,
        )
        # Add target lines
        axes[0].axhline(
            24,
            color="#d32f2f",
            linestyle="--",
            alpha=0.7,
            label="P1 Target (24h)",
        )
        axes[0].axhline(
            48,
            color="#f57c00",
            linestyle="--",
            alpha=0.7,
            label="P2 Target (48h)",
        )
        axes[0].axhline(
            96,
            color="#388e3c",
            linestyle="--",
            alpha=0.7,
            label="P3 Target (96h)",
        )
        axes[0].set_title(
            "A. Turnaround Time Distribution by Priority Tier",
            fontsize=13,
            fontweight="bold",
        )
        axes[0].set_xlabel("Priority Tier", fontsize=11)
        axes[0].set_ylabel("Turnaround Time (Hours)", fontsize=11)
        axes[0].legend(loc="upper right", frameon=True)

        # Panel B: SLA Breach Rate by Squad
        squad_breach = (
            cases_df.groupby("assigned_team")
            .agg(
                total=("case_id", "count"),
                breaches=("is_sla_breached", "sum"),
                closed=("is_case_closed", "sum"),
            )
            .reset_index()
        )
        squad_breach["breach_rate"] = (
            squad_breach["breaches"] / squad_breach["closed"] * 100.0
        )

        sns.barplot(
            x="assigned_team",
            y="breach_rate",
            hue="assigned_team",
            data=squad_breach,
            ax=axes[1],
            palette="magma",
            legend=False,
        )
        axes[1].set_title(
            "B. Closed Case SLA Breach Rate by Assigned Squad (%)",
            fontsize=13,
            fontweight="bold",
        )
        axes[1].set_xlabel("Assigned Investigation Squad", fontsize=11)
        axes[1].set_ylabel("SLA Breach Rate (%)", fontsize=11)
        axes[1].set_xticks(range(len(squad_breach)))
        axes[1].set_xticklabels(squad_breach["assigned_team"], rotation=15, ha="right")
        for p in axes[1].patches:
            axes[1].annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                xytext=(0, 3),
                textcoords="offset points",
            )

        plt.suptitle(
            "Investigation Operations: Capacity Constraints Drive P1 SLA Deterioration",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def plot_fig_06_woe_iv_ranking(self, iv_dict: dict[str, dict[str, Any]]) -> Path:
        """Plot Information Value (IV) Ranking Horizontal Bar Chart (Figure 6)."""
        output_file = self.figures_dir / "fig_06_woe_iv_ranking.png"
        logger.info(f"Generating Figure 6: {output_file.name}...")

        iv_df = (
            pd.DataFrame(
                [
                    {
                        "feature": k,
                        "iv": v["information_value"],
                        "tier": v["predictive_tier"],
                    }
                    for k, v in iv_dict.items()
                ]
            )
            .sort_values(by="iv", ascending=True)
            .reset_index(drop=True)
        )

        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
        colors = [
            "#2e7d32" if x >= 0.5 else "#1565c0" if x >= 0.3 else "#ef6c00"
            for x in iv_df["iv"]
        ]

        bars = ax.barh(iv_df["feature"], iv_df["iv"], color=colors, alpha=0.85)
        ax.set_xlabel("Information Value (IV)", fontsize=12, fontweight="bold")
        ax.set_title(
            "Feature Predictive Power Ranking (Information Value / WoE)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

        # Benchmark thresholds
        ax.axvline(
            0.50,
            color="#2e7d32",
            linestyle="--",
            alpha=0.7,
            label="Extraordinary Predictive Tier (IV >= 0.50)",
        )
        ax.axvline(
            0.30,
            color="#1565c0",
            linestyle="--",
            alpha=0.7,
            label="Strong Predictive Tier (IV >= 0.30)",
        )
        ax.axvline(
            0.10,
            color="#ef6c00",
            linestyle="--",
            alpha=0.7,
            label="Medium Predictive Tier (IV >= 0.10)",
        )

        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.3f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.legend(loc="lower right", frameon=True)
        plt.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return output_file

    def run_full_eda(self) -> dict[str, Any]:
        """Execute full exploratory statistical analysis, generate all figures, and persist summary."""
        logger.info("=" * 80)
        logger.info(
            "STARTING PHASE 10: EXPLORATORY DATA ANALYSIS & STATISTICAL TESTING"
        )
        logger.info("=" * 80)

        df = self.load_transactions_data()

        # 1. Statistical Tests
        mw_results = self.run_mann_whitney_tests(df)
        chi2_results = self.run_chi_square_tests(df)
        ks_results = self.run_kolmogorov_smirnov_tests(df)
        iv_results = self.compute_information_values(df)
        corr_matrix = self.compute_correlation_matrix(df)
        ops_metrics = self.compute_investigation_metrics()

        # 2. Visualizations
        fig1 = self.plot_fig_01_amount_distribution(df)
        fig2 = self.plot_fig_02_channel_vulnerability(df)
        fig3 = self.plot_fig_03_diurnal_hourly_risk(df)
        fig4 = self.plot_fig_04_correlation_heatmap(corr_matrix)
        fig5 = self.plot_fig_05_investigation_operations()
        fig6 = self.plot_fig_06_woe_iv_ranking(iv_results)

        generated_figures = [str(p.name) for p in [fig1, fig2, fig3, fig4, fig5, fig6]]

        summary = {
            "metadata": {
                "total_transactions": len(df),
                "fraud_transactions": int((df["isFraud"] == 1).sum()),
                "benign_transactions": int((df["isFraud"] == 0).sum()),
                "fraud_prevalence_pct": round(
                    float((df["isFraud"] == 1).mean() * 100.0), 4
                ),
            },
            "mann_whitney_tests": mw_results,
            "chi_square_tests": chi2_results,
            "kolmogorov_smirnov_tests": ks_results,
            "information_values": iv_results,
            "investigation_operations": ops_metrics,
            "figures_generated": generated_figures,
        }

        # Save summary JSON
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=decimal_serializer)

        logger.info(f"Statistical summary saved to: {self.summary_path}")
        logger.info(f"Generated {len(generated_figures)} publication-quality figures.")
        return summary


def main() -> None:
    """CLI Entrypoint for Phase 10 EDA & Statistical Testing."""
    parser = argparse.ArgumentParser(
        description="Phase 10 EDA and Statistical Testing Runner"
    )
    parser.parse_args()

    engine = FinancialRiskEDA()
    summary = engine.run_full_eda()

    print("\n" + "=" * 60)
    print("PHASE 10 EDA & STATISTICAL TESTING SUMMARY")
    print("=" * 60)
    print(
        f"Total Population: {summary['metadata']['total_transactions']:,} transactions"
    )
    print(
        f"Fraud Prevalence: {summary['metadata']['fraud_transactions']:,} ({summary['metadata']['fraud_prevalence_pct']:.4f}%)"
    )
    print("\nTop 5 Predictive Features by Information Value (IV):")
    for k, v in list(summary["information_values"].items())[:5]:
        print(
            f"  - {k:<28}: IV = {v['information_value']:.4f} ({v['predictive_tier']})"
        )
    print("\nGenerated Figures:")
    for fig in summary["figures_generated"]:
        print(f"  - reports/figures/{fig}")
    print("=" * 60)


if __name__ == "__main__":
    main()
