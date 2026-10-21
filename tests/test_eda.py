"""Test Suite for Phase 10: Exploratory Data Analysis & Statistical Hypothesis Testing.

Validates non-parametric hypothesis testing (Mann-Whitney U, Chi-Square, Kolmogorov-Smirnov),
Weight of Evidence (WoE) and Information Value (IV) computations, figure rendering,
and summary persistence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from financial_risk.config import DATA_PROCESSED_DIR, FIGURES_DIR
from financial_risk.eda import FinancialRiskEDA


@pytest.fixture(scope="module")
def eda_engine():
    """Provides an instance of FinancialRiskEDA."""
    return FinancialRiskEDA()


@pytest.fixture(scope="module")
def sample_tx_df(eda_engine):
    """Loads a representative sample of transaction features for quick test verification."""
    cols = [
        "amount",
        "log_amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "orig_depletion_complete",
        "orig_balance_discrepancy",
        "oldbalanceDest",
        "newbalanceDest",
        "dest_balance_discrepancy",
        "dest_inbound_prior_count",
        "type",
        "is_night_hours",
        "is_weekend",
        "is_merchant_dest",
        "is_transfer_or_cashout",
        "isFraud",
    ]
    # Read subset to optimize test execution
    return eda_engine.load_transactions_data(columns=cols)


def test_eda_initialization(eda_engine):
    """Verify EDA engine paths and directories exist."""
    assert eda_engine.features_path.exists()
    assert eda_engine.cases_path.exists()
    assert eda_engine.figures_dir.exists()


def test_load_investigations_data(eda_engine):
    """Verify investigation cases DataFrame loads correctly."""
    cases_df = eda_engine.load_investigations_data()
    assert len(cases_df) == 5000
    assert "turnaround_time_hours" in cases_df.columns
    assert "is_sla_breached" in cases_df.columns
    assert "priority" in cases_df.columns


def test_mann_whitney_tests(eda_engine, sample_tx_df):
    """Verify Mann-Whitney U rank-sum test execution and effect size computation."""
    res = eda_engine.run_mann_whitney_tests(
        sample_tx_df, continuous_cols=["amount", "log_amount", "oldbalanceOrg"]
    )
    assert len(res) == 3
    for metrics in res.values():
        assert metrics["u_statistic"] > 0
        assert metrics["p_value"] < 0.001
        assert metrics["is_statistically_significant"] is True
        assert -1.0 <= metrics["rank_biserial_correlation"] <= 1.0
        assert metrics["fraud_median"] >= 0
        assert metrics["benign_median"] >= 0


def test_chi_square_tests(eda_engine, sample_tx_df):
    """Verify Chi-Square independence test and Cramér's V effect size."""
    res = eda_engine.run_chi_square_tests(
        sample_tx_df,
        categorical_cols=["type", "is_night_hours", "orig_depletion_complete"],
    )
    assert len(res) == 3
    for metrics in res.values():
        assert metrics["chi2_statistic"] > 0
        assert metrics["p_value"] < 0.001
        assert metrics["is_statistically_significant"] is True
        assert 0.0 <= metrics["cramers_v"] <= 1.0
        assert metrics["degrees_of_freedom"] >= 1


def test_kolmogorov_smirnov_tests(eda_engine, sample_tx_df):
    """Verify Two-Sample K-S test max cumulative separation metric."""
    res = eda_engine.run_kolmogorov_smirnov_tests(
        sample_tx_df,
        continuous_cols=["amount", "oldbalanceOrg", "orig_balance_discrepancy"],
    )
    assert len(res) == 3
    for metrics in res.values():
        assert 0.0 <= metrics["ks_statistic_d"] <= 1.0
        assert metrics["p_value"] < 0.001
        assert metrics["is_statistically_significant"] is True

    # orig_balance_discrepancy must exhibit high separation (D > 0.70)
    assert res["orig_balance_discrepancy"]["ks_statistic_d"] > 0.70


def test_information_values(eda_engine, sample_tx_df):
    """Verify Information Value (IV) calculation and benchmark categorization."""
    res = eda_engine.compute_information_values(
        sample_tx_df,
        features=[
            "orig_depletion_complete",
            "is_transfer_or_cashout",
            "is_night_hours",
            "log_amount",
        ],
    )
    assert len(res) == 4
    for metrics in res.values():
        assert metrics["information_value"] > 0.0
        assert metrics["num_bins"] >= 2

    # Account depletion and channel type must be extraordinary predictors (IV >= 0.50)
    assert res["orig_depletion_complete"]["information_value"] >= 0.50
    assert res["is_transfer_or_cashout"]["information_value"] >= 0.50


def test_spearman_correlation_matrix(eda_engine, sample_tx_df):
    """Verify Spearman rank correlation matrix properties."""
    num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "isFraud"]
    corr = eda_engine.compute_correlation_matrix(sample_tx_df, num_cols=num_cols)
    assert corr.shape == (4, 4)
    # Diagonal must be identically 1.0
    for col in num_cols:
        assert np.isclose(corr.loc[col, col], 1.0)
    # Correlation values bounded between -1.0 and 1.0
    assert (corr.values >= -1.0).all() and (corr.values <= 1.0).all()


def test_investigation_metrics(eda_engine):
    """Verify operational investigation and SLA metrics summary."""
    metrics = eda_engine.compute_investigation_metrics()
    assert metrics["total_cases"] == 5000
    assert metrics["closed_count"] + metrics["open_backlog"] == 5000
    assert 30.0 <= metrics["overall_sla_breach_rate_pct"] <= 40.0
    assert (
        metrics["turnaround_time_hours"]["median_p50"]
        <= metrics["turnaround_time_hours"]["p90"]
    )
    assert len(metrics["priority_breakdown"]) == 3
    assert len(metrics["squad_breakdown"]) == 3


def test_figures_exist():
    """Verify all 6 publication-ready figures exist and have non-zero size."""
    expected_figures = [
        "fig_01_amount_distribution.png",
        "fig_02_channel_vulnerability.png",
        "fig_03_diurnal_hourly_risk.png",
        "fig_04_correlation_heatmap.png",
        "fig_05_investigation_operations.png",
        "fig_06_woe_iv_ranking.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 50_000, (
            f"Figure {fig_name} is too small (<50KB)"
        )


def test_statistical_summary_json():
    """Verify eda_statistical_summary.json exists and contains complete results."""
    summary_path = DATA_PROCESSED_DIR / "eda_statistical_summary.json"
    assert summary_path.exists()
    content = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "metadata" in content
    assert content["metadata"]["total_transactions"] == 6362620
    assert content["metadata"]["fraud_transactions"] == 8213
    assert "mann_whitney_tests" in content
    assert "chi_square_tests" in content
    assert "kolmogorov_smirnov_tests" in content
    assert "information_values" in content
    assert len(content["figures_generated"]) == 6
