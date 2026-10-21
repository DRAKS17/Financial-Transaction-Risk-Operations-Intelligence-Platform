"""Automated test suite for Phase 15 Interactive BI Risk Dashboard.

Validates Streamlit data caching, model loading, FinCEN SAR narrative generation,
dynamic asymmetric loss matrix simulation, and documentation artifacts.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_risk.dashboard import (
    generate_sar_narrative,
    get_cached_engine,
    get_cached_summaries,
    get_cached_test_data,
    simulate_loss_matrix_curve,
)
from financial_risk.explainability import RiskExplainabilityEngine


@pytest.fixture(scope="module")
def engine() -> RiskExplainabilityEngine:
    """Fixture providing initialized RiskExplainabilityEngine."""
    return get_cached_engine()


@pytest.fixture(scope="module")
def test_df() -> pd.DataFrame:
    """Fixture providing cached test holdout dataset."""
    return get_cached_test_data()


def test_cached_engine_initialization(engine: RiskExplainabilityEngine) -> None:
    """Verify cached engine contains valid champion model and TreeExplainer."""
    assert engine.model is not None
    assert engine.explainer is not None
    assert isinstance(engine.base_value, float)
    assert np.isfinite(engine.base_value)


def test_cached_test_data_integrity(test_df: pd.DataFrame) -> None:
    """Verify test holdout data partitioning, feature columns, and pre-computed risk scores."""
    assert len(test_df) == 89_688
    assert (test_df["day"] >= 27).all()
    assert "fraud_probability" in test_df.columns
    assert "risk_score" in test_df.columns
    assert "triage_decision" in test_df.columns

    # Check risk score bounds
    assert test_df["risk_score"].min() >= 0
    assert test_df["risk_score"].max() <= 1000

    # Check triage queue values
    valid_queues = {
        "AUTOMATED_APPROVAL",
        "INVESTIGATOR_REVIEW_QUEUE",
        "PRIORITY_ESCALATION_QUEUE",
    }
    assert set(test_df["triage_decision"].unique()).issubset(valid_queues)


def test_cached_summaries_loading() -> None:
    """Verify cached summaries dictionary loads advanced ML and explainability data."""
    summaries = get_cached_summaries()
    assert isinstance(summaries, dict)
    assert "advanced_ml" in summaries
    assert "explainability" in summaries
    assert "global_feature_attributions" in summaries["explainability"]


def test_generate_sar_narrative(
    engine: RiskExplainabilityEngine,
    test_df: pd.DataFrame,
) -> None:
    """Verify FinCEN SAR narrative synthesis and regulatory compliance formatting."""
    fraud_sample = test_df[test_df["isFraud"] == 1].iloc[0]
    explanation = engine.explain_transaction(fraud_sample)

    sar_text = generate_sar_narrative(fraud_sample, explanation)
    assert isinstance(sar_text, str)
    assert len(sar_text) > 500

    # Verify FinCEN mandatory sections
    assert "FINANCIAL CRIMES ENFORCEMENT NETWORK (FinCEN)" in sar_text
    assert "SUSPICIOUS ACTIVITY REPORT (SAR)" in sar_text
    assert f"#{fraud_sample['transaction_id']}" in sar_text
    assert f"${fraud_sample['amount']:,.2f} USD" in sar_text
    assert "TreeSHAP ATTRIBUTION" in sar_text
    assert "Federal Reserve SR 11-7" in sar_text
    assert "PRIORITY_ESCALATION_QUEUE" in sar_text


def test_simulate_loss_matrix_curve() -> None:
    """Verify dynamic threshold optimization curve simulation under varying review costs."""
    np.random.seed(42)
    n = 1000
    y_true = np.zeros(n, dtype=int)
    y_true[:50] = 1  # 5% fraud

    # Fraud probabilities: higher for true fraud
    y_probs = np.random.uniform(0.0, 0.2, size=n)
    y_probs[:50] = np.random.uniform(0.7, 1.0, size=50)

    amounts = np.random.uniform(100.0, 10_000.0, size=n)
    amounts[:50] = np.random.uniform(500_000.0, 2_000_000.0, size=50)

    # Test with standard $25 review cost
    best_tau_25, min_cost_25, records_25 = simulate_loss_matrix_curve(
        y_true, y_probs, amounts, review_cost=25.0, n_points=30
    )
    assert 0.01 <= best_tau_25 <= 0.99
    assert min_cost_25 > 0.0
    assert len(records_25) == 30

    # Test with high $100 review cost (should penalize false alerts more)
    _best_tau_100, min_cost_100, records_100 = simulate_loss_matrix_curve(
        y_true, y_probs, amounts, review_cost=100.0, n_points=30
    )
    assert min_cost_100 >= min_cost_25
    assert len(records_100) == 30


def test_dashboard_user_guide_artifact() -> None:
    """Verify dashboard user guide exists and contains comprehensive instructions."""
    guide_path = Path("docs/dashboard_user_guide.md")
    assert guide_path.exists(), "Missing docs/dashboard_user_guide.md"
    content = guide_path.read_text(encoding="utf-8")
    assert "Executive Overview" in content
    assert "Dashboard Navigation & Tab Architecture" in content
    assert "Mandatory Decision-Support Operating Policy" in content
    assert "FinCEN SAR Narrative Generation Standard" in content
    assert "Tier-1 Banking Deployment Specifications" in content
