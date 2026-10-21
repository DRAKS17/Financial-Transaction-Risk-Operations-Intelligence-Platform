"""Automated test suite for Phase 13 Model Explainability & Feature Attribution (SHAP).

Validates TreeSHAP engine initialization, axiomatic local accuracy (Efficiency Axiom),
reason code generation, case study consistency, and visualization artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from financial_risk.config import (
    DATA_PROCESSED_DIR,
    DOCS_DIR,
    FIGURES_DIR,
)
from financial_risk.explainability import RiskExplainabilityEngine


@pytest.fixture(scope="module")
def engine() -> RiskExplainabilityEngine:
    """Fixture providing initialized RiskExplainabilityEngine."""
    return RiskExplainabilityEngine()


@pytest.fixture(scope="module")
def summary_json() -> dict:
    """Fixture providing parsed explainability_summary.json."""
    summary_path = DATA_PROCESSED_DIR / "explainability_summary.json"
    assert summary_path.exists(), f"Missing summary file: {summary_path}"
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def test_engine_initialization(engine: RiskExplainabilityEngine) -> None:
    """Verify explainability engine config, base value, and reason code maps."""
    assert len(engine.FEATURE_COLS) == 16
    assert isinstance(engine.base_value, float)
    assert np.isfinite(engine.base_value)

    # Check reason code map covers all 16 features
    for col in engine.FEATURE_COLS:
        assert col in engine.REASON_CODE_MAP
        code, desc = engine.REASON_CODE_MAP[col]
        assert code.startswith("RC_")
        assert len(desc) > 10


def test_explain_transaction_efficiency_axiom(
    engine: RiskExplainabilityEngine,
) -> None:
    """Verify Local Accuracy Axiom: f(x) = E[f(x)] + Sum(phi_i) on test transaction."""
    tx_sample = {
        "log_amount": 15.5,
        "oldbalanceOrg": 5_000_000.0,
        "newbalanceOrig": 0.0,
        "amount_to_orig_balance_ratio": 1.0,
        "orig_depletion_complete": 1.0,
        "orig_balance_discrepancy": 5_000_000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "dest_balance_discrepancy": -5_000_000.0,
        "dest_inbound_prior_count": 0,
        "is_transfer": 1,
        "is_cash_out": 0,
        "is_merchant_dest": 0,
        "is_night_hours": 1,
        "sin_hour": 0.0,
        "cos_hour": 1.0,
    }

    explanation = engine.explain_transaction(tx_sample)

    assert "predicted_fraud_probability" in explanation
    assert 0.0 <= explanation["predicted_fraud_probability"] <= 1.0
    assert "model_raw_log_odds" in explanation
    assert "base_expected_log_odds" in explanation
    assert explanation["efficiency_axiom_verified"] is True
    assert explanation["reconstruction_error"] < 1e-4

    # Check reason codes structure
    reasons = explanation["primary_adverse_reason_codes"]
    assert len(reasons) > 0
    for r in reasons:
        assert r["reason_code"].startswith("RC_")
        assert r["direction"] == "ESCALATES_RISK"
        assert r["shap_value"] > 0.0


def test_explain_transaction_pandas_series_input(
    engine: RiskExplainabilityEngine,
) -> None:
    """Verify explain_transaction handles pd.Series input correctly."""
    series_sample = pd.Series(
        {
            "log_amount": 10.0,
            "oldbalanceOrg": 50_000.0,
            "newbalanceOrig": 30_000.0,
            "amount_to_orig_balance_ratio": 0.4,
            "orig_depletion_complete": 0.0,
            "orig_balance_discrepancy": 0.0,
            "oldbalanceDest": 10_000.0,
            "newbalanceDest": 30_000.0,
            "dest_balance_discrepancy": 0.0,
            "dest_inbound_prior_count": 2,
            "is_transfer": 0,
            "is_cash_out": 1,
            "is_merchant_dest": 0,
            "is_night_hours": 0,
            "sin_hour": 0.5,
            "cos_hour": 0.866,
        }
    )

    explanation = engine.explain_transaction(series_sample)
    assert explanation["efficiency_axiom_verified"] is True
    assert len(explanation["all_feature_shap_values"]) == 16


def test_summary_json_structure(summary_json: dict) -> None:
    """Verify structure and contents of explainability_summary.json."""
    meta = summary_json.get("metadata", {})
    assert "champion_lightgbm_tuned.joblib" in meta["model_analyzed"]
    assert meta["global_sample_size"] >= 1000
    assert meta["shap_computation_time_seconds"] > 0.0

    rankings = summary_json.get("global_feature_attributions", [])
    assert len(rankings) == 16
    total_pct = sum(r["global_importance_pct"] for r in rankings)
    assert pytest.approx(total_pct, abs=0.5) == 100.0

    # Top features must include newbalanceOrig and orig_balance_discrepancy
    top_2_feats = [rankings[0]["feature"], rankings[1]["feature"]]
    assert "newbalanceOrig" in top_2_feats
    assert "orig_balance_discrepancy" in top_2_feats


def test_case_studies_integrity(summary_json: dict) -> None:
    """Verify case studies contain expected scenarios and verified efficiency."""
    cases = summary_json.get("case_studies", {})
    assert len(cases) == 4

    expected_titles = [
        "Case 1: High-Value Account Drain Fraud",
        "Case 2: Nocturnal Transfer Fraud (03:00 AM)",
        "Case 3: Benign High-Value Commercial Transaction",
        "Case 4: Borderline Triage Investigation Case",
    ]
    for title in expected_titles:
        assert title in cases, f"Missing case: {title}"
        case_data = cases[title]
        assert "transaction_id" in case_data
        assert "amount" in case_data
        assert "predicted_fraud_probability" in case_data
        assert case_data["efficiency_axiom_verified"] is True
        assert case_data["reconstruction_error"] < 1e-4


def test_figures_and_documentation_artifacts_exist() -> None:
    """Verify publication figures and Markdown documentation exist on disk."""
    fig12 = FIGURES_DIR / "fig_12_shap_global_summary.png"
    fig13 = FIGURES_DIR / "fig_13_shap_local_waterfalls.png"
    doc_path = DOCS_DIR / "model_explainability_report.md"

    assert fig12.exists(), "Missing Figure 12"
    assert fig13.exists(), "Missing Figure 13"
    assert fig12.stat().st_size > 10_000, "Figure 12 is too small or corrupt"
    assert fig13.stat().st_size > 10_000, "Figure 13 is too small or corrupt"

    assert doc_path.exists(), "Missing explainability report"
    content = doc_path.read_text(encoding="utf-8")
    assert "Axiomatic Foundations of TreeSHAP" in content
    assert "Global Feature Attribution Hierarchy" in content
    assert "Local Transaction Case Studies & Operational Triage" in content
    assert "Model Risk Governance & Regulatory Compliance (SR 11-7 / OCC 2011-12)" in content
    assert "Tier-1 Financial Institution Interview Q&A" in content
