"""Test Suite for Phase 11: Baseline Modeling and Cost-Sensitive Learning.

Validates temporal out-of-time (OOT) partitioning, financial cost function calculation,
threshold optimization mechanics, model artifact serialization, and test performance invariants.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pytest

from financial_risk.config import DATA_PROCESSED_DIR, DOCS_DIR, FIGURES_DIR
from financial_risk.modeling import FinancialRiskModelingPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Provides an instance of FinancialRiskModelingPipeline."""
    return FinancialRiskModelingPipeline()


def test_pipeline_initialization(pipeline):
    """Verify pipeline initialization, paths, and directory structures."""
    assert pipeline.features_path.exists()
    assert pipeline.models_dir.exists()
    assert pipeline.figures_dir.exists()
    assert len(pipeline.FEATURE_COLS) == 16
    assert pipeline.review_cost == 25.0


def test_financial_cost_calculation(pipeline):
    """Verify mathematical correctness of asymmetric financial loss evaluation."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 0])
    amounts = np.array([500_000.0, 1_000_000.0, 50.0, 100.0, 200_000.0])

    # TP = index 0 ($500k captured)
    # FN = index 1 ($1M missed) and index 4 ($200k missed) -> $1.2M missed
    # FP = index 2 (review cost)
    # TN = index 3
    cost_metrics = pipeline.compute_financial_cost(y_true, y_pred, amounts)

    assert cost_metrics["true_positives"] == 1
    assert cost_metrics["false_positives"] == 1
    assert cost_metrics["false_negatives"] == 2
    assert cost_metrics["true_negatives"] == 1
    assert cost_metrics["alert_volume"] == 2

    assert cost_metrics["dollar_loss_missed"] == 1_200_000.0
    assert cost_metrics["dollar_loss_captured"] == 500_000.0
    assert cost_metrics["investigator_review_cost"] == 2 * 25.0  # $50.00
    assert cost_metrics["total_financial_cost"] == 1_200_050.0
    assert cost_metrics["dollar_capture_rate_pct"] == round(
        500_000 / 1_700_000 * 100.0, 2
    )


def test_threshold_optimization_logic(pipeline):
    """Verify threshold optimization finds a minimum on a simulated cost curve."""
    np.random.seed(42)
    y_true = np.array([1] * 20 + [0] * 1000)
    # High probabilities for fraud, low for benign
    y_probs = np.concatenate(
        [np.random.uniform(0.3, 0.9, 20), np.random.uniform(0.01, 0.2, 1000)]
    )
    amounts = np.concatenate(
        [np.random.uniform(10_000, 500_000, 20), np.random.uniform(10, 1000, 1000)]
    )

    best_tau, min_cost, records = pipeline.optimize_decision_threshold(
        y_true, y_probs, amounts, n_points=50
    )

    assert 0.01 <= best_tau <= 0.99
    assert len(records) == 50
    assert min_cost > 0.0
    # Min cost must be <= cost at default 0.50
    cost_at_05 = next(
        r["total_cost"] for r in records if abs(r["threshold"] - 0.50) < 0.03
    )
    assert min_cost <= cost_at_05


def test_model_artifacts_exist(pipeline):
    """Verify all 4 trained model artifacts are serialized in models/."""
    expected_models = [
        "logistic_regression.joblib",
        "decision_tree.joblib",
        "random_forest.joblib",
        "lightgbm.joblib",
    ]
    for model_file in expected_models:
        model_path = pipeline.models_dir / model_file
        assert model_path.exists(), f"Missing model: {model_path}"
        assert model_path.stat().st_size > 1_000, (
            f"Model file {model_file} is too small"
        )

        # Verify model can be loaded
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, "predict_proba")


def test_decision_tree_rules_exported():
    """Verify human-readable decision tree rules exist and contain expected root splits."""
    rules_path = DOCS_DIR / "decision_tree_rules.txt"
    assert rules_path.exists(), f"Missing rules: {rules_path}"
    content = rules_path.read_text(encoding="utf-8")
    assert "orig_balance_discrepancy" in content
    assert "newbalanceOrig" in content
    assert "class: 1" in content
    assert "class: 0" in content


def test_modeling_figures_exist():
    """Verify all 3 Phase 11 figures exist and have valid file sizes (>50KB)."""
    expected_figures = [
        "fig_07_pr_roc_curves.png",
        "fig_08_cost_curve_thresholds.png",
        "fig_09_model_dollar_capture_comparison.png",
    ]
    for fig_name in expected_figures:
        fig_path = FIGURES_DIR / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_path}"
        assert fig_path.stat().st_size > 50_000, (
            f"Figure {fig_name} is too small (<50KB)"
        )


def test_baseline_modeling_summary_json():
    """Verify baseline_modeling_summary.json exists and contains complete model metrics."""
    summary_path = DATA_PROCESSED_DIR / "baseline_modeling_summary.json"
    assert summary_path.exists()
    content = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "metadata" in content
    assert content["metadata"]["test_size"] == 89688
    assert content["metadata"]["test_fraud_count"] == 1328
    assert content["metadata"]["test_fraud_exposure"] > 2.0e9  # > $2 Billion

    assert "models" in content
    models_dict = content["models"]
    assert "Heuristic Baseline" in models_dict
    assert "Logistic Regression" in models_dict
    assert "Decision Tree" in models_dict
    assert "Random Forest" in models_dict
    assert "LightGBM" in models_dict

    # Check Random Forest and Decision Tree performance invariants
    rf_data = models_dict["Random Forest"]
    dt_data = models_dict["Decision Tree"]

    assert rf_data["pr_auc"] >= 0.99
    assert dt_data["pr_auc"] >= 0.99

    assert rf_data["metrics_at_optimal_threshold"]["dollar_capture_rate_pct"] >= 99.0
    assert dt_data["metrics_at_optimal_threshold"]["dollar_capture_rate_pct"] >= 99.0

    # Heuristic baseline must have catastrophic failure (< 5% recall)
    heur_data = models_dict["Heuristic Baseline"]
    assert heur_data["metrics_at_optimal_threshold"]["recall"] < 0.05
    assert heur_data["metrics_at_optimal_threshold"]["dollar_capture_rate_pct"] < 5.0
