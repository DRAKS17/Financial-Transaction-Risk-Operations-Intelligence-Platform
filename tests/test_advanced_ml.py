"""Automated test suite for Phase 12 Advanced Machine Learning & Hyperparameter Tuning.

Validates the Bayesian tuning engine, resampling benchmark outputs, champion model
serialization integrity, Stacking Meta-Learner inference, and visualization artifacts.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from financial_risk.config import (
    DATA_PROCESSED_DIR,
    DOCS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
)
from financial_risk.tuning import (
    AdvancedHyperparameterTuningEngine,
    FinancialStackingEnsemble,
)


@pytest.fixture(scope="module")
def engine() -> AdvancedHyperparameterTuningEngine:
    """Fixture providing initialized AdvancedHyperparameterTuningEngine."""
    return AdvancedHyperparameterTuningEngine()


@pytest.fixture(scope="module")
def summary_json() -> dict:
    """Fixture providing parsed advanced_ml_summary.json."""
    summary_path = DATA_PROCESSED_DIR / "advanced_ml_summary.json"
    assert summary_path.exists(), f"Missing summary file: {summary_path}"
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def test_engine_initialization(engine: AdvancedHyperparameterTuningEngine) -> None:
    """Verify tuning engine config, directories, and feature definitions."""
    assert len(engine.FEATURE_COLS) == 16
    assert "log_amount" in engine.FEATURE_COLS
    assert "orig_depletion_complete" in engine.FEATURE_COLS
    assert "orig_balance_discrepancy" in engine.FEATURE_COLS
    assert engine.review_cost == 25.0
    assert engine.models_dir.exists()
    assert engine.figures_dir.exists()


def test_summary_metadata_and_structure(summary_json: dict) -> None:
    """Verify metadata partition sizes and JSON structure."""
    meta = summary_json.get("metadata", {})
    assert meta["train_size"] == 6_064_036
    assert meta["val_size"] == 208_896
    assert meta["test_size"] == 89_688
    assert meta["test_fraud_count"] == 1328
    assert meta["test_fraud_exposure"] > 2.0e9  # > $2 Billion

    assert "resampling_benchmark" in summary_json
    assert "optuna_best_params" in summary_json
    assert "advanced_models" in summary_json
    assert "stacking_meta_weights" in summary_json


def test_resampling_benchmark_metrics(summary_json: dict) -> None:
    """Verify resampling benchmark results for all three strategies."""
    resamp = summary_json["resampling_benchmark"]
    for strategy in [
        "Cost-Sensitive Weighting",
        "Random Under-Sampling",
        "SMOTE Oversampling",
    ]:
        assert strategy in resamp, f"Missing strategy {strategy}"
        metrics = resamp[strategy]
        assert metrics["pr_auc"] >= 0.90
        assert metrics["roc_auc"] >= 0.95
        assert "brier_score" in metrics
        assert metrics["metrics_at_optimal_threshold"]["dollar_capture_rate_pct"] >= 95.0
        assert metrics["training_time_seconds"] > 0.0


def test_optuna_hyperparameters_validity(summary_json: dict) -> None:
    """Verify Optuna hyperparameter bounds and best score consistency."""
    best_params = summary_json["optuna_best_params"]
    assert "lightgbm" in best_params
    assert "xgboost" in best_params

    lgb_p = best_params["lightgbm"]
    assert 0.01 <= lgb_p["learning_rate"] <= 0.25
    assert 15 <= lgb_p["num_leaves"] <= 256
    assert 1.0 <= lgb_p["scale_pos_weight"] <= 100.0

    xgb_p = best_params["xgboost"]
    assert 0.01 <= xgb_p["learning_rate"] <= 0.25
    assert 3 <= xgb_p["max_depth"] <= 12
    assert 1.0 <= xgb_p["scale_pos_weight"] <= 100.0


def test_champion_lightgbm_model_loading_and_inference() -> None:
    """Verify tuned LightGBM model loads and executes inference correctly."""
    model_path = MODELS_DIR / "champion_lightgbm_tuned.joblib"
    assert model_path.exists(), f"Missing champion LightGBM model: {model_path}"
    model = joblib.load(model_path)

    # Synthetic inference check
    dummy_x = pd.DataFrame(
        np.random.randn(10, 16).astype(np.float32),
        columns=AdvancedHyperparameterTuningEngine.FEATURE_COLS,
    )
    probs = model.predict_proba(dummy_x)
    assert probs.shape == (10, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_champion_xgboost_model_loading_and_inference() -> None:
    """Verify tuned XGBoost model loads and executes inference correctly."""
    model_path = MODELS_DIR / "champion_xgboost_tuned.joblib"
    assert model_path.exists(), f"Missing champion XGBoost model: {model_path}"
    model = joblib.load(model_path)

    dummy_x = pd.DataFrame(
        np.random.randn(10, 16).astype(np.float32),
        columns=AdvancedHyperparameterTuningEngine.FEATURE_COLS,
    )
    probs = model.predict_proba(dummy_x)
    assert probs.shape == (10, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_stacking_ensemble_loading_and_inference() -> None:
    """Verify Stacking Meta-Learner composite pipeline loading and predictions."""
    model_path = MODELS_DIR / "stacking_ensemble.joblib"
    assert model_path.exists(), f"Missing Stacking Ensemble model: {model_path}"
    ensemble = joblib.load(model_path)
    assert isinstance(ensemble, FinancialStackingEnsemble)
    assert len(ensemble.base_models) == 5

    dummy_x = pd.DataFrame(
        np.random.randn(8, 16).astype(np.float32),
        columns=AdvancedHyperparameterTuningEngine.FEATURE_COLS,
    )
    probs = ensemble.predict_proba(dummy_x)
    assert probs.shape == (8, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    preds = ensemble.predict(dummy_x, threshold=0.50)
    assert preds.shape == (8,)
    assert set(preds).issubset({0, 1})


def test_stacking_ensemble_custom_fit_meta() -> None:
    """Verify FinancialStackingEnsemble meta-fitting on simulated validation data."""
    lr_dummy = LogisticRegression().fit(np.random.randn(20, 16), np.random.randint(0, 2, 20))
    base = {"LR1": lr_dummy, "LR2": lr_dummy}
    stacker = FinancialStackingEnsemble(base_models=base)

    val_x = np.random.randn(30, 16).astype(np.float32)
    val_y = np.random.randint(0, 2, 30)
    stacker.fit_meta(val_x, val_y)

    assert stacker.meta_classifier.coef_.shape == (1, 2)
    assert stacker.meta_classifier.intercept_.shape == (1,)


def test_figures_and_documentation_artifacts_exist() -> None:
    """Verify all publication figures and documentation files exist and are non-empty."""
    fig10 = FIGURES_DIR / "fig_10_hyperparameter_tuning_surface.png"
    fig11 = FIGURES_DIR / "fig_11_champion_model_comparison.png"
    doc_path = DOCS_DIR / "advanced_ml_tuning_report.md"

    assert fig10.exists(), "Missing Figure 10"
    assert fig11.exists(), "Missing Figure 11"
    assert fig10.stat().st_size > 10_000, "Figure 10 is too small or corrupt"
    assert fig11.stat().st_size > 10_000, "Figure 11 is too small or corrupt"

    assert doc_path.exists(), "Missing advanced ML tuning report"
    content = doc_path.read_text(encoding="utf-8")
    assert "Executive Summary & Core Results" in content
    assert "Resampling vs Cost-Sensitive Weighting Benchmark" in content
    assert "Heterogeneous Stacking Meta-Learner Architecture" in content
    assert "Model Governance & Regulatory Compliance (SR 11-7 / OCC 2011-12)" in content
    assert "Financial Institution Interview Preparation" in content
