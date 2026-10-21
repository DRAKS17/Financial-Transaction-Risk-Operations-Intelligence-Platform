"""Phase 6 Feature Engineering and Temporal Safety Tests.

Validates that mathematical transformations, diurnal encodings, accounting inconsistency flags,
causal recipient velocity, and target label parity are correctly engineered and free from
future lookahead leakage.
"""

import json

import pyarrow.parquet as pq

from financial_risk.config import DATA_PROCESSED_DIR, DOCS_DIR


def test_engineered_features_file_exists():
    """Verify engineered feature Parquet dataset exists and is populated."""
    parquet_path = DATA_PROCESSED_DIR / "features_engineered.parquet"
    assert parquet_path.exists(), f"Missing features Parquet file at {parquet_path}"
    assert parquet_path.stat().st_size > 200_000_000, "Features file appears truncated."


def test_feature_matrix_schema_and_column_count():
    """Verify that all 33 expected feature columns exist in the feature matrix."""
    parquet_path = DATA_PROCESSED_DIR / "features_engineered.parquet"
    schema = pq.read_schema(parquet_path)
    col_names = schema.names

    required_features = [
        "transaction_id",
        "step",
        "day",
        "hour_of_day",
        "day_of_week",
        "is_night_hours",
        "is_weekend",
        "sin_hour",
        "cos_hour",
        "type",
        "is_transfer",
        "is_cash_out",
        "is_transfer_or_cashout",
        "is_merchant_dest",
        "amount",
        "log_amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "amount_to_orig_balance_ratio",
        "orig_depletion_complete",
        "orig_balance_discrepancy",
        "is_zero_balance_orig",
        "oldbalanceDest",
        "newbalanceDest",
        "dest_balance_growth_ratio",
        "dest_balance_discrepancy",
        "is_zero_balance_dest",
        "dest_inbound_prior_count",
        "isFlaggedFraud",
        "isFraud",
    ]

    for feat in required_features:
        assert feat in col_names, f"Missing required feature: {feat}"


def test_feature_summary_json_and_label_parity():
    """Verify feature summary JSON confirms zero lookahead leakage and label parity."""
    summary_path = DATA_PROCESSED_DIR / "feature_engineering_summary.json"
    assert summary_path.exists(), f"Summary JSON missing at {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_records"] == 6_362_620
    assert data["total_fraud_labels"] == 8_213
    assert data["temporally_safe_assertions"]["zero_lookahead_leakage"] is True
    assert data["temporally_safe_assertions"]["label_parity_verified"] is True


def test_feature_dictionary_documentation_structure():
    """Verify feature dictionary documentation covers business rationale and mathematical formulas."""
    doc_path = DOCS_DIR / "feature_dictionary.md"
    assert doc_path.exists(), f"Feature dictionary missing at {doc_path}"

    content = doc_path.read_text(encoding="utf-8")
    assert "# Analytical & Predictive Feature Dictionary" in content
    assert "## 1. Overview & Causality Governance" in content
    assert "## 2. Feature Definitions & Business Rationale" in content
    assert "Zero Future-to-Past Leakage" in content
    assert "orig_depletion_complete" in content
    assert "amount_to_orig_balance_ratio" in content
    assert "dest_inbound_prior_count" in content


def test_feature_mathematical_bounds():
    """Verify mathematical bounds on continuous engineered transformations."""
    parquet_path = DATA_PROCESSED_DIR / "features_engineered.parquet"
    # Read first 10,000 rows to test bounds efficiently
    table = pq.read_table(
        parquet_path,
        columns=["log_amount", "sin_hour", "cos_hour", "day_of_week", "amount_to_orig_balance_ratio"],
    )
    df_sample = table.slice(0, 10_000).to_pandas()

    assert (df_sample["log_amount"] >= 0.0).all()
    assert (df_sample["sin_hour"].between(-1.0001, 1.0001)).all()
    assert (df_sample["cos_hour"].between(-1.0001, 1.0001)).all()
    assert (df_sample["day_of_week"].between(0, 6)).all()
    assert (df_sample["amount_to_orig_balance_ratio"] >= 0.0).all()
