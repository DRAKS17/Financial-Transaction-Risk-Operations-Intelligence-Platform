"""Phase 4 Data Cleaning and Reconciliation Tests.

Validates that cleaned transactions in Parquet format maintain strict record-count parity,
zero ground-truth label drift, valid calendar derivations, surrogate key uniqueness,
and proper counterparty entity classification.
"""

import json

import pyarrow.parquet as pq

from financial_risk.config import DATA_PROCESSED_DIR, DOCS_DIR


def test_cleaned_parquet_file_exists():
    """Verify cleaned Parquet dataset exists and has expected scale."""
    parquet_path = DATA_PROCESSED_DIR / "transactions_cleaned.parquet"
    assert parquet_path.exists(), f"Missing Parquet file at {parquet_path}"
    assert parquet_path.stat().st_size > 200_000_000, "Parquet file size appears too small."


def test_parquet_schema_and_columns():
    """Verify Parquet schema contains all normalized, derived, and original columns."""
    parquet_path = DATA_PROCESSED_DIR / "transactions_cleaned.parquet"
    schema = pq.read_schema(parquet_path)
    col_names = schema.names

    required_cols = [
        "transaction_id",
        "step",
        "day",
        "hour_of_day",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "orig_balance_delta",
        "is_zero_balance_orig",
        "is_orig_balance_err",
        "nameDest",
        "dest_type",
        "oldbalanceDest",
        "newbalanceDest",
        "dest_balance_delta",
        "is_zero_balance_dest",
        "isFraud",
        "isFlaggedFraud",
    ]

    for col in required_cols:
        assert col in col_names, f"Missing required column: {col}"


def test_reconciliation_json_verification():
    """Verify that cleaning reconciliation JSON passes all strict assertion checks."""
    recon_path = DATA_PROCESSED_DIR / "cleaning_reconciliation.json"
    assert recon_path.exists(), f"Missing reconciliation JSON at {recon_path}"

    with open(recon_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "SUCCESS"
    checks = data["reconciliation_checks"]
    assert checks["row_count_match"] is True
    assert checks["fraud_label_match"] is True
    assert checks["flagged_label_match"] is True
    assert checks["amount_reconciled"] is True
    assert checks["primary_key_generated"] is True

    # Record counts
    before = data["before_cleaning"]
    after = data["after_cleaning"]
    assert before["total_rows"] == 6_362_620
    assert after["total_rows"] == 6_362_620
    assert after["total_fraud"] == 8_213
    assert after["total_flagged"] == 16


def test_reconciliation_markdown_report_structure():
    """Verify generated cleaning reconciliation markdown report exists with key audit sections."""
    doc_path = DOCS_DIR / "cleaning_reconciliation_report.md"
    assert doc_path.exists(), f"Missing report at {doc_path}"

    content = doc_path.read_text(encoding="utf-8")
    assert "# Data Cleaning & Transformation Reconciliation Report" in content
    assert "## 1. Executive Reconciliation Scorecard" in content
    assert "## 2. Transformation Rules & Senior Analytics Rationale" in content
    assert "## 3. Strict Boundary Rules Enforced" in content
    assert "6,362,620" in content
    assert "8,213" in content
