"""Phase 5 Text Processing and Operational Case Normalization Tests.

Validates that unstructured investigation notes are properly cleaned, whitespace-normalized,
sentinel-imputed for missingness, and augmented with quantitative text metrics, keyword flags,
and operational turnaround time (TAT) and SLA breach indicators.
"""

import json

import pyarrow.parquet as pq

from financial_risk.config import DATA_PROCESSED_DIR, DOCS_DIR
from financial_risk.text_processing import CaseTextProcessor


def test_text_processor_whitespace_normalization():
    """Verify regex whitespace normalization strips edges and collapses internal whitespace."""
    raw = "   discrepancy   in   sender balance   \n\t  noted   "
    clean = CaseTextProcessor.normalize_text_whitespace(raw)
    assert clean == "discrepancy in sender balance noted"
    assert CaseTextProcessor.normalize_text_whitespace("") == ""
    assert CaseTextProcessor.normalize_text_whitespace(None) == ""


def test_processed_cases_parquet_exists_and_schema():
    """Verify processed case Parquet exists with 5000 rows and required columns."""
    parquet_path = DATA_PROCESSED_DIR / "investigation_cases_cleaned.parquet"
    assert parquet_path.exists(), f"Missing Parquet file at {parquet_path}"

    table = pq.read_table(parquet_path)
    assert table.num_rows == 5000

    col_names = table.schema.names
    required_cols = [
        "case_id",
        "linked_transaction_id",
        "tx_step",
        "tx_type",
        "tx_amount",
        "priority",
        "assigned_team",
        "case_status",
        "is_case_closed",
        "turnaround_time_hours",
        "is_sla_breached",
        "has_notes",
        "raw_investigation_notes",
        "cleaned_investigation_notes",
        "note_char_length",
        "note_word_count",
        "flag_account_depletion",
        "flag_velocity_surge",
        "flag_compromise_suspected",
        "flag_balance_discrepancy",
        "flag_benign_cleared",
    ]

    for col in required_cols:
        assert col in col_names, f"Missing required column: {col}"


def test_sentinel_imputation_for_missing_notes():
    """Verify cases without notes receive the standard sentinel token and zero word count."""
    parquet_path = DATA_PROCESSED_DIR / "investigation_cases_cleaned.parquet"
    df = pq.read_table(parquet_path).to_pandas()

    missing_subset = df[~df["has_notes"]]
    assert len(missing_subset) > 0

    # Verify all missing notes were imputed with sentinel
    assert (
        missing_subset["cleaned_investigation_notes"] == "[NO NOTES RECORDED]"
    ).all()
    assert (missing_subset["note_word_count"] == 0).all()
    assert (missing_subset["note_char_length"] == 0).all()


def test_text_summary_json_and_sla_metrics():
    """Verify summary JSON contains valid SLA and keyword flag distributions."""
    summary_path = DATA_PROCESSED_DIR / "case_text_summary.json"
    assert summary_path.exists(), f"Summary JSON missing at {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_cases_processed"] == 5000
    op = data["operational_metrics"]
    assert op["closed_cases"] > 0
    assert op["open_cases"] > 0
    assert op["sla_breaches"] > 0
    assert 0 < op["sla_breach_rate_pct"] < 100

    tat = op["turnaround_time_hours"]
    assert tat["median_p50"] > 0
    assert tat["p90"] >= tat["median_p50"]


def test_text_report_governance_section():
    """Verify text report contains strict governance warning against data leakage."""
    doc_path = DOCS_DIR / "text_processing_report.md"
    assert doc_path.exists(), f"Report missing at {doc_path}"

    content = doc_path.read_text(encoding="utf-8")
    assert "# Operational Case & Unstructured Text Processing Report" in content
    assert "## 2. Issue Category Keyword Indicators" in content
    assert "## 3. Operational Investigation Metrics & SLA Compliance" in content
    assert "CRITICAL DATA SCIENCE GOVERNANCE RULE" in content
    assert "catastrophic lookahead leakage" in content
