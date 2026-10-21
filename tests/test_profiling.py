"""Phase 3 Data Quality Profiling Tests.

Validates that data profiling metrics, statistical distributions, accounting integrity,
and fraud label concentrations are accurately computed and recorded.
"""

import json

from financial_risk.config import DATA_PROCESSED_DIR, DOCS_DIR
from financial_risk.profiling import DataQualityProfiler


def test_profiler_instantiation():
    """Verify DataQualityProfiler initializes with expected default configuration."""
    profiler = DataQualityProfiler()
    assert profiler.paysim_path.exists()
    assert profiler.cases_path.exists()


def test_summary_json_metrics():
    """Verify that the profiling summary JSON contains complete, accurate metrics."""
    summary_path = DATA_PROCESSED_DIR / "data_quality_summary.json"
    assert summary_path.exists(), f"Summary JSON missing at {summary_path}"

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "transactions" in data
    assert "investigations" in data

    tx = data["transactions"]
    assert tx["total_rows"] == 6_362_620
    assert tx["total_fraud"] == 8_213
    assert tx["total_flagged"] == 16
    assert 0.12 < tx["fraud_rate_pct"] < 0.14

    # Verify fraud occurs strictly in TRANSFER and CASH_OUT
    fraud_by_type = tx["fraud_by_type"]
    assert fraud_by_type.get("TRANSFER", 0) > 0
    assert fraud_by_type.get("CASH_OUT", 0) > 0
    assert fraud_by_type.get("PAYMENT", 0) == 0
    assert fraud_by_type.get("CASH_IN", 0) == 0
    assert fraud_by_type.get("DEBIT", 0) == 0

    # Verify amounts
    amt = tx["amount_distribution"]
    assert amt["min"] == 0.0
    assert amt["max"] > 90_000_000
    assert amt["negative_count"] == 0

    # Verify cases
    cases = data["investigations"]
    assert cases["total_cases"] == 5_000
    assert cases["referential_integrity_valid"] is True
    assert cases["text_quality_defects"]["missing_notes_count"] > 0


def test_markdown_report_structure():
    """Verify generated data quality report contains critical audit sections."""
    report_path = DOCS_DIR / "data_quality_report.md"
    assert report_path.exists(), f"Markdown report missing at {report_path}"

    content = report_path.read_text(encoding="utf-8")
    assert "# Data Quality Profiling & Audit Report" in content
    assert "## 1. Executive Summary & Quality Scorecard" in content
    assert "## 2. Transaction Stream Quality Profiling" in content
    assert "## 3. Operational Case Quality Profiling" in content
    assert "## 4. Key Takeaways & Recommendations" in content
    assert "6,362,620" in content
    assert "8,213" in content
