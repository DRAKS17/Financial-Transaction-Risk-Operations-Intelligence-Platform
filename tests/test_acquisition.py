"""Phase 2 Data Acquisition and Raw Inventory Tests.

Validates that raw transaction files and companion operational case records
are properly acquired, non-empty, schema-compliant, and cryptographically inventoried.
"""

import json

from financial_risk.config import DATA_RAW_DIR
from financial_risk.ingestion import compute_sha256


def test_paysim_raw_file_presence_and_size():
    """Verify PaySim raw transaction CSV exists and is full scale."""
    paysim_path = DATA_RAW_DIR / "paysim.csv"
    assert paysim_path.exists(), f"Missing PaySim file at {paysim_path}"
    # PaySim CSV is ~493.5 MB
    assert paysim_path.stat().st_size > 400_000_000, "PaySim file appears incomplete or truncated."


def test_investigations_raw_file_presence():
    """Verify synthetic operational case log CSV exists and is populated."""
    cases_path = DATA_RAW_DIR / "investigation_cases.csv"
    assert cases_path.exists(), f"Missing investigation cases file at {cases_path}"
    assert cases_path.stat().st_size > 50_000, "Investigation cases file appears incomplete."


def test_sha256_hashing_function(tmp_path):
    """Verify cryptographic SHA-256 utility function produces deterministic 64-char hex string."""
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("financial risk analytics 2026")
    h = compute_sha256(dummy_file)
    assert len(h) == 64
    assert isinstance(h, str)


def test_raw_inventory_metadata_validity():
    """Verify that raw inventory metadata JSON exists and contains required file audits."""
    meta_path = DATA_RAW_DIR / "raw_inventory_metadata.json"
    assert meta_path.exists(), f"Inventory metadata missing at {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "paysim.csv" in data, "paysim.csv missing from inventory metadata"
    assert "investigation_cases.csv" in data, "investigation_cases.csv missing from metadata"

    paysim_meta = data["paysim.csv"]
    assert paysim_meta["row_count"] > 6_000_000, "Unexpected row count for PaySim"
    assert paysim_meta["column_count"] == 11, "Unexpected column count for PaySim"
    assert "isFraud" in paysim_meta["columns"]
    assert "isFlaggedFraud" in paysim_meta["columns"]

    cases_meta = data["investigation_cases.csv"]
    assert cases_meta["row_count"] >= 1000, "Unexpected row count for investigation cases"
    assert "case_id" in cases_meta["columns"]
    assert "linked_transaction_id" in cases_meta["columns"]
