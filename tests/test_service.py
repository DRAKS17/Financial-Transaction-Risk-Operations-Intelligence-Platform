"""Automated test suite for Phase 14 Real-Time FastAPI Risk Scoring & Triage Service.

Validates the FastAPI REST API endpoints (/health, /thresholds, /score, /explain, /batch),
Pydantic validation schemas, decision-support routing rules, and TreeSHAP latency.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from financial_risk.config import DATA_PROCESSED_DIR
from financial_risk.service import app, cache


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Fixture providing initialized FastAPI TestClient with pre-loaded ModelCache."""
    cache.load()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def summary_json() -> dict:
    """Fixture providing parsed api_service_summary.json."""
    summary_path = DATA_PROCESSED_DIR / "api_service_summary.json"
    assert summary_path.exists(), f"Missing API summary file: {summary_path}"
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def test_health_check_endpoint(client: TestClient) -> None:
    """Verify GET /v1/health returns operational status and model metadata."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert data["service_name"] == "FinancialRiskEngine"
    assert data["version"] == "1.0.0"
    assert "champion_lightgbm_tuned.joblib" in data["champion_model"]
    assert data["cost_optimal_threshold"] == 0.140
    assert data["priority_escalation_threshold"] == 0.850
    assert "database_host" in data


def test_thresholds_governance_endpoint(client: TestClient) -> None:
    """Verify GET /v1/risk/thresholds exposes active loss matrix boundaries."""
    response = client.get("/v1/risk/thresholds")
    assert response.status_code == 200
    data = response.json()
    assert data["cost_optimal_threshold_tau"] == 0.140
    assert data["priority_escalation_threshold"] == 0.850
    assert data["investigator_review_cost_usd"] == 25.0
    rules = data["decision_support_operating_rules"]
    assert any("No automated account freezes" in rule for rule in rules)


def test_score_endpoint_fraud_priority_escalation(client: TestClient) -> None:
    """Verify POST /v1/risk/score triages account draining fraud to PRIORITY_ESCALATION_QUEUE."""
    payload = {
        "transaction_id": "TX_FRAUD_001",
        "amount": 2_500_000.0,
        "type": "TRANSFER",
        "oldbalanceOrg": 2_500_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "hour_of_day": 3,
        "dest_inbound_prior_count": 0,
        "is_merchant_dest": False,
    }
    response = client.post("/v1/risk/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX_FRAUD_001"
    assert data["fraud_probability"] >= 0.850
    assert data["risk_score"] >= 850
    assert data["triage_decision"] == "PRIORITY_ESCALATION_QUEUE"
    assert "Senior Compliance" in data["operational_queue"]
    assert data["sla_target"] == "1 Hour (High-Risk SLA)"
    assert data["latency_ms"] > 0.0


def test_score_endpoint_legitimate_auto_approval(client: TestClient) -> None:
    """Verify POST /v1/risk/score triages benign merchant payment to AUTOMATED_APPROVAL."""
    payload = {
        "transaction_id": "TX_BENIGN_001",
        "amount": 85.50,
        "type": "PAYMENT",
        "oldbalanceOrg": 12_500.0,
        "newbalanceOrig": 12_414.50,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "hour_of_day": 14,
        "dest_inbound_prior_count": 5,
        "is_merchant_dest": True,
    }
    response = client.post("/v1/risk/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX_BENIGN_001"
    assert data["fraud_probability"] < 0.140
    assert data["risk_score"] < 140
    assert data["triage_decision"] == "AUTOMATED_APPROVAL"
    assert "Auto-Clear" in data["operational_queue"]
    assert "Sub-second" in data["sla_target"]


def test_score_endpoint_schema_validation_errors(client: TestClient) -> None:
    """Verify Pydantic rejects invalid amounts, unknown types, and missing fields."""
    # Negative amount error
    neg_payload = {
        "transaction_id": "ERR_01",
        "amount": -500.0,
        "type": "TRANSFER",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 500.0,
    }
    resp1 = client.post("/v1/risk/score", json=neg_payload)
    assert resp1.status_code == 422

    # Invalid payment type error
    bad_type_payload = {
        "transaction_id": "ERR_02",
        "amount": 500.0,
        "type": "CRYPTOCURRENCY_SWAP",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 500.0,
    }
    resp2 = client.post("/v1/risk/score", json=bad_type_payload)
    assert resp2.status_code == 422

    # Missing mandatory oldbalanceOrg field
    missing_payload = {
        "transaction_id": "ERR_03",
        "amount": 500.0,
        "type": "TRANSFER",
        "newbalanceOrig": 500.0,
    }
    resp3 = client.post("/v1/risk/score", json=missing_payload)
    assert resp3.status_code == 422


def test_explain_endpoint_treeshap_reason_codes(client: TestClient) -> None:
    """Verify POST /v1/risk/explain computes TreeSHAP waterfall and verified efficiency."""
    payload = {
        "transaction_id": "TX_EXPLAIN_001",
        "amount": 1_200_000.0,
        "type": "TRANSFER",
        "oldbalanceOrg": 1_200_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
        "hour_of_day": 2,
        "dest_inbound_prior_count": 0,
        "is_merchant_dest": False,
    }
    response = client.post("/v1/risk/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "TX_EXPLAIN_001"
    assert data["efficiency_axiom_verified"] is True
    assert data["reconstruction_error"] < 1e-4

    # Verify adverse reason codes exist
    adverse_codes = data["primary_adverse_reason_codes"]
    assert len(adverse_codes) > 0
    code_names = [c["reason_code"] for c in adverse_codes]
    assert any(
        "RC_LEDGER_DISCREPANCY" in c or "RC_COMPLETE_DRAIN" in c for c in code_names
    )

    # Verify all 16 features are present in attribution dictionary
    assert len(data["all_feature_shap_values"]) == 16


def test_batch_scoring_endpoint(client: TestClient) -> None:
    """Verify POST /v1/risk/batch vectors multiple transactions in parallel."""
    batch_payload = {
        "transactions": [
            {
                "transaction_id": "BATCH_01",
                "amount": 100.0,
                "type": "PAYMENT",
                "oldbalanceOrg": 1000.0,
                "newbalanceOrig": 900.0,
            },
            {
                "transaction_id": "BATCH_02",
                "amount": 5_000_000.0,
                "type": "TRANSFER",
                "oldbalanceOrg": 5_000_000.0,
                "newbalanceOrig": 0.0,
                "hour_of_day": 4,
            },
            {
                "transaction_id": "BATCH_03",
                "amount": 25.0,
                "type": "DEBIT",
                "oldbalanceOrg": 500.0,
                "newbalanceOrig": 475.0,
            },
        ]
    }
    response = client.post("/v1/risk/batch", json=batch_payload)
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 3
    assert results[0]["transaction_id"] == "BATCH_01"
    assert results[0]["triage_decision"] == "AUTOMATED_APPROVAL"
    assert results[1]["transaction_id"] == "BATCH_02"
    assert results[1]["triage_decision"] == "PRIORITY_ESCALATION_QUEUE"
    assert results[2]["transaction_id"] == "BATCH_03"


def test_benchmark_summary_and_documentation_artifacts(summary_json: dict) -> None:
    """Verify benchmark summary JSON and API integration guide exist."""
    scoring = summary_json["scoring_latency_ms"]
    assert scoring["median_p50"] < 5.0  # Must be sub-5ms
    assert scoring["throughput_req_per_sec"] > 500.0

    guide_path = Path("docs/api_service_guide.md")
    assert guide_path.exists(), "Missing docs/api_service_guide.md"
    content = guide_path.read_text(encoding="utf-8")
    assert "Real-Time FastAPI Risk Engine & Operational Triage Service Guide" in content
    assert "Mandatory Decision-Support Operating Rules" in content
    assert "Tier-1 Financial Institution Integration Architectures" in content
