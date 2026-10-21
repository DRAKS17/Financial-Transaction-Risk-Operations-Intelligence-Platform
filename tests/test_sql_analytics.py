"""Test Suite for Phase 9: Advanced SQL Risk Analytics and Typologies.

Validates that complex analytical SQL queries execute correctly against PostgreSQL,
return sound financial metrics, adhere to window function properties,
and reconcile with portfolio-level accounting facts.
"""

from __future__ import annotations

import psycopg2
import pytest

from financial_risk.analytics import QUERIES, RiskAnalyticsEngine
from financial_risk.config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    SQL_DIR,
)


@pytest.fixture(scope="module")
def db_conn():
    """Provides a PostgreSQL connection for integration testing."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def analytics_engine():
    """Provides an instance of the RiskAnalyticsEngine."""
    return RiskAnalyticsEngine()


def test_sql_file_exists_and_content():
    """Verify sql/05_risk_analysis.sql exists and contains expected queries."""
    sql_path = SQL_DIR / "05_risk_analysis.sql"
    assert sql_path.exists(), f"Missing {sql_path}"
    content = sql_path.read_text(encoding="utf-8")
    assert "QUERY 1: Temporal Dynamics" in content
    assert "QUERY 2: Originator Rapid Structuring" in content
    assert "QUERY 3: Recipient Fan-In Network" in content
    assert "QUERY 4: Channel Vulnerability & Pareto Concentration" in content
    assert "QUERY 5: Operational Investigation Squad" in content
    assert "QUERY 6: Legacy Rule Failure" in content
    assert "QUERY 7: Account Draining" in content
    assert "QUERY 8: Diurnal / Hourly Fraud Vulnerability" in content


def test_all_queries_registered(analytics_engine):
    """Verify all 8 queries are registered in the analytics engine."""
    expected_keys = {
        "temporal_rolling_window",
        "origin_structuring_sequence",
        "recipient_fanin_mule_hubs",
        "pareto_decile_concentration",
        "investigation_operations_triage",
        "legacy_rule_failure",
        "account_draining_patterns",
        "diurnal_hourly_risk",
    }
    assert set(QUERIES.keys()) == expected_keys


def test_query_temporal_rolling_window(analytics_engine):
    """Verify Query 1 returns hourly rolling windows and cumulative sums."""
    res = analytics_engine.execute_query("temporal_rolling_window")
    rows = res["data"]
    assert len(rows) >= 740, f"Expected ~744 hourly steps, got {len(rows)}"
    first_row = rows[0]
    last_row = rows[-1]

    assert first_row["step"] == 1
    assert "rolling_24h_fraud_exposure" in first_row
    assert "cumulative_gtv" in last_row

    # Cumulative GTV at the final step must exceed 1 Trillion
    assert float(last_row["cumulative_gtv"]) > 1.0e12
    # Cumulative fraud exposure at final step must equal ~12.05 Billion
    assert float(last_row["cumulative_fraud_exposure"]) > 1.0e10
    # Rolling 24h count must be >= current hour count
    for r in rows[:50]:
        assert int(r["rolling_24h_fraud_count"]) >= int(r["fraud_count"])


def test_query_origin_structuring_sequence(analytics_engine):
    """Verify Query 2 detects sequential origin multi-transaction execution."""
    res = analytics_engine.execute_query("origin_structuring_sequence")
    rows = res["data"]
    assert len(rows) > 0, "Expected repeat origin transactions"
    for r in rows:
        assert r["prev_step"] is not None
        assert int(r["step_gap_hours"]) >= 0
        assert r["execution_latency_tier"] in {
            "Immediate Concurrent Execution (0h)",
            "Rapid Subsequent Execution (1h)",
            "Short Window Structuring (2-6h)",
            "Extended Gap (>6h)",
        }


def test_query_recipient_fanin_mule_hubs(analytics_engine):
    """Verify Query 3 identifies high fan-in recipient collection accounts."""
    res = analytics_engine.execute_query("recipient_fanin_mule_hubs")
    rows = res["data"]
    assert len(rows) > 0
    top_hub = rows[0]
    assert int(top_hub["inbound_tx_count"]) >= 10
    assert float(top_hub["total_inbound_gtv"]) > 1.0e6
    assert int(top_hub["fanin_velocity_rank"]) == 1


def test_query_pareto_decile_concentration(analytics_engine):
    """Verify Query 4 Pareto decile distribution and 90%+ fraud loss concentration in Decile 10."""
    res = analytics_engine.execute_query("pareto_decile_concentration")
    rows = res["data"]
    assert len(rows) == 10, f"Expected 10 deciles, got {len(rows)}"

    # Decile 10 checks
    decile_10 = rows[-1]
    assert decile_10["amount_decile"] == 10
    # Decile 10 must capture >= 90% of total fraud loss
    assert float(decile_10["share_of_total_fraud_loss_pct"]) >= 90.0
    # Final cumulative fraud loss must equal 100.0%
    assert float(decile_10["cum_fraud_loss_pct"]) == 100.0


def test_query_investigation_operations_triage(analytics_engine):
    """Verify Query 5 investigation triage matrix, SLA breaches, and percentiles."""
    res = analytics_engine.execute_query("investigation_operations_triage")
    rows = res["data"]
    assert len(rows) > 0

    total_allocated = sum(int(r["total_allocated_cases"]) for r in rows)
    assert total_allocated == 5000

    # Ensure median TAT <= P90 TAT for all squads
    for r in rows:
        assert float(r["p50_median_closed_tat_hours"]) <= float(
            r["p90_closed_tat_hours"]
        )
        assert float(r["squad_sla_breach_rate_pct"]) >= 0.0


def test_query_legacy_rule_failure(analytics_engine):
    """Verify Query 6 audits legacy is_flagged_fraud failure."""
    res = analytics_engine.execute_query("legacy_rule_failure")
    rows = res["data"]
    assert len(rows) > 0

    tp_rows = [r for r in rows if r["outcome_class"] == "True Positive (TP)"]
    fn_rows = [r for r in rows if "False Negative" in r["outcome_class"]]

    total_tp = sum(int(r["tx_count"]) for r in tp_rows)
    total_fn = sum(int(r["tx_count"]) for r in fn_rows)

    assert total_tp == 16
    assert total_fn == 8197

    # TP is exclusively on TRANSFER
    assert tp_rows[0]["transaction_type"] == "TRANSFER"

    # FN exists across both CASH_OUT (4,116) and TRANSFER (4,081)
    fn_by_type = {r["transaction_type"]: int(r["tx_count"]) for r in fn_rows}
    assert fn_by_type["CASH_OUT"] == 4116
    assert fn_by_type["TRANSFER"] == 4081

    # Recall of legacy rule is under 0.25%
    recall_pct = (total_tp / (total_tp + total_fn)) * 100
    assert recall_pct < 0.25

    # Total missed fraud loss is > 11 Billion
    total_fn_exposure = sum(float(r["total_exposure_amount"]) for r in fn_rows)
    assert total_fn_exposure > 1.1e10


def test_query_account_draining_patterns(analytics_engine):
    """Verify Query 7 accounts draining patterns capture > 95% of fraud for TRANSFER/CASH_OUT."""
    res = analytics_engine.execute_query("account_draining_patterns")
    rows = res["data"]
    assert len(rows) > 0

    cash_out_drains = [
        r
        for r in rows
        if r["transaction_type"] == "CASH_OUT"
        and "Complete Draining" in r["depletion_state"]
    ]
    transfer_drains = [
        r
        for r in rows
        if r["transaction_type"] == "TRANSFER"
        and "Complete Draining" in r["depletion_state"]
    ]

    assert len(cash_out_drains) == 1
    assert len(transfer_drains) == 1
    # Cash out draining captures > 95% of fraud count
    assert float(cash_out_drains[0]["share_of_channel_fraud_count_pct"]) > 95.0
    # Transfer draining captures > 70% of fraud count
    assert float(transfer_drains[0]["share_of_channel_fraud_count_pct"]) > 70.0


def test_query_diurnal_hourly_risk(analytics_engine):
    """Verify Query 8 captures 24 diurnal hours and nocturnal risk surge."""
    res = analytics_engine.execute_query("diurnal_hourly_risk")
    rows = res["data"]
    assert len(rows) == 24, f"Expected 24 hours, got {len(rows)}"

    # Hours 2, 3, 4 should have top risk prevalence ranks
    night_peak_hours = [r for r in rows if int(r["hour_of_day"]) in (2, 3, 4)]
    for r in night_peak_hours:
        assert int(r["risk_prevalence_rank"]) <= 5
        assert float(r["hourly_fraud_prevalence_pct"]) > 10.0


def test_explain_query(analytics_engine):
    """Verify EXPLAIN ANALYZE execution plan generation."""
    plan = analytics_engine.explain_query("diurnal_hourly_risk")
    assert "Parallel Seq Scan" in plan or "Seq Scan" in plan
    assert "Execution Time:" in plan
