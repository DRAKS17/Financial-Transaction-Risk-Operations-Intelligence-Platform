"""Advanced SQL Analytics Engine and Risk Typology Execution Framework.

Executes industry-grade analytical SQL queries directly against PostgreSQL star schema tables,
benchmarks execution performance, retrieves query plans (EXPLAIN ANALYZE),
and serializes executive risk summaries and operations metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

from financial_risk.config import (
    DATA_PROCESSED_DIR,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    DOCS_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def decimal_serializer(obj: Any) -> Any:
    """Helper serializer for json.dumps to handle Decimals, dates, and numpy types."""
    if isinstance(obj, Decimal):
        return float(obj)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# Individual SQL Query Statements extracted for targeted execution
QUERY_TEMPORAL_ROLLING_WINDOW = """
WITH hourly_rollup AS (
    SELECT
        t.step,
        d.day_number,
        d.hour_of_day,
        d.day_name,
        d.is_weekend,
        d.time_window_label,
        COUNT(t.transaction_id) AS tx_count,
        SUM(t.amount) AS gtv,
        SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
        SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0.0 END) AS fraud_exposure_amount
    FROM core.fact_transactions t
    JOIN core.dim_date d ON t.step = d.step
    GROUP BY t.step, d.day_number, d.hour_of_day, d.day_name, d.is_weekend, d.time_window_label
),
rolling_windows AS (
    SELECT
        step,
        day_number,
        hour_of_day,
        day_name,
        is_weekend,
        time_window_label,
        tx_count,
        ROUND(gtv, 2) AS gtv,
        fraud_count,
        ROUND(fraud_exposure_amount, 2) AS fraud_exposure_amount,
        ROUND(100.0 * fraud_count / NULLIF(tx_count, 0), 4) AS hourly_fraud_rate_pct,
        SUM(fraud_count) OVER (
            ORDER BY step 
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ) AS rolling_24h_fraud_count,
        ROUND(SUM(fraud_exposure_amount) OVER (
            ORDER BY step 
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ), 2) AS rolling_24h_fraud_exposure,
        ROUND(AVG(gtv) OVER (
            ORDER BY step 
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ), 2) AS rolling_24h_mean_gtv,
        ROUND(SUM(gtv) OVER (
            ORDER BY step 
            ROWS UNBOUNDED PRECEDING
        ), 2) AS cumulative_gtv,
        ROUND(SUM(fraud_exposure_amount) OVER (
            ORDER BY step 
            ROWS UNBOUNDED PRECEDING
        ), 2) AS cumulative_fraud_exposure
    FROM hourly_rollup
)
SELECT
    *,
    ROUND(
        (PERCENT_RANK() OVER (ORDER BY rolling_24h_fraud_exposure) * 100)::NUMERIC, 
        2
    ) AS rolling_exposure_stress_percentile
FROM rolling_windows
ORDER BY step ASC;
"""

QUERY_ORIGIN_STRUCTURING_SEQUENCE = """
WITH multi_tx_origs AS (
    SELECT orig_account_id
    FROM core.fact_transactions
    GROUP BY orig_account_id
    HAVING COUNT(*) > 1
),
orig_sequence AS (
    SELECT
        t.transaction_id,
        t.orig_account_id,
        t.dest_account_id,
        t.step,
        t.transaction_type,
        t.amount,
        t.oldbalance_org,
        t.newbalance_orig,
        t.is_fraud,
        LAG(t.step, 1) OVER (
            PARTITION BY t.orig_account_id 
            ORDER BY t.step, t.transaction_id
        ) AS prev_step,
        LAG(t.transaction_type, 1) OVER (
            PARTITION BY t.orig_account_id 
            ORDER BY t.step, t.transaction_id
        ) AS prev_transaction_type,
        LAG(t.amount, 1) OVER (
            PARTITION BY t.orig_account_id 
            ORDER BY t.step, t.transaction_id
        ) AS prev_amount,
        LAG(t.newbalance_orig, 1) OVER (
            PARTITION BY t.orig_account_id 
            ORDER BY t.step, t.transaction_id
        ) AS prev_end_balance
    FROM core.fact_transactions t
    JOIN multi_tx_origs m ON t.orig_account_id = m.orig_account_id
)
SELECT
    orig_account_id,
    transaction_id,
    step,
    prev_step,
    (step - prev_step) AS step_gap_hours,
    prev_transaction_type,
    transaction_type AS curr_transaction_type,
    ROUND(prev_amount, 2) AS prev_amount,
    ROUND(amount, 2) AS curr_amount,
    ROUND(prev_end_balance, 2) AS prev_end_balance,
    ROUND(oldbalance_org, 2) AS curr_start_balance,
    is_fraud,
    CASE
        WHEN (step - prev_step) = 0 THEN 'Immediate Concurrent Execution (0h)'
        WHEN (step - prev_step) = 1 THEN 'Rapid Subsequent Execution (1h)'
        WHEN (step - prev_step) <= 6 THEN 'Short Window Structuring (2-6h)'
        ELSE 'Extended Gap (>6h)'
    END AS execution_latency_tier
FROM orig_sequence
WHERE prev_step IS NOT NULL
ORDER BY (step - prev_step) ASC, curr_amount DESC
LIMIT 50;
"""

QUERY_RECIPIENT_FANIN_MULE_HUBS = """
WITH inbound_hubs AS (
    SELECT
        t.dest_account_id,
        t.dest_type,
        COUNT(t.transaction_id) AS inbound_tx_count,
        SUM(t.amount) AS total_inbound_gtv,
        AVG(t.amount) AS mean_inbound_ticket_size,
        MIN(t.amount) AS min_inbound_amount,
        MAX(t.amount) AS max_inbound_amount,
        SUM(t.is_fraud) AS fraud_tx_count,
        SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0.0 END) AS inbound_fraud_exposure
    FROM core.fact_transactions t
    WHERE t.transaction_type IN ('TRANSFER', 'CASH_OUT')
    GROUP BY t.dest_account_id, t.dest_type
    HAVING COUNT(t.transaction_id) > 1
)
SELECT
    dest_account_id,
    dest_type,
    inbound_tx_count,
    ROUND(total_inbound_gtv, 2) AS total_inbound_gtv,
    ROUND(mean_inbound_ticket_size, 2) AS mean_inbound_ticket_size,
    ROUND(min_inbound_amount, 2) AS min_inbound_amount,
    ROUND(max_inbound_amount, 2) AS max_inbound_amount,
    fraud_tx_count,
    ROUND(inbound_fraud_exposure, 2) AS inbound_fraud_exposure,
    ROUND(100.0 * fraud_tx_count / NULLIF(inbound_tx_count, 0), 2) AS hub_fraud_rate_pct,
    DENSE_RANK() OVER (ORDER BY inbound_tx_count DESC, total_inbound_gtv DESC) AS fanin_velocity_rank,
    DENSE_RANK() OVER (ORDER BY total_inbound_gtv DESC) AS total_exposure_rank
FROM inbound_hubs
ORDER BY inbound_tx_count DESC, total_inbound_gtv DESC
LIMIT 25;
"""

QUERY_PARETO_DECILE_CONCENTRATION = """
WITH transaction_deciles AS (
    SELECT
        transaction_id,
        amount,
        is_fraud,
        NTILE(10) OVER (ORDER BY amount ASC) AS amount_decile
    FROM core.fact_transactions
),
decile_summary AS (
    SELECT
        amount_decile,
        COUNT(*) AS tx_count,
        MIN(amount) AS min_amount,
        MAX(amount) AS max_amount,
        AVG(amount) AS avg_amount,
        SUM(amount) AS decile_gtv,
        SUM(is_fraud) AS decile_fraud_count,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0.0 END) AS decile_fraud_exposure
    FROM transaction_deciles
    GROUP BY amount_decile
)
SELECT
    amount_decile,
    tx_count,
    ROUND(min_amount, 2) AS decile_min_amount,
    ROUND(max_amount, 2) AS decile_max_amount,
    ROUND(avg_amount, 2) AS decile_mean_amount,
    ROUND(decile_gtv, 2) AS decile_gtv,
    decile_fraud_count,
    ROUND(decile_fraud_exposure, 2) AS decile_fraud_exposure,
    ROUND(100.0 * decile_fraud_count / NULLIF(tx_count, 0), 4) AS decile_fraud_rate_pct,
    ROUND(100.0 * decile_gtv / SUM(decile_gtv) OVER (), 2) AS share_of_total_gtv_pct,
    ROUND(100.0 * decile_fraud_exposure / SUM(decile_fraud_exposure) OVER (), 2) AS share_of_total_fraud_loss_pct,
    ROUND(100.0 * SUM(decile_gtv) OVER (ORDER BY amount_decile) / SUM(decile_gtv) OVER (), 2) AS cum_gtv_pct,
    ROUND(100.0 * SUM(decile_fraud_exposure) OVER (ORDER BY amount_decile) / SUM(decile_fraud_exposure) OVER (), 2) AS cum_fraud_loss_pct
FROM decile_summary
ORDER BY amount_decile ASC;
"""

QUERY_INVESTIGATION_OPERATIONS_TRIAGE = """
WITH operational_snapshot AS (
    SELECT MAX(created_at) AS max_snapshot_time FROM core.fact_investigations
),
case_aging AS (
    SELECT
        inv.case_id,
        inv.assigned_team,
        inv.priority,
        p.sla_target_hours,
        inv.case_status,
        inv.is_case_closed,
        inv.is_sla_breached,
        inv.turnaround_time_hours,
        CASE
            WHEN inv.is_case_closed = FALSE THEN
                ROUND((EXTRACT(EPOCH FROM (snap.max_snapshot_time - inv.created_at)) / 3600.0)::NUMERIC, 2)
            ELSE inv.turnaround_time_hours
        END AS effective_age_hours
    FROM core.fact_investigations inv
    JOIN core.dim_priority p ON inv.priority = p.priority_code
    CROSS JOIN operational_snapshot snap
),
squad_rollup AS (
    SELECT
        assigned_team,
        priority,
        sla_target_hours,
        COUNT(*) AS total_allocated_cases,
        SUM(CASE WHEN is_case_closed = FALSE THEN 1 ELSE 0 END) AS active_backlog_count,
        SUM(CASE WHEN is_case_closed = TRUE THEN 1 ELSE 0 END) AS closed_cases_count,
        SUM(CASE WHEN is_sla_breached = TRUE THEN 1 ELSE 0 END) AS total_sla_breaches,
        ROUND(AVG(CASE WHEN is_case_closed = TRUE THEN turnaround_time_hours ELSE NULL END), 2) AS mean_closed_tat_hours,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY CASE WHEN is_case_closed = TRUE THEN turnaround_time_hours ELSE NULL END
        )::NUMERIC, 2) AS p50_median_closed_tat_hours,
        ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (
            ORDER BY CASE WHEN is_case_closed = TRUE THEN turnaround_time_hours ELSE NULL END
        )::NUMERIC, 2) AS p90_closed_tat_hours,
        SUM(CASE WHEN is_case_closed = FALSE AND effective_age_hours < 24 THEN 1 ELSE 0 END) AS open_backlog_under_24h,
        SUM(CASE WHEN is_case_closed = FALSE AND effective_age_hours >= 24 AND effective_age_hours < 48 THEN 1 ELSE 0 END) AS open_backlog_24_to_48h,
        SUM(CASE WHEN is_case_closed = FALSE AND effective_age_hours >= 48 AND effective_age_hours < 72 THEN 1 ELSE 0 END) AS open_backlog_48_to_72h,
        SUM(CASE WHEN is_case_closed = FALSE AND effective_age_hours >= 72 THEN 1 ELSE 0 END) AS open_backlog_over_72h
    FROM case_aging
    GROUP BY assigned_team, priority, sla_target_hours
)
SELECT
    assigned_team,
    priority,
    sla_target_hours,
    total_allocated_cases,
    active_backlog_count,
    closed_cases_count,
    total_sla_breaches,
    ROUND(100.0 * total_sla_breaches / NULLIF(closed_cases_count, 0), 2) AS squad_sla_breach_rate_pct,
    mean_closed_tat_hours,
    p50_median_closed_tat_hours,
    p90_closed_tat_hours,
    open_backlog_under_24h,
    open_backlog_24_to_48h,
    open_backlog_48_to_72h,
    open_backlog_over_72h,
    RANK() OVER (
        PARTITION BY priority 
        ORDER BY mean_closed_tat_hours DESC
    ) AS tat_rank_within_priority
FROM squad_rollup
ORDER BY priority ASC, squad_sla_breach_rate_pct DESC;
"""

QUERY_LEGACY_RULE_FAILURE = """
WITH confusion_quadrants AS (
    SELECT
        t.transaction_type,
        CASE
            WHEN t.is_fraud = 1 AND t.is_flagged_fraud = 1 THEN 'True Positive (TP)'
            WHEN t.is_fraud = 0 AND t.is_flagged_fraud = 1 THEN 'False Positive (FP)'
            WHEN t.is_fraud = 1 AND t.is_flagged_fraud = 0 THEN 'False Negative (FN) - Missed Fraud'
            ELSE 'True Negative (TN) - Benign Pass'
        END AS outcome_class,
        COUNT(*) AS tx_count,
        SUM(t.amount) AS total_amount,
        AVG(t.amount) AS mean_ticket_size
    FROM core.fact_transactions t
    GROUP BY t.transaction_type, 2
),
portfolio_totals AS (
    SELECT
        SUM(CASE WHEN is_fraud = 1 THEN 1 ELSE 0 END) AS total_ground_truth_frauds,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0.0 END) AS total_ground_truth_loss,
        SUM(CASE WHEN is_flagged_fraud = 1 THEN 1 ELSE 0 END) AS total_flagged_count,
        SUM(CASE WHEN is_flagged_fraud = 1 THEN amount ELSE 0.0 END) AS total_flagged_exposure
    FROM core.fact_transactions
)
SELECT
    c.transaction_type,
    c.outcome_class,
    c.tx_count,
    ROUND(c.total_amount, 2) AS total_exposure_amount,
    ROUND(c.mean_ticket_size, 2) AS mean_ticket_size,
    ROUND(100.0 * c.tx_count / p.total_ground_truth_frauds, 4) AS pct_of_all_fraud_count,
    ROUND(100.0 * c.total_amount / p.total_ground_truth_loss, 4) AS pct_of_all_fraud_exposure
FROM confusion_quadrants c
CROSS JOIN portfolio_totals p
WHERE c.outcome_class IN ('True Positive (TP)', 'False Negative (FN) - Missed Fraud', 'False Positive (FP)')
ORDER BY c.transaction_type, c.outcome_class DESC;
"""

QUERY_ACCOUNT_DRAINING_PATTERNS = """
WITH balance_states AS (
    SELECT
        transaction_type,
        CASE
            WHEN oldbalance_org > 0 AND newbalance_orig = 0 THEN '1. Complete Draining (Zero Balance Left)'
            WHEN oldbalance_org > 0 AND newbalance_orig > 0 AND newbalance_orig < oldbalance_org THEN '2. Partial Depletion'
            WHEN oldbalance_org = 0 AND amount > 0 THEN '3. Zero Initial Balance Initiation'
            WHEN is_orig_balance_err = TRUE THEN '4. Accounting Ledger Discrepancy'
            ELSE '5. Balance Inflow / Other'
        END AS depletion_state,
        COUNT(*) AS tx_count,
        SUM(amount) AS total_gtv,
        SUM(is_fraud) AS fraud_count,
        SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0.0 END) AS fraud_exposure_amount
    FROM core.fact_transactions
    WHERE transaction_type IN ('TRANSFER', 'CASH_OUT')
    GROUP BY transaction_type, 2
)
SELECT
    transaction_type,
    depletion_state,
    tx_count,
    ROUND(total_gtv, 2) AS total_gtv,
    fraud_count,
    ROUND(fraud_exposure_amount, 2) AS fraud_exposure_amount,
    ROUND(100.0 * fraud_count / NULLIF(tx_count, 0), 4) AS state_fraud_rate_pct,
    ROUND(
        100.0 * fraud_count / SUM(fraud_count) OVER (PARTITION BY transaction_type), 
        2
    ) AS share_of_channel_fraud_count_pct,
    ROUND(
        100.0 * fraud_exposure_amount / NULLIF(SUM(fraud_exposure_amount) OVER (PARTITION BY transaction_type), 0), 
        2
    ) AS share_of_channel_fraud_exposure_pct
FROM balance_states
ORDER BY transaction_type, fraud_count DESC;
"""

QUERY_DIURNAL_HOURLY_RISK = """
WITH hourly_diurnal AS (
    SELECT
        d.hour_of_day,
        d.time_window_label,
        d.is_night_hours,
        COUNT(t.transaction_id) AS total_tx_count,
        SUM(t.amount) AS total_gtv,
        SUM(t.is_fraud) AS fraud_tx_count,
        SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0.0 END) AS fraud_loss_amount
    FROM core.fact_transactions t
    JOIN core.dim_date d ON t.step = d.step
    GROUP BY d.hour_of_day, d.time_window_label, d.is_night_hours
)
SELECT
    hour_of_day,
    time_window_label,
    is_night_hours,
    total_tx_count,
    ROUND(total_gtv, 2) AS total_gtv,
    fraud_tx_count,
    ROUND(fraud_loss_amount, 2) AS fraud_loss_amount,
    ROUND(100.0 * fraud_tx_count / NULLIF(total_tx_count, 0), 4) AS hourly_fraud_prevalence_pct,
    DENSE_RANK() OVER (ORDER BY (100.0 * fraud_tx_count / NULLIF(total_tx_count, 0)) DESC) AS risk_prevalence_rank,
    ROUND(100.0 * fraud_tx_count / SUM(fraud_tx_count) OVER (), 2) AS share_of_portfolio_fraud_pct,
    ROUND(100.0 * total_tx_count / SUM(total_tx_count) OVER (), 2) AS share_of_portfolio_volume_pct,
    ROUND(
        (100.0 * fraud_tx_count / NULLIF(total_tx_count, 0)) / 
        (100.0 * SUM(fraud_tx_count) OVER () / SUM(total_tx_count) OVER ()), 
        2
    ) AS diurnal_risk_multiplier_vs_baseline
FROM hourly_diurnal
ORDER BY hour_of_day ASC;
"""

QUERIES: dict[str, str] = {
    "temporal_rolling_window": QUERY_TEMPORAL_ROLLING_WINDOW,
    "origin_structuring_sequence": QUERY_ORIGIN_STRUCTURING_SEQUENCE,
    "recipient_fanin_mule_hubs": QUERY_RECIPIENT_FANIN_MULE_HUBS,
    "pareto_decile_concentration": QUERY_PARETO_DECILE_CONCENTRATION,
    "investigation_operations_triage": QUERY_INVESTIGATION_OPERATIONS_TRIAGE,
    "legacy_rule_failure": QUERY_LEGACY_RULE_FAILURE,
    "account_draining_patterns": QUERY_ACCOUNT_DRAINING_PATTERNS,
    "diurnal_hourly_risk": QUERY_DIURNAL_HOURLY_RISK,
}


class RiskAnalyticsEngine:
    """PostgreSQL Risk Analytics Execution Engine for Financial Transaction Surveillance."""

    def __init__(
        self,
        host: str = DB_HOST,
        port: int = DB_PORT,
        dbname: str = DB_NAME,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
        summary_path: Path = DATA_PROCESSED_DIR / "sql_analytics_summary.json",
        report_path: Path = DOCS_DIR / "sql_analytics_findings.md",
    ):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.password = password
        self.summary_path = summary_path
        self.report_path = report_path

    def get_connection(self) -> psycopg2.extensions.connection:
        """Establish a live psycopg2 connection to PostgreSQL."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
        )

    def execute_query(self, query_key: str) -> dict[str, Any]:
        """Execute a named query, timing execution and capturing result rows."""
        if query_key not in QUERIES:
            raise KeyError(
                f"Unknown query key: {query_key}. Available: {list(QUERIES.keys())}"
            )

        sql_text = QUERIES[query_key]
        t0 = time.perf_counter()

        with (
            self.get_connection() as conn,
            conn.cursor(cursor_factory=RealDictCursor) as cur,
        ):
            cur.execute(sql_text)
            rows = [dict(row) for row in cur.fetchall()]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "query_key": query_key,
            "row_count": len(rows),
            "execution_time_ms": round(elapsed_ms, 2),
            "columns": list(rows[0].keys()) if rows else [],
            "data": rows,
        }

    def explain_query(self, query_key: str) -> str:
        """Execute EXPLAIN ANALYZE on a query to inspect PostgreSQL query planner metrics."""
        if query_key not in QUERIES:
            raise KeyError(f"Unknown query key: {query_key}")

        sql_text = f"EXPLAIN (ANALYZE, BUFFERS) {QUERIES[query_key]}"
        with self.get_connection() as conn, conn.cursor() as cur:
            cur.execute(sql_text)
            lines = [row[0] for row in cur.fetchall()]
        return "\n".join(lines)

    def run_all_analyses(self) -> dict[str, Any]:
        """Run all 8 analytical queries sequentially, capturing timings and summary data."""
        logger.info("=" * 80)
        logger.info("STARTING PHASE 9 ADVANCED SQL RISK ANALYTICS SUITE")
        logger.info("=" * 80)

        results: dict[str, Any] = {
            "metadata": {
                "database": self.dbname,
                "host": self.host,
                "port": self.port,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "queries": {},
            "benchmarks": {},
        }

        total_start = time.perf_counter()

        for key in QUERIES:
            logger.info(f"Executing analytical query: [{key}]...")
            q_res = self.execute_query(key)
            results["queries"][key] = q_res
            results["benchmarks"][key] = {
                "row_count": q_res["row_count"],
                "execution_time_ms": q_res["execution_time_ms"],
            }
            logger.info(
                f"  -> [{key}] completed in {q_res['execution_time_ms']:.2f} ms "
                f"({q_res['row_count']:,} rows returned)"
            )

        total_elapsed_ms = (time.perf_counter() - total_start) * 1000.0
        results["benchmarks"]["total_suite_elapsed_ms"] = round(total_elapsed_ms, 2)

        # Save summary JSON
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=decimal_serializer)

        logger.info(
            f"Successfully executed all {len(QUERIES)} SQL queries in {total_elapsed_ms / 1000.0:.2f}s"
        )
        logger.info(f"Summary metrics saved to: {self.summary_path}")
        return results


def main() -> None:
    """CLI Entrypoint for running SQL Analytics."""
    parser = argparse.ArgumentParser(description="Phase 9 SQL Analytics Runner")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Execute a specific query key (e.g. pareto_decile_concentration)",
    )
    parser.add_argument(
        "--explain",
        type=str,
        default=None,
        help="Run EXPLAIN (ANALYZE, BUFFERS) on a specific query key",
    )
    args = parser.parse_args()

    engine = RiskAnalyticsEngine()

    if args.explain:
        logger.info(f"Explaining query: {args.explain}")
        plan = engine.explain_query(args.explain)
        print(plan)
    elif args.query:
        logger.info(f"Executing query: {args.query}")
        res = engine.execute_query(args.query)
        print(f"Executed in {res['execution_time_ms']} ms, Rows: {res['row_count']}")
        for row in res["data"][:10]:
            print(row)
    else:
        results = engine.run_all_analyses()
        print("\n" + "=" * 60)
        print("SQL ANALYTICS BENCHMARK SUMMARY")
        print("=" * 60)
        for k, v in results["benchmarks"].items():
            if k == "total_suite_elapsed_ms":
                print(f"Total Suite Elapsed: {v / 1000.0:.2f} seconds")
            else:
                print(
                    f"  - {k:<35}: {v['execution_time_ms']:>8.2f} ms ({v['row_count']} rows)"
                )
        print("=" * 60)


if __name__ == "__main__":
    main()
