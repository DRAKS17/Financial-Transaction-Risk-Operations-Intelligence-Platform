-- ============================================================================
-- 05_risk_analysis.sql
-- Financial Transaction Risk & Operations Intelligence Platform
-- Advanced SQL Analytics & Financial Risk Typologies
--
-- Target Database: PostgreSQL 14+ (Target Schema: core, mart)
-- Target Data Scale: 6,362,620 Transactions ($1.144 Trillion GTV)
--
-- Analytical Typologies Covered:
-- 1. Temporal Dynamics & 24-Hour Rolling Fraud Exposure Window
-- 2. Originator Rapid Structuring & Layering Sequence (LAG / Lead Delta)
-- 3. Recipient Fan-In Network & High-Velocity Mule Collection Hubs
-- 4. Channel Risk Exposure & Pareto Concentration (NTILE Decile Lorenz Curve)
-- 5. Operational Investigation Squad Performance & SLA Backlog Aging Matrix
-- 6. Legacy Heuristic Failure & Detection Efficacy Audit (is_flagged_fraud)
-- 7. Account Draining / Balance Depletion Risk Patterns
-- 8. Diurnal Diachronic Cycle & Off-Hours Vulnerability Analysis
-- ============================================================================

-- ============================================================================
-- QUERY 1: Temporal Dynamics & 24-Hour Rolling Fraud Exposure Window
--
-- Business Question:
-- How does fraud volume and monetary exposure evolve over simulation time,
-- and what are the peak 24-hour rolling burst windows where operational
-- defense capacity was under maximum stress?
--
-- Advanced SQL Concepts:
-- - Common Table Expression (CTE) for hourly pre-aggregation
-- - Sliding window frame: ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
-- - Cumulative running totals: ROWS UNBOUNDED PRECEDING
-- - Window percentage ranking: PERCENT_RANK() OVER (ORDER BY ...)
-- ============================================================================
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
        -- 24-Hour Sliding Window Metrics
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
        -- Cumulative Running Totals (Portfolio Trajectory)
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
    -- Exposure Stress Percentile
    ROUND(
        (PERCENT_RANK() OVER (ORDER BY rolling_24h_fraud_exposure) * 100)::NUMERIC, 
        2
    ) AS rolling_exposure_stress_percentile
FROM rolling_windows
ORDER BY step ASC;


-- ============================================================================
-- QUERY 2: Originator Rapid Structuring & Layering Sequence
--
-- Business Question:
-- Are origin accounts executing rapid sequential transactions across channels
-- (e.g. TRANSFER immediately followed by CASH_OUT) within narrow time deltas
-- (<= 1 hour), indicating automated script-driven account compromise or layering?
--
-- Advanced SQL Concepts:
-- - Multi-column LAG() window functions partitioned by account entity
-- - Inter-transaction temporal delta (step - prev_step)
-- - Inter-transaction channel transition matrix (prev_tx_type -> curr_tx_type)
-- - Balance delta verification between consecutive events
-- ============================================================================
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
ORDER BY (step - prev_step) ASC, curr_amount DESC;


-- ============================================================================
-- QUERY 3: Recipient Fan-In Network & High-Inbound Aggregator Hubs
--
-- Business Question:
-- Which destination accounts receive high inbound transfer velocity from multiple
-- originators, and how much gross capital and fraudulent exposure flows
-- into these potential mule aggregator nodes?
--
-- Advanced SQL Concepts:
-- - High-risk channel filtering to optimize index scan on core.fact_transactions
-- - DENSE_RANK() window function for ranking mule hubs by volume and ticket size
-- - Concentration ratio: Average inbound ticket size vs channel benchmark
-- ============================================================================
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
LIMIT 100;


-- ============================================================================
-- QUERY 4: Channel Vulnerability & Pareto Concentration (Decile Lorenz Curve)
--
-- Business Question:
-- How concentrated is transaction volume and fraud loss across ticket size deciles?
-- What percentage of total fraud losses is concentrated in the top 10% (Decile 10)?
--
-- Advanced SQL Concepts:
-- - NTILE(10) analytic window function partitioning 6.36M records into deciles
-- - Aggregation over window partitions
-- - Cumulative sum window functions to generate an empirical Lorenz / Pareto curve
-- ============================================================================
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
    -- Decile shares
    ROUND(100.0 * decile_gtv / SUM(decile_gtv) OVER (), 2) AS share_of_total_gtv_pct,
    ROUND(100.0 * decile_fraud_exposure / SUM(decile_fraud_exposure) OVER (), 2) AS share_of_total_fraud_loss_pct,
    -- Cumulative Lorenz / Pareto Trajectory
    ROUND(100.0 * SUM(decile_gtv) OVER (ORDER BY amount_decile) / SUM(decile_gtv) OVER (), 2) AS cum_gtv_pct,
    ROUND(100.0 * SUM(decile_fraud_exposure) OVER (ORDER BY amount_decile) / SUM(decile_fraud_exposure) OVER (), 2) AS cum_fraud_loss_pct
FROM decile_summary
ORDER BY amount_decile ASC;


-- ============================================================================
-- QUERY 5: Operational Investigation Squad Triage & SLA Backlog Aging Matrix
--
-- Business Question:
-- What is the current backlog aging distribution across teams and priority tiers,
-- which squads have the highest SLA breach risk, and how does turnaround time
-- distribute across the P50, P75, and P90 percentiles?
--
-- Advanced SQL Concepts:
-- - Continuous percentile window functions: PERCENTILE_CONT(0.50), PERCENTILE_CONT(0.90)
-- - Dynamic open ticket age calculation against maximum operational snapshot timestamp
-- - Partitioned window ranking: RANK() OVER (PARTITION BY priority ORDER BY mean_tat DESC)
-- - Conditional case aggregation for discrete SLA aging buckets
-- ============================================================================
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
        -- Closed Case TAT Metrics
        ROUND(AVG(CASE WHEN is_case_closed = TRUE THEN turnaround_time_hours ELSE NULL END), 2) AS mean_closed_tat_hours,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (
            ORDER BY CASE WHEN is_case_closed = TRUE THEN turnaround_time_hours ELSE NULL END
        )::NUMERIC, 2) AS p50_median_closed_tat_hours,
        ROUND(PERCENTILE_CONT(0.90) WITHIN GROUP (
            ORDER BY CASE WHEN is_case_closed = TRUE THEN turnaround_time_hours ELSE NULL END
        )::NUMERIC, 2) AS p90_closed_tat_hours,
        -- Open Backlog Aging Buckets
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


-- ============================================================================
-- QUERY 6: Legacy Rule Failure & Detection Efficacy Audit (is_flagged_fraud)
--
-- Business Question:
-- What is the mathematical and monetary shortfall of the bank's legacy static rule
-- (flagging TRANSFER > 200,000 as is_flagged_fraud)?
-- How much unmitigated fraud exposure slipped through the legacy defense perimeter?
--
-- Advanced SQL Concepts:
-- - Categorical confusion matrix segmentation
-- - Monetary exposure delta (detected vs missed gross exposure)
-- - Precision, Recall, and Capture Yield calculations in pure analytical SQL
-- ============================================================================
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


-- ============================================================================
-- QUERY 7: Account Draining / Balance Depletion Risk Patterns
--
-- Business Question:
-- What proportion of fraudulent transfers and cash-outs execute complete account
-- draining (newbalance_orig = 0), and how does the fraud rate differ across
-- balance depletion states?
--
-- Advanced SQL Concepts:
-- - Multi-condition state classification based on accounting balance deltas
-- - Partitioned window share: SUM(...) OVER (PARTITION BY transaction_type)
-- - Correlation proof for the orig_depletion_complete engineered feature
-- ============================================================================
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


-- ============================================================================
-- QUERY 8: Diurnal / Hourly Fraud Vulnerability & Off-Hours Risk Surge
--
-- Business Question:
-- How does transaction risk fluctuate across the 24-hour diurnal cycle?
-- Are nocturnal off-hours (00:00 - 06:00) disproportionately targeted by fraudsters,
-- and what is the exact signal-to-noise multiplier during the night?
--
-- Advanced SQL Concepts:
-- - Star schema join between core.fact_transactions and core.dim_date
-- - Window ranking of diurnal hours by fraud rate: DENSE_RANK() OVER (...)
-- - Portfolio share metrics for temporal load balancing
-- ============================================================================
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
