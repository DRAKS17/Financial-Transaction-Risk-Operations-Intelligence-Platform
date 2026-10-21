-- ============================================================================
-- 04_transformations.sql
-- Financial Transaction Risk & Operations Intelligence Platform
-- Dimensional Seeding and Data Mart Views (mart Schema)
-- ============================================================================

-- ============================================================================
-- 1. SEED DIMENSION TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Seed: core.dim_transaction_type
-- ----------------------------------------------------------------------------
INSERT INTO core.dim_transaction_type (type_code, type_name, direction, is_high_risk_channel, requires_dest_balance)
VALUES
    ('TRANSFER', 'Account Transfer', 'INTERNAL', TRUE, TRUE),
    ('CASH_OUT', 'Cash Withdrawal / Outflow', 'OUTBOUND', TRUE, TRUE),
    ('PAYMENT', 'Merchant Retail Payment', 'OUTBOUND', FALSE, FALSE),
    ('CASH_IN', 'Cash Deposit / Inflow', 'INBOUND', FALSE, TRUE),
    ('DEBIT', 'Direct Debit', 'OUTBOUND', FALSE, TRUE)
ON CONFLICT (type_code) DO UPDATE SET
    type_name = EXCLUDED.type_name,
    direction = EXCLUDED.direction,
    is_high_risk_channel = EXCLUDED.is_high_risk_channel,
    requires_dest_balance = EXCLUDED.requires_dest_balance;

-- ----------------------------------------------------------------------------
-- Seed: core.dim_case_status
-- ----------------------------------------------------------------------------
INSERT INTO core.dim_case_status (status_code, status_category, is_resolved, display_order)
VALUES
    ('OPEN', 'OPEN', FALSE, 1),
    ('UNDER_REVIEW', 'OPEN', FALSE, 2),
    ('CLOSED_CONFIRMED_FRAUD', 'CLOSED', TRUE, 3),
    ('CLOSED_FALSE_POSITIVE', 'CLOSED', TRUE, 4)
ON CONFLICT (status_code) DO UPDATE SET
    status_category = EXCLUDED.status_category,
    is_resolved = EXCLUDED.is_resolved,
    display_order = EXCLUDED.display_order;

-- ----------------------------------------------------------------------------
-- Seed: core.dim_priority
-- ----------------------------------------------------------------------------
INSERT INTO core.dim_priority (priority_code, priority_label, sla_target_hours, urgency_rank)
VALUES
    ('P1_CRITICAL', 'Priority 1 - Critical Risk', 24.0, 1),
    ('P2_HIGH', 'Priority 2 - High Risk', 48.0, 2),
    ('P3_STANDARD', 'Priority 3 - Standard Routine', 96.0, 3)
ON CONFLICT (priority_code) DO UPDATE SET
    priority_label = EXCLUDED.priority_label,
    sla_target_hours = EXCLUDED.sla_target_hours,
    urgency_rank = EXCLUDED.urgency_rank;

-- ----------------------------------------------------------------------------
-- Seed: core.dim_date (Populate 744 hourly steps = 31 days)
-- ----------------------------------------------------------------------------
INSERT INTO core.dim_date (step, day_number, hour_of_day, day_of_week_number, day_name, is_weekend, is_night_hours, time_window_label)
SELECT
    s AS step,
    ((s - 1) / 24 + 1)::SMALLINT AS day_number,
    ((s - 1) % 24)::SMALLINT AS hour_of_day,
    (((s - 1) / 24) % 7)::SMALLINT AS day_of_week_number,
    CASE (((s - 1) / 24) % 7)
        WHEN 0 THEN 'Monday'
        WHEN 1 THEN 'Tuesday'
        WHEN 2 THEN 'Wednesday'
        WHEN 3 THEN 'Thursday'
        WHEN 4 THEN 'Friday'
        WHEN 5 THEN 'Saturday'
        WHEN 6 THEN 'Sunday'
    END AS day_name,
    CASE WHEN (((s - 1) / 24) % 7) >= 5 THEN TRUE ELSE FALSE END AS is_weekend,
    CASE WHEN ((s - 1) % 24) < 6 THEN TRUE ELSE FALSE END AS is_night_hours,
    CASE
        WHEN ((s - 1) % 24) BETWEEN 0 AND 5 THEN 'Night (00-06)'
        WHEN ((s - 1) % 24) BETWEEN 6 AND 11 THEN 'Morning (06-12)'
        WHEN ((s - 1) % 24) BETWEEN 12 AND 17 THEN 'Afternoon (12-18)'
        ELSE 'Evening (18-24)'
    END AS time_window_label
FROM generate_series(1, 744) AS s
ON CONFLICT (step) DO NOTHING;

-- ============================================================================
-- 2. ANALYTICAL DATA MARTS (mart SCHEMA VIEWS)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Mart View: mart.mart_risk_executive_overview
-- Power BI Touchpoint: Page 1 (Executive Portfolio Overview)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW mart.mart_risk_executive_overview AS
SELECT
    d.day_number,
    d.day_name,
    d.is_weekend,
    t.transaction_type,
    tt.is_high_risk_channel,
    COUNT(t.transaction_id) AS total_transactions,
    SUM(t.amount) AS gross_transaction_value,
    SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_transactions,
    SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0.0 END) AS fraud_exposure_value,
    SUM(CASE WHEN t.is_flagged_fraud = 1 THEN 1 ELSE 0 END) AS heuristic_flagged_count,
    ROUND((SUM(CASE WHEN t.is_fraud = 1 THEN 1.0 ELSE 0.0 END) / COUNT(t.transaction_id)) * 100, 4) AS fraud_prevalence_pct,
    ROUND(AVG(t.amount), 2) AS mean_ticket_size,
    ROUND(AVG(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE NULL END), 2) AS mean_fraud_ticket_size
FROM core.fact_transactions t
JOIN core.dim_date d ON t.step = d.step
JOIN core.dim_transaction_type tt ON t.transaction_type = tt.type_code
GROUP BY d.day_number, d.day_name, d.is_weekend, t.transaction_type, tt.is_high_risk_channel;

-- ----------------------------------------------------------------------------
-- Mart View: mart.mart_investigation_operations
-- Power BI Touchpoint: Page 3 (Investigation Operations)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW mart.mart_investigation_operations AS
SELECT
    inv.assigned_team,
    p.priority_code,
    p.priority_label,
    p.sla_target_hours,
    cs.status_code,
    cs.status_category,
    COUNT(inv.case_id) AS total_cases,
    SUM(CASE WHEN cs.status_category = 'OPEN' THEN 1 ELSE 0 END) AS active_backlog,
    SUM(CASE WHEN cs.status_category = 'CLOSED' THEN 1 ELSE 0 END) AS closed_cases,
    SUM(CASE WHEN inv.is_sla_breached THEN 1 ELSE 0 END) AS sla_breaches,
    ROUND(
        (SUM(CASE WHEN inv.is_sla_breached THEN 1.0 ELSE 0.0 END) / 
        NULLIF(SUM(CASE WHEN cs.status_category = 'CLOSED' THEN 1 ELSE 0 END), 0)) * 100, 
        2
    ) AS sla_breach_rate_pct,
    ROUND(AVG(inv.turnaround_time_hours), 2) AS mean_turnaround_time_hours,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY inv.turnaround_time_hours)::NUMERIC, 2) AS median_turnaround_time_hours,
    SUM(CASE WHEN inv.flag_compromise_suspected THEN 1 ELSE 0 END) AS cases_compromise_suspected,
    SUM(CASE WHEN inv.flag_account_depletion THEN 1 ELSE 0 END) AS cases_account_depletion,
    SUM(CASE WHEN inv.flag_velocity_surge THEN 1 ELSE 0 END) AS cases_velocity_surge
FROM core.fact_investigations inv
JOIN core.dim_priority p ON inv.priority = p.priority_code
JOIN core.dim_case_status cs ON inv.case_status = cs.status_code
GROUP BY inv.assigned_team, p.priority_code, p.priority_label, p.sla_target_hours, cs.status_code, cs.status_category;

-- ----------------------------------------------------------------------------
-- Mart View: mart.mart_channel_exposure
-- Power BI Touchpoint: Page 2 (Transaction Risk Intelligence)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW mart.mart_channel_exposure AS
SELECT
    tt.type_code,
    tt.type_name,
    tt.direction,
    tt.is_high_risk_channel,
    COUNT(t.transaction_id) AS total_volume,
    SUM(t.amount) AS total_amount,
    SUM(CASE WHEN t.is_fraud = 1 THEN 1 ELSE 0 END) AS fraud_count,
    SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0.0 END) AS fraud_loss_value,
    ROUND(
        (SUM(CASE WHEN t.is_fraud = 1 THEN 1.0 ELSE 0.0 END) / COUNT(t.transaction_id)) * 100, 
        4
    ) AS fraud_rate_pct,
    ROUND(
        (SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0.0 END) / 
        NULLIF(SUM(t.amount), 0)) * 100, 
        4
    ) AS fraud_value_concentration_pct
FROM core.dim_transaction_type tt
LEFT JOIN core.fact_transactions t ON tt.type_code = t.transaction_type
GROUP BY tt.type_code, tt.type_name, tt.direction, tt.is_high_risk_channel;
