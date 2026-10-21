-- ============================================================================
-- 03_create_dimensions_facts.sql
-- Financial Transaction Risk & Operations Intelligence Platform
-- Dimensional Warehouse Layer (Star Schema) DDL
-- ============================================================================

DROP TABLE IF EXISTS core.fact_investigations CASCADE;
DROP TABLE IF EXISTS core.fact_transactions CASCADE;
DROP TABLE IF EXISTS core.dim_account CASCADE;
DROP TABLE IF EXISTS core.dim_priority CASCADE;
DROP TABLE IF EXISTS core.dim_case_status CASCADE;
DROP TABLE IF EXISTS core.dim_transaction_type CASCADE;
DROP TABLE IF EXISTS core.dim_date CASCADE;

-- ============================================================================
-- 1. DIMENSION TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Dimension: core.dim_date
-- Grain: 1 record per simulation hour (1 to 744, covering 31 calendar days)
-- ----------------------------------------------------------------------------
CREATE TABLE core.dim_date (
    step                          SMALLINT PRIMARY KEY,
    day_number                    SMALLINT NOT NULL CHECK (day_number BETWEEN 1 AND 31),
    hour_of_day                   SMALLINT NOT NULL CHECK (hour_of_day BETWEEN 0 AND 23),
    day_of_week_number            SMALLINT NOT NULL CHECK (day_of_week_number BETWEEN 0 AND 6),
    day_name                      VARCHAR(10) NOT NULL,
    is_weekend                    BOOLEAN NOT NULL,
    is_night_hours                BOOLEAN NOT NULL,
    time_window_label             VARCHAR(32) NOT NULL
);

-- ----------------------------------------------------------------------------
-- Dimension: core.dim_transaction_type
-- Grain: 1 record per transaction modality (5 rows)
-- ----------------------------------------------------------------------------
CREATE TABLE core.dim_transaction_type (
    type_code                     VARCHAR(16) PRIMARY KEY,
    type_name                     VARCHAR(32) NOT NULL,
    direction                     VARCHAR(8) NOT NULL CHECK (direction IN ('INBOUND', 'OUTBOUND', 'INTERNAL')),
    is_high_risk_channel          BOOLEAN NOT NULL,
    requires_dest_balance         BOOLEAN NOT NULL
);

-- ----------------------------------------------------------------------------
-- Dimension: core.dim_case_status
-- Grain: 1 record per operational case status (4 rows)
-- ----------------------------------------------------------------------------
CREATE TABLE core.dim_case_status (
    status_code                   VARCHAR(32) PRIMARY KEY,
    status_category               VARCHAR(16) NOT NULL CHECK (status_category IN ('OPEN', 'CLOSED')),
    is_resolved                   BOOLEAN NOT NULL,
    display_order                 SMALLINT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Dimension: core.dim_priority
-- Grain: 1 record per triage priority tier (3 rows)
-- ----------------------------------------------------------------------------
CREATE TABLE core.dim_priority (
    priority_code                 VARCHAR(16) PRIMARY KEY,
    priority_label                VARCHAR(32) NOT NULL,
    sla_target_hours              NUMERIC(6, 1) NOT NULL,
    urgency_rank                  SMALLINT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Dimension: core.dim_account
-- Grain: 1 record per unique entity (Customer or Merchant)
-- ----------------------------------------------------------------------------
CREATE TABLE core.dim_account (
    account_key                   BIGSERIAL PRIMARY KEY,
    account_id                    VARCHAR(32) UNIQUE NOT NULL,
    account_type                  VARCHAR(16) NOT NULL CHECK (account_type IN ('CUSTOMER', 'MERCHANT')),
    is_merchant                   BOOLEAN NOT NULL,
    first_seen_step               SMALLINT NOT NULL
);

-- ============================================================================
-- 2. FACT TABLES
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Fact: core.fact_transactions
-- Grain: 1 record per executed financial transaction (6,362,620 rows)
-- ----------------------------------------------------------------------------
CREATE TABLE core.fact_transactions (
    transaction_id                BIGINT PRIMARY KEY,
    step                          SMALLINT NOT NULL REFERENCES core.dim_date(step),
    transaction_type              VARCHAR(16) NOT NULL REFERENCES core.dim_transaction_type(type_code),
    amount                        NUMERIC(18, 2) NOT NULL CHECK (amount >= 0.0),
    log_amount                    NUMERIC(10, 4) NOT NULL,
    orig_account_id               VARCHAR(32) NOT NULL,
    dest_account_id               VARCHAR(32) NOT NULL,
    dest_type                     VARCHAR(16) NOT NULL,
    oldbalance_org                NUMERIC(18, 2) NOT NULL,
    newbalance_orig               NUMERIC(18, 2) NOT NULL,
    orig_balance_delta            NUMERIC(18, 2) NOT NULL,
    is_zero_balance_orig          BOOLEAN NOT NULL,
    is_orig_balance_err           BOOLEAN NOT NULL,
    orig_depletion_complete       BOOLEAN NOT NULL,
    orig_balance_discrepancy      NUMERIC(18, 2) NOT NULL,
    oldbalance_dest               NUMERIC(18, 2) NOT NULL,
    newbalance_dest               NUMERIC(18, 2) NOT NULL,
    dest_balance_delta            NUMERIC(18, 2) NOT NULL,
    is_zero_balance_dest          BOOLEAN NOT NULL,
    dest_balance_discrepancy      NUMERIC(18, 2) NOT NULL,
    dest_inbound_prior_count      INTEGER NOT NULL DEFAULT 0,
    is_fraud                      SMALLINT NOT NULL CHECK (is_fraud IN (0, 1)),
    is_flagged_fraud              SMALLINT NOT NULL CHECK (is_flagged_fraud IN (0, 1))
);

-- ----------------------------------------------------------------------------
-- Fact: core.fact_investigations
-- Grain: 1 record per operational case ticket (5,000 rows)
-- ----------------------------------------------------------------------------
CREATE TABLE core.fact_investigations (
    case_id                       VARCHAR(32) PRIMARY KEY,
    linked_transaction_id         BIGINT NOT NULL REFERENCES core.fact_transactions(transaction_id),
    step                          SMALLINT NOT NULL REFERENCES core.dim_date(step),
    transaction_type              VARCHAR(16) NOT NULL REFERENCES core.dim_transaction_type(type_code),
    case_status                   VARCHAR(32) NOT NULL REFERENCES core.dim_case_status(status_code),
    priority                      VARCHAR(16) NOT NULL REFERENCES core.dim_priority(priority_code),
    assigned_team                 VARCHAR(64) NOT NULL,
    created_at                    TIMESTAMPTZ NOT NULL,
    closed_at                     TIMESTAMPTZ,
    is_case_closed                BOOLEAN NOT NULL,
    turnaround_time_hours         NUMERIC(10, 2),
    is_sla_breached               BOOLEAN NOT NULL,
    has_notes                     BOOLEAN NOT NULL,
    cleaned_notes                 TEXT NOT NULL,
    note_word_count               SMALLINT NOT NULL,
    flag_compromise_suspected     BOOLEAN NOT NULL,
    flag_account_depletion        BOOLEAN NOT NULL,
    flag_velocity_surge           BOOLEAN NOT NULL,
    flag_balance_discrepancy      BOOLEAN NOT NULL,
    flag_benign_cleared           BOOLEAN NOT NULL
);

-- ============================================================================
-- 3. INDEXING STRATEGY FOR HIGH-PERFORMANCE ANALYTICS
-- ============================================================================

-- B-Tree Indexes on fact_transactions
CREATE INDEX idx_fact_tx_step ON core.fact_transactions (step);
CREATE INDEX idx_fact_tx_type ON core.fact_transactions (transaction_type);
CREATE INDEX idx_fact_tx_fraud ON core.fact_transactions (is_fraud);
CREATE INDEX idx_fact_tx_orig ON core.fact_transactions (orig_account_id);
CREATE INDEX idx_fact_tx_dest ON core.fact_transactions (dest_account_id);
CREATE INDEX idx_fact_tx_composite_risk ON core.fact_transactions (transaction_type, is_fraud, amount);

-- B-Tree Indexes on fact_investigations
CREATE INDEX idx_fact_inv_linked_tx ON core.fact_investigations (linked_transaction_id);
CREATE INDEX idx_fact_inv_status ON core.fact_investigations (case_status);
CREATE INDEX idx_fact_inv_priority ON core.fact_investigations (priority);
CREATE INDEX idx_fact_inv_team ON core.fact_investigations (assigned_team);
CREATE INDEX idx_fact_inv_sla ON core.fact_investigations (is_sla_breached);
