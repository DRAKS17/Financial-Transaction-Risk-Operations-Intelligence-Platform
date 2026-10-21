-- ============================================================================
-- 02_create_staging_tables.sql
-- Financial Transaction Risk & Operations Intelligence Platform
-- Staging Layer DDL
-- ============================================================================

DROP TABLE IF EXISTS stg.stg_investigations CASCADE;
DROP TABLE IF EXISTS stg.stg_transactions CASCADE;

-- ----------------------------------------------------------------------------
-- Table: stg.stg_transactions
-- Grain: 1 record per completed transaction event
-- ----------------------------------------------------------------------------
CREATE TABLE stg.stg_transactions (
    transaction_id                BIGINT PRIMARY KEY,
    step                          SMALLINT NOT NULL,
    day                           SMALLINT NOT NULL,
    hour_of_day                   SMALLINT NOT NULL,
    type                          VARCHAR(16) NOT NULL,
    amount                        NUMERIC(18, 2) NOT NULL,
    name_orig                     VARCHAR(32) NOT NULL,
    oldbalance_org                NUMERIC(18, 2) NOT NULL,
    newbalance_orig               NUMERIC(18, 2) NOT NULL,
    orig_balance_delta            NUMERIC(18, 2) NOT NULL,
    is_zero_balance_orig          BOOLEAN NOT NULL,
    is_orig_balance_err           BOOLEAN NOT NULL,
    name_dest                     VARCHAR(32) NOT NULL,
    dest_type                     VARCHAR(16) NOT NULL,
    oldbalance_dest               NUMERIC(18, 2) NOT NULL,
    newbalance_dest               NUMERIC(18, 2) NOT NULL,
    dest_balance_delta            NUMERIC(18, 2) NOT NULL,
    is_zero_balance_dest          BOOLEAN NOT NULL,
    is_fraud                      SMALLINT NOT NULL,
    is_flagged_fraud              SMALLINT NOT NULL
);

-- ----------------------------------------------------------------------------
-- Table: stg.stg_investigations
-- Grain: 1 record per operational case ticket
-- ----------------------------------------------------------------------------
CREATE TABLE stg.stg_investigations (
    case_id                       VARCHAR(32) PRIMARY KEY,
    linked_transaction_id         BIGINT NOT NULL,
    tx_step                       SMALLINT NOT NULL,
    tx_type                       VARCHAR(16) NOT NULL,
    tx_amount                     NUMERIC(18, 2) NOT NULL,
    priority                      VARCHAR(16) NOT NULL,
    assigned_team                 VARCHAR(64) NOT NULL,
    case_status                   VARCHAR(32) NOT NULL,
    case_created_at               TIMESTAMPTZ NOT NULL,
    closed_at                     TIMESTAMPTZ,
    is_case_closed                BOOLEAN NOT NULL,
    turnaround_time_hours         NUMERIC(10, 2),
    is_sla_breached               BOOLEAN NOT NULL,
    has_notes                     BOOLEAN NOT NULL,
    raw_investigation_notes       TEXT,
    cleaned_investigation_notes   TEXT,
    note_char_length              INTEGER NOT NULL,
    note_word_count               SMALLINT NOT NULL,
    note_avg_word_length          NUMERIC(6, 2) NOT NULL,
    flag_account_depletion        BOOLEAN NOT NULL,
    flag_velocity_surge           BOOLEAN NOT NULL,
    flag_compromise_suspected     BOOLEAN NOT NULL,
    flag_balance_discrepancy      BOOLEAN NOT NULL,
    flag_benign_cleared           BOOLEAN NOT NULL
);
