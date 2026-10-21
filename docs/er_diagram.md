# Relational Analytical Data Model & Entity-Relationship (ER) Architecture

## 1. Enterprise Star Schema Architecture

The relational warehouse for the **Financial Transaction Risk & Operations Intelligence Platform** is hosted in PostgreSQL (`financial_risk_analytics`). It implements a 3-tier warehouse pattern:
1. **Staging Schema (`stg`)**: 1:1 tabular landing zone for cleaned transaction streams and operational case records.
2. **Core Dimensional Schema (`core`)**: High-performance Star Schema separating business dimensions from high-volume transaction and investigation facts.
3. **Data Mart Schema (`mart`)**: Aggregated analytical views optimized for DirectQuery and Import models in **Power BI**.

---

## 2. Mermaid Entity-Relationship (ER) Diagram

```mermaid
erDiagram
    %% Dimensions
    DIM_DATE {
        smallint step PK "Chronological hour (1-744)"
        smallint day_number "Calendar day (1-31)"
        smallint hour_of_day "Hour (0-23)"
        smallint day_of_week_number "0=Mon to 6=Sun"
        varchar day_name "Monday..Sunday"
        boolean is_weekend "Flag: Sat/Sun"
        boolean is_night_hours "Flag: 00:00-05:59"
        varchar time_window_label "Night/Morning/Afternoon/Evening"
    }

    DIM_TRANSACTION_TYPE {
        varchar type_code PK "TRANSFER, CASH_OUT, PAYMENT, etc."
        varchar type_name "Descriptive channel name"
        varchar direction "INBOUND, OUTBOUND, INTERNAL"
        boolean is_high_risk_channel "Flag: Fraud occurs on channel"
        boolean requires_dest_balance "Flag: Balance tracked on dest"
    }

    DIM_CASE_STATUS {
        varchar status_code PK "OPEN, UNDER_REVIEW, etc."
        varchar status_category "OPEN or CLOSED"
        boolean is_resolved "Resolution flag"
        smallint display_order "UI sort order"
    }

    DIM_PRIORITY {
        varchar priority_code PK "P1_CRITICAL, P2_HIGH, P3_STANDARD"
        varchar priority_label "Descriptive SLA label"
        numeric sla_target_hours "Target resolution turnaround"
        smallint urgency_rank "1=Highest, 3=Lowest"
    }

    DIM_ACCOUNT {
        bigserial account_key PK "Surrogate account key"
        varchar account_id UK "Natural ID (C... or M...)"
        varchar account_type "CUSTOMER or MERCHANT"
        boolean is_merchant "Merchant flag"
        smallint first_seen_step "Earliest transaction step"
    }

    %% Facts
    FACT_TRANSACTIONS {
        bigint transaction_id PK "Surrogate PK (1..6,362,620)"
        smallint step FK "Hourly temporal step"
        varchar transaction_type FK "Channel modality"
        numeric amount "Transaction value ($)"
        numeric log_amount "log(1 + amount)"
        varchar orig_account_id "Origin account"
        varchar dest_account_id "Destination account"
        varchar dest_type "CUSTOMER or MERCHANT"
        numeric oldbalance_org "Pre-tx origin balance"
        numeric newbalance_orig "Post-tx origin balance"
        numeric orig_balance_delta "old - new balance"
        boolean is_zero_balance_orig "Zero initial balance flag"
        boolean is_orig_balance_err "Accounting violation flag"
        boolean orig_depletion_complete "100% liquidation flag"
        numeric orig_balance_discrepancy "Accounting delta"
        numeric oldbalance_dest "Pre-tx dest balance"
        numeric newbalance_dest "Post-tx dest balance"
        numeric dest_balance_delta "new - old balance"
        boolean is_zero_balance_dest "Zero dest balance flag"
        numeric dest_balance_discrepancy "Accounting delta"
        integer dest_inbound_prior_count "Causal recipient velocity"
        smallint is_fraud "Ground truth (0/1)"
        smallint is_flagged_fraud "Legacy rule (0/1)"
    }

    FACT_INVESTIGATIONS {
        varchar case_id PK "CASE-YYYY-XXXXXX"
        bigint linked_transaction_id FK "Evaluated transaction pointer"
        smallint step FK "Case creation hour"
        varchar transaction_type FK "Transaction modality"
        varchar case_status FK "Workflow status"
        varchar priority FK "Urgency tier"
        varchar assigned_team "Operational squad"
        timestamptz created_at "Ticket creation timestamp"
        timestamptz closed_at "Ticket resolution timestamp"
        boolean is_case_closed "Resolution state"
        numeric turnaround_time_hours "Turnaround Time (TAT)"
        boolean is_sla_breached "SLA breach flag"
        boolean has_notes "Notes recorded flag"
        text cleaned_notes "Normalized notes"
        smallint note_word_count "Word count"
        boolean flag_compromise_suspected "NLP keyword: compromise"
        boolean flag_account_depletion "NLP keyword: depletion"
        boolean flag_velocity_surge "NLP keyword: velocity"
        boolean flag_balance_discrepancy "NLP keyword: mismatch"
        boolean flag_benign_cleared "NLP keyword: cleared"
    }

    %% Relationships
    DIM_DATE ||--o{ FACT_TRANSACTIONS : "synchronizes"
    DIM_TRANSACTION_TYPE ||--o{ FACT_TRANSACTIONS : "classifies"
    FACT_TRANSACTIONS ||--o| FACT_INVESTIGATIONS : "triggers review"
    DIM_DATE ||--o{ FACT_INVESTIGATIONS : "times"
    DIM_TRANSACTION_TYPE ||--o{ FACT_INVESTIGATIONS : "categorizes"
    DIM_CASE_STATUS ||--o{ FACT_INVESTIGATIONS : "tracks"
    DIM_PRIORITY ||--o{ FACT_INVESTIGATIONS : "enforces SLA"
```

---

## 3. Schema Catalog & Entity Specifications

### A. `core.fact_transactions`
- **Business Meaning**: Represents the fundamental financial fact event (an individual monetary transfer or liquidation request).
- **Grain**: 1 row = 1 executed financial transaction.
- **Primary Key**: `transaction_id` (`BIGINT`, range `1` to `6,362,620`).
- **Foreign Keys**:
  - `step` $\rightarrow$ `core.dim_date(step)`
  - `transaction_type` $\rightarrow$ `core.dim_transaction_type(type_code)`
- **Indexing Strategy**:
  - `idx_fact_tx_step`: Optimizes temporal sliding windows and time-series queries.
  - `idx_fact_tx_type`: Accelerates channel filtering (`TRANSFER` and `CASH_OUT`).
  - `idx_fact_tx_fraud`: Speeds up fraud exposure filtering and loss calculations.
  - `idx_fact_tx_orig` and `idx_fact_tx_dest`: Supports customer 360 lookup and entity velocity joins.
  - `idx_fact_tx_composite_risk`: Composite index `(transaction_type, is_fraud, amount)` for sub-second dashboard aggregation.

---

### B. `core.fact_investigations`
- **Business Meaning**: Represents the operational human investigation ticket initiated when an alert rule triggers.
- **Grain**: 1 row = 1 operational case investigation ticket (5,000 cases).
- **Primary Key**: `case_id` (`VARCHAR(32)`).
- **Foreign Keys**:
  - `linked_transaction_id` $\rightarrow$ `core.fact_transactions(transaction_id)`
  - `step` $\rightarrow$ `core.dim_date(step)`
  - `transaction_type` $\rightarrow$ `core.dim_transaction_type(type_code)`
  - `case_status` $\rightarrow$ `core.dim_case_status(status_code)`
  - `priority` $\rightarrow$ `core.dim_priority(priority_code)`
- **Indexing Strategy**:
  - `idx_fact_inv_linked_tx`: Enables rapid 1:1 and 1:N joins between case queues and underlying ledger transactions.
  - `idx_fact_inv_status` and `idx_fact_inv_priority`: Optimizes operations backlog views.
  - `idx_fact_inv_sla`: Filters operational SLA breaches for risk leadership audits.

---

### C. Conformed Dimension Tables
1. **`core.dim_date`**: 744 rows covering steps 1 to 744 (31 calendar days). Provides attributes for diurnal cycles, day of week, weekend flags, and temporal day grouping.
2. **`core.dim_transaction_type`**: 5 rows (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`). Identifies flow direction and flags high-risk fraud rails.
3. **`core.dim_case_status`**: 4 rows (`OPEN`, `UNDER_REVIEW`, `CLOSED_CONFIRMED_FRAUD`, `CLOSED_FALSE_POSITIVE`). Categorizes tickets into open backlog vs resolved states.
4. **`core.dim_priority`**: 3 rows (`P1_CRITICAL`, `P2_HIGH`, `P3_STANDARD`). Governs SLA target hours (24h, 48h, 96h).
5. **`core.dim_account`**: Unique registry of customer and merchant identifiers.

---

## 4. Analytical Data Marts (`mart` Schema Views)

To decouple business reporting queries from the massive transaction fact table, we defined three pre-aggregated views:

1. **`mart.mart_risk_executive_overview`**:
   - Aggregates by `day_number`, `day_name`, and `transaction_type`.
   - Metrics: Gross Transaction Value (GTV), total transaction volume, fraud count, fraud exposure value, fraud prevalence %, and mean ticket size.
   - Purpose: Direct feed for **Page 1 (Executive Overview)** of the Power BI dashboard.
2. **`mart.mart_investigation_operations`**:
   - Aggregates by `assigned_team`, `priority_code`, and `case_status`.
   - Metrics: Total tickets, open backlog, closed tickets, SLA breach count, SLA breach rate %, median/mean turnaround time (TAT), and keyword flags.
   - Purpose: Direct feed for **Page 3 (Investigation Operations)** of the Power BI dashboard.
3. **`mart.mart_channel_exposure`**:
   - Modality risk concentration metrics comparing total transaction volume against fraud value concentration.
   - Purpose: Direct feed for **Page 2 (Transaction Risk Intelligence)**.

---

## 5. Architectural Distinctions: Genuine Source vs Synthetic Case Data

- **Transaction Records**: Derived from the benchmark PaySim multi-agent simulation model, representing ledger accounting events.
- **Investigation Records**: Generated as an explicit operational simulation to model ticket workflows, SLA monitoring, and text processing.
- **Integrity Boundary**: The foreign key `linked_transaction_id` maps cleanly to `transaction_id`. Operational features exist solely in `fact_investigations` and are strictly excluded from the pre-transaction feature matrix.
