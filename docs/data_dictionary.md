# Data Dictionary

## 1. Overview
This document defines the schema, types, constraints, and business definitions for all raw tables in the **Financial Transaction Risk & Operations Intelligence Platform**.

---

## 2. Table: `raw_transactions` (PaySim Stream)

- **Source File**: `data/raw/paysim.csv`
- **Grain**: 1 record per completed financial transaction event.
- **Estimated Rows**: 6,362,620
- **Primary Key Candidate**: Synthetic surrogate key (e.g. `transaction_id` generated during ingestion/staging), as natural composite keys `(step, nameOrig, nameDest, amount)` have edge-case collisions in high-velocity multi-threading simulations.

### Column Definitions:

| Column Name | Inferred Data Type | Nullable | Domain / Allowed Values | Description & Analytical Role |
|---|---|---|---|---|
| `step` | `INTEGER` | No | `1` to `744` | Hourly chronological index from simulation start. `step 1` = Day 1, Hour 1; `step 744` = Day 30, Hour 24. Used for temporal train/test split. |
| `type` | `VARCHAR(16)` | No | `PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEBIT`, `CASH_IN` | Modality of transaction. Empirically, fraudulent transfers in PaySim concentrate in `TRANSFER` and `CASH_OUT`. |
| `amount` | `NUMERIC(18, 2)` | No | `>= 0.00` | Transaction value in local fiat currency. Zero-amount and micro-transactions exist as probe attempts. |
| `nameOrig` | `VARCHAR(32)` | No | Regex: `^C[0-9]+$` | Originating customer account ID. Initiates the debit/transfer. |
| `oldbalanceOrg` | `NUMERIC(18, 2)` | No | `>= 0.00` | Account balance of the originator immediately before the transaction was executed. |
| `newbalanceOrig` | `NUMERIC(18, 2)` | No | `>= 0.00` | Account balance of the originator immediately after the transaction was executed. |
| `nameDest` | `VARCHAR(32)` | No | Regex: `^[CM][0-9]+$` | Recipient entity ID. `C...` denotes customer peer; `M...` denotes merchant terminal. |
| `oldbalanceDest` | `NUMERIC(18, 2)` | No | `>= 0.00` | Account balance of recipient immediately before transaction (often 0.0 for merchants). |
| `newbalanceDest` | `NUMERIC(18, 2)` | No | `>= 0.00` | Account balance of recipient immediately after transaction (often 0.0 for merchants). |
| `isFraud` | `SMALLINT` | No | `0, 1` | Ground-truth fraud label. `1` = Transaction simulated under malicious takeover/theft behavior; `0` = Benign. |
| `isFlaggedFraud` | `SMALLINT` | No | `0, 1` | Naive heuristic rule flag. `1` = Single transaction attempt > 200,000 currency units on transfer rails. |

---

## 3. Table: `raw_investigations` (Operational Case Logs)

- **Source File**: `data/raw/investigation_cases.csv`
- **Grain**: 1 record per generated fraud investigation case.
- **Estimated Rows**: 5,000
- **Primary Key**: `case_id`
- **Foreign Key**: `linked_transaction_id` -> `raw_transactions(row_id)`

### Column Definitions:

| Column Name | Inferred Data Type | Nullable | Domain / Allowed Values | Description & Analytical Role |
|---|---|---|---|---|
| `case_id` | `VARCHAR(32)` | No | Format: `CASE-YYYY-XXXXXX` | Globally unique case investigation ticket reference. |
| `linked_transaction_id` | `INTEGER` | No | `>= 1` | Relational pointer linking the case to the flagged transaction. |
| `tx_step` | `INTEGER` | No | `1` to `744` | Transaction step corresponding to alert trigger. |
| `tx_type` | `VARCHAR(16)` | No | Valid transaction types | Category of transaction under review. |
| `tx_amount` | `NUMERIC(18, 2)` | No | `>= 0.00` | Exposure value under investigation. |
| `case_created_at` | `TIMESTAMP WITH TIME ZONE` | No | ISO 8601 string | Creation timestamp of the investigation ticket. |
| `case_status` | `VARCHAR(32)` | No | `OPEN`, `UNDER_REVIEW`, `CLOSED_CONFIRMED_FRAUD`, `CLOSED_FALSE_POSITIVE` | Current lifecycle state of the ticket. |
| `priority` | `VARCHAR(16)` | No | `P1_CRITICAL`, `P2_HIGH`, `P3_STANDARD` | Urgency tier for investigator queue sorting and SLA monitoring. |
| `assigned_team` | `VARCHAR(64)` | No | `Tier 1 Operations`, `Senior Fraud Risk`, `Special Investigations Unit (SIU)` | Operational squad assigned to resolve ticket. |
| `investigation_notes` | `TEXT` | Yes | Free text | Qualitative notes by analysts. Contains intentional defects for text cleaning. |
| `closed_at` | `TIMESTAMP WITH TIME ZONE` | Yes | ISO 8601 string or `NULL` | Timestamp of final case resolution. `NULL` for ongoing or open cases. |

---

## 4. Known Data Limitations & Modeling Constraints

1. **Merchant Balances (`nameDest` starting with `M`)**: In PaySim, merchant accounts do not track `oldbalanceDest` and `newbalanceDest` (consistently recorded as `0.00`). Balance delta features must account for merchant destinations to avoid artificial feature bias.
2. **Account Identifier Masking**: Identifiers (`C123...`) do not contain demographic, geographical, or device telemetry (e.g., IP address, device fingerprint). All behavioral features must rely on transaction step, frequency, and amount patterns.
3. **Synthetic Ground Truth**: The `isFraud` label reflects programmed heuristic agents in the PaySim simulator. While realistic, real-world fraud schemes evolve dynamically (adversarial drift).
