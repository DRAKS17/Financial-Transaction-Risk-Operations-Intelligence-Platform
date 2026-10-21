# PostgreSQL Database Loading & Integrity Validation Report

**Database**: `financial_risk_analytics` on `localhost:5432`  
**Warehouse Architecture**: Staging (`stg`), Dimensional Star Schema (`core`), Analytical Marts (`mart`)  
**Validation Status**: **PASSED (All 8 Integrity Assertions Passed)**  
**Validated At**: 2026-10-21 06:58:45 UTC  

---

## 1. Executive Warehouse Load Scorecard

| Target Table / View | Schema | Expected Records | Loaded Records | Verification Status |
|---|---|---|---|---|
| `dim_date` | `core` | 744 hours (31 days) | **744** | ✅ PASSED |
| `dim_transaction_type` | `core` | 5 modalities | **5** | ✅ PASSED |
| `dim_case_status` | `core` | 4 statuses | **4** | ✅ PASSED |
| `dim_priority` | `core` | 3 tiers | **3** | ✅ PASSED |
| `fact_transactions` | `core` | 6,362,620 rows | **6,362,620** | ✅ PASSED (Exact Match) |
| `fact_investigations` | `core` | 5,000 cases | **5,000** | ✅ PASSED (Exact Match) |
| `mart_risk_executive_overview` | `mart` | Active View | **152** aggregate rows | ✅ PASSED |
| `mart_investigation_operations` | `mart` | Active View | **20** aggregate rows | ✅ PASSED |
| `mart_channel_exposure` | `mart` | Active View | **5** channel rows | ✅ PASSED |

---

## 2. Financial Parity & Metric Reconciliation

- **Gross Transaction Value (GTV)**: **$1,144,392,944,759.77** (Exact parity with raw ledger, variance: $0.00).
- **Ground-Truth Fraud Volume**: **8,213 frauds** (Prevalence: 0.1291%).
- **Legacy Heuristic Rule Flags**: **16 flags**.
- **Referential Integrity**: **0 orphan records** (Every single `linked_transaction_id` in `fact_investigations` references an active `fact_transactions` row).
- **Active Operational Backlog**: **1,142 open cases** (22.8%).
- **Closed Cases**: **3,858 closed cases**.
- **Operational SLA Breaches**: **1,384 breached cases** (35.9% breach rate).

---

## 3. High-Throughput Streaming COPY Strategy

Standard row-by-row or ORM insertions (`DataFrame.to_sql`) on 6.36 million rows take 25 to 45 minutes and incur substantial memory locks.
Our pipeline implements **PostgreSQL streaming `COPY` with `io.StringIO` buffers**:
- Loads in chunks of 500,000 records.
- Directly pushes raw CSV streams through PostgreSQL's internal binary parser.
- Entire 6.36M transaction fact table loaded in **under 45 seconds**.
