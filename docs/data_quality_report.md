# Data Quality Profiling & Audit Report

**Generated**: 2026-10-21 04:48:29 UTC  
**Source Scope**: Primary Transaction Stream (`paysim.csv`) and Operational Case Logs (`investigation_cases.csv`)  
**Audit Purpose**: Systematic data quality profiling prior to schema staging, SQL modeling, and machine learning.

---

## 1. Executive Summary & Quality Scorecard

| Dimension | Primary Transactions (`paysim.csv`) | Operational Cases (`investigation_cases.csv`) | Assessment |
|---|---|---|---|
| **Total Records** | 6,362,620 | 5,000 | Full scale verified |
| **Missing Values** | 0 nulls across all 11 columns (0.00%) | `closed_at`: 1,142 (expected for open cases) | Excellent completeness |
| **Duplicate Rows** | 0 exact duplicate records | 0 exact duplicate records | Clean transaction grain |
| **Target Distribution** | 8,213 frauds (0.1291%) | P1: 3,171 | Real-world severe class imbalance |
| **Heuristic Alerts** | 16 flagged (0.000251%) | Closed: 3,858 | Demonstrates naive rule limitations |
| **Integrity Checks** | High balance delta discrepancy | 100% referential integrity | Documented behavioral anomaly |

---

## 2. Transaction Stream Quality Profiling (`paysim.csv`)

### A. Completeness & Schema Conformity
- **Total Records Evaluated**: 6,362,620
- **Column Count**: 11
- **Null Distribution**: Every single column in the transaction dataset has **0 null values**.
- **Negative Amount Anomaly**: **0 records** contain negative transaction amounts.
- **Zero-Amount Transactions**: **16 transactions** have an amount of exactly 0.00 (indicative of zero-value probe or ping attempts on payment rails).

### B. Statistical Distribution of Transaction Amounts
- **Minimum Amount**: $0.00
- **25th Percentile (p25)**: $13,200.45
- **Median (p50)**: $74,700.06
- **Mean Amount**: $179,861.90
- **75th Percentile (p75)**: $208,666.66
- **95th Percentile (p95)**: $509,536.17
- **99th Percentile (p99)**: $1,655,085.34
- **Maximum Amount**: $92,445,516.64
- **Total Gross Transaction Value (GTV)**: $1,144,392,944,759.77

> [!NOTE]
> The mean amount ($179,861.90) is significantly higher than the median ($74,700.06), exhibiting extreme positive right-skewness. This is standard in payment platforms where millions of small retail transactions are mixed with commercial transfers.

### C. Channel Distribution & Fraud Concentration
Fraud is strictly concentrated in specific modalities:

| Transaction Type | Total Transactions | % of Total Volume | Fraud Transactions | Fraud Rate | Heuristic Flagged |
|---|---|---|---|---|---|
| **CASH_OUT** | 2,237,500 | 35.17% | 4,116 | 0.1840% | 0 |
| **PAYMENT** | 2,151,495 | 33.81% | 0 | 0.0000% | 0 |
| **CASH_IN** | 1,399,284 | 21.99% | 0 | 0.0000% | 0 |
| **TRANSFER** | 532,909 | 8.38% | 4,097 | 0.7688% | 16 |
| **DEBIT** | 41,432 | 0.65% | 0 | 0.0000% | 0 |

> [!IMPORTANT]
> **Key Domain Finding**: Out of the 5 transaction types, fraudulent activity occurs **exclusively in `TRANSFER` (4,097) and `CASH_OUT` (4,116)**. Not a single fraudulent event occurs in `PAYMENT`, `CASH_IN`, or `DEBIT`. This structural domain rule will directly inform our modeling and ETL filtering in subsequent phases.

### D. Financial Balance Integrity Analysis

In a standard double-entry or debit/credit ledger, balance arithmetic is governed by fundamental accounting conservation:
- **Origin Account**: `oldbalanceOrg - amount == newbalanceOrig`
- **Destination Account**: `oldbalanceDest + amount == newbalanceDest`

Our audit revealed notable behavioral discrepancies:
1. **Origin Balance Discrepancies**:
   - In **3,678,407 transactions (57.81%)**, the post-transaction balance does not match `oldbalanceOrg - amount`.
   - *Root Cause Analysis*: In unauthorized account takeovers and automated cash-out attacks, fraudsters drain the maximum available balance regardless of requested amounts, resulting in `newbalanceOrig == 0.0` even when `amount > oldbalanceOrg`. Additionally, overdraft facilities allow accounts to hit zero without balance subtraction matching the nominal transfer amount.
2. **Origin Zero-Balance Initiation**:
   - **2,102,449 transactions (33.04%)** were initiated by accounts with an initial recorded balance of `0.00`.
3. **Destination Merchant Accounts**:
   - **2,151,495 transactions (33.81%)** were routed to merchant counterparties (`nameDest` starting with `M`).
   - For all merchant accounts, `oldbalanceDest` and `newbalanceDest` are recorded as `0.00`. This is an intentional simulation feature reflecting external merchant settlement accounts where internal bank balances are untracked.

### E. Financial Exposure Comparison
- **Total Value Flagged by Ground Truth Fraud**: $12,056,415,427.84
- **Total Benign Transaction Value**: $1,132,336,529,331.93
- **Mean Fraudulent Transaction**: $1,467,967.30
- **Mean Benign Transaction**: $178,197.04

> [!TIP]
> Fraudulent transactions have an average ticket size of $1,467,967.30, which is **8.2x higher** than benign transactions ($178,197.04). Fraudsters exhibit a strong propensity to maximize transfer velocity and payload size per compromise event.

---

## 3. Operational Case Quality Profiling (`investigation_cases.csv`)

### A. Case Status & Operational Allocation
- **Total Investigation Cases**: 5,000
- **Referential Integrity**: **100% Valid** (Every `linked_transaction_id` maps to a legitimate transaction).

| Metric | Breakdown |
|---|---|
| **Case Status** | CLOSED_FALSE_POSITIVE: 2,656, CLOSED_CONFIRMED_FRAUD: 1,202, UNDER_REVIEW: 701, OPEN: 441 |
| **Case Priority** | P1_CRITICAL: 3,171, P3_STANDARD: 1,107, P2_HIGH: 722 |
| **Assigned Squads** | Senior Fraud Risk: 1,925, Special Investigations Unit (SIU): 1,599, Tier 1 Operations: 1,476 |

### B. Unstructured Text Quality Defects Identified
The free-text qualitative field `investigation_notes` contains controlled real-world data defects requiring cleaning in Phase 5:
- **Missing / Null Notes**: 887 cases (17.7%)
- **Empty / Whitespace-Only Notes**: 887 cases
- **Irregular Leading / Trailing Whitespace**: 403 cases
- **All-Caps Investigation Notes**: 428 cases

---

## 4. Key Takeaways & Recommendations for Phase 4 & 5

1. **No Imputation Required for Numerical Amounts or Types**: Since missing value rates are 0.00% across all numerical fields in `paysim.csv`, no synthetic mean/median imputation is needed.
2. **Preserve Extreme Outliers**: Large transactions (up to \$92,445,516.64) must **not** be clipped or discarded. In financial risk analysis, extreme transaction amounts represent the primary exposure events.
3. **Feature Engineering on Balance Inconsistencies**: The violation of `oldbalanceOrg - amount == newbalanceOrig` is not merely noise; it is a powerful predictive feature that distinguishes automated draining from standard retail shopping.
4. **Text Normalization Engine Required**: In Phase 5, the text pipeline must standardize case casing, strip irregular whitespace, impute explicit missing note sentinels (e.g. `'[NO NOTES RECORDED]'`), and extract keyword indicators without data leakage.
