# Data Cleaning & Transformation Reconciliation Report

**Pipeline Execution**: Automated Batch Normalization Engine  
**Execution Duration**: 12.22 seconds  
**Source Dataset**: `data/raw/paysim.csv` (470.67 MB)  
**Cleaned Target**: `data/processed/transactions_cleaned.parquet` (345.74 MB, Compression: 1.36x)  

---

## 1. Executive Reconciliation Scorecard

| Dimension | Raw Input | Cleaned Output | Variance | Verification Status |
|---|---|---|---|---|
| **Total Record Count** | 6,362,620 | 6,362,620 | 0 | ✅ PASSED (Exact Match) |
| **Ground-Truth Frauds (`isFraud`)** | 8,213 | 8,213 | 0 | ✅ PASSED (Zero Label Drift) |
| **Heuristic Alerts (`isFlaggedFraud`)** | 16 | 16 | 0 | ✅ PASSED (Exact Match) |
| **Gross Transaction Value ($)** | $1,144,392,944,759.77 | $1,144,392,944,759.77 | $0.00 | ✅ PASSED (Exact Monetary Parity) |
| **Primary Key Integrity** | Inferred | 1 to 6,362,620 (`transaction_id`) | N/A | ✅ PASSED (Surrogate Key Generated) |

---

## 2. Transformation Rules & Senior Analytics Rationale

### A. Surrogate Primary Key Assignment (`transaction_id`)
- **Action**: Assigned monotonic integer surrogate keys `1` to `6,362,620`.
- **Rationale**: Natural composite keys `(step, nameOrig, nameDest, amount)` in high-throughput simulated payment streams exhibit edge collisions. Assigning an explicit surrogate key ensures strict referential integrity with our companion operational cases table (`linked_transaction_id`).

### B. Calendar Temporal Normalization (`day`, `hour_of_day`)
- **Action**: Decomposed continuous step index (`step` 1 to 744) into:
  - `day = (step - 1) // 24 + 1` (integer 1 to 31, representing 1 month of operations).
  - `hour_of_day = (step - 1) % 24` (integer 0 to 23, representing diurnal cycles).
- **Rationale**: Fraud patterns display pronounced cyclical and diurnal rhythms (e.g., automated draining during off-peak night hours). Deriving modular hours allows downstream SQL and ML models to detect temporal surges.

### C. Counterparty Entity Classification (`dest_type`)
- **Action**: Segmented `nameDest` into `'MERCHANT'` (where `nameDest` starts with `M`) and `'CUSTOMER'` (starts with `C`).
- **Breakdown**:
  - Customer Peer Transfers: **4,211,125 (66.19%)**
  - Merchant Payments: **2,151,495 (33.81%)**
- **Rationale**: Merchant accounts do not maintain balances in the raw source (`oldbalanceDest = 0.00` and `newbalanceDest = 0.00`). Explicitly typing counterparty entities prevents algorithms from treating merchant zero balances as customer balance anomalies.

### D. Financial Accounting Delta & Inconsistency Flags
- **Action**: Engineered explicit delta vectors:
  - `orig_balance_delta = oldbalanceOrg - newbalanceOrig`
  - `dest_balance_delta = newbalanceDest - oldbalanceDest`
  - `is_zero_balance_orig = oldbalanceOrg == 0.0` (2,102,449 accounts)
  - `is_zero_balance_dest = oldbalanceDest == 0.0` (2,704,388 accounts)
  - `is_orig_balance_err`: Flagged **3,678,407 transactions (57.81%)** where outbound transfers violate `abs((oldbalanceOrg - amount) - newbalanceOrig) > 0.01`.
- **Rationale**: Rather than discarding or artificially coercing accounts that violate accounting balances, we encapsulate this behavior into boolean flags. In account takeover and SIM-swap fraud, attacker bots empty balances to zero regardless of requested ticket size.

### E. Storage Optimization via Apache Parquet
- **Action**: Serialized normalized records to Apache Parquet with Snappy compression.
- **Storage Impact**: Compressed from **470.67 MB (raw CSV)** down to **345.74 MB (processed Parquet)**.
- **Compression Ratio**: **1.36x**.
- **Performance**: Columnar Parquet enables predicate pushdown and sub-second SQL / Pandas filtering without re-parsing raw text.

---

## 3. Strict Boundary Rules Enforced

1. **Zero Row Truncation**: No rows were dropped. All 6,362,620 transactions are preserved.
2. **Zero Label Modification**: Ground-truth `isFraud` (8,213) and heuristic `isFlaggedFraud` (16) counts are identical before and after. We did not downsample, oversample, or manipulate labels.
3. **Zero Outlier Removal**: High-value transactions (up to $92.4M) were fully preserved.
4. **Physical Storage Separation**: Raw data in `data/raw/paysim.csv` was untouched; transformed outputs reside strictly in `data/processed/`.
