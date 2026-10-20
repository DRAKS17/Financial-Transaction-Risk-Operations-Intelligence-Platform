# Data Source Strategy & Candidate Evaluation

## 1. Candidate Comparison & Evaluation Matrix

To satisfy the technical, analytical, and interview-readiness goals of the platform, two leading financial transaction benchmarks were evaluated:

| Evaluation Dimension | Option A: IBM AMLSim | Option B: PaySim (Lopez-Rojas et al.) | Decision / Assessment |
|---|---|---|---|
| **Domain Scope** | Anti-Money Laundering (AML) transaction networks & layering | High-velocity mobile money & payment platform transactions | PaySim aligns closely with payment platforms (Paytm, UPI) and retail banking (HSBC). |
| **Record Grain** | Transaction-level with network topology | Transaction-level with origin/destination account balances | Both provide clean transaction grain. |
| **Available Schema** | `timestamp`, `source_account`, `target_account`, `amount`, `is_sar` | `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`, `isFlaggedFraud` | PaySim provides rich pre- and post-transaction balance states, crucial for financial logic validation. |
| **Ground-Truth Label** | Synthetic SAR (Suspicious Activity Report) flags | `isFraud` (1 = verified fraud, 0 = legitimate) + `isFlaggedFraud` (heuristic baseline) | PaySim provides both a validated binary target and a legacy rule baseline for performance comparison. |
| **Dataset Scale** | Parameterized (typically 10K - 1M) | 6,362,620 transactions (744 hours = 30-day temporal window) | PaySim offers scale, realistic 0.129% fraud prevalence, and temporal depth. |
| **Setup & Dependencies** | Requires Java 8+, Gradle, and compilation of the simulator engine | Direct single CSV download (~493 MB uncompressed, ~180 MB zipped) | PaySim is easily reproducible across standard Python/PostgreSQL environments without JVM overhead. |
| **Licensing** | Apache 2.0 (IBM Research) | CC BY-SA 4.0 (Creative Commons) | Both allow open research and educational portfolio use. |

---

## 2. Primary Selection: PaySim (Mobile Money Fraud Dataset)

### Rationale
1. **Industry Realism**: Simulates 30 days of actual mobile-money logs from an African financial operator, capturing real payment platform flows (`TRANSFER`, `CASH_OUT`, `PAYMENT`, `CASH_IN`, `DEBIT`).
2. **Deterministic Temporal Splitting**: The `step` column (1 hour increments from 1 to 744) enables strict time-based train/test splitting, preventing future-to-past data leakage.
3. **Balance Integrity Checks**: Having both origin and destination before/after balances enables engineering advanced financial consistency features and running database integrity tests.
4. **Extreme Class Imbalance**: The 0.129% fraud rate reflects real-world banking environments (needle-in-a-haystack problem), requiring precision-recall optimization rather than misleading accuracy metrics.

---

## 3. Unstructured Operational Component: Synthetic Investigation Cases

To evaluate risk operations, turnaround time (TAT), case aging, and text normalization, we design a complementary synthetic operational dataset:
- **Entity**: `fact_investigations`
- **Link**: Foreign key `linked_transaction_id` referencing PaySim transactions flagged by the model or heuristic baseline.
- **Fields**:
  - `case_id`: Unique investigation identifier (e.g. `CASE-2026-0001`).
  - `linked_transaction_id`: Identifier of the flagged transaction.
  - `case_created_at`: Calculated from transaction `step`.
  - `priority`: `P1_CRITICAL`, `P2_HIGH`, `P3_STANDARD`.
  - `case_status`: `OPEN`, `IN_PROGRESS`, `CLOSED_CONFIRMED_FRAUD`, `CLOSED_FALSE_POSITIVE`.
  - `assigned_team`: `Tier 1 Operations`, `Senior Fraud Risk`, `Special Investigations Unit (SIU)`.
  - `investigation_notes`: Unstructured text with controlled data quality challenges (inconsistent whitespace, mixed casing, noise tokens, free-text keywords).
  - `closed_at`: Timestamp for turnaround time and SLA calculation.

---

## 4. Acquisition & Storage Protocol

1. **Preservation**: The original transaction download is placed into `data/raw/` and treated as strictly immutable (read-only).
2. **Version Control**: Raw data files are excluded via `.gitignore` to prevent repository bloat while preserving reproducible download instructions.
3. **Data Integrity**: SHA-256 checksums are calculated and logged upon initial download to verify artifact integrity.
