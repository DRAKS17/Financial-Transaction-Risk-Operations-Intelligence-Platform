# Analytical & Predictive Feature Dictionary

## 1. Overview & Causality Governance

All features in `data/processed/features_engineered.parquet` are strictly **temporally safe**.
- **Zero Future-to-Past Leakage**: No aggregate is computed across future steps or global dataset partitions.
- **Inference Feasibility**: Every feature in this matrix can be calculated in real time at transaction authorization ($T_0$).
- **No Post-Investigation Contamination**: Investigation case outcomes and analyst notes are completely excluded from this matrix.

---

## 2. Feature Definitions & Business Rationale

### A. Exposure & Financial Scale Features
| Feature Name | Type | Formula / Method | Business Rationale & Signal |
|---|---|---|---|
| `amount` | `float64` | Raw transaction value | Direct baseline financial exposure. |
| `log_amount` | `float32` | `log(1 + amount)` | Stabilizes extreme right-skewness ($0 to $92.4M). Trees and linear models benefit from variance stabilization. |
| `amount_to_orig_balance_ratio` | `float32` | `amount / (oldbalanceOrg + 1.0)` | Ratio of transfer size to available sender funds. Ratios $\ge 1.0$ indicate the account is being fully drained or overdrawn. |
| `orig_depletion_complete` | `bool` | `(oldbalanceOrg > 0) & (newbalanceOrig == 0)` | Binary flag indicating 100% balance liquidation. Highly indicative of automated credential takeover. |
| `dest_balance_growth_ratio` | `float32` | `amount / (oldbalanceDest + 1.0)` | Relative surge of incoming funds on recipient ledger (0.0 for merchants). Identifies newly minted money-mule accounts receiving large sums. |

---

### B. Double-Entry Accounting Inconsistency Vectors
| Feature Name | Type | Formula / Method | Business Rationale & Signal |
|---|---|---|---|
| `orig_balance_discrepancy` | `float32` | `(oldbalanceOrg - amount) - newbalanceOrig` | Mathematical variance from expected debit accounting. Deviations highlight automated bot cashouts and overdraft exploitation. |
| `dest_balance_discrepancy` | `float32` | `(oldbalanceDest + amount) - newbalanceDest` | Mathematical variance on recipient side (0.0 for merchants). |
| `is_zero_balance_orig` | `bool` | `oldbalanceOrg == 0.0` | Transaction initiated with empty origin balance. Common in overdraft exploitation or synthetic identity testing. |
| `is_zero_balance_dest` | `bool` | `oldbalanceDest == 0.0` | Recipient account possessed no prior balance. |

---

### C. Modality & Counterparty Typology Features
| Feature Name | Type | Formula / Method | Business Rationale & Signal |
|---|---|---|---|
| `is_transfer` | `bool` | `type == 'TRANSFER'` | Fund exfiltration channel (532,909 transactions; 4,097 frauds). |
| `is_cash_out` | `bool` | `type == 'CASH_OUT'` | Liquidation channel (2,237,500 transactions; 4,116 frauds). |
| `is_transfer_or_cashout` | `bool` | `type in ['TRANSFER', 'CASH_OUT']` | Primary high-risk rail. Encapsulates 100% of ground-truth fraud occurrences. |
| `is_payment` | `bool` | `type == 'PAYMENT'` | Retail merchant payment (0% fraud prevalence). |
| `is_cash_in` | `bool` | `type == 'CASH_IN'` | Inbound deposit rail (0% fraud prevalence). |
| `is_debit` | `bool` | `type == 'DEBIT'` | Account-to-account debit (0% fraud prevalence). |
| `is_merchant_dest` | `bool` | `nameDest.startswith('M')` | Flags external merchant terminal counterparties. |

---

### D. Temporal & Diurnal Cyclical Features
| Feature Name | Type | Formula / Method | Business Rationale & Signal |
|---|---|---|---|
| `step` | `int16` | Chronological hour (1 to 744) | Sequential time index used for temporal train/test splitting. |
| `day` | `int8` | `(step - 1) // 24 + 1` (1 to 31) | Day of monthly simulation cycle. |
| `hour_of_day` | `int8` | `(step - 1) % 24` (0 to 23) | Hour of daily operating cycle. |
| `is_night_hours` | `bool` | `hour_of_day < 6` | Flags off-peak night hours (00:00 to 05:59), when automated attacker scripts operate. |
| `sin_hour` | `float32` | `sin(2 * pi * hour / 24)` | Continuous cyclical representation preserving cyclical proximity (23:00 is close to 00:00). |
| `cos_hour` | `float32` | `cos(2 * pi * hour / 24)` | Cosine component of cyclical daily rhythm. |
| `day_of_week` | `int8` | `(day - 1) % 7` (0=Mon to 6=Sun) | Weekly operational periodicity. |
| `is_weekend` | `bool` | `day_of_week >= 5` | Weekend volume vs weekday commercial business. |

---

### E. Recipient History (Temporally Safe Velocity)
| Feature Name | Type | Formula / Method | Business Rationale & Signal |
|---|---|---|---|
| `dest_inbound_prior_count` | `int32` | Cumulative inbound count for `nameDest` prior to transaction $T_0$ | Causal recipient velocity. Identifies recipient accounts acting as high-velocity fund collectors or mule accounts. |

---

### F. Target & Baseline Labels
| Feature Name | Type | Formula / Method | Role |
|---|---|---|---|
| `isFraud` | `int8` | Simulated ground-truth label (1/0) | Primary supervised classification target (8,213 frauds). |
| `isFlaggedFraud` | `int8` | Legacy heuristic rule flag (1/0) | Baseline benchmark: single transfer > 200,000 (16 flags). |
