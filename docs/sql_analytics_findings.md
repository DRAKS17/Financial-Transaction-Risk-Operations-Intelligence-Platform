# Advanced SQL Analytics & Risk Typology Findings Report

**Document Reference**: `docs/sql_analytics_findings.md`  
**Execution Environment**: PostgreSQL 18.0 Relational Engine on `localhost:5432`  
**Database**: `financial_risk_analytics` (Schemas: `stg`, `core`, `mart`)  
**Data Scale Processed**: 6,362,620 transactions ($1,144,392,944,759.77 GTV) + 5,000 Investigation Case Tickets  
**Automation Code**: [`src/financial_risk/analytics.py`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/src/financial_risk/analytics.py)  
**Production SQL**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql)  
**Test Suite**: [`tests/test_sql_analytics.py`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/tests/test_sql_analytics.py) (52/52 tests passing)  

---

## 1. Executive Summary & Portfolio Baseline

This report documents the advanced SQL analytical investigations executed directly against the 6.36 million row dimensional warehouse. The analytical engine solves core surveillance and operations questions using industry-standard SQL typologies: multi-stage Common Table Expressions (CTEs), multi-tier analytic window functions (`ROW_NUMBER`, `DENSE_RANK`, `LAG`, `PERCENTILE_CONT`, `PERCENT_RANK`, `NTILE`), sliding time window aggregations, and categorical confusion matrix modeling.

### Global Portfolio Summary
| Financial Metric | Portfolio Total | Fraud Component | Benign Component | Fraud Share / Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Transaction Volume** | 6,362,620 transactions | 8,213 transactions | 6,354,407 transactions | 0.1291% (1 in 775 tx) |
| **Gross Transaction Value (GTV)** | $1,144,392,944,759.77 ($1.144 Trillion) | $12,056,415,427.84 ($12.056 Billion) | $1,132,336,529,331.93 ($1.132 Trillion) | 1.0535% of GTV |
| **Mean Ticket Size** | $179,861.90 | $1,467,967.30 | $178,197.04 | **8.24x higher for fraud** |
| **Max Single Transaction** | $92,445,516.64 | $10,000,000.00 | $92,445,516.64 | High-value wire cap |
| **Legacy Flagged Fraud (`isFlaggedFraud = 1`)**| 16 transactions | 16 transactions | 0 transactions | **0.1948% recall** |
| **Operational Backlog** | 5,000 cases | 3,858 closed cases | 1,142 open backlog | 35.87% SLA breach rate |

---

## 2. Deep Dive: Analytical SQL Queries & Financial Typologies

### Query 1: Temporal Risk Dynamics & 24-Hour Rolling Fraud Exposure Window
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L25-L95)
- **Business Question**: How does fraud volume and monetary exposure evolve over simulation time, and what are the peak 24-hour rolling burst windows where operational defense capacity was under maximum stress?
- **SQL Patterns**:
  - Two-tier CTE with `core.dim_date` temporal join.
  - Sliding window: `SUM(fraud_exposure_amount) OVER (ORDER BY step ROWS BETWEEN 23 PRECEDING AND CURRENT ROW)`.
  - Unbounded running totals: `SUM(gtv) OVER (ORDER BY step ROWS UNBOUNDED PRECEDING)`.
  - Analytic exposure percentile: `PERCENT_RANK() OVER (ORDER BY rolling_24h_fraud_exposure)`.
- **Query Performance**: Execution time: **2,625 ms** across 6.36M rows.
- **Key Empirical Results**:
  - Across the 743 recorded hourly steps (31 days), the rolling 24-hour fraud exposure averages **$388.9 Million**.
  - Peak 24-hour burst window occurs at **step 719-743** (Day 30-31), where rolling 24-hour fraud count surges to **1,254 fraudulent transactions** and rolling exposure reaches **$1.84 Billion** (99.8th percentile stress level).
  - Benign transaction volume exhibits a sharp weekly oscillation (weekdays average $42B GTV/day vs weekend $18B GTV/day), but fraudulent attacks execute at a continuous, steady pace, causing weekend fraud prevalence rates to jump by **2.3x**.

```
[Simulation Timeline (Steps 1 - 743)]
Step 1 .................................................. Step 743
Cumulative GTV:  $285.4M  ------------------------>  $1,144,392.9M ($1.144 Trillion)
Cumulative Loss: $3.74M    ------------------------>  $12,056.4M ($12.056 Billion)
```

---

### Query 2: Origin Account Rapid Structuring & Layering Sequence
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L97-L165)
- **Business Question**: Are origin accounts executing rapid sequential transactions across channels (e.g. `TRANSFER` immediately followed by `CASH_OUT`) within narrow time deltas ($\le 1$ hour), indicating automated script-driven account compromise or layering?
- **SQL Patterns**:
  - Filtering CTE with `HAVING COUNT(*) > 1` on `orig_account_id`.
  - Multi-column window functions:
    - `LAG(t.step, 1) OVER (PARTITION BY t.orig_account_id ORDER BY t.step, t.transaction_id)`
    - `LAG(t.transaction_type, 1) OVER (...)`
    - `LAG(t.amount, 1) OVER (...)`
    - `LAG(t.newbalance_orig, 1) OVER (...)`
  - Inter-transaction step delta calculation (`step - prev_step`).
- **Query Performance**: Execution time: **5,838 ms**.
- **Key Empirical Results**:
  - There are **9,313 duplicate originator transactions** generated by accounts executing repeated actions.
  - Of repeat transactions, **100% of concurrent executions (step gap = 0 hours)** transition from `TRANSFER` or `CASH_IN` directly into `CASH_OUT`.
  - Example detected entity: Originator `C389626503` executed a `CASH_OUT` of $139,133.13 immediately followed by a `TRANSFER` of $919,487.76 at step 139 with zero hourly latency, completely wiping out the account balance.

---

### Query 3: Recipient Fan-In Network & High-Velocity Mule Collection Hubs
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L167-L215)
- **Business Question**: Which destination accounts receive high inbound transfer velocity from multiple originators, and how much gross capital and fraudulent exposure flows into these potential mule aggregator nodes?
- **SQL Patterns**:
  - Channel filtering: `transaction_type IN ('TRANSFER', 'CASH_OUT')`.
  - Aggregation CTE: `COUNT(transaction_id) > 1`.
  - Window ranking: `DENSE_RANK() OVER (ORDER BY inbound_tx_count DESC, total_inbound_gtv DESC) AS fanin_velocity_rank`.
- **Query Performance**: Execution time: **7,378 ms**.
- **Key Empirical Results**:
  - Top 10 destination collection hubs aggregate over **$1.12 Billion** across 276 transactions.
  - Destination node `C665576141` received **31 inbound transfers** totaling **$72,019,816.20** ($72.02 Million) with a mean ticket size of $2.32M.
  - Destination node `C1789550256` received **25 inbound transfers** totaling **$161,157,696.10** ($161.16 Million) with a mean ticket size of $6.44M.
  - In a financial crime setting (e.g. HSBC AML or Paytm P2P), accounts with $>10$ inbound transfers exceeding $\$10\text{M}$ total inflow without merchant categorization are prioritized as high-confidence money mule networks or shell companies.

---

### Query 4: Channel Vulnerability & Pareto Concentration (Decile Lorenz Curve)
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L217-L270)
- **Business Question**: How concentrated is transaction volume and fraud loss across ticket size deciles? What percentage of total fraud losses is concentrated in the top 10% (Decile 10)?
- **SQL Patterns**:
  - Analytic partitioning: `NTILE(10) OVER (ORDER BY amount ASC) AS amount_decile`.
  - Cumulative percentage sums: `SUM(decile_fraud_exposure) OVER (ORDER BY amount_decile) / SUM(...) OVER () * 100`.
- **Query Performance**: Execution time: **23,327 ms** (sorting 6.36M rows).
- **Empirical Pareto Distribution Table**:

| Decile | Ticket Size Min ($) | Ticket Size Max ($) | Decile GTV ($) | Fraud Tx Count | Fraud Exposure ($) | Fraud Rate (%) | Share of Total Fraud ($) | Cumulative Fraud Loss (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.00 | $4,501.30 | $1,420,747,947.82 | 148 | $250,460.58 | 0.0233% | 0.002% | **0.00%** |
| **2** | $4,501.30 | $9,866.15 | $4,490,762,499.16 | 128 | $931,531.24 | 0.0201% | 0.008% | **0.01%** |
| **3** | $9,866.16 | $18,092.00 | $8,641,516,493.02 | 148 | $2,083,049.32 | 0.0233% | 0.017% | **0.03%** |
| **4** | $18,092.04 | $36,371.32 | $16,431,516,245.53 | 365 | $9,580,485.50 | 0.0574% | 0.080% | **0.11%** |
| **5** | $36,371.37 | $74,871.80 | $34,489,997,745.99 | 617 | $33,894,225.48 | 0.0970% | 0.281% | **0.39%** |
| **6** | $74,872.08 | $122,563.76 | $62,533,156,781.98 | 592 | $58,652,156.56 | 0.0930% | 0.486% | **0.87%** |
| **7** | $122,563.82 | $176,801.88 | $94,738,539,594.14 | 581 | $86,920,593.33 | 0.0913% | 0.721% | **1.60%** |
| **8** | $176,802.01 | $246,611.16 | $133,434,312,009.81 | 503 | $106,516,284.61 | 0.0791% | 0.883% | **2.48%** |
| **9** | $246,611.46 | $365,423.29 | $189,921,104,101.74 | 719 | $219,103,198.06 | 0.1130% | 1.817% | **4.30%** |
| **10** | **$365,423.48** | **$92,445,516.64** | **$598,291,291,340.58** | **4,412** | **$11,538,483,443.16** | **0.6934%** | **95.702%** | **100.00%** |

- **Critical Risk Finding**:
  - The top decile (Decile 10: transactions $\ge \$365,423.48$) represents **52.28% of total portfolio GTV**, but accounts for **95.70% of all gross fraud losses** ($11.538 Billion out of $12.056 Billion).
  - Deciles 1 through 7 combined account for barely **1.60%** of total fraud exposure.
  - This mathematically proves the necessity of **value-tiered monitoring policies** (e.g. Morgan Stanley / HSBC Wire Surveillance): applying strict stepped-up authentication and manual review holds on transactions above $365k protects over 95% of total at-risk capital while impacting only 10% of customers.

---

### Query 5: Operational Investigation Squad Triage & SLA Backlog Aging Matrix
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L272-L345)
- **Business Question**: What is the current backlog aging distribution across teams and priority tiers, which squads have the highest SLA breach risk, and how does turnaround time distribute across the P50, P75, and P90 percentiles?
- **SQL Patterns**:
  - Operational snapshot timestamp anchoring: `MAX(created_at)`.
  - Continuous percentiles: `PERCENTILE_CONT(0.50)` and `PERCENTILE_CONT(0.90)`.
  - Partitioned ranking: `RANK() OVER (PARTITION BY priority ORDER BY mean_closed_tat_hours DESC)`.
  - Four discrete aging buckets: `<24h`, `24-48h`, `48-72h`, `>72h`.
- **Query Performance**: Execution time: **56 ms**.
- **Squad Performance Matrix Table**:

| Assigned Squad | Priority Tier | SLA Target | Total Cases | Open Backlog | Closed Cases | SLA Breaches | SLA Breach Rate (%) | Mean TAT (h) | P50 Median TAT (h) | P90 TAT (h) | Backlog > 72h | Priority TAT Rank |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Senior Fraud Risk** | `P1_CRITICAL` | 24.0 h | 1,572 | 308 | 1,264 | 552 | **43.67%** | 42.47 h | 24.00 h | 120.00 h | 284 | 1 (Slowest) |
| **SIU (Special Investigations)** | `P1_CRITICAL` | 24.0 h | 1,599 | 316 | 1,283 | 562 | **43.80%** | 41.49 h | 24.00 h | 120.00 h | 292 | 2 |
| **Senior Fraud Risk** | `P2_HIGH` | 48.0 h | 353 | 87 | 266 | 80 | **30.08%** | 42.42 h | 24.00 h | 120.00 h | 87 | 1 |
| **Tier 1 Operations** | `P2_HIGH` | 48.0 h | 369 | 109 | 260 | 79 | **30.38%** | 41.06 h | 24.00 h | 120.00 h | 109 | 2 |
| **Tier 1 Operations** | `P3_STANDARD` | 96.0 h | 1,107 | 322 | 785 | 111 | **14.14%** | 40.30 h | 24.00 h | 120.00 h | 322 | 1 |

- **Operational Diagnosis**:
  - The highest risk bottleneck is in `P1_CRITICAL` cases handled by Senior Fraud Risk and SIU, where the SLA breach rate exceeds **43.7%** against a 24-hour target.
  - P90 turnaround time is **120.0 hours (5 full days)** across all priority tiers, revealing a long-tail operational drag.
  - Of 1,142 open cases in the backlog, **1,094 cases (95.8%) are currently aged $>72$ hours**, posing severe regulatory and recovery failure risks (funds cannot be recalled after 72 hours).

---

### Query 6: Legacy Rule Failure & Detection Efficacy Audit (`is_flagged_fraud`)
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L347-L395)
- **Business Question**: What is the mathematical and monetary shortfall of the bank's legacy static rule (flagging `TRANSFER` $> 200,000$ as `is_flagged_fraud`)? How much unmitigated fraud exposure slipped through the legacy defense perimeter?
- **SQL Patterns**:
  - Complete categorical confusion matrix: TP, FP, FN, TN.
  - Monetary loss aggregation and cross-join with portfolio totals.
- **Query Performance**: Execution time: **2,153 ms**.
- **Confusion Matrix & Audit Table**:

| Outcome Category | Channel | Tx Count | Gross Exposure ($) | Mean Ticket ($) | % of Fraud Tx | % of Fraud Loss ($) | Detection Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **True Positive (TP)** | `TRANSFER` | **16** | **$77,785,563.69** | $4,861,597.73 | **0.1948%** | **0.6452%** | Successfully flagged by legacy rule |
| **False Positive (FP)**| *All Channels* | **0** | **$0.00** | $0.00 | 0.0000% | 0.0000% | No benign transactions flagged |
| **False Negative (FN)**| `TRANSFER` | **4,081** | **$5,939,531,041.52** | $1,455,410.69 | 49.6895% | 49.2645% | **Missed Fraud: Transfer bypass** |
| **False Negative (FN)**| `CASH_OUT` | **4,116** | **$6,039,098,822.63** | $1,467,225.18 | 50.1157% | 50.0903% | **Missed Fraud: Rule blind to Cash Out**|
| **Total False Negatives**| *Combined* | **8,197** | **$11,978,629,864.15** | $1,461,343.16 | **99.8052%** | **99.3548%** | **$11.98 Billion unmitigated exposure** |
| **True Negative (TN)** | *All Channels* | 6,354,407 | $1,132,336,529,331.93 | $178,197.04 | N/A | N/A | Correctly cleared benign traffic |

- **Audit Findings for Risk Committee**:
  1. **Precision = 100.0%**: Whenever the legacy rule flags a transaction, it is genuine fraud ($16 / 16$).
  2. **Recall = 0.1948%**: The rule captures only 16 out of 8,213 frauds. **It misses 99.8% of all fraudulent activity.**
  3. **Channel Blind Spot**: The legacy rule was hardcoded exclusively for `TRANSFER` and completely ignored `CASH_OUT`. Consequently, 100% of cash-out fraud ($6.04 Billion) passed through without any alert.
  4. **Monetary Shortfall**: The rule captured $77.79M, but allowed **$11.978 Billion** in fraudulent transactions to settle undetected.
  5. **Strategic Mandate**: This empirical audit provides the core business justification for Phase 11-13 (Machine Learning Scoring System). An interpretable XGBoost/LightGBM model with probabilistic thresholding is required to replace brittle static rules.

---

### Query 7: Account Draining / Balance Depletion Risk Patterns
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L397-L445)
- **Business Question**: What proportion of fraudulent transfers and cash-outs execute complete account draining (`newbalance_orig = 0`), and how does the fraud rate differ across balance depletion states?
- **SQL Patterns**:
  - State classification conditional aggregation.
  - Partitioned window shares across channel types.
- **Query Performance**: Execution time: **2,155 ms**.
- **Depletion State Breakdown Table**:

| Channel | Balance Depletion State | Total Transactions | Channel GTV ($) | Fraud Tx Count | Fraud Exposure ($) | Fraud Rate (%) | Share of Channel Fraud Count (%) | Share of Channel Fraud Loss (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CASH_OUT` | **Complete Draining (Balance = 0)** | 959,412 | $198,736,039,465.01 | **4,078** | $5,984,235,874.09 | **0.4251%** | **99.08%** | **99.09%** |
| `CASH_OUT` | Zero Initial Balance Initiation | 1,025,783 | $173,025,503,228.02 | 21 | $4,924,306.92 | 0.0020% | 0.51% | 0.08% |
| `CASH_OUT` | Other / Inflow Increase | 16 | $0.00 | 16 | $0.00 | 100.0000% | 0.39% | 0.00% |
| `CASH_OUT` | Partial Depletion | 252,285 | $22,651,082,350.09 | 1 | $42,062.82 | 0.0004% | 0.02% | 0.00% |
| `CASH_OUT` | Accounting Discrepancy Error | 4 | $370,181.37 | 0 | $0.00 | 0.0000% | 0.00% | 0.00% |
| `TRANSFER` | **Complete Draining (Balance = 0)** | 228,662 | $217,159,520,624.33 | **3,934** | $4,563,822,403.36 | **1.7204%** | **96.02%** | **75.85%** |
| `TRANSFER` | Partial Depletion | 21,447 | $4,537,101,546.82 | 143 | $1,420,994,453.20 | 0.6668% | 3.49% | 23.62% |
| `TRANSFER` | Accounting Discrepancy Error | 17 | $77,954,670.66 | 16 | $77,785,563.69 | 94.1176% | 0.39% | 1.29% |
| `TRANSFER` | Zero Initial Balance Initiation | 282,783 | $263,517,410,421.36 | 4 | $4,610,763.76 | 0.0014% | 0.10% | 0.08% |

- **Key Analytical Insight**:
  - Across both high-risk channels, **8,012 out of 8,213 frauds (97.55%) execute complete balance draining**.
  - For `CASH_OUT`, 99.08% of fraud drains the origin account completely to $0.00.
  - For `TRANSFER`, 96.02% of fraud drains the origin account completely to $0.00.
  - This validates that our Phase 6 engineered feature [`orig_depletion_complete`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/src/financial_risk/feature_engineering.py) and [`amount_to_orig_balance_ratio`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/src/financial_risk/feature_engineering.py) are primary features for predictive modeling.

---

### Query 8: Diurnal / Hourly Fraud Vulnerability & Off-Hours Risk Surge
- **SQL File**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql#L447-L495)
- **Business Question**: How does transaction risk fluctuate across the 24-hour diurnal cycle? Are nocturnal off-hours (00:00 - 06:00) disproportionately targeted by fraudsters, and what is the exact signal-to-noise multiplier during the night?
- **SQL Patterns**:
  - Star schema aggregation on `core.dim_date(hour_of_day, is_night_hours)`.
  - Window ranking: `DENSE_RANK() OVER (ORDER BY fraud_rate DESC)`.
  - Portfolio multiplier: `(hourly_fraud_rate) / (portfolio_average_fraud_rate)`.
- **Query Performance**: Execution time: **958 ms**.
- **Diurnal Profile Table (Hourly Breakdown)**:

| Hour of Day | Time Window | Night Hours? | Total Transactions | Total GTV ($) | Fraud Tx Count | Fraud Loss ($) | Fraud Prevalence (%) | Risk Rank | Share of All Fraud (%) | Share of Volume (%) | Risk Multiplier vs Baseline |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **00:00** | Night (00-06) | TRUE | 27,111 | $3,200,401,133.58 | 358 | $366,685,648.61 | **1.3205%** | 8 | 4.36% | 0.43% | **10.23x** |
| **01:00** | Night (00-06) | TRUE | 9,018 | $1,555,136,461.33 | 372 | $700,179,276.58 | **4.1251%** | 5 | 4.53% | 0.14% | **31.96x** |
| **02:00** | Night (00-06) | TRUE | 2,007 | $917,807,529.16 | 326 | $628,560,056.95 | **16.2431%** | 3 | 3.97% | 0.03% | **125.84x** |
| **03:00** | Night (00-06) | TRUE | 1,241 | $452,670,433.94 | 274 | $340,582,284.76 | **22.0790%** | 2 | 3.34% | 0.02% | **171.05x** |
| **04:00** | Night (00-06) | TRUE | 1,641 | $637,355,461.27 | 366 | $502,723,699.68 | **22.3035%** | **1 (Peak)** | 4.46% | 0.03% | **172.79x** |
| **05:00** | Night (00-06) | TRUE | 3,420 | $733,140,777.32 | 358 | $446,186,370.27 | **10.4678%** | 4 | 4.36% | 0.05% | **81.09x** |
| **06:00** | Morning (06-12) | FALSE | 8,988 | $1,342,656,886.21 | 328 | $330,311,639.84 | 3.6493% | 6 | 3.99% | 0.14% | 28.27x |
| **07:00** | Morning (06-12) | FALSE | 26,915 | $4,771,771,726.72 | 368 | $621,345,818.05 | 1.3673% | 7 | 4.48% | 0.42% | 10.59x |
| **08:00** | Morning (06-12) | FALSE | 283,518 | $57,230,312,371.27 | 341 | $413,074,885.38 | 0.1203% | 13 | 4.15% | 4.46% | 0.93x |
| **09:00** | Morning (06-12) | FALSE | 425,729 | $90,728,860,570.56 | 375 | $688,339,630.73 | 0.0881% | 14 | 4.57% | 6.69% | 0.68x |
| **10:00** | Morning (06-12) | FALSE | 445,992 | $86,836,289,140.77 | 324 | $362,885,498.68 | 0.0726% | 20 | 3.94% | 7.01% | 0.56x |
| **11:00** | Morning (06-12) | FALSE | 483,418 | $95,365,234,623.53 | 339 | $457,632,561.84 | 0.0701% | 21 | 4.13% | 7.60% | 0.54x |
| **12:00** | Afternoon (12-18) | FALSE | 499,310 | $94,942,668,293.75 | 346 | $481,215,920.91 | 0.0693% | 22 | 4.21% | 7.85% | 0.54x |
| **13:00** | Afternoon (12-18) | FALSE | 489,613 | $92,593,296,819.38 | 326 | $453,745,715.19 | 0.0666% | 23 | 3.97% | 7.70% | 0.52x |
| **14:00** | Afternoon (12-18) | FALSE | 473,733 | $88,719,533,707.99 | 334 | $427,358,013.91 | 0.0705% | 19 | 4.07% | 7.45% | 0.55x |
| **15:00** | Afternoon (12-18) | FALSE | 465,584 | $86,929,481,749.61 | 350 | $515,648,154.26 | 0.0752% | 18 | 4.26% | 7.32% | 0.58x |
| **16:00** | Afternoon (12-18) | FALSE | 467,235 | $87,511,634,682.38 | 348 | $530,135,163.66 | 0.0745% | 17 | 4.24% | 7.34% | 0.58x |
| **17:00** | Afternoon (12-18) | FALSE | 471,080 | $87,908,582,347.16 | 362 | $576,013,446.54 | 0.0768% | 16 | 4.41% | 7.40% | 0.60x |
| **18:00** | Evening (18-24) | FALSE | 476,389 | $89,129,566,381.16 | 366 | $548,220,131.06 | 0.0768% | 15 | 4.46% | 7.49% | 0.60x |
| **19:00** | Evening (18-24) | FALSE | 462,096 | $86,515,550,227.08 | 341 | $573,959,738.92 | 0.0738% | 12 | 4.15% | 7.26% | 0.57x |
| **20:00** | Evening (18-24) | FALSE | 440,698 | $81,387,978,811.23 | 313 | $453,952,192.11 | 0.0710% | 24 (Lowest) | 3.81% | 6.93% | 0.55x |
| **21:00** | Evening (18-24) | FALSE | 373,729 | $66,616,160,940.38 | 342 | $510,757,678.93 | 0.0915% | 11 | 4.16% | 5.87% | 0.71x |
| **22:00** | Evening (18-24) | FALSE | 266,039 | $43,770,391,339.69 | 345 | $490,443,391.24 | 0.1297% | 10 | 4.20% | 4.18% | 1.00x |
| **23:00** | Evening (18-24) | FALSE | 137,258 | $19,697,642,883.38 | 338 | $548,477,888.75 | 0.2462% | 9 | 4.12% | 2.16% | 1.91x |

- **Nocturnal Attack Vector Breakdown**:
  - The fraud count per hour is **strictly invariant across the 24 hours**: it remains between 313 and 375 transactions every hour, accounting for ~4.1% of total fraud per hour.
  - Legitimate human transaction volume collapses by **99.7%** between 01:00 and 05:00 (dropping from 499,310 tx at 12:00 to just 1,241 tx at 03:00).
  - Because fraud is automated by botnets/scripts executing round-the-clock while human volume declines, the conditional probability of fraud surges to **22.30% at 04:00 (a 172.8x risk multiplier over baseline)**!
  - **Operational Recommendation**: Financial institutions should implement dynamic, time-aware risk rules. At 04:00 AM, stepping up authentication (2FA / biometric selfie / delay on out-of-pattern transfers) achieves extraordinary precision without burdening daytime user experience.

---

## 3. SQL Performance Benchmarking & Query Planner Insights

All 8 queries were benchmarked on a single PostgreSQL instance processing the 6.36 million row dataset.

| Query Identifier | Primary Typology | Execution Time (ms) | Rows Returned | Query Plan Driver (`EXPLAIN ANALYZE`) |
| :--- | :--- | :---: | :---: | :--- |
| `investigation_operations_triage` | CTE + Percentiles + Aging | **56.18 ms** | 5 | Seq Scan + HashAggregate on indexed `core.fact_investigations` |
| `diurnal_hourly_risk` | Star Join + DENSE_RANK | **958.30 ms** | 24 | Parallel Seq Scan (2 workers) + Hash Join on `step` |
| `legacy_rule_failure` | Confusion Matrix Aggregation | **2,153.75 ms** | 3 | Parallel Seq Scan + Partial HashAggregate on `(is_fraud, is_flagged_fraud)` |
| `account_draining_patterns` | Partitioned Window + States | **2,155.96 ms** | 9 | Bitmap Heap Scan on `idx_fact_tx_type` + WindowAgg |
| `temporal_rolling_window` | 24h Sliding Window + Percentiles | **2,625.27 ms** | 743 | Parallel Hash Join + Two-tier WindowAgg (`ROWS 23 PRECEDING`) |
| `origin_structuring_sequence` | Multi-column `LAG()` | **5,838.06 ms** | 50 | Hash Join with multi-tx CTE + Sort by `(orig_account_id, step)` |
| `recipient_fanin_mule_hubs` | Hub Grouping + DENSE_RANK | **7,378.88 ms** | 25 | Filtered Seq Scan on `TRANSFER/CASH_OUT` + Memory HashAggregate |
| `pareto_decile_concentration` | `NTILE(10)` over 6.36M rows | **23,327.09 ms** | 10 | Parallel Seq Scan + External Sort on `amount` + WindowAgg |
| **Total Analytics Suite** | **All 8 Analytical Typologies** | **44,500.00 ms (44.5s)** | **869** | **Full portfolio analytics executed in < 45 seconds** |

---

## 4. Financial Institution Interview Discussion Points

### Paytm (Digital Wallets, UPI, High-Velocity Retail P2P)
- **Key Insight**: UPI and digital wallet frauds rely on high-frequency account draining (`orig_depletion_complete`) and recipient mule networks (`recipient_fanin_mule_hubs`).
- **Interview Response**:  
  *"At Paytm's scale of hundreds of millions of daily transactions, you cannot afford complex graph walks or correlated subqueries in real-time scoring. In our SQL analytics pipeline, we pre-aggregated recipient fan-in degree and identified that collection nodes receiving $\ge 25$ inbound transfers accumulated over $\$160\text{M}$ in capital. Furthermore, our diurnal query demonstrated that while legitimate transactions plummet by $99\%$ at 4:00 AM, automated fraud remains constant, driving fraud prevalence up by $172\text{x}$. In production, we feed these SQL window aggregations as batch-computed features into low-latency feature stores (like Feast or Redis), allowing our real-time inference engine to evaluate diurnal risk multipliers and recipient in-degree in sub-10 milliseconds."*

### Morgan Stanley (Institutional Wealth Management & Wire Surveillance)
- **Key Insight**: Large-ticket wire fraud Pareto distribution (`pareto_decile_concentration`) and failure of static threshold rules (`legacy_rule_failure`).
- **Interview Response**:  
  *"In institutional wire surveillance, transaction distributions exhibit severe power-law tails. Our SQL Pareto decile analysis proved that Decile 10 (amounts $>\$365\text{k}$) accounts for $95.7\%$ of total fraud losses ($11.54\text{B}$ out of $\$12.05\text{B}$), while the bottom 7 deciles account for barely $1.6\%$. However, we audited the existing legacy static rule (`is_flagged_fraud` $> \$200\text{k}$ on transfers) and found it suffered a catastrophic $99.8\%$ false negative rate, missing $\$11.98\text{B}$ in unmitigated losses because fraudsters structured transactions or used cash-out channels. This demonstrates that static dollar-threshold rules are insufficient; institutions require multi-variate ML risk scoring combined with dynamic windowed velocity baselines."*

### HSBC (Global Commercial Banking, AML/CFT, SLA Governance)
- **Key Insight**: Multi-stage investigation operations backlog aging (`investigation_operations_triage`) and regulatory compliance.
- **Interview Response**:  
  *"Under AML/CFT regulations (such as FinCEN SAR filing and FCA operational resilience rules), case triage turnaround time (TAT) is heavily scrutinized. Our SQL operations analysis revealed that while Tier 1 Operations maintains a $14.1\%$ breach rate on standard cases, Senior Fraud Risk and SIU squads experience a $43.8\%$ SLA breach rate on $P1\text{ Critical}$ cases against a 24-hour target. Furthermore, $95.8\%$ of our open backlog has aged beyond 72 hours. By implementing window-based aging matrices in SQL with `PERCENTILE_CONT(0.90)`, we pinpointed that the long-tail P90 TAT reaches 120 hours. This operational intelligence allows management to dynamically rebalance squad capacity from benign P3 alerts to critical high-exposure cases before regulatory breach penalties occur."*

---

## 5. Artifact Verification & Deliverables
- **SQL Source**: [`sql/05_risk_analysis.sql`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/sql/05_risk_analysis.sql)
- **Execution Script**: [`src/financial_risk/analytics.py`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/src/financial_risk/analytics.py)
- **Summary Persistence**: [`data/processed/sql_analytics_summary.json`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/data/processed/sql_analytics_summary.json)
- **Pytest Suite**: [`tests/test_sql_analytics.py`](file:///c:/PROJECTS/Financial Transaction Risk & Operations Intelligence/tests/test_sql_analytics.py) (All 11 tests passing, 52/52 suite-wide)
