# Operational Case & Unstructured Text Processing Report

**Pipeline Component**: NLP & Case Management Normalization Engine  
**Execution Duration**: 0.24 seconds  
**Source Dataset**: `data/raw/investigation_cases.csv` (5,000 tickets)  
**Output Target**: `data/processed/investigation_cases_cleaned.parquet`  

---

## 1. Executive Summary & Text Quality Scorecard

| Dimension | Raw Input Value | Normalized Value / Treatment | Assessment |
|---|---|---|---|
| **Total Case Records** | 5,000 | 5,000 | 100% record retention |
| **Missing Investigation Notes** | 887 (17.74%) | Imputed with `'[NO NOTES RECORDED]'` + `has_notes=False` | Handled via sentinel imputation |
| **Whitespace Irregularities** | Trailing/leading tabs, multiple spaces | Normalized to single space via regex `\s+` | Standardized formatting |
| **Original Text Preservation** | Free text | Preserved in `raw_investigation_notes` | Full audit trail retained |
| **Quantitative Text Metrics** | None | `note_char_length`, `note_word_count`, `note_avg_word_length` | Extracted for operations BI |

---

## 2. Issue Category Keyword Indicators

Rule-based regex matching extracted transparent domain indicators from qualitative notes:

| Extracted Category Flag | Matching Patterns | Total Cases Tagged | % of Documented Cases | Operational Context |
|---|---|---|---|---|
| `flag_compromise_suspected` | `unauthorized`, `compromise`, `takeover`, `freeze` | **810** | 19.7% | Credential theft or unauthorized device access |
| `flag_account_depletion` | `depletion`, `drain`, `rapid account`, `wiped` | **407** | 9.9% | Complete balance drainage attempts |
| `flag_velocity_surge` | `velocity`, `surge`, `unusual velocity`, `burst` | **423** | 10.3% | Frequency threshold spike within single step |
| `flag_balance_discrepancy` | `discrepancy`, `mismatch`, `mismatched` | **791** | 19.2% | Accounting variance in settlement ledger |
| `flag_benign_cleared` | `false alert`, `commercial payroll`, `cleared`, `authorized` | **833** | 20.3% | Case verified as legitimate commercial flow |

> [!NOTE]
> **Limitations of Keyword-Derived Indicators**:
> Simple keyword rules serve as transparent heuristics for triage dashboards. However, they do not replace contextual semantic understanding. In regulated financial crime investigations, keyword matching must never be claimed as a standalone NLP fraud detector.

---

## 3. Operational Investigation Metrics & SLA Compliance

- **Closed Cases**: **3,858** (77.2%)
- **Active / Open Backlog**: **1,142** (22.8%)
- **Median Turnaround Time (p50)**: **24.0 hours**
- **Mean Turnaround Time**: **41.61 hours**
- **90th Percentile Turnaround Time (p90)**: **120.0 hours**
- **SLA Breaches**: **1,384 cases** breached the operational SLA threshold (35.87% of closed cases).

---

## 4. Governance & Anti-Leakage Boundary Rule

> [!CAUTION]
> **CRITICAL DATA SCIENCE GOVERNANCE RULE**:
> Investigation notes, turnaround times, and resolution outcomes are generated **after** an alert is routed to human investigators.
> - **Permitted Use**: Operational capacity analytics, backlog aging analysis (Page 3 of Power BI Dashboard), and investigator workload modeling.
> - **Prohibited Use**: Training the pre-transaction Decision Tree classifier (Phases 11-12). Joining post-investigation text features into the transaction feature matrix would introduce **catastrophic lookahead leakage** (predicting whether a transaction is fraudulent based on an investigator note written 24 hours later).
