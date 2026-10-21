"""Text Processing and Operational Case Normalization Pipeline.

Cleans and standardizes unstructured investigation case notes, handles missing text,
computes quantitative text metrics, extracts transparent rule-based domain keyword indicators,
and calculates operational turnaround time (TAT) and SLA metrics.

CRITICAL GOVERNANCE RULE:
Operational case notes and investigation resolution outcomes are recorded AFTER an alert is triggered.
They MUST NOT be used as predictive features for pre-investigation transaction fraud classification,
as that would introduce severe retrospective data leakage.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from financial_risk.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, DOCS_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class CaseTextProcessor:
    """Enterprise text processor for operational investigation case logs."""

    # Transparent, auditable keyword patterns for issue categorization
    KEYWORD_PATTERNS: ClassVar[dict[str, list[str]]] = {
        "flag_account_depletion": [
            r"\bdeplet",
            r"\bdrain",
            r"\brapid account",
            r"\bbalance wiped",
        ],
        "flag_velocity_surge": [
            r"\bvelocity",
            r"\bsurge",
            r"\bunusual velocity",
            r"\bburst",
        ],
        "flag_compromise_suspected": [
            r"\bunauthorized",
            r"\bcompromise",
            r"\btakeover",
            r"\bfreez",
        ],
        "flag_balance_discrepancy": [r"\bdiscrepanc", r"\bmismatch", r"\bmismatched"],
        "flag_benign_cleared": [
            r"\bfalse alert",
            r"\bcommercial payroll",
            r"\bordinary commercial",
            r"\bcleared",
            r"\bauthorized via",
        ],
    }

    # SLA thresholds in hours by priority tier
    SLA_HOURS_BY_PRIORITY: ClassVar[dict[str, float]] = {
        "P1_CRITICAL": 24.0,
        "P2_HIGH": 48.0,
        "P3_STANDARD": 96.0,
    }

    def __init__(
        self,
        raw_cases_path: Path = DATA_RAW_DIR / "investigation_cases.csv",
        output_parquet_path: Path = DATA_PROCESSED_DIR
        / "investigation_cases_cleaned.parquet",
        summary_json_path: Path = DATA_PROCESSED_DIR / "case_text_summary.json",
        report_path: Path = DOCS_DIR / "text_processing_report.md",
    ):
        self.raw_cases_path = raw_cases_path
        self.output_parquet_path = output_parquet_path
        self.summary_json_path = summary_json_path
        self.report_path = report_path

    @staticmethod
    def normalize_text_whitespace(text: str) -> str:
        """Strip leading/trailing whitespace and collapse internal repeated spaces/tabs/newlines."""
        if not text:
            return ""
        # Collapse multiple whitespace characters into single space
        return re.sub(r"\s+", " ", text).strip()

    def process_cases(self) -> dict[str, Any]:
        """Execute text normalization and operational feature extraction."""
        logger.info(f"Loading raw case records from {self.raw_cases_path}...")
        start_time = time.time()

        df = pd.read_csv(self.raw_cases_path)
        total_cases = len(df)

        raw_notes = df["investigation_notes"].fillna("")

        # 1. Text Completeness and Sentinel Imputation
        has_notes = df["investigation_notes"].notna() & (raw_notes.str.strip() != "")
        missing_count = int((~has_notes).sum())

        cleaned_notes = []
        char_lengths = []
        word_counts = []
        avg_word_lengths = []

        for raw_text, present in zip(raw_notes, has_notes, strict=False):
            if not present:
                cleaned_notes.append("[NO NOTES RECORDED]")
                char_lengths.append(0)
                word_counts.append(0)
                avg_word_lengths.append(0.0)
            else:
                norm = self.normalize_text_whitespace(raw_text)
                cleaned_notes.append(norm)
                c_len = len(norm)
                words = norm.split()
                w_count = len(words)
                avg_w_len = (
                    round(sum(len(w) for w in words) / w_count, 2)
                    if w_count > 0
                    else 0.0
                )

                char_lengths.append(c_len)
                word_counts.append(w_count)
                avg_word_lengths.append(avg_w_len)

        # 2. Transparent Rule-Based Keyword Indicators
        keyword_flags: dict[str, list[bool]] = {
            flag: [] for flag in self.KEYWORD_PATTERNS
        }
        for note in cleaned_notes:
            note_lower = note.lower()
            for flag, patterns in self.KEYWORD_PATTERNS.items():
                matched = any(re.search(p, note_lower) for p in patterns)
                keyword_flags[flag].append(matched)

        # 3. Operational Timestamps, Turnaround Time (TAT), and SLA Compliance
        created_at_dt = pd.to_datetime(df["case_created_at"], utc=True)
        closed_at_dt = pd.to_datetime(df["closed_at"], utc=True)

        is_closed = df["case_status"].str.startswith("CLOSED")
        tat_hours = np.where(
            is_closed,
            (closed_at_dt - created_at_dt).dt.total_seconds() / 3600.0,
            np.nan,
        )

        sla_limits = df["priority"].map(self.SLA_HOURS_BY_PRIORITY).fillna(48.0)
        is_sla_breached = np.where(
            is_closed,
            tat_hours > sla_limits,
            False,
        )

        # Construct Cleaned DataFrame
        processed_df = pd.DataFrame(
            {
                "case_id": df["case_id"].astype("string"),
                "linked_transaction_id": df["linked_transaction_id"].astype("int64"),
                "tx_step": df["tx_step"].astype("int16"),
                "tx_type": df["tx_type"].astype("category"),
                "tx_amount": df["tx_amount"].astype("float64"),
                "priority": df["priority"].astype("category"),
                "assigned_team": df["assigned_team"].astype("category"),
                "case_status": df["case_status"].astype("category"),
                "case_created_at": df["case_created_at"].astype("string"),
                "closed_at": df["closed_at"].fillna("").astype("string"),
                "is_case_closed": is_closed.astype("bool"),
                "turnaround_time_hours": tat_hours,
                "is_sla_breached": is_sla_breached.astype("bool"),
                "has_notes": has_notes.values.astype("bool"),
                "raw_investigation_notes": raw_notes.astype("string"),
                "cleaned_investigation_notes": pd.Series(cleaned_notes, dtype="string"),
                "note_char_length": np.array(char_lengths, dtype=np.int32),
                "note_word_count": np.array(word_counts, dtype=np.int16),
                "note_avg_word_length": np.array(avg_word_lengths, dtype=np.float32),
            }
        )

        # Append keyword flags
        for flag_name, flag_values in keyword_flags.items():
            processed_df[flag_name] = np.array(flag_values, dtype=bool)

        # Write to Parquet
        self.output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(processed_df, preserve_index=False)
        pq.write_table(table, self.output_parquet_path, compression="snappy")

        elapsed = time.time() - start_time
        parquet_size = self.output_parquet_path.stat().st_size

        # Compute summary metrics
        valid_tat = tat_hours[~np.isnan(tat_hours)]
        tat_p50 = float(np.median(valid_tat)) if len(valid_tat) > 0 else 0.0
        tat_p90 = float(np.percentile(valid_tat, 90)) if len(valid_tat) > 0 else 0.0
        tat_mean = float(np.mean(valid_tat)) if len(valid_tat) > 0 else 0.0

        summary = {
            "total_cases_processed": total_cases,
            "execution_duration_seconds": round(elapsed, 2),
            "parquet_size_bytes": parquet_size,
            "text_quality_audit": {
                "total_cases_with_notes": int(has_notes.sum()),
                "total_missing_notes": missing_count,
                "missing_notes_pct": round((missing_count / total_cases) * 100, 2),
                "mean_word_count": round(
                    float(np.mean([w for w in word_counts if w > 0])), 1
                )
                if any(word_counts)
                else 0.0,
                "mean_char_length": round(
                    float(np.mean([c for c in char_lengths if c > 0])), 1
                )
                if any(char_lengths)
                else 0.0,
            },
            "keyword_flag_frequencies": {
                flag: int(sum(vals)) for flag, vals in keyword_flags.items()
            },
            "operational_metrics": {
                "closed_cases": int(is_closed.sum()),
                "open_cases": int((~is_closed).sum()),
                "sla_breaches": int(is_sla_breached.sum()),
                "sla_breach_rate_pct": round(
                    (int(is_sla_breached.sum()) / int(is_closed.sum())) * 100, 2
                )
                if is_closed.sum() > 0
                else 0.0,
                "turnaround_time_hours": {
                    "mean": round(tat_mean, 2),
                    "median_p50": round(tat_p50, 2),
                    "p90": round(tat_p90, 2),
                },
            },
        }

        # Save summary JSON
        with open(self.summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary JSON saved to {self.summary_json_path}")

        # Save Markdown Report
        md_report = self.generate_text_report(summary)
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        logger.info(f"Text processing report saved to {self.report_path}")

        return summary

    def generate_text_report(self, summary: dict[str, Any]) -> str:
        """Construct executive markdown report for text processing and case operations."""
        tq = summary["text_quality_audit"]
        kf = summary["keyword_flag_frequencies"]
        op = summary["operational_metrics"]
        tat = op["turnaround_time_hours"]

        report = f"""# Operational Case & Unstructured Text Processing Report

**Pipeline Component**: NLP & Case Management Normalization Engine  
**Execution Duration**: {summary["execution_duration_seconds"]} seconds  
**Source Dataset**: `data/raw/investigation_cases.csv` ({summary["total_cases_processed"]:,} tickets)  
**Output Target**: `data/processed/investigation_cases_cleaned.parquet`  

---

## 1. Executive Summary & Text Quality Scorecard

| Dimension | Raw Input Value | Normalized Value / Treatment | Assessment |
|---|---|---|---|
| **Total Case Records** | {summary["total_cases_processed"]:,} | {summary["total_cases_processed"]:,} | 100% record retention |
| **Missing Investigation Notes** | {tq["total_missing_notes"]:,} ({tq["missing_notes_pct"]}%) | Imputed with `'[NO NOTES RECORDED]'` + `has_notes=False` | Handled via sentinel imputation |
| **Whitespace Irregularities** | Trailing/leading tabs, multiple spaces | Normalized to single space via regex `\\s+` | Standardized formatting |
| **Original Text Preservation** | Free text | Preserved in `raw_investigation_notes` | Full audit trail retained |
| **Quantitative Text Metrics** | None | `note_char_length`, `note_word_count`, `note_avg_word_length` | Extracted for operations BI |

---

## 2. Issue Category Keyword Indicators

Rule-based regex matching extracted transparent domain indicators from qualitative notes:

| Extracted Category Flag | Matching Patterns | Total Cases Tagged | % of Documented Cases | Operational Context |
|---|---|---|---|---|
| `flag_compromise_suspected` | `unauthorized`, `compromise`, `takeover`, `freeze` | **{kf["flag_compromise_suspected"]:,}** | {kf["flag_compromise_suspected"] / tq["total_cases_with_notes"] * 100:.1f}% | Credential theft or unauthorized device access |
| `flag_account_depletion` | `depletion`, `drain`, `rapid account`, `wiped` | **{kf["flag_account_depletion"]:,}** | {kf["flag_account_depletion"] / tq["total_cases_with_notes"] * 100:.1f}% | Complete balance drainage attempts |
| `flag_velocity_surge` | `velocity`, `surge`, `unusual velocity`, `burst` | **{kf["flag_velocity_surge"]:,}** | {kf["flag_velocity_surge"] / tq["total_cases_with_notes"] * 100:.1f}% | Frequency threshold spike within single step |
| `flag_balance_discrepancy` | `discrepancy`, `mismatch`, `mismatched` | **{kf["flag_balance_discrepancy"]:,}** | {kf["flag_balance_discrepancy"] / tq["total_cases_with_notes"] * 100:.1f}% | Accounting variance in settlement ledger |
| `flag_benign_cleared` | `false alert`, `commercial payroll`, `cleared`, `authorized` | **{kf["flag_benign_cleared"]:,}** | {kf["flag_benign_cleared"] / tq["total_cases_with_notes"] * 100:.1f}% | Case verified as legitimate commercial flow |

> [!NOTE]
> **Limitations of Keyword-Derived Indicators**:
> Simple keyword rules serve as transparent heuristics for triage dashboards. However, they do not replace contextual semantic understanding. In regulated financial crime investigations, keyword matching must never be claimed as a standalone NLP fraud detector.

---

## 3. Operational Investigation Metrics & SLA Compliance

- **Closed Cases**: **{op["closed_cases"]:,}** ({op["closed_cases"] / summary["total_cases_processed"] * 100:.1f}%)
- **Active / Open Backlog**: **{op["open_cases"]:,}** ({op["open_cases"] / summary["total_cases_processed"] * 100:.1f}%)
- **Median Turnaround Time (p50)**: **{tat["median_p50"]} hours**
- **Mean Turnaround Time**: **{tat["mean"]} hours**
- **90th Percentile Turnaround Time (p90)**: **{tat["p90"]} hours**
- **SLA Breaches**: **{op["sla_breaches"]:,} cases** breached the operational SLA threshold ({op["sla_breach_rate_pct"]}% of closed cases).

---

## 4. Governance & Anti-Leakage Boundary Rule

> [!CAUTION]
> **CRITICAL DATA SCIENCE GOVERNANCE RULE**:
> Investigation notes, turnaround times, and resolution outcomes are generated **after** an alert is routed to human investigators.
> - **Permitted Use**: Operational capacity analytics, backlog aging analysis (Page 3 of Power BI Dashboard), and investigator workload modeling.
> - **Prohibited Use**: Training the pre-transaction Decision Tree classifier (Phases 11-12). Joining post-investigation text features into the transaction feature matrix would introduce **catastrophic lookahead leakage** (predicting whether a transaction is fraudulent based on an investigator note written 24 hours later).
"""
        return report


if __name__ == "__main__":
    processor = CaseTextProcessor()
    processor.process_cases()
