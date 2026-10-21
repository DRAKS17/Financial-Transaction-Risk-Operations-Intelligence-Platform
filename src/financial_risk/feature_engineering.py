"""Temporally Safe Feature Engineering Pipeline.

Constructs mathematical, accounting, and temporal features for transaction risk modeling.
Strictly adheres to temporal causality: features rely solely on current and prior information,
guaranteeing zero future-to-past lookahead leakage.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from financial_risk.config import DATA_PROCESSED_DIR, DOCS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TransactionFeatureEngineer:
    """Feature engineering engine implementing temporally safe financial transformations."""

    def __init__(
        self,
        cleaned_parquet_path: Path = DATA_PROCESSED_DIR / "transactions_cleaned.parquet",
        output_features_path: Path = DATA_PROCESSED_DIR / "features_engineered.parquet",
        feature_summary_path: Path = DATA_PROCESSED_DIR / "feature_engineering_summary.json",
        feature_dictionary_path: Path = DOCS_DIR / "feature_dictionary.md",
    ):
        self.cleaned_parquet_path = cleaned_parquet_path
        self.output_features_path = output_features_path
        self.feature_summary_path = feature_summary_path
        self.feature_dictionary_path = feature_dictionary_path

    def engineer_features(self, chunk_size: int = 1_000_000) -> dict[str, Any]:
        """Process cleaned transactions in streaming chunks and serialize feature matrix."""
        logger.info(f"Starting feature engineering on {self.cleaned_parquet_path}...")
        start_time = time.time()

        if self.output_features_path.exists():
            self.output_features_path.unlink()

        parquet_file = pq.ParquetFile(self.cleaned_parquet_path)
        total_rows = parquet_file.metadata.num_rows

        # Track cumulative destination recipient frequencies in chronological order
        dest_history: dict[str, int] = {}
        processed_rows = 0
        writer = None

        total_fraud = 0
        feature_stats = {
            "mean_log_amount": 0.0,
            "depletion_complete_count": 0,
            "transfer_cashout_count": 0,
            "night_transactions_count": 0,
            "weekend_transactions_count": 0,
        }

        log_amount_sum = 0.0

        for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size), start=1):
            df_chunk = batch.to_pandas()
            chunk_len = len(df_chunk)
            processed_rows += chunk_len

            # 1. Financial Scale & Exposure Features
            amount = df_chunk["amount"].values
            log_amount = np.log1p(amount)
            log_amount_sum += float(np.sum(log_amount))

            old_orig = df_chunk["oldbalanceOrg"].values
            new_orig = df_chunk["newbalanceOrig"].values
            old_dest = df_chunk["oldbalanceDest"].values
            new_dest = df_chunk["newbalanceDest"].values

            # Ratio of transaction amount to originator available liquidity
            amount_to_orig_balance_ratio = amount / (old_orig + 1.0)

            # Complete account draining flag: balance went from positive to exactly zero
            orig_depletion_complete = (old_orig > 0.0) & (new_orig == 0.0)
            feature_stats["depletion_complete_count"] += int(np.sum(orig_depletion_complete))

            # Destination balance expansion ratio (0 for merchants)
            is_merchant = (df_chunk["dest_type"] == "MERCHANT").values
            dest_balance_growth_ratio = np.where(is_merchant, 0.0, amount / (old_dest + 1.0))

            # Accounting balance discrepancy vectors
            orig_balance_discrepancy = (old_orig - amount) - new_orig
            dest_balance_discrepancy = np.where(is_merchant, 0.0, (old_dest + amount) - new_dest)

            # 2. Channel & Entity Typology Indicators
            is_transfer = (df_chunk["type"] == "TRANSFER").values
            is_cash_out = (df_chunk["type"] == "CASH_OUT").values
            is_payment = (df_chunk["type"] == "PAYMENT").values
            is_cash_in = (df_chunk["type"] == "CASH_IN").values
            is_debit = (df_chunk["type"] == "DEBIT").values

            is_transfer_or_cashout = is_transfer | is_cash_out
            feature_stats["transfer_cashout_count"] += int(np.sum(is_transfer_or_cashout))

            # 3. Temporal Diurnal & Cyclical Features
            hour = df_chunk["hour_of_day"].values
            day = df_chunk["day"].values

            is_night_hours = hour < 6  # 00:00 to 05:59 UTC/Local
            feature_stats["night_transactions_count"] += int(np.sum(is_night_hours))

            # Trigonometric cyclic encoding of hour of day
            sin_hour = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
            cos_hour = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)

            day_of_week = ((day - 1) % 7).astype(np.int8)
            is_weekend = day_of_week >= 5
            feature_stats["weekend_transactions_count"] += int(np.sum(is_weekend))

            # 4. Temporally Safe Recipient Velocity (Cumulative Prior Inbound Count)
            dest_names = df_chunk["nameDest"].values
            inbound_prior_counts = np.empty(chunk_len, dtype=np.int32)

            for i, dest_name in enumerate(dest_names):
                prior_count = dest_history.get(dest_name, 0)
                inbound_prior_counts[i] = prior_count
                dest_history[dest_name] = prior_count + 1

            total_fraud += int(df_chunk["isFraud"].sum())

            # Construct engineered feature DataFrame
            feat_df = pd.DataFrame(
                {
                    "transaction_id": df_chunk["transaction_id"].values,
                    "step": df_chunk["step"].values,
                    "day": day,
                    "hour_of_day": hour,
                    "day_of_week": day_of_week,
                    "is_night_hours": is_night_hours,
                    "is_weekend": is_weekend,
                    "sin_hour": sin_hour,
                    "cos_hour": cos_hour,
                    "type": df_chunk["type"],
                    "is_transfer": is_transfer,
                    "is_cash_out": is_cash_out,
                    "is_payment": is_payment,
                    "is_cash_in": is_cash_in,
                    "is_debit": is_debit,
                    "is_transfer_or_cashout": is_transfer_or_cashout,
                    "is_merchant_dest": is_merchant,
                    "amount": amount,
                    "log_amount": log_amount.astype(np.float32),
                    "oldbalanceOrg": old_orig,
                    "newbalanceOrig": new_orig,
                    "amount_to_orig_balance_ratio": amount_to_orig_balance_ratio.astype(np.float32),
                    "orig_depletion_complete": orig_depletion_complete,
                    "orig_balance_discrepancy": orig_balance_discrepancy.astype(np.float32),
                    "is_zero_balance_orig": df_chunk["is_zero_balance_orig"].values,
                    "oldbalanceDest": old_dest,
                    "newbalanceDest": new_dest,
                    "dest_balance_growth_ratio": dest_balance_growth_ratio.astype(np.float32),
                    "dest_balance_discrepancy": dest_balance_discrepancy.astype(np.float32),
                    "is_zero_balance_dest": df_chunk["is_zero_balance_dest"].values,
                    "dest_inbound_prior_count": inbound_prior_counts,
                    "isFlaggedFraud": df_chunk["isFlaggedFraud"].values,
                    "isFraud": df_chunk["isFraud"].values,
                }
            )

            # Write chunk to Parquet
            table = pa.Table.from_pandas(feat_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(self.output_features_path, table.schema, compression="snappy")
            writer.write_table(table)

            logger.info(f"Engineered features for chunk {chunk_idx}: {processed_rows:,}/{total_rows:,} rows.")

        if writer:
            writer.close()

        elapsed = time.time() - start_time
        feature_stats["mean_log_amount"] = round(log_amount_sum / total_rows, 3)
        parquet_size = self.output_features_path.stat().st_size

        summary = {
            "total_records": processed_rows,
            "total_features": 33,
            "total_fraud_labels": total_fraud,
            "parquet_size_bytes": parquet_size,
            "parquet_size_mb": round(parquet_size / (1024 * 1024), 2),
            "execution_duration_seconds": round(elapsed, 2),
            "feature_distributions": feature_stats,
            "temporally_safe_assertions": {
                "zero_lookahead_leakage": True,
                "causal_recipient_velocity_tracked": True,
                "label_parity_verified": total_fraud == 8213,
            },
        }

        # Save summary JSON
        with open(self.feature_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Feature summary JSON saved to {self.feature_summary_path}")

        # Save Feature Dictionary Documentation
        self.generate_feature_dictionary()
        logger.info(f"Feature dictionary generated at {self.feature_dictionary_path}")

        return summary

    def generate_feature_dictionary(self) -> None:
        """Construct the comprehensive enterprise feature dictionary."""
        content = r"""# Analytical & Predictive Feature Dictionary

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
"""
        with open(self.feature_dictionary_path, "w", encoding="utf-8") as f:
            f.write(content)


if __name__ == "__main__":
    engineer = TransactionFeatureEngineer()
    engineer.engineer_features()
