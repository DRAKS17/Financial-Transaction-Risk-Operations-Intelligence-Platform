"""Data Acquisition and Ingestion Module.

Handles automated download of raw transaction datasets, cryptographic checksum
verification, and generation of linked operational investigation cases for unstructured analysis.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from financial_risk.config import DATA_RAW_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PAYSIM_URL = "https://huggingface.co/datasets/theman10/paysim/resolve/main/paysim.csv"
PAYSIM_RAW_PATH = DATA_RAW_DIR / "paysim.csv"
INVESTIGATIONS_RAW_PATH = DATA_RAW_DIR / "investigation_cases.csv"
METADATA_PATH = DATA_RAW_DIR / "raw_inventory_metadata.json"


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the cryptographic SHA-256 hash of a file for auditability."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_paysim_dataset(
    url: str = PAYSIM_URL, dest_path: Path = PAYSIM_RAW_PATH
) -> Path:
    """Download the PaySim transaction CSV with chunked streaming and progress logging."""
    if dest_path.exists() and dest_path.stat().st_size > 400_000_000:
        logger.info(
            f"PaySim raw dataset already exists at {dest_path} ({dest_path.stat().st_size:,} bytes). Skipping download."
        )
        return dest_path

    logger.info(f"Downloading PaySim transaction dataset from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    temp_path = dest_path.with_suffix(".tmp")
    start_time = time.time()
    downloaded = 0

    with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 4 * 1024 * 1024  # 4 MB chunks

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = (downloaded / total_size) * 100
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                if int(pct) % 20 == 0:
                    logger.info(
                        f"Download progress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({pct:.1f}%)"
                    )

    temp_path.replace(dest_path)
    elapsed = time.time() - start_time
    logger.info(
        f"Download complete in {elapsed:.1f}s. Saved to {dest_path} ({dest_path.stat().st_size:,} bytes)."
    )
    return dest_path


def generate_synthetic_investigation_cases(
    paysim_path: Path = PAYSIM_RAW_PATH,
    output_path: Path = INVESTIGATIONS_RAW_PATH,
    sample_size: int = 5000,
    random_seed: int = 42,
) -> Path:
    """Generate a realistic operational investigation case dataset linked to PaySim transactions.

    Includes realistic data quality scenarios:
    - Inconsistent whitespace and casing
    - Varied investigator note phrasing
    - Missing notes
    - Plausible turnaround times and resolution statuses
    """
    if output_path.exists() and output_path.stat().st_size > 100_000:
        logger.info(
            f"Synthetic investigation cases already exist at {output_path}. Skipping generation."
        )
        return output_path

    logger.info(
        f"Sampling transactions from {paysim_path} to generate linked investigation cases..."
    )
    random.seed(random_seed)

    flagged_txs: list[dict[str, Any]] = []
    normal_txs: list[dict[str, Any]] = []

    # Read PaySim with row index as linked_transaction_id (1-based index)
    with open(paysim_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # Prioritize all heuristic flagged and actual fraud cases to create realistic alert review queues
            is_fraud = row["isFraud"] == "1"
            is_flagged = row["isFlaggedFraud"] == "1"

            if is_fraud or is_flagged:
                row["_tx_id"] = idx
                flagged_txs.append(row)
            elif len(normal_txs) < sample_size and random.random() < 0.005:
                row["_tx_id"] = idx
                normal_txs.append(row)

    combined = flagged_txs + normal_txs
    selected = random.sample(combined, min(sample_size, len(combined)))
    selected.sort(key=lambda x: int(x["step"]))

    logger.info(f"Generating {len(selected)} operational investigation records...")

    statuses = [
        "CLOSED_CONFIRMED_FRAUD",
        "CLOSED_FALSE_POSITIVE",
        "UNDER_REVIEW",
        "OPEN",
    ]

    note_templates = [
        "Rapid account depletion noted. Immediate verification initiated for customer {nameOrig}.",
        "High-value {type} transaction flagged by threshold rule. Destination account {nameDest} queried.",
        "Unusual velocity surge observed within step {step}. Customer contacted via registered SMS rail.",
        "Pattern resembles smurfing/layering activity. Escalated to Tier 2 specialist for entity resolution.",
        "Confirmed customer authorized transaction via secondary channel OTP. Cleared as false alert.",
        "Destination balance mismatched post-settlement. Flagged for potential credential compromise.",
        "Account {nameOrig} reported unauthorized access attempt. Freezing outbound rail pending review.",
        "   discrepancy in sender balance noted   ",  # Intentional whitespace defect
        "SUSPICIOUS HIGH VALUE OUTFLOW TO UNKNOWN COUNTERPARTY",  # Intentional uppercase
        "Reviewed customer transaction history. Ordinary commercial payroll pattern confirmed.",
        "",  # Intentional missing note
        None,  # Intentional null note
    ]

    base_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)

    fieldnames = [
        "case_id",
        "linked_transaction_id",
        "tx_step",
        "tx_type",
        "tx_amount",
        "case_created_at",
        "case_status",
        "priority",
        "assigned_team",
        "investigation_notes",
        "closed_at",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()

        for c_idx, tx in enumerate(selected, start=1):
            case_id = f"CASE-{2026}-{c_idx:06d}"
            tx_id = tx["_tx_id"]
            step = int(tx["step"])
            tx_type = tx["type"]
            amount = float(tx["amount"])

            # Create timestamp based on step (step 1 = 1 hour past base_time)
            created_at = base_time + timedelta(
                hours=step, minutes=random.randint(5, 55)
            )

            # Realistic priority assignment
            if amount > 500_000 or tx["isFraud"] == "1":
                priority = "P1_CRITICAL"
                team = random.choice(
                    ["Senior Fraud Risk", "Special Investigations Unit (SIU)"]
                )
            elif amount > 100_000:
                priority = "P2_HIGH"
                team = random.choice(["Tier 1 Operations", "Senior Fraud Risk"])
            else:
                priority = "P3_STANDARD"
                team = "Tier 1 Operations"

            # Case status & closing time
            status = random.choices(
                statuses,
                weights=[0.35 if tx["isFraud"] == "1" else 0.05, 0.45, 0.12, 0.08],
                k=1,
            )[0]

            if status.startswith("CLOSED"):
                tat_hours = random.choice([2, 6, 18, 24, 48, 72, 120])
                closed_at = (created_at + timedelta(hours=tat_hours)).isoformat()
            else:
                closed_at = ""

            template = random.choice(note_templates)
            if template:
                notes = template.format(
                    nameOrig=tx["nameOrig"],
                    nameDest=tx["nameDest"],
                    type=tx_type,
                    step=step,
                )
            elif template == "":
                notes = ""
            else:
                notes = ""

            writer.writerow(
                {
                    "case_id": case_id,
                    "linked_transaction_id": tx_id,
                    "tx_step": step,
                    "tx_type": tx_type,
                    "tx_amount": amount,
                    "case_created_at": created_at.isoformat(),
                    "case_status": status,
                    "priority": priority,
                    "assigned_team": team,
                    "investigation_notes": notes,
                    "closed_at": closed_at,
                }
            )

    logger.info(
        f"Synthetic case records saved to {output_path} ({output_path.stat().st_size:,} bytes)."
    )
    return output_path


def profile_raw_inventory() -> dict[str, Any]:
    """Inspect and inventory all raw files in data/raw/."""
    inventory: dict[str, Any] = {}

    for file_path in DATA_RAW_DIR.iterdir():
        if file_path.name.startswith(".") or not file_path.is_file():
            continue

        file_size = file_path.stat().st_size
        sha256 = compute_sha256(file_path)

        # Count rows and inspect columns
        row_count = 0
        columns = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            try:
                columns = next(reader)
                for _ in reader:
                    row_count += 1
            except StopIteration:
                pass

        inventory[file_path.name] = {
            "filename": file_path.name,
            "file_size_bytes": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "sha256": sha256,
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
        }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    logger.info(f"Inventory metadata written to {METADATA_PATH}")
    return inventory


if __name__ == "__main__":
    download_paysim_dataset()
    generate_synthetic_investigation_cases()
    profile_raw_inventory()
