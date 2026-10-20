"""Phase 1 Environment and Dependency Verification Tests.

Verifies that the Python environment, required packages, and core configuration
are functioning correctly and reproducible.
"""

import sys
import importlib
import pytest


REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "sklearn",
    "sqlalchemy",
    "psycopg2",
    "dotenv",
    "pytest",
]


def test_python_version():
    """Verify Python runtime version is at least 3.12."""
    assert sys.version_info >= (3, 12), f"Expected Python >= 3.12, got {sys.version}"


@pytest.mark.parametrize("package_name", REQUIRED_PACKAGES)
def test_required_packages_installed(package_name: str):
    """Verify that each core required library imports without error."""
    mod = importlib.import_module(package_name)
    assert mod is not None, f"Failed to import {package_name}"


def test_project_config_paths():
    """Verify project configuration and directory paths resolution."""
    from financial_risk.config import (
        PROJECT_ROOT,
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        DATA_EXTERNAL_DIR,
        DOCS_DIR,
        SQL_DIR,
    )

    assert PROJECT_ROOT.exists(), f"Project root does not exist: {PROJECT_ROOT}"
    assert DATA_RAW_DIR.exists(), f"Raw data directory does not exist: {DATA_RAW_DIR}"
    assert DATA_PROCESSED_DIR.exists(), f"Processed data dir does not exist: {DATA_PROCESSED_DIR}"
    assert DATA_EXTERNAL_DIR.exists(), f"External data dir does not exist: {DATA_EXTERNAL_DIR}"
    assert DOCS_DIR.exists(), f"Docs dir does not exist: {DOCS_DIR}"
    assert SQL_DIR.exists(), f"SQL dir does not exist: {SQL_DIR}"
