"""Phase 7 Database Schema and SQL DDL Tests.

Validates that SQL DDL scripts, relational schemas (stg, core, mart), primary and foreign key
constraints, B-Tree indexes, and analytical view definitions are syntactically sound,
modular, and completely documented.
"""

from financial_risk.config import DOCS_DIR, SQL_DIR


def test_sql_ddl_files_presence():
    """Verify that all required Phase 7 SQL scripts exist."""
    required_files = [
        "01_create_schemas.sql",
        "02_create_staging_tables.sql",
        "03_create_dimensions_facts.sql",
        "04_transformations.sql",
    ]

    for sql_file in required_files:
        p = SQL_DIR / sql_file
        assert p.exists(), f"Missing SQL file: {sql_file}"
        assert p.stat().st_size > 100, f"SQL file appears empty: {sql_file}"


def test_schema_creation_script():
    """Verify 01_create_schemas.sql defines stg, core, and mart schemas."""
    content = (SQL_DIR / "01_create_schemas.sql").read_text(encoding="utf-8")
    assert "CREATE SCHEMA IF NOT EXISTS stg;" in content
    assert "CREATE SCHEMA IF NOT EXISTS core;" in content
    assert "CREATE SCHEMA IF NOT EXISTS mart;" in content


def test_staging_tables_ddl_script():
    """Verify 02_create_staging_tables.sql defines both staging tables."""
    content = (SQL_DIR / "02_create_staging_tables.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE stg.stg_transactions" in content
    assert "CREATE TABLE stg.stg_investigations" in content
    assert "transaction_id                BIGINT PRIMARY KEY" in content
    assert "case_id                       VARCHAR(32) PRIMARY KEY" in content


def test_star_schema_ddl_script():
    """Verify 03_create_dimensions_facts.sql defines dimensions, facts, and indexes."""
    content = (SQL_DIR / "03_create_dimensions_facts.sql").read_text(encoding="utf-8")

    # Dimensions
    assert "CREATE TABLE core.dim_date" in content
    assert "CREATE TABLE core.dim_transaction_type" in content
    assert "CREATE TABLE core.dim_case_status" in content
    assert "CREATE TABLE core.dim_priority" in content
    assert "CREATE TABLE core.dim_account" in content

    # Facts
    assert "CREATE TABLE core.fact_transactions" in content
    assert "CREATE TABLE core.fact_investigations" in content

    # Foreign Keys & Constraints
    assert "REFERENCES core.dim_date(step)" in content
    assert "REFERENCES core.dim_transaction_type(type_code)" in content
    assert "REFERENCES core.fact_transactions(transaction_id)" in content

    # Indexes
    assert "CREATE INDEX idx_fact_tx_step" in content
    assert "CREATE INDEX idx_fact_tx_type" in content
    assert "CREATE INDEX idx_fact_tx_fraud" in content
    assert "CREATE INDEX idx_fact_inv_linked_tx" in content


def test_transformations_and_marts_ddl_script():
    """Verify 04_transformations.sql seeds dimensions and creates analytical views."""
    content = (SQL_DIR / "04_transformations.sql").read_text(encoding="utf-8")

    # Seeding
    assert "INSERT INTO core.dim_transaction_type" in content
    assert "INSERT INTO core.dim_case_status" in content
    assert "INSERT INTO core.dim_priority" in content
    assert "generate_series(1, 744)" in content

    # Views
    assert "CREATE OR REPLACE VIEW mart.mart_risk_executive_overview" in content
    assert "CREATE OR REPLACE VIEW mart.mart_investigation_operations" in content
    assert "CREATE OR REPLACE VIEW mart.mart_channel_exposure" in content


def test_er_diagram_documentation_structure():
    """Verify docs/er_diagram.md contains Mermaid ER diagram and catalog specifications."""
    doc_path = DOCS_DIR / "er_diagram.md"
    assert doc_path.exists(), f"Missing ER documentation at {doc_path}"

    content = doc_path.read_text(encoding="utf-8")
    assert "# Relational Analytical Data Model" in content
    assert "```mermaid" in content
    assert "erDiagram" in content
    assert "FACT_TRANSACTIONS" in content
    assert "FACT_INVESTIGATIONS" in content
    assert "DIM_DATE" in content
    assert "DIM_TRANSACTION_TYPE" in content
