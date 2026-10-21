"""Phase 8 Database Loading and Verification Tests.

Validates that core dimensional tables, conformed dimensions, and analytical mart views
are properly loaded and reconciled in the target PostgreSQL instance.
"""

import psycopg2
import pytest

from financial_risk.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


def is_database_accessible():
    """Helper to check if local PostgreSQL instance is reachable with configured credentials."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            connect_timeout=2,
        )
        conn.close()
        return True
    except psycopg2.Error:
        return False


@pytest.mark.skipif(
    not is_database_accessible(),
    reason="PostgreSQL instance not yet connected with local password.",
)
def test_fact_transactions_row_count():
    """Verify core.fact_transactions contains all 6,362,620 loaded transactions."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM core.fact_transactions;")
            count = cur.fetchone()[0]
            assert count == 6_362_620
    finally:
        conn.close()


@pytest.mark.skipif(
    not is_database_accessible(),
    reason="PostgreSQL instance not yet connected with local password.",
)
def test_fact_investigations_row_count():
    """Verify core.fact_investigations contains all 5,000 loaded case tickets."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM core.fact_investigations;")
            count = cur.fetchone()[0]
            assert count == 5_000
    finally:
        conn.close()


@pytest.mark.skipif(
    not is_database_accessible(),
    reason="PostgreSQL instance not yet connected with local password.",
)
def test_database_monetary_parity():
    """Verify total gross transaction value in PostgreSQL matches $1.144 Trillion."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT SUM(amount) FROM core.fact_transactions;")
            gtv = float(cur.fetchone()[0])
            assert round(gtv, 2) == 1144392944759.77
    finally:
        conn.close()
