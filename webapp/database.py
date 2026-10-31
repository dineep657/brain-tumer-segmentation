"""
database.py — MySQL connection and helper for the Brain Tumor Web App.
"""

import os
import mysql.connector
from mysql.connector import Error  # noqa: F401 — re-exported for callers

DB_CONFIG = {
    "host": os.environ.get("BT_DB_HOST", "localhost"),
    "user": os.environ.get("BT_DB_USER", "root"),
    "password": os.environ.get("BT_DB_PASSWORD", "Dineep123"),
    "database": os.environ.get("BT_DB_NAME", "brain_tumor_app"),
}


def get_connection():
    """Return a new MySQL connection."""
    return mysql.connector.connect(**DB_CONFIG)


def execute_query(query: str, params: tuple = (), fetch: bool = False):
    """
    Execute a parameterised query.
    - fetch=False → INSERT/UPDATE/DELETE, returns lastrowid
    - fetch=True  → SELECT, returns list of dicts
    """
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(query, params)
        if fetch:
            result = cursor.fetchall()
            return result
        else:
            conn.commit()
            return cursor.lastrowid
    finally:
        cursor.close()
        conn.close()


def write_audit_log(
    actor_id: int | None,
    action: str,
    comments: str | None = None,
    patient_id: int | None = None,
    report_id: int | None = None,
):
    """Insert an audit trail row. Failures are swallowed so UX is not blocked."""
    try:
        execute_query(
            """
            INSERT INTO audit_logs (actor_id, patient_id, report_id, action, comments)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (actor_id, patient_id, report_id, action, comments),
        )
    except Exception:
        pass
