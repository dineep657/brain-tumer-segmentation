"""
database.py — DB connection helper for the Brain Tumor Web App.

Backends:
  - sqlite (default for free Render deploy): set BT_DB_BACKEND=sqlite
  - mysql (local / Railway): set BT_DB_BACKEND=mysql (or leave unset when MYSQLHOST is present)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def _detect_backend() -> str:
    explicit = _env("BT_DB_BACKEND").lower()
    if explicit in {"sqlite", "mysql"}:
        return explicit
    # Auto: MySQL when a host plugin var is present, otherwise SQLite for free hosts.
    if _env("MYSQLHOST", "BT_DB_HOST"):
        return "mysql"
    return "sqlite"


DB_BACKEND = _detect_backend()

# Prefer BT_* vars; fall back to Railway/Render MySQL plugin vars.
DB_CONFIG = {
    "host": _env("BT_DB_HOST", "MYSQLHOST", default="localhost"),
    "user": _env("BT_DB_USER", "MYSQLUSER", default="root"),
    "password": _env("BT_DB_PASSWORD", "MYSQLPASSWORD", default="Dineep123"),
    "database": _env("BT_DB_NAME", "MYSQLDATABASE", default="brain_tumor_app"),
    "port": int(_env("BT_DB_PORT", "MYSQLPORT", default="3306")),
}

_DEFAULT_SQLITE = Path(__file__).resolve().parent / "data" / "brainscan.db"
SQLITE_PATH = Path(_env("BT_SQLITE_PATH", default=str(_DEFAULT_SQLITE)))


class Error(Exception):
    """Compatibility alias previously imported from mysql.connector."""


def _sqlite_connect() -> sqlite3.Connection:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SQLITE_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_connection():
    """Return a new DB connection for the active backend."""
    if DB_BACKEND == "sqlite":
        return _sqlite_connect()

    import mysql.connector

    return mysql.connector.connect(**DB_CONFIG)


def _adapt_sql(query: str) -> str:
    if DB_BACKEND == "sqlite":
        return query.replace("%s", "?")
    return query


def execute_query(query: str, params: tuple = (), fetch: bool = False):
    """
    Execute a parameterised query.
    - fetch=False → INSERT/UPDATE/DELETE, returns lastrowid
    - fetch=True  → SELECT, returns list of dicts
    """
    conn = get_connection()
    sql = _adapt_sql(query)
    try:
        if DB_BACKEND == "sqlite":
            cursor = conn.cursor()
            try:
                cursor.execute(sql, params)
                if fetch:
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                conn.commit()
                return cursor.lastrowid
            finally:
                cursor.close()
        else:
            cursor = conn.cursor(dictionary=True)
            try:
                cursor.execute(sql, params)
                if fetch:
                    return cursor.fetchall()
                conn.commit()
                return cursor.lastrowid
            finally:
                cursor.close()
    finally:
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
