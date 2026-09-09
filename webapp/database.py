"""
database.py — MySQL connection and helper for the Brain Tumor Web App.
"""

import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Dineep123",
    "database": "brain_tumor_app",
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
