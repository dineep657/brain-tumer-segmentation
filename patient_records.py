"""
patient_records.py — SQLite persistence layer for the Brain Tumor Segmentation app.

Owns all database interactions. app.py calls only the public functions below;
it never constructs SQL strings directly. All SQL uses parameterised queries only
— no string interpolation — to prevent SQL injection.
"""

import os
import sqlite3
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DB_PATH: str = os.path.join("data", "patient_records.db")

# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------


def init_db() -> None:
    """
    Creates the ``data/`` directory and ``patient_records`` table if they do
    not already exist.  Also creates indexes on ``patient_id`` and
    ``scan_timestamp`` for efficient history queries.

    Safe to call multiple times — all DDL statements use IF NOT EXISTS.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patient_records (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id     TEXT    NOT NULL,
                patient_name   TEXT    NOT NULL,
                scan_timestamp TEXT    NOT NULL,
                tumor_detected INTEGER NOT NULL,
                tumor_type     TEXT    NOT NULL,
                confidence     REAL,
                tumor_area_px  INTEGER NOT NULL,
                brain_coverage REAL    NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_patient_id
            ON patient_records (patient_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON patient_records (scan_timestamp DESC)
            """
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_patient_fields(patient_id: str, patient_name: str) -> "str | None":
    """
    Validates that both ``patient_id`` and ``patient_name`` are non-empty
    after stripping surrounding whitespace.

    Returns:
        ``None`` when both fields are valid.
        An error message string when either field is empty or whitespace-only.
    """
    if not patient_id.strip():
        return "Patient ID must not be empty."
    if not patient_name.strip():
        return "Patient Name must not be empty."
    return None


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


def insert_patient_record(
    patient_id: str,
    patient_name: str,
    metrics: dict,
    cls_info: dict,
) -> None:
    """
    Writes a single Patient_Record row to the database.

    Args:
        patient_id:   Non-empty patient identifier string.
        patient_name: Non-empty patient display name string.
        metrics:      Dict from the segmentation pipeline, expected keys:
                        ``tumor_detected`` (bool),
                        ``tumor_area_px``  (int),
                        ``brain_coverage`` (float).
        cls_info:     Dict from the classifier, expected keys:
                        ``tumor_type``  (str),
                        ``confidence``  (float | None).

    Raises:
        ValueError: If ``validate_patient_fields`` fails (caller should
                    pre-validate, but this acts as a second safety net).
        sqlite3.DatabaseError: On any DB-level failure (disk full, locked, …).
    """
    error = validate_patient_fields(patient_id, patient_name)
    if error:
        raise ValueError(error)

    scan_timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    tumor_detected = int(bool(metrics.get("tumor_detected", False)))
    tumor_area_px = int(metrics.get("tumor_area_px", 0))
    brain_coverage = round(float(metrics.get("brain_coverage", 0.0)), 4)

    tumor_type = cls_info.get("tumor_type", "")
    raw_confidence = cls_info.get("confidence")
    confidence = round(float(raw_confidence), 4) if raw_confidence is not None else None

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO patient_records
                (patient_id, patient_name, scan_timestamp, tumor_detected,
                 tumor_type, confidence, tumor_area_px, brain_coverage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id.strip(),
                patient_name.strip(),
                scan_timestamp,
                tumor_detected,
                tumor_type,
                confidence,
                tumor_area_px,
                brain_coverage,
            ),
        )
        conn.commit()


def query_patient_history(patient_id: str) -> list:
    """
    Returns all Patient_Records for the given ``patient_id``, ordered by
    ``scan_timestamp`` descending (most recent first).

    Args:
        patient_id: The patient identifier to query.

    Returns:
        A list of dicts (one per row).  Returns an empty list when no records
        exist for the given patient ID — never raises on a missing patient.
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT id, patient_id, patient_name, scan_timestamp,
                       tumor_detected, tumor_type, confidence,
                       tumor_area_px, brain_coverage
                FROM   patient_records
                WHERE  patient_id = ?
                ORDER  BY scan_timestamp DESC
                """,
                (patient_id,),
            )
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Table may not exist yet on a fresh DB — return empty list gracefully.
        return []


def delete_all_records() -> int:
    """
    Deletes **all** rows from ``patient_records``.

    Returns:
        The count of rows deleted.

    Raises:
        sqlite3.OperationalError: If the DB file is locked or the table
            cannot be accessed (e.g. concurrent write, missing schema).
        sqlite3.DatabaseError:    On any other DB-level failure (disk full,
            corruption, etc.).
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute("DELETE FROM patient_records")
            conn.commit()
            return cursor.rowcount
    except sqlite3.OperationalError:
        raise
    except sqlite3.DatabaseError:
        raise


# ---------------------------------------------------------------------------
# Module-level safety initialisation
# ---------------------------------------------------------------------------
# Called at import time so that the DB and schema exist before any call
# to insert_patient_record or query_patient_history is made, regardless of
# whether app.py has explicitly called init_db() yet.

try:
    init_db()
except Exception:  # pragma: no cover — environment-level failures (permissions, etc.)
    pass
