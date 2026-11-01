"""
Create tables (if missing) and seed demo admin/doctor accounts.
Safe to run on every boot.
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.auth import hash_password
from webapp.database import DB_BACKEND, execute_query, get_connection


MYSQL_TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id            INT AUTO_INCREMENT PRIMARY KEY,
        name          VARCHAR(100)  NOT NULL,
        email         VARCHAR(255)  NOT NULL UNIQUE,
        password_hash VARCHAR(255)  NOT NULL,
        role          ENUM('patient','doctor','admin') NOT NULL DEFAULT 'patient',
        verification_status ENUM('none','pending','verified','rejected','suspended')
                            NOT NULL DEFAULT 'none',
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scans (
        id         INT AUTO_INCREMENT PRIMARY KEY,
        patient_id INT NOT NULL,
        filename   VARCHAR(255) NOT NULL,
        image_path VARCHAR(255) NOT NULL,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status     ENUM('pending','approved','rejected','further_review') NOT NULL DEFAULT 'pending',
        FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_results (
        id             INT AUTO_INCREMENT PRIMARY KEY,
        scan_id        INT NOT NULL UNIQUE,
        tumor_detected BOOLEAN NOT NULL DEFAULT FALSE,
        tumor_type     VARCHAR(50) NOT NULL,
        confidence     FLOAT NULL,
        gradcam_path   VARCHAR(255) NULL,
        overlay_path   VARCHAR(255) NULL,
        FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doctor_reviews (
        id          INT AUTO_INCREMENT PRIMARY KEY,
        scan_id     INT NOT NULL UNIQUE,
        doctor_id   INT NOT NULL,
        decision    ENUM('approved','rejected','further_review') NOT NULL DEFAULT 'approved',
        diagnosis   TEXT NOT NULL,
        notes       TEXT,
        reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
        FOREIGN KEY (doctor_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doctor_verifications (
        id INT AUTO_INCREMENT PRIMARY KEY,
        user_id INT NOT NULL,
        full_name VARCHAR(150) NOT NULL,
        medical_registration_number VARCHAR(100) NOT NULL,
        qualification VARCHAR(255) NOT NULL,
        hospital VARCHAR(255) NOT NULL,
        contact_phone VARCHAR(40) NULL,
        contact_email VARCHAR(255) NULL,
        document_path VARCHAR(255) NULL,
        application_status ENUM('pending','approved','rejected') NOT NULL DEFAULT 'pending',
        admin_notes TEXT NULL,
        reviewed_by INT NULL,
        reviewed_at TIMESTAMP NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE KEY uq_doctor_verifications_user (user_id),
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INT AUTO_INCREMENT PRIMARY KEY,
        actor_id INT NULL,
        patient_id INT NULL,
        report_id INT NULL,
        action VARCHAR(100) NOT NULL,
        comments TEXT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (actor_id) REFERENCES users(id) ON DELETE SET NULL,
        FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE SET NULL,
        FOREIGN KEY (report_id) REFERENCES scans(id) ON DELETE SET NULL
    )
    """,
]

SQLITE_TABLE_SQL = [
    """
    CREATE TABLE IF NOT EXISTS users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT NOT NULL,
        email         TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL,
        role          TEXT NOT NULL DEFAULT 'patient',
        verification_status TEXT NOT NULL DEFAULT 'none',
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS scans (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        filename   TEXT NOT NULL,
        image_path TEXT NOT NULL,
        upload_time TEXT DEFAULT CURRENT_TIMESTAMP,
        status     TEXT NOT NULL DEFAULT 'pending',
        FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_results (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id        INTEGER NOT NULL UNIQUE,
        tumor_detected INTEGER NOT NULL DEFAULT 0,
        tumor_type     TEXT NOT NULL,
        confidence     REAL,
        gradcam_path   TEXT,
        overlay_path   TEXT,
        FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doctor_reviews (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id     INTEGER NOT NULL UNIQUE,
        doctor_id   INTEGER NOT NULL,
        decision    TEXT NOT NULL DEFAULT 'approved',
        diagnosis   TEXT NOT NULL,
        notes       TEXT,
        reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
        FOREIGN KEY (doctor_id) REFERENCES users(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS doctor_verifications (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        full_name TEXT NOT NULL,
        medical_registration_number TEXT NOT NULL,
        qualification TEXT NOT NULL,
        hospital TEXT NOT NULL,
        contact_phone TEXT,
        contact_email TEXT,
        document_path TEXT,
        application_status TEXT NOT NULL DEFAULT 'pending',
        admin_notes TEXT,
        reviewed_by INTEGER,
        reviewed_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        actor_id INTEGER,
        patient_id INTEGER,
        report_id INTEGER,
        action TEXT NOT NULL,
        comments TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (actor_id) REFERENCES users(id) ON DELETE SET NULL,
        FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE SET NULL,
        FOREIGN KEY (report_id) REFERENCES scans(id) ON DELETE SET NULL
    )
    """,
]

SEED_USERS = [
    ("Admin", "admin@brainscan.ai", "admin123", "admin", "verified"),
    ("Dr Demo", "doctor@brainscan.ai", "doctor123", "doctor", "verified"),
]


def ensure_tables() -> None:
    table_sql = SQLITE_TABLE_SQL if DB_BACKEND == "sqlite" else MYSQL_TABLE_SQL
    conn = get_connection()
    try:
        if DB_BACKEND == "sqlite":
            cursor = conn.cursor()
            try:
                for stmt in table_sql:
                    cursor.execute(stmt)
                conn.commit()
            finally:
                cursor.close()
        else:
            cursor = conn.cursor()
            try:
                for stmt in table_sql:
                    cursor.execute(stmt)
                conn.commit()
            finally:
                cursor.close()
        print(f"Database tables ready ({DB_BACKEND}).")
    finally:
        conn.close()


def seed_demo_users() -> None:
    for name, email, password, role, verification in SEED_USERS:
        existing = execute_query(
            "SELECT id FROM users WHERE email = %s",
            (email,),
            fetch=True,
        )
        if existing:
            print(f"= User exists: {email}")
            continue
        execute_query(
            """
            INSERT INTO users (name, email, password_hash, role, verification_status)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (name, email, hash_password(password), role, verification),
        )
        print(f"+ Seeded {role}: {email}")


def init_database() -> None:
    ensure_tables()
    if DB_BACKEND == "mysql":
        try:
            from webapp.migrate_schema import migrate

            migrate()
        except Exception as exc:
            # Fresh installs already match schema; migration is best-effort.
            print(f"Migration skipped/failed (ok on fresh DB): {exc}")
    else:
        print("SQLite schema created from scratch; MySQL migration skipped.")
    seed_demo_users()


if __name__ == "__main__":
    init_database()
