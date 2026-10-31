"""
Safe, idempotent schema migration for doctor verification + audit logs.
Does not destroy existing user or scan data.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webapp.database import get_connection


def column_exists(cursor, table: str, column: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS c FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s AND COLUMN_NAME = %s
        """,
        (table, column),
    )
    return cursor.fetchone()["c"] > 0


def table_exists(cursor, table: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS c FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """,
        (table,),
    )
    return cursor.fetchone()["c"] > 0


def migrate():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # --- users.verification_status ---
        if not column_exists(cursor, "users", "verification_status"):
            cursor.execute(
                """
                ALTER TABLE users
                ADD COLUMN verification_status
                    ENUM('none','pending','verified','rejected','suspended')
                    NOT NULL DEFAULT 'none'
                    AFTER role
                """
            )
            print("+ Added users.verification_status")
        else:
            print("= users.verification_status already exists")

        # Backfill: existing doctors/admins are verified; patients none
        cursor.execute(
            """
            UPDATE users SET verification_status = 'verified'
            WHERE role IN ('doctor', 'admin') AND verification_status = 'none'
            """
        )
        cursor.execute(
            """
            UPDATE users SET verification_status = 'none'
            WHERE role = 'patient' AND verification_status NOT IN ('pending','rejected')
            """
        )

        # --- Expand scans.status ---
        cursor.execute(
            """
            ALTER TABLE scans
            MODIFY COLUMN status
                ENUM('pending','approved','rejected','further_review','reviewed')
                NOT NULL DEFAULT 'pending'
            """
        )
        cursor.execute(
            "UPDATE scans SET status = 'approved' WHERE status = 'reviewed'"
        )
        cursor.execute(
            """
            ALTER TABLE scans
            MODIFY COLUMN status
                ENUM('pending','approved','rejected','further_review')
                NOT NULL DEFAULT 'pending'
            """
        )
        print("+ Updated scans.status enum")

        # --- doctor_reviews.decision ---
        if not column_exists(cursor, "doctor_reviews", "decision"):
            cursor.execute(
                """
                ALTER TABLE doctor_reviews
                ADD COLUMN decision
                    ENUM('approved','rejected','further_review')
                    NOT NULL DEFAULT 'approved'
                    AFTER doctor_id
                """
            )
            print("+ Added doctor_reviews.decision")
        else:
            print("= doctor_reviews.decision already exists")

        # --- doctor_verifications ---
        if not table_exists(cursor, "doctor_verifications"):
            cursor.execute(
                """
                CREATE TABLE doctor_verifications (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    full_name VARCHAR(150) NOT NULL,
                    medical_registration_number VARCHAR(100) NOT NULL,
                    qualification VARCHAR(255) NOT NULL,
                    hospital VARCHAR(255) NOT NULL,
                    contact_phone VARCHAR(40) NULL,
                    contact_email VARCHAR(255) NULL,
                    document_path VARCHAR(255) NULL,
                    application_status ENUM('pending','approved','rejected')
                        NOT NULL DEFAULT 'pending',
                    admin_notes TEXT NULL,
                    reviewed_by INT NULL,
                    reviewed_at TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_doctor_verifications_user (user_id),
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL
                )
                """
            )
            print("+ Created doctor_verifications")
        else:
            print("= doctor_verifications already exists")

        # --- audit_logs ---
        if not table_exists(cursor, "audit_logs"):
            cursor.execute(
                """
                CREATE TABLE audit_logs (
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
                """
            )
            print("+ Created audit_logs")
        else:
            print("= audit_logs already exists")

        conn.commit()
        print("Migration complete.")
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    migrate()
