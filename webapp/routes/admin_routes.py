"""
admin_routes.py — Admin portal: users, doctor verification, scans, audit.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr, Field
from webapp.auth import require_admin, hash_password
from webapp.database import execute_query, write_audit_log


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

router = APIRouter()


class CreateDoctorRequest(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class ApplicationDecision(BaseModel):
    admin_notes: str = Field(default="", max_length=2000)


class SuspendRequest(BaseModel):
    reason: str = Field(default="", max_length=1000)


@router.get("/stats")
def get_stats(current_user: dict = Depends(require_admin)):
    patients = execute_query(
        "SELECT COUNT(*) as c FROM users WHERE role='patient'", fetch=True
    )[0]["c"]
    doctors = execute_query(
        """
        SELECT COUNT(*) as c FROM users
        WHERE role='doctor' AND verification_status='verified'
        """,
        fetch=True,
    )[0]["c"]
    pending_apps = execute_query(
        """
        SELECT COUNT(*) as c FROM doctor_verifications
        WHERE application_status='pending'
        """,
        fetch=True,
    )[0]["c"]
    scans = execute_query("SELECT COUNT(*) as c FROM scans", fetch=True)[0]["c"]
    pending = execute_query(
        "SELECT COUNT(*) as c FROM scans WHERE status IN ('pending','further_review')",
        fetch=True,
    )[0]["c"]
    reviewed = execute_query(
        "SELECT COUNT(*) as c FROM scans WHERE status IN ('approved','rejected')",
        fetch=True,
    )[0]["c"]
    return {
        "patients": patients,
        "doctors": doctors,
        "pending_applications": pending_apps,
        "total_scans": scans,
        "pending_reviews": pending,
        "reviewed": reviewed,
    }


@router.get("/users")
def get_all_users(role: str = None, current_user: dict = Depends(require_admin)):
    if role:
        rows = execute_query(
            """
            SELECT id, name, email, role, verification_status, created_at
            FROM users WHERE role = %s ORDER BY created_at DESC
            """,
            (role,),
            fetch=True,
        )
    else:
        rows = execute_query(
            """
            SELECT id, name, email, role, verification_status, created_at
            FROM users ORDER BY created_at DESC
            """,
            fetch=True,
        )
    return rows


@router.post("/create-doctor")
def create_doctor(req: CreateDoctorRequest, current_user: dict = Depends(require_admin)):
    """Admin-provisioned doctor — immediately verified."""
    existing = execute_query(
        "SELECT id FROM users WHERE email = %s", (req.email.lower().strip(),), fetch=True
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(req.password)
    user_id = execute_query(
        """
        INSERT INTO users (name, email, password_hash, role, verification_status)
        VALUES (%s, %s, %s, 'doctor', 'verified')
        """,
        (req.name.strip(), req.email.lower().strip(), hashed),
    )
    write_audit_log(
        current_user["id"],
        "admin_created_doctor",
        f"Created verified doctor user_id={user_id} email={req.email}",
        patient_id=user_id,
    )
    return {"message": f"Verified doctor account created for {req.email}"}


@router.get("/doctor-applications")
def list_doctor_applications(
    status_filter: str = "pending",
    current_user: dict = Depends(require_admin),
):
    if status_filter and status_filter != "all":
        rows = execute_query(
            """
            SELECT dv.*, u.email AS user_email, u.role AS user_role,
                   u.verification_status, rev.name AS reviewer_name
            FROM doctor_verifications dv
            JOIN users u ON u.id = dv.user_id
            LEFT JOIN users rev ON rev.id = dv.reviewed_by
            WHERE dv.application_status = %s
            ORDER BY dv.created_at ASC
            """,
            (status_filter,),
            fetch=True,
        )
    else:
        rows = execute_query(
            """
            SELECT dv.*, u.email AS user_email, u.role AS user_role,
                   u.verification_status, rev.name AS reviewer_name
            FROM doctor_verifications dv
            JOIN users u ON u.id = dv.user_id
            LEFT JOIN users rev ON rev.id = dv.reviewed_by
            ORDER BY
              CASE dv.application_status
                WHEN 'pending' THEN 0
                WHEN 'approved' THEN 1
                ELSE 2
              END,
              dv.created_at DESC
            """,
            fetch=True,
        )
    return rows


@router.post("/doctor-applications/{application_id}/approve")
def approve_doctor_application(
    application_id: int,
    body: ApplicationDecision = ApplicationDecision(),
    current_user: dict = Depends(require_admin),
):
    apps = execute_query(
        "SELECT * FROM doctor_verifications WHERE id = %s",
        (application_id,),
        fetch=True,
    )
    if not apps:
        raise HTTPException(status_code=404, detail="Application not found")
    app = apps[0]
    if app["application_status"] != "pending":
        raise HTTPException(status_code=400, detail="Application is not pending")

    execute_query(
        """
        UPDATE doctor_verifications
        SET application_status='approved', admin_notes=%s,
            reviewed_by=%s, reviewed_at=%s
        WHERE id=%s
        """,
        (body.admin_notes or None, current_user["id"], _utc_now(), application_id),
    )
    execute_query(
        """
        UPDATE users
        SET role='doctor', verification_status='verified', name=%s
        WHERE id=%s
        """,
        (app["full_name"], app["user_id"]),
    )
    write_audit_log(
        current_user["id"],
        "doctor_application_approved",
        body.admin_notes or f"Approved application #{application_id}",
        patient_id=app["user_id"],
    )
    return {"message": "Doctor application approved. User now has verified doctor access."}


@router.post("/doctor-applications/{application_id}/reject")
def reject_doctor_application(
    application_id: int,
    body: ApplicationDecision = ApplicationDecision(),
    current_user: dict = Depends(require_admin),
):
    apps = execute_query(
        "SELECT * FROM doctor_verifications WHERE id = %s",
        (application_id,),
        fetch=True,
    )
    if not apps:
        raise HTTPException(status_code=404, detail="Application not found")
    app = apps[0]
    if app["application_status"] != "pending":
        raise HTTPException(status_code=400, detail="Application is not pending")

    execute_query(
        """
        UPDATE doctor_verifications
        SET application_status='rejected', admin_notes=%s,
            reviewed_by=%s, reviewed_at=%s
        WHERE id=%s
        """,
        (body.admin_notes or None, current_user["id"], _utc_now(), application_id),
    )
    execute_query(
        """
        UPDATE users SET verification_status='rejected', role='patient'
        WHERE id=%s
        """,
        (app["user_id"],),
    )
    write_audit_log(
        current_user["id"],
        "doctor_application_rejected",
        body.admin_notes or f"Rejected application #{application_id}",
        patient_id=app["user_id"],
    )
    return {"message": "Doctor application rejected."}


@router.post("/doctors/{user_id}/suspend")
def suspend_doctor(
    user_id: int,
    body: SuspendRequest = SuspendRequest(),
    current_user: dict = Depends(require_admin),
):
    rows = execute_query(
        "SELECT id, role FROM users WHERE id = %s", (user_id,), fetch=True
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    if rows[0]["role"] != "doctor":
        raise HTTPException(status_code=400, detail="User is not a doctor")

    execute_query(
        "UPDATE users SET verification_status='suspended' WHERE id=%s",
        (user_id,),
    )
    write_audit_log(
        current_user["id"],
        "doctor_suspended",
        body.reason or "Suspended by admin",
        patient_id=user_id,
    )
    return {"message": "Doctor access suspended."}


@router.post("/doctors/{user_id}/revoke")
def revoke_doctor(
    user_id: int,
    body: SuspendRequest = SuspendRequest(),
    current_user: dict = Depends(require_admin),
):
    rows = execute_query(
        "SELECT id, role FROM users WHERE id = %s", (user_id,), fetch=True
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    if rows[0]["role"] != "doctor":
        raise HTTPException(status_code=400, detail="User is not a doctor")

    execute_query(
        """
        UPDATE users SET role='patient', verification_status='none'
        WHERE id=%s
        """,
        (user_id,),
    )
    write_audit_log(
        current_user["id"],
        "doctor_revoked",
        body.reason or "Doctor privileges revoked",
        patient_id=user_id,
    )
    return {"message": "Doctor privileges revoked. Account reverted to patient."}


@router.delete("/user/{user_id}")
def delete_user(user_id: int, current_user: dict = Depends(require_admin)):
    if user_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    rows = execute_query(
        "SELECT role FROM users WHERE id = %s", (user_id,), fetch=True
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    if rows[0]["role"] == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete admin accounts")
    execute_query("DELETE FROM users WHERE id = %s", (user_id,))
    write_audit_log(
        current_user["id"],
        "user_deleted",
        f"Deleted user_id={user_id}",
    )
    return {"message": "User deleted"}


@router.get("/scans")
def get_all_scans(current_user: dict = Depends(require_admin)):
    rows = execute_query(
        """
        SELECT s.id, s.filename, s.upload_time, s.status,
               u.name AS patient_name, u.email AS patient_email,
               r.tumor_detected, r.tumor_type, r.confidence,
               dr.decision, dr.diagnosis, d.name AS doctor_name
        FROM scans s
        JOIN users u ON u.id = s.patient_id
        LEFT JOIN ai_results r ON r.scan_id = s.id
        LEFT JOIN doctor_reviews dr ON dr.scan_id = s.id
        LEFT JOIN users d ON d.id = dr.doctor_id
        ORDER BY s.upload_time DESC
        """,
        fetch=True,
    )
    return rows


@router.get("/audit-logs")
def get_audit_logs(limit: int = 100, current_user: dict = Depends(require_admin)):
    limit = max(1, min(limit, 500))
    rows = execute_query(
        """
        SELECT a.id, a.action, a.comments, a.created_at,
               a.actor_id, a.patient_id, a.report_id,
               actor.name AS actor_name,
               patient.name AS patient_name
        FROM audit_logs a
        LEFT JOIN users actor ON actor.id = a.actor_id
        LEFT JOIN users patient ON patient.id = a.patient_id
        ORDER BY a.created_at DESC
        LIMIT %s
        """,
        (limit,),
        fetch=True,
    )
    return rows
