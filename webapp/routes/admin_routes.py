"""
admin_routes.py — Admin portal API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from webapp.auth import require_admin, hash_password
from webapp.database import execute_query

router = APIRouter()


class CreateDoctorRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


@router.get("/stats")
def get_stats(current_user: dict = Depends(require_admin)):
    patients = execute_query("SELECT COUNT(*) as c FROM users WHERE role='patient'", fetch=True)[0]["c"]
    doctors  = execute_query("SELECT COUNT(*) as c FROM users WHERE role='doctor'",  fetch=True)[0]["c"]
    scans    = execute_query("SELECT COUNT(*) as c FROM scans", fetch=True)[0]["c"]
    pending  = execute_query("SELECT COUNT(*) as c FROM scans WHERE status='pending'", fetch=True)[0]["c"]
    reviewed = execute_query("SELECT COUNT(*) as c FROM scans WHERE status='reviewed'", fetch=True)[0]["c"]
    return {"patients": patients, "doctors": doctors, "total_scans": scans, "pending_reviews": pending, "reviewed": reviewed}


@router.get("/users")
def get_all_users(role: str = None, current_user: dict = Depends(require_admin)):
    if role:
        rows = execute_query(
            "SELECT id, name, email, role, created_at FROM users WHERE role = %s ORDER BY created_at DESC",
            (role,), fetch=True,
        )
    else:
        rows = execute_query(
            "SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC",
            fetch=True,
        )
    return rows


@router.post("/create-doctor")
def create_doctor(req: CreateDoctorRequest, current_user: dict = Depends(require_admin)):
    existing = execute_query("SELECT id FROM users WHERE email = %s", (req.email,), fetch=True)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed = hash_password(req.password)
    execute_query(
        "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'doctor')",
        (req.name, req.email, hashed),
    )
    return {"message": f"Doctor account created for {req.email}"}


@router.delete("/user/{user_id}")
def delete_user(user_id: int, current_user: dict = Depends(require_admin)):
    if user_id == current_user["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    rows = execute_query("SELECT role FROM users WHERE id = %s", (user_id,), fetch=True)
    if not rows:
        raise HTTPException(status_code=404, detail="User not found")
    if rows[0]["role"] == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete admin accounts")
    execute_query("DELETE FROM users WHERE id = %s", (user_id,))
    return {"message": "User deleted"}


@router.get("/scans")
def get_all_scans(current_user: dict = Depends(require_admin)):
    rows = execute_query(
        """SELECT s.id, s.filename, s.upload_time, s.status,
                  u.name AS patient_name, u.email AS patient_email,
                  r.tumor_detected, r.tumor_type, r.confidence,
                  dr.diagnosis, d.name AS doctor_name
           FROM scans s
           JOIN users u ON u.id = s.patient_id
           LEFT JOIN ai_results r ON r.scan_id = s.id
           LEFT JOIN doctor_reviews dr ON dr.scan_id = s.id
           LEFT JOIN users d ON d.id = dr.doctor_id
           ORDER BY s.upload_time DESC""",
        fetch=True,
    )
    return rows
