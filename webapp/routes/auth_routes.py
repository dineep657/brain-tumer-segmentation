"""
auth_routes.py — Register, login, doctor application endpoints.
Public registration always creates a patient. Roles from the client are ignored.
"""

import os
import uuid
import shutil

from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from webapp.database import execute_query, write_audit_log
from webapp.auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
)

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_UPLOAD_DIR = os.path.join(BASE_DIR, "uploads", "doctor_docs")
os.makedirs(DOC_UPLOAD_DIR, exist_ok=True)


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # silently drop client-supplied role, etc.
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)
    # Intentionally no role field — never trust client-supplied roles.


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    email: EmailStr
    password: str


@router.post("/register")
def register(req: RegisterRequest):
    """Public registration — always creates a patient account."""
    existing = execute_query(
        "SELECT id FROM users WHERE email = %s", (req.email,), fetch=True
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(req.password)
    user_id = execute_query(
        """
        INSERT INTO users (name, email, password_hash, role, verification_status)
        VALUES (%s, %s, %s, 'patient', 'none')
        """,
        (req.name.strip(), req.email.lower().strip(), hashed),
    )
    write_audit_log(user_id, "patient_registered", f"Patient registered: {req.email}")
    return {"message": "Registration successful. Please log in."}


@router.post("/login")
def login(req: LoginRequest):
    rows = execute_query(
        """
        SELECT id, name, email, password_hash, role, verification_status
        FROM users WHERE email = %s
        """,
        (req.email.lower().strip(),),
        fetch=True,
    )
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = rows[0]
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if user.get("verification_status") == "suspended":
        raise HTTPException(
            status_code=403,
            detail="Account suspended. Contact the administrator.",
        )

    # JWT carries only identity; authorization always re-checks the DB.
    token = create_access_token({"user_id": user["id"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "verification_status": user.get("verification_status", "none"),
        "name": user["name"],
    }


@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Current profile from the database (source of truth for role)."""
    app = execute_query(
        """
        SELECT id, application_status, medical_registration_number, hospital,
               qualification, created_at, reviewed_at, admin_notes
        FROM doctor_verifications
        WHERE user_id = %s
        ORDER BY created_at DESC LIMIT 1
        """,
        (current_user["id"],),
        fetch=True,
    )
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "email": current_user["email"],
        "role": current_user["role"],
        "verification_status": current_user.get("verification_status", "none"),
        "doctor_application": app[0] if app else None,
    }


@router.post("/apply-doctor")
async def apply_doctor(
    full_name: str = Form(...),
    email: EmailStr = Form(...),
    password: str = Form(...),
    medical_registration_number: str = Form(...),
    qualification: str = Form(...),
    hospital: str = Form(...),
    contact_phone: str = Form(""),
    document: UploadFile | None = File(None),
):
    """
    Doctor access request. Creates a patient account (or reuses existing patient)
    with verification_status=pending. No doctor permissions until admin approval.
    """
    email_norm = email.lower().strip()
    full_name = full_name.strip()
    reg_no = medical_registration_number.strip()
    qualification = qualification.strip()
    hospital = hospital.strip()

    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if not reg_no or not qualification or not hospital or not full_name:
        raise HTTPException(status_code=400, detail="All required fields must be provided")

    existing = execute_query(
        "SELECT id, role, verification_status FROM users WHERE email = %s",
        (email_norm,),
        fetch=True,
    )

    if existing:
        user = existing[0]
        if user["role"] in ("doctor", "admin"):
            raise HTTPException(
                status_code=400,
                detail="This account already has elevated privileges.",
            )
        if user["verification_status"] == "pending":
            raise HTTPException(
                status_code=400,
                detail="A doctor verification request is already pending for this email.",
            )
        if user["verification_status"] == "verified" and user["role"] == "doctor":
            raise HTTPException(status_code=400, detail="Already a verified doctor.")

        # Re-authenticate ownership of existing patient account
        rows = execute_query(
            "SELECT password_hash FROM users WHERE id = %s", (user["id"],), fetch=True
        )
        if not verify_password(password, rows[0]["password_hash"]):
            raise HTTPException(
                status_code=401,
                detail="Email already registered. Enter the correct password to apply.",
            )
        user_id = user["id"]
        execute_query(
            "UPDATE users SET name=%s, verification_status='pending' WHERE id=%s",
            (full_name, user_id),
        )
    else:
        hashed = hash_password(password)
        user_id = execute_query(
            """
            INSERT INTO users (name, email, password_hash, role, verification_status)
            VALUES (%s, %s, %s, 'patient', 'pending')
            """,
            (full_name, email_norm, hashed),
        )

    # Prevent duplicate open applications
    open_app = execute_query(
        """
        SELECT id FROM doctor_verifications
        WHERE user_id = %s AND application_status = 'pending'
        """,
        (user_id,),
        fetch=True,
    )
    if open_app:
        raise HTTPException(status_code=400, detail="You already have a pending application.")

    doc_path = None
    if document and document.filename:
        ext = os.path.splitext(document.filename)[1].lower()
        if ext not in (".pdf", ".png", ".jpg", ".jpeg"):
            raise HTTPException(
                status_code=400,
                detail="Verification document must be PDF or image (PNG/JPG).",
            )
        doc_name = f"doc_{user_id}_{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(DOC_UPLOAD_DIR, doc_name)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(document.file, f)
        doc_path = f"doctor_docs/{doc_name}"

    app_id = execute_query(
        """
        INSERT INTO doctor_verifications
            (user_id, full_name, medical_registration_number, qualification,
             hospital, contact_phone, contact_email, document_path, application_status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
        """,
        (
            user_id,
            full_name,
            reg_no,
            qualification,
            hospital,
            contact_phone.strip() or None,
            email_norm,
            doc_path,
        ),
    )

    write_audit_log(
        user_id,
        "doctor_application_submitted",
        f"Application #{app_id} for {full_name} / reg {reg_no}",
    )

    return {
        "message": (
            "Doctor verification request submitted. "
            "Your account remains a patient account until an administrator approves it."
        ),
        "application_id": app_id,
        "status": "pending",
    }
