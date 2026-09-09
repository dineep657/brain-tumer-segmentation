"""
auth_routes.py — Register and login endpoints.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr
from webapp.database import execute_query
from webapp.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter()


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    # Role is always patient on public registration.
    # Doctor accounts are created only via the /admin/create-doctor endpoint.


class LoginRequest(BaseModel):
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
    execute_query(
        "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'patient')",
        (req.name, req.email, hashed),
    )
    return {"message": "Registration successful. Please log in."}


@router.post("/login")
def login(req: LoginRequest):
    rows = execute_query(
        "SELECT id, name, email, password_hash, role FROM users WHERE email = %s",
        (req.email,),
        fetch=True,
    )
    if not rows:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user = rows[0]
    if not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"user_id": user["id"], "role": user["role"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "name": user["name"],
    }


@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the current user's profile from the DB (used by frontend to verify role)."""
    return {
        "id": current_user["id"],
        "name": current_user["name"],
        "email": current_user["email"],
        "role": current_user["role"],
    }


class CreateDoctorRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    admin_secret: str  # simple shared secret to authorize doctor creation


ADMIN_SECRET = "brainscan_admin_2024"  # Change this in production


@router.post("/admin/create-doctor")
def create_doctor(req: CreateDoctorRequest):
    """
    Admin-only endpoint to create doctor accounts.
    Requires a shared admin secret — not exposed in the UI.
    """
    if req.admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")

    existing = execute_query(
        "SELECT id FROM users WHERE email = %s", (req.email,), fetch=True
    )
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(req.password)
    execute_query(
        "INSERT INTO users (name, email, password_hash, role) VALUES (%s, %s, %s, 'doctor')",
        (req.name, req.email, hashed),
    )
    return {"message": f"Doctor account created for {req.email}"}


@router.get("/admin/list-doctors")
def list_doctors(admin_secret: str):
    """Returns all doctor accounts — admin only."""
    if admin_secret != ADMIN_SECRET:
        raise HTTPException(status_code=403, detail="Invalid admin secret")
    rows = execute_query(
        "SELECT id, name, email, created_at FROM users WHERE role = 'doctor' ORDER BY created_at DESC",
        fetch=True,
    )
    return rows
