"""
auth.py — JWT authentication and role/verification checks.
Roles and verification status are always loaded from the database,
never trusted solely from the JWT payload.
"""

import os
from datetime import datetime, timedelta
import bcrypt
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from webapp.database import execute_query

SECRET_KEY = os.environ.get("BT_JWT_SECRET", "brain_tumor_app_secret_key_2024_change_in_prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(data: dict) -> str:
    from datetime import timezone
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    rows = execute_query(
        """
        SELECT id, name, email, role, verification_status
        FROM users WHERE id = %s
        """,
        (user_id,),
        fetch=True,
    )
    if not rows:
        raise credentials_exception

    user = rows[0]
    if user.get("verification_status") == "suspended":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account suspended. Contact the administrator.",
        )
    return user


def require_role(role: str):
    """Require an exact role from the live database record."""
    def checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Requires role: {role}",
            )
        return current_user
    return checker


def require_verified_doctor(current_user: dict = Depends(get_current_user)):
    """Doctor endpoints: role must be doctor AND verification_status must be verified."""
    if current_user["role"] != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied. Requires verified doctor account.",
        )
    if current_user.get("verification_status") != "verified":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor account is not verified. Await administrator approval.",
        )
    return current_user


def require_admin(current_user: dict = Depends(get_current_user)):
    """Admin-only routes — role checked from database."""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required.",
        )
    return current_user
