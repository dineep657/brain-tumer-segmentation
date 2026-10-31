"""
RBAC and doctor-verification security tests.
Uses the live MySQL database with temporary users that are cleaned up.
"""

import os
import sys
import uuid

import pytest
from fastapi.testclient import TestClient

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from webapp.main import app
from webapp.database import execute_query
from webapp.auth import hash_password, create_access_token

client = TestClient(app)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}@example.com"


def _create_user(name, email, role, verification_status="none", password="secret123"):
    return execute_query(
        """
        INSERT INTO users (name, email, password_hash, role, verification_status)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (name, email, hash_password(password), role, verification_status),
    )


def _delete_user(user_id):
    if user_id:
        execute_query("DELETE FROM users WHERE id = %s", (user_id,))


def _auth(user_id):
    return {"Authorization": f"Bearer {create_access_token({'user_id': user_id})}"}


@pytest.fixture
def patient():
    email = _uid("patient")
    uid = _create_user("Test Patient", email, "patient", "none")
    yield {"id": uid, "email": email, "password": "secret123"}
    _delete_user(uid)


@pytest.fixture
def pending_applicant():
    email = _uid("pending")
    uid = _create_user("Pending Doc", email, "patient", "pending")
    execute_query(
        """
        INSERT INTO doctor_verifications
          (user_id, full_name, medical_registration_number, qualification, hospital, application_status)
        VALUES (%s, %s, %s, %s, %s, 'pending')
        """,
        (uid, "Pending Doc", "REG-TEST", "MD", "Test Hospital"),
    )
    yield {"id": uid, "email": email}
    _delete_user(uid)


@pytest.fixture
def verified_doctor():
    email = _uid("doctor")
    uid = _create_user("Verified Doctor", email, "doctor", "verified")
    yield {"id": uid, "email": email, "password": "secret123"}
    _delete_user(uid)


@pytest.fixture
def admin_user():
    email = _uid("admin")
    uid = _create_user("Test Admin", email, "admin", "verified")
    yield {"id": uid, "email": email}
    _delete_user(uid)


def test_register_ignores_role_and_creates_patient():
    email = _uid("reg")
    res = client.post(
        "/api/auth/register",
        json={"name": "New User", "email": email, "password": "secret123", "role": "doctor"},
    )
    assert res.status_code == 200
    rows = execute_query(
        "SELECT role, verification_status FROM users WHERE email=%s", (email,), fetch=True
    )
    assert rows[0]["role"] == "patient"
    assert rows[0]["verification_status"] == "none"
    execute_query("DELETE FROM users WHERE email=%s", (email,))


def test_patient_cannot_access_doctor_queue(patient):
    res = client.get("/api/doctor/queue", headers=_auth(patient["id"]))
    assert res.status_code == 403


def test_patient_cannot_access_admin(patient):
    res = client.get("/api/admin/stats", headers=_auth(patient["id"]))
    assert res.status_code == 403


def test_pending_doctor_cannot_access_queue(pending_applicant):
    # Still a patient role while pending
    res = client.get("/api/doctor/queue", headers=_auth(pending_applicant["id"]))
    assert res.status_code == 403


def test_unverified_doctor_role_blocked():
    email = _uid("unverified")
    uid = _create_user("Unverified", email, "doctor", "pending")
    try:
        res = client.get("/api/doctor/queue", headers=_auth(uid))
        assert res.status_code == 403
        assert "not verified" in res.json()["detail"].lower()
    finally:
        _delete_user(uid)


def test_verified_doctor_can_access_queue(verified_doctor):
    res = client.get("/api/doctor/queue", headers=_auth(verified_doctor["id"]))
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_doctor_cannot_review_own_scan(verified_doctor):
    # Create a scan owned by the doctor user (edge case)
    scan_id = execute_query(
        "INSERT INTO scans (patient_id, filename, image_path, status) VALUES (%s,%s,%s,'pending')",
        (verified_doctor["id"], "self.png", "self.png"),
    )
    try:
        res = client.post(
            f"/api/doctor/review/{scan_id}",
            headers=_auth(verified_doctor["id"]),
            json={"decision": "approved", "diagnosis": "Self approve", "notes": ""},
        )
        assert res.status_code == 403
    finally:
        execute_query("DELETE FROM scans WHERE id=%s", (scan_id,))


def test_admin_approves_doctor_application(admin_user, pending_applicant):
    apps = execute_query(
        "SELECT id FROM doctor_verifications WHERE user_id=%s AND application_status='pending'",
        (pending_applicant["id"],),
        fetch=True,
    )
    assert apps
    app_id = apps[0]["id"]
    res = client.post(
        f"/api/admin/doctor-applications/{app_id}/approve",
        headers=_auth(admin_user["id"]),
        json={"admin_notes": "Credentials reviewed"},
    )
    assert res.status_code == 200
    user = execute_query(
        "SELECT role, verification_status FROM users WHERE id=%s",
        (pending_applicant["id"],),
        fetch=True,
    )[0]
    assert user["role"] == "doctor"
    assert user["verification_status"] == "verified"

    # Now doctor endpoints work
    res2 = client.get("/api/doctor/queue", headers=_auth(pending_applicant["id"]))
    assert res2.status_code == 200


def test_login_returns_verification_status(patient):
    res = client.post(
        "/api/auth/login",
        json={"email": patient["email"], "password": patient["password"]},
    )
    assert res.status_code == 200
    body = res.json()
    assert "access_token" in body
    assert body["role"] == "patient"
    assert body["verification_status"] == "none"


def test_jwt_role_claim_ignored_for_authorization(patient):
    """Even a forged role in JWT must not elevate privileges — DB is source of truth."""
    forged = create_access_token({"user_id": patient["id"], "role": "admin"})
    res = client.get(
        "/api/admin/stats",
        headers={"Authorization": f"Bearer {forged}"},
    )
    assert res.status_code == 403
