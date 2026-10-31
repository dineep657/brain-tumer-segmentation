"""
doctor_routes.py — Verified-doctor review queue and diagnosis submission.
"""

from typing import Literal
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from webapp.auth import require_verified_doctor
from webapp.database import execute_query, write_audit_log

router = APIRouter()

VALID_DECISIONS = ("approved", "rejected", "further_review")


class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected", "further_review"]
    diagnosis: str = Field(min_length=1, max_length=2000)
    notes: str = Field(default="", max_length=5000)


@router.get("/queue")
def get_review_queue(current_user: dict = Depends(require_verified_doctor)):
    """Pending and further-review scans awaiting doctor action."""
    rows = execute_query(
        """
        SELECT s.id AS scan_id, s.filename, s.upload_time, s.status,
               u.name AS patient_name, u.email AS patient_email,
               r.tumor_detected, r.tumor_type, r.confidence,
               r.gradcam_path, r.overlay_path
        FROM scans s
        JOIN users u ON u.id = s.patient_id
        LEFT JOIN ai_results r ON r.scan_id = s.id
        WHERE s.status IN ('pending', 'further_review')
          AND s.patient_id <> %s
        ORDER BY
          CASE WHEN s.status = 'further_review' THEN 0 ELSE 1 END,
          s.upload_time ASC
        """,
        (current_user["id"],),
        fetch=True,
    )
    return rows


@router.get("/case/{scan_id}")
def get_case(scan_id: int, current_user: dict = Depends(require_verified_doctor)):
    """Full case detail for review."""
    rows = execute_query(
        """
        SELECT s.id AS scan_id, s.filename, s.image_path, s.upload_time, s.status,
               s.patient_id,
               u.name AS patient_name, u.email AS patient_email,
               r.tumor_detected, r.tumor_type, r.confidence,
               r.gradcam_path, r.overlay_path,
               dr.decision, dr.diagnosis, dr.notes, dr.reviewed_at,
               d.name AS doctor_name
        FROM scans s
        JOIN users u ON u.id = s.patient_id
        LEFT JOIN ai_results r ON r.scan_id = s.id
        LEFT JOIN doctor_reviews dr ON dr.scan_id = s.id
        LEFT JOIN users d ON d.id = dr.doctor_id
        WHERE s.id = %s
        """,
        (scan_id,),
        fetch=True,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Case not found")
    if rows[0]["patient_id"] == current_user["id"]:
        raise HTTPException(
            status_code=403,
            detail="Doctors cannot review their own patient scans.",
        )
    return rows[0]


@router.post("/review/{scan_id}")
def submit_review(
    scan_id: int,
    req: ReviewRequest,
    current_user: dict = Depends(require_verified_doctor),
):
    """Submit doctor's decision for a scan. Writes an audit log entry."""
    if req.decision not in VALID_DECISIONS:
        raise HTTPException(status_code=400, detail="Invalid decision")

    scans = execute_query(
        "SELECT id, status, patient_id FROM scans WHERE id = %s",
        (scan_id,),
        fetch=True,
    )
    if not scans:
        raise HTTPException(status_code=404, detail="Scan not found")

    patient_id = scans[0]["patient_id"]
    if patient_id == current_user["id"]:
        raise HTTPException(
            status_code=403,
            detail="Doctors cannot review their own scans.",
        )

    existing = execute_query(
        "SELECT id FROM doctor_reviews WHERE scan_id = %s", (scan_id,), fetch=True
    )
    if existing:
        execute_query(
            """
            UPDATE doctor_reviews
            SET doctor_id=%s, decision=%s, diagnosis=%s, notes=%s, reviewed_at=NOW()
            WHERE scan_id=%s
            """,
            (current_user["id"], req.decision, req.diagnosis, req.notes, scan_id),
        )
    else:
        execute_query(
            """
            INSERT INTO doctor_reviews (scan_id, doctor_id, decision, diagnosis, notes)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (scan_id, current_user["id"], req.decision, req.diagnosis, req.notes),
        )

    execute_query(
        "UPDATE scans SET status=%s WHERE id=%s",
        (req.decision, scan_id),
    )

    write_audit_log(
        actor_id=current_user["id"],
        patient_id=patient_id,
        report_id=scan_id,
        action=f"doctor_review_{req.decision}",
        comments=f"Diagnosis: {req.diagnosis}. Notes: {req.notes or '—'}",
    )

    return {
        "message": "Review submitted successfully.",
        "decision": req.decision,
        "scan_status": req.decision,
    }


@router.get("/history")
def get_reviewed_cases(current_user: dict = Depends(require_verified_doctor)):
    """All cases reviewed by this doctor."""
    rows = execute_query(
        """
        SELECT s.id AS scan_id, s.filename, s.upload_time, s.status,
               u.name AS patient_name,
               r.tumor_type, r.confidence,
               dr.decision, dr.diagnosis, dr.reviewed_at
        FROM doctor_reviews dr
        JOIN scans s ON s.id = dr.scan_id
        JOIN users u ON u.id = s.patient_id
        LEFT JOIN ai_results r ON r.scan_id = s.id
        WHERE dr.doctor_id = %s
        ORDER BY dr.reviewed_at DESC
        """,
        (current_user["id"],),
        fetch=True,
    )
    return rows
