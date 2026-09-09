"""
doctor_routes.py — Doctor review queue and diagnosis submission.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from webapp.auth import require_role
from webapp.database import execute_query

router = APIRouter()


class ReviewRequest(BaseModel):
    diagnosis: str
    notes: str = ""


@router.get("/queue")
def get_review_queue(current_user: dict = Depends(require_role("doctor"))):
    """All pending scans awaiting doctor review."""
    rows = execute_query(
        """SELECT s.id AS scan_id, s.filename, s.upload_time,
                  u.name AS patient_name, u.email AS patient_email,
                  r.tumor_detected, r.tumor_type, r.confidence,
                  r.gradcam_path, r.overlay_path
           FROM scans s
           JOIN users u ON u.id = s.patient_id
           LEFT JOIN ai_results r ON r.scan_id = s.id
           WHERE s.status = 'pending'
           ORDER BY s.upload_time ASC""",
        fetch=True,
    )
    return rows


@router.get("/case/{scan_id}")
def get_case(scan_id: int, current_user: dict = Depends(require_role("doctor"))):
    """Full case detail for review."""
    rows = execute_query(
        """SELECT s.id AS scan_id, s.filename, s.image_path, s.upload_time, s.status,
                  u.name AS patient_name, u.email AS patient_email,
                  r.tumor_detected, r.tumor_type, r.confidence,
                  r.gradcam_path, r.overlay_path,
                  dr.diagnosis, dr.notes, dr.reviewed_at,
                  d.name AS doctor_name
           FROM scans s
           JOIN users u ON u.id = s.patient_id
           LEFT JOIN ai_results r ON r.scan_id = s.id
           LEFT JOIN doctor_reviews dr ON dr.scan_id = s.id
           LEFT JOIN users d ON d.id = dr.doctor_id
           WHERE s.id = %s""",
        (scan_id,),
        fetch=True,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Case not found")
    return rows[0]


@router.post("/review/{scan_id}")
def submit_review(
    scan_id: int,
    req: ReviewRequest,
    current_user: dict = Depends(require_role("doctor")),
):
    """Submit doctor's diagnosis for a scan."""
    # Check scan exists
    scans = execute_query(
        "SELECT id, status, patient_id FROM scans WHERE id = %s", (scan_id,), fetch=True
    )
    if not scans:
        raise HTTPException(status_code=404, detail="Scan not found")

    # Prevent doctor from reviewing their own scan
    # (in case someone has both patient and doctor accounts with same user_id)
    if scans[0]["patient_id"] == current_user["id"]:
        raise HTTPException(
            status_code=403,
            detail="Doctors cannot review their own scans."
        )

    # Upsert review
    existing = execute_query(
        "SELECT id FROM doctor_reviews WHERE scan_id = %s", (scan_id,), fetch=True
    )
    if existing:
        execute_query(
            """UPDATE doctor_reviews
               SET doctor_id=%s, diagnosis=%s, notes=%s, reviewed_at=NOW()
               WHERE scan_id=%s""",
            (current_user["id"], req.diagnosis, req.notes, scan_id),
        )
    else:
        execute_query(
            """INSERT INTO doctor_reviews (scan_id, doctor_id, diagnosis, notes)
               VALUES (%s, %s, %s, %s)""",
            (scan_id, current_user["id"], req.diagnosis, req.notes),
        )

    # Mark scan as reviewed
    execute_query(
        "UPDATE scans SET status='reviewed' WHERE id=%s", (scan_id,)
    )

    return {"message": "Review submitted successfully."}


@router.get("/history")
def get_reviewed_cases(current_user: dict = Depends(require_role("doctor"))):
    """All cases reviewed by this doctor."""
    rows = execute_query(
        """SELECT s.id AS scan_id, s.filename, s.upload_time,
                  u.name AS patient_name,
                  r.tumor_type, r.confidence,
                  dr.diagnosis, dr.reviewed_at
           FROM doctor_reviews dr
           JOIN scans s ON s.id = dr.scan_id
           JOIN users u ON u.id = s.patient_id
           LEFT JOIN ai_results r ON r.scan_id = s.id
           WHERE dr.doctor_id = %s
           ORDER BY dr.reviewed_at DESC""",
        (current_user["id"],),
        fetch=True,
    )
    return rows
