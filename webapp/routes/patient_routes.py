"""
patient_routes.py — Patient scan upload and history endpoints.
"""

import os
import shutil
import uuid

import numpy as np
import torch.nn.functional as F
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import Response
from webapp.auth import require_role
from webapp.database import execute_query, write_audit_log
from webapp.report_generator import generate_clinical_report

# ML imports
from inference_2d import preprocess_2d, predict_2d
from classifier_2d import classify_tumor_2d, load_trained_classifier
from gradcam import generate_gradcam, resize_activation_map, blend_heatmap_with_image
from image_processing import save_segmentation_overlay_fast
from PIL import Image

router = APIRouter()

# Absolute path to uploads dir — always relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UPLOAD_DIR = os.path.join(BASE_DIR, "webapp", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

REVIEWED_STATUSES = frozenset({"approved", "rejected", "further_review"})


def _resolve_upload(relative_name: str | None) -> str | None:
    if not relative_name:
        return None
    path = os.path.join(UPLOAD_DIR, relative_name)
    return path if os.path.isfile(path) else None


def _fetch_owned_scan(scan_id: int, patient_id: int) -> dict:
    rows = execute_query(
        """SELECT s.id, s.filename, s.image_path, s.upload_time, s.status,
                  r.tumor_detected, r.tumor_type, r.confidence,
                  r.gradcam_path, r.overlay_path,
                  dr.decision, dr.diagnosis, dr.notes, dr.reviewed_at,
                  u.name AS doctor_name
           FROM scans s
           LEFT JOIN ai_results r ON r.scan_id = s.id
           LEFT JOIN doctor_reviews dr ON dr.scan_id = s.id
           LEFT JOIN users u ON u.id = dr.doctor_id
           WHERE s.id = %s AND s.patient_id = %s""",
        (scan_id, patient_id),
        fetch=True,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Scan not found")
    return rows[0]


@router.post("/upload")
async def upload_scan(
    file: UploadFile = File(...),
    current_user: dict = Depends(require_role("patient")),
):
    # Save uploaded file
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(status_code=400, detail="Only PNG/JPG images allowed")

    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Save scan record
    scan_id = execute_query(
        "INSERT INTO scans (patient_id, filename, image_path) VALUES (%s, %s, %s)",
        (current_user["id"], file.filename, filename),
    )

    # Run ML inference
    try:
        tensor, raw_original, _ = preprocess_2d(save_path)
        pred_info = predict_2d(tensor, raw_original)

        if pred_info["is_valid_mri"]:
            cls_info = classify_tumor_2d(tensor, tumor_detected=pred_info["tumor_detected"])
        else:
            cls_info = {"tumor_type": "Invalid Input", "confidence": None, "classifier_called": False}

        mask = pred_info["mask"]

        # 1. Fast PIL overlay (Matplotlib is too slow on free Render CPU)
        overlay_filename = None
        try:
            overlay_filename = f"overlay_{uuid.uuid4().hex}.png"
            if pred_info["is_valid_mri"]:
                save_segmentation_overlay_fast(
                    raw_original,
                    mask,
                    os.path.join(UPLOAD_DIR, overlay_filename),
                    alpha=0.5,
                    max_side=512,
                )
            else:
                # Still provide a preview of the uploaded slice
                gray = Image.fromarray(
                    (raw_original / max(float(raw_original.max()), 1.0) * 255).astype("uint8")
                )
                gray = gray.convert("RGB")
                gray.thumbnail((512, 512))
                gray.save(os.path.join(UPLOAD_DIR, overlay_filename), format="PNG", optimize=True)
        except Exception:
            overlay_filename = None

        # 2. Grad-CAM heatmap (reuse cached classifier; downscale source for speed)
        gradcam_filename = None
        if pred_info["is_valid_mri"] and pred_info["tumor_detected"] and cls_info.get("classifier_called"):
            try:
                from train_2d_classifier import CLASSES
                model = load_trained_classifier()
                top_idx = CLASSES.index(cls_info["tumor_type"]) if cls_info["tumor_type"] in CLASSES else 0
                tensor_128 = F.interpolate(tensor, size=(128, 128), mode="bilinear", align_corners=False)
                activation = generate_gradcam(model, tensor_128, top_idx)
                # Blend at most 512px side for free-tier CPU
                h0, w0 = raw_original.shape
                scale = min(1.0, 512.0 / float(max(h0, w0)))
                th, tw = max(1, int(h0 * scale)), max(1, int(w0 * scale))
                if scale < 1.0:
                    small = np.array(
                        Image.fromarray(
                            (raw_original / max(float(raw_original.max()), 1.0) * 255).astype("uint8")
                        ).resize((tw, th), Image.Resampling.BILINEAR)
                    ).astype("float32")
                else:
                    small = raw_original
                activation_resized = resize_activation_map(activation, small.shape[0], small.shape[1])
                gradcam_img = blend_heatmap_with_image(small, activation_resized, alpha=0.5)
                gradcam_filename = f"gradcam_{uuid.uuid4().hex}.png"
                Image.fromarray(gradcam_img).save(
                    os.path.join(UPLOAD_DIR, gradcam_filename), optimize=True
                )
            except Exception:
                gradcam_filename = None

        # 3. Save AI result to DB
        execute_query(
            """INSERT INTO ai_results
               (scan_id, tumor_detected, tumor_type, confidence, gradcam_path, overlay_path)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                scan_id,
                bool(pred_info["tumor_detected"]),
                cls_info.get("tumor_type", "Unknown"),
                cls_info.get("confidence"),
                gradcam_filename,
                overlay_filename,
            ),
        )

        write_audit_log(
            current_user["id"],
            "scan_uploaded",
            f"Scan #{scan_id} uploaded: {file.filename}",
            patient_id=current_user["id"],
            report_id=scan_id,
        )

        return {
            "scan_id": scan_id,
            "status": "pending",
            "tumor_detected": pred_info["tumor_detected"],
            "tumor_type": cls_info.get("tumor_type"),
            "confidence": cls_info.get("confidence"),
            "image_path": filename,
            "overlay_path": overlay_filename,
            "gradcam_path": gradcam_filename,
            "message": "Scan analysed. Awaiting doctor review.",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@router.get("/scans")
def get_my_scans(current_user: dict = Depends(require_role("patient"))):
    rows = execute_query(
        """SELECT s.id, s.filename, s.upload_time, s.status,
                  r.tumor_detected, r.tumor_type, r.confidence,
                  r.gradcam_path, r.overlay_path,
                  dr.decision, dr.diagnosis, dr.notes, dr.reviewed_at,
                  u.name AS doctor_name
           FROM scans s
           LEFT JOIN ai_results r ON r.scan_id = s.id
           LEFT JOIN doctor_reviews dr ON dr.scan_id = s.id
           LEFT JOIN users u ON u.id = dr.doctor_id
           WHERE s.patient_id = %s
           ORDER BY s.upload_time DESC""",
        (current_user["id"],),
        fetch=True,
    )
    return rows


@router.get("/scan/{scan_id}")
def get_scan_detail(scan_id: int, current_user: dict = Depends(require_role("patient"))):
    return _fetch_owned_scan(scan_id, current_user["id"])


@router.get("/scan/{scan_id}/report")
def download_clinical_report(
    scan_id: int,
    current_user: dict = Depends(require_role("patient")),
):
    """
    PDF clinical report — available only after a verified doctor has reviewed the scan.
    """
    scan = _fetch_owned_scan(scan_id, current_user["id"])

    if scan.get("status") not in REVIEWED_STATUSES or not scan.get("decision"):
        raise HTTPException(
            status_code=400,
            detail="Report is available only after a physician has completed review.",
        )

    pdf_bytes = generate_clinical_report(
        patient_name=current_user["name"],
        patient_email=current_user["email"],
        scan=scan,
        image_paths={
            "original": _resolve_upload(scan.get("image_path")),
            "overlay": _resolve_upload(scan.get("overlay_path")),
            "gradcam": _resolve_upload(scan.get("gradcam_path")),
        },
    )

    write_audit_log(
        current_user["id"],
        "clinical_report_downloaded",
        f"Patient downloaded report for scan #{scan_id}",
        patient_id=current_user["id"],
        report_id=scan_id,
    )

    safe_name = f"BrainScan_Report_{scan_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
    )
