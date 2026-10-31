"""
Clinical PDF report for patient scans after physician review.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from fpdf import FPDF


def _safe(text: Any) -> str:
    """FPDF core fonts are Latin-1; normalize common Unicode and drop the rest."""
    if text is None:
        return "-"
    s = str(text)
    for src, dst in (
        ("\u2014", "-"),
        ("\u2013", "-"),
        ("\u2018", "'"),
        ("\u2019", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2022", "-"),
        ("\u00a0", " "),
    ):
        s = s.replace(src, dst)
    return s.encode("latin-1", "replace").decode("latin-1")


def _fmt_dt(value: Any) -> str:
    if value is None:
        return "-"
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d %H:%M")
    return _safe(value)


def _fmt_pct(confidence: Any) -> str:
    if confidence is None:
        return "-"
    try:
        return f"{float(confidence) * 100:.1f}%"
    except (TypeError, ValueError):
        return _safe(confidence)


def _decision_label(decision: str | None) -> str:
    return {
        "approved": "Approved",
        "rejected": "Rejected",
        "further_review": "Further review requested",
    }.get(decision or "", _safe(decision) if decision else "-")


class ClinicalReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(11, 63, 82)
        self.cell(0, 9, "BrainScan AI - Clinical Imaging Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 116, 139)
        self.cell(0, 5, "Patient summary after verified physician review", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(14, 116, 144)
        self.set_line_width(0.4)
        self.line(12, 24, 198, 24)
        self.ln(8)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(
            0,
            8,
            f"Academic / decision-support use only | Page {self.page_no()}/{{nb}}",
            align="C",
        )


def _section(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")


def _kv_row(pdf: FPDF, label: str, value: str, label_w: float = 55):
    x = pdf.l_margin
    y = pdf.get_y()
    value_w = pdf.epw - label_w
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(label_w, 6, _safe(label))
    pdf.set_xy(x + label_w, y)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.multi_cell(value_w, 6, _safe(value))


def _add_image_block(pdf: FPDF, title: str, path: str | None, max_w: float = 85):
    if not path or not os.path.exists(path):
        return False
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    try:
        # Keep images reasonably sized; leave room for caption
        y_before = pdf.get_y()
        if y_before > 230:
            pdf.add_page()
        pdf.image(path, w=max_w)
        pdf.ln(4)
        return True
    except Exception:
        pdf.set_font("Helvetica", "I", 8)
        pdf.cell(0, 5, "(Image could not be embedded)", new_x="LMARGIN", new_y="NEXT")
        return False


def generate_clinical_report(
    *,
    patient_name: str,
    patient_email: str,
    scan: dict[str, Any],
    image_paths: dict[str, str | None],
) -> bytes:
    """
    Build a PDF report for a physician-reviewed scan.

    image_paths keys: original, overlay, gradcam (filesystem paths or None)
    """
    pdf = ClinicalReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Disclaimer
    pdf.set_fill_color(254, 242, 242)
    pdf.set_draw_color(220, 38, 38)
    pdf.set_text_color(153, 27, 27)
    pdf.set_font("Helvetica", "B", 8)
    pdf.multi_cell(
        0,
        5,
        "IMPORTANT: This report combines AI decision-support outputs with a recorded "
        "physician review. It is not a standalone medical diagnosis and must not replace "
        "in-person clinical judgment, imaging protocols, or treatment advice.",
        border=1,
        fill=True,
    )
    pdf.ln(6)

    # Patient & scan identity
    _section(pdf, "1. Patient & scan information")
    _kv_row(pdf, "Report ID", f"BS-{scan.get('id')}")
    _kv_row(pdf, "Patient name", _safe(patient_name))
    _kv_row(pdf, "Patient email", _safe(patient_email))
    _kv_row(pdf, "Original file", _safe(scan.get("filename")))
    _kv_row(pdf, "Uploaded at", _fmt_dt(scan.get("upload_time")))
    _kv_row(pdf, "Report generated", datetime.now().strftime("%Y-%m-%d %H:%M"))
    pdf.ln(3)

    # AI findings
    _section(pdf, "2. AI analysis summary")
    detected = scan.get("tumor_detected")
    if detected is True:
        detection = "Tumor region suggested by segmentation model"
    elif detected is False:
        detection = "No tumor region suggested"
    else:
        detection = "Unavailable"

    _kv_row(pdf, "Detection", detection)
    _kv_row(pdf, "Predicted type", _safe(scan.get("tumor_type") or "N/A"))
    _kv_row(pdf, "Model confidence", _fmt_pct(scan.get("confidence")))
    _kv_row(pdf, "Model notes", "Classes: Glioma, Meningioma, Pituitary, No Tumor")
    pdf.ln(2)

    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(
        0,
        4,
        "AI outputs are probabilistic. Low confidence or atypical anatomy may require "
        "additional imaging or specialist review.",
    )
    pdf.ln(4)

    # Physician review — the authoritative section
    _section(pdf, "3. Physician review (authoritative)")
    pdf.set_fill_color(240, 253, 250)
    pdf.set_draw_color(13, 148, 136)
    pdf.set_xy(pdf.l_margin, pdf.get_y())
    x0, y0 = pdf.get_x(), pdf.get_y()
    pdf.ln(2)
    _kv_row(pdf, "Decision", _decision_label(scan.get("decision") or scan.get("status")))
    _kv_row(pdf, "Clinical diagnosis", _safe(scan.get("diagnosis")))
    notes = (scan.get("notes") or "").strip()
    _kv_row(pdf, "Physician notes", _safe(notes) if notes else "None recorded")
    _kv_row(pdf, "Reviewed by", f"Dr. {_safe(scan.get('doctor_name') or 'Unknown')}")
    _kv_row(pdf, "Reviewed at", _fmt_dt(scan.get("reviewed_at")))
    pdf.ln(2)
    # Draw a light box around review block (approximate height)
    y1 = pdf.get_y()
    pdf.set_draw_color(45, 212, 191)
    pdf.rect(x0 - 1, y0 - 1, pdf.epw + 2, max(y1 - y0 + 2, 10))
    pdf.ln(6)

    # Imaging
    _section(pdf, "4. Imaging attachments")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 4, "Embedded images from this case (when available).")
    pdf.ln(2)

    # Two-up layout where possible
    orig = image_paths.get("original")
    overlay = image_paths.get("overlay")
    gradcam = image_paths.get("gradcam")

    placed = 0
    start_y = pdf.get_y()
    left_x = pdf.l_margin
    col_w = 88

    def place(path, title, x, y):
        nonlocal placed
        if not path or not os.path.exists(path):
            return y
        try:
            pdf.set_xy(x, y)
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(51, 65, 85)
            pdf.cell(col_w, 5, title, new_x="LMARGIN", new_y="NEXT")
            pdf.set_x(x)
            pdf.image(path, x=x, w=col_w)
            placed += 1
            return pdf.get_y() + 4
        except Exception:
            return y

    if start_y > 200:
        pdf.add_page()
        start_y = pdf.get_y()

    y_left = place(orig, "Original MRI", left_x, start_y)
    y_right = place(overlay, "Segmentation overlay", left_x + col_w + 6, start_y)
    pdf.set_y(max(y_left, y_right, start_y))

    if gradcam and os.path.exists(gradcam):
        if pdf.get_y() > 210:
            pdf.add_page()
        _add_image_block(pdf, "Grad-CAM attention map", gradcam, max_w=100)

    if placed == 0 and not (gradcam and os.path.exists(gradcam or "")):
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "No imaging attachments were available for embedding.", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    # Closing
    _section(pdf, "5. Interpretation guidance")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(
        0,
        5,
        "- The physician decision in section 3 takes precedence over AI suggestions.\n"
        "- Share this report with your care team if requested; keep a personal copy for records.\n"
        "- For urgent neurological symptoms, seek emergency care immediately - do not wait on this report.\n"
        "- Generated by BrainScan AI for academic / demonstration workflows.",
    )

    return bytes(pdf.output())
