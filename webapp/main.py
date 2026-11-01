"""
main.py — FastAPI entry point for the Brain Tumor Web App.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from webapp.routes import auth_routes, patient_routes, doctor_routes, admin_routes

app = FastAPI(title="BrainScan AI — Medical Imaging Platform", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(BASE_DIR, "templates")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADS_DIR, "doctor_docs"), exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

app.include_router(auth_routes.router, prefix="/api/auth", tags=["Auth"])
app.include_router(patient_routes.router, prefix="/api/patient", tags=["Patient"])
app.include_router(doctor_routes.router, prefix="/api/doctor", tags=["Doctor"])
app.include_router(admin_routes.router, prefix="/api/admin", tags=["Admin"])


def _page(*parts: str):
    return FileResponse(os.path.join(TEMPLATES, *parts))


@app.get("/")
def landing():
    return _page("landing.html")


@app.get("/login")
def login_page():
    return _page("index.html")


@app.get("/apply-doctor")
def apply_doctor_page():
    return _page("apply_doctor.html")


@app.get("/patient/dashboard")
def patient_dashboard():
    return _page("patient", "dashboard.html")


@app.get("/patient/upload")
def patient_upload():
    return _page("patient", "upload.html")


@app.get("/doctor/dashboard")
def doctor_dashboard():
    return _page("doctor", "dashboard.html")


@app.get("/doctor/review/{scan_id}")
def doctor_review(scan_id: int):
    return _page("doctor", "review.html")


@app.get("/doctor/pending")
def doctor_pending():
    return _page("doctor", "pending.html")


@app.get("/admin/create-doctor")
def admin_create_doctor():
    return _page("admin", "create_doctor.html")


@app.get("/admin/dashboard")
def admin_dashboard():
    return _page("admin", "dashboard.html")
