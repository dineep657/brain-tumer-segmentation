"""
main.py — FastAPI entry point for the Brain Tumor Web App.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

from webapp.routes import auth_routes, patient_routes, doctor_routes, admin_routes

app = FastAPI(title="Brain Tumor Detection App", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/uploads", StaticFiles(directory=os.path.join(BASE_DIR, "uploads")), name="uploads")

app.include_router(auth_routes.router,    prefix="/api/auth",    tags=["Auth"])
app.include_router(patient_routes.router, prefix="/api/patient", tags=["Patient"])
app.include_router(doctor_routes.router,  prefix="/api/doctor",  tags=["Doctor"])
app.include_router(admin_routes.router,   prefix="/api/admin",   tags=["Admin"])

from fastapi.responses import FileResponse

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

@app.get("/login")
def login_page():
    return FileResponse(os.path.join(BASE_DIR, "templates", "index.html"))

@app.get("/patient/dashboard")
def patient_dashboard():
    return FileResponse(os.path.join(BASE_DIR, "templates", "patient", "dashboard.html"))

@app.get("/patient/upload")
def patient_upload():
    return FileResponse(os.path.join(BASE_DIR, "templates", "patient", "upload.html"))

@app.get("/doctor/dashboard")
def doctor_dashboard():
    return FileResponse(os.path.join(BASE_DIR, "templates", "doctor", "dashboard.html"))

@app.get("/doctor/review/{scan_id}")
def doctor_review(scan_id: int):
    return FileResponse(os.path.join(BASE_DIR, "templates", "doctor", "review.html"))

@app.get("/admin/create-doctor")
def admin_create_doctor():
    return FileResponse(os.path.join(BASE_DIR, "templates", "admin", "create_doctor.html"))

@app.get("/admin/dashboard")
def admin_dashboard():
    return FileResponse(os.path.join(BASE_DIR, "templates", "admin", "dashboard.html"))
