# BrainScan AI -- Academic 2D Brain Tumor Segmentation & Classification Platform

## Project Overview

BrainScan AI is an academic B.Tech final-year project that combines deep learning-based brain tumor segmentation, multi-class classification, and explainability (Grad-CAM) into a full-stack web platform. The system accepts 2D brain MRI scans, runs a PyTorch UNet to segment tumor regions, classifies the tumor type using a PyTorch CNN (Glioma, Meningioma, Pituitary, or No Tumor), generates a Grad-CAM heatmap to explain the classification decision, and routes results through a role-based clinical review workflow (Patient -> AI -> Doctor -> Patient).

The project includes two independent frontends sharing the same ML pipeline:
- A **Streamlit** app for quick single-user demonstration
- A **FastAPI** web application with JWT auth, role-based access, MySQL persistence, and a multi-user clinical workflow

---

## Tech Stack

### Machine Learning
| Component | Technology |
|-----------|-----------|
| Segmentation Model | PyTorch LightweightUNet2D (U-Net architecture) |
| Classification Model | PyTorch LightweightClassifier2D (ResNet-style CNN) |
| Explainability | Grad-CAM (custom PyTorch hook-based implementation) |
| Training Framework | PyTorch + custom training loops |
| Model Format | .pth checkpoint files |

### Streamlit Application
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit 1.61 |
| Charts/Plots | Matplotlib |
| PDF Reports | fpdf2 |
| Local Database | SQLite (patient_records.db) |
| Image Processing | PIL (Pillow), NumPy |

### FastAPI Web Application
| Component | Technology |
|-----------|-----------|
| Backend Framework | FastAPI 0.162 |
| ASGI Server | Uvicorn |
| Authentication | JWT (python-jose) + bcrypt password hashing |
| Database | MySQL 8.0 (mysql-connector-python) |
| Frontend | Vanilla HTML/CSS/JS (no framework) |
| File Uploads | python-multipart (multipart/form-data) |

### Testing and Tooling
| Component | Technology |
|-----------|-----------|
| Test Framework | pytest |
| Property-Based Testing | Hypothesis |
| Python Version | 3.13.5 |
| Virtual Environment | venv |

### Infrastructure
| Component | Technology |
|-----------|-----------|
| OS | Windows 11 |
| Database Server | MySQL 8.0 (localhost) |
| Version Control | Git |
| IDE | VS Code with Kiro AI |

---

## File Structure -- Organized by Category

### 1. Streamlit Application (Single-User Demo Frontend)

| File | Title | Description |
|------|-------|-------------|
| app.py | Streamlit Main Entry Point | The primary Streamlit web UI -- handles MRI upload, runs the full AI pipeline, displays metrics/segmentation/Grad-CAM, and provides patient record + PDF export sidebar features |

### 2. ML Pipeline -- Segmentation

| File | Title | Description |
|------|-------|-------------|
| inference_2d.py | 2D Inference Pipeline | Preprocesses MRI images (grayscale, resize 256x256, normalize 0-1), runs UNet segmentation, applies tumor detection threshold, validates MRI input |
| image_processing.py | Segmentation Rendering | Renders matplotlib segmentation figures (overlay, contours), exports PNG overlays |
| train_2d_unet.py | UNet Training Script | Trains the LightweightUNet2D model on the 2D tumor segmentation dataset |
| inference.py | 3D Inference (Legacy) | Older 3D volumetric inference pipeline using MONAI |

### 3. ML Pipeline -- Classification

| File | Title | Description |
|------|-------|-------------|
| classifier_2d.py | CNN Tumor Classifier | Defines LightweightClassifier2D (ResNet-style CNN), loads trained checkpoint, runs 4-class classification (Glioma, Meningioma, Pituitary, No Tumor) |
| train_2d_classifier.py | Classifier Training Script | Trains the LightweightClassifier2D on the classification dataset with data augmentation |

### 4. ML Pipeline -- Explainability

| File | Title | Description |
|------|-------|-------------|
| gradcam.py | Grad-CAM Heatmap Generator | Implements Grad-CAM using PyTorch forward/backward hooks on the classifier res3 block, resizes activation maps, blends heatmaps with grayscale MRI using jet colormap |

### 5. ML Pipeline -- Metrics and Display Logic

| File | Title | Description |
|------|-------|-------------|
| metrics_2d.py | Tumor Metrics Analysis | Computes tumor area (pixels), brain coverage ratio, classification confidence, and derives quantitative clinical metrics from segmentation mask + classifier output |
| confidence_display.py | Confidence Threshold Warning | Centralizes the 60% confidence threshold logic; resolves tumor-type label vs. low-confidence warning badge display |
| metrics.py | 3D Metrics (Legacy) | Older 3D volumetric metrics computation |

### 6. Data and Records

| File | Title | Description |
|------|-------|-------------|
| patient_records.py | SQLite Patient Records | CRUD layer for the Streamlit app -- init_db, validate fields, insert/query/delete patient records in local SQLite |
| pdf_generator.py | PDF Report Generator | Generates downloadable PDF summary reports with scan metrics and overlay images using fpdf2 |

### 7. Dataset Preparation and Sample Generation

| File | Title | Description |
|------|-------|-------------|
| dataset_generator.py | Synthetic 2D Dataset Generator | Generates synthetic 2D brain MRI + tumor mask pairs for initial training |
| build_brisc_dataset.py | BRATS Dataset Builder | Builds dataset from BRATS-format multi-modal MRI volumes |
| classification_dataset_generator.py | Classification Dataset Builder | Generates classification training data from segmentation masks + tumor labels |
| prepare_real_classification_dataset.py | Real Classification Dataset Prep | Prepares real MRI data for the classification training pipeline |
| prepare_real_unet_dataset.py | Real UNet Dataset Prep | Prepares real MRI data for the UNet segmentation training pipeline |
| create_sample_2d.py | Sample 2D MRI Creator | Creates synthetic 2D brain MRI sample images for demo |
| create_sample_data.py | Sample Data Generator | Generates base sample MRI images for the Streamlit demo |
| create_sample_non_mri.py | Non-MRI Test Sample | Creates a plastic bottle image to test the MRI validator rejection of non-brain inputs |
| create_sample_normal_2d.py | Normal Brain MRI Sample | Creates a healthy (no-tumor) brain MRI sample for the normal preset |
| create_sample_subtypes.py | Tumor Subtype Samples | Creates sample MRI images for each tumor subtype (glioma, meningioma, pituitary) |
| update_demo_samples.py | Demo Sample Updater | Updates/replaces the demo sample images in the data/ directory |

### 8. Evaluation and Testing

| File | Title | Description |
|------|-------|-------------|
| evaluate_test_sets.py | Model Evaluator | Evaluates trained UNet and classifier models on held-out test sets, reports accuracy/Dice/metrics |
| test_classifier_pipeline.py | Classifier Pipeline Tests | Integration tests for the classifier inference pipeline |
| test_model_pipeline.py | Model Pipeline Tests | Integration tests for the segmentation inference pipeline |
| debug_alignment.py | Mask Alignment Debugger | Debugging utility to verify segmentation mask alignment with original MRI image |
| tests/conftest.py | Test Fixtures | Shared pytest fixtures -- in-memory SQLite path, dummy tensor factory, dummy metrics dict |
| tests/__init__.py | Tests Package Init | Marks tests/ as a Python package |

### 9. FastAPI Web Application (Multi-User Clinical Platform)

#### Backend

| File | Title | Description |
|------|-------|-------------|
| webapp/main.py | FastAPI Entry Point | App initialization, CORS, static file mounting, route registration, HTML page serving |
| webapp/auth.py | JWT Authentication | Password hashing (bcrypt), JWT token creation/verification, role-based access control (require_role, require_admin) |
| webapp/database.py | MySQL Database Layer | MySQL connection pool, parameterized query executor (fetch/insert/update) |
| webapp/routes/auth_routes.py | Auth API Routes | /api/auth/register, /api/auth/login, /api/auth/me endpoints |
| webapp/routes/patient_routes.py | Patient API Routes | /api/patient/upload (MRI upload + AI inference + overlay/Grad-CAM generation), /api/patient/scans, /api/patient/scan/id |
| webapp/routes/doctor_routes.py | Doctor API Routes | /api/doctor/queue, /api/doctor/case/id, /api/doctor/review/id (diagnosis submission), /api/doctor/history |
| webapp/routes/admin_routes.py | Admin API Routes | /api/admin/stats, /api/admin/users, /api/admin/create-doctor, /api/admin/user/id (delete), /api/admin/scans |

#### Frontend Templates

| File | Title | Description |
|------|-------|-------------|
| webapp/templates/index.html | Login and Register Page | Glassmorphism card UI with tabbed login/register forms, JWT token storage, role-based redirect |
| webapp/templates/patient/dashboard.html | Patient Dashboard | Stats cards (total/pending/reviewed scans), scan history table with AI results + doctor verdict, upload button |
| webapp/templates/patient/upload.html | MRI Upload and Results Page | Drag-drop upload zone, loading spinner, 3-column image panels (original/segmentation/Grad-CAM), metrics cards, AI summary, doctor review display |
| webapp/templates/doctor/dashboard.html | Doctor Dashboard | Review queue (pending scans) + review history tabs, stats cards, patient info + AI results table |
| webapp/templates/doctor/review.html | Doctor Review Page | AI results display + diagnosis submission form (diagnosis + notes) |
| webapp/templates/admin/dashboard.html | Admin Dashboard | Platform stats (5 metrics), user management table, create-doctor form, all-scans overview table |
| webapp/templates/admin/create_doctor.html | Create Doctor Page | Doctor account creation form (to be redesigned per UI spec) |

### 10. Configuration and Database Setup

| File | Title | Description |
|------|-------|-------------|
| requirements.txt | Python Dependencies | Core ML (torch, monai), Streamlit (streamlit, matplotlib, fpdf2), Web App (fastapi, uvicorn, mysql-connector, python-jose, bcrypt, pydantic), Testing (pytest, hypothesis) |
| schema.sql | MySQL Database Schema | Creates brain_tumor_app database with 4 tables: users, scans, ai_results, doctor_reviews |
| .gitignore | Git Ignore Rules | Excludes venv, __pycache__, data files, model checkpoints, logs |

### 11. Trained Model Checkpoints

| File | Title | Description |
|------|-------|-------------|
| models/brain_tumor_unet_2d.pth | Trained UNet Checkpoint | Serialized PyTorch state_dict for the LightweightUNet2D segmentation model |
| models/brain_tumor_classifier_2d.pth | Trained Classifier Checkpoint | Serialized PyTorch state_dict for the LightweightClassifier2D classification model |

### 12. Data Directories

| Path | Description |
|------|-------------|
| data/ | Contains sample MRI images for the Streamlit demo (glioma, meningioma, pituitary, normal, non-MRI bottle), generated overlays, and the SQLite patient records database |
| classification_dataset/ | Prepared classification training dataset (image + label pairs) |
| dataset/ | Prepared segmentation training dataset (image + mask pairs) |
| webapp/uploads/ | Uploaded MRI scans and generated overlay/Grad-CAM images for the web app |

### 13. Project Specifications (Kiro Specs)

| Path | Title | Description |
|------|-------|-------------|
| .kiro/specs/brain-tumor-enhancements/ | Streamlit Enhancements Spec | Requirements, design, and tasks for confidence threshold warning, patient records, and Grad-CAM features (COMPLETED) |
| .kiro/specs/webapp-tier2-enhancements/ | Web App Tier 2 Spec | Requirements for DICOM support, interactive viewer, clinical reports, notifications, and longitudinal comparison (PENDING) |
| .kiro/specs/webapp-ui-redesign/ | UI Redesign Spec | Design and tasks for shared CSS, landing page, responsive layout, and dedicated doctor creation page (PENDING) |

---

## Features

### ML / AI Features
- **2D MRI Segmentation** -- PyTorch UNet identifies tumor pixels and produces a binary mask with contour overlay
- **4-Class Tumor Classification** -- CNN classifies into Glioma, Meningioma, Pituitary, or No Tumor with softmax confidence
- **Grad-CAM Explainability** -- Heatmap overlay showing which image regions drove the classification decision
- **Confidence Threshold Warning** -- Surfaces a LOW CONFIDENCE warning when classifier confidence is below 60%
- **Input Validation** -- Rejects non-MRI images (e.g., plastic bottle test) with a descriptive error message
- **Quantitative Metrics** -- Tumor area (pixels), brain coverage ratio (%), classification confidence (%)

### Streamlit App Features
- MRI upload (PNG/JPG) or preset sample selection (5 presets)
- Real-time inference pipeline with execution time display
- Adjustable overlay opacity, high-contrast mode, contour highlighting
- 3-panel visualization (input, segmentation overlay, Grad-CAM)
- Patient record system (SQLite) -- save, query history, clear all
- PDF report download with metrics and segmentation images
- Model architecture telemetry panel (viva presentation)

### FastAPI Web App Features
- **Role-Based Access Control** -- Three roles: Patient, Doctor, Admin with JWT-authenticated route guards
- **Patient Workflow** -- Register, upload MRI, AI runs automatically, view results, await doctor review, see final diagnosis
- **Doctor Workflow** -- View pending review queue, examine AI results + segmentation/Grad-CAM images, submit diagnosis + notes
- **Admin Workflow** -- Platform statistics dashboard, create doctor accounts, manage all users, view all scans
- **MySQL Persistence** -- Full scan lifecycle stored in relational database (users, scans, AI results, doctor reviews)
- **Image Storage** -- Uploaded scans, segmentation overlays, and Grad-CAM heatmaps saved to disk and served via static file mount

---

## Architecture Overview

The ML pipeline modules (inference_2d.py, classifier_2d.py, gradcam.py, metrics_2d.py, image_processing.py) are shared between both frontends. Neither frontend modifies the trained model weights during inference.

Pipeline flow:
1. preprocess_2d (grayscale, resize 256x256, normalize 0-1)
2. predict_2d (UNet segmentation, tumor detection threshold)
3. classify_tumor_2d (CNN 4-class classification with softmax)
4. generate_gradcam (Grad-CAM heatmap on res3 block)
5. analyze_tumor_metrics (tumor area, brain coverage, confidence)

Two frontends consume this pipeline:
- Streamlit (app.py) -- single-user demo with SQLite records and PDF export
- FastAPI (webapp/) -- multi-user platform with MySQL + JWT auth + role-based clinical workflow

---

## Database Schema (FastAPI Web App -- MySQL)

- **users** -- id, name, email, password_hash, role (patient/doctor/admin), created_at
- **scans** -- id, patient_id (FK), filename, image_path, upload_time, status (pending/reviewed)
- **ai_results** -- id, scan_id (FK), tumor_detected, tumor_type, confidence, gradcam_path, overlay_path
- **doctor_reviews** -- id, scan_id (FK), doctor_id (FK), diagnosis, notes, reviewed_at

---

## Running the Applications

| App | URL |
|-----|-----|
| Streamlit | localhost:8501 |
| FastAPI | localhost:8000 |

**Test Accounts (FastAPI):**
- Admin: admin@brainscan.ai / admin123
- Doctor: doctor@brainscan.ai / doctor123
- Patient: register via the web interface

---

## Planned Enhancements (Specs)

| Spec | Status | Scope |
|------|--------|-------|
| Streamlit Enhancements | COMPLETE | Confidence threshold, patient records, Grad-CAM |
| Web App Tier 2 | REQUIREMENTS READY | DICOM support, interactive viewer, clinical reports, notifications, longitudinal comparison |
| UI Redesign | DESIGN + TASKS READY | Shared CSS, landing page, responsive layout, dedicated doctor creation page |
