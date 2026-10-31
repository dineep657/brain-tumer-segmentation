-- Brain Tumor Web App Database Schema
CREATE DATABASE IF NOT EXISTS brain_tumor_app CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE brain_tumor_app;

CREATE TABLE IF NOT EXISTS users (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    name          VARCHAR(100)  NOT NULL,
    email         VARCHAR(255)  NOT NULL UNIQUE,
    password_hash VARCHAR(255)  NOT NULL,
    role          ENUM('patient','doctor','admin') NOT NULL DEFAULT 'patient',
    verification_status ENUM('none','pending','verified','rejected','suspended')
                        NOT NULL DEFAULT 'none',
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS scans (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT NOT NULL,
    filename   VARCHAR(255) NOT NULL,
    image_path VARCHAR(255) NOT NULL,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status     ENUM('pending','approved','rejected','further_review') NOT NULL DEFAULT 'pending',
    FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ai_results (
    id             INT AUTO_INCREMENT PRIMARY KEY,
    scan_id        INT NOT NULL UNIQUE,
    tumor_detected BOOLEAN NOT NULL DEFAULT FALSE,
    tumor_type     VARCHAR(50) NOT NULL,
    confidence     FLOAT NULL,
    gradcam_path   VARCHAR(255) NULL,
    overlay_path   VARCHAR(255) NULL,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS doctor_reviews (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    scan_id     INT NOT NULL UNIQUE,
    doctor_id   INT NOT NULL,
    decision    ENUM('approved','rejected','further_review') NOT NULL DEFAULT 'approved',
    diagnosis   TEXT NOT NULL,
    notes       TEXT,
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE,
    FOREIGN KEY (doctor_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS doctor_verifications (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    full_name VARCHAR(150) NOT NULL,
    medical_registration_number VARCHAR(100) NOT NULL,
    qualification VARCHAR(255) NOT NULL,
    hospital VARCHAR(255) NOT NULL,
    contact_phone VARCHAR(40) NULL,
    contact_email VARCHAR(255) NULL,
    document_path VARCHAR(255) NULL,
    application_status ENUM('pending','approved','rejected') NOT NULL DEFAULT 'pending',
    admin_notes TEXT NULL,
    reviewed_by INT NULL,
    reviewed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_doctor_verifications_user (user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    actor_id INT NULL,
    patient_id INT NULL,
    report_id INT NULL,
    action VARCHAR(100) NOT NULL,
    comments TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (actor_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (patient_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (report_id) REFERENCES scans(id) ON DELETE SET NULL
);
