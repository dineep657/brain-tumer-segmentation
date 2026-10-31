"""
Shared pytest fixtures for the brain-tumor-enhancements test suite.
"""

import os
import tempfile

import numpy as np
import pytest
import torch


@pytest.fixture
def tmp_db_path(monkeypatch, tmp_path):
    """
    Monkeypatch patient_records.DB_PATH to a temporary file path so tests
    never touch the real data/patient_records.db file.
    """
    db_file = str(tmp_path / "test_patient_records.db")
    import patient_records
    monkeypatch.setattr(patient_records, "DB_PATH", db_file)
    return db_file


@pytest.fixture
def dummy_tensor():
    """
    Returns a random (1, 1, 128, 128) float32 torch tensor with values
    normalised to [0, 1].
    """
    t = torch.rand(1, 1, 128, 128, dtype=torch.float32)
    return t


@pytest.fixture
def dummy_metrics():
    """
    Returns a minimal metrics dict matching the fields expected by
    insert_patient_record / query_patient_history.
    """
    return {
        "tumor_detected": True,
        "tumor_area_px": 500,
        "brain_coverage": 12.5,
    }


@pytest.fixture
def dummy_cls_info():
    """
    Returns a minimal classification-info dict matching the fields expected by
    insert_patient_record.
    """
    return {
        "tumor_type": "Glioma",
        "confidence": 0.85,
    }
