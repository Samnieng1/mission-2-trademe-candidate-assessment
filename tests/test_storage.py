import json
from pathlib import Path

import pytest

from src.storage import save_run
from src.config import settings


def test_save_run_creates_files(tmp_path, monkeypatch):
    # Point results directory to tmp
    monkeypatch.setattr(settings, "RESULTS_DIRECTORY", tmp_path / "results")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "SECRET_KEY")

    record = {
        "case_id": "TEST",
        "provider": "test",
        "model": "m",
        "success": True,
        "validation_status": False,
        "elapsed_seconds": 0.1,
        "job_description": "JD",
        "candidate_profile": "CP",
        "cover_letter": "CL",
    }

    out_path = save_run(record, save_raw_input=False)
    assert out_path.exists()

    # summary.csv created
    summary = (tmp_path / "results") / "summary.csv"
    assert summary.exists()

    # When save_raw_input is False, job_description should be removed
    with out_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        assert "job_description" not in data
        assert "candidate_profile" not in data
        assert "cover_letter" not in data

    # Ensure secrets from settings are not accidentally written into the saved JSON
    text = out_path.read_text(encoding="utf-8")
    assert "SECRET_KEY" not in text


def test_save_run_keeps_raw_when_requested(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RESULTS_DIRECTORY", tmp_path / "results")

    record = {
        "case_id": "TEST2",
        "provider": "test",
        "model": "m",
        "success": True,
        "validation_status": False,
        "elapsed_seconds": 0.2,
        "job_description": "JD2",
        "candidate_profile": "CP2",
        "cover_letter": "CL2",
    }

    out_path = save_run(record, save_raw_input=True)
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
        assert data.get("job_description") == "JD2"
