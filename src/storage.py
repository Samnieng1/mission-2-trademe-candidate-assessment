"""Storage helpers for experiment results.

Functions to save individual run JSON files and maintain a CSV summary.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from .config import settings


RESULTS_DIR: Path = settings.RESULTS_DIRECTORY
SUMMARY_CSV = RESULTS_DIR / "summary.csv"


def _timestamp() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def save_run(record: Dict[str, Any], save_raw_input: bool = False) -> Path:
    """Save a single run record as JSON and append a row to the summary CSV.

    If `save_raw_input` is False the keys 'job_description', 'candidate_profile', and
    'cover_letter' will be removed from the saved JSON to reduce sensitive text storage.
    Returns the path to the saved JSON file.
    """
    results_dir = settings.RESULTS_DIRECTORY
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = record.get("run_id") or str(uuid4())
    ts = record.get("timestamp") or _timestamp()
    filename = f"run_{ts.replace(':', '-')}_{run_id}.json"
    out_path = results_dir / filename

    to_store = dict(record)
    if not save_raw_input:
        for k in ("job_description", "candidate_profile", "cover_letter"):
            to_store.pop(k, None)

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(to_store, fh, ensure_ascii=False, indent=2)

    # Append summary row
    summary_row = {
        "run_id": run_id,
        "timestamp": ts,
        "case_id": record.get("case_id", ""),
        "provider": record.get("provider", ""),
        "model": record.get("model", ""),
        "success": record.get("success", False),
        "schema_valid": record.get("validation_status", False),
        "elapsed_seconds": record.get("elapsed_seconds", ""),
    }

    summary_csv = results_dir / "summary.csv"
    write_header = not summary_csv.exists()
    with summary_csv.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(summary_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)

    return out_path

