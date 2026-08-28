import pytest

from src.normalise import normalise_payload
from src.scoring import stats_from_runs


def test_candidate_evidence_list_normalised():
    payload = {
        "requirement_matches": [
            {"requirement_id": "M1", "status": "met", "candidate_evidence": ["e1", "e2"]}
        ]
    }
    norm, warnings = normalise_payload(payload)
    assert norm is not None
    matches = norm.get("requirement_matches")
    assert matches[0]["candidate_evidence"] == "e1; e2"


def test_status_as_singleton_list():
    payload = {"requirement_matches": [{"requirement_id": "M1", "status": ["met"], "candidate_evidence": "x"}]}
    norm, warnings = normalise_payload(payload)
    assert norm["requirement_matches"][0]["status"] == "matched"


def test_stats_from_runs_skips_non_numeric():
    vals = [1.0, "bad", None, 2]
    stats = stats_from_runs(vals)
    assert stats["mean"] == pytest.approx(1.5)


def test_phi4_assessment_evidence_notes_format():
    payload = {
        "requirement_matches": [
            {"id": "M1", "assessment": "met", "evidence": ["a", "b"], "notes": "ok"}
        ]
    }
    norm, warnings = normalise_payload(payload)
    assert norm["requirement_matches"][0]["requirement_id"] == "M1"
    assert norm["requirement_matches"][0]["status"] == "matched"
    assert norm["requirement_matches"][0]["candidate_evidence"] == "a; b"
    assert norm["requirement_matches"][0]["reason"] == "ok"
