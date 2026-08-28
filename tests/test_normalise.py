import pytest
from src.normalise import normalise_payload


def test_unwrap_single_item_list():
    payload = [{"mandatory_requirements": [], "preferred_requirements": [], "requirement_matches": []}]
    norm, warnings = normalise_payload(payload)
    assert norm is not None
    assert "unwrapped single-item list" in warnings


def test_assessment_met_normalised_to_matched():
    payload = {"requirement_matches": [{"requirement_id": "M1", "assessment": "met"}]}
    norm, warnings = normalise_payload(payload)
    assert norm is not None
    assert norm["requirement_matches"][0]["status"] == "matched"


def test_evidence_list_normalised_joined():
    payload = {"requirement_matches": [{"requirement_id": "M1", "evidence": ["a","b"]}]}
    norm, warnings = normalise_payload(payload)
    assert norm is not None
    assert "a" in norm["requirement_matches"][0]["candidate_evidence"]

def test_top_level_multiple_list_errors():
    payload = [{"a":1}, {"b":2}]
    norm, warnings = normalise_payload(payload)
    assert norm is None
    assert any("not supported" in w for w in warnings)
