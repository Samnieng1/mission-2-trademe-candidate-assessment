import json

import pytest
from pydantic import ValidationError

from src.schemas import (
    ModelResponse,
    RequirementItem,
    RequirementMatch,
    MatchStatus,
)


def make_valid_response():
    return {
        "mandatory_requirements": [
            {"id": "M1", "requirement": "Req1", "source_evidence": "evidence"}
        ],
        "preferred_requirements": [
            {"id": "P1", "requirement": "Pref1", "source_evidence": "evidence"}
        ],
        "requirement_matches": [
            {"requirement_id": "M1", "status": "matched", "candidate_evidence": "candidate text", "reason": "ok"}
        ],
        "missing_evidence": [],
    }


def test_valid_model_response_parses():
    data = make_valid_response()
    m = ModelResponse.parse_obj(data)
    assert m.mandatory_requirements[0].id == "M1"
    assert m.requirement_matches[0].status == MatchStatus.matched


def test_invalid_match_status_raises():
    data = make_valid_response()
    data["requirement_matches"][0]["status"] = "invalid_status"
    with pytest.raises(ValidationError):
        ModelResponse.parse_obj(data)


def test_missing_candidate_evidence_for_matched_raises():
    data = make_valid_response()
    data["requirement_matches"][0].pop("candidate_evidence")
    with pytest.raises(ValidationError):
        ModelResponse.parse_obj(data)
