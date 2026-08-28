import pytest
from src.candidate_scoring import build_candidate_fit_breakdown, calculate_candidate_fit_score_components
from src.schemas import ModelResponse


def test_calculate_candidate_fit_score_components_cv_only():
    parsed = {
        "mandatory_requirements": [{"id": "M1"}, {"id": "M2"}, {"id": "M3"}, {"id": "M4"}, {"id": "M5"}],
        "preferred_requirements": [{"id": "P1"}, {"id": "P2"}, {"id": "P3"}],
        "requirement_matches": [
            {"requirement_id": "M1", "status": "matched"},
            {"requirement_id": "M2", "status": "matched"},
            {"requirement_id": "M3", "status": "matched"},
            {"requirement_id": "M4", "status": "matched"},
            {"requirement_id": "P1", "status": "matched"},
            {"requirement_id": "P2", "status": "matched"},
        ],
    }

    comps = calculate_candidate_fit_score_components(parsed)
    # mandatory: 4/5 * 80 = 64.0
    assert comps["mandatory_component"] == pytest.approx(64.0)
    # preferred: 2/3 * 20
    assert comps["preferred_component"] == pytest.approx(round((2 / 3) * 20.0, 2))
    # total
    assert comps["total_percentage"] == pytest.approx(comps["mandatory_component"] + comps["preferred_component"], rel=1e-2)


def test_build_candidate_fit_breakdown_preserves_score_and_counts():
    parsed = {
        "mandatory_requirements": [{"id": "M1"}, {"id": "M2"}],
        "preferred_requirements": [{"id": "P1"}],
        "requirement_matches": [
            {"requirement_id": "M1", "status": "matched"},
            {"requirement_id": "M2", "status": "uncertain"},
            {"requirement_id": "P1", "status": "partially_matched"},
        ],
        "missing_evidence": [{"requirement_id": "M2", "reason": "missing"}],
    }

    breakdown = build_candidate_fit_breakdown(parsed)

    assert breakdown["mandatory_component"] == pytest.approx(50.0)
    assert breakdown["preferred_component"] == pytest.approx(10.0)
    assert breakdown["total_percentage"] == pytest.approx(60.0)
    assert breakdown["mandatory_counts"]["matched"] == 1
    assert breakdown["mandatory_counts"]["uncertain"] == 1
    assert breakdown["preferred_counts"]["partially_matched"] == 1
    assert breakdown["missing_evidence_count"] == 1


def test_build_candidate_fit_breakdown_accepts_model_response_with_empty_lists():
    parsed = ModelResponse.parse_obj(
        {
            "mandatory_requirements": [
                {"id": "M1", "requirement": "Req1", "source_evidence": "Job ad"}
            ],
            "preferred_requirements": [],
            "requirement_matches": [
                {"requirement_id": "M1", "status": "matched", "candidate_evidence": "Evidence", "reason": "ok"}
            ],
            "missing_evidence": [],
        }
    )

    breakdown = build_candidate_fit_breakdown(parsed)

    assert breakdown["mandatory_component"] == pytest.approx(80.0)
    assert breakdown["preferred_component"] == pytest.approx(0.0)
    assert breakdown["missing_evidence_count"] == 0
