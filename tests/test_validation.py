from __future__ import annotations

from typing import List

from src.schemas import ModelResponse, ProviderResult
from src.validation import (
    build_assessment_audit_export,
    classify_simple_compound_requirement,
    run_validated_analysis,
    validate_candidate_response,
)


class SequenceProvider:
    def __init__(self, results: List[ProviderResult]):
        self.results = list(results)
        self.calls = 0

    def analyse(self, job_description: str, candidate_profile: str) -> ProviderResult:
        self.calls += 1
        if not self.results:
            raise AssertionError("No provider results left for test")
        return self.results.pop(0)


def _provider_result(provider_name: str, parsed: ModelResponse | None = None, *, success: bool = True, validation_status: bool = True, error: str | None = None) -> ProviderResult:
    return ProviderResult(
        provider_name=provider_name,
        model_name=f"{provider_name}-model",
        parsed_response=parsed,
        raw_response="{}",
        elapsed_seconds=0.1,
        success=success,
        validation_status=validation_status,
        error=error,
    )


def _response(status: str = "matched", evidence: str = "Built Python dashboards for customer reporting.") -> ModelResponse:
    return ModelResponse.parse_obj(
        {
            "mandatory_requirements": [
                {
                    "id": "M1",
                    "requirement": "3+ years of Python experience",
                    "source_evidence": "Job ad",
                }
            ],
            "preferred_requirements": [
                {
                    "id": "P1",
                    "requirement": "Experience with Docker",
                    "source_evidence": "Job ad",
                }
            ],
            "requirement_matches": [
                {
                    "requirement_id": "M1",
                    "status": status,
                    "candidate_evidence": evidence,
                    "reason": "supported",
                },
                {
                    "requirement_id": "P1",
                    "status": "matched",
                    "candidate_evidence": "Worked with Docker in delivery pipelines.",
                    "reason": "supported",
                },
            ],
            "missing_evidence": [],
        }
    )


def test_valid_phi4_result_does_not_call_gpt5():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider([_provider_result("phi4", _response(evidence="Candidate has 4 years of Python experience."))])
    gpt5 = SequenceProvider([_provider_result("openai", _response())])

    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    assert result.final_result.success is True
    assert result.gpt5_used is False
    assert phi4.calls == 1
    assert gpt5.calls == 0
    assert result.primary_validation is not None
    assert result.review_validation is None


def test_invalid_phi4_is_retried_once_before_success():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider(
        [
            _provider_result("phi4", None, success=False, validation_status=False, error="Response did not contain valid JSON"),
            _provider_result("phi4", _response(evidence="Candidate has 4 years of Python experience.")),
        ]
    )
    gpt5 = SequenceProvider([_provider_result("openai", _response())])

    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    assert result.final_result.success is True
    assert result.gpt5_used is False
    assert result.phi4_attempts == 2
    assert phi4.calls == 2
    assert gpt5.calls == 0


def test_invalid_phi4_twice_falls_back_to_gpt5():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider(
        [
            _provider_result("phi4", None, success=False, validation_status=False, error="bad json"),
            _provider_result("phi4", None, success=False, validation_status=False, error="bad json again"),
        ]
    )
    gpt5 = SequenceProvider([_provider_result("openai", _response(evidence="Candidate has 4 years of Python experience."))])

    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    assert result.final_result.success is True
    assert result.gpt5_used is True
    assert phi4.calls == 2
    assert gpt5.calls == 1
    assert result.provider_sequence == ["phi4", "phi4", "openai"]
    assert result.primary_validation is None
    assert result.review_validation is not None


def test_mandatory_uncertainty_triggers_gpt5_review():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider([_provider_result("phi4", _response(status="uncertain", evidence="Candidate has 4 years of Python experience."))])
    gpt5 = SequenceProvider([_provider_result("openai", _response(evidence="Candidate has 4 years of Python experience."))])

    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    assert result.gpt5_used is True
    assert any("Mandatory requirement M1 is uncertain" in reason for reason in result.review_reasons)
    assert result.final_result.success is True


def test_mandatory_uncertainty_after_gpt5_review_is_not_a_hard_failure():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider([_provider_result("phi4", _response(status="uncertain", evidence="Candidate has 4 years of Python experience."))])
    gpt5 = SequenceProvider([_provider_result("openai", _response(status="uncertain", evidence="Candidate has 4 years of Python experience."))])

    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    assert result.gpt5_used is True
    assert result.final_result.success is True
    assert result.validation_outcome is not None
    assert result.validation_outcome.review_required is False
    assert any(issue.code == "mandatory_uncertain" for issue in result.validation_outcome.issues)
    assert result.review_validation is not None


def test_unsupported_evidence_triggers_gpt5_review():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider([_provider_result("phi4", _response(evidence="Led neurosurgery operations for five years."))])
    gpt5 = SequenceProvider([_provider_result("openai", _response(evidence="Candidate has 4 years of Python experience."))])

    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    assert result.gpt5_used is True
    assert result.final_result.success is True
    assert any(issue.code == "unsupported_evidence" for issue in result.validation_outcome.issues) is False


def test_explicit_evidence_can_be_corrected_without_gpt5():
    parsed = _response(evidence="Built Python dashboards for customer reporting.")
    outcome = validate_candidate_response(parsed, "Built Python dashboards for customer reporting. Worked with Docker in delivery pipelines.")

    assert outcome.corrected is True
    assert outcome.review_required is False
    corrected_match = outcome.parsed_response.requirement_matches[0]
    assert corrected_match.status.value == "not_matched"


def test_compound_requirement_classifier_cases():
    matched = classify_simple_compound_requirement(
        "Experience with medication administration and IV therapy",
        "Medication administration and IV therapy in acute care.",
        "Medication administration and IV therapy in acute care.",
    )
    assert matched is not None
    assert matched["status"] == "matched"

    partial = classify_simple_compound_requirement(
        "Experience with medication administration and IV therapy",
        "Medication administration in acute care.",
        "Medication administration in acute care.",
    )
    assert partial is not None
    assert partial["status"] == "partially_matched"

    uncertain = classify_simple_compound_requirement(
        "Experience with medication administration and IV therapy",
        "Clinical ward experience caring for acute patients.",
        "Clinical ward experience caring for acute patients and medication safety awareness.",
    )
    assert uncertain is not None
    assert uncertain["status"] == "uncertain"

    not_matched = classify_simple_compound_requirement(
        "Experience with medication administration and IV therapy",
        "Retail leadership and rostering.",
        "Retail leadership and rostering.",
    )
    assert not_matched is not None
    assert not_matched["status"] == "not_matched"


def test_compound_requirement_correction_can_raise_uncertain_to_partial_without_changing_formula():
    parsed = ModelResponse.parse_obj(
        {
            "mandatory_requirements": [
                {
                    "id": "M1",
                    "requirement": "Experience with medication administration and IV therapy",
                    "source_evidence": "Job ad",
                }
            ],
            "preferred_requirements": [],
            "requirement_matches": [
                {
                    "requirement_id": "M1",
                    "status": "uncertain",
                    "candidate_evidence": "Medication administration in acute care.",
                    "reason": "support unclear",
                }
            ],
            "missing_evidence": [],
        }
    )

    outcome = validate_candidate_response(parsed, "Medication administration in acute care.")

    assert outcome.corrected is True
    assert outcome.parsed_response.requirement_matches[0].status.value == "partially_matched"
    assert any(issue.code == "compound_requirement_corrected" for issue in outcome.issues)


def test_compound_requirement_does_not_downgrade_supported_match_to_uncertain():
    parsed = ModelResponse.parse_obj(
        {
            "mandatory_requirements": [
                {
                    "id": "M4",
                    "requirement": "Ability to manage multiple acute patients and escalate appropriately",
                    "source_evidence": "Job ad",
                }
            ],
            "preferred_requirements": [],
            "requirement_matches": [
                {
                    "requirement_id": "M4",
                    "status": "matched",
                    "candidate_evidence": "Provide nursing care for multiple acute patients across medical and surgical wards. Monitor patients for clinical deterioration and escalate concerns appropriately to senior nurses and medical teams.",
                    "reason": "Evidence covers both managing multiple acute patients and appropriate escalation.",
                }
            ],
            "missing_evidence": [],
        }
    )

    outcome = validate_candidate_response(
        parsed,
        "Provide nursing care for multiple acute patients across medical and surgical wards. Monitor patients for clinical deterioration and escalate concerns appropriately to senior nurses and medical teams.",
    )

    assert outcome.corrected is False
    assert outcome.review_required is False
    assert outcome.parsed_response.requirement_matches[0].status.value == "matched"
    assert not any(issue.code == "compound_requirement_corrected" for issue in outcome.issues)


def test_audit_export_preserves_primary_only_flow():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider([_provider_result("phi4", _response(evidence="Candidate has 4 years of Python experience."))])
    gpt5 = SequenceProvider([_provider_result("openai", _response())])
    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    export_obj = build_assessment_audit_export(
        result,
        {"case_id": "CASE1", "job_title": "Example Job"},
        {"mandatory_component": 80.0, "preferred_component": 20.0, "total_percentage": 100.0},
    )

    assert export_obj["audit"]["primary_provider"] == "phi4"
    assert export_obj["audit"]["review_triggered"] is False
    assert export_obj["audit"]["review_result"] is None
    assert export_obj["audit"]["final_provider"] == "phi4"
    assert export_obj["audit"]["primary_result"]["parsed_response"] is not None


def test_audit_export_preserves_review_flow_and_final_result():
    cv_text = "Candidate has 4 years of Python experience. Worked with Docker in delivery pipelines."
    phi4 = SequenceProvider([_provider_result("phi4", _response(status="uncertain", evidence="Candidate has 4 years of Python experience."))])
    gpt5 = SequenceProvider([_provider_result("openai", _response(status="uncertain", evidence="Candidate has 4 years of Python experience."))])
    result = run_validated_analysis("Job description", cv_text, phi4_provider=phi4, gpt5_provider=gpt5)

    export_obj = build_assessment_audit_export(
        result,
        {"case_id": "CASE2", "job_title": "Example Job"},
        {"mandatory_component": 50.0, "preferred_component": 20.0, "total_percentage": 70.0},
    )

    assert export_obj["audit"]["review_triggered"] is True
    assert export_obj["audit"]["review_reasons"]
    assert export_obj["audit"]["primary_result"]["parsed_response"] is not None
    assert export_obj["audit"]["review_result"]["parsed_response"] is not None
    assert export_obj["audit"]["review_validation"] is not None
    assert export_obj["final_assessment"]["provider_name"] == "openai"
    assert export_obj["candidate_fit_score"]["total_percentage"] == 70.0