"""Validation and fallback orchestration for candidate-fit analysis.

This module keeps provider selection and deterministic validation rules out of
the Streamlit UI so the runtime flow stays easy to explain and test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

from .config import settings
from .providers.openai_provider import OpenAIProvider
from .providers.phi4_provider import Phi4Provider
from .scoring import evaluate_fuzzy_evidence_support, normalize_text
from .schemas import MatchStatus, ModelResponse, ProviderResult


WORK_RIGHTS_PATTERN = re.compile(
    r"\b(right to work|work rights?|eligible to work|permanent resid|permanent resident|citizen|citizenship|visa)\b",
    flags=re.IGNORECASE,
)
QUALIFICATION_PATTERN = re.compile(
    r"\b(bachelor|master|degree|diploma|certificate|certification|certified|licen[cs]e|registration|registered)\b",
    flags=re.IGNORECASE,
)
EXPLICIT_REQUIREMENT_PATTERN = re.compile(
    r"\b(right to work|eligib|years?|licen[cs]e|certif|qualif|degree|diploma|registration|registered)\b",
    flags=re.IGNORECASE,
)
YEARS_PATTERN = re.compile(r"(\d+)\s*\+?\s*years?", flags=re.IGNORECASE)


@dataclass
class ValidationIssue:
    code: str
    message: str
    requirement_id: Optional[str] = None
    severity: str = "review_required"


@dataclass
class ValidationOutcome:
    parsed_response: ModelResponse
    review_required: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    evidence_metrics: Optional[Dict[str, Any]] = None
    corrected: bool = False


@dataclass
class ValidatedAnalysisResult:
    final_result: ProviderResult
    primary_result: ProviderResult
    review_result: Optional[ProviderResult]
    validation_outcome: Optional[ValidationOutcome]
    primary_validation: Optional[ValidationOutcome]
    review_validation: Optional[ValidationOutcome]
    provider_sequence: List[str]
    phi4_attempts: int
    gpt5_used: bool
    review_reasons: List[str]


def _status_name(status: Any) -> str:
    return normalize_text(getattr(status, "value", status))


def _build_requirement_map(parsed_response: ModelResponse) -> Dict[str, str]:
    requirement_map: Dict[str, str] = {}
    for requirement in list(parsed_response.mandatory_requirements) + list(parsed_response.preferred_requirements):
        requirement_map[requirement.id] = requirement.requirement
    return requirement_map


def _normalised_tokens(text: str) -> List[str]:
    keep_short = {"iv", "rn", "emr", "apc", "icu", "ed", "nz"}
    return [token for token in normalize_text(text).split() if len(token) >= 3 or token in keep_short]


def _simple_compound_components(requirement_text: str) -> Optional[List[str]]:
    raw = (requirement_text or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    if "," in raw or ";" in raw or " or " in lowered:
        return None
    if lowered.count(" and ") != 1:
        return None
    left, right = re.split(r"\band\b", raw, maxsplit=1, flags=re.IGNORECASE)
    left = left.strip(" .")
    right = right.strip(" .")
    if not left or not right:
        return None
    return [left, right]


def _simplify_component_text(component: str) -> str:
    simplified = (component or "").strip()
    simplified = re.sub(
        r"^(experience\s+(with|in)|ability\s+to|ability\s+for|familiarity\s+with|knowledge\s+of)\s+",
        "",
        simplified,
        flags=re.IGNORECASE,
    )
    return simplified.strip(" .")


def classify_simple_compound_requirement(
    requirement_text: str,
    candidate_evidence: str,
    candidate_profile_text: str,
) -> Optional[Dict[str, Any]]:
    """Classify simple two-part compound requirements conservatively.

    This helper only activates when a requirement cleanly reads as two material
    parts joined by a single 'and' without other list punctuation. It is used to
    correct clearly inconsistent status choices, not to broadly parse all job
    requirement language.
    """
    components = _simple_compound_components(requirement_text)
    if not components:
        return None

    corpus_text = "\n".join(part for part in (candidate_evidence or "", candidate_profile_text or "") if part)
    corpus_normalized = normalize_text(corpus_text)
    corpus_tokens = set(_normalised_tokens(corpus_text))
    support_by_component = []

    for component in components:
        simplified_component = _simplify_component_text(component)
        component_normalized = normalize_text(simplified_component)
        component_tokens = _normalised_tokens(simplified_component)
        if component_normalized and component_normalized in corpus_normalized:
            support = "explicit"
        else:
            overlap = len([token for token in component_tokens if token in corpus_tokens])
            if overlap == 0:
                support = "absent"
            else:
                support = "ambiguous"
        support_by_component.append(
            {
                "component": component,
                "support": support,
            }
        )

    explicit_count = len([row for row in support_by_component if row["support"] == "explicit"])
    ambiguous_count = len([row for row in support_by_component if row["support"] == "ambiguous"])
    absent_count = len([row for row in support_by_component if row["support"] == "absent"])

    if explicit_count == len(support_by_component):
        status = "matched"
    elif explicit_count >= 1 and absent_count >= 1 and ambiguous_count == 0:
        status = "partially_matched"
    elif explicit_count == 0 and absent_count == len(support_by_component):
        status = "not_matched"
    else:
        status = "uncertain"

    return {
        "status": status,
        "components": support_by_component,
    }


def _extract_year_values(text: str) -> List[int]:
    return [int(match.group(1)) for match in YEARS_PATTERN.finditer(text or "")]


def _explicit_requirement_kind(requirement_text: str) -> Optional[str]:
    text = requirement_text or ""
    if not EXPLICIT_REQUIREMENT_PATTERN.search(text):
        return None
    if WORK_RIGHTS_PATTERN.search(text):
        return "work_rights"
    if YEARS_PATTERN.search(text) or "year" in text.lower():
        return "years"
    if QUALIFICATION_PATTERN.search(text):
        return "qualification"
    return "explicit_other"


def _explicit_support(requirement_text: str, candidate_evidence: str, candidate_profile_text: str) -> Tuple[bool, bool]:
    kind = _explicit_requirement_kind(requirement_text)
    if kind is None:
        return True, False

    evidence_text = candidate_evidence or ""
    combined_text = "\n".join(part for part in (evidence_text, candidate_profile_text or "") if part)

    if kind == "work_rights":
        supported = bool(WORK_RIGHTS_PATTERN.search(combined_text))
        return supported, not supported

    if kind == "qualification":
        supported = bool(QUALIFICATION_PATTERN.search(combined_text))
        return supported, not supported

    if kind == "years":
        required_values = _extract_year_values(requirement_text)
        found_values = _extract_year_values(combined_text)
        if not required_values:
            return True, False
        if not found_values:
            return False, True
        supported = max(found_values) >= max(required_values)
        return supported, not supported

    explicit_hint = bool(EXPLICIT_REQUIREMENT_PATTERN.search(combined_text))
    return explicit_hint, not explicit_hint


def _apply_explicit_evidence_corrections(
    parsed_response: ModelResponse,
    candidate_profile_text: str,
) -> Tuple[ModelResponse, List[ValidationIssue], bool]:
    payload = parsed_response.dict()
    requirement_map = _build_requirement_map(parsed_response)
    corrected = False
    issues: List[ValidationIssue] = []

    for match in payload.get("requirement_matches", []):
        requirement_id = match.get("requirement_id")
        requirement_text = requirement_map.get(requirement_id, "")
        raw_status = match.get("status")
        status_name = normalize_text(getattr(raw_status, "value", raw_status))
        if status_name not in {"matched", "partially_matched"}:
            continue

        supported, clear_absence = _explicit_support(
            requirement_text,
            str(match.get("candidate_evidence") or ""),
            candidate_profile_text,
        )
        if supported or not clear_absence:
            continue

        corrected = True
        corrected_status = "not_matched"
        original_reason = str(match.get("reason") or "").strip()
        correction_reason = "Adjusted by validation because explicit evidence was not found in the uploaded CV."
        match["status"] = corrected_status
        match["candidate_evidence"] = None
        match["reason"] = correction_reason if not original_reason else f"{correction_reason} Original model reason: {original_reason}"
        issues.append(
            ValidationIssue(
                code="explicit_evidence_corrected",
                message=f"Requirement {requirement_id} was downgraded because explicit evidence was not found in the uploaded CV.",
                requirement_id=requirement_id,
                severity="corrected",
            )
        )

    if not corrected:
        return parsed_response, issues, False

    return ModelResponse.parse_obj(payload), issues, True


def _should_apply_compound_correction(current_status: str, derived_status: str, has_candidate_evidence: bool) -> bool:
    if derived_status == current_status:
        return False
    if derived_status == "matched":
        return False
    if derived_status == "partially_matched" and not has_candidate_evidence:
        return False
    if current_status == "matched" and derived_status in {"partially_matched", "not_matched"}:
        return True
    if current_status == "uncertain" and derived_status in {"partially_matched", "not_matched"}:
        return True
    if current_status == "partially_matched" and derived_status == "not_matched":
        return True
    return False


def _apply_compound_requirement_corrections(
    parsed_response: ModelResponse,
    candidate_profile_text: str,
) -> Tuple[ModelResponse, List[ValidationIssue], bool]:
    payload = parsed_response.dict()
    requirement_map = _build_requirement_map(parsed_response)
    existing_missing = {item.get("requirement_id") for item in payload.get("missing_evidence", []) if isinstance(item, dict)}
    corrected = False
    issues: List[ValidationIssue] = []

    for match in payload.get("requirement_matches", []):
        requirement_id = match.get("requirement_id")
        requirement_text = requirement_map.get(requirement_id, "")
        analysis = classify_simple_compound_requirement(
            requirement_text,
            str(match.get("candidate_evidence") or ""),
            candidate_profile_text,
        )
        if analysis is None:
            continue

        raw_status = match.get("status")
        current_status = normalize_text(getattr(raw_status, "value", raw_status))
        derived_status = analysis["status"]
        if not _should_apply_compound_correction(current_status, derived_status, bool(match.get("candidate_evidence"))):
            continue

        explicit_components = [row["component"] for row in analysis["components"] if row["support"] == "explicit"]
        absent_components = [row["component"] for row in analysis["components"] if row["support"] == "absent"]
        ambiguous_components = [row["component"] for row in analysis["components"] if row["support"] == "ambiguous"]

        message_parts = []
        if explicit_components:
            message_parts.append("supported: " + ", ".join(explicit_components))
        if absent_components:
            message_parts.append("not evidenced: " + ", ".join(absent_components))
        if ambiguous_components:
            message_parts.append("ambiguous: " + ", ".join(ambiguous_components))
        detail = "; ".join(message_parts) if message_parts else "compound requirement evidence was reassessed"
        correction_reason = f"Adjusted by validation for compound requirement semantics: {detail}."

        corrected = True
        match["status"] = derived_status
        original_reason = str(match.get("reason") or "").strip()
        match["reason"] = correction_reason if not original_reason else f"{correction_reason} Original model reason: {original_reason}"
        if derived_status == "not_matched":
            match["candidate_evidence"] = None
        if absent_components and requirement_id not in existing_missing:
            payload.setdefault("missing_evidence", []).append(
                {
                    "requirement_id": requirement_id,
                    "reason": "Missing required component evidence: " + ", ".join(absent_components),
                }
            )
            existing_missing.add(requirement_id)

        issues.append(
            ValidationIssue(
                code="compound_requirement_corrected",
                message=f"Requirement {requirement_id} status was adjusted to {derived_status} using compound requirement evidence rules.",
                requirement_id=requirement_id,
                severity="corrected",
            )
        )

    if not corrected:
        return parsed_response, issues, False

    return ModelResponse.parse_obj(payload), issues, True


def validate_candidate_response(
    parsed_response: ModelResponse,
    candidate_profile_text: str,
    threshold: Optional[float] = None,
    require_review_for_mandatory_uncertain: bool = True,
) -> ValidationOutcome:
    corrected_response, correction_issues, corrected = _apply_explicit_evidence_corrections(
        parsed_response,
        candidate_profile_text,
    )
    corrected_response, compound_issues, compound_corrected = _apply_compound_requirement_corrections(
        corrected_response,
        candidate_profile_text,
    )
    issues = list(correction_issues) + list(compound_issues)
    effective_threshold = settings.EVIDENCE_FUZZY_THRESHOLD if threshold is None else threshold

    requirement_map = _build_requirement_map(corrected_response)
    mandatory_ids = {item.id for item in corrected_response.mandatory_requirements}
    evidence_required_ids = set()

    for match in corrected_response.requirement_matches:
        status_name = _status_name(match.status)
        if status_name in {"matched", "partially_matched"}:
            evidence_required_ids.add(match.requirement_id)
            if not normalize_text(match.candidate_evidence):
                issues.append(
                    ValidationIssue(
                        code="missing_supporting_evidence",
                        message=f"Requirement {match.requirement_id} is {status_name} but has no supporting evidence.",
                        requirement_id=match.requirement_id,
                    )
                )

        if match.requirement_id in mandatory_ids and status_name == "uncertain":
            issues.append(
                ValidationIssue(
                    code="mandatory_uncertain",
                    message=f"Mandatory requirement {match.requirement_id} is uncertain and requires review.",
                    requirement_id=match.requirement_id,
                    severity="review_required" if require_review_for_mandatory_uncertain else "note",
                )
            )

    evidence_matches = [
        match
        for match in corrected_response.requirement_matches
        if _status_name(match.status) in {"matched", "partially_matched"}
    ]
    evidence_metrics = evaluate_fuzzy_evidence_support(
        evidence_matches,
        candidate_profile_text,
        effective_threshold,
        gold_evidence_by_requirement=None,
    )

    debug_rows = evidence_metrics.get("evidence_fuzzy_debug", []) if evidence_metrics else []
    for row in debug_rows:
        requirement_id = row.get("requirement_id")
        if requirement_id not in evidence_required_ids:
            continue
        if row.get("fuzzy_supported"):
            continue
        issues.append(
            ValidationIssue(
                code="unsupported_evidence",
                message=(
                    f"Requirement {requirement_id} cites evidence that could not be grounded in the uploaded CV"
                    f" for '{requirement_map.get(requirement_id, requirement_id)}'."
                ),
                requirement_id=requirement_id,
            )
        )

    review_required = any(issue.severity == "review_required" for issue in issues)
    return ValidationOutcome(
        parsed_response=corrected_response,
        review_required=review_required,
        issues=issues,
        evidence_metrics=evidence_metrics,
        corrected=(corrected or compound_corrected),
    )


def _serialize_issue(issue: ValidationIssue) -> Dict[str, Any]:
    return {
        "code": issue.code,
        "message": issue.message,
        "requirement_id": issue.requirement_id,
        "severity": issue.severity,
    }


def serialize_validation_outcome(outcome: Optional[ValidationOutcome]) -> Optional[Dict[str, Any]]:
    if outcome is None:
        return None
    return {
        "review_required": outcome.review_required,
        "corrected": outcome.corrected,
        "issues": [_serialize_issue(issue) for issue in outcome.issues],
        "evidence_metrics": outcome.evidence_metrics,
        "parsed_response": outcome.parsed_response.dict() if outcome.parsed_response else None,
    }


def serialize_provider_result(result: Optional[ProviderResult]) -> Optional[Dict[str, Any]]:
    if result is None:
        return None
    return {
        "provider_name": result.provider_name,
        "model_name": result.model_name,
        "success": result.success,
        "validation_status": result.validation_status,
        "elapsed_seconds": result.elapsed_seconds,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "estimated_cost": result.estimated_cost,
        "normalization_applied": result.normalization_applied,
        "normalization_warnings": list(result.normalization_warnings),
        "error": result.error,
        "parsed_response": result.parsed_response.dict() if result.parsed_response else None,
    }


def build_assessment_audit_export(
    analysis_result: ValidatedAnalysisResult,
    case_payload: Dict[str, Any],
    score_breakdown: Dict[str, Any],
) -> Dict[str, Any]:
    final_result = analysis_result.final_result
    return {
        "case_id": case_payload.get("case_id"),
        "job_title": case_payload.get("job_title"),
        "audit": {
            "primary_provider": analysis_result.primary_result.provider_name,
            "primary_result": serialize_provider_result(analysis_result.primary_result),
            "primary_validation": serialize_validation_outcome(analysis_result.primary_validation),
            "review_triggered": analysis_result.gpt5_used,
            "review_reasons": list(analysis_result.review_reasons),
            "phi4_attempts": analysis_result.phi4_attempts,
            "review_provider": analysis_result.review_result.provider_name if analysis_result.review_result else None,
            "review_result": serialize_provider_result(analysis_result.review_result),
            "review_validation": serialize_validation_outcome(analysis_result.review_validation),
            "corrected_issues": [
                _serialize_issue(issue)
                for issue in (analysis_result.validation_outcome.issues if analysis_result.validation_outcome else [])
                if issue.severity == "corrected"
            ],
            "final_provider": final_result.provider_name,
            "provider_sequence": list(analysis_result.provider_sequence),
        },
        "final_assessment": serialize_provider_result(final_result),
        "candidate_fit_score": score_breakdown,
    }


def _finalize_result(
    result: ProviderResult,
    outcome: Optional[ValidationOutcome],
    validation_failed: bool = False,
) -> ProviderResult:
    if outcome is not None:
        result.parsed_response = outcome.parsed_response
    if validation_failed and outcome is not None:
        result.success = False
        result.error = "Validation failed after review: " + "; ".join(issue.message for issue in outcome.issues if issue.severity == "review_required")
    elif outcome is not None:
        result.success = True
    return result


def run_validated_analysis(
    job_description: str,
    candidate_profile_text: str,
    phi4_provider: Optional[Any] = None,
    gpt5_provider: Optional[Any] = None,
) -> ValidatedAnalysisResult:
    phi4 = phi4_provider or Phi4Provider()
    gpt5 = gpt5_provider or OpenAIProvider()

    phi4_attempts = 0
    provider_sequence: List[str] = []
    primary_result: Optional[ProviderResult] = None
    review_result: Optional[ProviderResult] = None
    primary_outcome: Optional[ValidationOutcome] = None

    while phi4_attempts < 2:
        phi4_attempts += 1
        primary_result = phi4.analyse(job_description, candidate_profile_text)
        provider_sequence.append(getattr(primary_result, "provider_name", "phi4"))
        if primary_result.parsed_response and primary_result.validation_status:
            primary_outcome = validate_candidate_response(primary_result.parsed_response, candidate_profile_text)
            if not primary_outcome.review_required:
                final_result = _finalize_result(primary_result, primary_outcome)
                return ValidatedAnalysisResult(
                    final_result=final_result,
                    primary_result=primary_result,
                    review_result=None,
                    validation_outcome=primary_outcome,
                    primary_validation=primary_outcome,
                    review_validation=None,
                    provider_sequence=provider_sequence,
                    phi4_attempts=phi4_attempts,
                    gpt5_used=False,
                    review_reasons=[issue.message for issue in primary_outcome.issues],
                )
            break

    review_reasons: List[str] = []
    if primary_result is not None:
        if primary_outcome is not None:
            review_reasons = [issue.message for issue in primary_outcome.issues if issue.severity == "review_required"]
        elif primary_result.error:
            review_reasons = [primary_result.error]
        else:
            review_reasons = ["Phi-4 did not return a valid structured response."]

    if not gpt5 or not hasattr(gpt5, "analyse"):
        final_result = primary_result or ProviderResult(provider_name="phi4", model_name="")
        final_result.success = False
        final_result.error = "GPT-5 review provider is unavailable. " + "; ".join(review_reasons)
        return ValidatedAnalysisResult(
            final_result=final_result,
            primary_result=primary_result or final_result,
            review_result=None,
            validation_outcome=primary_outcome,
            primary_validation=primary_outcome,
            review_validation=None,
            provider_sequence=provider_sequence,
            phi4_attempts=phi4_attempts,
            gpt5_used=False,
            review_reasons=review_reasons,
        )

    review_result = gpt5.analyse(job_description, candidate_profile_text)
    provider_sequence.append(getattr(review_result, "provider_name", "openai"))

    if review_result.parsed_response and review_result.validation_status:
        review_outcome = validate_candidate_response(
            review_result.parsed_response,
            candidate_profile_text,
            require_review_for_mandatory_uncertain=False,
        )
        validation_failed = review_outcome.review_required
        final_result = _finalize_result(review_result, review_outcome, validation_failed=validation_failed)
        return ValidatedAnalysisResult(
            final_result=final_result,
            primary_result=primary_result or final_result,
            review_result=review_result,
            validation_outcome=review_outcome,
            primary_validation=primary_outcome,
            review_validation=review_outcome,
            provider_sequence=provider_sequence,
            phi4_attempts=phi4_attempts,
            gpt5_used=True,
            review_reasons=review_reasons,
        )

    if review_result.error:
        review_reasons.append(review_result.error)
    final_result = review_result
    final_result.success = False
    if not final_result.error:
        final_result.error = "GPT-5 review did not return a valid structured response."
    return ValidatedAnalysisResult(
        final_result=final_result,
        primary_result=primary_result or final_result,
        review_result=review_result,
        validation_outcome=None,
        primary_validation=primary_outcome,
        review_validation=None,
        provider_sequence=provider_sequence,
        phi4_attempts=phi4_attempts,
        gpt5_used=True,
        review_reasons=review_reasons,
    )