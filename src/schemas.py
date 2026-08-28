"""Pydantic schemas for model responses and provider results.

These models implement the required schema for validated model output and the
provider result wrapper used by the experiment orchestration.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, Extra, root_validator, validator


class MatchStatus(str, Enum):
    matched = "matched"
    partially_matched = "partially_matched"
    not_matched = "not_matched"
    uncertain = "uncertain"


class RequirementItem(BaseModel):
    id: str
    requirement: str
    source_evidence: str

    class Config:
        anystr_strip_whitespace = True


class RequirementMatch(BaseModel):
    requirement_id: str
    status: MatchStatus
    candidate_evidence: Optional[str] = None
    reason: Optional[str] = None

    @validator("candidate_evidence", always=True)
    def evidence_required_for_matches(cls, v, values):
        status = values.get("status")
        if status in (MatchStatus.matched, MatchStatus.partially_matched) and not v:
            raise ValueError("candidate_evidence is required for matched or partially_matched status")
        return v


class MissingEvidence(BaseModel):
    requirement_id: str
    reason: str


class UnsupportedClaim(BaseModel):
    claim: str
    reason: str


class CoverLetterFeedback(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ModelResponse(BaseModel):
    case_id: Optional[str] = None
    mandatory_requirements: List[RequirementItem] = Field(default_factory=list)
    preferred_requirements: List[RequirementItem] = Field(default_factory=list)
    requirement_matches: List[RequirementMatch] = Field(default_factory=list)
    missing_evidence: List[MissingEvidence] = Field(default_factory=list)
    # CV-only schema: do not include cover-letter related fields

    class Config:
        # Forbid unexpected fields (eg. overall_score, hiring_probability)
        extra = Extra.forbid
        anystr_strip_whitespace = True

    @root_validator(skip_on_failure=True)
    def ensure_ids_exist(cls, values):
        # Ensure requirement_matches reference valid IDs present in mandatory or preferred lists
        mandatory = {r.id for r in values.get("mandatory_requirements", [])}
        preferred = {r.id for r in values.get("preferred_requirements", [])}
        known = mandatory.union(preferred)

        for match in values.get("requirement_matches", []):
            if match.requirement_id not in known:
                raise ValueError(f"requirement_matches contains unknown requirement_id: {match.requirement_id}")

        for miss in values.get("missing_evidence", []):
            if miss.requirement_id not in known:
                raise ValueError(f"missing_evidence contains unknown requirement_id: {miss.requirement_id}")

        return values


class ProviderResult(BaseModel):
    provider_name: str
    model_name: str
    parsed_response: Optional[ModelResponse] = None
    raw_response: Optional[str] = None
    elapsed_seconds: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    normalization_applied: bool = False
    normalization_warnings: List[str] = Field(default_factory=list)
    success: bool = False
    validation_status: bool = False
    error: Optional[str] = None

