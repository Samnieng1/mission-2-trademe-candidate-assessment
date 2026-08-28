"""Deterministic candidate scoring utilities.

All functions are pure and never call external services or AI models.
They operate on simple Python structures so they are easy to unit test.
"""
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "build_candidate_fit_breakdown",
    "calculate_candidate_fit_score_components",
    "calculate_candidate_fit_score",
    "generate_summary",
]


def _field_list(obj: Any, field_name: str) -> List[Any]:
    if hasattr(obj, field_name):
        value = getattr(obj, field_name)
        return value if value is not None else []
    if isinstance(obj, dict):
        value = obj.get(field_name, [])
        return value if value is not None else []
    return []


def _score_value_from_status(status: Any) -> float:
    mapping = {
        "matched": 1.0,
        "partially_matched": 0.5,
        "uncertain": 0.25,
        "not_matched": 0.0,
    }
    try:
        # Accept enums (with .value) or raw strings
        if hasattr(status, "value"):
            s = str(status.value).strip().lower()
        else:
            s = str(status).strip().lower()
            if "." in s:
                s = s.split(".")[-1]
    except Exception:
        s = str(status)
    return mapping.get(s, 0.0)


def _status_name(status: Any) -> str:
    try:
        if hasattr(status, "value"):
            s = str(status.value).strip().lower()
        else:
            s = str(status).strip().lower()
            if "." in s:
                s = s.split(".")[-1]
    except Exception:
        s = str(status)
    return s


def build_candidate_fit_breakdown(parsed_response: Any) -> Dict[str, Any]:
    """Return score components plus a stable breakdown for the UI.

    The numeric scoring algorithm remains unchanged:
    - mandatory requirements contribute 80 points total
    - preferred requirements contribute 20 points total
    """
    mand = _field_list(parsed_response, "mandatory_requirements")
    pref = _field_list(parsed_response, "preferred_requirements")
    matches = _field_list(parsed_response, "requirement_matches")
    missing_evidence = _field_list(parsed_response, "missing_evidence")

    mandatory_ids = [r.id if hasattr(r, "id") else r.get("id") for r in mand]
    preferred_ids = [r.id if hasattr(r, "id") else r.get("id") for r in pref]
    status_map: Dict[str, str] = {}
    for match in matches:
        requirement_id = match.requirement_id if hasattr(match, "requirement_id") else match.get("requirement_id")
        status_map[requirement_id] = _status_name(match.status if hasattr(match, "status") else match.get("status"))

    def counts_for(requirement_ids: List[str]) -> Dict[str, int]:
        counts = {
            "matched": 0,
            "partially_matched": 0,
            "uncertain": 0,
            "not_matched": 0,
        }
        for requirement_id in requirement_ids:
            counts[_status_name(status_map.get(requirement_id, "not_matched"))] = counts.get(
                _status_name(status_map.get(requirement_id, "not_matched")),
                0,
            ) + 1
        return counts

    mandatory_counts = counts_for([requirement_id for requirement_id in mandatory_ids if requirement_id])
    preferred_counts = counts_for([requirement_id for requirement_id in preferred_ids if requirement_id])
    components = calculate_candidate_fit_score_components(parsed_response)

    return {
        **components,
        "mandatory_total": len([requirement_id for requirement_id in mandatory_ids if requirement_id]),
        "preferred_total": len([requirement_id for requirement_id in preferred_ids if requirement_id]),
        "mandatory_counts": mandatory_counts,
        "preferred_counts": preferred_counts,
        "missing_evidence_count": len(missing_evidence),
    }


def calculate_candidate_fit_score_components(parsed_response: Any) -> Dict[str, float]:
    """Calculate the candidate fit score components from a parsed ModelResponse-like object.

    Returns dict with keys: mandatory_component, preferred_component, unsupported_component, total_percentage
    """
    logger.debug("calculate_candidate_fit_score_components: parsed_response type=%s", type(parsed_response))
    # Extract lists
    mand = _field_list(parsed_response, "mandatory_requirements")
    pref = _field_list(parsed_response, "preferred_requirements")
    matches = _field_list(parsed_response, "requirement_matches")

    # Build map of requirement_id -> status
    status_map: Dict[str, Any] = {}
    for m in matches:
        try:
            rid = m.requirement_id if hasattr(m, "requirement_id") else m.get("requirement_id")
            raw_status = m.status if hasattr(m, "status") else m.get("status")
            if hasattr(raw_status, "value"):
                st = str(raw_status.value).strip().lower()
            else:
                st = str(raw_status).strip().lower()
                if "." in st:
                    st = st.split(".")[-1]
            status_map[rid] = st
        except Exception:
            logger.debug("candidate_scoring: failed to parse match entry: %s", getattr(m, "__dict__", str(m)))
            continue

    mandatory_ids = [r.id if hasattr(r, "id") else r.get("id") for r in mand]
    preferred_ids = [r.id if hasattr(r, "id") else r.get("id") for r in pref]

    logger.debug(
        "candidate_scoring: mand=%d pref=%d matches=%d",
        len(mandatory_ids),
        len(preferred_ids),
        len(status_map),
    )

    # Sum values
    mand_total = len([x for x in mandatory_ids if x])
    pref_total = len([x for x in preferred_ids if x])

    mand_sum = 0.0
    for rid in mandatory_ids:
        val = _score_value_from_status(status_map.get(rid, "not_matched"))
        mand_sum += val

    pref_sum = 0.0
    for rid in preferred_ids:
        val = _score_value_from_status(status_map.get(rid, "not_matched"))
        pref_sum += val

    # Components (TASKS6 CV-only): 80% mandatory, 20% preferred, no cover-letter component
    mandatory_component = (mand_sum / mand_total * 80.0) if mand_total else 0.0
    preferred_component = (pref_sum / pref_total * 20.0) if pref_total else 0.0

    total = mandatory_component + preferred_component
    total = max(0.0, min(100.0, total))

    return {
        "mandatory_component": round(mandatory_component, 2),
        "preferred_component": round(preferred_component, 2),
        "total_percentage": round(total, 2),
    }


def calculate_candidate_fit_score(parsed_response: Any) -> float:
    """Backward-compatible wrapper returning the total percentage as float."""
    comps = calculate_candidate_fit_score_components(parsed_response)
    return comps["total_percentage"]


def generate_summary(mandatory_matches: List[str], mandatory_missing: List[str], preferred_matches: List[str], preferred_missing: List[str]) -> Dict[str, List[str]]:
    return {
        "mandatory_matched_evidence": mandatory_matches,
        "mandatory_missing_evidence": mandatory_missing,
        "preferred_matched_evidence": preferred_matches,
        "preferred_missing_evidence": preferred_missing,
    }
