"""Normalization helpers for model JSON payloads.

Functions here defensively coerce variant model outputs into the canonical
ModelResponse-shaped dict expected by the application.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

ALLOWED_STATUSES = {"matched", "partially_matched", "not_matched", "uncertain"}
STATUS_MAP = {
    "met": "matched",
    "partial": "partially_matched",
    "partially_met": "partially_matched",
    "partiallymatched": "partially_matched",
    "not met": "not_matched",
    "not_met": "not_matched",
    "missing": "not_matched",
    "yes": "matched",
    "no": "not_matched",
}


def _map_status(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    key = str(s).strip().lower()
    if key in ALLOWED_STATUSES:
        return key
    if key in STATUS_MAP:
        return STATUS_MAP[key]
    return None


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        # join list items into a best-effort semicolon-separated string
        return "; ".join(str(v) for v in value if v is not None)
    return str(value)


def _normalize_status(s: Any) -> Tuple[str, Optional[str]]:
    """Normalize status values into canonical MatchStatus strings.

    Returns (normalized_status, warning_or_none)
    """
    if s is None:
        return "uncertain", None
    # unwrap single-item lists
    if isinstance(s, list) and len(s) == 1:
        s = s[0]
    key = str(s).strip().lower()
    if key in ("met", "match", "matched"):
        return "matched", None
    if key in ("partial", "partially met", "partially_met", "partiallymatched", "partiallymatched"):
        return "partially_matched", None
    if key in ("missing", "unmet", "not met", "not_met", "notmatched", "not_matched"):
        return "not_matched", None
    if key in ("unclear", "unknown", "uncertain"):
        return "uncertain", None
    # fallback
    warn = f"unrecognized status '{s}' mapped to 'uncertain'"
    logger.debug("normalise: %s", warn)
    return "uncertain", warn


def normalise_payload(payload: Any) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Normalise various model output shapes into the canonical dict.

    Returns (normalised_payload_or_none, warnings)
    If payload is irreparably malformed (eg. top-level list with >1 items)
    returns (None, [error message]).
    """
    warnings: List[str] = []

    if payload is None:
        return None, ["empty payload"]

    # If payload is a list with single item, unwrap it
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            warnings.append("unwrapped single-item list")
            payload = payload[0]
        else:
            return None, ["top-level list with multiple items is not supported"]

    if not isinstance(payload, dict):
        return None, ["payload is not a JSON object"]

    norm: Dict[str, Any] = {}

    # case_id passthrough
    if "case_id" in payload:
        norm["case_id"] = payload.get("case_id")

    # normalize requirements lists: accept multiple possible names
    def _norm_reqs(key_candidates):
        items = []
        for key in key_candidates:
            if key in payload and isinstance(payload[key], (list, dict)):
                items = payload[key]
                break
        # if dict mapping id -> data, convert
        out = []
        if isinstance(items, dict):
            for rid, rdata in items.items():
                if not isinstance(rdata, dict):
                    continue
                out.append({
                    "id": rid,
                    "requirement": rdata.get("requirement") or rdata.get("description") or rdata.get("text") or "",
                    "source_evidence": _as_text(rdata.get("evidence") or rdata.get("source") or rdata.get("evidence_job_ad")),
                })
        elif isinstance(items, list):
            for itm in items:
                if not isinstance(itm, dict):
                    continue
                out.append({
                    "id": itm.get("id") or itm.get("req_id") or itm.get("requirement_id") or "",
                    "requirement": itm.get("requirement") or itm.get("text") or itm.get("description") or "",
                    "source_evidence": _as_text(itm.get("source_evidence") or itm.get("evidence_job_ad") or itm.get("evidence") or itm.get("source")),
                })
        return out

    norm["mandatory_requirements"] = _norm_reqs(["mandatory_requirements", "mandatory", "must", "required"])
    norm["preferred_requirements"] = _norm_reqs(["preferred_requirements", "preferred", "optional"])

    # Normalize requirement_matches
    raw_matches = None
    for k in ["requirement_matches", "matches", "assessments", "requirements"]:
        if k in payload:
            raw_matches = payload[k]
            break

    norm_matches: List[Dict[str, Any]] = []
    if isinstance(raw_matches, dict):
        # dict mapping id -> data
        for rid, rdata in raw_matches.items():
            if not isinstance(rdata, dict):
                continue
            raw_status = rdata.get("status") or rdata.get("assessment") or rdata.get("result")
            status, warn = _normalize_status(raw_status)
            if warn:
                warnings.append(warn)
            candidate_evidence = _as_text(rdata.get("candidate_evidence") or rdata.get("evidence") or rdata.get("evidence_candidate"))
            reason = rdata.get("reason") or rdata.get("explanation") or rdata.get("notes") or None
            norm_matches.append({"requirement_id": rid, "status": status or "not_matched", "candidate_evidence": candidate_evidence or None, "reason": reason})
    elif isinstance(raw_matches, list):
        for itm in raw_matches:
            if not isinstance(itm, dict):
                continue
            rid = itm.get("requirement_id") or itm.get("id") or itm.get("req_id") or itm.get("requirement")
            raw_status = itm.get("status") or itm.get("assessment") or itm.get("result")
            status, warn = _normalize_status(raw_status)
            if warn:
                warnings.append(warn)
            candidate_evidence = _as_text(itm.get("candidate_evidence") or itm.get("evidence") or itm.get("evidence_candidate") or itm.get("evidence_list"))
            reason = itm.get("reason") or itm.get("explanation") or itm.get("notes") or None
            norm_matches.append({"requirement_id": rid or "", "status": status or "not_matched", "candidate_evidence": candidate_evidence or None, "reason": reason})
    else:
        # Try to infer matches from requirements structures (some models include assessment inside requirement entries)
        for r in norm["mandatory_requirements"] + norm["preferred_requirements"]:
            # look for fields in payload matching requirement id
            rid = r.get("id")
            if not rid:
                continue
            # search for an object under payload with this id?
        # If still empty, leave as empty list
        norm_matches = []

    norm["requirement_matches"] = norm_matches

    # missing_evidence
    missing = payload.get("missing_evidence") or payload.get("missing_mandatory") or []
    out_missing = []
    if isinstance(missing, list):
        for m in missing:
            if isinstance(m, dict):
                out_missing.append({"requirement_id": m.get("requirement_id") or m.get("id") or "", "reason": m.get("reason") or ""})
            else:
                out_missing.append({"requirement_id": str(m), "reason": ""})
    norm["missing_evidence"] = out_missing

    # CV-only mode: do not include cover-letter related fields

    return norm, warnings
