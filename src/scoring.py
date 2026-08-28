"""Deterministic scoring utilities for comparing model output to benchmarks.

Implements strict text normalisation and strict-mode comparison functions, plus
helpers for precision/recall/F1 and simple accuracy metrics.
"""
from __future__ import annotations

import difflib
import re
import statistics
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def normalize_text(s: Any) -> str:
    """Normalise text for strict comparison.

    Steps:
    - lower-case
    - strip
    - collapse whitespace
    - remove basic punctuation (.,;:()[]"')
    """
    if s is None:
        return ""
    if isinstance(s, list):
        s = " ".join(str(item) for item in s if item is not None)
    else:
        s = str(s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s


def set_of_normalised(items: Iterable[str]) -> Set[str]:
    return {normalize_text(i) for i in items if i}


def precision_recall_f1(preds: Iterable[str], gold: Iterable[str]) -> Tuple[float, float, float]:
    """Compute precision, recall, F1 between two lists of textual items (strict mode).

    Returns (precision, recall, f1). If both sets empty, returns (1.0, 1.0, 1.0).
    """
    pset = set_of_normalised(preds)
    gset = set_of_normalised(gold)

    if not pset and not gset:
        return 1.0, 1.0, 1.0

    tp = len(pset & gset)
    prec = tp / len(pset) if pset else 0.0
    rec = tp / len(gset) if gset else 0.0
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return prec, rec, f1


def accuracy(preds: Iterable[str], gold: Iterable[str]) -> float:
    """Compute simple accuracy as fraction of gold items correctly predicted.

    Uses normalised set membership.
    """
    pset = set_of_normalised(preds)
    gset = set_of_normalised(gold)
    if not gset:
        return 1.0 if not pset else 0.0
    tp = len(pset & gset)
    return tp / len(gset)


def containment_score(preds: Iterable[str], gold: Iterable[str]) -> float:
    """Alias for accuracy for clarity in some metrics.
    """
    return accuracy(preds, gold)


def stats_from_runs(values: List[float]) -> dict:
    """Return mean, median, stdev for a list of floats. stdev is 0.0 if fewer than 2 values."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}
    # Accept only numeric values (int, float, bool). Convert to float.
    nums: List[float] = []
    skipped = 0
    for v in values:
        if isinstance(v, (int, float, bool)):
            try:
                nums.append(float(v))
            except Exception:
                skipped += 1
        else:
            skipped += 1

    if skipped:
        logger.debug("stats_from_runs: skipped %d non-numeric values", skipped)

    if not nums:
        return {"mean": 0.0, "median": 0.0, "stdev": 0.0}

    mean_val = statistics.mean(nums)
    median_val = statistics.median(nums)
    stdev_val = statistics.pstdev(nums) if len(nums) > 1 else 0.0
    return {"mean": mean_val, "median": median_val, "stdev": stdev_val}


def split_text_segments(text: Any) -> List[str]:
    normalized_text = normalize_text(text)
    if not normalized_text:
        return []

    raw_text = "" if text is None else ("\n".join(str(item) for item in text if item is not None) if isinstance(text, list) else str(text))
    segments: List[str] = []

    paragraphs = re.split(r"\n\s*\n+", raw_text)
    segments.extend(paragraphs)

    lines = [line for line in raw_text.splitlines() if line.strip()]
    segments.extend(lines)

    sentences = re.split(r"(?<=[.!?])\s+", raw_text)
    segments.extend(sentences)

    seen = set()
    out: List[str] = []
    for segment in segments:
        normalized_segment = normalize_text(segment)
        if normalized_segment and normalized_segment not in seen:
            seen.add(normalized_segment)
            out.append(normalized_segment)
    return out


def evaluate_fuzzy_evidence_support(
    requirement_matches: Iterable[Any],
    candidate_profile_text: Any,
    threshold: float,
    gold_evidence_by_requirement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate fuzzy and exact support for model-produced evidence.

    If `gold_evidence_by_requirement` is provided and contains a non-empty
    value for a requirement id, the comparison is performed against the
    gold evidence. Otherwise the comparison is performed against
    `candidate_profile_text` (CV) as before.
    """
    claimed_statuses = {"matched", "partially_matched", "uncertain"}

    cv_normalized = normalize_text(candidate_profile_text)
    cv_segments = split_text_segments(candidate_profile_text)

    debug_rows: List[Dict[str, Any]] = []
    supported_flags: List[float] = []
    exact_flags: List[float] = []
    similarity_values: List[float] = []

    for match in requirement_matches or []:
        requirement_id = getattr(match, "requirement_id", None)
        status = getattr(match, "status", None)
        candidate_evidence = getattr(match, "candidate_evidence", None)

        status_value = getattr(status, "value", status)
        status_normalized = normalize_text(status_value)
        if status_normalized not in claimed_statuses:
            continue

        evidence_normalized = normalize_text(candidate_evidence)

        # Determine comparison target.
        # If a gold mapping was explicitly provided, only compare when gold exists
        # for the requirement. Do NOT fall back to CV in that case.
        gold_evidence = None
        use_gold_mapping = gold_evidence_by_requirement is not None
        if use_gold_mapping and requirement_id is not None:
            gold_evidence = gold_evidence_by_requirement.get(requirement_id)

        if use_gold_mapping:
            if not gold_evidence:
                # Record debug row indicating skipped comparison due to missing gold
                debug_rows.append(
                    {
                        "requirement_id": requirement_id,
                        "candidate_evidence": candidate_evidence,
                        "comparison_target": "gold_missing",
                        "direct_match": False,
                        "highest_similarity": 0.0,
                        "matched_segment": "",
                        "fuzzy_supported": False,
                        "skipped_no_gold": True,
                    }
                )
                # Skip this match when gold mapping was requested but missing
                continue
            target_normalized = normalize_text(gold_evidence)
            target_segments = split_text_segments(gold_evidence)
            target_label = "gold"
        else:
            # No gold mapping provided: compare against candidate profile (CV)
            target_normalized = cv_normalized
            target_segments = cv_segments
            target_label = "cv"

        direct_match = bool(evidence_normalized and target_normalized and evidence_normalized in target_normalized)
        highest_similarity = 1.0 if direct_match else 0.0
        matched_segment = evidence_normalized if direct_match else ""

        if not direct_match and evidence_normalized:
            for segment in target_segments:
                similarity = difflib.SequenceMatcher(None, evidence_normalized, segment).ratio()
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    matched_segment = segment

        fuzzy_supported = bool(evidence_normalized) and (direct_match or highest_similarity >= threshold)
        exact_supported = bool(evidence_normalized) and direct_match

        exact_flags.append(1.0 if exact_supported else 0.0)
        supported_flags.append(1.0 if fuzzy_supported else 0.0)
        similarity_values.append(float(highest_similarity))
        debug_rows.append(
            {
                "requirement_id": requirement_id,
                "candidate_evidence": candidate_evidence,
                "comparison_target": target_label,
                "direct_match": direct_match,
                "highest_similarity": float(highest_similarity),
                "matched_segment": matched_segment,
                "fuzzy_supported": fuzzy_supported,
            }
        )

    if not supported_flags:
        return {
            "evidence_exact_match_accuracy": None,
            "evidence_fuzzy_match_accuracy": None,
            "mean_evidence_similarity": None,
            "evidence_fuzzy_debug": debug_rows,
        }

    return {
        "evidence_exact_match_accuracy": sum(exact_flags) / len(exact_flags),
        "evidence_fuzzy_match_accuracy": sum(supported_flags) / len(supported_flags),
        "mean_evidence_similarity": statistics.mean(similarity_values),
        "evidence_fuzzy_debug": debug_rows,
    }

