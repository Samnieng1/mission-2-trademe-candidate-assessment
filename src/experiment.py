"""Experiment orchestration: run cases against providers and evaluate results."""
from __future__ import annotations

import json
import time
from statistics import mean
from typing import Any, Dict, List, Optional

from .config import settings
from .schemas import ProviderResult, ModelResponse
from .scoring import evaluate_fuzzy_evidence_support, precision_recall_f1, containment_score, stats_from_runs
from .storage import save_run


def load_case(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_benchmark(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_against_benchmark(parsed: ModelResponse, benchmark: Dict, candidate_profile_text: Optional[str] = None) -> Dict:
    """Compute metrics comparing a validated ModelResponse against a benchmark dict."""
    def _status_value(value):
        return getattr(value, "value", value)

    # Compare by IDs for mandatory and preferred requirements
    pred_mand_ids = [r.id for r in parsed.mandatory_requirements]
    gold_mand_ids = [r["id"] for r in benchmark.get("mandatory_requirements", [])]

    mand_prec, mand_rec, mand_f1 = precision_recall_f1(pred_mand_ids, gold_mand_ids)

    pred_pref_ids = [r.id for r in parsed.preferred_requirements]
    gold_pref_ids = [r["id"] for r in benchmark.get("preferred_requirements", [])]

    pref_prec, pref_rec, pref_f1 = precision_recall_f1(pred_pref_ids, gold_pref_ids)

    # Exact evidence agreement: preserve existing status-by-id comparison behavior.
    gold_matches = {m["requirement_id"]: _status_value(m.get("status")) for m in benchmark.get("requirement_matches", [])}
    pred_matches = {m.requirement_id: _status_value(m.status) for m in parsed.requirement_matches}
    match_items = list(gold_matches.keys())
    missing_matches = []
    if match_items:
        # Ensure every gold id appears in pred_matches; treat missing as not_matched and record warning
        for k in match_items:
            if k not in pred_matches:
                missing_matches.append(k)
                pred_matches[k] = "not_matched"

        match_acc = sum(1 for k in match_items if str(pred_matches.get(k)) == str(gold_matches.get(k))) / len(match_items)
    else:
        match_acc = 1.0

    # Build gold evidence mapping by requirement id from the benchmark if present
    gold_by_req: Dict[str, Any] = {}
    for m in (benchmark.get("requirement_matches") or []):
        if isinstance(m, dict):
            rid = m.get("requirement_id")
            gold_by_req[rid] = m.get("candidate_evidence")
        else:
            rid = getattr(m, "requirement_id", None)
            gold_by_req[rid] = getattr(m, "candidate_evidence", None)

    # Fall back to requirement-level source_evidence when per-match candidate_evidence
    # isn't provided in the benchmark payload.
    for r in (benchmark.get("mandatory_requirements") or []) + (benchmark.get("preferred_requirements") or []):
        if isinstance(r, dict):
            rid = r.get("id")
            src = r.get("source_evidence")
            if rid is not None and (rid not in gold_by_req or not gold_by_req.get(rid)):
                gold_by_req[rid] = src

    fuzzy_metrics = evaluate_fuzzy_evidence_support(
        parsed.requirement_matches,
        candidate_profile_text,
        settings.EVIDENCE_FUZZY_THRESHOLD,
        gold_evidence_by_requirement=gold_by_req,
    )

    # Missing-evidence detection accuracy
    gold_missing = {m["requirement_id"] for m in benchmark.get("missing_evidence", [])}
    pred_missing = {m.requirement_id for m in parsed.missing_evidence}
    if gold_missing:
        missing_acc = len(pred_missing & gold_missing) / len(gold_missing)
    else:
        missing_acc = 1.0

    return {
        "mandatory_precision": mand_prec,
        "mandatory_recall": mand_rec,
        "mandatory_f1": mand_f1,
        "preferred_precision": pref_prec,
        "preferred_recall": pref_rec,
        "preferred_f1": pref_f1,
        "evidence_match_accuracy": match_acc,
        "evidence_exact_match_accuracy": fuzzy_metrics["evidence_exact_match_accuracy"],
        "evidence_fuzzy_match_accuracy": fuzzy_metrics["evidence_fuzzy_match_accuracy"],
        "mean_evidence_similarity": fuzzy_metrics["mean_evidence_similarity"],
        "missing_evidence_accuracy": missing_acc,
        # CV-only: unsupported claim metrics removed
        "missing_requirement_matches": missing_matches,
        "evidence_fuzzy_debug": fuzzy_metrics["evidence_fuzzy_debug"],
    }


def run_single(
    provider,
    case: Dict,
    benchmark: Dict,
    repetitions: int = 3,
    save_raw_input: bool = False,
    candidate_profile_text: Optional[str] = None,
) -> Dict:
    """Run a single case against one provider multiple times and aggregate results."""
    times: List[float] = []
    validations = 0
    metrics_list = []
    json_valid_count = 0
    raw_responses = []
    errors = []
    profile_text = candidate_profile_text if candidate_profile_text is not None else case["candidate_profile"]

    for i in range(repetitions):
        start = time.time()
        result: ProviderResult = provider.analyse(case["job_description"], profile_text)
        elapsed = result.elapsed_seconds or (time.time() - start)
        times.append(elapsed)
        raw_responses.append(result.raw_response)
        if result.error:
            errors.append(result.error)

        if result.parsed_response:
            validations += 1
            metrics = evaluate_against_benchmark(result.parsed_response, benchmark, profile_text)
            metrics_list.append(metrics)
            json_valid_count += 1 if result.validation_status else 0

        # Save each run
        record = {
            "run_id": None,
            "timestamp": None,
            "case_id": case.get("case_id"),
            "provider": getattr(provider, "__class__", provider).__class__.__name__ if not isinstance(provider, str) else str(provider),
            "model": getattr(result, "model_name", ""),
            "success": result.success,
            "validation_status": result.validation_status,
            "elapsed_seconds": elapsed,
            "raw_response": result.raw_response,
            "parsed_response": result.parsed_response.dict() if result.parsed_response else None,
            "errors": errors,
            "job_description": case.get("job_description"),
            "candidate_profile": profile_text,
        }
        save_run(record, save_raw_input=save_raw_input)

    agg = {
        "mean_latency": stats_from_runs(times)["mean"],
        "median_latency": stats_from_runs(times)["median"],
        "stdev_latency": stats_from_runs(times)["stdev"],
        "num_valid_json": json_valid_count,
        "num_runs": repetitions,
        "errors": errors,
    }

    # Aggregate metric means across repetitions
    if metrics_list:
        agg_metrics = {}
        keys = metrics_list[0].keys()
        for k in keys:
            vals = [m[k] for m in metrics_list]
            numeric_vals = [v for v in vals if isinstance(v, (int, float, bool))]
            if not numeric_vals:
                continue
            agg_metrics[k + "__mean"] = stats_from_runs(numeric_vals)["mean"]
            agg_metrics[k + "__stdev"] = stats_from_runs(numeric_vals)["stdev"]
        agg.update(agg_metrics)

    return agg

