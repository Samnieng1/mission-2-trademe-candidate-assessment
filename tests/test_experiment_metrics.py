from types import SimpleNamespace

from src.config import settings
from src.experiment import evaluate_against_benchmark, run_single
from src.schemas import ModelResponse, ProviderResult
from src.scoring import evaluate_fuzzy_evidence_support, normalize_text


class StatefulProvider:
    def __init__(self, results):
        self._results = list(results)

    def analyse(self, job_description, candidate_profile):
        return self._results.pop(0)


def _build_response(evidence_text="Built Python dashboards for customer reporting.", status="matched"):
    return ModelResponse.parse_obj(
        {
            "mandatory_requirements": [
                {"id": "M1", "requirement": "Python dashboards", "source_evidence": "Python dashboards required"}
            ],
            "preferred_requirements": [],
            "requirement_matches": [
                {
                    "requirement_id": "M1",
                    "status": status,
                    "candidate_evidence": evidence_text,
                    "reason": "supported",
                }
            ],
            "missing_evidence": [],
        }
    )


BENCHMARK = {
    "mandatory_requirements": [{"id": "M1", "requirement": "Python dashboards", "source_evidence": "Python dashboards required"}],
    "preferred_requirements": [],
    "requirement_matches": [{"requirement_id": "M1", "status": "matched", "candidate_evidence": "Built Python dashboards for customer reporting."}],
    "missing_evidence": [],
}


def test_normalize_text_joins_list_values():
    assert normalize_text(["Hello,", "WORLD!"]) == "hello world"



def test_exact_quotation_passes_both_metrics():
    matches = [SimpleNamespace(requirement_id="M1", status="matched", candidate_evidence="Built Python dashboards for customer reporting.")]
    cv_text = "Built Python dashboards for customer reporting. Managed delivery across teams."

    metrics = evaluate_fuzzy_evidence_support(matches, cv_text, settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_exact_match_accuracy"] == 1.0
    assert metrics["evidence_fuzzy_match_accuracy"] == 1.0
    assert metrics["mean_evidence_similarity"] == 1.0



def test_punctuation_and_caps_pass_fuzzy_matching():
    matches = [SimpleNamespace(requirement_id="M1", status="matched", candidate_evidence="BUILT, PYTHON dashboards for customer reporting!")]
    cv_text = "Built Python dashboards for customer reporting."

    metrics = evaluate_fuzzy_evidence_support(matches, cv_text, settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_fuzzy_match_accuracy"] == 1.0



def test_close_paraphrase_passes_fuzzy_but_fails_exact():
    matches = [SimpleNamespace(requirement_id="M1", status="matched", candidate_evidence="Built Python dashboards for customer reporting")]
    cv_text = "Created Python dashboards for customer reports and analytics."

    metrics = evaluate_fuzzy_evidence_support(matches, cv_text, settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_exact_match_accuracy"] == 0.0
    assert metrics["evidence_fuzzy_match_accuracy"] == 1.0
    assert metrics["mean_evidence_similarity"] >= settings.EVIDENCE_FUZZY_THRESHOLD



def test_unrelated_evidence_fails_both_metrics():
    matches = [SimpleNamespace(requirement_id="M1", status="matched", candidate_evidence="Performed advanced neurosurgery")]
    cv_text = "Built Python dashboards for customer reporting."

    metrics = evaluate_fuzzy_evidence_support(matches, cv_text, settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_exact_match_accuracy"] == 0.0
    assert metrics["evidence_fuzzy_match_accuracy"] == 0.0



def test_null_evidence_for_matched_status_counts_as_unsupported():
    matches = [SimpleNamespace(requirement_id="M1", status="matched", candidate_evidence=None)]

    metrics = evaluate_fuzzy_evidence_support(matches, "Relevant CV text", settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_exact_match_accuracy"] == 0.0
    assert metrics["evidence_fuzzy_match_accuracy"] == 0.0



def test_null_evidence_for_not_matched_status_is_excluded():
    matches = [SimpleNamespace(requirement_id="M1", status="not_matched", candidate_evidence=None)]

    metrics = evaluate_fuzzy_evidence_support(matches, "Relevant CV text", settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_exact_match_accuracy"] is None
    assert metrics["evidence_fuzzy_match_accuracy"] is None
    assert metrics["mean_evidence_similarity"] is None



def test_all_seven_supported_evidence_items_produce_fuzzy_accuracy_one():
    matches = [
        SimpleNamespace(requirement_id=f"M{i}", status="matched", candidate_evidence=f"Evidence item {i}")
        for i in range(1, 8)
    ]
    cv_text = "\n".join(f"Evidence item {i}" for i in range(1, 8))

    metrics = evaluate_fuzzy_evidence_support(matches, cv_text, settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_fuzzy_match_accuracy"] == 1.0
    assert metrics["evidence_exact_match_accuracy"] == 1.0



def test_no_claimed_evidence_returns_null():
    metrics = evaluate_fuzzy_evidence_support([], "Any CV text", settings.EVIDENCE_FUZZY_THRESHOLD)

    assert metrics["evidence_exact_match_accuracy"] is None
    assert metrics["evidence_fuzzy_match_accuracy"] is None
    assert metrics["mean_evidence_similarity"] is None



def test_similarity_values_are_flat_numeric_values():
    matches = [SimpleNamespace(requirement_id="M1", status="uncertain", candidate_evidence="Python reporting dashboards")]
    cv_text = "Built Python reporting dashboards for customer analytics."

    metrics = evaluate_fuzzy_evidence_support(matches, cv_text, settings.EVIDENCE_FUZZY_THRESHOLD)

    debug_row = metrics["evidence_fuzzy_debug"][0]
    assert isinstance(debug_row["highest_similarity"], float)



def test_evaluate_against_benchmark_exposes_new_metrics():
    parsed = _build_response()
    cv_text = "Built Python dashboards for customer reporting."

    metrics = evaluate_against_benchmark(parsed, BENCHMARK, cv_text)

    assert metrics["evidence_match_accuracy"] == 1.0
    assert metrics["evidence_exact_match_accuracy"] == 1.0
    assert metrics["evidence_fuzzy_match_accuracy"] == 1.0
    assert metrics["mean_evidence_similarity"] == 1.0



def test_repeated_run_aggregation_handles_null_values_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RESULTS_DIRECTORY", tmp_path / "results")
    case = {
        "case_id": "case-1",
        "job_description": "Need Python dashboards",
        "candidate_profile": "Built Python dashboards for customer reporting.",
    }
    provider = StatefulProvider(
        [
            ProviderResult(
                provider_name="test",
                model_name="test-model",
                parsed_response=_build_response(),
                elapsed_seconds=1.0,
                success=True,
                validation_status=True,
            ),
            ProviderResult(
                provider_name="test",
                model_name="test-model",
                parsed_response=ModelResponse.parse_obj(
                    {
                        "mandatory_requirements": [{"id": "M1", "requirement": "Python dashboards", "source_evidence": "Python dashboards required"}],
                        "preferred_requirements": [],
                        "requirement_matches": [{"requirement_id": "M1", "status": "not_matched", "reason": "missing"}],
                        "missing_evidence": [{"requirement_id": "M1", "reason": "missing"}],
                    }
                ),
                elapsed_seconds=1.2,
                success=True,
                validation_status=True,
            ),
        ]
    )

    agg = run_single(provider, case, BENCHMARK, repetitions=2, save_raw_input=False)

    assert agg["evidence_exact_match_accuracy__mean"] == 1.0
    assert agg["evidence_fuzzy_match_accuracy__mean"] == 1.0
    assert agg["mean_evidence_similarity__mean"] == 1.0


def test_run_single_accepts_candidate_profile_override(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RESULTS_DIRECTORY", tmp_path / "results")
    case = {
        "case_id": "case-1",
        "job_description": "Need Python dashboards",
        "candidate_profile": "Stored profile does not contain the evidence.",
    }
    override_cv = "Built Python dashboards for customer reporting."
    provider = StatefulProvider(
        [
            ProviderResult(
                provider_name="test",
                model_name="test-model",
                parsed_response=_build_response(),
                elapsed_seconds=1.0,
                success=True,
                validation_status=True,
            )
        ]
    )

    agg = run_single(
        provider,
        case,
        BENCHMARK,
        repetitions=1,
        save_raw_input=False,
        candidate_profile_text=override_cv,
    )

    assert agg["evidence_exact_match_accuracy__mean"] == 1.0
    assert agg["evidence_fuzzy_match_accuracy__mean"] == 1.0
    assert agg["mean_evidence_similarity__mean"] == 1.0
