from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiment import run_single
from src.providers.openai_provider import OpenAIProvider
from src.providers.phi4_provider import Phi4Provider


CASES_DIR = ROOT / "data" / "cases"
BENCHMARKS_DIR = ROOT / "data" / "benchmarks"


def load_cases() -> List[Dict]:
    cases = []
    for path in sorted(CASES_DIR.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    return cases


def load_benchmarks() -> Dict[str, Dict]:
    benchmarks: Dict[str, Dict] = {}
    for path in sorted(BENCHMARKS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        benchmarks[payload.get("case_id")] = payload
    return benchmarks


def select_cases(cases: List[Dict], case_id: str) -> Iterable[Dict]:
    if case_id == "all":
        return cases
    return [case for case in cases if case.get("case_id") == case_id]


def provider_for(name: str):
    if name == "phi4":
        return Phi4Provider()
    if name == "gpt5":
        return OpenAIProvider()
    raise ValueError(f"Unsupported provider: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled Mission 2 evaluation cases without the live Streamlit UI.")
    parser.add_argument("--provider", choices=["phi4", "gpt5", "both"], default="phi4")
    parser.add_argument("--case-id", default="all", help="Case ID to run, or 'all'")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--save-raw-input", action="store_true")
    args = parser.parse_args()

    cases = load_cases()
    benchmarks = load_benchmarks()
    selected_cases = list(select_cases(cases, args.case_id))
    if not selected_cases:
        print(json.dumps({"status": "error", "error": f"No cases found for case_id={args.case_id}"}, indent=2))
        return 1

    providers = [args.provider] if args.provider != "both" else ["phi4", "gpt5"]
    results = []
    for case in selected_cases:
        benchmark = benchmarks.get(case.get("case_id"))
        if benchmark is None:
            print(json.dumps({"status": "warning", "case_id": case.get("case_id"), "warning": "Missing benchmark, skipping case"}, indent=2))
            continue

        for provider_name in providers:
            provider = provider_for(provider_name)
            aggregates = run_single(
                provider,
                case,
                benchmark,
                repetitions=args.repetitions,
                save_raw_input=args.save_raw_input,
            )
            results.append(
                {
                    "case_id": case.get("case_id"),
                    "job_title": case.get("job_title"),
                    "provider": provider_name,
                    "aggregates": aggregates,
                }
            )

    print(json.dumps({"status": "ok", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())