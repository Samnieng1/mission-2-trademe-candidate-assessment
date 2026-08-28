# Trade Me AI Candidate Fit Prototype

This repository contains a Streamlit prototype for a Trade Me Jobs candidate-fit workflow.

The live app is intentionally simple:
- select a job
- upload or paste a CV
- run AI analysis
- validate the structured output before showing it
- calculate a deterministic Candidate Fit Score

Mission 2 keeps Phi-4 as the primary model and uses GPT-5 only when validation requires internal review.

## Mission 2 architecture

Runtime flow:

```text
CV
-> text extraction
-> job description
-> Phi-4
-> structured response
-> validation layer
-> PASS: deterministic scoring
-> REVIEW REQUIRED: GPT-5 review -> validation -> deterministic scoring
```

Validation is implemented separately from the UI in `src/validation.py`.

GPT-5 is not exposed as a candidate choice in the live demo.

## Deterministic scoring

The final numeric score is calculated in `src/candidate_scoring.py`, not by the model.

Current weighting:
- mandatory requirements: 80 points total
- preferred requirements: 20 points total

Status values:
- `matched` = 1.0
- `partially_matched` = 0.5
- `uncertain` = 0.25
- `not_matched` = 0.0

The score is a prototype fit indicator, not a hiring decision or hiring probability.

## Project structure

```text
app.py
data/
	benchmarks/
	cases/
scripts/
	run_evaluation.py
src/
	candidate_scoring.py
	config.py
	document_parser.py
	experiment.py
	normalise.py
	prompts.py
	schemas.py
	scoring.py
	storage.py
	validation.py
	providers/
		base.py
		openai_provider.py
		phi4_provider.py
tests/
results/
```

`app.py` is the live demo.

`scripts/run_evaluation.py` is the separate Mission 2 evaluation entry point for controlled benchmark runs.

## Setup

1. Create a virtual environment: `py -3.11 -m venv .venv`
2. Activate it in PowerShell: `.venv\Scripts\Activate.ps1`
3. Install dependencies: `py -3.11 -m pip install -r requirements.txt`

## Environment variables

Copy `.env.example` to `.env` and set your own credentials.

Required variables:
- `OPENAI_API_KEY`
- `OPENAI_MODEL` default: `gpt-5`
- `HF_TOKEN`
- `HF_MODEL` default: `microsoft/Phi-4`

Optional variables:
- `REQUEST_TIMEOUT_SECONDS`
- `MAX_OUTPUT_TOKENS`
- `EVIDENCE_FUZZY_THRESHOLD`
- `RESULTS_DIRECTORY`

`.env` must remain uncommitted.

## Running the app

Start the live demo:

`python -m streamlit run app.py`

## Running Mission 2 evaluation

Example commands:

- `python scripts/run_evaluation.py --provider phi4 --case-id all --repetitions 3`
- `python scripts/run_evaluation.py --provider gpt5 --case-id SD001 --repetitions 3`
- `python scripts/run_evaluation.py --provider both --case-id all --repetitions 5`

The evaluation runner reuses the existing providers, benchmark logic, scoring utilities, and result storage.

## File support and limitations

- Supported CV inputs: `TXT`, `DOCX`, text-based `PDF`
- Maximum upload size: 5 MB
- Scanned PDFs are rejected because extractable text is required
- Uploaded CV validation uses the CV text itself as the grounding source
- Benchmark precision/recall/F1 metrics are kept for controlled evaluation only and are not shown in the live candidate flow

## Testing

Run the test suite with:

`py -3.11 -m pytest -q`
