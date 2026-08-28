Create a complete Python project named `trademe-llm-vs-slm-experiment`.

## Project purpose

Build a small, reproducible experiment that compares:

1. An OpenAI frontier language model, configured through `OPENAI_MODEL`.
2. Microsoft Phi-4 accessed through a configurable hosted Hugging Face inference endpoint.

The experiment supports a university Technology Evaluation Brief for a fictional Trade Me Jobs feature called an “AI Application Fit Coach”.

The feature compares:

* a job advertisement;
* a fictional candidate profile or CV;
* a fictional cover letter.

It must evaluate how well each model can:

* extract mandatory job requirements;
* extract preferred job requirements;
* match candidate evidence to requirements;
* identify missing candidate evidence;
* detect unsupported claims in the cover letter;
* return valid structured JSON.

This is an evaluation tool, not a production recruitment system. Do not predict whether a candidate will be hired and do not ask either model to generate an overall match score.

## Important constraints

* Use Python 3.11 or later.
* Use Streamlit for the user interface.
* Use the official OpenAI Python SDK and the current Responses API.
* Keep model names configurable through environment variables.
* Use a separate provider adapter for OpenAI and Phi-4.
* Access Phi-4 through a hosted endpoint. Do not attempt to download or run Phi-4 locally.
* Do not fine-tune either model.
* Send exactly the same task instructions and input data to both models.
* Use fictional candidate data only.
* Never commit API keys.
* Do not hard-code secrets.
* Add `.env` to `.gitignore`.
* Validate model output with Pydantic.
* Record errors rather than silently ignoring them.
* Keep the implementation simple enough for a university experiment.
* Include clear comments and type hints.
* Do not fabricate API responses.
* Do not invent provider-specific endpoint formats. Isolate any hosted Phi-4 assumptions inside its provider adapter and document what may need changing for the selected Hugging Face endpoint.

## Technology stack

Use:

* Python
* Streamlit
* OpenAI Python SDK
* `httpx`
* Pydantic
* python-dotenv
* pandas
* pytest

Use standard-library modules where appropriate, including:

* `json`
* `time`
* `uuid`
* `datetime`
* `pathlib`
* `statistics`

## Required project structure

Create this structure:

```text
trademe-llm-vs-slm-experiment/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── scoring.py
│   ├── experiment.py
│   ├── storage.py
│   │
│   └── providers/
│       ├── __init__.py
│       ├── base.py
│       ├── openai_provider.py
│       └── phi4_provider.py
│
├── data/
│   ├── cases/
│   │   ├── software_developer.json
│   │   ├── registered_nurse.json
│   │   └── retail_manager.json
│   │
│   └── benchmarks/
│       ├── software_developer_benchmark.json
│       ├── registered_nurse_benchmark.json
│       └── retail_manager_benchmark.json
│
├── results/
│   └── .gitkeep
│
└── tests/
    ├── test_schemas.py
    ├── test_scoring.py
    └── test_storage.py
```

## Environment variables

Create `.env.example` containing:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5

PHI4_API_KEY=
PHI4_ENDPOINT=
PHI4_MODEL=microsoft/phi-4

REQUEST_TIMEOUT_SECONDS=120
MAX_OUTPUT_TOKENS=3000
RESULTS_DIRECTORY=results
```

Explain in the README that:

* the user should create a separate OpenAI project and API key for this experiment;
* `OPENAI_MODEL` must be replaced if the named model is unavailable to the user’s OpenAI project;
* `PHI4_ENDPOINT` depends on the hosted Hugging Face inference service selected;
* the Phi-4 adapter may need minor adjustment based on the endpoint’s documented request and response format;
* the user must never paste keys into source code or commit the `.env` file.

## Data schema

Create Pydantic models for the following output.

Each model response must contain:

```json
{
  "mandatory_requirements": [
    {
      "id": "M1",
      "requirement": "Three years of commercial software development experience",
      "source_evidence": "Quoted or closely paraphrased evidence from the job advertisement"
    }
  ],
  "preferred_requirements": [
    {
      "id": "P1",
      "requirement": "Experience with Microsoft Azure",
      "source_evidence": "Quoted or closely paraphrased evidence from the job advertisement"
    }
  ],
  "requirement_matches": [
    {
      "requirement_id": "M1",
      "status": "matched",
      "candidate_evidence": "Exact evidence from the candidate profile",
      "reason": "Brief explanation"
    }
  ],
  "missing_evidence": [
    {
      "requirement_id": "M2",
      "reason": "No explicit evidence was found in the candidate profile"
    }
  ],
  "unsupported_cover_letter_claims": [
    {
      "claim": "I have extensive Kubernetes experience",
      "reason": "The candidate profile contains no supporting Kubernetes evidence"
    }
  ],
  "cover_letter_feedback": {
    "strengths": [],
    "weaknesses": [],
    "suggestions": []
  }
}
```

Use enums or literals for match status:

* `matched`
* `partially_matched`
* `not_matched`
* `uncertain`

Require every matched or partially matched requirement to contain candidate evidence.

Do not allow the model to include:

* protected personal characteristics;
* hiring probability;
* employability predictions;
* personality judgements;
* an overall candidate score.

## Prompt design

Create one shared system/task prompt used by both providers.

The prompt must instruct the model to:

1. Work only from the supplied job advertisement, candidate profile and cover letter.
2. Separate mandatory requirements from preferred requirements.
3. Treat wording such as “must”, “required”, “essential” and explicit licences as mandatory unless context clearly indicates otherwise.
4. Treat wording such as “preferred”, “desirable”, “advantageous” and “nice to have” as preferred.
5. Do not infer experience unless the candidate text provides reasonable evidence.
6. Do not treat related technologies as exact matches without explaining the relationship.
7. Mark ambiguous evidence as `uncertain` or `partially_matched`.
8. Quote or closely paraphrase evidence from the supplied text.
9. Identify cover-letter claims that are not supported by the candidate profile.
10. Return only valid JSON matching the required schema.
11. Never calculate an overall match score.
12. Never predict hiring success.
13. Ignore names, age, gender, nationality, ethnicity, disability, religion and other protected or irrelevant personal characteristics.

Create a helper that constructs the user message from:

* case ID;
* job advertisement;
* candidate profile;
* cover letter.

The same generated message must be passed to both providers.

## Provider abstraction

Create an abstract `ModelProvider` interface with a method similar to:

```python
def analyse(
    self,
    job_description: str,
    candidate_profile: str,
    cover_letter: str
) -> ProviderResult:
    ...
```

`ProviderResult` should include:

* provider name;
* configured model name;
* parsed structured response;
* raw response text;
* elapsed time in seconds;
* input token count when available;
* output token count when available;
* estimated cost when available;
* success status;
* validation status;
* error message when applicable.

### OpenAI provider

Implement the OpenAI provider using:

```python
from openai import OpenAI
```

Use the current Responses API.

Read the API key and model name from environment variables.

Request structured JSON where supported. Even when structured output is requested, validate the final result with Pydantic.

Do not hard-code model pricing. Cost should be:

* calculated only when pricing values are explicitly supplied through a configurable pricing dictionary; or
* shown as unavailable.

Do not invent token usage if the API does not return it.

Handle:

* missing API key;
* authentication errors;
* rate limits;
* request timeouts;
* malformed responses;
* JSON validation errors.

### Phi-4 provider

Implement Phi-4 using `httpx`.

Read:

* endpoint URL;
* API key;
* model name;
* timeout

from environment variables.

Keep the request-building and response-parsing logic in small functions so it can be adjusted for the selected Hugging Face hosted endpoint.

Support common response shapes cautiously, but do not pretend every Hugging Face endpoint uses the same schema.

If the endpoint is OpenAI-compatible, document where to adapt the request.

If the endpoint returns generated text, extract and validate the JSON.

Handle:

* missing endpoint;
* missing API key;
* HTTP errors;
* timeouts;
* malformed response bodies;
* invalid JSON;
* Pydantic validation failure.

## Fictional experiment cases

Create three clearly fictional test cases:

1. Software Developer
2. Registered Nurse
3. Retail Manager

Each JSON case must include:

```json
{
  "case_id": "",
  "industry": "",
  "job_title": "",
  "job_description": "",
  "candidate_profile": "",
  "cover_letter": "",
  "cover_letter_quality": ""
}
```

Use fictional names or remove names completely.

Each job advertisement should contain:

* four to six mandatory requirements;
* two to four preferred requirements;
* at least one requirement that requires semantic reasoning;
* at least one explicit requirement such as a licence, certification or years of experience.

Each candidate should:

* clearly meet some requirements;
* clearly miss some requirements;
* partially meet at least one requirement;
* have at least one transferable skill.

Cover-letter quality:

* Software Developer: strong
* Registered Nurse: average
* Retail Manager: deliberately weak

The deliberately weak cover letter should contain at least one unsupported claim so the models can be tested on hallucination or evidence checking.

Avoid using real people, employers, contact information or sensitive data.

## Human benchmark files

Create one benchmark file for each case.

The benchmark must be written manually in structured JSON and include:

* expected mandatory requirements;
* expected preferred requirements;
* expected requirement-to-evidence matches;
* expected missing requirements;
* expected unsupported cover-letter claims.

Use stable benchmark IDs such as `M1`, `M2`, `P1` and `P2`.

Add a note in each benchmark file stating that it is a researcher-created reference answer, not an objective hiring decision.

## Deterministic evaluation

Create scoring functions that compare each model’s validated output with the human benchmark.

Calculate these metrics:

1. Mandatory requirement precision.
2. Mandatory requirement recall.
3. Mandatory requirement F1.
4. Preferred requirement precision.
5. Preferred requirement recall.
6. Preferred requirement F1.
7. Evidence-match accuracy.
8. Missing-evidence detection accuracy.
9. Unsupported-claim detection accuracy.
10. JSON/schema validity.
11. Response time.

Because natural-language requirements may be paraphrased, implement two comparison modes:

### Strict mode

Use normalised text comparison:

* lowercase;
* trim whitespace;
* remove repeated spaces;
* remove basic punctuation.

### Researcher review mode

Allow the user to manually mark each predicted item as:

* correct;
* partially correct;
* incorrect.

Store these manual judgements with the experiment result.

Do not use another AI model as the judge by default because this would introduce another model into the comparison.

## Repeated runs

Allow each provider to run the same test case between one and five times.

Default to three repetitions.

For repeated runs, calculate:

* mean response time;
* median response time;
* standard deviation of response time;
* number of valid JSON responses;
* metric mean;
* metric variation;
* output consistency.

Define output consistency using the overlap of normalised requirement items between repeated runs. Clearly label this as a simple experimental consistency measure, not a universal model reliability metric.

## Storage

Store every experiment run as a timestamped JSON file under `results/`.

The stored record must include:

* unique run ID;
* timestamp in ISO 8601 format;
* case ID;
* provider;
* model;
* prompt version;
* response metrics;
* raw response;
* validated response when available;
* benchmark comparison;
* manual researcher judgements;
* errors;
* elapsed time;
* token counts when available;
* cost when available.

Also maintain a CSV summary containing one row per run.

Never save API keys.

Add a configurable option to avoid storing full candidate input text. Default this option to not storing the original candidate text, even though the supplied data is fictional.

## Streamlit interface

Build a clean but simple Streamlit interface.

### Sidebar

Include:

* provider selection:

  * OpenAI
  * Phi-4
  * Both
* test case selection;
* repetition count from one to five;
* strict benchmark evaluation toggle;
* save raw input toggle, default off;
* prompt version display;
* provider configuration status without revealing secrets.

### Main page

Display:

* experiment title;
* research question;
* warning that only fictional candidate data should be used;
* job advertisement;
* candidate profile;
* cover letter;
* human benchmark in an expandable section.

Add buttons:

* `Run selected experiment`
* `Run all cases`
* `Clear displayed results`

When both providers are selected, execute them sequentially for simpler debugging and fairer observation. Clearly state this in the interface.

### Results

Display:

* provider and model;
* success or failure;
* schema validity;
* elapsed time;
* token usage when available;
* estimated cost when configured;
* extracted mandatory requirements;
* extracted preferred requirements;
* evidence matches;
* missing evidence;
* unsupported cover-letter claims;
* cover-letter feedback;
* benchmark metrics;
* raw JSON in an expander.

Create a side-by-side comparison table for GPT and Phi-4 containing:

* mandatory F1;
* preferred F1;
* evidence-match accuracy;
* missing-evidence accuracy;
* unsupported-claim accuracy;
* schema-valid response rate;
* mean latency;
* token usage;
* estimated cost;
* consistency.

Do not automatically declare a winner. Show the evidence so the researcher can interpret the trade-offs.

Add download buttons for:

* current result JSON;
* summary CSV.

## README

Write a comprehensive README containing:

1. Project overview.
2. Research question.
3. Why GPT and Phi-4 are being compared.
4. Important distinction between:

   * base Phi-4 used in the experiment;
   * a possible future recruitment-specific fine-tuned Phi-4.
5. Architecture diagram in Mermaid.
6. Folder structure.
7. Windows setup instructions.
8. Virtual environment setup.
9. Dependency installation.
10. Environment-variable configuration.
11. OpenAI project-key guidance.
12. Hosted Phi-4 endpoint guidance.
13. How to run Streamlit.
14. How to run tests.
15. How to run one case.
16. How to run all cases.
17. Explanation of every metric.
18. Explanation of strict versus researcher-reviewed evaluation.
19. Privacy and security limitations.
20. Experiment limitations.
21. Suggested report wording.

Use Windows-friendly commands:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
pytest
```

Include this limitation clearly:

“The hosted Phi-4 experiment evaluates baseline model capability, latency and output behaviour. It does not demonstrate the full privacy benefit that could be achieved through self-hosting Phi-4 within Trade Me’s controlled infrastructure.”

Also include:

“The experiment uses a base, off-the-shelf Phi-4 model rather than a recruitment-specific fine-tuned version. Therefore, its results should not be interpreted as the maximum performance achievable by a domain-adapted SLM.”

## Testing

Create tests for:

* valid model output;
* invalid match status;
* missing evidence for a claimed match;
* malformed JSON handling;
* text normalisation;
* precision, recall and F1 calculations;
* empty predictions;
* duplicate predictions;
* result-file creation;
* ensuring secrets are not written to result files.

Mock all external API requests in tests.

Tests must not require real API keys or make network calls.

## Code quality

Use:

* type hints;
* docstrings;
* small functions;
* helpful exception messages;
* separation of UI, provider, evaluation and storage logic.

Avoid:

* unnecessary frameworks;
* databases;
* Docker;
* authentication systems;
* asynchronous processing;
* background jobs;
* excessive UI styling;
* complex dependency injection.

## Implementation order

Work in this order:

1. Create the folder structure.
2. Create configuration and Pydantic schemas.
3. Create fictional cases and human benchmarks.
4. Create the shared prompt.
5. Create deterministic scoring.
6. Create the provider abstraction.
7. Implement the OpenAI provider.
8. Implement the configurable Phi-4 hosted provider.
9. Implement experiment orchestration.
10. Implement JSON and CSV storage.
11. Build the Streamlit interface.
12. Add tests.
13. Write the README.
14. Check imports and resolve errors.

After creating the files, provide:

* a concise summary of what was generated;
* the exact commands to install and run it;
* any sections of the Phi-4 adapter that must be changed after I choose my hosted Hugging Face endpoint;
* any assumptions made;
* a checklist for completing the first experiment.

Do not provide only an outline. Generate the actual project files and working code.
