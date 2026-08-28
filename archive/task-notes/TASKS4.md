Refactor the existing Phi-4 provider implementation to use Hugging Face Inference Providers instead of any placeholder implementation or Azure endpoint.

Note: the workspace is currently running in a CV-only mode for the Candidate Assessment demo — the cover-letter UI and related display/export logic have been moved to `backup/cover_letter_ui_backup.py`. This change was made to reduce output variability during testing; restore the original UI from the backup file if needed.

Do not change any application behaviour or UI.

The objective is for GPT-5 and Phi-4 to be interchangeable through the existing provider abstraction.
# Dependencies
Install and use: huggingface_hub

Do not use transformers, ollama, llama.cpp, or local model inference.

The application must use the hosted Hugging Face Inference Providers service.
# Environment Variables
Replace the existing Phi-4 configuration.

Remove support for: PHI4_API_KEY
PHI4_ENDPOINT
Instead use:
HF_TOKEN=<huggingface_access_token>

HF_MODEL=microsoft/Phi-4
Continue using:OPENAI_API_KEY=...

# Provider Architecture

Reuse the existing provider abstraction.

Maintain a common interface similar to:
provider.generate(
    system_prompt,
    user_prompt,
    response_schema
)

Both providers should return exactly the same internal response object.

The rest of the application must not know whether GPT-5 or Phi-4 generated the response.

# Create or Update
Update: src/providers/phi4_provider.py

# Hugging Face Client
Use: from huggingface_hub import InferenceClient
Initialise: client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)
Use the model: os.getenv("HF_MODEL")
Default: microsoft/Phi-4

# Chat Completion

Use the Hugging Face Chat Completions API.

The provider should send:

system message
user message

Use:

Temperature:0

Use a configurable maximum token limit.

# Prompt

Reuse exactly the same prompt currently used by GPT-5.

Do not create a separate Phi-4 prompt.

Both models must receive identical instructions.

This ensures a fair comparison.

# JSON Output

The application already expects structured JSON.

Require Phi-4 to return exactly the same JSON schema as GPT-5.

Validate the returned JSON.

If validation fails:

retry once
if still invalid, return a structured provider error

Do not crash the application.

# Error Handling

Handle:

invalid Hugging Face token
rate limiting
timeout
unavailable model
malformed JSON
network failure

Return meaningful error messages that can be displayed inside Streamlit.

# Logging

Reuse the existing logging.

Log:

provider name
model name
response time
prompt tokens (if available)
completion tokens (if available)
estimated cost (if available)

If Hugging Face does not return token usage, leave those fields as null rather than inventing values.

# Provider Selection

Keep the existing selector:
GPT-5

Phi-4

Both

When "Both" is selected:

Execute both providers independently.

Display both outputs side-by-side.

# Comparison

Ensure both providers produce compatible output so the comparison table continues working.

Compare:

Candidate Fit Score
Mandatory matches
Preferred matches
Missing requirements
Unsupported claims
Latency
Token usage (when available)
Estimated cost (when available)

# Code Quality

Reuse existing code wherever possible.

Avoid duplicate logic.

Shared functionality such as:

prompt construction
JSON validation
response parsing
scoring

should remain outside the provider.

Providers should only be responsible for:

calling the model
returning the raw structured response

# README

Update the README.

Document:

GPT-5

Provider:

OpenAI API

Authentication:

OPENAI_API_KEY

# Phi-4

Provider:

Hugging Face Inference Providers

Authentication:

HF_TOKEN

Model:

microsoft/Phi-4

No local GPU is required.

# Acceptance Criteria

The implementation is complete when:

GPT-5 continues to work unchanged.
Phi-4 uses Hugging Face Inference Providers.
Both providers produce the same JSON schema.
The application can switch between GPT-5 and Phi-4 without changing any other code.
"Both" mode successfully executes both providers and displays the comparison.
All existing benchmark and Candidate Application Demo functionality continues to work without modification.

# One additional recommendation

create a small BaseProvider abstract class if you don't already have one. For example:

class BaseProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        pass

Then implement:
OpenAIProvider(BaseProvider)
Phi4Provider(BaseProvider)
