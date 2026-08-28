import os, json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env if present so scripts work from project root without manual env setup
load_dotenv()

MODEL = os.getenv("HF_MODEL", "microsoft/Phi-4:deepinfra")

# Read token from environment (including .env)
client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": "You are a brief assistant."},
                  {"role": "user", "content": "What is 2+2?"}],
        temperature=0,
        max_tokens=10,
    )
    # Print a compact success summary
    out = {
        "status": "ok",
        "model": MODEL,
        "response_preview": str(resp).replace('\n', ' ')[:100],
    }
    print(json.dumps(out, indent=2))
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
