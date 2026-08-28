import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env for convenience when running locally
load_dotenv()

MODEL = os.environ.get("HF_MODEL", "microsoft/phi-4:deepinfra")
TOKEN = os.environ.get("HF_TOKEN")

try:
    print(f"Using model: {MODEL}")
    client = InferenceClient(api_key=TOKEN)
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0,
        max_tokens=20,
    )

    # Try multiple ways to print result
    try:
        # object-like
        msg = completion.choices[0].message
        print("message attr:", msg)
    except Exception:
        try:
            print("raw completion:", json.dumps(completion, default=str)[:1000])
        except Exception:
            print("completion repr:", repr(completion)[:1000])
except Exception as e:
    print("Error:", e)