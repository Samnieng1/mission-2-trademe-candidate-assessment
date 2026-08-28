import os, json
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("HF_MODEL", "microsoft/Phi-4:deepinfra")
TOKEN = os.getenv("HF_TOKEN")

try:
    from openai import OpenAI
except Exception as e:
    print(json.dumps({"status": "error", "error": f"openai package not available: {e}"}))
    raise SystemExit(1)

if not TOKEN:
    print(json.dumps({"status": "error", "error": "HF_TOKEN not set in environment"}))
    raise SystemExit(1)

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKEN)

try:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    out = {"status": "ok", "model": MODEL, "resp": str(completion)[:200]}
    print(json.dumps(out, indent=2))
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
