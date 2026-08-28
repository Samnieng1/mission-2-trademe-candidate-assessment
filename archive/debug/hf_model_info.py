import os, json
from dotenv import load_dotenv
import httpx

load_dotenv()
MODEL = os.getenv("HF_MODEL", "microsoft/Phi-4:deepinfra")
TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    print(json.dumps({"status": "error", "error": "HF_TOKEN not set"}))
    raise SystemExit(1)

# The model id in URL should be the part before any colon
model_id = MODEL.split(":")[0]
url = f"https://api-inference.huggingface.co/models/{model_id}"
headers = {"Authorization": f"Bearer {TOKEN}"}

try:
    r = httpx.get(url, headers=headers, timeout=15.0)
    try:
        data = r.json()
    except Exception:
        data = r.text
    out = {"status": "ok", "http_status": r.status_code, "response": data}
    print(json.dumps(out, indent=2))
except Exception as e:
    print(json.dumps({"status": "error", "error": str(e)}))
