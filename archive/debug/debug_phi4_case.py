from dotenv import load_dotenv
load_dotenv()

import json
from pathlib import Path
from src.providers.phi4_provider import Phi4Provider
from src.prompts import build_user_message

CASE_FILE = Path("data/cases/software_developer.json")

data = json.loads(CASE_FILE.read_text(encoding="utf-8"))
job = data.get("job_description", "")
profile = data.get("candidate_profile", "")

print("Loaded case:", data.get("case_id"))
print("Job length:", len(job), "Profile length:", len(profile))

provider = Phi4Provider()
print("client present:", bool(provider.client), "router present:", bool(provider.router_client), "model:", provider.model)

payload = build_user_message("case", job, profile)
print("--- Instructions (truncated) ---\n", payload.get("instructions")[:400])
print("--- User content (truncated) ---\n", payload.get("content")[:800])

res = provider.analyse(job, profile)
print("success:", res.success)
print("error:", res.error)
print("validation_status:", res.validation_status)
print("elapsed_seconds:", res.elapsed_seconds)
print("raw_response:\n", res.raw_response)
if res.parsed_response:
    print("parsed keys:", list(res.parsed_response.dict().keys()))
