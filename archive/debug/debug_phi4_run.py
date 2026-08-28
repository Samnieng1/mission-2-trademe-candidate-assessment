import json
import traceback
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers.phi4_provider import Phi4Provider

p = Path('data/cases')
case_files = sorted(p.glob('*.json'))
if not case_files:
    print('No case files found')
    raise SystemExit(1)

case = json.loads(case_files[0].read_text(encoding='utf-8'))
print('Testing case:', case.get('case_id'))
prov = Phi4Provider()
try:
    res = prov.analyse(case['job_description'], case['candidate_profile'])
    print('Success:', res.success)
    print('Error:', res.error)
    print('Parsed response type:', type(res.parsed_response))
    print('Raw response snippet:', str(res.raw_response)[:1000])
except Exception as e:
    traceback.print_exc()
    print('Exception repr:', repr(e))
    raise
