import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.candidate_scoring import calculate_candidate_fit_score_components
from types import SimpleNamespace

matches = [
    SimpleNamespace(requirement_id='M1', status='matched'),
    SimpleNamespace(requirement_id='M2', status='matched'),
    SimpleNamespace(requirement_id='M3', status='matched'),
    SimpleNamespace(requirement_id='M4', status='matched'),
    SimpleNamespace(requirement_id='M5', status='matched'),
    SimpleNamespace(requirement_id='P1', status='matched'),
    SimpleNamespace(requirement_id='P2', status='matched'),
]
mand = [SimpleNamespace(id='M1'), SimpleNamespace(id='M2'), SimpleNamespace(id='M3'), SimpleNamespace(id='M4'), SimpleNamespace(id='M5')]
pref = [SimpleNamespace(id='P1'), SimpleNamespace(id='P2')]
parsed = SimpleNamespace(mandatory_requirements=mand, preferred_requirements=pref, requirement_matches=matches)
print(calculate_candidate_fit_score_components(parsed))
