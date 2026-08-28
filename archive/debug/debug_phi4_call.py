from dotenv import load_dotenv
load_dotenv()

from src.providers.phi4_provider import Phi4Provider

p = Phi4Provider()
print('client present:', bool(p.client), 'router present:', bool(p.router_client), 'model:', p.model)
res = p.analyse('Must have Python and SQL.','I have 5 years Python and some SQL experience.')
print('success:', res.success)
print('error:', res.error)
print('validation_status:', res.validation_status)
print('elapsed_seconds:', res.elapsed_seconds)
print('raw_response:\n', res.raw_response)
if res.parsed_response:
    print('parsed keys:', res.parsed_response.dict().keys())
