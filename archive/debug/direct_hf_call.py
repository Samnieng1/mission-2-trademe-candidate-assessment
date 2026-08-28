from dotenv import load_dotenv
load_dotenv()
from huggingface_hub import InferenceClient
import os, json

MODEL = os.getenv('HF_MODEL')
client = InferenceClient(api_key=os.getenv('HF_TOKEN'))
print('using model:', MODEL)
try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{'role':'system','content':'You are a brief assistant.'},{'role':'user','content':'Say hello and output valid JSON: {"ok": true}'}],
        temperature=0,
        max_tokens=200,
    )
    print('type(resp)=', type(resp))
    try:
        print('repr(resp)=', repr(resp)[:1000])
    except Exception:
        print('could not repr')
    try:
        rdict = resp if isinstance(resp, dict) else getattr(resp, '__dict__', None)
        print('as dict (truncated)=', json.dumps(str(rdict))[:1000])
    except Exception:
        pass
    # try to access choices
    try:
        choices = getattr(resp, 'choices', None)
        print('choices attr:', choices)
        if choices:
            print('first choice type:', type(choices[0]))
            print('first choice repr:', repr(choices[0])[:1000])
    except Exception as e:
        print('error reading choices', e)
except Exception as e:
    print('call error:', e)
