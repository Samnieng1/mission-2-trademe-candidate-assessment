from src.providers.openai_provider import OpenAIProvider
from src.providers.phi4_provider import Phi4Provider

o = OpenAIProvider()
p = Phi4Provider()
print("OpenAI initialized:", o.health())
print("Phi4/HF initialized:", p.health())