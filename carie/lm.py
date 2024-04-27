from functools import lru_cache

import dspy

_LLAMA3 = "meta-llama/Meta-Llama-3-8B"
_MISTRAL_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
_MISTRAL = "mistralai/Mistral-7B-v0.1"
_ORCA = "microsoft/Orca-2-7b"


@lru_cache
def get_mistral(port=8000, url="http://127.0.0.1", **kwargs):
    lm = dspy.HFClientVLLM(model=_MISTRAL, port=port, url=url, **kwargs)
    return lm


@lru_cache
def get_llama3(port=8000, url="http://127.0.0.1", **kwargs):
    lm = dspy.HFClientVLLM(model=_LLAMA3, port=port, url=url, **kwargs)
    return lm
