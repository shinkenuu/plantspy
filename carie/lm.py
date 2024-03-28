import dspy

_MISTRAL_INSTRUCT = "mistralai/Mistral-7B-Instruct-v0.2"
_MISTRAL = "mistralai/Mistral-7B-v0.1"
_ORCA = "microsoft/Orca-2-7b"


def get_lm(model=_MISTRAL, port=8000, url="http://127.0.0.1", **kwargs):
    lm = dspy.HFClientVLLM(model=model, port=port, url=url, **kwargs)
    return lm
