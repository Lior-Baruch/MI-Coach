"""PATCH-POINT INDIRECTION for the characterization suite.

Every fixture and test reaches shared app state through the dotted paths
below, so when the readability refactor moves a symbol to a new module,
the ONLY test change required is updating the matching constant here.
"""

import importlib

VLLM_CLIENT = "agent.graph._vllm"
OPENAI_CLIENT = "agent.graph._openai"
CUSTOM_FILE = "agent.graph.CUSTOM_QUESTIONNAIRES_FILE"
CUSTOM_DICT = "agent.graph.CUSTOM_QUESTIONNAIRES"
SESSIONS_DICT = "app.main.SESSIONS"
SCORING_FLAG = "app.main.SCORING_ENABLED"
LIST_MODELS = "app.main._list_models"


def resolve(dotted: str):
    """Return the object a dotted patch-point path currently refers to."""
    module, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)
