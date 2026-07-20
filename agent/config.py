"""Environment, generation parameters, pricing, and the two API clients.

Layer 0 of the agent package: everything here derives from the environment
(repo-root ``.env`` included) and is shared by every other module. The two
clients are module-level singletons on purpose — ``vllm_client`` talks to the
local vLLM server (free, never billed), ``openai_client`` to the OpenAI API
(billed; every response's usage is folded into a session accumulator via
``add_usage``).

Access rule: other modules call ``config.vllm_client`` / ``config.openai_client``
through the module (never ``from agent.config import vllm_client``), so tests
can monkeypatch the two attributes in exactly one place.
"""

import os
from pathlib import Path

from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]


def _load_env_file() -> None:
    """Load repo-root .env (OPENAI_API_KEY etc.) without overriding real env vars."""
    if (REPO / ".env").is_file():
        for line in (REPO / ".env").read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
PATIENT_MODEL = os.environ.get("PATIENT_MODEL", "gpt-4o-mini")

# Models offered for the judge in Advanced settings, with $/1M-token (input,
# output) prices for the session cost display. gpt-4o-mini stays the default.
JUDGE_MODEL_CHOICES = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]
_PRICES_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
}

# Generation knobs exposed via the API and the UI "Advanced settings" accordion.
DEFAULT_PARAMS = {
    "therapist_temperature": 0.7,
    "therapist_max_tokens": 300,
    "patient_temperature": 0.8,
    "judge_model": JUDGE_MODEL,
    "seed": None,  # int seeds vLLM + OpenAI calls (best-effort); None = unseeded
}


def resolve_params(params: dict | None) -> dict:
    """DEFAULT_PARAMS overlaid with any non-None user overrides."""
    return {**DEFAULT_PARAMS, **{k: v for k, v in (params or {}).items() if v is not None}}


def seed_kwargs(params: dict) -> dict:
    """The optional sampling seed as create(**kwargs), when one is set."""
    return {"seed": int(params["seed"])} if params.get("seed") is not None else {}


def empty_usage() -> dict:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}


def add_usage(acc: dict, usage, model: str) -> dict:
    """Accumulate one OpenAI response's token usage and $ cost into acc (in place)."""
    if usage is not None:
        inp, out = _PRICES_PER_MTOK.get(model, _PRICES_PER_MTOK["gpt-4o-mini"])
        acc["calls"] += 1
        acc["prompt_tokens"] += usage.prompt_tokens
        acc["completion_tokens"] += usage.completion_tokens
        acc["cost_usd"] = round(
            acc["cost_usd"] + (usage.prompt_tokens * inp + usage.completion_tokens * out) / 1e6, 6)
    return acc


vllm_client = OpenAI(base_url=VLLM_URL, api_key="unused")  # local vLLM; never billed
openai_client = OpenAI()  # OPENAI_API_KEY from env (or repo .env)
