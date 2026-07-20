"""Characterization-test harness.

The suite pins CURRENT behavior so the readability refactor cannot silently
change it. No test ever touches the network: both OpenAI-style clients are
replaced by fakes (autouse fixture), and the one import-time HTTP call in
``app.main`` (the served-model fetch inside ``_build_ui``) is forced onto its
except-fallback path, which also pins a deterministic model list.

Symbol locations are indirected through ``tests/patch_points.py`` so refactor
commits update one constant per moved symbol instead of every test.
"""

import os
import unittest.mock
from types import SimpleNamespace

import pytest

# Must precede the first agent import: OpenAI() raises without an API key, and
# a fixed fake key makes SCORING_ENABLED deterministically True. The repo .env
# loader uses os.environ.setdefault, so a real key can no longer leak in.
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import httpx  # noqa: E402

with unittest.mock.patch.object(
    httpx, "get", side_effect=httpx.ConnectError("vLLM down (test)")
):
    import app.main  # noqa: E402  (builds the Gradio UI at import time)

from fastapi.testclient import TestClient  # noqa: E402

from tests.fakes import FakeChatClient  # noqa: E402
from tests.patch_points import (  # noqa: E402
    CUSTOM_DICT,
    CUSTOM_FILE,
    OPENAI_CLIENT,
    SCORING_FLAG,
    SESSIONS_DICT,
    VLLM_CLIENT,
    resolve,
)


@pytest.fixture(autouse=True)
def fake_clients(monkeypatch):
    """Replace both API clients with fakes; returns them for scripting/asserts."""
    fakes = SimpleNamespace(
        vllm=FakeChatClient("Fake therapist reply."),
        openai=FakeChatClient("Fake patient reply."),
    )
    monkeypatch.setattr(VLLM_CLIENT, fakes.vllm)
    monkeypatch.setattr(OPENAI_CLIENT, fakes.openai)
    return fakes


@pytest.fixture(autouse=True)
def isolate_state(monkeypatch, tmp_path):
    """Redirect custom-questionnaire persistence to tmp and reset shared dicts.

    Both dicts cross modules by object identity, so they are cleared and
    restored IN PLACE (never rebound).
    """
    monkeypatch.setattr(CUSTOM_FILE, tmp_path / "custom_questionnaires.json")
    customs = resolve(CUSTOM_DICT)
    sessions = resolve(SESSIONS_DICT)
    saved_customs, saved_sessions = dict(customs), dict(sessions)
    customs.clear()
    sessions.clear()
    yield
    customs.clear()
    customs.update(saved_customs)
    sessions.clear()
    sessions.update(saved_sessions)


@pytest.fixture
def scoring_disabled(monkeypatch):
    """Simulate a missing OPENAI_API_KEY at the app layer."""
    monkeypatch.setattr(SCORING_FLAG, False)


@pytest.fixture
def custom_file():
    """The (tmp) path custom questionnaires persist to during this test."""
    return resolve(CUSTOM_FILE)


@pytest.fixture(scope="session")
def client():
    with TestClient(app.main.app) as test_client:
        yield test_client
