"""Characterization tests for the session REST API (create/message/report/list)."""

import pytest

from agent.graph import GREETING
from tests.fakes import COST_PER_MINI_CALL
from tests.patch_points import LIST_MODELS

DEFAULT_MODEL = "mi-coach-pto-iter10"
DEFAULT_PARAMS_JSON = {
    "therapist_temperature": 0.7,
    "therapist_max_tokens": 300,
    "patient_temperature": 0.8,
    "judge_model": "gpt-4o-mini",
    "seed": None,
}


def test_health_ok_and_down(client, monkeypatch):
    async def fake_models():
        return ["meta-llama/Llama-3.2-1B", DEFAULT_MODEL]

    monkeypatch.setattr(LIST_MODELS, fake_models)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "vllm_models": ["meta-llama/Llama-3.2-1B", DEFAULT_MODEL],
        "live_scoring": True,
    }

    async def broken_models():
        raise RuntimeError("boom")

    monkeypatch.setattr(LIST_MODELS, broken_models)
    resp = client.get("/health")
    assert resp.status_code == 503
    assert "vLLM unreachable" in resp.json()["detail"]


def test_create_session_defaults(client):
    resp = client.post("/sessions", json={})
    assert resp.status_code == 201
    body = resp.json()
    assert body["model"] == DEFAULT_MODEL
    assert body["role"] == "patient"
    assert body["greeting"] == GREETING  # exact thesis greeting, incl. the backtick in I`m
    assert body["turn_questionnaires"] == ["Q1"]
    assert body["report_questionnaires"] == ["Q2", "MITI"]
    assert body["params"] == DEFAULT_PARAMS_JSON
    # therapist role gets no greeting (the human opens the session)
    resp = client.post("/sessions", json={"role": "therapist"})
    assert resp.json()["greeting"] is None


def test_get_session_key_contract(client):
    """Pins the absent-until-set contract: report appears only after /report,
    comparison/persona/params never appear in GET /sessions/{id}."""
    sid = client.post("/sessions", json={}).json()["session_id"]
    data = client.get(f"/sessions/{sid}").json()
    assert set(data) == {"id", "created_at", "kind", "role", "model",
                         "messages", "turn_scores", "usage"}
    assert data["kind"] == "practice"
    assert [m["role"] for m in data["messages"]] == ["system", "assistant"]
    assert data["turn_scores"] == []

    client.post(f"/sessions/{sid}/report")
    data = client.get(f"/sessions/{sid}").json()
    assert set(data) == {"id", "created_at", "kind", "role", "model",
                         "messages", "turn_scores", "usage", "report"}

    assert client.get("/sessions/nope").status_code == 404


def test_create_session_validation(client):
    assert client.post("/sessions", json={"role": "supervisor"}).status_code == 422

    resp = client.post("/sessions", json={"turn_questionnaires": ["Q1", "NOPE"]})
    assert resp.status_code == 422
    assert "NOPE" in resp.json()["detail"]

    resp = client.post("/sessions", json={"gender": "Robot"})
    assert resp.status_code == 422
    assert "invalid persona option" in resp.json()["detail"]

    resp = client.post("/sessions", json={"params": {"judge_model": "gpt-5"}})
    assert resp.status_code == 422
    assert "judge_model" in resp.json()["detail"]


def test_send_message_patient_flow(client, fake_clients):
    sid = client.post("/sessions", json={}).json()["session_id"]
    body = client.post(f"/sessions/{sid}/message",
                       json={"content": "Hi David, I keep smoking."}).json()
    assert body["reply"] == "Fake therapist reply."
    assert body["reply_role"] == "therapist"
    assert body["turns"] == 4  # system + greeting + patient + reply
    assert body["turn_score"]["therapist_turns"] == 2  # greeting counts
    assert body["turn_score"]["means"] == {"Q1": 1.0}
    assert body["usage"] == {"calls": 1, "prompt_tokens": 100,
                             "completion_tokens": 20, "cost_usd": COST_PER_MINI_CALL}
    # the therapist call hit vLLM with the session model, stops, and defaults
    call = fake_clients.vllm.calls[-1]
    assert call["model"] == DEFAULT_MODEL
    assert call["stop"] == ["<|im_end|>", "<|im_start|>"]
    assert call["max_tokens"] == 300
    assert call["temperature"] == 0.7
    # usage accumulates across turns
    body = client.post(f"/sessions/{sid}/message", json={"content": "It calms me."}).json()
    assert body["usage"]["calls"] == 2


def test_send_message_variants(client, fake_clients):
    # explicit [] disables per-turn judging (and its cost) entirely
    sid = client.post("/sessions", json={"turn_questionnaires": []}).json()["session_id"]
    body = client.post(f"/sessions/{sid}/message", json={"content": "hi"}).json()
    assert body["reply"] == "Fake therapist reply."
    assert body["turn_score"] == {}
    assert body["usage"]["calls"] == 0
    assert fake_clients.openai.n_calls == 0

    # therapist role: human's line stored as assistant, sim patient answers,
    # judges score the HUMAN's therapist turns
    sid = client.post("/sessions", json={"role": "therapist"}).json()["session_id"]
    body = client.post(f"/sessions/{sid}/message",
                       json={"content": "Welcome, what brings you here?"}).json()
    assert body["reply"] == "Fake patient reply."
    assert body["reply_role"] == "patient"
    assert body["turn_score"]["means"] == {"Q1": 1.0}
    data = client.get(f"/sessions/{sid}").json()
    assert [m["role"] for m in data["messages"]] == ["assistant", "user"]  # no system msg

    assert client.post("/sessions/nope/message", json={"content": "x"}).status_code == 404


def test_therapist_mode_needs_key(client, scoring_disabled):
    sid = client.post("/sessions", json={"role": "therapist"}).json()["session_id"]
    resp = client.post(f"/sessions/{sid}/message", json={"content": "hello"})
    assert resp.status_code == 503


def test_report_shape_and_caching(client, fake_clients):
    sid = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{sid}/message", json={"content": "hi"})
    report = client.post(f"/sessions/{sid}/report").json()
    assert report["means"] == {"Q2": 1.0, "MITI": 1.0}
    assert report["results"]["Q2"]["scores"] == {f"Q2_{i}": 1 for i in range(1, 18)}
    assert set(report["results"]["MITI"]) == {"mean", "scores", "globals", "behaviors"}
    assert report["per_turn_means"] == [{"Q1": 1.0}]
    assert set(report["assessment"]) == {"overall_rating", "summary", "strengths",
                                         "growth_areas", "tip"}
    assert report["assessment"]["overall_rating"] == 1
    # snapshot: 1 turn judge + 2 report judges + 1 supervisor assessment
    assert report["usage"]["calls"] == 4

    calls_before = fake_clients.openai.n_calls
    assert client.post(f"/sessions/{sid}/report").json() == report  # cached
    assert fake_clients.openai.n_calls == calls_before


def test_report_needs_key(client, scoring_disabled):
    sid = client.post("/sessions", json={}).json()["session_id"]
    assert client.post(f"/sessions/{sid}/report").status_code == 503


def test_list_and_delete_sessions(client):
    sid = client.post("/sessions", json={}).json()["session_id"]
    client.post(f"/sessions/{sid}/message", json={"content": "hi"})
    rows = client.get("/sessions").json()
    assert len(rows) == 1
    assert rows[0] == {
        "id": sid,
        "created_at": rows[0]["created_at"],  # timestamp, format pinned by export test
        "kind": "practice",
        "role": "patient",
        "model": DEFAULT_MODEL,
        "therapist_turns": 2,
        "scored_turns": 1,
        "has_report": False,
        "cost_usd": COST_PER_MINI_CALL,
    }
    assert client.delete(f"/sessions/{sid}").status_code == 204
    assert client.delete(f"/sessions/{sid}").status_code == 404
    assert client.get("/sessions").json() == []
