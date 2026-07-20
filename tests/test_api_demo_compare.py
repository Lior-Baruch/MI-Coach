"""Characterization tests for POST /demo and POST /compare/review."""

from tests.fakes import COST_PER_MINI_CALL


def test_demo_endpoint(client, fake_clients):
    body = client.post("/demo", json={"max_patient_turns": 2,
                                      "cooperation": "Low", "age": 27}).json()
    assert body["model"] == "mi-coach-pto-iter10"
    # system prompt is dropped from the response; greeting comes first
    assert [m["role"] for m in body["messages"]] == \
        ["assistant", "user", "assistant", "user", "assistant"]
    assert len(body["turn_scores"]) == 2
    assert body["report"]["assessment"]["overall_rating"] == 1
    assert "27 years old" in body["patient_persona"]  # persona options honored
    # 2 patient + 2 turn judges + 2 report judges + 1 assessment
    assert body["usage"]["calls"] == 7
    assert body["usage"]["cost_usd"] == round(7 * COST_PER_MINI_CALL, 6)

    # the finished run is stored as a session so it shows up in history
    rows = client.get("/sessions").json()
    assert [row["kind"] for row in rows] == ["demo"]
    assert rows[0]["id"] == body["session_id"]
    assert rows[0]["has_report"] is True
    assert rows[0]["cost_usd"] == body["usage"]["cost_usd"]


def test_demo_needs_key(client, scoring_disabled):
    assert client.post("/demo", json={}).status_code == 503


def test_demo_validates_persona(client):
    resp = client.post("/demo", json={"cooperation": "Sometimes"})
    assert resp.status_code == 422
    assert "invalid persona option" in resp.json()["detail"]


def test_compare_review(client, fake_clients):
    sid_a = client.post("/sessions", json={}).json()["session_id"]
    sid_b = client.post("/sessions", json={"model": "mi-coach-grpo-iter8"}).json()["session_id"]

    # both sides need at least one turn; only therapist-role sessions start
    # EMPTY (patient-role sessions always contain the greeting), so the 422
    # branch is reachable only through one of those
    sid_empty = client.post("/sessions", json={"role": "therapist"}).json()["session_id"]
    resp = client.post("/compare/review", json={"session_a": sid_a, "session_b": sid_empty})
    assert resp.status_code == 422

    client.post(f"/sessions/{sid_a}/message", json={"content": "hi"})
    client.post(f"/sessions/{sid_b}/message", json={"content": "hi"})
    verdict = client.post("/compare/review",
                          json={"session_a": sid_a, "session_b": sid_b}).json()
    assert set(verdict) == {"model_a", "model_b", "preferred", "summary",
                            "key_differences", "a_strengths", "b_strengths",
                            "recommendation"}
    assert verdict["model_a"] == "mi-coach-pto-iter10"
    assert verdict["model_b"] == "mi-coach-grpo-iter8"
    assert verdict["preferred"] == "A"  # enum-first in the fake judge

    # reports were generated on demand; the verdict never leaks into the GET body
    data_a = client.get(f"/sessions/{sid_a}").json()
    assert "report" in data_a and "comparison" not in data_a
    # the comparison call itself is billed to side A
    data_b = client.get(f"/sessions/{sid_b}").json()
    assert data_a["usage"]["calls"] == 4 + 1  # turn judge + report(3) + comparison
    assert data_b["usage"]["calls"] == 4

    assert client.post("/compare/review",
                       json={"session_a": sid_a, "session_b": "nope"}).status_code == 404


def test_compare_review_needs_key(client, scoring_disabled):
    resp = client.post("/compare/review", json={"session_a": "x", "session_b": "y"})
    assert resp.status_code == 503
