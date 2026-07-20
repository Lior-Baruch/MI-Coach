"""Characterization tests for the custom-questionnaire endpoints."""

BUILTINS = {"Q1", "Q2", "WAI-SR", "CSQ-8", "MI-SAT", "MITI", "PCT", "MICI"}


def test_questionnaire_endpoints(client):
    body = client.get("/questionnaires").json()
    assert set(body["builtin"]) == BUILTINS
    assert body["custom"] == {}

    resp = client.post("/questionnaires",
                       json={"name": "LISTEN-2", "description": "d", "items": ["s1", "s2"]})
    assert resp.status_code == 201
    assert resp.json() == {"name": "LISTEN-2", "description": "d", "items": ["s1", "s2"]}
    assert client.get("/questionnaires").json()["custom"] == \
        {"LISTEN-2": {"description": "d", "items": ["s1", "s2"]}}

    # the new instrument passes session validation and is judged like a built-in
    created = client.post("/sessions", json={"turn_questionnaires": ["LISTEN-2"]})
    assert created.status_code == 201
    sid = created.json()["session_id"]
    body = client.post(f"/sessions/{sid}/message", json={"content": "hi"}).json()
    assert body["turn_score"]["means"] == {"LISTEN-2": 1.0}
    assert body["turn_score"]["results"]["LISTEN-2"]["scores"] == {"s1": 1, "s2": 1}

    # name clash with a built-in -> 422 carrying the ValueError text
    resp = client.post("/questionnaires", json={"name": "Q1", "items": ["x"]})
    assert resp.status_code == 422
    assert "built-in" in resp.json()["detail"]

    assert client.delete("/questionnaires/LISTEN-2").status_code == 204
    assert client.delete("/questionnaires/LISTEN-2").status_code == 404
