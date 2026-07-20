"""Characterization tests for the UI's streaming engine and shared helpers
(app/ui/shared.py)."""

import pytest

from app import sessions
from app.ui import shared


def test_ui_mounted(client):
    assert client.get("/ui").status_code == 200


def test_merge_streams_interleaves_and_reraises():
    def gen(items):
        yield from items

    merged = list(shared.merge_streams(gen([1, 2]), gen(["x"])))
    assert [item for tag, item in merged if tag == "a"] == [1, 2]  # per-side order kept
    assert [item for tag, item in merged if tag == "b"] == ["x"]

    def boom():
        yield 1
        raise RuntimeError("pump failed")

    with pytest.raises(RuntimeError, match="pump failed"):
        list(shared.merge_streams(boom(), gen([])))


def test_stream_reply_finalizes_transcript(fake_clients):
    session = sessions.new_session("mi-coach-pto-iter10")
    fake_clients.vllm.queue("Streamed reply.")
    history = [{"role": "assistant", "content": "greeting"}]
    # NOTE: stream_reply yields the SAME list object and mutates the last
    # bubble in place (unlike demo_stream, which yields fresh dicts), so the
    # first yield must be checked before the generator is consumed further.
    stream = shared.stream_reply(session, "hello", history)
    first = next(stream)
    assert first[-1] == {"role": "assistant", "content": ""}  # empty bubble opens
    last = first
    for last in stream:
        pass
    assert last[-1] == {"role": "assistant", "content": "Streamed reply."}
    assert session["messages"][-2:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Streamed reply."},
    ]


def test_demo_stream_event_sequence(fake_clients):
    session = sessions.new_session("mi-coach-pto-iter10")
    events = [event for _, event in shared.demo_stream(session, 1)]
    assert events[0] == "chunk"
    assert "scored" in events
    assert events[-2:] == ["reporting", "done"]
    assert len(session["turn_scores"]) == 1
    assert "report" in session
    # patient SESSION ENDED stops before the therapist answers
    session = sessions.new_session("mi-coach-pto-iter10")
    fake_clients.openai.queue("Thanks, that is all I needed. SESSION ENDED 2")
    list(shared.demo_stream(session, 5))
    assert sum(1 for m in session["messages"] if m["role"] == "user") == 1
    assert session["turn_scores"] == []  # ended before a scoreable therapist turn


def test_params_from_controls():
    params = shared.params_from_controls(0.5, 128, "0.9", "gpt-4o", " 42 ")
    assert params == {"therapist_temperature": 0.5, "therapist_max_tokens": 128,
                      "patient_temperature": 0.9, "judge_model": "gpt-4o", "seed": 42}
    # blank or non-numeric seed is simply omitted
    assert "seed" not in shared.params_from_controls(0.7, 300, 0.8, "gpt-4o-mini", "")
    assert "seed" not in shared.params_from_controls(0.7, 300, 0.8, "gpt-4o-mini", "abc")


def test_get_or_create_session_reuse_rules():
    first = shared.get_or_create_session(None, "m1", "patient", ["Q1"], ["Q2"], None,
                                         {}, False, False, "practice")
    # same model+role -> reused, with settings re-synced from the UI
    same = shared.get_or_create_session(first["id"], "m1", "patient", ["Q2"], ["MITI"], None,
                                        {"seed": 1}, True, True, "practice")
    assert same is first
    assert same["turn_questionnaires"] == ["Q2"]
    assert same["params"] == {"seed": 1} and same["turn_rationale"] is True
    # model changed -> a fresh session starts
    fresh = shared.get_or_create_session(first["id"], "m2", "patient", ["Q1"], ["Q2"], None,
                                         {}, False, False, "practice")
    assert fresh is not first and fresh["model"] == "m2"


def test_model_catalog_parsing_and_choices(monkeypatch):
    import httpx

    def refuse(*args, **kwargs):
        raise httpx.ConnectError("vLLM down (test)")

    monkeypatch.setattr(httpx, "get", refuse)
    catalog = shared.load_model_catalog()  # vLLM down -> deterministic fallback list
    assert catalog.served == ["mi-coach-pto-iter10", "mi-coach-grpo-iter8"]
    assert catalog.adapters == {"pto": [10], "grpo": [8]}
    assert catalog.base_model == "base"  # no non-adapter model in the fallback list
    assert catalog.model_name("pto", 10) == "mi-coach-pto-iter10"
    assert catalog.model_name("base", 3) == "base"
    assert catalog.iter_choices("pto") == [("iteration 10 ★ best", 10)]
    assert catalog.iter_choices("grpo") == [("iteration 8 ★ best", 8)]
    assert catalog.method_choices[-1] == ("Base model (no adapter)", "base")
